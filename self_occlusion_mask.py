#!/usr/bin/env python
"""
Render the robot's own body into each camera as a pixel mask.

A wrist camera sees its own gripper from a few centimetres away: a large,
textureless, extremely near object that occupies part of every frame. Those
pixels cannot contribute scene geometry, and there is reason to suspect they
distort the network's depth and pose estimate for that view.

This projects the robot's collision geometry through forward kinematics into
each camera and rasterises it, so the pixels can be excluded before inference
(to test that suspicion) or the resulting points dropped afterwards (to keep the
robot out of an obstacle map).

The mask is deliberately generous. In image space an over-wide mask costs only
background that another view can supply or that stays unobserved, whereas a
too-narrow one leaves a piece of robot in the map labelled as an obstacle.

Triangles are clipped against the near plane rather than dropped: the gripper
routinely straddles it, and dropping those triangles would punch a hole in the
mask exactly where the robot is closest.

Example:
  python self_occlusion_mask.py \
      --session ~/MapAnythingTest/TestData/session_20260721_232012 \
      --output-root ~/MapAnythingTest/outputs_g2 \
      --urdf G2_parameters/G2_t2_crs_omnipicker/urdf/G2_t2_crs_omnipicker.urdf \
      --preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capture_contract import VIEW_NAMES
from robot_profiles import detect_profile, get_profile
from urdf_model import UrdfRobot

NEAR_PLANE_M = 0.02
MASK_FILE = "self_occlusion_mask.npz"


def clip_triangle_near(triangle, near=NEAR_PLANE_M):
    """Clip one camera-frame triangle against ``z >= near``.

    Returns a list of triangles (0, 1 or 2) covering the visible part.  A
    triangle straddling the near plane must be cut rather than discarded: the
    gripper sits centimetres from the lens and would otherwise leave a hole in
    the mask exactly where the robot is closest to the camera.
    """
    inside = [point for point in triangle if point[2] >= near]
    outside = [point for point in triangle if point[2] < near]
    if len(inside) == 3:
        return [triangle]
    if not inside:
        return []

    def cut(a, b):
        t = (near - a[2]) / (b[2] - a[2])
        return a + t * (b - a)

    if len(inside) == 1:
        a = inside[0]
        return [np.stack([a, cut(a, outside[0]), cut(a, outside[1])])]
    a, b = inside
    c = outside[0]
    a_cut, b_cut = cut(a, c), cut(b, c)
    return [np.stack([a, b, a_cut]), np.stack([b, b_cut, a_cut])]


def encloses_camera(local_vertices, faces):
    """True when a link's solid geometry contains the camera's optical centre.

    The camera sits inside its own housing, so that housing's mesh contains the
    optical centre.  Projecting it fills most of the frame, which is
    geometrically consistent and physically nonsense: a camera cannot occlude
    itself.  Detecting the condition beats hardcoding link names, which differ
    per robot.

    This has to be real containment, not a bounding-box test.  The gripper wraps
    around the camera closely enough that its *box* contains the optical centre
    while its solid does not, and excluding the gripper is precisely the wrong
    outcome: it is the part most worth masking.
    """
    if not len(local_vertices) or not len(faces):
        return False
    minimum = local_vertices.min(axis=0)
    maximum = local_vertices.max(axis=0)
    if not np.all((minimum < 0.0) & (maximum > 0.0)):
        return False  # Cheap rejection before the ray test.

    # Ray-parity containment, computed directly rather than through a spatial
    # index, so this needs nothing beyond numpy.  Three random directions are
    # cast and the majority wins, which removes the ambiguity of a ray that
    # happens to graze an edge or a vertex.
    triangles = local_vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    generator = np.random.default_rng(0)
    votes = 0
    for _ in range(3):
        direction = generator.normal(size=3)
        direction /= np.linalg.norm(direction)
        pvec = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, pvec)
        parallel = np.abs(determinant) < 1e-12
        safe = np.where(parallel, 1.0, determinant)
        tvec = -triangles[:, 0]
        u = np.einsum("ij,ij->i", tvec, pvec) / safe
        qvec = np.cross(tvec, edge1)
        v = np.einsum("j,ij->i", direction, qvec) / safe
        t = np.einsum("ij,ij->i", edge2, qvec) / safe
        hit = ~parallel & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        votes += int(hit.sum()) % 2
    return votes >= 2


def parse_shrink(values):
    """Parse ``substring=factor`` pairs into a mapping."""
    result = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--shrink-links wants substring=factor, got {item!r}")
        key, _, factor = item.partition("=")
        value = float(factor)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"Shrink factor must be in (0, 1], got {value}")
        result[key] = value
    return result


def render_mask(
    robot, joint_positions, K, base_T_cam, width, height, *, dilate_px=12, links=None, shrink=None
):
    """Rasterise the robot's collision geometry into one camera."""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    world_T_cam = np.asarray(base_T_cam, dtype=np.float64)
    cam_T_world = np.linalg.inv(world_T_cam)
    stats = {
        "links_drawn": {},
        "triangles_drawn": 0,
        "triangles_clipped": 0,
        "links_enclosing_camera": [],
    }

    for link, vertices, faces in robot.world_geometry(
        joint_positions, "collision", links=links, shrink=shrink
    ):
        local = vertices @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
        if not len(faces):
            continue
        if encloses_camera(local, faces):
            if link not in stats["links_enclosing_camera"]:
                stats["links_enclosing_camera"].append(link)
            continue
        before = int(mask.any())
        drawn = 0
        for face in faces:
            triangle = local[face]
            if triangle[:, 2].max() < NEAR_PLANE_M:
                continue
            pieces = clip_triangle_near(triangle)
            if len(pieces) != 1 or not np.array_equal(pieces[0], triangle):
                stats["triangles_clipped"] += 1
            for piece in pieces:
                u = piece[:, 0] / piece[:, 2] * K[0, 0] + K[0, 2]
                v = piece[:, 1] / piece[:, 2] * K[1, 1] + K[1, 2]
                polygon = np.stack([u, v], axis=1)
                if not np.isfinite(polygon).all():
                    continue
                # Clamping keeps a triangle that runs off-frame filling the part
                # that is on-frame instead of being rejected outright.
                polygon = np.clip(polygon, [-2 * width, -2 * height], [3 * width, 3 * height])
                cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
                drawn += 1
        stats["triangles_drawn"] += drawn
        covered = int(mask.sum())
        stats["links_drawn"][link] = {"triangles": drawn, "cumulative_pixels": covered}
        del before

    raw_pixels = int(mask.sum())
    if dilate_px > 0 and raw_pixels:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    stats.update(
        {
            "raw_pixels": raw_pixels,
            "dilated_pixels": int(mask.sum()),
            "coverage_fraction": float(mask.sum()) / (width * height),
            "dilate_px": int(dilate_px),
            "near_plane_m": NEAR_PLANE_M,
            "shrink_links": dict(shrink or {}),
        }
    )
    return mask.astype(bool), stats


def camera_parameters(output_root: Path, capture: str, view: str):
    undistorted = output_root / "undistorted" / capture
    with (undistorted / f"{view}_K.json").open(encoding="utf-8") as handle:
        intrinsics = json.load(handle)
    with (undistorted / "camera_poses_opencv_cam2world.json").open(encoding="utf-8") as handle:
        poses = json.load(handle)["poses"]
    return (
        np.asarray(intrinsics["K"], dtype=np.float64),
        int(intrinsics["width"]),
        int(intrinsics["height"]),
        np.asarray(poses[view], dtype=np.float64),
    )


def build_capture_masks(
    session, output_root, capture, profile, robot, *, dilate_px=12, preview=None, shrink=None
):
    loaded = profile.load(session, capture)
    if not loaded.joint_positions:
        raise ValueError(f"{capture} carries no joint state; cannot place the robot")
    undistorted = output_root / "undistorted" / capture
    payload = {}
    report = {}
    for view in profile.view_names:
        K, width, height, base_T_cam = camera_parameters(output_root, capture, view)
        mask, stats = render_mask(
            robot, loaded.joint_positions, K, base_T_cam, width, height,
            dilate_px=dilate_px, shrink=shrink,
        )
        payload[f"{view}_self_mask"] = mask
        report[view] = stats
        if preview is not None:
            _write_preview(preview, capture, view, undistorted / f"{view}.png", mask)
    payload["views"] = np.asarray(list(profile.view_names))
    payload["dilate_px"] = np.asarray(dilate_px)
    payload["urdf"] = np.asarray(str(robot.path))
    payload["shrink_links"] = np.asarray(json.dumps(shrink or {}))
    np.savez_compressed(undistorted / MASK_FILE, **payload)
    return report


def _write_preview(directory: Path, capture: str, view: str, image_path: Path, mask):
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    overlay = image.copy()
    overlay[mask] = (0, 0, 255)
    blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / f"{capture}_{view}_selfmask.png"), blended)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, action="append", default=None)
    parser.add_argument("--captures", nargs="*", default=None)
    parser.add_argument("--robot", default=None)
    parser.add_argument(
        "--dilate-px",
        type=int,
        default=12,
        help=(
            "Grow the mask by this many pixels. Generous is the safe direction: "
            "extra background can be recovered from another view, a missed piece "
            "of robot becomes an obstacle."
        ),
    )
    parser.add_argument(
        "--shrink-links",
        action="append",
        default=None,
        metavar="SUBSTRING=FACTOR",
        help=(
            "Scale matching links about their own centroid, e.g. gripper=0.7. "
            "G2's gripper collision mesh is a swept volume covering the whole "
            "open/close travel, so at wrist range it masks far more than the "
            "gripper occupies. Shrinking gives up the guarantee that the shape "
            "encloses the real part, so it is opt-in and recorded in the output."
        ),
    )
    parser.add_argument("--preview", type=Path, help="Write mask overlays here for inspection")
    args = parser.parse_args()

    profile = get_profile(args.robot) if args.robot else detect_profile(args.session)
    captures = args.captures or profile.discover(args.session)
    shrink = parse_shrink(args.shrink_links)
    robot = UrdfRobot(args.urdf, mesh_roots=args.mesh_root)
    if shrink:
        print(f"Shrinking links about their centroid: {shrink}")
    print(f"URDF: {robot.path}")
    print(f"Robot profile: {profile.name}, {len(captures)} captures")

    for capture in captures:
        try:
            report = build_capture_masks(
                args.session, args.output_root, capture, profile, robot,
                dilate_px=args.dilate_px, preview=args.preview, shrink=shrink,
            )
        except (OSError, ValueError, KeyError) as exc:
            print(f"  {capture}: FAILED {type(exc).__name__}: {exc}")
            continue
        summary = ", ".join(
            f"{view} {100 * stats['coverage_fraction']:.1f}%" for view, stats in report.items()
        )
        print(f"  {capture}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
