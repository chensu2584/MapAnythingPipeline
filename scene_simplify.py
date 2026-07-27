"""Turn a raw voxelized reconstruction into a clean, coarse obstacle map.

The voxel grid coming out of ``voxelize.py`` still contains the robot's own
gripper (the wrist cameras stare straight at it), single-voxel measurement
noise, and every object rendered as a fuzzy surface shell.  A path planner
does not want any of that -- it wants "here is the ground, here are a handful
of solid boxes to stay away from."  This module produces exactly that:

  1. Gripper removal -- purely spatial, NOT from the URDF.  The gripper that is
     physically bolted to this robot is a different part from the one in the
     G2 URDF (the URDF end-effector / camera_link is a placeholder), so its FK
     mesh cannot be trusted to sit where the real gripper's voxels are.  What
     IS trustworthy is the wrist-camera pose (ground-truth verified).  A
     conservative 0.3 m sphere about each wrist-camera centre removes the
     nearest self-observed gripper shell without reaching nearby blue bins.

  2. Denoise -- drop under-observed voxels, then conservatively remove only
     isolated specks and tiny disconnected fragments.  Denoising and object
     abstraction use separate thresholds so a small real object is not erased
     merely because it is too small to become a planning box.

  3. Simplify -- fit the dominant table surface, retain it as a support box,
     then cluster raised geometry in XY and fit boxes or vertical cylinders.
     Extending each primitive back down to the support surface preserves the
     main body of partially observed bins and low objects.  The GLB also carries
     named head/hand-camera, gripper-centre, and world-origin pose markers.

Depends only on numpy / scipy / scikit-learn / trimesh -- no torch, no GPU.

Example
-------
    python scene_simplify.py OUT/snapshot_x/voxels.npz \
        --extrinsics RAW/snapshot_x/camera_extrinsics.json \
        --out-dir OUT/snapshot_x
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN

# A 0.4 m sphere reaches the blue bins when an arm moves over the workspace in
# snapshots 4/5.  At 0.3 m it still covers the self-observed gripper shell while
# leaving the nearby scene geometry intact.
DEFAULT_GRIPPER_RADIUS_M = 0.30
HAND_CAMERA_KEYS = ("hand_left_rgb", "hand_right_rgb")
MARKER_COLORS = {
    "origin": [255, 255, 255],
    "head_camera": [255, 214, 64],
    "left_camera": [55, 210, 255],
    "right_camera": [220, 80, 255],
    "left_gripper": [80, 220, 110],
    "right_gripper": [255, 135, 55],
}


@dataclass
class SimplifyReport:
    voxel_size_m: float
    world_frame: str
    input_voxels: int
    removed_gripper: int = 0
    removed_low_count: int = 0
    removed_low_conf: int = 0
    removed_noise_clusters: int = 0
    kept_voxels: int = 0
    ground_plane: dict | None = None
    obstacle_voxels: int = 0
    ground_voxels: int = 0
    support_voxels: int = 0
    support_boxes: list = field(default_factory=list)
    obstacle_boxes: list = field(default_factory=list)
    scene_boxes: list = field(default_factory=list)
    robot_markers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def load_voxels(path: str):
    """Return (world_points, colors, conf, counts, source_views, meta)."""
    z = np.load(path, allow_pickle=True)
    idx = z["indices"].astype(np.float64)
    origin = np.asarray(z["origin"], dtype=np.float64)
    voxel_size = float(z["voxel_size"])
    world = origin + (idx + 0.5) * voxel_size
    meta = {
        "voxel_size": voxel_size,
        "origin": origin,
        "dims": np.asarray(z["dims"]),
        "world_frame": str(z["world_frame"]) if "world_frame" in z else "unknown",
        "translation_unit": str(z["translation_unit"]) if "translation_unit" in z else "meter",
    }
    colors = z["colors"] if "colors" in z else np.full((len(world), 3), 200, np.uint8)
    conf = z["conf"] if "conf" in z else np.ones(len(world), np.float32)
    counts = z["counts"] if "counts" in z else np.ones(len(world), np.int32)
    source_views = z["source_views"] if "source_views" in z else np.zeros(len(world), np.uint8)
    return world, colors, conf, counts, source_views, meta


def read_camera_centers(extrinsics_path: str):
    """Return (wrist_centers, head_z) in base frame from the raw capture."""
    with open(extrinsics_path, encoding="utf-8") as handle:
        data = json.load(handle)
    ext = data.get("extrinsics", data)
    wrists = []
    for key in HAND_CAMERA_KEYS:
        if key in ext and "matrix" in ext[key]:
            wrists.append(np.asarray(ext[key]["matrix"], dtype=np.float64)[:3, 3])
    if not wrists:
        raise ValueError(f"No wrist cameras {HAND_CAMERA_KEYS} in {extrinsics_path}")
    head_z = None
    if "head_rgb" in ext and "matrix" in ext["head_rgb"]:
        head_z = float(np.asarray(ext["head_rgb"]["matrix"])[2, 3])
    return wrists, head_z


def read_scene_markers(extrinsics_path: str | None):
    """Read camera/gripper poses and always include the base-frame origin."""
    markers = [{
        "id": "origin",
        "role": "marker",
        "kind": "origin",
        "center_m": [0.0, 0.0, 0.0],
        "pose_matrix": np.eye(4).tolist(),
        "color": MARKER_COLORS["origin"],
    }]
    if not extrinsics_path:
        return markers

    with open(extrinsics_path, encoding="utf-8") as handle:
        data = json.load(handle)
    ext = data.get("extrinsics", data)
    fk = data.get("fk_base_T_link", {})
    sources = (
        ("head_camera", "camera", ext.get("head_rgb")),
        ("left_camera", "camera", ext.get("hand_left_rgb")),
        ("right_camera", "camera", ext.get("hand_right_rgb")),
        ("left_gripper", "gripper_center", fk.get("arm_l_end_link")),
        ("right_gripper", "gripper_center", fk.get("arm_r_end_link")),
    )
    for marker_id, kind, record in sources:
        if not isinstance(record, dict) or "matrix" not in record:
            continue
        pose = np.asarray(record["matrix"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue
        markers.append({
            "id": marker_id,
            "role": "marker",
            "kind": kind,
            "center_m": pose[:3, 3].round(5).tolist(),
            "pose_matrix": pose.round(8).tolist(),
            "color": MARKER_COLORS[marker_id],
        })
    return markers


def remove_gripper(world: np.ndarray, centers, radius: float) -> np.ndarray:
    """Return a keep-mask that drops voxels within `radius` of any wrist camera."""
    keep = np.ones(len(world), dtype=bool)
    for c in centers:
        keep &= np.linalg.norm(world - c, axis=1) >= radius
    return keep


def find_support_surface(points: np.ndarray, z_band=None, bin_m: float = 0.02):
    """Locate the working surface (table top) the objects sit on.

    A single RANSAC plane is unreliable here: the scene stacks floor, table and
    walls at different heights, so one plane snaps to whichever slab is densest
    and the "above the plane" test then mislabels the table itself.  The world
    frame is base_link with +Z up and the table is nearly level (normal within a
    couple degrees of +Z), so the robust signal is the height histogram: the
    densest horizontal layer is the table surface.  Its histogram-bin upper
    edge is used as the support top.  Walking upward through all merely
    non-empty bins is deliberately avoided: object sides keep those bins dense
    enough to make the old method swallow the lower half of every object.

    Some frames have a denser floor/base slab lower down that would steal the
    mode, so the search is restricted to `z_band` (lo, hi) when given -- derived
    from the head-camera height, since the table sits a roughly fixed drop below
    the fixed head and well above the floor.

    Returns (table_top_z, mode_z, hist_peak_count).
    """
    z = points[:, 2]
    if len(z) < 3:
        return (float(z.max()) if len(z) else 0.0), 0.0, 0
    sel = np.ones(len(z), bool)
    if z_band is not None:
        sel = (z >= z_band[0]) & (z <= z_band[1])
        if sel.sum() < 3:
            sel = np.ones(len(z), bool)  # band empty -> fall back to full range
    zb = z[sel]
    edges = np.arange(zb.min(), zb.max() + bin_m, bin_m)
    hist, edges = np.histogram(zb, bins=edges)
    peak = int(np.argmax(hist))
    mode_z = float(edges[peak] + bin_m / 2)
    table_top_z = float(edges[peak + 1])
    return table_top_z, mode_z, int(hist[peak])


def fit_support_box(points: np.ndarray, colors: np.ndarray, mode_z: float,
                    table_top_z: float, eps: float, voxel_size: float,
                    thickness_m: float):
    """Fit one coarse table-top box from the largest XY component at `mode_z`."""
    half_band = max(voxel_size, 0.01)
    layer_mask = np.abs(points[:, 2] - mode_z) <= half_band + 1e-8
    layer = points[layer_mask]
    layer_colors = colors[layer_mask]
    if len(layer) < 3:
        return None, 0

    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(layer[:, :2])
    valid = [lab for lab in set(labels) if lab >= 0]
    if not valid:
        return None, 0
    best = max(valid, key=lambda lab: int((labels == lab).sum()))
    component = labels == best
    support = layer[component]
    support_colors = layer_colors[component]
    pad = voxel_size / 2
    lo_xy = support[:, :2].min(0) - pad
    hi_xy = support[:, :2].max(0) + pad
    thickness = max(float(thickness_m), voxel_size)
    lo = np.array([lo_xy[0], lo_xy[1], table_top_z - thickness])
    hi = np.array([hi_xy[0], hi_xy[1], table_top_z])
    box = {
        "id": 0,
        "role": "support",
        "label": "table",
        "primitive": "box",
        "center_m": ((lo + hi) / 2).round(4).tolist(),
        "size_m": (hi - lo).round(4).tolist(),
        "min_m": lo.round(4).tolist(),
        "max_m": hi.round(4).tolist(),
        "voxel_count": int(component.sum()),
        "color": [int(v) for v in support_colors.mean(0)],
    }
    return box, int(component.sum())


def footprint_metrics(points: np.ndarray):
    """Return conservative 2D shape metrics for primitive selection."""
    xy = np.unique(np.round(points[:, :2], 5), axis=0)
    if len(xy) < 3:
        return {
            "aspect_ratio": float("inf"),
            "circularity": 0.0,
            "span_m": [0.0, 0.0],
        }
    span = np.maximum(xy.max(0) - xy.min(0), 1e-9)
    aspect = float(span.max() / span.min())
    circularity = 0.0
    try:
        hull = ConvexHull(xy)
        if hull.area > 0:
            circularity = float(4 * np.pi * hull.volume / (hull.area ** 2))
    except QhullError:
        pass
    return {
        "aspect_ratio": aspect,
        "circularity": circularity,
        "span_m": span.tolist(),
    }


def axis_aligned_boxes(points: np.ndarray, colors: np.ndarray, eps: float,
                       min_samples: int, min_cluster: int,
                       support_top_z: float | None = None,
                       min_height_m: float = 0.0,
                       voxel_size: float = 0.0,
                       primitive_mode: str = "auto",
                       cylinder_max_diameter: float = 0.20,
                       cylinder_max_aspect: float = 1.6,
                       cylinder_min_circularity: float = 0.82,
                       max_footprint_aspect: float = 3.0):
    """Cluster raised geometry in XY and fit a box or vertical cylinder.

    XY clustering tolerates holes in vertical surfaces.  A cluster must still
    have several voxels above `min_height_m`, which rejects shallow table noise.
    Its lower face is extended back to the support top.  Compact, circular,
    small footprints become cylinders; large or rectangular footprints remain
    boxes.  Very thin high-aspect fragments are rejected as table-edge noise.
    """
    if len(points) == 0:
        return [], np.array([], dtype=int)
    if support_top_z is not None and min_height_m > 0:
        seed_mask = points[:, 2] >= support_top_z + min_height_m
    else:
        seed_mask = np.ones(len(points), dtype=bool)
    seed_indices = np.where(seed_mask)[0]
    if len(seed_indices) == 0:
        return [], np.full(len(points), -1, dtype=int)

    seed_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
        points[seed_indices, :2]
    )
    labels = np.full(len(points), -1, dtype=int)
    labels[seed_indices] = seed_labels
    boxes = []
    for lab in sorted(set(seed_labels)):
        if lab < 0:
            continue
        seed_local = seed_labels == lab
        if seed_local.sum() < min_cluster:
            continue
        seed_pts = points[seed_indices[seed_local]]
        metrics = footprint_metrics(seed_pts)
        seed_height = float(np.ptp(seed_pts[:, 2]) + voxel_size)
        if (
            metrics["aspect_ratio"] > max_footprint_aspect
            and seed_height < max(min_height_m * 2, 0.08)
        ):
            continue

        # Recover the lower body only in the seed cluster's local XY footprint.
        # This fills a bin/object down to the table without allowing shallow
        # table ripples elsewhere to connect two independent objects.
        grow = max(voxel_size, eps / 2)
        body_lo_xy = seed_pts[:, :2].min(0) - grow
        body_hi_xy = seed_pts[:, :2].max(0) + grow
        body = np.all(
            (points[:, :2] >= body_lo_xy) & (points[:, :2] <= body_hi_xy), axis=1
        )
        labels[(labels < 0) & body] = lab
        pts = points[body]
        box_colors = colors[body]
        pad = voxel_size / 2
        lo = pts.min(0) - pad
        hi = pts.max(0) + pad
        if support_top_z is not None:
            lo[2] = min(lo[2], support_top_z)

        seed_span = np.asarray(metrics["span_m"])
        use_cylinder = (
            primitive_mode == "cylinder"
            or (
                primitive_mode == "auto"
                and seed_span.max() + voxel_size <= cylinder_max_diameter
                and metrics["aspect_ratio"] <= cylinder_max_aspect
                and metrics["circularity"] >= cylinder_min_circularity
            )
        )
        primitive = "cylinder" if use_cylinder else "box"
        shape = {
            "id": len(boxes),
            "role": "object",
            "label": "object",
            "primitive": primitive,
            "center_m": ((lo + hi) / 2).round(4).tolist(),
            "size_m": (hi - lo).round(4).tolist(),
            "min_m": lo.round(4).tolist(),
            "max_m": hi.round(4).tolist(),
            "voxel_count": int(body.sum()),
            "seed_voxel_count": int(seed_local.sum()),
            "color": [int(v) for v in box_colors.mean(0)],
            "shape_metrics": {
                "aspect_ratio": round(metrics["aspect_ratio"], 4),
                "circularity": round(metrics["circularity"], 4),
            },
        }
        if use_cylinder:
            center_xy = (seed_pts[:, :2].min(0) + seed_pts[:, :2].max(0)) / 2
            radial = np.linalg.norm(seed_pts[:, :2] - center_xy, axis=1)
            radius = float(np.percentile(radial, 98) + grow + pad)
            cylinder_lo = np.array([
                center_xy[0] - radius,
                center_xy[1] - radius,
                lo[2],
            ])
            cylinder_hi = np.array([
                center_xy[0] + radius,
                center_xy[1] + radius,
                hi[2],
            ])
            shape.update({
                "center_m": ((cylinder_lo + cylinder_hi) / 2).round(4).tolist(),
                "size_m": (cylinder_hi - cylinder_lo).round(4).tolist(),
                "min_m": cylinder_lo.round(4).tolist(),
                "max_m": cylinder_hi.round(4).tolist(),
                "radius_m": round(radius, 4),
                "height_m": round(float(cylinder_hi[2] - cylinder_lo[2]), 4),
                "axis": [0.0, 0.0, 1.0],
            })
        boxes.append(shape)
    return boxes, labels


def voxels_to_glb(world: np.ndarray, colors: np.ndarray, voxel_size: float):
    """One small cube per voxel, merged, with the same X-flip voxelize.py uses
    so the result overlays scene.glb / voxels.glb in the same viewer pose."""
    cube = trimesh.creation.box(extents=[voxel_size * 0.95] * 3)
    v = cube.vertices
    f = cube.faces
    V = (world[:, None, :] + v[None, :, :]).reshape(-1, 3)
    F = (f[None] + (np.arange(len(world)) * len(v))[:, None, None]).reshape(-1, 3)
    vc = np.repeat(colors, len(v), axis=0)
    mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=vc, process=False)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return mesh


def primitives_to_glb(primitives: list, inflate_m: float = 0.0,
                      markers: list | None = None):
    """Render fitted primitives plus named robot/world pose markers."""
    scene = trimesh.Scene()
    flip = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    for primitive in primitives:
        # Keep the support slab faithful to the observed table footprint.
        # Inflation remains useful for movable/object avoidance boxes.
        inflation = 0.0 if primitive.get("role") == "support" else inflate_m
        if primitive.get("primitive") == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=max(float(primitive["radius_m"]) + inflation, 1e-3),
                height=max(float(primitive["height_m"]) + 2 * inflation, 1e-3),
                sections=32,
            )
        else:
            size = np.asarray(primitive["size_m"]) + 2 * inflation
            mesh = trimesh.creation.box(extents=np.maximum(size, 1e-3))
        T = np.eye(4)
        T[:3, 3] = primitive["center_m"]
        mesh.apply_transform(T)
        mesh.apply_transform(flip)
        col = primitive.get("color", [200, 60, 60])
        mesh.visual.vertex_colors = np.array([*col, 120], np.uint8)
        name = f"primitive_{primitive['id']}_{primitive.get('primitive', 'box')}"
        scene.add_geometry(mesh, geom_name=name)

    marker_lookup = {m["id"]: m for m in (markers or [])}
    for marker in markers or []:
        pose = np.asarray(marker["pose_matrix"], dtype=np.float64)
        marker_id = marker["id"]
        kind = marker["kind"]
        radius = 0.025 if kind == "origin" else (0.03 if marker_id == "head_camera" else 0.02)
        center_mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        center_mesh.apply_translation(pose[:3, 3])
        center_mesh.apply_transform(flip)
        center_mesh.visual.vertex_colors = np.array([*marker["color"], 255], np.uint8)
        scene.add_geometry(center_mesh, geom_name=f"marker_{marker_id}_center")

        axis_length = 0.20 if kind == "origin" else 0.10
        axes = trimesh.creation.axis(
            origin_size=radius * 0.55,
            transform=pose,
            origin_color=np.array([*marker["color"], 255], np.uint8),
            axis_radius=0.003,
            axis_length=axis_length,
        )
        axes.apply_transform(flip)
        scene.add_geometry(axes, geom_name=f"marker_{marker_id}_axes")

    for side in ("left", "right"):
        camera = marker_lookup.get(f"{side}_camera")
        gripper = marker_lookup.get(f"{side}_gripper")
        if not camera or not gripper:
            continue
        segment = np.asarray([camera["center_m"], gripper["center_m"]])
        link = trimesh.creation.cylinder(radius=0.004, segment=segment, sections=12)
        link.apply_transform(flip)
        link.visual.vertex_colors = np.array([*MARKER_COLORS[f"{side}_camera"], 210], np.uint8)
        scene.add_geometry(link, geom_name=f"marker_{side}_camera_to_gripper")
    return scene


def boxes_to_glb(boxes: list, inflate_m: float = 0.0):
    """Backward-compatible wrapper for callers that only provide primitives."""
    return primitives_to_glb(boxes, inflate_m=inflate_m)


def simplify(voxels_path: str, extrinsics_path: str | None, out_dir: str,
             gripper_radius: float, min_count: int, min_conf: float | None,
             cluster_eps: float, min_cluster: int, obstacle_height: float,
             box_inflate: float, surface_z: float | None = None,
             denoise_min_cluster: int = 4, table_thickness: float = 0.06,
             primitive_mode: str = "auto",
             cylinder_max_diameter: float = 0.20,
             cylinder_max_aspect: float = 1.6,
             cylinder_min_circularity: float = 0.82,
             max_footprint_aspect: float = 3.0,
             include_markers: bool = True):
    world, colors, conf, counts, source_views, meta = load_voxels(voxels_path)
    vs = meta["voxel_size"]
    rep = SimplifyReport(voxel_size_m=vs, world_frame=meta["world_frame"],
                         input_voxels=len(world))

    keep = np.ones(len(world), dtype=bool)
    head_z = None

    # 1. gripper (spatial, not URDF)
    if extrinsics_path:
        centers, head_z = read_camera_centers(extrinsics_path)
        gmask = remove_gripper(world, centers, gripper_radius)
        rep.removed_gripper = int((~gmask & keep).sum())
        keep &= gmask

    # 2. under-observed / low-confidence voxels
    if min_count > 1:
        m = counts >= min_count
        rep.removed_low_count = int((~m & keep).sum())
        keep &= m
    if min_conf is not None:
        m = conf >= min_conf
        rep.removed_low_conf = int((~m & keep).sum())
        keep &= m

    # 3. Conservative DBSCAN denoise.  This threshold is intentionally separate
    # from min_cluster: a fragment can be real scene geometry even when it is too
    # small to deserve its own planning box.
    idx_keep = np.where(keep)[0]
    if len(idx_keep):
        labels = DBSCAN(eps=cluster_eps, min_samples=2).fit_predict(world[idx_keep])
        drop = np.zeros(len(idx_keep), bool)
        for lab in set(labels):
            m = labels == lab
            if lab < 0 or m.sum() < denoise_min_cluster:
                drop |= m
        rep.removed_noise_clusters = int(drop.sum())
        keep[idx_keep[drop]] = False

    W = world[keep]
    C = colors[keep]
    rep.kept_voxels = int(keep.sum())

    # 4. Preserve the support/table as one box, then abstract raised objects.
    support_boxes = []
    object_boxes = []
    if len(W) >= 3:
        if surface_z is not None:
            table_top_z = float(surface_z)
            mode_z = table_top_z - vs
            peak = -1
        else:
            # the table sits a roughly fixed drop below the fixed head camera and
            # well above the floor; restrict the mode search to that band so a
            # denser floor/base slab cannot steal it.
            band = (head_z - 1.0, head_z - 0.4) if head_z is not None else None
            table_top_z, mode_z, peak = find_support_surface(W, z_band=band)

        support_box, support_count = fit_support_box(
            W, C, mode_z, table_top_z,
            eps=max(cluster_eps, vs * 2.5),
            voxel_size=vs,
            thickness_m=table_thickness,
        )
        if support_box is not None:
            support_boxes.append(support_box)
            footprint_lo = np.asarray(support_box["min_m"])[:2] - 2 * vs
            footprint_hi = np.asarray(support_box["max_m"])[:2] + 2 * vs
            in_footprint = np.all(
                (W[:, :2] >= footprint_lo) & (W[:, :2] <= footprint_hi), axis=1
            )
        else:
            in_footprint = np.ones(len(W), dtype=bool)

        # Start immediately above the support surface.  `obstacle_height` is a
        # validation height, not a destructive slicing height.
        obstacle_mask = in_footprint & (W[:, 2] > table_top_z)
        ground_mask = ~obstacle_mask
        rep.ground_plane = {
            "method": "z_histogram_mode",
            "table_top_z": round(table_top_z, 4),
            "mode_z": round(mode_z, 4),
            "obstacle_cut_z": round(table_top_z, 4),
            "object_validation_z": round(table_top_z + obstacle_height, 4),
            "peak_voxels": peak,
        }
        rep.ground_voxels = int(ground_mask.sum())
        rep.obstacle_voxels = int(obstacle_mask.sum())
        rep.support_voxels = support_count
        object_boxes, _ = axis_aligned_boxes(
            W[obstacle_mask], C[obstacle_mask],
            eps=max(cluster_eps, vs * 2.5),
            min_samples=2,
            min_cluster=min_cluster,
            support_top_z=table_top_z,
            min_height_m=obstacle_height,
            voxel_size=vs,
            primitive_mode=primitive_mode,
            cylinder_max_diameter=cylinder_max_diameter,
            cylinder_max_aspect=cylinder_max_aspect,
            cylinder_min_circularity=cylinder_min_circularity,
            max_footprint_aspect=max_footprint_aspect,
        )
        for i, box in enumerate(object_boxes, start=len(support_boxes)):
            box["id"] = i
    scene_boxes = support_boxes + object_boxes
    rep.support_boxes = support_boxes
    rep.obstacle_boxes = object_boxes
    rep.scene_boxes = scene_boxes
    robot_markers = read_scene_markers(extrinsics_path) if include_markers else []
    rep.robot_markers = robot_markers

    # 5. write outputs
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "cleaned_voxels.npz"),
                        points=W.astype(np.float32), colors=C,
                        voxel_size=np.float32(vs),
                        world_frame=np.asarray(meta["world_frame"]),
                        translation_unit=np.asarray("meter"))
    voxels_to_glb(W, C, vs).export(os.path.join(out_dir, "cleaned_voxels.glb"))
    if scene_boxes or robot_markers:
        primitives_to_glb(
            scene_boxes,
            inflate_m=box_inflate,
            markers=robot_markers,
        ).export(
            os.path.join(out_dir, "obstacles.glb"))
    with open(os.path.join(out_dir, "obstacles.json"), "w", encoding="utf-8") as f:
        json.dump({"world_frame": meta["world_frame"], "unit": "meter",
                   "box_inflation_m": box_inflate,
                   "primitive_mode": primitive_mode,
                   "boxes": scene_boxes,
                   "support_boxes": support_boxes,
                   "object_boxes": object_boxes,
                   "markers": robot_markers}, f, indent=2)
    with open(os.path.join(out_dir, "simplify_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep.to_dict(), f, indent=2)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voxels", help="path to voxels.npz")
    ap.add_argument("--extrinsics", default=None,
                    help="raw camera_extrinsics.json (for spatial gripper removal)")
    ap.add_argument("--out-dir", default=None, help="defaults to the voxels.npz folder")
    ap.add_argument("--gripper-radius", type=float, default=DEFAULT_GRIPPER_RADIUS_M)
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop voxels backed by fewer than N points")
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--cluster-eps", type=float, default=0.03,
                    help="DBSCAN neighbourhood radius in metres")
    ap.add_argument("--denoise-min-cluster", type=int, default=4,
                    help="delete only disconnected fragments smaller than this")
    ap.add_argument("--min-cluster", type=int, default=24,
                    help="minimum raised seed voxels for a component to become a primitive")
    ap.add_argument("--obstacle-height", type=float, default=0.03,
                    help="minimum raised height used to validate an object cluster")
    ap.add_argument("--table-thickness", type=float, default=0.06,
                    help="thickness of the retained coarse table/support slab")
    ap.add_argument("--surface-z", type=float, default=None,
                    help="override auto table-height detection with a fixed z (base frame)")
    ap.add_argument("--box-inflate", type=float, default=0.0,
                    help="metres to grow each object box in the visualization")
    ap.add_argument("--primitive-mode", choices=("auto", "box", "cylinder"), default="auto",
                    help="auto fits compact circular clusters as cylinders")
    ap.add_argument("--cylinder-max-diameter", type=float, default=0.20)
    ap.add_argument("--cylinder-max-aspect", type=float, default=1.6)
    ap.add_argument("--cylinder-min-circularity", type=float, default=0.82)
    ap.add_argument("--max-footprint-aspect", type=float, default=3.0,
                    help="reject thinner raised fragments as edge noise")
    ap.add_argument("--no-markers", action="store_true",
                    help="omit robot camera/gripper/head and world-origin markers")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.voxels))
    rep = simplify(
        args.voxels, args.extrinsics, out_dir,
        args.gripper_radius, args.min_count, args.min_conf,
        args.cluster_eps, args.min_cluster, args.obstacle_height,
        args.box_inflate, surface_z=args.surface_z,
        denoise_min_cluster=args.denoise_min_cluster,
        table_thickness=args.table_thickness,
        primitive_mode=args.primitive_mode,
        cylinder_max_diameter=args.cylinder_max_diameter,
        cylinder_max_aspect=args.cylinder_max_aspect,
        cylinder_min_circularity=args.cylinder_min_circularity,
        max_footprint_aspect=args.max_footprint_aspect,
        include_markers=not args.no_markers,
    )
    d = rep.to_dict()
    print(f"input voxels        : {d['input_voxels']}")
    print(f"  - gripper         : {d['removed_gripper']}")
    print(f"  - low count       : {d['removed_low_count']}")
    print(f"  - low conf        : {d['removed_low_conf']}")
    print(f"  - noise clusters  : {d['removed_noise_clusters']}")
    print(f"kept voxels         : {d['kept_voxels']}  (ground {d['ground_voxels']}, obstacle {d['obstacle_voxels']})")
    primitive_counts = {
        kind: sum(p.get("primitive", "box") == kind for p in d["scene_boxes"])
        for kind in ("box", "cylinder")
    }
    print(f"scene primitives    : {len(d['scene_boxes'])}  ({primitive_counts})")
    print(f"pose markers        : {len(d['robot_markers'])}")
    for b in d["scene_boxes"]:
        s = b["size_m"]
        print(f"    {b['primitive']} {b['id']} [{b['role']}]: center {b['center_m']} size {s} ({b['voxel_count']} vox)")
    print(f"written to {out_dir}: cleaned_voxels.npz/.glb, obstacles.json/.glb, simplify_report.json")


if __name__ == "__main__":
    main()
