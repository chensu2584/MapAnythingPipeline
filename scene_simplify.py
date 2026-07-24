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
     IS trustworthy is the wrist-camera pose (ground-truth verified).  The real
     gripper is the only thing within ~0.4 m of each wrist camera -- the table
     is ~0.6 m below it -- so a sphere about each wrist-camera centre carves the
     gripper out cleanly without touching the scene.

  2. Denoise -- drop under-observed voxels, then DBSCAN and delete any cluster
     too small to be a real object (floating measurement specks).

  3. Simplify -- fit the dominant ground/table plane, split everything above it
     into obstacle clusters, and emit one axis-aligned box per cluster.  The
     boxes are the coarse-planning obstacle set; a shelf of shaped blue-bin
     voxels becomes a single box a planner can inflate and avoid.

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
from sklearn.cluster import DBSCAN

# The wrist cameras are the only views that see the gripper, and the gripper is
# the nearest thing to them; the table sits far enough below that a sphere this
# size never reaches it.  Tunable, but 0.4 m is a safe default for the G2 wrist.
DEFAULT_GRIPPER_RADIUS_M = 0.40
HAND_CAMERA_KEYS = ("hand_left_rgb", "hand_right_rgb")


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
    obstacle_boxes: list = field(default_factory=list)

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
    data = json.load(open(extrinsics_path, encoding="utf-8"))
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
    densest horizontal layer is the table surface, and its upper edge is where
    objects begin.

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
    # walk up from the peak while the layer stays dense (the table slab has
    # thickness -- rim + top + noise); the top of that slab is where objects rise
    thresh = hist[peak] * 0.15
    top = peak
    while top + 1 < len(hist) and hist[top + 1] >= thresh:
        top += 1
    table_top_z = float(edges[top + 1]) if top + 1 < len(edges) else float(edges[-1])
    return table_top_z, mode_z, int(hist[peak])


def axis_aligned_boxes(points: np.ndarray, colors: np.ndarray, eps: float,
                       min_samples: int, min_cluster: int):
    """Cluster obstacle points and describe each cluster as an AABB."""
    if len(points) == 0:
        return [], np.array([], dtype=int)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    boxes = []
    for lab in sorted(set(labels)):
        if lab < 0:
            continue
        m = labels == lab
        if m.sum() < min_cluster:
            continue
        pts = points[m]
        lo = pts.min(0)
        hi = pts.max(0)
        boxes.append({
            "id": len(boxes),
            "center_m": ((lo + hi) / 2).round(4).tolist(),
            "size_m": (hi - lo).round(4).tolist(),
            "min_m": lo.round(4).tolist(),
            "max_m": hi.round(4).tolist(),
            "voxel_count": int(m.sum()),
            "color": [int(v) for v in colors[m].mean(0)],
        })
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


def boxes_to_glb(boxes: list, inflate_m: float = 0.0):
    """Translucent inflated boxes for the planner's obstacle set."""
    scene = trimesh.Scene()
    flip = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    for b in boxes:
        size = np.asarray(b["size_m"]) + 2 * inflate_m
        box = trimesh.creation.box(extents=np.maximum(size, 1e-3))
        T = np.eye(4)
        T[:3, 3] = b["center_m"]
        box.apply_transform(T)
        box.apply_transform(flip)
        col = b.get("color", [200, 60, 60])
        box.visual.vertex_colors = np.array([*col, 120], np.uint8)
        scene.add_geometry(box)
    return scene


def simplify(voxels_path: str, extrinsics_path: str | None, out_dir: str,
             gripper_radius: float, min_count: int, min_conf: float | None,
             cluster_eps: float, min_cluster: int, obstacle_height: float,
             box_inflate: float, surface_z: float | None = None):
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

    # 3. DBSCAN denoise: delete clusters too small to be real objects
    idx_keep = np.where(keep)[0]
    if len(idx_keep):
        labels = DBSCAN(eps=cluster_eps, min_samples=2).fit_predict(world[idx_keep])
        drop = np.zeros(len(idx_keep), bool)
        for lab in set(labels):
            m = labels == lab
            if lab < 0 or m.sum() < min_cluster:
                drop |= m
        rep.removed_noise_clusters = int(drop.sum())
        keep[idx_keep[drop]] = False

    W = world[keep]
    C = colors[keep]
    rep.kept_voxels = int(keep.sum())

    # 4. support surface (table) + obstacle split via height histogram
    boxes = []
    if len(W) >= 3:
        if surface_z is not None:
            table_top_z, mode_z, peak = float(surface_z), float(surface_z), -1
        else:
            # the table sits a roughly fixed drop below the fixed head camera and
            # well above the floor; restrict the mode search to that band so a
            # denser floor/base slab cannot steal it.
            band = (head_z - 1.0, head_z - 0.4) if head_z is not None else None
            table_top_z, mode_z, peak = find_support_surface(W, z_band=band)
        cut = table_top_z + obstacle_height
        obstacle_mask = W[:, 2] > cut          # objects standing on the table
        ground_mask = W[:, 2] <= cut
        rep.ground_plane = {"method": "z_histogram_mode", "table_top_z": round(table_top_z, 4),
                            "mode_z": round(mode_z, 4), "obstacle_cut_z": round(cut, 4),
                            "peak_voxels": peak}
        rep.ground_voxels = int(ground_mask.sum())
        rep.obstacle_voxels = int(obstacle_mask.sum())
        boxes, _ = axis_aligned_boxes(W[obstacle_mask], C[obstacle_mask],
                                      eps=cluster_eps, min_samples=2,
                                      min_cluster=min_cluster)
    rep.obstacle_boxes = boxes

    # 5. write outputs
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "cleaned_voxels.npz"),
                        points=W.astype(np.float32), colors=C,
                        voxel_size=np.float32(vs),
                        world_frame=np.asarray(meta["world_frame"]),
                        translation_unit=np.asarray("meter"))
    voxels_to_glb(W, C, vs).export(os.path.join(out_dir, "cleaned_voxels.glb"))
    if boxes:
        boxes_to_glb(boxes, inflate_m=box_inflate).export(
            os.path.join(out_dir, "obstacles.glb"))
    with open(os.path.join(out_dir, "obstacles.json"), "w", encoding="utf-8") as f:
        json.dump({"world_frame": meta["world_frame"], "unit": "meter",
                   "box_inflation_m": box_inflate, "boxes": boxes}, f, indent=2)
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
    ap.add_argument("--min-cluster", type=int, default=40,
                    help="clusters smaller than this many voxels are noise")
    ap.add_argument("--obstacle-height", type=float, default=0.03,
                    help="metres above the support surface to count as an obstacle")
    ap.add_argument("--surface-z", type=float, default=None,
                    help="override auto table-height detection with a fixed z (base frame)")
    ap.add_argument("--box-inflate", type=float, default=0.0,
                    help="metres to grow each obstacle box (safety margin) in the glb/json")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.voxels))
    rep = simplify(args.voxels, args.extrinsics, out_dir,
                   args.gripper_radius, args.min_count, args.min_conf,
                   args.cluster_eps, args.min_cluster, args.obstacle_height,
                   args.box_inflate, surface_z=args.surface_z)
    d = rep.to_dict()
    print(f"input voxels        : {d['input_voxels']}")
    print(f"  - gripper         : {d['removed_gripper']}")
    print(f"  - low count       : {d['removed_low_count']}")
    print(f"  - low conf        : {d['removed_low_conf']}")
    print(f"  - noise clusters  : {d['removed_noise_clusters']}")
    print(f"kept voxels         : {d['kept_voxels']}  (ground {d['ground_voxels']}, obstacle {d['obstacle_voxels']})")
    print(f"obstacle boxes      : {len(d['obstacle_boxes'])}")
    for b in d["obstacle_boxes"]:
        s = b["size_m"]
        print(f"    box {b['id']}: center {b['center_m']} size {s} ({b['voxel_count']} vox)")
    print(f"written to {out_dir}: cleaned_voxels.npz/.glb, obstacles.json/.glb, simplify_report.json")


if __name__ == "__main__":
    main()
