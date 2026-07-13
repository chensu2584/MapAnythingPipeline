# G2 Pipeline

3D reconstruction pipeline for G2 robot camera captures using [MapAnything](https://github.com/facebookresearch/map-anything).

Each capture contains 3 RGB images (`head`, `hand_left`, `hand_right`) plus per-camera intrinsics JSONs. The pipeline undistorts the images, runs MapAnything multi-view inference to get a metric world-frame point cloud, and exports filtered GLB/PLY point clouds.

## Pipeline overview

```
MapAnythingTestData/<capture>/{head,hand_left,hand_right}.png + intrinsic_*.json
        │
        ▼  undistort.py            (Step A: OpenCV undistortion, adjusted K)
outputs/undistorted/<capture>/{name}.png + {name}_K.json
        │
        ▼  run_inference.py        (Step B: MapAnything inference, GPU)
outputs/<capture>/scene.glb + scene.ply + views.npz + summary.json
        │
        ▼  filter_export.py        (Step C: filtered re-export, CPU only)
outputs/<capture>/scene_filtered.glb + scene_filtered.ply
```

## Setup

Requires the MapAnything library (not bundled in this repo):

```bash
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything && pip install -e .
```

Other dependencies: `opencv-python`, `numpy`, `torch`, `trimesh` (installed with map-anything).

Expected data layout (paths are hardcoded at the top of each script):

- Input captures: `~/MapAnything/MapAnythingTestData/<capture>/`
- Outputs: `~/MapAnything/outputs/`

## Usage

```bash
# Step A: undistort all captures
python undistort.py

# Step B: inference on all captures (needs CUDA GPU)
python run_inference.py
# or a subset, with export filters applied to the GLB/PLY:
python run_inference.py --captures g2_smoke_20260702_142817 --max_radius 2.0

# Step C: re-export with filters from an existing views.npz (no GPU needed)
python filter_export.py --captures g2_smoke_20260702_142817 \
    --max_radius 2.0 --min_conf 0.5 --show_cameras
```

---

## Function reference

### `undistort.py` — Step A: undistortion preprocessing

For each capture and each of the 3 RGB images: builds the camera matrix from the intrinsics JSON, undistorts with OpenCV (`alpha=0`), crops to the valid ROI, and saves the undistorted PNG plus the adjusted intrinsics.

| Function | Description |
|---|---|
| `load_K_dist(intrinsic_path)` | Reads an intrinsics JSON (`Fx, Fy, Cx, Cy, k1, k2, p1, p2, k3`) and returns the 3×3 camera matrix `K` and the distortion vector `dist = [k1, k2, p1, p2, k3]`. |
| `undistort_image(img, K, dist)` | Undistorts one image: `getOptimalNewCameraMatrix(alpha=0)` → `initUndistortRectifyMap` → `remap` → crop to ROI. Shifts the new principal point by the crop offset. Returns `(undistorted_img, adjusted_K, roi)`. Falls back to the full image if the ROI is degenerate. |
| `main()` | Loops over `CAPTURES` × 3 views; writes `outputs/undistorted/<capture>/<name>.png` and `<name>_K.json` (adjusted K, output/original sizes, ROI) and prints per-image stats. |

### `run_inference.py` — Step B: MapAnything 3D reconstruction

Loads the undistorted images + adjusted intrinsics, runs `facebook/map-anything` multi-view inference on GPU, and saves per-capture reconstruction artifacts.

| Function | Description |
|---|---|
| `load_views(capture)` | Loads the 3 undistorted PNGs (as RGB uint8) and their adjusted `K` matrices into the MapAnything view format `[{"img", "intrinsics"}, ...]`. |
| `run_capture(model, capture, minibatch_size=None, max_radius=None, bbox=None, min_conf=None)` | Full per-capture inference: preprocesses views, runs `model.infer(...)` (memory-efficient, bf16 AMP, edge masking), unprojects depth to world-frame points, computes per-view stats (valid-pixel %, confidence, depth range, camera translation) and inter-camera baselines, optionally applies export filters (via `build_filter_mask` from `filter_export.py`; `views.npz` stays unfiltered), then writes `scene.glb`, `scene.ply`, `views.npz`, and `summary.json` to `outputs/<capture>/`. Returns the summary dict. |
| `main()` | CLI entry point (`--captures`, `--minibatch_size`, `--max_radius`, `--bbox`, `--min_conf`). Loads the model onto CUDA and runs each capture; on CUDA OOM, retries the capture with `minibatch_size=1`. |

**Outputs per capture** (`outputs/<capture>/`):

- `scene.glb` — colored point cloud of all 3 views merged (masked)
- `scene.ply` — same masked points as a raw PLY point cloud
- `views.npz` — per-view `depth_z`, `intrinsics`, `camera_pose` (4×4 cam2world), `mask`, `pts3d`, `img`, `conf` (unfiltered)
- `summary.json` — per-view stats, inter-camera baselines (m), point count, applied export filter (if any)

### `filter_export.py` — Step C: filtered point-cloud export

Re-exports filtered point clouds from an existing `views.npz` without GPU or re-inference. Geometry is verified against the stored `pts3d` and the original `scene.ply` before export.

| Function | Description |
|---|---|
| `camera_frustum_mesh(K, pose, img_hw, color, frustum_depth=0.15)` | Builds a solid frustum pyramid mesh for one camera: apex at the camera center, base at the image corners unprojected to `frustum_depth` meters, transformed to world frame. Used for the `--show_cameras` markers. |
| `build_filter_mask(pts3d, conf, max_radius=None, bbox=None, min_conf=None)` | Pointwise keep-mask over world-frame points. Filters are ANDed and each is optional: radius from world origin, world-frame bounding box, minimum confidence. Raises if `min_conf` is requested but no confidence data exists. Also imported by `run_inference.py` for inference-time export filtering. |
| `unproject_view(npz, name)` | Recomputes world-frame points for one view from `depth_z` + `intrinsics` + `camera_pose` using the same `depthmap_to_world_frame` utility as inference. Returns `(pts3d, valid_mask)`. |
| `fallback_colors(capture, name, target_hw)` | Approximate per-pixel colors when `views.npz` has no stored image: loads the undistorted PNG, center-crops to the depth-map aspect ratio, and resizes. Returns float RGB in [0, 1]. |
| `process_capture(capture, max_radius=None, bbox=None, min_conf=None, show_cameras=False)` | Full per-capture export: loads `views.npz`, reconstructs geometry (stored `pts3d` preferred, verified against fresh unprojection within 1 cm) and colors (stored `img` preferred, else `fallback_colors`), cross-checks point count/coordinates against `scene.ply`, applies `build_filter_mask`, optionally adds colored camera frustums (head=red, hand_left=green, hand_right=blue; GLB only), and writes `scene_filtered.glb` / `scene_filtered.ply`. Returns a stats dict (points before/after, kept %, filtered bbox). |
| `main()` | CLI entry point (`--captures`, `--max_radius`, `--bbox`, `--min_conf`, `--show_cameras`); processes each capture and prints a JSON summary. |

## Docs

- `PROJECT_LOG.md` — running project log
- `PLAN_SEMANTIC_VOXEL.md` — plan for the semantic voxel task
- `TECH_DETAIL_TASK2.md` — technical details for task 2
