# G1/G2 MapAnything Pipeline

3D reconstruction pipeline for G1/G2 robot camera captures using [MapAnything](https://github.com/facebookresearch/map-anything).

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
        │
        ▼  voxelize.py             (Step D: occupancy grid, CPU only)
outputs/<capture>/voxels.npz + voxels.glb
```

## Setup

See `requirements.txt`. Two tiers:

- **Voxelization only** (`voxelize.py`): `pip install numpy open3d trimesh` — pure CPU, runs on Apple Silicon macOS or any Linux PC given a copied `views.npz`.
- **Full pipeline** (Steps A–C) additionally requires `opencv-python`, `torch` (CUDA), and the MapAnything library (not on PyPI):

```bash
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything && pip install -e .
```

Expected data layout (defaults are resolved relative to this repository;
override them with CLI flags or the legacy `G2_DATA_ROOT` / `G2_OUT_ROOT`
environment variables):

- Input captures: `~/MapAnything/MapAnythingTestData/<capture>/`
- Outputs: `~/MapAnything/outputs/`

## Usage

```bash
# Current G1 data on ck's machine
export G2_DATA_ROOT=/home/ck/MapAnythingTest/TestData
export G2_OUT_ROOT=/home/ck/MapAnythingTest/outputs

# Step A: auto-discover and undistort every compatible capture
python undistort.py
# or process only the new GUI capture
python undistort.py --captures g1_capture_20260715_121059

# Validate image/K/pose loading and MapAnything preprocessing without CUDA
python run_inference.py --captures g1_capture_20260715_121059 --validate-only

# Step B: inference on all captures (needs CUDA GPU)
python run_inference.py
# or a subset, with export filters applied to the GLB/PLY:
python run_inference.py --captures g2_smoke_20260702_142817 --max_radius 2.0

# Step C: re-export with filters from an existing views.npz (no GPU needed)
python filter_export.py --captures g2_smoke_20260702_142817 \
    --max_radius 2.0 --min_conf 0.5 --show_cameras

# Rig diagnostic: head/left/right points in red/green/blue plus three frustums
python filter_export.py --captures g1_capture_20260715_121059 \
    --max_radius 2.0 --show_cameras --color_by_view --frustum_depth 0.35

# Step D: sparse occupancy grid from views.npz (no GPU needed)
python voxelize.py --max_radius 2.0                 # all captures, 2 cm voxels
python voxelize.py --voxel_size 0.05 --captures g2_smoke_20260702_142817
```

All four steps auto-discover compatible captures when `--captures` is omitted.
Step A/B require a valid metric pose file by default; intentional pose-free,
arbitrary-scale inference requires the explicit `--allow-missing-poses` flag.

## Camera-pose contract

For calibrated G1 captures, `camera_poses_opencv_cam2world.json` must provide all
three 4×4 poses as RGB optical-center, OpenCV RDF (`+X` right, `+Y` down, `+Z`
forward), camera-to-world matrices in meters. The file must declare its
`world_frame`; the current GUI capture uses `head_rgb_opencv_at_capture`, so the
head pose must be identity.

The pipeline validates matrix shape/finite values, SO(3), homogeneous bottom
row, units, direction, view completeness, image/K consistency, and pose
preservation through `preprocess_inputs`. Provenance files are copied through to
the reconstruction output.

MapAnything treats supplied poses as conditioning priors and internally makes
them relative to view 0; its predicted output poses are not hard constraints.
Therefore `run_inference.py` uses the exact calibrated input pose for final depth
unprojection and exports it as `<view>_camera_pose`. The network prediction is
retained separately as `<view>_model_camera_pose_head_reference` for diagnostic
comparison.

---

## Function reference

### `undistort.py` — Step A: undistortion preprocessing

For each capture and each of the 3 RGB images: builds the camera matrix from the intrinsics JSON, undistorts with OpenCV (`alpha=0`), crops to the valid ROI, and saves the undistorted PNG plus the adjusted intrinsics.

| Function | Description |
|---|---|
| `load_K_dist(intrinsic_path)` | Reads an intrinsics JSON (`Fx, Fy, Cx, Cy, k1, k2, p1, p2, k3`) and returns the 3×3 camera matrix `K` and the distortion vector `dist = [k1, k2, p1, p2, k3]`. |
| `undistort_image(img, K, dist)` | Undistorts one image: `getOptimalNewCameraMatrix(alpha=0)` → `initUndistortRectifyMap` → `remap` → crop to ROI. Shifts the new principal point by the crop offset. Returns `(undistorted_img, adjusted_K, roi)`. Falls back to the full image if the ROI is degenerate. |
| `main()` | Resolves explicit `--captures` or auto-discovers compatible folders; writes `outputs/undistorted/<capture>/<name>.png`, `<name>_K.json`, copied pose/provenance, and `pipeline_preprocess_manifest.json`. |

### `run_inference.py` — Step B: MapAnything 3D reconstruction

Loads the undistorted images + adjusted intrinsics, runs `facebook/map-anything` multi-view inference on GPU, and saves per-capture reconstruction artifacts.

| Function | Description |
|---|---|
| `load_views(capture)` | Loads and validates the 3 undistorted PNGs, adjusted `K` matrices, and metric RDF cam2world poses into MapAnything view dictionaries. |
| `run_capture(model, capture, minibatch_size=None, max_radius=None, bbox=None, min_conf=None)` | Full per-capture inference: preprocesses views, runs `model.infer(...)` (memory-efficient, bf16 AMP, edge masking), unprojects depth to world-frame points, computes per-view stats (valid-pixel %, confidence, depth range, camera translation) and inter-camera baselines, optionally applies export filters (via `build_filter_mask` from `filter_export.py`; `views.npz` stays unfiltered), then writes `scene.glb`, `scene.ply`, `views.npz`, and `summary.json` to `outputs/<capture>/`. Returns the summary dict. |
| `main()` | CLI entry point. Supports root overrides, auto-discovery, `--validate-only`, and explicit pose-free opt-in; full inference loads the model on `--device` and retries CUDA OOM with `minibatch_size=1`. |

**Outputs per capture** (`outputs/<capture>/`):

- `scene.glb` — colored point cloud of all 3 views merged (masked)
- `scene.ply` — same masked points as a raw PLY point cloud
- `views.npz` — per-view `depth_z`, `intrinsics`, exact calibrated `camera_pose`, model-predicted diagnostic pose, `mask`, `pts3d`, `img`, `conf` (unfiltered)
- `summary.json` — pose contract/provenance, preprocessing checks, model-vs-input pose diagnostics, per-view stats, calibrated baselines (m), point count, and export filter

### `filter_export.py` — Step C: filtered point-cloud export

Re-exports filtered point clouds from an existing `views.npz` without GPU or re-inference. Geometry is verified against the stored `pts3d` and the original `scene.ply` before export.

| Function | Description |
|---|---|
| `camera_frustum_mesh(K, pose, img_hw, color, frustum_depth=0.15)` | Builds a solid frustum pyramid mesh for one camera: apex at the camera center, base at the image corners unprojected to `frustum_depth` meters, transformed to world frame. Used for the `--show_cameras` markers. |
| `build_filter_mask(pts3d, conf, max_radius=None, bbox=None, min_conf=None)` | Pointwise keep-mask over world-frame points. Filters are ANDed and each is optional: radius from world origin, world-frame bounding box, minimum confidence. Raises if `min_conf` is requested but no confidence data exists. Also imported by `run_inference.py` for inference-time export filtering. |
| `unproject_view(npz, name)` | Recomputes world-frame points for one view from `depth_z` + `intrinsics` + `camera_pose` using the same `depthmap_to_world_frame` utility as inference. Returns `(pts3d, valid_mask)`. |
| `fallback_colors(capture, name, target_hw)` | Approximate per-pixel colors when `views.npz` has no stored image: loads the undistorted PNG, center-crops to the depth-map aspect ratio, and resizes. Returns float RGB in [0, 1]. |
| `process_capture(capture, ...)` | Full per-capture export: loads `views.npz`, reconstructs geometry, replays any inference-time filter before cross-checking `scene.ply`, applies new filters, and optionally adds calibrated frustums. `--color_by_view` exports head/left/right as red/green/blue to diagnose rig separation. |
| `main()` | CLI entry point (`--captures`, `--max_radius`, `--bbox`, `--min_conf`, `--show_cameras`); processes each capture and prints a JSON summary. |

### `voxelize.py` — Step D: sparse occupancy grid

Bins the merged world-frame point cloud from `views.npz` into a fixed-resolution sparse voxel grid (`idx = floor((pts − origin) / voxel_size)`), aggregating per-voxel point count, mean color, and max confidence. Pure numpy for the grid; Open3D voxelizes the same points as a cross-check (skipped with a warning if Open3D can't load, e.g. headless server without libGL). Depends only on `numpy`/`open3d`/`trimesh` — no torch, no GPU. Implements Task 1 (P1) of `PLAN_SEMANTIC_VOXEL.md`.

| Function | Description |
|---|---|
| `build_filter_mask(pts3d, conf, max_radius=None, bbox=None, min_conf=None)` | Same pre-filter semantics as `filter_export.build_filter_mask`, duplicated locally so this script stays free of that module's torch/mapanything imports. |
| `load_points(capture)` | Loads and merges the masked points, colors ([0, 1] float), and confidences of all 3 views from `views.npz` into flat `(N, 3)` / `(N,)` arrays. |
| `voxelize_points(pts, cols, conf, voxel_size, origin, dims)` | Core numpy voxelization: integer binning → `np.unique` on flattened indices → per-voxel aggregation (`counts` via `bincount`, mean colors via weighted `bincount`, max conf via `np.maximum.at`). Returns sparse arrays sorted by flat voxel index. |
| `crosscheck_open3d(pts, cols, voxel_size, origin, dims, np_indices)` | Builds an Open3D `VoxelGrid` from the same points over identical bounds and asserts ≥ 99.9 % IoU between the occupied-voxel index sets (float32 vs float64 boundary rounding tolerance). Returns None and warns if Open3D is unavailable. |
| `voxels_to_glb_mesh(indices, colors, voxel_size, origin)` | Single merged trimesh with one cube per occupied voxel (2 % shrunk for visual separation), per-face mean colors, and the same 180° X flip as `predictions_to_glb` so `voxels.glb` overlays `scene.glb`. |
| `process_capture(capture, voxel_size, max_radius=None, bbox=None, min_conf=None)` | Full per-capture run: load + pre-filter points, derive grid frame (`--bbox` fixes origin/extent, else tight bounds), voxelize, cross-check, write `voxels.npz` + `voxels.glb`. Returns a stats dict. |
| `main()` | CLI entry point (`--captures`, `--voxel_size` [default 0.02 m], `--max_radius`, `--bbox`, `--min_conf`); processes each capture and prints a JSON summary. |

**Outputs per capture** (`outputs/<capture>/`):

- `voxels.npz` — sparse grid: `indices (N,3) int32`, `origin (3,)`, `voxel_size`, `dims (3,)`, `counts`, `colors (N,3) uint8`, `conf`, plus `labels`/`label_scores` reserved as zeros for Task 2 (semantic lift)
- `voxels.glb` — colored cube per occupied voxel, viewable alongside `scene.glb`

## Docs

- `PROJECT_LOG.md` — running project log
- `PLAN_SEMANTIC_VOXEL.md` — plan for the semantic voxel task
- `TECH_DETAIL_TASK2.md` — technical details for task 2
