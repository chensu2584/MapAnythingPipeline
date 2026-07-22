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
                   + scene_filtered_by_view.glb (optional; GUI default)
                   + scene_filtered_per_camera_k*.glb (experimental GUI option)
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

On the current G1 machine, capture tooling lives in
`/home/ck/MapAnythingTest/capture`, raw captures live in
`/home/ck/MapAnythingTest/TestData`, and Pipeline outputs live under
`/home/ck/MapAnythingTest/outputs*`. See `../capture/README.md` for the migrated
capture scripts, tests, documentation, and external robot dependencies.

## Usage

```bash
# Optional GUI: select captures, stages, output geometry mode and output folder
python pipeline_gui.py

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

# Optional speed mode: higher peak VRAM; CUDA OOM retries the safe path
python run_inference.py --captures g1_capture_20260721_115621 --fast-inference

# Diagnostics: unscaled model geometry, or the old model-depth/calibrated-pose hybrid
python run_inference.py --captures g2_smoke_20260702_142817 \
    --pose-export-mode model-relative-head-anchored
python run_inference.py --captures g2_smoke_20260702_142817 \
    --pose-export-mode calibrated-input

# Step C: re-export with filters from an existing views.npz (no GPU needed)
python filter_export.py --captures g2_smoke_20260702_142817 \
    --max_radius 2.0 --min_conf 0.5 --show_cameras

# Rig diagnostic: keep normal scene_filtered.glb/.ply and additionally write
# scene_filtered_by_view.glb with head/left/right in red/green/blue, plus small
# camera center/frustum markers and the world-origin XYZ frame
python filter_export.py --captures g1_capture_20260715_121059 \
    --max_radius 2.0 --show_cameras --show_grippers --color_by_view

# Experimental ChArUco-derived K policy: head=calibrated K, left=model K,
# right=model focal lengths + calibrated principal point. This adds RGB and,
# with --color_by_view, red/green/blue comparison GLBs without replacing output.
python filter_export.py --captures g1_capture_20260720_155200 \
    --show_cameras --color_by_view --per_camera_k_ab

# Step D: sparse occupancy grid from views.npz (no GPU needed)
python voxelize.py --max_radius 2.0 --show_grippers # all captures, 2 cm voxels
python voxelize.py --voxel_size 0.05 --captures g2_smoke_20260702_142817
```

For G1 `base_link` captures, `--show_grippers` reads the saved WBC Link7 poses,
cross-checks `base_T_Link_hand_l/r` against `/home/ck/robot_test/G1.urdf`,
and appends the fixed `0.14308 m` local-Z gripper-center displacement from the
Omnipicker URDF. It adds an orange left / cyan right tool-center marker to each
GLB and saves exact positions, URDF hashes, and cross-check errors in
`gripper_poses_base_link.json`. The legacy Omnipicker arm/base chain is not used.

All four steps auto-discover compatible captures when `--captures` is omitted.
Step A/B require a valid metric pose file by default; intentional pose-free,
arbitrary-scale inference requires the explicit `--allow-missing-poses` flag.

The GUI displays a live elapsed clock for the active stage and the complete
pipeline. Every completed stage is written to the live log. `run_inference`
additionally reports model-load, input-load, preprocessing, GPU inference,
reconstruction postprocessing, GLB/PLY export, compressed NPZ write, and total
times; per-capture timings are saved in `summary.json`.

Two conservative speed controls are available in the GUI:

- **Reuse unchanged undistorted inputs** is enabled by default. Step A fingerprints
  all three source images, intrinsics, pose mode, and provenance content. It skips
  OpenCV remap only when that fingerprint and every expected output match. Old
  manifests are recomputed once because they have no cache key.
- **Fast inference dense head** is opt-in. The local MapAnything implementation
  documents its memory-efficient dense-head path as slower; fast mode runs the
  three dense heads together and may use substantially more VRAM. An OOM
  automatically retries memory-efficient inference with minibatch size 1.

Image preprocessing remains enabled: pose/intrinsic preservation checks and edge
masking affect reconstruction validity and are not removed for speed. The GUI sends
all selected captures to one inference process, so model weights load once per batch.

## G2 support

`--robot g2` (or auto-detection from the folder layout) reads a G2 capture
session of `snapshot_*` folders instead of the flat G1 capture layout:

```
session_.../snapshot_.../
    head_rgb.png  hand_left_rgb.png  hand_right_rgb.png   (three colour views)
    head_depth_raw16.png                                  (uint16 millimetres)
    camera_extrinsics.json                                (intrinsics + base_T_camera + joints)
```

G2 already ships metric `base_T_camera` matrices in OpenCV RDF, so no forward
kinematics has to be re-derived. Step A validates them (SO(3), the redundant
quaternion/inverse/translation copies, camera synchronisation, the capture
script's own FK-vs-SDK check) and then rewrites them into the same
`camera_poses_opencv_cam2world.json` every later stage already consumes. Steps
B-D therefore do not need to know which robot produced the capture.

```bash
# Step A: undistort + register depth into the undistorted colour frame
python undistort.py --robot g2 \
    --data-root /path/to/session_20260721_232012 \
    --output-root /path/to/outputs

# Step B: inference, with the metric depth used only as a quality report
python run_inference.py --captures snapshot_20260721_232128_0001

# Step B, version 1: let the measured depth set the reconstruction scale
python run_inference.py --captures snapshot_20260721_232128_0001 \
    --pose-export-mode model-relative-head-anchored-depth-scaled

# Step B, version 2: feed the depth to the model as a fourth input modality,
# withholding 30 percent of pixels so the diagnostic stays honest
python run_inference.py --captures snapshot_20260721_232128_0001 \
    --depth-input --depth-holdout 0.3
```

### Metric depth: report, scale anchor, or model input

The head depth camera is metric, so it can play three different roles. They are
independent and can be combined:

| Role | Flag | What it does |
|---|---|---|
| Quality report | *(always on when depth exists)* | Fits predicted vs measured head depth and writes the result to `summary.json`; does not change the reconstruction |
| Scale anchor | `--pose-export-mode ...-depth-scaled` | Uses that fit as the similarity scale instead of the three camera baselines |
| Model input | `--depth-input` | Passes `depth_z` + `is_metric_scale` into `model.infer` as a per-view prior |

The report is what makes the other two judgeable, so it always runs. It records
the fitted scale, residual RMSE/median/P95, the inlier ratio against an absolute
2 cm / 5 % tolerance, and an affine test `reference ~= a * model + b`. **A `b`
well above the residual RMSE means the error is not a pure scale error and no
single global scalar can fix the reconstruction** - the baseline estimator
cannot detect this at all, because three camera-centre distances contain no
information about scene depth.

The depth-scaled mode refuses to export a scale it cannot defend: an implausible
scale, fewer than 5000 co-visible pixels, or an inlier ratio below 0.5 all fall
back to the baseline fit and record the rejected depth fit beside it.

`--depth-input` and the report interact: depth fed to the model will of course
agree with the model afterwards, so `summary.json` marks such a view
`was_fed_to_model` and warns that the agreement is circular. Use
`--depth-holdout` to keep a random subset of pixels out of the model, or judge
the two hand views, which are never fed.

## Camera-pose contract

For calibrated G1 captures, `camera_poses_opencv_cam2world.json` must provide all
three 4×4 poses as RGB optical-center, OpenCV RDF (`+X` right, `+Y` down, `+Z`
forward), camera-to-world matrices in meters. The file must declare its
`world_frame`; current G1 GUI captures use `base_link`. Legacy head-centered captures
may declare `head_rgb_opencv_at_capture`, in which case the head pose must be identity.

The pipeline validates matrix shape/finite values, SO(3), homogeneous bottom
row, units, direction, view completeness, image/K consistency, and pose
preservation through `preprocess_inputs`. Provenance files are copied through to
the reconstruction output.

MapAnything treats supplied poses as conditioning priors and internally makes
them relative to view 0; its predicted depth and poses form one network-estimated
geometry, while the input poses are not hard constraints. Real G1 A/B testing showed
that reprojecting the network depth with the exact calibrated poses can split an
otherwise aligned scene by about 5 cm.

The default
`--pose-export-mode model-relative-head-anchored-baseline-scaled` preserves the
model-predicted three-camera geometry while fitting one uniform scale from all three
calibrated/model camera baseline lengths. It applies the same scale to model depth
and head-relative camera translations before anchoring:

```text
s = argmin Σij (s * model_baseline_ij - calibrated_baseline_ij)^2
model_head_T_camera = inverse(model_reference_T_head)
                      @ model_reference_T_camera
model_head_T_camera.translation *= s
depth_z *= s
world_T_camera = calibrated_world_T_head @ model_head_T_camera
```

This is one similarity transform of the entire reconstruction: it keeps the head
exactly in the declared calibrated world frame and does not split the views.
`model-relative-head-anchored` preserves the previous unscaled model geometry for
diagnosis; `calibrated-input` preserves the former hybrid. RGB-only `--ignore-poses`
keeps unanchored model pose and arbitrary scale.

Real G1 scale validation:

| capture | fitted scale | baseline RMSE before/after |
|---|---:|---:|
| `170603` | `0.897974` | `53.31 / 2.50 mm` |
| `170700` | `0.893756` | `55.80 / 2.99 mm` |
| `102356` | `0.851619` | `93.45 / 2.25 mm` |

All three completed inference→filter_export→voxelize checks; PLY replay was exact
and voxel IoU was 1.0000. The different scales show why the correction is fitted per
capture rather than hard-coded.

The scale fit constrains camera-center distances only. Final model-relative camera
rotations and translation directions still come from MapAnything. Therefore a
remaining cross-view offset is not, by itself, proof of bad robot extrinsics. Use a
shared rigid ChArUco/AprilTag target over multiple captures: a stable per-camera SE(3)
residual supports fixed extrinsic/optical-frame error; capture-varying residuals
support model pose/depth error; joint-dependent residuals support FK/frame error.

---

## Function reference

### `undistort.py` — Step A: undistortion preprocessing

For each capture and each of the 3 RGB images: builds the camera matrix from the intrinsics JSON, undistorts with OpenCV (`alpha=0`), crops to the valid ROI, and saves the undistorted PNG plus the adjusted intrinsics.

| Function | Description |
|---|---|
| `load_K_dist(intrinsic_path)` | Reads an intrinsics JSON (`Fx, Fy, Cx, Cy, k1, k2, p1, p2, k3`) and returns the 3×3 camera matrix `K` and the distortion vector `dist = [k1, k2, p1, p2, k3]`. |
| `undistort_image(img, K, dist)` | Undistorts one image: `getOptimalNewCameraMatrix(alpha=0)` → `initUndistortRectifyMap` → `remap` → crop to ROI. Shifts the new principal point by the crop offset. Returns `(undistorted_img, adjusted_K, roi)`. Falls back to the full image if the ROI is degenerate. |
| `main()` | Resolves explicit `--captures` or auto-discovers compatible folders; writes `outputs/undistorted/<capture>/<name>.png`, `<name>_K.json`, copied pose/provenance, and `pipeline_preprocess_manifest.json`. `--reuse-existing` skips remap only after a complete content-fingerprint match. |

### `run_inference.py` — Step B: MapAnything 3D reconstruction

Loads the undistorted images + adjusted intrinsics, runs `facebook/map-anything` multi-view inference on GPU, and saves per-capture reconstruction artifacts.

| Function | Description |
|---|---|
| `load_views(capture)` | Loads and validates the 3 undistorted PNGs, adjusted `K` matrices, and metric RDF cam2world poses into MapAnything view dictionaries. |
| `run_capture(model, capture, ..., pose_export_mode=DEFAULT_POSE_EXPORT_MODE)` | Full per-capture inference: preprocesses views, runs `model.infer(...)`, baseline-scales and head-anchors self-consistent model geometry by default, unprojects corrected depth, computes diagnostics and optional filters, then writes reconstruction, provenance, and section timings. |
| `main()` | CLI entry point. Supports root overrides, auto-discovery, `--validate-only`, explicit pose-free opt-in, and opt-in `--fast-inference`; full inference loads the model once and retries CUDA OOM through the safe memory-efficient path. |

**Outputs per capture** (`outputs/<capture>/`):

- `scene.glb` — colored point cloud of all 3 views merged (masked)
- `scene.ply` — same masked points as a raw PLY point cloud
- `views.npz` — per-view corrected `depth_z`, `intrinsics`, final export `camera_pose`, original `calibrated_input_camera_pose`, `model_camera_pose_head_reference`, model metric factor, applied similarity scale, `mask`, self-consistent `pts3d`, `img`, and `conf` (unfiltered)
- `camera_poses_used_for_export.json` — exact final pose set, effective mode, world frame, head anchor and full baseline-scale fit report
- `summary.json` — input pose contract, requested/effective export mode, scale fit, anchor, model-vs-input and export-vs-calibrated diagnostics, final baselines, point count, filters, and per-stage timings

### `filter_export.py` — Step C: filtered point-cloud export

Re-exports filtered point clouds from an existing `views.npz` without GPU or re-inference. Geometry is verified against the stored `pts3d` and the original `scene.ply` before export.

| Function | Description |
|---|---|
| `camera_frustum_mesh(K, pose, img_hw, color, frustum_depth=0.06)` | Builds a compact solid frustum for one camera; the 6 cm default avoids obscuring nearby geometry. `--show_cameras` also adds an exact center sphere for every camera and a small RGB XYZ frame at the world origin. |
| `build_filter_mask(pts3d, conf, max_radius=None, bbox=None, min_conf=None)` | Pointwise keep-mask over world-frame points. Filters are ANDed and each is optional: radius from world origin, world-frame bounding box, minimum confidence. Raises if `min_conf` is requested but no confidence data exists. Also imported by `run_inference.py` for inference-time export filtering. |
| `unproject_view(npz, name)` | Recomputes world-frame points for one view from `depth_z` + `intrinsics` + `camera_pose` using the same `depthmap_to_world_frame` utility as inference. Returns `(pts3d, valid_mask)`. |
| `fallback_colors(capture, name, target_hw)` | Approximate per-pixel colors when `views.npz` has no stored image: loads the undistorted PNG, center-crops to the depth-map aspect ratio, and resizes. Returns float RGB in [0, 1]. |
| `process_capture(capture, ...)` | Full per-capture export: loads `views.npz`, reconstructs geometry, replays any inference-time filter before cross-checking `scene.ply`, and applies new filters. It always writes the normal RGB `scene_filtered.glb/.ply`; `--color_by_view` adds the red/green/blue diagnostic; `--show_grippers` resolves and overlays G1 tool centers; `--per_camera_k_ab` adds experimental RGB and optional view-colored GLBs using the ChArUco-derived per-camera K policy. |
| `main()` | CLI entry point (`--captures`, filters, scene/camera/gripper markers, `--color_by_view`, `--per_camera_k_ab`); processes each capture and prints a JSON summary. |

### `voxelize.py` — Step D: sparse occupancy grid

Bins the merged world-frame point cloud from `views.npz` into a fixed-resolution sparse voxel grid (`idx = floor((pts − origin) / voxel_size)`), aggregating per-voxel point count, mean color, and max confidence. Pure numpy for the grid; Open3D voxelizes the same points as a cross-check (skipped with a warning if Open3D can't load, e.g. headless server without libGL). Depends only on `numpy`/`open3d`/`trimesh` — no torch, no GPU. Implements Task 1 (P1) of `PLAN_SEMANTIC_VOXEL.md`.

| Function | Description |
|---|---|
| `build_filter_mask(pts3d, conf, max_radius=None, bbox=None, min_conf=None)` | Same pre-filter semantics as `filter_export.build_filter_mask`, duplicated locally so this script stays free of that module's torch/mapanything imports. |
| `load_points(capture)` | Loads and merges the masked points, colors ([0, 1] float), and confidences of all 3 views from `views.npz` into flat `(N, 3)` / `(N,)` arrays. |
| `voxelize_points(pts, cols, conf, voxel_size, origin, dims)` | Core numpy voxelization: integer binning → `np.unique` on flattened indices → per-voxel aggregation (`counts` via `bincount`, mean colors via weighted `bincount`, max conf via `np.maximum.at`). Returns sparse arrays sorted by flat voxel index. |
| `crosscheck_open3d(pts, cols, voxel_size, origin, dims, np_indices)` | Builds an Open3D `VoxelGrid` from the same points over identical bounds and asserts ≥ 99.9 % IoU between the occupied-voxel index sets (float32 vs float64 boundary rounding tolerance). Returns None and warns if Open3D is unavailable. |
| `voxels_to_glb_mesh(indices, colors, voxel_size, origin)` | Single merged trimesh with one cube per occupied voxel (2 % shrunk for visual separation), per-face mean colors, and the same 180° X flip as `predictions_to_glb` so `voxels.glb` overlays `scene.glb`. |
| `process_capture(capture, voxel_size, max_radius=None, bbox=None, min_conf=None)` | Full per-capture run: load + pre-filter points, derive grid frame (`--bbox` fixes origin/extent, else tight bounds), voxelize, cross-check, and write `voxels.npz` + `voxels.glb`; `--show_grippers` adds the same resolved G1 tool-center markers to the GLB. Returns a stats dict. |
| `main()` | CLI entry point (`--captures`, `--voxel_size` [default 0.02 m], `--max_radius`, `--bbox`, `--min_conf`); processes each capture and prints a JSON summary. |

**Outputs per capture** (`outputs/<capture>/`):

- `voxels.npz` — sparse grid: `indices (N,3) int32`, `origin (3,)`, `voxel_size`, `dims (3,)`, `counts`, `colors (N,3) uint8`, `conf`, plus `labels`/`label_scores` reserved as zeros for Task 2 (semantic lift)
- `voxels.glb` — colored cube per occupied voxel, viewable alongside `scene.glb`

## Docs

- `PROJECT_LOG.md` — running project log
- `PLAN_SEMANTIC_VOXEL.md` — plan for the semantic voxel task
- `TECH_DETAIL_TASK2.md` — technical details for task 2
