# G1 GUI Capture → MapAnything Pipeline 审查（2026-07-15）

## 范围与数据

- Pipeline：`/home/ck/MapAnythingTest/MapAnythingPipeline`
- MapAnything 源码：`/home/ck/MapAnythingTest/map-anything`
- 新 capture：`/home/ck/MapAnythingTest/TestData/g1_capture_20260715_121059`
- 新 capture 在 `TestData` Git 仓库中仍为 untracked；本次未修改原始 capture。

## 审查结论

新 capture 的数据契约可以被正确处理：三张 RGB、原始内参/畸变、三路 pose 均
自洽。pose 声明为 OpenCV RDF cam2world，单位 meter，共同 world 为
`head_rgb_opencv_at_capture`，head pose 为 identity。

原 pipeline 有四个实质风险：

1. Step A/B/C/D 默认 capture 列表硬编码为 `g_1_Test_1..4`，新 capture 默认不会运行。
2. pose 缺失时 Step A/B 静默退化到任意尺度，可能产生“看似成功”的错误结果。
3. loader 只取 `poses` 字段，不校验 convention、方向、单位、刚体矩阵或 K/图像尺寸；
   注释还把所有 world 错写成 robot `end`。
4. MapAnything 内部先把 pose 变换到第 0 视图，并仅将其作为条件先验；输出 pose 是
   网络重新预测的。旧 pipeline 用预测 pose 做反投影并保存，无法保证输出严格注册到
   capture 标定坐标系。

## 已实现修正

- `capture_contract.py`：自动发现、路径约束、内参/畸变校验、pose contract 和刚体校验。
- `undistort.py`：CLI roots/captures、默认要求 metric pose、先完整验证再写输出、复制
  provenance、输出 `pipeline_preprocess_manifest.json`。
- `run_inference.py`：严格读取图像/K/pose，显式 metric 标记，验证 preprocessing 不改变
  pose，增加 `--validate-only`。有标定 pose 时，最终深度反投影固定使用输入 pose；模型
  预测 pose 另存为 `<view>_model_camera_pose_head_reference` 并报告偏差。
- `filter_export.py` / `voxelize.py`：自动发现含 `views.npz` 的输出并支持 `--output-root`。
- `tests/test_capture_contract.py`：6 个回归测试，包括方向反转、非 SO(3)、head-world 不一致。

## 真实数据验证结果

原始图像/内参：

| View | 原始分辨率 | Fx / Fy | Cx / Cy |
|---|---:|---:|---:|
| head | 1280×720 | 645.264 / 644.381 | 642.154 / 362.271 |
| hand_left | 848×480 | 435.384 / 434.830 | 423.830 / 241.469 |
| hand_right | 848×480 | 431.547 / 430.939 | 421.239 / 236.622 |

pose 数值：

| View | det(R) | 最大正交误差 | 相机中心（head world，m） |
|---|---:|---:|---|
| head | 1.0 | 0 | [0, 0, 0] |
| hand_left | 1.0 | 1.78e-15 | [-0.492144, 0.058303, -0.293802] |
| hand_right | 1.0 | 1.89e-15 | [0.530125, -0.053896, -0.335664] |

baseline：head-left 0.576129 m，head-right 0.629768 m，left-right 1.029260 m。

Step A 在 `/tmp/mapanything_pipeline_audit` 的输出：head 1279×719，左右腕均
847×479。源 pose 与复制文件 SHA-256 同为
`7288f9a99c063771b12a9639592da2b9556707692d340ef255ea2122c7f04ccd`。

Step B `--validate-only` 实际执行了 MapAnything `preprocess_inputs`：三视图统一为
518×294；每一路 pose 的 preprocessing 最大绝对差均为 0.0。6 个 unittest 全部通过。
GUI 同时生成的 `mapanything_views_input.npz` 与 pose JSON 的最大误差为 0（head）和约
2.5e-8（腕部 float32 量化），内容一致。

另用 CPU 确定性假模型走完与真实推理相同的后处理/导出路径：`views.npz` 的三路
`camera_pose` 与源 JSON 最大差均为 0，模型预测 pose 被单独保存；标定 baseline 保持为
0.5761 / 0.6298 / 1.0293 m。随后 Step C 成功逐点复核 456,876 个点并导出 frustum，
Step D 的 numpy/Open3D 体素占用 IoU 为 1.0000。该测试覆盖坐标注册和所有下游代码，
但不替代真实神经网络的深度/重建质量验证。

## 仍未关闭的风险

- `pose_validation_report.json` 仍明确说明 wrist parent=`hand-base` 和腕相机 optical-frame
  语义是待物理验证的工作假设。本次只能证明 pipeline 无损、按声明正确处理，不能替代
  标定板/重建视觉验证。
- 当前主机 `torch.cuda.is_available()` 为 false，`nvidia-smi` 无法连接驱动，因此没有
  对新 capture 运行完整 MapAnything 网络。需要在 GPU 恢复后做一次完整推理，并查看
  GLB/frustum/重投影结果。

## 2026-07-15 完整推理结果的后续诊断

用户随后生成了完整输出。检查 `summary.json` / `views.npz` 证明三路相机没有被合并：

```text
head center       = [ 0.000000,  0.000000,  0.000000] m
hand_left center  = [-0.492144,  0.058303, -0.293802] m
hand_right center = [ 0.530125, -0.053896, -0.335664] m
baselines         = 0.576129 / 0.629768 / 1.029260 m
```

导出 pose 与源 JSON 差为 0。MapAnything 自己预测的 pose 也没有重合，且相对输入仅偏
约 0.6 / 15.5 / 24.5 mm 和 0 / 1.49 / 1.18 deg。

真正可疑的是视轴：head 的 forward 为 `[0,0,1]`，两腕 forward 的 head-frame Z 分量
分别为 `-0.821` 和 `-0.798`。结果中 head 点云主要位于 `+Z`，腕部点云主要位于 `-Z`。
对 hand-base/link7 × parent_T_camera/camera_T_parent 四种候选重新解算后，腕部 forward
仍全部为负 Z；因此只更换 Link7/hand-base 或矩阵方向不能解释异常。最需要确认的是腕部
JSON 的 camera frame 是否真为 RGB OpenCV optical，或是否缺少 raw/depth/link/OpenGL
到 RGB OpenCV 的固定轴变换。

同时修复了 `filter_export.py` 的一个独立问题：当 inference 使用 `--max_radius` 时，
`scene.ply` 已过滤但 `views.npz` 保留全量数据，旧代码会错误断言点数相同。现在会根据
`summary.json` 重放原过滤器。新增 `--color_by_view --frustum_depth`，已生成：

```text
/home/ck/MapAnythingTest/outputs/g1_capture_20260715_121059/scene_filtered_by_view.glb
```

其中 head/left/right 分别为红/绿/蓝，包含三个使用精确输入 pose 的独立 frustum。

## 推荐运行命令

```bash
cd /home/ck/MapAnythingTest/MapAnythingPipeline
export G2_DATA_ROOT=/home/ck/MapAnythingTest/TestData
export G2_OUT_ROOT=/home/ck/MapAnythingTest/outputs
conda run -n MAP python undistort.py --captures g1_capture_20260715_121059
conda run -n MAP python run_inference.py --captures g1_capture_20260715_121059 --validate-only

# GPU 正常后：
conda run -n MAP python run_inference.py --captures g1_capture_20260715_121059 --max_radius 2.0
conda run -n MAP python filter_export.py --captures g1_capture_20260715_121059 --show_cameras --max_radius 2.0
conda run -n MAP python voxelize.py --captures g1_capture_20260715_121059 --voxel_size 0.02 --max_radius 2.0
```

## 2026-07-15 新 capture 的 CPU/GPU 反投影容差修正

对 `g1_capture_20260715_135717` 完成真实推理后，Step C 原固定断言
`max_abs_error < 0.01 m` 在 hand_left 得到 `0.0114818 m` 而失败。复算证明误差来自 GPU
BF16/TF32 与 CPU float32 重放路径，且随点距离增长：该视图 99.9% 坐标误差小于
9.30 mm，最大误差位于约 21.7 m 的远点；本次 `--max_radius 2.0` 实际保留点的最大误差
仅 0.71 mm。

`filter_export.py` 已改为逐点尺度容差：

```text
L_inf_error <= 0.002 m + 5e-4 * ||world_point||
```

所以 2 m 工作距离允许 3 mm、20 m 允许 12 mm；它仍会拒绝工作距离内 1 cm 的 K/pose/
depth 错配。输出同时报告 max、p99.9 和最大 tolerance ratio。新增三个回归测试覆盖远点
GPU 数值差、近距离厘米级错误和空 mask。

修复后真实命令成功：三路 tolerance ratio 为 0.665/0.888/0.675；136,965 个过滤后点
与原 `scene.ply` 逐坐标完全一致（最大差 0），并生成：

```text
/home/ck/MapAnythingTest/outputs/g1_capture_20260715_135717/scene_filtered_by_view.glb
/home/ck/MapAnythingTest/outputs/g1_capture_20260715_135717/scene_filtered_by_view.ply
```

## 2026-07-17 MapAnything pose/depth hybrid 修正

stage bundle 四组真实数据中，`170323/170536` 人工判定对齐良好，而 `170603/170700` 在 arm、
waist 和保存腕 pose 几乎不变时严重分离。源码确认旧 metric 导出把 MapAnything 的网络
`depth/K` 强制套到 calibrated input pose；坏组网络腕 pose 与输入相差约 `49–51 mm`。

对同一 `views.npz` 离线保持网络三路相对 pose、只用 calibrated head 做全局刚性锚定后，坏组
left-right symmetric p10 从 `10.56/8.33 mm` 改善到 `4.92/5.76 mm`，用户交互 GLB 确认恢复。

Pipeline 当时新增以下默认模式（现已被 2026-07-20 的 baseline-scaled 默认模式取代）：

```text
--pose-export-mode model-relative-head-anchored   # 当时默认，现仅诊断
--pose-export-mode calibrated-input               # 旧 hybrid，仅诊断
--ignore-poses                                    # RGB-only，任意尺度
```

默认模式只施加一个共同左乘：

```text
world_T_model_reference = calibrated_world_T_head @ inverse(model_reference_T_head)
world_T_camera          = world_T_model_reference @ model_reference_T_camera
```

`views.npz` 的 `<view>_camera_pose/pts3d` 使用最终自洽几何，同时另存
`<view>_calibrated_input_camera_pose` 与 `<view>_model_camera_pose_head_reference`。
`camera_poses_used_for_export.json` 和 `summary.json` 明确记录 effective mode、anchor、最终 pose 与
每路相对 calibrated pose 的差异。GUI 默认选择推荐模式并保留 legacy 下拉选项。

## 2026-07-20 metric scale 的 baseline similarity 修正

用户对蓝盒和圆柱实测确认新 model-relative/head-anchor 重建仍统一偏大约 `11%`。独立相机
baseline 给出相同结论：`170603/170700` 的模型三条 baseline 比 calibrated baseline 大
约 `11–13%`，三对最小二乘校正分别为 `0.8980/0.8938`。作为对照，原好组 `170323/170536`
只需 `0.9857/0.9832`，所以不能使用固定全局 `0.895`。

默认模式升级为逐 capture 求解：

```text
--pose-export-mode model-relative-head-anchored-baseline-scaled  # 默认/推荐
--pose-export-mode model-relative-head-anchored                  # 旧未缩放诊断
--pose-export-mode calibrated-input                              # 旧 hybrid 诊断
```

求解 `s = argmin Σ(s*b_model-b_calibrated)^2`，然后在 model head frame 中同时执行
`relative camera translation *= s` 与 `depth_z *= s`，再锚定 calibrated head。这是对完整重建的
一个统一 similarity，不改变三路旋转/方向，也不会重新引入 depth/pose hybrid。

`summary.json` 和 `camera_poses_used_for_export.json` 保存三对 baseline、LS scale、校正前后 RMSE
及 ratio spread；`views.npz` 保存 `pose_export_similarity_scale` 和模型原始
`metric_scaling_factor`，其 `depth_z/camera_pose/pts3d` 均属于校正后的同一几何。

真实 GPU 已对 `170603/170700` 完成 inference→filter_export→voxelize：分别自动得到
`s=0.897974/0.893756`，baseline RMSE 从 `53.31/55.80 mm` 降至 `2.50/2.99 mm`，2.3 m
过滤后保留 `352688/348984` 点。六路反投影最大偏差 `0.89–1.60 mm`，PLY 逐点一致，两组
numpy/Open3D voxel IoU 均为 `1.0000`。18 项 Pipeline 测试通过。验证输出和报告位于：

```text
/home/ck/MapAnythingTest/outputs_pose_scale_fix_20260720
/home/ck/MapAnythingTest/outputs_pose_scale_fix_20260720/REAL_SCALE_VALIDATION.md
```

新采集 `g1_capture_20260720_102356` 随后完成相同 A→D 流程：自动 `s=0.851619`，baseline
RMSE `93.45 -> 2.25 mm`，三对 ratio spread `1.218%`；Step C 三路最大误差
`1.36/2.00/1.27 mm`，PLY 完全一致，voxel IoU `1.0000`。输出：

```text
/home/ck/MapAnythingTest/outputs_pose_scale_test_20260720_102356
/home/ck/MapAnythingTest/outputs_pose_scale_test_20260720_102356/VALIDATION.md
```

用户确认不同数据的实体尺度已经准确很多，但仍有少量未对齐。当前默认模式只从 calibrated pose
取得 baseline 长度和 head world anchor；最终相对 rotation/translation direction 来自模型。
所以剩余偏差不能直接定性为标定错误。下一步用三路共同刚性靶跨多组求 per-camera residual
SE(3)：跨 capture 固定才支持外参/optical frame，随场景变化支持模型 pose/depth，随关节变化
才回查 FK；同时做 raw K/D 与 undistorted/new K 的同名角点 A/B。
