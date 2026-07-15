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
