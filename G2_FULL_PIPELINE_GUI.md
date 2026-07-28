# G2 采集到三版本重建 GUI

## 当前状态

`g2_full_pipeline_gui.py` 已把以下步骤放进一个可停止、可查看日志的 Tk GUI：

1. 用指定的 G2 四相机采集程序获取三路 RGB、头部深度、实时关节和相机外参；
2. 按当前生产参数执行去畸变、MapAnything 推理、过滤和 1 cm 体素化；
3. 从注册后的度量深度独立重建体素；
4. 对深度和 MapAnything 两路执行同一桌面范围裁剪、去夹爪、DBSCAN 去噪和桌面下方裁剪；
5. 导出 only 深度、融合、only MapAnything 三份最终 NPZ/GLB。

启动：

```bash
cd /home/ck/MapAnythingTest/MapAnythingPipeline
conda run --no-capture-output -n MAP python g2_full_pipeline_gui.py
```

可用 `G2_FULL_PIPELINE_GUI_SCALE=0.8` 或 `1.2` 调整界面缩放。窗口初始尺寸会限制在当前屏幕
范围内，左右区域和日志区可随窗口缩放。

## 采集契约

GUI 不重写外参算法，而是直接运行：

```text
/home/ck/MapAnythingTest/g2_four_camera_extrinsic_capture.py
```

界面允许选择别的文件，但启动前会检查它是否仍包含指定脚本的关键契约。当前调用固定为：

```text
--pose-source fk
--sensor-dir <G2>/G2_parameters/sensor
--urdf <G2>/G2_parameters/G2_t2_crs_omnipicker/urdf/G2_t2_crs_omnipicker.urdf
--save-dir <run>/in
```

最终运行的 G2 机器具备完整采集 GDK，因此“采集 Python”和“处理 Python”默认使用当前
解释器即可，不要求额外部署步骤。界面仍允许两者独立选择，便于以后把 GDK 与
MapAnything/CUDA 拆到不同环境；两条解释器路径都会写入运行配置。

因此图像和外参的实际行为与指定脚本一致：

- 先取头部深度，再按其时间戳取最近的头部 RGB；
- 左右手 RGB 和缺失帧处理沿用原脚本；
- 实时读取 GDK 关节状态；
- 用 G2 URDF FK 求 `base_T_head_link3`、`base_T_arm_l/r_end_link`；
- 与 `G2_parameters/sensor` 静态传感器标定相乘得到六个 `base_T_camera`；
- 保存 `camera_extrinsics.json`、原始 16 位头部深度、三路 RGB 和 FK/SDK TF 校验摘要。

采集会打开原脚本的 OpenCV 四画面窗口。单击画面或按 `S`/空格保存，按 `Q`/Esc 结束；Tk
主窗口在采集期间继续刷新 Snapshot 列表。

## 生产参数

GUI 固定当前最优生产组合，只有设备、预处理复用和人工桌面 XY 范围开放编辑：

| 项 | 值 |
|---|---|
| Feed metric depth | 开 |
| Depth holdout | `0.0` |
| Pose export | `model-relative-head-anchored-baseline-scaled` |
| View order | `head, hand_left, hand_right` |
| Hand max depth | 左右均 `1.0 m` |
| Roll normalize | 关 |
| Self-mask input | 关 |
| 首跑复用预处理 | 关 |
| Memory-efficient inference | 开 |
| Max radius | `2.3 m` |
| Voxel size | `0.01 m` |
| Fusion | occupancy |
| Snap distance | `0.03 m` |
| Surface tolerance | `0.04 m` |
| DBSCAN | `eps=0.03 m`, `min_cluster=24` |
| Table thickness | `0.06 m` |

每次运行把配置、采集参考脚本 SHA-256、Capture 列表和最终路径写入：

```text
<run>/g2_full_pipeline_config.json
```

全部处理完成后还会逐个复读三份 NPZ，验证 `world_frame=base_link`、非空、所有体素中心均在
所选桌面 XY 内，并核对 `fusion_report.json` 的边界参数。只有全部通过才写：

```text
<run>/three_version_validation.json
```

GUI 结果页的“已裁剪”状态来自该验证文件，不只根据 GLB 是否存在判断。

## 桌面裁剪

三份最终结果强制使用 GUI 中同一组人工标定 `base_link` XY 范围，默认：

```text
X = [0.239, 1.019] m
Y = [-0.694, 0.706] m
```

它们都来自 Avoid `clean_cloud()` 的同一次调用链：

```text
人工桌面 XY 裁剪
  -> 基于左右腕相机位姿的实测代理盒去夹爪
  -> DBSCAN 去噪
  -> 支撑面高度检测并删除 table_top_z - 0.06 m 以下内容
```

`versions/` 下的三份才是正式的统一裁剪输出。`map/`、`depth/` 中仍会保留未走完最终清理的
中间证据，不能因为文件名也是 GLB 就直接交给绕障。

## 输出目录

```text
<run>/
  in/<snapshot>/                         原始采集
  map/undistorted/<snapshot>/            去畸变、注册深度
  map/<snapshot>/                        MapAnything 中间结果
  depth/<snapshot>/                      直接深度中间结果
  versions/<snapshot>/
    depth_only_voxels.npz
    depth_only_voxels.glb
    fused_voxels.npz
    fused_voxels.glb
    mapanything_only_voxels.npz
    mapanything_only_voxels.glb
    fusion_report.json
  three_version_validation.json
```

最终 GLB 使用采集颜色，GUI 把融合 provenance tint 设为 `0.0`，并隐藏红色夹爪删除示意盒。
三份 GLB 都含 `base_link` 原点、头部/左右手相机、左右法兰参考中心和简约左右手。

简约手和法兰中心只用于视觉检查。参数包 URDF 的 omnipicker 不是实机夹爪，当前去夹爪仍是
以腕相机位姿锚定的操作者实测代理盒，不是已标定 TCP 或完整实机碰撞模型。

## 验证记录

2026-07-28：

- Pipeline：`108` 项通过；其中 `2` 项显示相关布局测试在桌面环境补跑；
- Avoid 全套 pytest：`83 passed, 3 skipped`；
- Tk GUI 在本机显示环境完成建窗、控件构建和立即关闭冒烟；
- 用 `snapshot_20260724_040712_0001` 真实后处理得到：
  - only 深度：`9,709` 体素；
  - 融合：`17,592` 体素；
  - only MapAnything：`18,270` 体素；
- 三份输出的人工桌面 XY 越界体素均为 `0`；
- 三份 GLB 均可由 Trimesh 复读，并含三处相机、简约左右手、法兰和原点标记。

本次开发机没有重新跑完整 CUDA MapAnything 推理，也没有连接真实 G2 相机；目标部署机已具备
完整 G2 采集 GDK。三份重建仍是粗粒度规划输入，不解除 Avoid 对未知实机夹爪、
TCP、轨迹时间参数化和真实执行的安全闭锁。
