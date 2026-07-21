# G2 机器人相机 MapAnything 三维重建 — 工作日志与迁移指南

> 最后更新：2026-07-21。用于未来 resume 工作或迁移到其他机器时快速上手。

## 2026-07-21：Inference GUI 计时与保守加速

- Pipeline GUI 增加活动阶段和整条 pipeline 的实时计时；每个完成阶段在日志中记录秒数。
- `run_inference.py` 增加模型加载、输入读取、preprocess、GPU inference、重建后处理、
  GLB/PLY、NPZ 和单 capture 总耗时，并将 capture 分项写入 `summary.json`。
- GUI 默认复用内容哈希完全一致且输出完整的 undistort 结果；旧 manifest 首次重算。
- 新增显式 fast dense-head 模式，用更多峰值显存换速度；CUDA OOM 自动退回
  memory-efficient + minibatch 1。没有移除 pose/K 校验或 edge mask。
- 当前执行环境无法连接 NVIDIA driver，因此只确认了本地源码中的速度/显存语义，没有声称
  未测得的 GPU 加速比例；正式比例以后由新增计时在同一 capture 上 A/B 得到。

## 2026-07-20：语义体素方案审阅

- 审阅并重写 `PLAN_SEMANTIC_VOXEL.md`，将目标从“单次 YOLO 标签写入占据格”明确为
  面向机器人操作的稀疏语义体素地图。
- 当前 `voxelize.py` 只实现 P1 表面占据体素；`labels/label_scores` 仍是占位，语义提升、
  稳定实例 ID、多帧融合和操作接口均未实现。
- 确定语义图需要固定 `frame/origin/voxel_size/dims`，区分 unknown/free/occupied，并同时
  保存类别票数、实例 ID、观测来源、次数与时间戳；GLB 只作调试，NPZ/JSON 是数据接口。
- 头部 YOLO 实例 mask 直接索引 `head_pts3d` 写入可见表面；三相机只先补几何，未直接
  观察部分必须受颜色/法向/连通性约束传播，不能沿检测射线或最近邻无条件扩散。
- 结合当前 ChArUco 结果，采用 2–3 cm 全局语义体素；5–10 mm 只用于动作前实时局部精修，
  不把 voxel size 误当作绝对定位精度。
- 最小闭环定为：单 capture YOLO-Seg → 固定 2 cm 网格 → class/instance 语义 GLB →
  `objects.json`/box OBB → 改变头部姿态复测一致性。

## 2026-07-20：G1 自洽 pose、逐采集尺度与剩余误差边界

- 旧 metric 导出的 `model depth/K + forced calibrated pose` hybrid 会令部分 capture 分离；默认
  已改为保留 MapAnything 三路相对 pose，再以 calibrated head 锚定到 `base_link`。
- 用户实体测量确认部分重建约大 11%；固定 scale 不可用，因为 calibrated/model baseline 给出
  `170323/170536≈0.9857/0.9832`、`170603/170700≈0.8980/0.8938`。
- 当前默认 `model-relative-head-anchored-baseline-scaled` 对每个 capture 用三条 camera baseline
  最小二乘求 `s`，同时执行 `depth_z *= s` 和 head-relative translation `*= s`。
- 真实 GPU `170603/170700` 得到 `0.897974/0.893756`，baseline RMSE 降到
  `2.50/2.99 mm`；新 `102356` 得到 `0.851619`，RMSE `93.45 -> 2.25 mm`。三组 Step C、PLY、
  voxel 全通过，voxel IoU `1.0000`；18 项测试通过。
- 用户确认不同数据的尺度明显准确，但仍有少量未对齐。scale 只约束相机中心距离，最终相对
  rotation/translation direction 仍来自模型，故不能直接归咎机器人标定。下一步用跨多组共同
  刚性靶求 per-camera residual SE(3)：固定 residual 才支持外参，随场景变化支持模型，随关节
  变化才支持 FK；另做 raw K/D 与 undistorted/new K A/B。
- 最新输出：`outputs_pose_scale_fix_20260720` 和
  `outputs_pose_scale_test_20260720_102356`；各目录保存完整 provenance/validation。

## 2026-07-15：G1 GUI capture 输入审查与修正

- 新数据：`TestData/g1_capture_20260715_121059`。
- 真实数据已通过图像、内参、畸变、RDF cam2world、米制、SO(3)、head identity、
  baseline 和 MapAnything preprocessing 校验。
- 移除了四个脚本对旧 `g_1_Test_1..4` 的默认硬编码，改为自动发现或显式
  `--captures`。
- Step A/B 默认要求 pose，只有显式 `--allow-missing-poses` 才允许任意尺度模式。
- 修正旧注释把所有 world frame 误写为 robot `end` 的问题；world 现在来自 pose JSON。
- 关键语义修正：MapAnything 只把输入 pose 当条件先验并重新预测 pose。最终反投影、
  `views.npz` 和相机 frustum 现在使用原始标定 pose；模型预测 pose 单独保存做诊断。
- 新增 `capture_contract.py`、`--validate-only`、preprocess manifest、provenance 复制和
  6 个回归测试。
- 真实 capture 的 Step A + Step B validate-only 冒烟通过：输出尺寸 head 1279×719、
  wrists 847×479；模型输入统一为 518×294；所有 pose preprocess diff 为 0。
- CPU 确定性假模型完整走通 inference 后处理、GLB/PLY/NPZ、filter/frustum 和 voxelize；
  导出 pose 与源 JSON 差为 0，numpy/Open3D voxel IoU 为 1.0。
- 当前机器 NVIDIA 驱动不可用，尚未执行本次新数据的完整模型推理/视觉验收。
  详细审查见 `G1_CAPTURE_PIPELINE_AUDIT_20260715.md`。
- 后续完整输出证明三路相机中心没有合并；异常是 head 点云在 +Z、两腕点云在 -Z。
  四种 wrist parent/direction 候选都不能消除负 Z，下一优先级是确认 wrist JSON 的
  raw camera frame 到 RGB OpenCV optical 的固定轴转换。新增按视图着色/frustum 诊断
  GLB，并修复 inference-time filter 导致 `filter_export.py` 点数误断言的问题。

## 1. 项目概述

用 [MapAnything](https://github.com/facebookresearch/map-anything)（facebook/map-anything，米制多视图 3D 重建模型）对 G2 机器人的三路 RGB 相机（head 640×400、hand_left / hand_right 1280×1056 广角）做**联合**三维重建，输出米制彩色点云。

- 输入数据仓库：`~/MapAnything/MapAnythingTestData`（GitHub: `chensu2584/MapAnythingTestData`）
- 模型代码仓库：`~/MapAnything/map-anything`（已生成 CLAUDE.md 供 Claude Code 使用）
- 本管线脚本：`~/MapAnything/g2_pipeline/`
- 输出：`~/MapAnything/outputs/`

## 2. 已完成的工作（2026-07-09）

### 环境搭建
- conda env `mapanything`（Python 3.12），`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` + `pip install -e ~/MapAnything/map-anything`
- **坑**：pip 默认装 torch cu130 wheel，但本机驱动 570.x 只支持 CUDA 12.8，导致 `cuda available: False`，必须用 cu128 index-url 重装（见 §5 迁移注意事项）

### 管线（三个脚本，均在本目录）
1. **`undistort.py`** — 去畸变。OpenCV Brown–Conrady（JSON 中 k1,k2,p1,p2,k3 与 OpenCV 参数顺序一致），`getOptimalNewCameraMatrix(alpha=0)` → `initUndistortRectifyMap` → `remap` → ROI 裁剪并平移主点。输出去畸变 PNG + `<name>_K.json`（修正后内参）到 `outputs/undistorted/<capture>/`
2. **`run_inference.py`** — 联合推理。三路图 + 修正内参组成 views → `preprocess_inputs()`（自动统一分辨率到 518 集合、同步缩放内参）→ 一次 `model.infer()` 联合求解（`memory_efficient_inference=True, bf16 AMP`）。输出 `scene.glb/.ply`、`views.npz`、`summary.json`
3. **`filter_export.py`** — 从 `views.npz` 重新导出（无需 GPU/重推理）。可叠加过滤器 `--max_radius` / `--bbox` / `--min_conf`；`--show_cameras` 在 GLB 中加相机锥台标记（头红、左手绿、右手蓝）。输出 `scene_filtered.glb/.ply`

### 结果（全部 4 个 capture，`--max_radius 2.0`）

| Capture | 过滤后点数（保留率） | 过滤后包围盒 (m) | 基线 头–左/头–右/左–右 (m) |
|---|---|---|---|
| 142817 | 356,338 (65.7%) | 3.6×2.3×1.5 | 0.42 / 0.62 / 0.95 |
| 144239 | 416,984 (79.2%) | 3.2×2.6×1.5 | 0.59 / 0.54 / 0.31 |
| 144354 | 427,025 (81.7%) | 3.1×2.6×1.6 | 0.56 / 0.51 / 0.33 |
| 144728 | 435,300 (80.6%) | 3.0×2.7×1.7 | 0.67 / 0.45 / 0.45 |

### GitHub 交付
- `reconstruction_outputs/` 已推送到 MapAnythingTestData 仓库（commit `5b68be8`）
- ⚠️ **未完成**：加相机标记后的 GLB 更新（本地 commit `8a71ea0`）推送失败 —— 用户的 fine-grained token 已撤销，**需要新 token 重新 push**

## 3. 关键发现与思考

1. **去畸变是必须的且已定量验证**。MapAnything 全程假设无畸变针孔模型（`geometry.py:154`、`wai/camera.py:208` 明确 "Undistort first"）。手腕广角相机（105° HFOV）角落像素偏移约 140–150 px；去畸变后地面黄线弯曲度从 12.3 px 降至 1.1 px。去畸变后图像边缘的"拉伸感"是**透视拉伸**（正确的针孔投影），不是残余畸变，不要试图去除。
2. **推理是三路联合的**（单次 `model.infer` 多视图 alternating attention），输出共享世界坐标系（head 相机 = 原点）和全局一致的米制尺度，相机相对位姿是模型的输出而非输入。
3. **尺度是学出来的先验，无物理锚定**。纯图像+内参输入下，绝对尺度预计有 5–15% 误差。验证手段：用机器人运动学（URDF/FK）的真实头–腕距离对比模型估计的基线；系统性偏差可全局乘系数校正。
4. **置信度异常是共视不足的信号**。capture 1 头部 conf 1.27、手腕贴地板 1.0；后 3 个 capture 全部视图贴地板（双手举在身前，共视更少）。这类 capture 的跨视图对齐精度要打折扣，`--min_conf` 过滤在这种情况下无效（所有点同值）。
5. **过滤应在导出层做，不动输入**。远处地面/墙参与推理有利于位姿求解，只是不该出现在最终点云；`views.npz` 存了 `pts3d`+颜色，改过滤参数秒级重导出。
6. **下一步（用户已确认计划）**：提供相机外参 + 头部深度图后走"图像+内参+深度+位姿"满配模式 —— 外参解决跨视图对齐和尺度锚定，深度锚定头部几何并通过联合推理传播到手腕视图。建议用 `ignore_pose_inputs` / `ignore_depth_inputs` 做三种模式的消融对比。

## 4. 数据格式约定（重要）

### 输入
- capture 文件夹：`g2_smoke_<timestamp>/`，含 `head.png`、`hand_left.png`、`hand_right.png`（`.npy` 是相同的 RGB 数组，非深度，忽略）
- 内参 JSON（顶层，按文件名对应相机）：`intrinsic_{head_front,hand_left,hand_right}_rgb.json`，字段 `Fx, Fy, Cx, Cy, k1, k2, p1, p2, k3, SN`，Brown–Conrady 模型，与原始（未去畸变）分辨率对应
- **给模型的内参必须用去畸变后的 `<name>_K.json` 中的 newK，不是原始 K**

### 未来的外参与深度输入
- 外参：**OpenCV cam2world 约定**（+X右 +Y下 +Z前），必须是 RGB **光心**位姿（手眼标定结果，不是 URDF link 系，需转换）；去畸变不改变外参，可直接用；view 0（head）必须带位姿
- 深度：**z-depth**（沿光轴），单位米，配准到去畸变后的 RGB 逐像素对齐，无效像素填 0；每个带物理输入的 view 设 `is_metric_scale=True`

### 输出 `views.npz`（每视图 6 个数组，前缀 `head_` / `hand_left_` / `hand_right_`）
`_pts3d`（H×W×3 世界坐标，米）、`_depth_z`、`_camera_pose`（4×4 cam2world）、`_intrinsics`、`_conf`、`_mask`、`_img`（uint8 RGB）

### GLB 坐标注意
`predictions_to_glb` 对整个场景做了 **180° 绕 X 轴翻转**（GLB viewer 朝向习惯）。给 GLB 场景后添加任何几何体（如相机标记）必须应用同样的翻转；`scene.ply` / `views.npz` 是未翻转的原始世界坐标。

## 5. 迁移到其他机器的注意事项

### 环境
1. Python ≥3.10（本项目用 3.12），先装 torch 再 `pip install -e <map-anything>`
2. **torch wheel 必须匹配目标机的驱动**：先 `nvidia-smi` 看右上角 CUDA Version，再选 `--index-url https://download.pytorch.org/whl/cuXXX`。pip 默认最新 cu 版本常常超过驱动支持（本机就中招：cu130 wheel vs 12.8 驱动 → CUDA 不可用）
3. 模型权重缓存在 `~/.cache/huggingface/hub`（`models--facebook--map-anything` + DINOv2 骨干，约 6 GB）。离线机器可直接拷贝这个目录避免下载；在线机器首跑自动下载
4. GPU 显存：3 视图推理配合 `memory_efficient_inference=True` 实测峰值很小（在只剩 ~24 GB 的共享 H200 上无压力）；更小的卡可加 `minibatch_size=1`

### 脚本
- 三个脚本的路径常量是硬编码的（`TEST_DATA`、`OUT_ROOT`、`UNDIST_ROOT`，均 `os.path.expanduser("~/MapAnything/...")`），迁移后要么保持相同目录布局，要么改这几个常量（都在文件顶部）
- capture 列表 `CAPTURES` 也是硬编码的，新数据需更新（或用 `--captures` 参数覆盖）
- 解释器：脚本无 shebang 依赖，直接用目标机 env 的 python 运行即可
- 复跑顺序：`undistort.py` → `run_inference.py`（需 GPU）→ `filter_export.py`（纯 CPU，可反复调参数）

### 其他
- `map-anything` 仓库本体只有代码（18 MB），git clone 即可；本管线只依赖它的 pip 包安装，未修改仓库源码
- GitHub 推送：MapAnythingTestData 用 fine-grained PAT（需勾选该仓库 + Contents Read/write）；token 不要留在 shell 历史/对话里，用完撤销
- 数据处理子包 `data_processing/wai_processing` 与主环境冲突（hydra 版本），**不要**装进同一个 env（本项目未用到）

## 6. 已完成的工作（2026-07-14）— G1 数据 + 外参条件推理

> 本节工作在分支 **`g1-with-extrinsics`**（commit `8edbefe`），即 §3.6 计划中"图像+内参+位姿"模式的落地。

### 新数据集:MapAnythingTestData1(G1 机器人)
- 仓库 `chensu2584/MapAnythingTestData1`,本地 `~/MapAnything/MapAnythingTestData1`,4 个 capture:`g_1_Test_1..4`
- 与旧 G2 数据的差异:**内参/外参按 capture 文件夹内置**(不再是仓库顶层共享);相机序列号与标定值和 G2 不同(head SN `CPBC853000EL`),**绝不可复用旧内参**
- 每个 capture 含:3 张 RGB、`intrinsic_*.json` + pipeline 别名 `intrinsic_*_rgb.json`、四元数外参 `extrinsic_*.json`、**`camera_poses_opencv_cam2world.json`**(4×4 cam2world,OpenCV 约定,世界系=机器人 "end",光心位姿,单位米)、`manifest.json`、`mapanything_views_input.npz`(原始畸变图+原始 K+位姿,不能直接喂,仍需先去畸变)

### 管线改动(4 个脚本)
1. **路径改为环境变量**:`G2_DATA_ROOT`(capture 父目录)+ `G2_OUT_ROOT`(输出根),默认值为本机路径,跨机器零改码(§5 迁移流程相应简化)
2. **`undistort.py`**:内参改为从 capture 文件夹内读取;外参 JSON 原样拷贝到去畸变输出目录(去畸变只改 K 不改位姿)
3. **`run_inference.py`**:读取外参并以 `camera_poses`(4×4 tensor)喂给 `model.infer`;无外参文件时自动回退纯图像+内参模式。**故意不显式传 `is_metric_scale`**——不传时框架默认 True 且生成正确形状的 bool tensor;显式传 Python bool 会在 model.py:924 的 tensor 索引处崩

### 实跑结果(4 captures 全部成功,H200)
- 每 capture 52–55 万点,有效像素 72–92%;输出基线与外参真值吻合:hand_left↔hand_right 0.191–0.203 m vs 真值 ≈0.199 m → **米制尺度锚定生效**(对比 §3.3 纯图像模式的 5–15% 尺度误差)
- 输出已推送 `MapAnythingTestData1` main:`724669c`(reconstruction_outputs/)+ `fb22e26`(measure_viewer.html)

### 新工具:measure_viewer.html(两点测距,验证尺度精度)
- 单个自包含 HTML(3.5 MB,内嵌 g_1_Test_1 六毫米降采样点云),浏览器双击打开、离线可用;单击两点出米制距离,支持拖入任意 `scene.ply` 看全分辨率
- 严肃精度评估仍推荐 CloudCompare(Tools → Point picking)

### 重要结论/注意事项
1. **输出世界系 = head 相机系,不是 "end" 系**:模型总把 view 0 归一化为原点(即使喂了世界系位姿)。距离/尺度不受影响,但 `--max_radius`/`--bbox` 在 head 系下生效;要 "end" 系坐标需用 head 外参再变换一次
2. **外参是条件不是硬约束**:输出位姿仍是模型回归结果,基线相对真值有 ±4% 以内浮动,该浮动属于模型能力范畴
3. **数据操作链路已审计**:去畸变数学、内参配对、位姿透传、通道顺序均无系统性误差;质量上限由模型 + 518 推理分辨率决定
4. §2 遗留的 token 问题已解决:MapAnythingTestData1 用新 PAT 推送成功(旧 MapAnythingTestData 的 `8a71ea0` 是否补推,视需要)
