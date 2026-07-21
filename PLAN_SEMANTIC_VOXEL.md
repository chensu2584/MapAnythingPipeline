# 计划：面向机器人操作的语义体素地图

> 初版：2026-07-11
>
> 审阅更新：2026-07-20
>
> 状态：几何体素 P1 已实现；语义提升、实例融合和操作接口尚未实现。
>
> 本文是当前方案的唯一决策基线。`TECH_DETAIL_TASK2.md` 保留早期单帧实现细节，若与本文冲突，以本文为准。

## 1. 目标与边界

目标不是只给 GLB/点云染色，而是建立可供规则系统查询的**稀疏语义体素地图**：

```text
三路 MapAnything 几何
        +
头部 YOLO 类别/实例掩码
        ↓
固定坐标系下的 Semantic Voxel Map
        ↓
对象实例 / 空间关系 / 碰撞几何 / 操作候选
        ↓
抓取、移动物体、拉抽屉等规则操作
```

地图必须同时回答：

1. 该空间是 `unknown / free / occupied` 中的哪一种；
2. 被占据体素属于什么类别；
3. 属于哪个对象实例，而不只是“都是 box”；
4. 结论来自哪些相机/帧、置信度多少、最后何时看到；
5. 如何从体素恢复可操作对象的中心、尺寸、朝向、表面和部件。

当前不把语义地图视为安全认证传感器，也不要求一次 MapAnything snapshot 直接提供毫米级抓取位姿。全局地图用于发现目标、规则推理和粗定位；执行抓取/拉抽屉前，用当前相机或深度传感器进行局部在线校正。

## 2. 当前事实与审阅结论

### 2.1 已完成

- `voxelize.py` 已能把 `views.npz` 的三路世界坐标点云体素化；默认 2 cm。
- 当前输出 `voxels.npz` 保存稀疏体素索引、颜色、点数和几何置信度，`labels/label_scores` 只是全零占位。
- `voxels.glb` 已用于立方体可视化，并通过 numpy/Open3D 占据索引一致性检查。
- `views.npz` 保存每个视图逐像素对应的 `<view>_pts3d`，所以头部 2D mask 可以直接索引得到同一像素的 3D 世界点，不必再次手写 `K^-1` 反投影。

### 2.2 尚未完成

- 尚未接入实际 YOLO 权重、类别表和实例 mask。
- 尚无 `detect_objects.py` / `semantic_lift.py`。
- 尚无多帧语义累积、稳定实例 ID、动态物体更新和对象级 `objects.json`。
- 当前只是**表面占据体素**，没有沿相机射线雕刻 free space；因此未观测区域必须视为 `unknown`，不能写成背景或自由空间。

### 2.3 本次审阅后确定的七项原则

1. **几何与语义分文件保存。** 不覆盖原始 `voxels.npz`，生成可重算的 `semantic_voxels.npz`。
2. **固定网格。** 离线预览可沿用点云 tight bbox；跨 capture/在线操作必须显式给定 `frame + origin + voxel_size + dims`，推荐最终使用 `base_link`。
3. **类别和实例并存。** 每格既有 `semantic_class`，也有 `instance_id`；对象级规则依赖后者。
4. **累积概率，不做最后一次覆盖。** 多帧观测更新类别/实例票数，保存观测数、来源和时间戳。
5. **只直接标记可见表面。** YOLO mask 不得把射线后方的体素整列标成目标；手相机未识别部分只能做受约束的几何传播。
6. **GLB 只是调试产物。** 完整语义、实例和时间信息以 NPZ + JSON 为准。
7. **全局粗、操作局部细。** 当前 ChArUco 检测显示跨相机典型误差仍是厘米级；2 cm 全局体素合理，5 mm 全局网格只会表达虚假精度。

## 3. 表示设计

### 3.1 网格元数据

`semantic_voxels_metadata.json`：

```json
{
  "schema_version": 1,
  "frame": "base_link",
  "voxel_size_m": 0.02,
  "origin_m": [-1.5, -1.5, -0.3],
  "dims": [150, 150, 120],
  "classes": ["unknown", "box", "drawer_front", "drawer_handle"],
  "source_capture": "g1_capture_..."
}
```

若当前 capture 的 pose contract 还不能输出 `base_link`，保留它声明的 `world_frame`，不得把任意模型坐标系默认为机器人基座系。

### 3.2 稀疏体素字段

`semantic_voxels.npz` 的目标 schema：

| 字段 | 类型/形状 | 含义 |
|---|---|---|
| `indices` | `N×3 int32` | 稀疏体素坐标 |
| `occupancy_state` | `N int8` | `-1 unknown, 0 free, 1 occupied`；P2 初期只写 occupied |
| `occupancy_log_odds` | `N float32` | 多帧占据概率；P2 可先由几何置信度初始化 |
| `colors` | `N×3 uint8` | 融合颜色 |
| `geometry_confidence` | `N float32` | MapAnything 几何置信度聚合 |
| `semantic_votes` | `N×C float32` | 各类别累计票数；类别少时直接存稠密矩阵 |
| `semantic_class` | `N int16` | `argmax(semantic_votes)` 的缓存结果 |
| `semantic_score` | `N float32` | 归一化类别置信度 |
| `instance_id` | `N int32` | 稳定对象实例，`-1` 表示未分配 |
| `instance_score` | `N float32` | 该实例归属置信度 |
| `observation_count` | `N uint16` | 有效观测次数 |
| `source_view_mask` | `N uint8` | head/left/right 三位来源掩码 |
| `last_seen_ns` | `N int64` | 在线模式最后观测时间；离线可为 capture 时间 |

对象类别数变大时再把 `semantic_votes` 改为 top-k 稀疏存储；首版不提前复杂化。

### 3.3 对象级派生结果

体素是权威空间表示，但规则系统不应逐格遍历。由语义类别、实例 ID 和 3D 连通域派生 `objects.json`：

```json
{
  "frame": "base_link",
  "objects": [{
    "instance_id": 3,
    "class": "box",
    "confidence": 0.91,
    "centroid_m": [0.72, -0.18, 0.64],
    "aabb": {"min": [], "max": []},
    "obb": {"center": [], "axes": [], "extent": []},
    "voxel_count": 4821,
    "last_seen_ns": 0,
    "affordances": ["graspable", "movable"]
  }]
}
```

`objects.json` 是体素地图的派生缓存，可以随时重建，不能反过来覆盖原始体素证据。

## 4. 语义写入与融合

### 4.1 2D 入口

- 优先使用 YOLO 实例分割：输出 `class + score + instance mask`。
- 若当前模型只有检测框，框不能直接提升到 3D；短期使用 YOLO box 提示 SAM2 得到 mask，纯几何聚类只作为降级方案。
- YOLO 在其训练域的全分辨率头图上运行；结果按照 `undistort.py` 的 remap 和 MapAnything preprocessing 精确变换到 `<view>_pts3d` 的 H×W 像素域。
- 每个阶段输出 overlay，先证明 2D mask 与去畸变/缩放后的图像一致，再写 3D。

### 4.2 直接提升到头部表面体素

```python
valid = instance_mask & head_mask & geometry_confidence_mask
points_world = head_pts3d[valid]
voxel_index = floor((points_world - grid_origin) / voxel_size)
```

对类别 `c` 的一次观测，建议票重：

```text
w = yolo_score × geometry_weight × mask_interior_weight × view_weight
semantic_votes[voxel, c] += w
```

- mask 边缘降低权重或腐蚀 1–2 个全分辨率像素，抑制深度边界渗色。
- 同一像素只写其已重建表面所在体素；没有自由空间 raycast 前，不推断该射线上其他体素。
- 类别冲突保留完整票数和置信度，不立即删除低票类别。

### 4.3 实例关联

单帧 YOLO 的实例序号不是稳定 ID。跨帧按以下顺序关联：

1. 类别相同；
2. 3D voxel IoU / OBB IoU；
3. 质心距离与尺寸一致性；
4. 必要时加入颜色/图像 embedding；
5. 达不到门限则新建实例，短时丢失保留旧实例并降低置信度。

动态对象不能永久烙在静态体素地图中。在线阶段需按 `last_seen` 衰减或清除旧实例占据，并区分静态层与动态对象层。

### 4.4 三相机几何补全

首版只让头部 YOLO 决定语义；三相机均可贡献几何。头部直接标记后：

- 落入同一体素的手相机点自然继承该体素语义；
- 邻域传播必须同时满足空间邻接、颜色/法向相似、没有明显深度断层、属于同一几何连通分量；
- 桌面与盒子、柜体与抽屉面板的接触区域要阻止无条件 region growing；
- 每个传播标签保存来源，能区分“YOLO 直接观测”和“几何推断”。

将来若手相机也运行合适的分割模型，则作为独立观测写入同一投票系统，而不是覆盖头部结果。

## 5. 面向具体操作的派生逻辑

### 5.1 移动盒子

1. 取 `class=box, instance_id=k` 的连通体素；
2. 去除小分量并拟合 AABB/OBB；
3. 从外表面体素估计可见平面、顶面和候选夹持面；
4. 生成粗抓取位姿和碰撞盒；
5. 抓取前以当前头/腕相机建立 5–10 mm 局部地图，重新估计最终抓取位姿。

### 5.2 拉抽屉

YOLO 类别至少拆成 `drawer_front` 与 `drawer_handle`，柜体可另设 `cabinet`：

1. 对 `drawer_front` 体素拟合平面与法向；
2. 对 `drawer_handle` 实例提取中心、主轴和可夹持区域；
3. 由面板法向生成预抓取与拉动方向；
4. 视觉估计只提供初值，实际拉动使用轨迹约束并结合力/力矩或接触检测。

语义标签本身不能表达抽屉关节；后续对象层还需保存 `joint_type=prismatic`、轴向、开合状态和估计行程。

## 6. 分辨率策略

| 地图 | 建议体素 | 用途 |
|---|---:|---|
| 全局语义地图 | 2–3 cm | 查找对象、规则推理、粗定位、碰撞包围 |
| 粗避障地图 | 3–5 cm | 保守障碍占据与膨胀 |
| 操作局部地图 | 5–10 mm | 抓取、把手定位和接触前校正 |

已有 ChArUco A/B 结果显示：相机坐标深度偏差约 1–4 cm，当前世界同角点典型误差约 3–4 cm。因此首版全局 2 cm 是分辨率上限，不代表 2 cm 绝对定位精度。局部 5–10 mm 地图必须来自执行时的新观测和局部配准。

## 7. 实施阶段与交付物

| 阶段 | 状态 | 工作 | 主要产物 |
|---|---|---|---|
| P1 几何体素基线 | **已完成（离线版）** | 三路表面点体素化与 GLB 可视化 | `voxels.npz`, `voxels.glb` |
| P1.1 操作网格契约 | 待做 | 固定 bbox/origin/frame；schema/version 校验；不再依赖每帧 tight bbox | metadata + 回归测试 |
| P2 单帧语义提升 | 待做 | YOLO adapter、mask remap、head mask→3D、语义投票 | overlays, masks, `semantic_voxels.npz/.glb` |
| P2.1 对象派生 | 待做 | 连通域、AABB/OBB、实例 ID、对象导出 | `objects.json`, object debug GLB |
| P3 多帧融合 | 待做 | 固定 base 网格、多帧累计、ID 关联、过期/动态层 | 持久语义地图与轨迹日志 |
| P4 操作接口 | 待做 | box 抓取面、drawer front/handle、局部精修接口 | 规则系统查询 API |
| P5 占据增强 | 远期 | ray carving、free/unknown、TSDF/ESDF、在线后端 | 可供规划器直接查询的地图 |

P2 开始前需要确定：YOLO 框架/版本、权重路径、类别表、det 或 seg、训练图像是否去畸变、输入尺寸和默认阈值。

## 8. 验收门槛

每个阶段必须留下可复现实验，不以“GLB 看起来差不多”为唯一标准。

### P1.1

- 同一世界点在不同 capture 中得到相同 voxel index；
- metadata 明确 frame、单位、origin、dims、voxel size 和 schema version；
- 越界点有统计和报警，不静默 clip 到边界格。

### P2

- 保存原图、去畸变图和 MapAnything 输入域三层 mask overlay；
- 语义 GLB 可按 class 和 instance 两种方式着色；
- 统计每实例 2D 像素数、有效 3D 点数、占据体素数、被过滤比例和背景泄漏；
- 在人工选取的盒子/桌面边界样本上单独检查标签纯度。

### P3/P4

- 同一静态对象从不同头部姿态观察时保持实例 ID；
- 报告对象中心/OBB 跨帧波动，不把体素大小当作精度；
- 规则层能按 `class + instance_id + confidence + freshness` 查询；
- 最终抓取或拉动前，必须检查局部观测是否新鲜以及不确定度是否低于任务门限。

## 9. 风险与非目标

| 风险 | 处理 |
|---|---|
| mask 边缘混入背景 | 边缘降权/腐蚀、3D 聚类、连通域纯度检查 |
| 三相机仍有厘米级未对齐 | 全局体素不追求虚假高分辨率；局部重观测与配准 |
| 透明、反光、细把手几何缺失 | 保留 unknown；换视角主动观察；必要时使用 RGB-D/已知模型 |
| 物体移动后留下旧语义 | 动态层、时间戳、置信度衰减和清除策略 |
| 只有类别没有对象身份 | 强制保存并维护 `instance_id` |
| 只看表面点却声称 free space | P5 ray carving 前严格区分 unknown 与 free |
| 语义正确但位姿不够抓取 | 操作前局部精修；已知物体可增加 6D pose 模块 |

## 10. 当前推荐的最小闭环

第一轮不要同时实现 TSDF、在线 ROS 和开放词汇模型。先完成一个可测闭环：

```text
一个带实例 mask 的头部 YOLO capture
    → 固定 2 cm 网格
    → head mask 直接写表面语义票数
    → class/instance 着色 GLB
    → 从 box 体素生成 objects.json + OBB
    → 改变头部姿态再采一帧，检查同一对象的位置和标签一致性
```

该闭环通过后，再按 P3 增加多帧实例融合，最后接入抓取与抽屉规则。
