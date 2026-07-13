# 计划:体素化 + 语义分类对齐(2026-07-11)

> 前置阅读:`PROJECT_LOG.md`。本文档是两个新任务的调研结论与实施计划:
> ① 点云 → 体素(occupancy grid);② 头部相机跑物体分类/检测,并与点云对齐,服务机器人操作。

## 0. 核心结论(先回答问题)

**MapAnything 没有任何语义功能,只做几何重建。** 证据(已在代码库核实):

- 预测头输出适配器(`configs/model/pred_head/adaptor_config/`)只有:pointmap、ray directions、depth、pose、confidence、mask、scale —— 全是几何量,没有类别/分割头
- 数据集代码里出现的 "segmentation" 只是训练用的有效像素掩码(non-ambiguous mask),不是语义标签
- 模型内部虽然用 DINOv2 做图像编码(DINOv2 特征本身含语义信息),但没有暴露任何语义输出接口

因此语义必须由**外部 2D 模型**提供,再"提升"(lift)到 3D。好消息:我们的
`views.npz` 已经存了 `head_pts3d`(H×W×3,世界坐标,米),即**每个像素都有对应 3D 点**
—— 2D 掩码 → 3D 点云只是一次数组索引,不需要任何投影计算。这是整个方案的基石。

## 1. 任务一:点云体素化(occupancy grid)

### 1.1 调研结论

智能驾驶里的"小方块"= **占据栅格(occupancy grid)**,固定分辨率的 3D 格子,每格存占据状态(+可选语义)。主流方案对比:

| 方案 | 特点 | 适用 |
|---|---|---|
| **numpy/Open3D 固定分辨率栅格** | 实现最简单,~50 万点毫秒级,离线一次成图 | ✅ 当前阶段(离线 capture) |
| OctoMap / Bonxai | 八叉树多分辨率,log-odds 概率融合,ROS 生态 | 未来在线建图 |
| Voxblox / **nvblox** | TSDF+ESDF,GPU 加速,常数时间插入,直接给规划器用 | 未来实时操作闭环 |

当前数据是离线单帧 capture,场景经 `--max_radius 2.0` 过滤后约 3×3×2 m,**不需要**上重型框架。选 2 cm 体素 → 约 150×150×100 = 225 万格,稀疏存储(实际占据 ~几万格),纯 numpy 即可。留出接口,未来切 nvblox 时数据结构可平移。

### 1.2 设计

新脚本 **`voxelize.py`**(输入 `views.npz`,纯 CPU,可反复调参,风格同 `filter_export.py`):

- 体素大小 `--voxel_size`(默认 0.02 m,操作级;可视化可用 0.05)
- 工作空间 AABB `--bbox`(默认取过滤后点云包围盒)
- 每个体素聚合:`occupied`、点数、平均颜色、最大 conf、(任务二写入)语义标签+分数
- 实现:`idx = floor((pts - origin)/voxel_size)` → `np.unique(展平索引)` → 分组聚合
- 输出:
  - `voxels.npz` — 稀疏格式:`indices (N×3 int)`、`origin`、`voxel_size`、`colors`、`counts`、`conf`(+ 语义字段)—— 给机器人/下游用
  - `voxels.glb` — 每个占据体素画立方体(trimesh box 实例化,合并成单 mesh 保证性能)。⚠️ 必须应用与 `predictions_to_glb` 相同的**180° 绕 X 翻转**(见 PROJECT_LOG §4)
- 复用 `filter_export.py` 的过滤器逻辑(radius/bbox/conf)作为体素化前的预过滤

### 1.3 进阶(Phase 3,可选)

- **自由空间雕刻(free-space carving)**:利用每视图 `_depth_z` + `_camera_pose` 从光心向每个像素射线投射,射线穿过的体素标记为 free(区分 free/occupied/unknown)。规划避障时有用;纯占据图不需要
- **多 capture 融合**:有了真实外参(全配模式)后,多帧点云在同一机器人坐标系下 log-odds 累积

## 2. 任务二:头部相机物体分类 + 点云对齐

> 逐步技术详解(数据流、每步的原因与陷阱)另见 **`TECH_DETAIL_TASK2.md`**。

### 2.1 模型选型(已定:用户自训 YOLO,2026-07-11 更新)

**分类器采用用户已(大致)训练好的 YOLO 模型**,不再引入 Grounding DINO 等开放词汇方案(原调研结论存档见 git 历史/本节末备注)。这带来两个必须先确认的分支点:

**分支 A:检测版(boxes)还是分割版(seg,带掩码)?**

| 情况 | 2D→3D 提升方式 |
|---|---|
| YOLO-seg(有掩码) | 最理想,掩码直接进 §2.2 流程,无需其他模型 |
| YOLO-det(只有框) | 框内像素 ≠ 物体像素(含背景),需二选一:<br>**A1(推荐)**:框作为 prompt 喂给 SAM2 出掩码 —— YOLO 管"是什么",SAM2 管"哪些像素",各司其职,掩码质量最好<br>**A2(零依赖备选)**:纯几何法 —— 取框内全部 `pts3d`,按深度中值/DBSCAN 聚类取最近的主簇作为物体点。无需装 SAM2,但细长/镂空物体和贴近背景的物体会失真 |

**分支 B:YOLO 训练用的是原始(带畸变)图还是去畸变图?**

对齐要求掩码/框最终落在**去畸变图像域**(与 `head_pts3d` 逐像素对应)。头部相机畸变较小,但不能忽略:

| 情况 | 处理 |
|---|---|
| 训练用去畸变图 | 直接在去畸变 PNG 上推理,零转换 |
| 训练用原始图 | 在原始图上推理(保持训练域,精度最优),然后把结果**用 `undistort.py` 的同一套 remap 映射到去畸变域**:掩码直接 `cv2.remap`(最近邻);框不能只映射四角(直线在畸变下弯曲),应先框→填充掩码→remap。若懒得转换也可直接在去畸变图上推理碰运气 —— 头部相机畸变小,域差可能可接受,但需实测 mAP 不掉 |

**其余需要用户提供的信息**(P2 开工前):框架与版本(ultralytics YOLOv8/11?还是 darknet 等)、权重文件路径、类别列表、训练输入分辨率、置信度阈值习惯值。

### 2.2 对齐方案(关键设计)

**不需要外参、不需要投影矩阵运算** —— `views.npz` 的 `head_pts3d` 与 `head_img` 逐像素对齐:

1. **YOLO 跑在全分辨率头部图上**(而非 518 缩放图,小物体检测质量更好);推理域按 §2.1 分支 B 决定,若在原始图上推理则先把结果 remap 到去畸变域
2. 得到掩码(seg 版直接出;det 版走 §2.1 分支 A1/A2),用**最近邻**缩放到 `head_pts3d` 的分辨率(推理时的 518 集合分辨率)
3. `obj_pts = head_pts3d[mask & valid_mask]` —— 直接得到该物体的世界坐标点云
4. **清洗**:掩码先腐蚀 1–2 px(防止物体边缘深度不连续处的"渗色",这是最常见的误差源)→ radius 过滤 → DBSCAN 取最大簇(去除掩码泄漏到背景的散点)
5. 每个物体输出:`label, score, centroid(m), AABB/OBB, 主轴, 点数` → `objects.json`(世界坐标系,head 相机=原点)
6. **写入任务一的体素栅格**:物体点所在体素记语义标签,冲突时按检测分数加权多数投票 → **语义占据栅格**(即智能驾驶里的 semantic occupancy)

### 2.3 与机器人操作的衔接

- 当前世界系 = head 相机光心系。拿到**头部相机→机器人基座的外参**(正好是全配模式计划要接入的同一份手眼标定)后,一个 4×4 矩阵把 `objects.json` 和体素图变换到 base 系,机械臂可直接用
- 手腕相机视图同样有 `_pts3d`,同一世界系 → **多视图语义融合免费获得**(左右手看到同一物体,体素级投票消歧),作为可选增强
- 已知尺寸物体(如标准水杯)的 OBB 尺寸可反过来**验证米制尺度**(呼应 PROJECT_LOG §3.3 的尺度校验)

### 2.4 已知风险

| 风险 | 对策 |
|---|---|
| 掩码边缘深度渗色 | 腐蚀掩码 + DBSCAN 最大簇(§2.2 步骤 4) |
| 透明/反光物体深度差 | MapAnything 固有限制;用 conf 过滤,必要时物体点云取掩码内中值深度重建 |
| 518 分辨率下小物体 pts3d 稀疏 | 检测在全分辨率做;物体太小则点数阈值报警 |
| YOLO 训练域 ≠ 推理域(畸变/光照/相机差异) | 按 §2.1 分支 B 保持训练域推理再 remap;先在 4 个 capture 上做定性验收(overlay PNG 人工检查) |
| "大致训练好" —— 精度未定量 | P2 第一步就是在真实 capture 上跑通并出 overlay 图评估;漏检/误检严重则反馈重训,不阻塞管线开发(管线对权重文件是热插拔的) |
| 闭集类别:操作对象不在训练类别里就不可见 | 属于模型能力边界,新物体需回炉加数据重训;若未来需要临时识别新物体,可再补开放词汇方案(原调研:Grounding DINO + SAM2)作旁路 |

## 3. 实施顺序

| 阶段 | 内容 | 依赖 |
|---|---|---|
| **P0 环境** | `pip install open3d` + YOLO 运行时(ultralytics 系则 `pip install ultralytics`,按用户框架定)+ 仅当走分支 A1 时装 SAM2;⚠️ 装前 `pip check`,防止重蹈 hydra 冲突覆辙(PROJECT_LOG §5);全部装入现有 `mapanything` env,冲突则单开 env。另需用户提供 YOLO 权重文件与类别表 | 无 |
| **P1 体素化** | `voxelize.py`:几何占据栅格 + GLB 可视化,4 个 capture 全跑 | P0 |
| **P2 语义** | `detect_objects.py`(检测+掩码,存 overlay PNG + masks.npz)→ `semantic_lift.py`(2D→3D 提升、清洗、objects.json、写语义进体素) | P0, P1 |
| **P3 整合** | 接入外参+深度全配模式(原计划)后:base 系输出、多视图语义融合、自由空间雕刻、多帧融合 | 外参/深度数据到位 |
| **P4 实时化**(远期) | 自训 YOLO 本身已具备实时能力,主要工作在建图侧(nvblox 增量语义占据图)| 机器人侧需求明确后 |

P1、P2 与"外参+深度接入"完全解耦(都只消费 `views.npz` 和去畸变图),可以先做,不用等标定数据。

## 4. 参考来源

- [Grounding DINO (IDEA-Research)](https://github.com/idea-research/groundingdino) — 零样本开放词汇检测,COCO zero-shot 52.5 AP
- [DINO-X: Unified Open-World Detection](https://arxiv.org/html/2411.14347v1)
- [Grounded SAM 对比 (Roboflow)](https://playground.roboflow.com/models/idea-research/grounded-sam)
- [Semantic OctoMap 分割方法对比 (MDPI 2025)](https://www.mdpi.com/2076-3417/15/13/7285) — 语义体素:每格存类别+置信度+占据概率
- [KRVF: 边缘移动操作的语义体素世界表示 (2026)](https://arxiv.org/pdf/2606.26321)
- [OctoMap vs Voxel Grid (Robotics SE)](https://answers.ros.org/question/186783/difference-between-octomap-and-voxel-grid/)
- [OmniMap: 光学+几何+语义统一建图框架](https://arxiv.org/pdf/2509.07500)
