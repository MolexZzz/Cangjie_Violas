# 科研实践个人工作计划（二）：仓颉 HDMG/Benchmark 与图像数据实验

| 项目 | 内容 |
|---|---|
| 学生 | 待填写 |
| 实践周期 | 2026 年 7 月 21 日—7 月 30 日 |
| 进展汇报 | 7 月 21 日、7 月 23 日、7 月 28 日、7 月 30 日下班前 |
| 主要代码范围 | 仓颉 HDMG 构建/搜索、benchmark reader/runner、指标、配置与结果输出 |
| 主要数据范围 | caltech、cub、coco |

## 一、研究背景与工作定位

本次科研实践以开源 Python 版 Violas 为冻结参考实现，在仓颉侧复现主要实验结果并提高代码
质量、灵活性和全量运行能力。Python 代码仅用于阅读和生成参考结果，不进行功能修改。

当前仓颉版已经包含 HDMG 和评测代码，但初步 review 发现，HDMG 与其他逻辑集中在较大的
`vectormap.cj` 中，构图存在 O(G²) 路径，部分搜索参数硬编码，benchmark 依赖交互式输入，
外部数据库列仍为 `N/A`，也缺少标准 Result JSON 和正式自动化测试。当前 caltech、cub、
coco 文件分别只有 2,020、4,000、20 条向量，其中 COCO 显然只是 smoke 样例，不能直接作为
全量实验依据。

本人将独立完成一条完整闭环，而不是等待另一名成员完成仓颉核心后再开始实验：

> HDMG/Benchmark review → 仓颉代码修改 → 图像数据准备 → Python/仓颉/Faiss/数据库对比 → 准确率与性能调优。

## 二、工作目标

1. 系统 review 仓颉 HDMG 和 benchmark 主路径，形成 P0/P1/P2 问题清单。
2. 对齐 Python 版 HDMG 的构建、入口、遍历、候选和 rerank 行为，并改善模块边界与可测试性。
3. 将 benchmark 改为 config/CLI 驱动，建立稳定 query/GT、指标和 Result JSON。
4. 审计并准备 caltech、cub、coco 三个图像数据集的 sample/full artifact。
5. 在相同输入和参数下对比冻结 Python、修改前后仓颉和 Faiss Flat/HNSW。
6. 统一负责 Milvus、Qdrant、Chroma 三个外部数据库，完成真实服务的可复现实验。
7. profiling HDMG 构建/搜索并完成至少一项准确率改进和一项性能优化。

## 三、现状基线与 Faiss 统计任务

当前初步统计结果如下：

| 范围 | 物理行 | 非空行 |
|---|---:|---:|
| 仓颉 `bench`（当前 4 个文件） | 1,811 | 1,655 |
| 仓颉 `storage`（当前 6 个文件） | 2,118 | 1,879 |
| 全部仓颉 `src`（当前 13 个文件） | 4,172 | 3,750 |
| Faiss CPU 核心（初步） | 130,438 | 112,448 |
| 全部 `faiss/`（初步） | 182,695 | 156,564 |

Faiss 初步统计固定在 commit `7d4bb39f7eb3e9bb4d160aa38ec821ee1a407afc`。正式提交中将保存
统计命令、commit、日期、文件类型和排除项，并复核“CPU 核心”和“全部源码”两个口径。

Faiss 是包含多种索引、量化、SIMD、并行和平台适配的成熟库，与当前 Violas 不属于相同工程
规模。LOC 只用于说明范围，不能说明谁更快或质量更高。本人将把 Faiss Flat/HNSW 作为图像
数据上的 Exact/ANN 基线，用实验指标完成真正有意义的对比。

### 3.1 截至 7 月 21 日的工作基础

- HDMG 已增加数据版本、失效规则、参数校验、fallback 和稳定排序；
- 已删除完整 G×G `Float64` 距离矩阵，构图额外距离内存由 O(G²) 降为 O(G)，但计算量仍为 O(G²)；
- 已拆出 `hdmg.cj` 中的配置与评分边界，graph state/build/walk 尚未完整迁出 `VectorMap`；
- benchmark 已具备非交互命令、Recall/NDCG、mean/P50/P95、build trace 和 Result JSON；
- 已建立稳定 recordId、sample manifest 和 artifact exporter；六个现有文件仍未验证为 full；
- 已建立 Milvus/Qdrant/Chroma 统一适配器和仓颉进程协议；Mock 及三个真实 Docker 服务的
  Caltech smoke 均已通过，Python 90/10/full 正式实验尚未完成。
- 已确认 Python 图像 benchmark 默认使用每类 90/10、`random_state=42` 的随机切分，而仓颉当前
  precomputed runner 使用五折连续切分且缺独立 CLIP 文本 key vector；该差异列为下一项 P0。

以上内容只证明整改和实验框架已经形成，不代表三个数据库或图像 full 实验已经完成。

## 四、具体工作内容

### 4.1 HDMG 代码 Review

重点对照以下行为：

- HDMG node 与 micro-cluster 的定义；
- embedding neighbor、semantic intra-key edge 和 bridge edge；
- 构图参数、边数限制和确定性；
- entry node 的选择；
- graph walk 的深度、hop、visited 与停止条件；
- candidate pool、mixed score 缓存和最终 rerank；
- 图未构建、空图、小图和参数越界时的 fallback；
- 数据变化后的图失效和重建；
- O(G²) 构图、全量排序、重复打分和内存分配。

问题优先级：

- P0：改变 HDMG 结果、GT 或指标；
- P1：阻塞 full 数据、自动运行、性能分析或后端接入；
- P2：非主路径 API、风格或长期增强。

### 4.2 HDMG 代码修改

1. 在现有配置/评分模块基础上，将 HDMG graph state、构建和遍历继续从 `VectorMap` 迁入独立 index 对象；
2. 通过冻结接口接入 VectorMap，避免与基础/mixed 模块交叉修改；
3. 对齐 embedding-k、semantic-intra-k、bridge、candidate pool 和 extra hops；
4. 对齐入口选择、graph walk、visited、score cache 和 rerank；
5. 明确 build/rebuild/invalidation 与 fallback 状态；
6. 增加 build、entry、walk、rerank 分阶段 trace 和耗时；
7. 针对 O(G²)、重复距离和临时数组开展 profiling；
8. 只提交具有正确性测试及 before/after 的优化。

### 4.3 Benchmark 工程修改

- 将数据集、路径、运行规模、seed、topK 和后端改为 config/CLI 参数；
- 移除对交互式 `readln()` 的依赖，支持批处理和 CI；
- 移除隐含 query/folder/vector cap，配置中明确 smoke/regression/full；
- 建 FakeBackend 测试 runner，建 Exact backend 校验 GT 和指标；
- 实现统一 `BackendHit(recordId,key,distance)`；
- 实现 Recall@K、NDCG@K、build time、mean/P50/P95；
- 输出包含 dataset/config hash、commit 和环境的 Result JSON；
- reporter 直接读取 Result JSON，不使用手工抄表或仅保留截图；
- 保留错误、超时和资源不足状态，禁止用 `N/A` 冒充真实实验结果。
- 先将数据库入口对齐 Python 90/10 split；五折作为独立协议实现集合隔离、逐折重建和汇总。

### 4.4 回归测试

- 使用独立小图验证节点、边、入口和遍历顺序；
- 对 Python expected JSON 检查候选池和最终排序；
- 测试未 build、重复 build、数据改变和空图 fallback；
- 使用 Fake Result 验证 Recall/NDCG 和分位延迟；
- 使用 Exact backend 验证 ground truth；
- reader/exporter 做数量、维度和 hash 的 round-trip；
- 测试使用冻结的 mixed scoring 接口和本地 fixture，不依赖另一名成员后续修改或文本 full 数据。

### 4.5 图像数据准备

本人独立负责 caltech、cub、coco：

- 核对当前向量数、维度、类别数和每类采样量；
- 查明 full 数据来源、许可、官方规模和原始划分；
- 固定 image embedding/key-vector 模型、版本、维度和归一化方式；
- 生成稳定 train records、queries、key vectors 和 ground truth；
- 使用稳定 `recordId`，不使用数据库内部 ID 或临时 UUID；
- 输出 manifest、数据 hash、seed 和完整预处理命令；
- 建立 smoke、regression、full 三档配置；
- 检测缺失、重复、维度错误和类别数量异常；
- 数据本体、索引和数据库文件不提交 Git。

COCO 当前只有 20 条向量，正式结果必须标为 smoke。若周期内 full 数据无法获得，将明确记录
数据状态和阻塞原因，不以当前文件代替 full。

### 4.6 图像实验与 Faiss/数据库对比

每个数据集至少比较：

| 方法 | 作用 |
|---|---|
| 冻结 Python Violas | HDMG 参考行为 |
| 修改前仓颉 Violas | before 基线 |
| 修改后仓颉 HDMG/representative | 本人主要结果 |
| Faiss IndexFlat | 精确检索/GT 校验 |
| Faiss HNSW | 图像 ANN 性能基线 |
| Milvus | 外部数据库基线之一 |
| Qdrant | 外部数据库基线之一 |
| Chroma | 外部数据库基线之一 |

所有方法使用相同 artifact、query、GT、distance、topK 和 seed。对于三个数据库，记录实际
索引类型、持久化模式和搜索参数；如果数据库只返回 embedding candidates，则在实验层使用
与参考实现一致的 mixed rerank，不能把纯 embedding 和 mixed search 当成相同方法。

仓颉侧当前通过统一进程协议调用 Python SDK，而不是分别实现三个仓颉原生客户端。正式实验需
同时记录 Python/SDK 版本、服务镜像或版本、endpoint、collection 配置和进程开销；数据库内部
ID 不参与指标，只使用冻结的 `recordId`。

### 4.7 指标与参数调优

准确率指标：Recall@K、NDCG@K、逐 query overlap。  
性能指标：build time、mean/P50/P95 latency；条件允许时记录峰值内存和索引大小。

HDMG 主要扫描：

- embedding-k、semantic-intra-k；
- bridge keys、bridge per key；
- candidate pool、extra hops；
- beta、topK 和图规模。

Faiss 记录 HNSW `M`、construction/search 参数；数据库记录相应索引参数。Exact 和 approximate
分表呈现，并分析准确率—延迟权衡。

## 五、时间安排与阶段成果

| 日期 | 计划工作 | 下班前提交/汇报 |
|---|---|---|
| **7 月 21 日** | 完成 HDMG/benchmark review；复核首轮修改；审计图像 sample 和数据库框架 | 本个人计划；已完成修改、P0/P1 遗留和当前数据/代码量证据 |
| **7 月 22 日** | 对齐 Python 90/10 split 和 CLIP 文本 key vector；扩大图像 sample 对照 | 数据协议、差异清单、Result JSON 和三数据库真实 smoke |
| **7 月 23 日** | 收敛数据/HDMG P0；提交图像 full 获取方案和正式数据库方案 | “数据对齐 + 真实服务 + 图像实验”闭环；明确阻塞项 |
| **7 月 24—27 日** | 准备 full artifact；在三数据库运行 Python 90/10，另做可选五折；完成 profiling | 配置/hash/commit；正式数据库结果；性能 before/after |
| **7 月 28 日** | 收敛 HDMG 差异，至少跑通一个图像 full 数据 | Python/仓颉/Faiss/数据库对照；调优结果；剩余风险 |
| **7 月 29 日** | 合并公共代码后，在统一 commit 上重跑三个图像数据集 | 合并回归结果和最终复现命令 |
| **7 月 30 日** | 整理图像部分最终结果、测试、已知限制和后续建议 | 图像实验包、个人总结和最终报告对应章节 |

## 六、预期产出

1. HDMG/benchmark 主路径问题清单；
2. 独立 HDMG 模块、非交互 runner、指标和结构化结果输出；
3. HDMG/reader/runner/metrics 回归测试及 Python expected trace；
4. 三个图像数据集的 manifest、artifact 工具和运行配置；
5. Python、修改前后仓颉、Faiss、Milvus、Qdrant、Chroma 的结果 JSON；
6. Faiss LOC 统计命令、固定 commit 和口径说明；
7. HDMG 参数扫描、性能 before/after 和已知限制说明；
8. 可由命令、commit、config 和 hash 重复验证的个人实验报告。

## 七、验收标准

- 仓颉项目持续可构建，HDMG/benchmark 测试可独立运行；
- `git diff -- violas_python` 为空；
- 三个图像 sample 均有 Python、修改前后仓颉和 Faiss 结果；
- 7 月 28 日前至少一个图像 full 数据集完成闭环；最终目标为三个图像 full 数据均可运行；确因来源、许可或资源无法完成的，必须提供客观证据且不得以 sample 替代；
- HDMG 主要差异有逐 query、entry/walk/rerank trace；
- 至少一项正确性修复和一项性能优化有测试及 before/after；
- 三个数据库已在同一 smoke artifact 上完成端到端实验；正式结论必须先复现 Python 90/10 和
  full 数据，五折作为单独的稳健性实验；若服务、资源或数据受阻，应提交错误日志和复现命令，
  不以 Mock 结果替代；
- Faiss 源码与实验比较口径清楚，不以 LOC 判断优劣；
- 实验结果绑定统一 artifact、配置、commit 和环境。

## 八、协作边界与风险控制

本人主要维护 HDMG 模块、image runner、通用 benchmark 指标、图像 manifest、数据库适配器和
数据库结果。另一名成员负责 cluster/mixed、基础 VectorMap facade、文本数据和文本 Faiss。
双方冻结公共 recordId、五折、指标和 Result JSON；公共文件由指定 owner 修改，另一方只提交
review 意见。HDMG state 后续通过独立 index 边界迁出，不继续把实现堆入基础 facade。

若 mixed/文本工作延期，本人的 HDMG、Fake/Exact runner、数据库 adapter 和图像实验仍可独立推进；
若某个图像 full 数据受阻，则继续另外两个图像集并记录证据；若数据库服务不能全部完成，按
Milvus、Qdrant、Chroma 分别保留版本、配置、错误日志和复现命令，不产生占位结果。合并后本人
重跑自己负责的三个图像数据集和三数据库，不把最终实验转交给另一名成员。

## 九、可选 CodeAgent 工作

仅在必做项完成后，设计 symbol lookup、跨文件依赖扩展和相关代码召回 benchmark，配合另一名
成员提出的代码 VectorGroup 层级，形成最小实验方案或 POC，不影响 full 数据和 HDMG 收尾。
