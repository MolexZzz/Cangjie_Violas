# Violas 仓颉版两人垂直并行工作计划

周期：2026 年 7 月 21 日—7 月 30 日  
汇报：7 月 21 日、7 月 23 日、7 月 28 日、7 月 30 日下班前

## 1. 分工原则

不能让一个人只负责代码、另一个人只负责实验。正确分法是让两个人各自完成一条包含
“代码 review/修改 + 数据准备 + Faiss/数据库对比 + 调优”的完整闭环：

| 成员 | 代码主线（目标 1） | 数据与实验主线（目标 2） |
|---|---|---|
| A | 基础检索、自动聚类、representative、mixed search、索引生命周期 | news20、ohsumed、yahoo；Python/仓颉/Faiss |
| B | HDMG 构建与搜索、benchmark loader/runner/指标、配置化 | caltech、cub、coco；Python/仓颉/Faiss/Milvus/Qdrant/Chroma |

两人从 7 月 21 日同时开始：先在各自的三个小样本上跑通，再各自准备对应 full 数据并调优。
任何一方都不把另一方“完成全部代码”作为启动实验的条件。

`violas_python/` 是冻结参考版本，两人均不得修改。

## 2. 已完成的公共准备：只做一次，不构成阶段串行

为避免两人同时修改同一大文件，7 月 21 日已形成以下公共基础：

1. 冻结 [`reproduction-contract.md`](reproduction-contract.md) 中的 record、query、hit 和 Result JSON；
2. 以稳定 recordId、五折、Recall/NDCG 和 Result JSON 作为公共实验契约；
3. 新增 `storage/clustering.cj`、`storage/mixed_scoring.cj` 和 `storage/hdmg.cj` 模块边界；
4. 新增非交互 benchmark、artifact/manifest、Faiss 和外部数据库工具；
5. 完成仓颉构建、核心回归和 Mock 数据库端到端验证。

后续公共接口非必要不改；需要调整时由指定 owner 单独提交，不能夹在个人算法修改中。HDMG 的
配置和评分边界已经拆出，但 graph state/build/walk 仍在 `VectorMap`，这是 B 后续负责的 P1。

## 3. 成员 A：基础/Mixed 检索 + 文本实验闭环

### A1. 代码 review 与修改

- 对照 Python 检查 `VectorMap` 基础插入、搜索和结果排序；
- 对齐自动聚类数量 `n^(1-alpha)`、随机种子和确定性 KMeans；
- 对齐 representative 生成和候选筛选；
- 对齐 base key、key vector、semantic distance、mixed score 与 rerank；
- 修复空的 `buildRepIndex/buildSingleIndex` 或明确替换其语义；
- 建立 `dataVersion/builtFromVersion`，防止陈旧索引继续查询；
- 将距离、聚类、beta、topK 等参数改为显式配置；
- 为基础检索、聚类、mixed search 和索引失效建立回归测试；
- 对照冻结 Python expected JSON 输出逐 query trace。

### A2. 负责的数据集

- news20；
- ohsumed；
- yahoo。

A 独立完成这三个数据集的：

- 当前 sample 数量、维度、类别和采样规则检查；
- full 数据来源、许可、预期规模、预处理步骤和 hash；
- train/query/key-vector/ground-truth artifact；
- sample → regression → full 三档运行配置；
- 数据完整性和可重复性检查。

### A3. 实验与横向对比

在三个文本数据集上独立运行：

- 冻结 Python Violas 参考结果；
- 修改前与修改后的仓颉 representative/mixed search；
- Faiss IndexFlat 精确基线；
- Faiss IVF 或适合文本向量的 ANN 基线；

统一记录 Recall@K、NDCG@K、构建时间、mean/P50/P95 查询延迟，以及条件允许时的内存和索引大小。

### A4. 性能与准确率调优

- 扫描 alpha、beta、cluster count、candidate pool、topK；
- profiling 重复距离计算、全量排序和聚类开销；
- 每项优化保存 before/after、配置、dataset hash 和 commit；
- 不以降低正确率为代价只报告延迟，必须同时给出准确率变化。

### A5. A 的独立验收

- A 的分支可独立构建和运行；
- 三个文本 sample 均有仓颉/Python/Faiss 结果；
- 至少一个文本 full 数据完成闭环，其他 full 状态有证据；
- mixed/representative 与 Python 的差异有逐 query 解释；
- 至少一项正确性修复和一项性能优化有测试；
- 文本侧 Faiss 使用与仓颉相同的 records、queries、GT 和距离口径；
- `git diff -- violas_python` 为空。

## 4. 成员 B：HDMG/Benchmark + 图像实验闭环

### B1. 代码 review 与修改

- 对照 Python 检查 HDMG 节点、边、入口、候选池、graph walk 和 rerank；
- 检查 O(G²) 构图、重复打分、缓存和 Float64 内存成本；
- 将 HDMG 构建和搜索迁入独立模块，避免继续膨胀 `vectormap.cj`；
- 修复 benchmark 硬编码路径、交互式 `readln()`、query cap 和结果 `N/A`；
- 建立 config/CLI 驱动的 image runner；
- 实现 Recall@K、NDCG@K、build time、mean/P50/P95 和 Result JSON；
- 建 Fake/Exact backend 测试 runner，不等待 A 的 mixed search 修改；
- 为 HDMG 构图、搜索、fallback、指标和 reader 建回归测试。

### B2. 负责的数据集

- caltech；
- cub；
- coco。

B 独立完成这三个数据集的：

- 当前 sample 数量、维度、类别和采样规则检查；
- full 数据来源、许可、预期规模、预处理步骤和 hash；
- train/query/key-vector/ground-truth artifact；
- sample → regression → full 三档运行配置；
- 数据完整性、维度、重复项和 round-trip 检查。

特别标明当前 COCO 文件只有 20 条向量，不能作为 full 结果报告。

### B3. 实验与横向对比

在三个图像数据集上独立运行：

- 冻结 Python Violas 参考结果；
- 修改前与修改后的仓颉 HDMG/representative；
- Faiss IndexFlat 精确基线；
- Faiss HNSW 或适合图像向量的 ANN 基线；
- Milvus、Qdrant、Chroma 三个真实 backend。

统一记录 Recall@K、NDCG@K、构建时间、mean/P50/P95 查询延迟，以及条件允许时的内存和索引大小。

### B4. 性能与准确率调优

- 扫描 embedding-k、semantic-intra-k、bridge keys、cluster pool 和 extra hops；
- profiling HDMG 构图、入口搜索、graph walk 和 rerank；
- 比较 exact 与 approximate，不能将不同索引参数混为同一结果；
- 每项优化保存 before/after、配置、dataset hash 和 commit。

### B5. B 的独立验收

- B 的分支可独立构建和运行；
- 三个图像 sample 均有仓颉/Python/Faiss 结果；
- 至少一个图像 full 数据完成闭环，其他 full 状态有证据；
- HDMG 与 Python 的差异有逐 query/逐阶段 trace；
- 非交互 runner 和指标可由 Fake/Exact backend 独立验证；
- 至少一项 HDMG 正确性修复和一项性能优化有测试；
- 三个数据库先在同一 smoke artifact 上完成端到端实验；正式结论使用 full 数据和五折；
- `git diff -- violas_python` 为空。

## 5. LOC 与 Faiss 的共同结论、分别取证

- A 负责统计 Python Violas 核心、仓颉基础/mixed 核心的代码量和功能范围；
- B 负责固定 Faiss commit，统计 Faiss CPU 核心/全部源码，并统计仓颉 HDMG/benchmark；
- 两人各自保存统计命令和结果，最终共同说明统计口径；
- LOC 只用于展示工程规模，不能作为性能或质量结论；
- 真正的 Faiss 对比来自各自三个数据集上的 Exact/ANN 实验。

## 6. 时间表

| 日期 | 成员 A：基础/Mixed + 文本 | 成员 B：HDMG/Benchmark + 图像 | 汇报内容 |
|---|---|---|---|
| **7/21 周二** | 提交 A 计划；复核聚类/mixed 首轮修改和文本 sample | 提交 B 计划；复核 HDMG/benchmark 首轮修改和数据库框架 | 展示已完成修改、P0/P1 遗留和公共实验契约 |
| **7/22** | 扩大文本 sample 对照，定位 Yahoo 差异，复核 Faiss | 扩大图像 sample 对照，补 HDMG/指标测试，准备三数据库服务 | 两条闭环均可独立执行 |
| **7/23 周四** | 提交 mixed/聚类测试、文本基线和首批修复 | 提交 HDMG/runner 测试、图像基线和首批修复 | 两人分别展示代码修改与对应实验，不汇报纯计划 |
| **7/24—7/27** | 准备三个文本 full；完成 Faiss、Float32/索引评估和 profiling | 准备三个图像 full；完成三数据库、HDMG 参数扫描和 profiling | 每人独立推进 full 与调优 |
| **7/28 周二** | 提交文本 full 状态、正确性对齐、性能 before/after | 提交图像 full 状态、HDMG 对齐、性能 before/after | 两条垂直任务分别达到 merge-ready |
| **7/29** | 合并后在统一 commit 上重跑文本实验 | 合并后在同一 commit 上重跑图像实验 | 两人在同一版本上并行回归各自数据集 |
| **7/30 周四** | 提交文本最终结果和已知限制 | 提交图像最终结果和已知限制 | 汇总六数据集、Faiss/数据库比较和后续路线 |

## 7. 合并与冲突控制

- A 主要拥有 `storage/clustering.cj`、`storage/mixed_scoring.cj`、基础 `VectorMap` facade 和文本实验配置；
- B 主要拥有 `storage/hdmg.cj`、`bench/**`、外部数据库适配器和图像实验配置；
- `vectormap.cj` 在公共机械拆分后由 A 维护 facade，B 不直接加入 HDMG 实现；
- B 的 HDMG 通过冻结接口接入，先在 B 自己的 image runner 中直接测试；
- 数据本体、生成索引、数据库文件和大结果不提交 Git；
- 7 月 28 日前各自保持分支可构建，7 月 29 日合并后两人同时重跑各自实验；
- 交叉 review 只给意见，由 owner 修改对应文件。

## 8. 风险与降级

| 风险 | A 的继续路径 | B 的继续路径 |
|---|---|---|
| 某个 full 数据获取失败 | 在另外两个文本集继续，失败数据给 manifest、证据和复现步骤 | 在另外两个图像集继续，COCO 明确标记 sample |
| 公共模块拆分延期 | 在独立 cluster/mixed 模块和文本 runner 开发 | 在独立 HDMG 模块和 image runner 开发 |
| 外部数据库接入失败 | 继续完成文本 Faiss，不重复接入数据库 | 分别保留三数据库版本、配置、错误日志和复现命令，不生成占位结果 |
| 合并出现问题 | 保留 A 分支完整文本结果 | 保留 B 分支完整图像结果 |
| 调优时间不足 | 先提交正确性、profiling 和一项有证据的优化 | 先提交正确性、profiling 和一项有证据的优化 |

## 9. CodeAgent 可选阶段

只有两条垂直任务都完成核心验收后才进入：

- A 负责 repository/module/file/symbol/chunk 的 VectorGroup 与 mixed 检索设计；
- B 负责 symbol lookup、跨文件依赖扩展和相关代码召回 benchmark；
- 只做最小设计或 POC，不影响六数据集结果收尾。
