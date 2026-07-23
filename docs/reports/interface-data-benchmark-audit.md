# 仓颉版复现 Python 论文实验：必要修改清单

更新日期：2026-07-21

## 1. 目标和边界

目标不是完整重写 Python Violas，也不是一次性补齐所有公开接口，而是让仓颉版在相同数据、相同切分和相同参数下，得到与 Python 论文版本接近的 Recall/NDCG，并能够比较查询延迟。

本阶段遵循两条规则：

- `violas_python/` 冻结，不修改；它是参考实现和 expected result 来源。
- 只修改影响论文实验的数据、算法、参数、指标和性能路径；CRUD、持久化、CodeAgent、多模态关系 API 暂不处理。

## 2. 当前为什么仍不能直接宣称完成复现

目前有四类阻塞。

### 2.1 当前数据不是全量数据

| 数据集 | 当前向量数 | 当前状态 |
|---|---:|---|
| 20 Newsgroups | 400 | 20/class 抽样 |
| OHSUMED | 460 | 20/class 抽样 |
| Yahoo Answers | 200 | 20/class；Python 默认还设置 `sample_ratio=0.01` |
| Caltech-101 | 2,020 | 20/class 抽样 |
| CUB-200 | 4,000 | 20/class；官方数据为 11,788 张图 |
| COCO | 20 | 只有 11 个 folder；不是 COCO-10k |

六个 TXT 可以用于回归测试，但不能代表全量论文实验。六个原始数据集当前也不在工作区。

仓库现已新增独立 artifact audit/exporter 和 manifest，但 full 原始数据尚未取得和验证。因此即使
仓颉选择 `full`，目前仍只是在当前抽样 TXT 上取消数量限制。

### 2.2 仓颉与 Python 的关键算法不一致

首轮 Review 已修复其中一部分；当前状态如下：

1. **自动聚类已首轮对齐。** 仓颉已实现 `n^(1-alpha)`、MT19937、KMeans++、`n_init=10` 和 Lloyd 语义；Yahoo 少量边界 query 仍需继续量化。
2. **Flat 索引状态已显式化，但仍是 exact scan。** `buildRepIndex()`、`buildSingleIndex()` 不再无状态成功，但尚未实现真实 ANN。
3. **HDMG 参数和配置边界已首轮对齐。** 完整 G×G 距离矩阵已移除；graph state/build/walk 仍需独立化，并在 full 下 profiling O(G²) 计算。
4. **数值与计算路径不同。** Python 主要使用 NumPy/float32 矩阵计算；仓颉使用 `Array<Float64>` 和对象循环，结果误差与性能边界都不同。
5. **索引状态管理已首轮修复。** 数据修改会使索引失效；公开容器仍允许调用方绕过受控 API，因此必须调用 `markDirty()`。

group ID、对象 CRUD、关系检索等差异不会直接决定当前六数据集论文指标，可以后置。

### 2.3 评测口径没有完全对齐

仓颉当前评测虽然有 Recall 和 latency，但还存在以下问题：

- Python 默认通常只跑 200 个 query；仓颉 `full` 会跑全部 query，二者不能直接比较。
- Python 与仓颉必须使用完全相同的 train/test split、5 folds、seed 和 representative。
- ground truth 必须来自同一份 exact search 结果，不能由两端各自重新生成。
- beta、alpha、expansion factor、HDMG graph 参数需要逐项相同。
- 仓颉已输出稳定 Result JSON 和逐 query 对照；full 五折汇总仍待完成。
- 已有 build time 和 mean/p50/p95；仍缺统一 warm-up、重复次数、QPS、peak memory 和 index size。

### 2.4 三个横向数据库框架已接入，但真实实验未完成

Python benchmark 已包含 Milvus、Qdrant、Chroma，默认通过 `VIOLAS_ENABLE_EXTERNAL_DBS=1` 开启。

仓颉目前已通过统一进程协议接入 Milvus、Qdrant、Chroma 的建库、插入、查询和 mixed rerank，
并已在三个真实 Docker 服务上完成 Caltech smoke。随后又用同一份 Caltech
`python-paper-90-10` validation artifact 完成仓颉、Faiss 和三个真实数据库的读取验证。
虽然已改用真实 CLIP 文本 key vector，但图片输入仍是 sample，因此这仍不是 90/10/full 正式横向结果，
不得用 smoke、Mock 或 `N/A` 代替。

需要注意，Python 的 mixed database 结果并不是数据库原生混合检索，而是：

1. 数据库按 embedding 取 `topK × 80` 候选；
2. Python 根据 key vector 和 beta 在本地重排。

仓颉要复现该结果，也应该在仓颉 benchmark 层实现同样的候选重排公式。

## 3. 为复现实验必须修改什么

按先后顺序分为五项。

### 修改一：建立相同的实验输入

需要新增独立数据工具，不修改 `violas_python/`：

- 为六个数据集建立 manifest，记录版本、原始数量、split、seed、向量模型、维度和 hash；
- 使用 Python 参考版本只读生成 train/query/key-vector/representative；
- 导出仓颉可流式读取的 artifact；
- 同时导出每个 query 的 exact ground truth；
- 取消 `sample_ratio`、per-class cap 和 `max_queries` 后才算全量。

验收：Python 和仓颉读取同一份 train/query/GT，数量和 hash 完全一致。

### 修改二：对齐仓颉的实验算法

优先改以下路径：

1. 将仓颉自动聚类改为与 Python 相同的簇数公式和 KMeans；固定 random seed。
2. 对齐 representative、base key、key vector 和 mixed distance 的计算公式。
3. 对齐 HDMG 的构图参数、entry selection、候选池和 rerank 逻辑。
4. 集中管理 index dirty/built 状态，数据变化后强制重建。
5. 给 `buildRepIndex/buildSingleIndex` 明确行为：实现真实索引，或在论文复现路径中明确只使用 Exact/HDMG，不能保留无效占位接口。

验收：在当前 7,100 条回归集上，仓颉逐 query 的 top-k、Recall 和 NDCG 与 Python 接近；误差阈值需要在第一次基线后固定。

### 修改三：统一 benchmark runner

仓颉 benchmark 需要支持非交互参数：

```text
dataset artifact
query artifact
ground-truth artifact
alpha / beta
top-k
HDMG graph parameters
max-queries（0 表示全部）
seed
output JSON
```

JSON 至少记录：

- commit、数据 hash、参数和硬件；
- query 数、Recall@K、NDCG@K；
- build time、平均 latency、p50、p95；
- HDMG candidate pool、hops 和 rerank time；
- 失败、跳过和 OOM 原因。

验收：一条非交互命令可以重复运行同一实验，并得到结构相同的结果文件。

### 修改四：解决全量运行的性能阻塞

在当前回归集对齐准确率后，再处理全量性能：

1. HDMG 当前完整距离矩阵是 O(G²)，需要先 profiling；如果 full 数据无法构建，改为 block kNN 或近似构图。
2. Exact/Flat 查询不要对全部结果完整排序，改用 top-k 有界堆。
3. 避免每个 beta、每个 fold 重建相同的 split 和 HDMG。
4. 评估连续 Float32 向量存储，降低 full 数据内存占用；但必须先通过数值误差回归。
5. 将数据读取、构图和查询分别计时，避免把 I/O 混入 query latency。

验收：六个 full artifact 均能完成构建和查询；若受硬件限制失败，必须记录峰值内存、失败阶段和最大可运行规模。

### 修改五：补仓颉横向数据库 benchmark

在仓颉 benchmark 层定义统一 backend：

```text
build(records)
search(query, topN)
stats()
close()
```

依次接入：

1. Qdrant；
2. Milvus；
3. Chroma。

每个 backend 返回相同的 object ID 和 embedding distance。仓颉再按 Python 公式完成 mixed rerank。

正式比较时必须记录索引类型和参数。Exact 与 approximate 分开比较，不能把 Milvus IVF、Qdrant/Chroma HNSW 和 ExactSearch 当成同一种索引直接下结论。

验收：仓颉结果 JSON 中三库不再是硬编码 `N/A`，并且每个结果都能追溯到实际建库、查询和参数。

## 4. 除了对齐 Python，仓颉还应改进什么

这些建议不是为了改变 Python 论文算法，而是为了让仓颉版本更容易替换组件、增加实验和定位性能问题。原则是：本次修改碰到相关模块时顺手建立边界，不要求两周内完成所有后端。

### 4.1 用类型化配置代替字符串和硬编码

当前 mode、distance method、数据路径和大部分 benchmark 参数依靠字符串或源码常量。建议增加：

- `SearchMode`、`DistanceMetric`、`ClusterStrategy` 等枚举；
- `SearchConfig`、`HdmgConfig`、`BenchmarkConfig`；
- config 文件/CLI → 类型化配置的统一解析入口；
- 参数范围校验和最终 effective config 输出。

保留现有字符串入口作为兼容 wrapper，内部统一转换为类型化配置。这样可以减少拼写错误，也方便批量参数扫描。

### 4.2 把可替换算法放到小接口后面

建议逐步形成以下扩展点：

```text
DistanceMetric       距离计算
Clusterer            KMeans 或其他分簇策略
VectorIndex          Exact、Flat、HDMG 或后续索引
DatasetReader        TXT、二进制或流式输入
ExternalVectorBackend  Milvus、Qdrant、Chroma
BenchmarkReporter    JSON、CSV、终端表格
```

Python 兼容行为作为默认实现。新增算法只需实现接口，不应继续向 `VectorMap` 增加更多字符串分支。

### 4.3 分离存储核心、检索策略和 benchmark

当前 `VectorMap` 同时负责数据、索引、搜索、统计和打印。建议至少划分为：

- `storage`：VectorGroup、元数据和受控写入；
- `index`：构建、失效、查询和 index stats；
- `search`：representative、mixed、HDMG 等策略；
- `bench`：数据切分、指标、外部数据库和输出。

本次不必大规模重写，但新增三库、JSON reporter 和配置时必须放入 `bench`，不能继续扩大核心类。

### 4.4 建立明确的索引生命周期

索引应记录所对应的数据版本：

- 每次数据修改增加 `dataVersion`；
- 索引构建后记录 `builtFromVersion`；
- 查询前检查索引是否过期；
- 支持显式 `build/rebuild/invalidate/stats`；
- 不允许空实现将状态标记为“已构建”。

这比在各个 mutation 方法中零散修改多个 Bool 更可靠，也为后续增量索引预留空间。

### 4.5 统一错误、结果和可观测性

核心库不应通过 `println`、`-1` 和 `false` 混合表达失败。建议：

- 定义维度错误、参数错误、数据缺失、索引过期等明确错误；
- 查询返回结构化结果和 trace，不直接打印；
- trace 至少包含候选数、距离计算次数、graph hops 和各阶段耗时；
- reporter 决定输出终端、JSON 或 CSV。

这样既方便 benchmark，也便于未来把 Violas 嵌入其他仓颉程序。

### 4.6 数据输入不应绑定单一 TXT 格式

现有 TXT 适合调试，但全量数据解析慢且占空间。建议 `DatasetReader` 先抽象统一 record 流，再提供：

- 当前 TXT reader，保证兼容；
- 带版本/header/checksum 的紧凑二进制 reader；
- batch/stream 接口，避免调用方一次性加载所有对象；
- schema version，防止 artifact 字段变化后静默误读。

是否切换二进制格式由 profiling 决定，但 reader 边界应在本次 exporter 开发时建立。

### 4.7 建立分层测试，而不只依赖最终 benchmark

建议保留三层验证：

1. 单元测试：距离、聚类、top-k、参数校验；
2. Python snapshot 兼容测试：同一 fixture 的代表向量、候选和最终结果；
3. 数据集回归测试：当前 7,100 条 artifact 的 Recall/NDCG/latency 阈值。

全量 benchmark 用于论文结果，不能替代前两层快速测试。

### 4.8 本次最值得立即落地的四个边界

为控制两周范围，优先真正实现：

1. `BenchmarkConfig`：消除交互入口和硬编码参数；
2. `DatasetReader`：支撑当前 TXT 与后续 full artifact；
3. `ExternalVectorBackend`：统一三个横向数据库；
4. `BenchmarkResult/Reporter`：统一指标、trace 和 JSON 输出。

`DistanceMetric`、`Clusterer`、`VectorIndex` 可以先给出最小接口并让现有实现接入；更复杂的新增算法放到论文结果复现之后。

## 5. 本阶段暂时不需要改什么

以下内容不影响当前六数据集论文结果，暂不纳入第一优先级：

- 修改 Python 参考版本；
- 完整对象 CRUD 和 assign/delete；
- `VectorRef/VectorRelation` 全量移植；
- context、dependency、multimodal 等应用接口；
- 持久化、GPU backend 和 CodeAgent 数据模型；
- 为了“架构好看”大规模重写全部 `VectorMap`。

只有实验主路径对齐并跑通 full 数据后，再处理这些长期能力。

## 6. 两人分工

两人开始实现前，只共同冻结一次
[`reproduction-contract.md`](../plans/reproduction-contract.md)。此后按文件所有权并行开发，
不以对方尚未完成的代码作为自己的前置条件。

### 成员 A：基础/Mixed + 文本闭环

- review 并修改基础检索、自动聚类、representative、mixed search 和索引生命周期；
- 负责 news20、ohsumed、yahoo 的 sample/full artifact；
- 在三个文本集上运行 Python、修改前后仓颉和 Faiss Flat/IVF/HNSW；
- 独立完成准确率、性能调优和结果报告。

### 成员 B：HDMG/Benchmark + 图像闭环

- review 并修改 HDMG、数据 reader、非交互 runner、指标和配置化；
- 负责 caltech、cub、coco 的 sample/full artifact；
- 在三个图像集上运行 Python、修改前后仓颉、Faiss Flat/HNSW、Milvus、Qdrant 和 Chroma；
- 独立完成准确率、性能调优和结果报告。

两人都同时承担目标（1）的代码 review/修改和目标（2）的全量实验/Faiss 对比。完整边界见
[`two-person-parallel-work-plan.md`](../plans/two-person-parallel-work-plan.md)。

### 共同检查点

| 日期 | 必须看到的结果 |
|---|---|
| 7 月 21 日 | 本修改清单、当前数据规模、仓颉缺口结论 |
| 7 月 23 日 | A 提交文本代码修复与文本基线；B 提交 HDMG/benchmark 修复与图像基线 |
| 7 月 28 日 | A、B 各自至少跑通一个 full 数据集，并分别给出准确率/性能 before-after |
| 7 月 30 日 | 合并后 A 重跑三个文本集、B 重跑三个图像集，汇总六数据集和 Faiss/数据库结果 |

## 7. 最终验收标准

只有满足以下条件，才能说仓颉复现了接近 Python 论文版本的结果：

1. `git diff -- violas_python` 为空；
2. Python 与仓颉使用相同 artifact、query、GT、seed 和参数；
3. 当前回归集上逐 query 结果差异有明确阈值和解释；
4. 六个全量数据集没有隐含 sample/query cap；
5. Recall/NDCG 与 Python 接近，并同时给出 latency 和资源数据；
6. Milvus、Qdrant、Chroma 的结果来自真实仓颉 benchmark 调用，不是表格占位；
7. 所有结论可以由 manifest、命令、commit 和结果 JSON 重复验证。
