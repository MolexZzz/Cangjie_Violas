# 历史报告：仓颉版 Violas 代码审查与修改记录

> 本文保留开发阶段审查过程；发布架构与限制以正式文档为准。

日期：2026 年 7 月 21 日  
参考版本：仓库当前冻结的 `violas_python/`  
修改边界：核心实现仅修改 `cj_core/`；新增独立的 `tools/`、`manifests/` 实验工具，`violas_python/` 保持只读

## 1. Review 结论

A/B 两份计划覆盖了实验主路径的大部分主要问题，包括：

- Python—仓颉算法差异；
- 自动聚类、representative 和 mixed search；
- HDMG 构建、搜索、候选池和 rerank；
- 索引生命周期与空索引接口；
- benchmark 硬编码、交互运行、指标和结果输出；
- full 数据、Faiss 和三个外部数据库；
- 参数化、测试、profiling 和模块边界。

但原计划仍遗漏了几项会直接影响正确性或可复现性的基础问题：

1. 距离函数在向量维度不一致时返回 `0.0`，会把错误输入当成最相似结果；
2. `addVector` 修改数据后未使 HDMG 失效；
3. `HashMap` 遍历顺序与 `maxQueries` 截断组合后，可能导致每次评测选择不同 query；
4. 同距离结果没有稳定 tie-break，排序可能漂移；
5. HDMG 构建分配完整 G×G `Float64` 距离矩阵，full 数据时内存风险高；
6. beta、topK、候选数、HDMG 邻居数等参数缺少边界检查；
7. TXT reader 未充分检查字段数和向量/代表向量维度；
8. 文档要求 NDCG、P50、P95，但代码实际只有 Recall 和平均延迟；
9. COCO 当前只有 20 条向量，runner 却硬编码显示 `COCO-10k`；
10. `buildRepIndex/buildSingleIndex` 是无状态空实现，调用成功无法判断索引是否有效。

因此，两份计划的方向正确，但不能认为 review 已经穷尽问题。本次将上述遗漏纳入 P0/P1。

### 1.1 Review 成果与后续复核责任分配

为便于两人分别向老师汇报，同时避免把工作拆成“先改代码、后做实验”的串行阶段，现将本轮
Review 成果按两条完整技术链平均分配。这里的“负责”包括：解释问题、复核修改、维护回归测试、
补充对应数据集实验以及对该部分结论负责；公共配置和结果格式由两人交叉检查。

| 成员 A：基础/Mixed 与文本闭环 | 成员 B：HDMG/Benchmark 与图像闭环 |
|---|---|
| 自动聚类与 Python KMeans 行为对照 | HDMG 构建、搜索、候选池和 rerank 行为对照 |
| representative、single、mixed search 正确性 | HDMG 生命周期、失效规则和构建统计 |
| 距离计算、维度检查、稳定排序和 tie-break | benchmark 参数校验、数据读取和非交互入口 |
| rep/single 索引生命周期和 exact-scan 状态说明 | Recall/NDCG、P50/P95 和 Result JSON 输出 |
| `clustering.cj`、`mixed_scoring.cj` 模块边界 | `hdmg.cj`、`runner.cj`、`evaluations.cj` 模块边界 |
| News20、OHSUMED、Yahoo 三个文本数据集的逐 query 对照 | Caltech、CUB、COCO 三个图像数据集的逐 query 对照 |
| Faiss Flat/IVF/HNSW 文本侧基线与参数记录 | Milvus/Qdrant/Chroma 统一框架及图像侧数据库实验 |
| 文本侧 full manifest、query/GT 和 profiling | 图像侧 full manifest、query/GT 和 profiling |

两人的公共验收项仅包括稳定 recordId、明确命名的 split 协议、指标定义、实验 JSON schema 和最终汇总表，
不要求双方共同修改同一核心文件。A 可以用 Mock/Faiss 和文本 sample 独立验证自己的链路；B 可以用
Mock/外部数据库适配器和图像 sample 独立验证自己的链路。真实 full 数据未就绪或数据库服务失败时，
均应显式记录阻塞条件，不以 `N/A`、Mock 或 sample 结果代替正式结果。

## 2. 本轮已经完成的修改

### 2.1 自动聚类

- 聚类数量改为真实的 `round(n^(1-alpha))`，并限制在 `[1,n]`；
- 移除只在 alpha 接近 0/1 时处理、其他情况固定近似 `sqrt(n)` 的逻辑；
- 移除按输入顺序平均切块的伪聚类；
- 首轮增加确定性 Lloyd KMeans；后续改为兼容 NumPy MT19937、KMeans++、`random_state=42`、
  `n_init=10` 和 Lloyd 收敛语义的实现；
- 聚类名称和子 key 统一使用四位编号；
- 校验组内向量维度。

兼容 KMeans 使 OHSUMED、Caltech、CUB 的已测 representative 差异消失，并显著缩小 Yahoo 差异。
由于 scikit-learn 的 Cython Lloyd 在并行求和、浮点累计和空簇边界上仍可能与纯仓颉实现不同，
不能宣称逐位等价；最终仍以逐 query Top-K 报告验收。

### 2.2 索引生命周期

- 增加 `dataVersion`；
- 增加 rep、single、HDMG 的 `builtFromVersion`；
- insert、setKeyVectors、addVector 后统一失效；
- HDMG 搜索发现未构建或版本陈旧时走明确 fallback；
- `buildRepIndex/buildSingleIndex` 不再是完全无状态空函数，记录精确遍历快照版本；
- 新增 `getIndexState()`；
- `getIndexState()` 明确返回 `exact_scan_snapshot`，避免把版本快照接口误称为 ANN；
- 新增 `markDirty()`，供直接修改公开容器的调用方显式失效索引。

当前 rep/single 仍是 exact-scan 语义，并未实现真实 ANN 索引，不能表述为索引加速已经完成。

### 2.3 检索正确性与稳定性

- 维度不匹配改为抛出 `IllegalArgumentException`；
- 未知距离方法和检索模式不再静默回退；
- topK≤0 安全返回空结果；
- beta/alpha 限制到 `[0,1]`；
- group expansion、candidate pool 和 HDMG 搜索参数增加边界处理；
- SearchResult、FlatScore 和 HDMG score 增加稳定 tie-break；
- VectorMap key、dataset folder 和 HDMG base key 使用稳定排序；
- query cap 按稳定 fold/folder/vector 顺序执行。

### 2.4 HDMG 全量可运行性

- 保留 O(G²) 邻居计算，但移除完整 G×G `Float64` 距离矩阵；
- 构图峰值额外距离内存从 O(G²) 降为单节点候选所需的 O(G)；
- 同 key 和 bridge 候选距离改为按需计算；
- 构建和搜索参数增加非负校验；
- build version 与数据 version 绑定。
- build stats 记录节点数、embedding/semantic 边数、数据版本和构建耗时。

该修改降低内存风险，但没有消除 O(G²) 计算时间。后续需要在 full group 数量下 profiling，
再决定是否引入 ANN 邻居构建。

### 2.5 Benchmark 与数据读取

- `EvalScale` 增加 folds、topK 和规模参数校验；
- TXT parser 增加 FOLDER/VECTOR/GLOBAL_REP/FOLD_REP 字段检查；
- 检查同 folder 的向量、global rep 和 fold rep 维度；
- `limitDataset` 改为稳定排序后截断；
- COCO 标签改为“scope 由 manifest 确认”，不再声称当前文件是 10k；
- 新增单行自动化入口：`bench <smoke|partial|full> <1..6|t|v|a>`；
- 实现二元相关性 NDCG@K；
- 保存每次查询延迟，并输出 mean、P50 和 P95；
- 修复 key-level ground truth 含重复 key 时 NDCG 理想分母重复计数。

### 2.6 数据 artifact、recordId 与 ground truth

新增独立工具 `tools/precomputed_artifacts.py`，不修改或调用 Python 参考实现：

- `audit` 对六个 precomputed TXT 计算 SHA-256、字节数、folder/vector/rep 数量和维度；
- `export` 按稳定 folder/向量顺序生成 `records.jsonl`、`queries.jsonl`、`ground_truth.jsonl` 和 `manifest.json`；
- recordId 统一为 `<dataset>/<folder>/<zero-based-index:08d>`；
- query/ground truth 固定 5-fold 划分和 beta-mixed exact 口径；
- 生成可提交的 `manifests/current-artifacts.json`，大体积导出放在被 `.gitignore` 排除的 `artifacts/`。

当前清单的结论如下：

| 数据集 | folders | vectors | dim | 结论 |
|---|---:|---:|---:|---|
| news20 | 20 | 400 | 400 | sample，未验证 full |
| ohsumed | 23 | 460 | 460 | sample，未验证 full |
| yahoo | 10 | 200 | 200 | sample，未验证 full |
| caltech | 101 | 2020 | 512 | sample，未验证 full |
| cub | 200 | 4000 | 512 | sample，未验证 full |
| coco | 11 | 20 | 512 | sample，未验证 full |

因此当前六个文件都不能作为“六个全量数据集已经齐全”的证据。

### 2.7 Frozen Python—仓颉对照

新增 `tools/compare_python_cangjie.py`：

- Python 侧只 import 冻结的 `violas_python/`；
- 仓颉侧由核心回归入口输出机器可读 TRACE；
- 比较 cluster partition 以及左右 query 的 Top-3 id；
- 结果写入 `manifests/python-cangjie-fixture.json`；
- 当前固定双簇样例三项比较均为 `equal: true`。

### 2.8 可复现实验配置与 Result JSON

- 新增 `AlignedBenchmarkConfig`，集中管理 cluster alpha、beta 列表、HDMG 建图和搜索参数；
- benchmark 构建和查询不再分别硬编码两套参数；
- 每个 beta 记录索引构建 mean ms/fold；
- 仓颉侧输出单行 `RESULT_JSON|...`，内容包括 scale、完整 config、Recall/NDCG、mean/P50/P95、
  HDMG trace 和 build time；
- 新增 `tools/run_cangjie_benchmark.py`，运行非交互 benchmark、校验标准 JSON，并补充运行时间、
  Git commit、命令、OS/CPU 架构、Cangjie/cjpm 版本等 provenance 后写入 `results/`；
- `results/` 已加入 `.gitignore`，避免把本机实验输出误提交为基准结果。

### 2.9 回归入口

新增仓颉核心回归检查，覆盖：

- `n^(1-alpha)` 聚类数量；
- 交替排列的两个簇不会被顺序切块；
- rep/single/HDMG build state；
- rep/single 的 exact-scan 语义及 HDMG 构建统计；
- addVector 后索引失效；
- topK=0；
- beta 限幅；
- 维度不匹配显式失败。

运行方式：

```powershell
cd cj_core
"2" | cjpm run
```

### 2.10 六数据集真实 sample 的逐 query 对照

新增仓颉非交互入口：

```powershell
"parity 1 3 0.3" | cjpm run
```

它在真实 precomputed 数据上固定 fold-0、每类最多 20 条、Top-3 和 beta=0.3，逐 query 输出
exact、representative、mixed、HDMG 的稳定 recordId。新增
`tools/compare_sample_queries.py`，使用冻结的 Python `VectorMap` 重建相同数据、fold 和参数，并生成
`manifests/python-cangjie-sample-queries.json`。

recordId 使用 `<folder>/<zero-based-index:08d>`，不再依赖可能重复的文件名或文本 description。
六数据集当前每个最多检查前三个 query（COCO fold-0 当前只有一个 query），结果为：

| 数据集 | exact overlap@3 | representative overlap@3 | mixed overlap@3 | HDMG overlap@3 |
|---|---:|---:|---:|---:|
| news20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ohsumed | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| yahoo | 1.0000 | 0.8889 | 1.0000 | 1.0000 |
| caltech | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| cub | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| coco | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

前三个 query 口径下，除 Yahoo representative 外均已完全一致。进一步扩大 Yahoo 到 20 queries 后，
exact overlap@3=1.0、representative=0.9、mixed=0.9833、HDMG=0.9667。剩余差异集中在少量聚类
边界和候选排序，不修改冻结 Python 版本，也不将其隐藏为平均指标。

### 2.11 仓颉存储算法模块拆分

新增三个独立模块：

- `storage/clustering.cj`：cluster count、编号和确定性 Lloyd KMeans；
- `storage/mixed_scoring.cj`：alpha/beta 限幅、base-key 规则和统一 mixed distance；
- `storage/hdmg.cj`：HDMG score、`HdmgBuildConfig` 和 `HdmgSearchConfig`。

`VectorMap` 已删除对应的私有重复实现，并新增 `buildHdmgWithConfig`、`searchHdmgWithConfig`；
aligned benchmark 和 parity 路径均通过配置对象调用。核心回归增加独立聚类、mixed scoring、HDMG
配置构建/搜索检查。当前拆分完成了可替换算法和配置边界，但 HDMG graph state 仍由 `VectorMap` 持有，
若未来要支持多种 graph backend，还需把 state/build/walk 完整迁移为独立 index 对象。

### 2.12 Faiss exact/IVF/HNSW 基线

新增 `tools/run_faiss_baseline.py`，不修改 Python 论文版本，支持：

- `IndexFlatIP` exact cosine 基线；
- `IndexIVFFlat` approximate 基线；
- `IndexHNSWFlat` approximate 基线；
- 稳定 brute-force ground truth；
- Recall@K、NDCG@K、build time、mean/P50/P95、序列化索引大小；
- float32、Faiss/NumPy/OS/Git provenance；
- 与仓颉结果相同的 JSON 顶层结构风格。

运行方式：

```powershell
python tools\run_faiss_baseline.py --dataset news20
```

当前 Faiss 1.13.2、news20 sample、fold-0、320 个训练向量、20 queries、Top-3 的一次本机验证中，
Flat/IVF/HNSW Recall@3 均为 1.0；平均查询耗时分别约 0.0275/0.0337/0.0451 ms。数据量太小，
不能据此判断 approximate index 优劣。Faiss 基线只比较单向量 cosine 检索，不包含 Violas 的语义
mixed score，后续报告必须分栏呈现，不能把两种任务混为一个排名。

### 2.13 Milvus/Qdrant/Chroma 统一框架

新增仓颉 `bench/external_backends.cj`：

- 定义统一 `ExternalVectorBackend`；
- `ProcessExternalVectorBackend` 使用仓颉标准库 `std.process.executeWithOutput` 启动适配器；
- 新增命令 `dbbench <mock|milvus|qdrant|chroma> <1..6> <smoke|partial|full>`；
- 仓颉侧接收并输出 `DB_RESULT_JSON`，数据库失败会显式抛错，不再写 N/A 冒充结果。

新增 `tools/external_db_benchmark.py`：

- 统一 MockExact、Milvus、Qdrant、Chroma backend 协议；
- 数据库内部 ID 不参与指标，统一使用稳定 recordId；
- 同时输出数据库原始 embedding Top-K 和 `Top-K × 80` 候选的 Python 论文公式 mixed rerank；
- 输出 Recall/NDCG、建库时间、数据库 P50/P95、rerank/total latency 和 backend 配置；
- 当前已通过 `dbbench mock 1 smoke` 的仓颉端到端验证；
- 2026-07-21 已启动真实 Milvus 2.3.15、Qdrant 1.16.1 和 Chroma 1.5.5 服务；
- 三个后端均通过 `dbbench <backend> 4 smoke`，使用 Caltech fold-0、1616 条训练向量和 20 条查询；
- 真实 smoke 只证明连接、建库、查询和结果回传可用，不作为数据库性能排名或 full 结果。

使用说明见 [`external-database-benchmark.md`](../guides/external-database-benchmark.md)，可选依赖见
`tools/requirements-external-db.txt`。当前入口固定 fold-0 用于框架验证；正式复现应先支持 Python
90/10 split，新增五折实验则需逐折重建集合。

### 2.14 Python 图像数据流程与仓颉 artifact 的口径差异

进一步核对冻结 Python 的 Caltech/CUB/COCO benchmark 后发现，现有 Python 代码默认按类别调用
`train_test_split(test_size=0.1, random_state=42, shuffle=True)`，即 90% 建库、10% 查询；图像向量
来自 CLIP `ViT-B/32`，semantic key vector 来自文本模板 `a photo of a {key}` 的归一化 CLIP 文本向量。

当前仓颉 precomputed runner 则按 TXT 顺序做五折连续切分，并把 fold train representative 同时用作
key vector。现有 TXT 也没有独立 `KEY_VECTOR` 字段。因此当前逐 query parity 只能证明 Python
`VectorMap` 与仓颉在“同一 precomputed artifact 口径”下基本一致，不能直接证明已经复现论文
benchmark 的原始数据流程。

后续必须将两种协议分开命名：

1. `python-paper-90-10`：复现 Python 指标，使用相同 CLIP、同一图片顺序、同一随机切分和独立文本 key vector；统一 artifact 及仓颉/Faiss/三数据库读取入口已经实现，详见 [`python-paper-90-10-protocol.md`](../guides/python-paper-90-10-protocol.md)；
2. `five-fold`：作为新增稳健性实验，Python、仓颉、Faiss 和数据库共同读取同一冻结 split。

仓颉仍保持两阶段设计：第一阶段使用冻结 Python 预处理逻辑一次性生成共享 artifact；第二阶段仓颉
只负责读取 artifact、建图和检索。仓颉侧不重复实现 CLIP，否则模型版本和预处理差异会引入新变量。

## 3. 已完成验证

### 构建与核心回归

```text
cjpm build
→ cjpm build success

"2" | cjpm run
→ Core regression checks passed.
```

### 真实 artifact smoke

```powershell
"bench smoke 1" | cjpm run
```

news20 当前 400 条向量、20 类、400 维 artifact 已完成 5-fold、20 query cap、11 组 beta 的
端到端运行。输出已经包含 Recall@3、NDCG@3、mean/P50/P95 和 HDMG entry/walk/rerank trace。

例如当前 smoke 在 beta=0.30 时得到：

- HDMG mixed Recall@3：0.8833；
- HDMG mixed NDCG@3：0.9117。

这些数字只证明当前小 artifact 链路可运行，不能作为论文 full 结果。

`git diff -- violas_python` 为空。

### Artifact 与跨语言对照

```powershell
python tools\precomputed_artifacts.py audit
python tools\precomputed_artifacts.py export --dataset news20 --output-dir artifacts\news20-smoke --max-queries 5
python tools\compare_python_cangjie.py
```

上述命令已经成功生成六数据集清单、news20 稳定 query/GT artifact，并得到
`manifests/python-cangjie-fixture.json` 的 `allEqual: true`。

### Result JSON 落盘

```powershell
python tools\run_cangjie_benchmark.py --scale smoke --dataset 1
```

已成功生成并用 Python 标准库解析 `results/cangjie/dataset-1-smoke.json`。文件包含 11 个 beta 的
指标、延迟、构建时间、HDMG trace、配置和 provenance。该目录是本机产物，不纳入 Git。

当前仓颉源码规模为 13 个 `.cj` 文件、4172 个物理行、3750 个非空行。这个数值是修改后的工程规模，
后续与 Faiss 比较时应区分“本仓库实现代码”“benchmark/示例代码”和“Faiss 整个成熟项目”，不能直接用
一个总行数推导性能或开发质量。

## 4. 尚未完成、不得提前宣称完成的事项

下面的问题是当前版本的真实边界。P0 直接决定能否声称“接近复现 Python 论文实验”；P1 不一定
立即改变 sample 准确率，但会影响 full 数据运行、性能结论和后续开发。主责只表示推进人，最终实验
口径仍需两人共同确认。

### P0：复现实验前必须继续处理

| 当前问题 | 影响 | 主责 |
|---|---|---|
| 三个图像 full 原始数据已下载并核验规模；三个文本 full 数据以及六套统一预处理仍待完成 | 只有原始图片不等于已完成全量复现 | A 负责文本，B 负责图像 |
| 逐 query 对照目前只覆盖少量 sample query | 小样例一致不能证明 full 一致 | A 负责文本，B 负责图像 |
| Yahoo 扩展测试仍存在少量 representative/mixed/HDMG Top-K 差异 | 可能影响最终 Recall/NDCG | A |
| full query、Python 90/10 split、可选五折和 ground truth 尚未冻结 SHA-256 | 不同实现可能实际使用不同实验输入 | A/B 各负责三个数据集 |
| Python 论文脚本的采样、候选池、GT 和指标口径仍需逐项核对 | 可能出现“指标同名但任务不同” | A 复核搜索公式，B 复核 benchmark/GT |
| 三套图像 full artifact、固定 CLIP 哈希和逐行共享输入审计已完成；Caltech 已跑 200 query，CUB/COCO 及外部数据库 full 指标待跑 | 目前只能报告 Caltech 阶段结果，不能宣称三套图像实验全部完成 | B |
| 三个真实数据库已完成 Caltech 统一 90/10 artifact smoke，但尚未完成 full 正式运行 | 当前只能证明同一冻结输入下的真实服务链路可用 | B，A 交叉复核统一指标 |

### P1：全量性能和灵活性

| 当前问题 | 影响 | 主责 |
|---|---|---|
| rep/single 仍是 exact scan，不是真实 ANN 索引 | 接口名称容易使人误认为已经有索引加速 | A |
| HDMG graph state/build/walk 仍由 `VectorMap` 持有 | 后续替换图后端和独立测试不够灵活 | B |
| 主数据通路仍以 Float64 为主，尚未评估 Float32 | 与 Faiss/数据库比较时内存和吞吐口径不一致 | A 做基础通路，B 做 benchmark 统计 |
| HDMG 构图虽已取消 O(G²) 内存矩阵，但计算量仍为 O(G²) | full group 数量较大时可能成为主要瓶颈 | B |
| 尚未统一记录峰值内存、索引大小、预热和重复次数 | 性能数据不足以形成公平结论 | B 设计口径，A 补文本结果 |
| 三数据库真实 smoke 已跑通，但服务版本、索引参数和重复次数尚未完成正式校准 | 横向结果可能受默认参数影响 | A 负责 Faiss，B 负责三个数据库 |
| 目前依赖可执行回归入口，尚无正式 `cjpm test` 测试目录 | 自动化回归和持续集成能力不足 | A 负责算法测试，B 负责 benchmark 测试 |

### 已知接口限制

- `VectorMap.data`、`VectorGroup.vectors` 仍是公开可变容器，绕过 VectorMap 修改时必须调用 `markDirty()`；
- append 后 representative 不会自动重算，因为 representative 可能是外部模型生成值而不是均值；
- 仓颉 groupId 仍是本地 16 位 hash，与 Python MD5 32 位不一致；
- 仓颉侧通过进程协议调用三个数据库的 Python SDK，并非仓颉原生数据库客户端；当前优点是接口统一、
  可替换，代价是需要管理 Python 环境、服务地址和进程错误；
- 数据库框架已能读取 Python 90/10 artifact；五折自动循环、collection 隔离和结果汇总仍待完成；
- 尚未实现持久化、GPU backend、完整 CRUD/relations 或 CodeAgent 数据模型。

## 5. 下一步建议顺序

1. A 扩大三个文本 sample 的 query 对照并定位 Yahoo 边界差异；B 同时扩大三个图像 sample 对照；
2. A/B 分别获取文本/图像 full artifact；图像先冻结 Python 90/10 query/GT，五折作为独立协议；
3. A 在文本侧运行仓颉与 Faiss；B 在图像侧运行仓颉、Faiss 与三个数据库，统一记录资源和延迟；
4. A 推进 Float32、rep/single 索引和算法测试；B 推进 HDMG profiling、graph state 抽离和 benchmark 测试；
5. 两人交换一组数据复跑，检查 schema、指标和实验命令可复现性；
6. 汇总修改前/后、Python、Faiss、三个数据库的结果，并明确 exact、ANN 和 mixed 任务的区别；
7. 如果前述任务按期完成，再进入 CodeAgent/仓颉项目组织的可选阶段。
## 2026-07-22 HDMG 性能对齐

Caltech 冻结 `python-paper-90-10` artifact 的性能审计发现，仓颉工程虽然输出到
`target/release`，但编译器未收到优化选项，实际使用默认 `-O0`；同时每次余弦距离
都会重复计算两个向量范数。Python 参考实现则使用预归一化 Float32 矩阵、批量矩阵
乘法和部分 Top-K 选择。

本轮在不改变 HDMG 搜索参数和评价口径的前提下完成：

- 仓颉编译选项改为 `-O2`；
- 冻结论文 artifact 在加载时统一执行 L2 归一化；
- 新增 `cosine_normalized` 距离路径，以 `1 - dot(a, b)` 代替重复范数计算；
- HDMG 构图、入口选择、图遍历、候选重排以及论文协议的精确检索均使用相同的
  归一化快速路径；
- `PAPER_SUMMARY` 写入 `implementation=normalized-f64-o2`，防止 `-Resume` 复用旧
  O0 性能结果；
- 每次查询继续与冻结的 Python Ground Truth 对照，不一致立即终止。

同一台机器、Caltech 7,766 条训练向量、3 个固定查询、11 个 beta 下，HDMG 平均延迟
由 `76.50 ms/query` 降至 `1.25 ms/query`，约加速 `61.4x`；构图时间由约 `77.6 s`
降至约 `0.91 s`，约加速 `85x`，所有 Recall 行与优化前一致。

进一步使用正式 200 个查询验证，11 个 beta 的 HDMG 延迟为 `0.90–1.48 ms/query`，
平均 `1.29 ms/query`；最低 Mixed Recall@3 为 `0.9983`。这表明当前仓颉实现已经达到
Python 论文约 2 ms/query 的同一性能量级。Float32 连续存储仍可将向量内存约减半，
但当前不再是查询延迟的首要瓶颈，后续应作为独立内存优化实验并重新验证排序稳定性。
