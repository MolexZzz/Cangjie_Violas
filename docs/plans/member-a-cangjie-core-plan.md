# 科研实践个人工作计划（一）：仓颉基础/Mixed 检索与文本数据实验

| 项目 | 内容 |
|---|---|
| 学生 | 待填写 |
| 实践周期 | 2026 年 7 月 21 日—7 月 30 日 |
| 进展汇报 | 7 月 21 日、7 月 23 日、7 月 28 日、7 月 30 日下班前 |
| 主要代码范围 | 仓颉版基础检索、自动聚类、representative、mixed search、索引生命周期 |
| 主要数据范围 | news20、ohsumed、yahoo |

## 一、研究背景与工作定位

本次科研实践以开源 Python 版 Violas 为参考实现，完善仓颉版本的正确性、性能和可开发性。
Python 代码保持冻结，只用于阅读、生成参考输出和运行对照实验，不通过修改 Python 版本来
迁就仓颉实现。

当前仓颉版本已经能够构建并在小规模预计算数据上运行，但初步 review 发现，实验主路径仍
存在影响论文结果复现的问题，例如自动聚类策略与 Python 版本不一致、部分索引接口为空、
检索参数硬编码、索引生命周期不明确，以及缺少逐 query 回归测试。现有 news20、ohsumed、
yahoo 文件分别只有 400、460、200 条向量，主要适合作为回归样例，不能在未验证来源和完整性
的情况下直接称为论文全量数据。

本人的工作不是单独完成“代码阶段”后再交给另一名成员实验，而是独立负责一条完整闭环：

> 基础/Mixed 检索 review → 仓颉代码修改 → 文本数据准备 → Python/仓颉/Faiss 对比 → 准确率与性能调优。

## 二、工作目标

1. 系统 review 仓颉基础检索、自动聚类、representative 和 mixed search 的实验主路径，形成有证据的 P0/P1/P2 问题清单。
2. 修复会影响实验正确性、全量运行和后续扩展的仓颉实现，并建立可重复的回归测试。
3. 审计并准备 news20、ohsumed、yahoo 三个文本数据集的 sample/full 实验 artifact。
4. 在相同输入、query、ground truth 和参数下，对比冻结 Python 版本、修改前后仓颉版本和 Faiss 基线。
5. 评估 Float32 数据通路和 rep/single 真实索引方案，解释 Violas mixed rerank 与标准向量检索的差异。
6. 通过参数扫描和 profiling 完成至少一项准确率改进和一项有 before/after 证据的性能优化。

## 三、现状基线与统计口径

当前初步统计结果如下：

| 范围 | 物理行 | 非空行 |
|---|---:|---:|
| Python Violas 核心 | 3,519 | 2,960 |
| 仓颉 `storage`（当前 6 个文件） | 2,118 | 1,879 |
| 全部仓颉 `src`（当前 13 个文件） | 4,172 | 3,750 |

正式报告将固定统计命令、Git commit、文件后缀和排除规则。代码行数只用于说明工程规模和
实现范围，不作为代码质量或性能优劣的直接证据。Faiss 是成熟的底层向量索引库，真正有意义
的比较是相同数据上的 Recall、NDCG、延迟、构建时间和资源使用。

本人负责解释 Python/仓颉基础与 mixed 核心的代码量和功能范围，并负责文本侧 Faiss
Flat/IVF/HNSW 实验。Faiss 整体源码规模只作背景材料，最终与另一名成员共同复核统计口径。

### 3.1 截至 7 月 21 日的工作基础

- 已完成 KMeans 聚类数量、KMeans++、MT19937、`random_state=42`、`n_init=10` 和 Lloyd 语义的首轮对齐；
- 已完成维度错误、非法参数、稳定排序和 tie-break 的正确性处理；
- 已完成 rep/single/HDMG 的数据版本与失效状态管理；
- 已拆出 `clustering.cj` 和 `mixed_scoring.cj`，并建立机器可读跨语言回归入口；
- 六个 sample 的少量 query 已完成对照；Yahoo 扩展到 20 query 后仍有少量边界差异；
- 已建立 Faiss Flat/IVF/HNSW 基线脚本，但当前 sample 太小，尚不能形成性能结论。

以上内容是本计划后续复核和实验的起点，不代表 full 数据、正式性能优化或论文结果已经完成。

## 四、具体工作内容

### 4.1 Python—仓颉行为对照与代码 Review

围绕论文实验主路径建立逐项对照，不追求一次性补齐全部 Python API：

- 插入、分组、base key 和稳定记录标识；
- representative 的生成与更新；
- 自动聚类数量、初始化、随机种子和确定性；
- key vector、semantic distance、beta 和 mixed score；
- candidate pool、rerank、top-k 和排序稳定性；
- 空 group、重复 key、维度错误和 `topK` 越界；
- build、search、数据修改、rebuild 和索引失效；
- 硬编码参数、重复逻辑、错误处理和可观测性。

问题按以下优先级记录：

- P0：会改变论文实验结果或导致结果不可复现；
- P1：会阻塞 full 数据运行、性能调优或后端替换；
- P2：长期 API 完整性、风格和非实验功能改进。

本周期优先处理 P0 和关键 P1，不为了架构形式进行大规模无关重写。

### 4.2 仓颉代码修改

本轮已完成前七项的首轮实现，后续由本人复核并在 full 数据下继续验证：

1. 对齐 Python 的 `n^(1-alpha)` 聚类数量规则；
2. 替换当前顺序切块式聚类，提供可设置 seed 的确定性 KMeans 路径；
3. 对齐 representative、key vector、base key 和 mixed score 的计算语义；
4. 对齐 candidate generation 与最终 rerank，统一距离“越小越相似”的约定；
5. 处理 `buildRepIndex/buildSingleIndex` 空实现，避免占位接口静默报告成功；
6. 引入 `dataVersion/builtFromVersion` 或等价机制，检测陈旧索引；
7. 使用有类型配置承载 alpha、beta、distance、topK 和 candidate pool；
8. 为构建和查询返回必要统计，便于定位准确率与性能问题；
9. 将聚类和 mixed 检索迁入清晰模块，降低继续扩展索引时的耦合；
10. 评估 Float64→Float32 对准确率、内存和 Faiss 对比口径的影响；
11. 为 rep/single 实现真实 ANN，或在 API 中继续明确其 exact-scan 语义。

### 4.3 回归测试与参考输出

- 建立包含多个 key、多个 group、重复和边界输入的小型 fixture；
- 使用冻结 Python 版本一次性生成 expected JSON；
- 分层测试 distance、cluster count、cluster assignment、representative、mixed score、top-k 和 index invalidation；
- 对每个 query 保存候选、embedding distance、semantic distance、mixed score 和最终排序；
- 对浮点结果设置明确容差，不以“肉眼看起来接近”作为验收；
- 保证测试可在不依赖另一名成员代码和 full 数据的情况下运行。

### 4.4 文本数据准备

本人独立负责 news20、ohsumed、yahoo：

- 核对当前文件的向量数、维度、类别数和每类采样量；
- 查明 full 数据来源、许可、预期规模和原始划分；
- 明确 embedding/key-vector 的模型、版本、维度和归一化方式；
- 生成稳定 train records、queries、key vectors 和 ground truth；
- 使用稳定 `recordId`，不使用数组下标、临时 UUID 或数据库内部 ID；
- 输出 manifest、数据 hash、随机 seed 和转换命令；
- 建立 smoke、regression、full 三档配置；
- 原始数据和大规模生成文件不提交 Git。

若周期内无法合法获得某个 full 数据集，将明确报告其状态、缺失原因和复现步骤，不能把当前
小样本标记为 full。

### 4.5 文本实验与 Faiss 对比

每个数据集至少比较：

| 方法 | 作用 |
|---|---|
| 冻结 Python Violas | 论文实现参考 |
| 修改前仓颉 Violas | before 基线 |
| 修改后仓颉 representative/mixed | 本人主要结果 |
| Faiss IndexFlat | 精确检索/ground-truth 校验 |
| Faiss IVF | ANN 性能基线 |

所有方法使用相同 records、queries、GT、距离、topK 和 seed。Exact 与 approximate 结果分开
汇报，不把不同召回条件下的延迟直接比较。Violas 的 key-vector/mixed 语义与 Faiss 的纯
embedding 检索存在差异，报告中将同时给出流程差异和统一指标，不用单一延迟下结论。

外部数据库由成员 B 统一负责，成员 A 不重复维护 Qdrant 接入；A 只使用双方冻结的 Result JSON
读取数据库结果，并检查其 query、GT、topK 和距离口径是否与文本侧 Faiss 实验一致。

### 4.6 指标与调优

准确率指标：Recall@K、NDCG@K、逐 query overlap。  
性能指标：build time、mean/P50/P95 latency；条件允许时记录峰值内存和索引大小。

主要参数包括：

- alpha、cluster count、KMeans seed；
- beta、candidate pool、topK；
- Faiss nlist、nprobe 等 IVF 参数；
- Float32/Float64、索引类型和线程等运行参数。

profiling 重点检查重复距离计算、全量排序、聚类开销和 Float64 内存成本。每项优化都必须绑定
commit、config、dataset hash 和 before/after，不只保留截图。

## 五、时间安排与阶段成果

| 日期 | 计划工作 | 下班前提交/汇报 |
|---|---|---|
| **7 月 21 日** | 完成现状 review；复核聚类/mixed、索引生命周期首轮修改；审计文本 sample | 本个人计划；已完成修改、P0/P1 遗留和当前 LOC 基线 |
| **7 月 22 日** | 扩大三个文本 sample 的逐 query 对照；定位 Yahoo 边界差异；复核 Faiss 命令 | 可运行命令、Result JSON、逐 query 差异 trace |
| **7 月 23 日** | 收敛聚类/mixed P0；补算法回归；提交文本 full 数据获取与转换方案 | “Review 修改 + 文本实验”闭环；明确未完成和阻塞项 |
| **7 月 24—27 日** | 准备 full artifact；评估 Float32/真实索引；运行 Faiss、profiling 和参数扫描 | 每项实验的 config/hash/commit；性能 before/after |
| **7 月 28 日** | 收敛主要正确性差异，至少跑通一个文本 full 数据 | Python/仓颉/Faiss 对照表；调优结果；剩余风险 |
| **7 月 29 日** | 合并公共代码后，在统一 commit 上重跑三个文本数据集 | 合并回归结果和最终可复现命令 |
| **7 月 30 日** | 整理文本部分最终结果、测试、已知限制和后续建议 | 文本实验包、个人总结和最终报告对应章节 |

## 六、预期产出

1. 基础/mixed 主路径差异与问题清单；
2. 仓颉基础检索、聚类、mixed search 和索引生命周期修改；
3. 分层回归测试、Python expected JSON 和逐 query trace；
4. 三个文本数据集的 manifest、artifact 生成/检查工具和实验配置；
5. Python、修改前后仓颉和 Faiss 的结果 JSON；
6. 参数扫描、性能 before/after 和已知限制说明；
7. 可由命令、commit、config 和 hash 重复验证的个人实验报告。

## 七、验收标准

- 仓颉项目持续可构建，核心测试可独立运行；
- `git diff -- violas_python` 为空；
- 三个文本 sample 均有 Python、修改前后仓颉和 Faiss 结果；
- 7 月 28 日前至少一个文本 full 数据集完成闭环；最终目标为三个文本 full 数据均可运行；确因来源、许可或资源无法完成的，必须提供客观证据且不得以 sample 替代；
- 主要准确率差异有逐 query 解释和明确阈值；
- 至少一项正确性修复和一项性能优化有测试及 before/after；
- 文本侧 Faiss Flat/IVF/HNSW 使用与仓颉相同的 records、queries 和 GT；
- 实验结果绑定统一 artifact、配置、commit 和环境；
- 不将 LOC、sample 数据或占位数据库结果表述为性能结论。

## 八、协作边界与风险控制

本人主要维护 cluster/mixed 模块、VectorMap 基础 facade、文本 runner、文本 manifest、文本
Faiss 配置和结果。另一名成员负责 HDMG、通用 benchmark、图像数据和三个数据库。双方冻结
公共 recordId、五折、指标和 Result JSON；公共文件由指定 owner 修改，另一方只提交 review 意见。

如果 HDMG 或图像工作延期，本人的基础/mixed 修改和文本实验仍可独立完成。如果某个文本
full 数据受阻，则继续另外两个文本集并保留完整阻塞证据。最终仅在合并 commit 上重跑时进行
一次集成回归，避免形成长期串行等待。

## 九、可选 CodeAgent 工作

仅在本计划必做项完成后，探索 repository/module/file/symbol/chunk 的 VectorGroup 层级、
key-vector 语义和 mixed 检索设计，为仓颉 CodeAgent 的代码组织与相关代码召回提供最小设计，
不挤占正确性和 full 实验收尾时间。
