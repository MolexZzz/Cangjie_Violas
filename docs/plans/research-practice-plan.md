# Violas 两周科研实践计划（2026-07-20 ～ 2026-07-30）

详细现状证据、接口差异、数据集完整性和横向数据库审计见
[`interface-data-benchmark-audit.md`](../reports/interface-data-benchmark-audit.md)。本计划中的分工与验收应以该审计结论为准。

> **修改边界：** `violas_python/` 是冻结的参考版本，本实践不修改 Python 核心、benchmark 或依赖文件。所有功能补齐和优化均在 `cj_core/` 完成；新增数据/实验工具只能放在独立目录，并以只读方式调用 Python 生成参考结果。

## 1. 目标与范围

本次实践由两人协作，按以下优先级推进：

1. **必须完成：代码 review 与仓颉核心改造。** 以 Python 冻结版本为规范，梳理接口差异，在仓颉侧补齐功能、修复不一致并增强扩展性，建立仓颉自动化测试和可复现的代码量统计。
2. **必须完成：全量 benchmark。** 建立统一、非交互、可恢复的评测流程，在全量数据上记录准确率、性能和资源指标，并与精确检索及 Faiss 基线对比。
3. **可选：CodeAgent 场景验证。** 只有前两项达到验收门槛后，才尝试用仓颉版 Violas 组织代码项目，形成设计说明或小型 benchmark。

本阶段不修改 Python，也不追求逐行翻译；目标是在仓颉侧按优先级实现 Python 的核心行为，并在兼容默认结果的前提下改善类型、封装和性能。

## 2. 当前基线（截至 2026-07-21 首轮整改）

### 2.1 可复现代码行数

统计规则：仅统计 Git 已跟踪的实现文件；Python/仓颉分别统计 `.py`、`.cj`，脚本统计 `.sh`；同时给出物理行和非空行；排除数据、文档、图片、构建目录、缓存和第三方依赖。后续汇报必须记录统计命令、Git commit 和日期。

| 范围 | 文件数 | 物理行 | 非空行 |
|---|---:|---:|---:|
| Python 核心包 `violas_python/violas` | 10 | 3,519 | 2,960 |
| Python benchmarks | 26 | 12,394 | 10,968 |
| Python 全部实现（含脚本/示例） | 42 | 16,070 | 14,059 |
| 仓颉存储核心 `cj_core/src/storage` | 6 | 2,118 | 1,879 |
| 仓颉 benchmark | 4 | 1,811 | 1,655 |
| 仓颉全部实现 | 13 | 4,172 | 3,750 |
| Violas 当前全部实现 | 55 | 20,242 | 17,809 |

Faiss 对照使用官方仓库 `facebookresearch/faiss` 的 commit `7d4bb39f7eb3e9bb4d160aa38ec821ee1a407afc`（提交日期 2026-07-17），采用相同的“物理行/非空行”规则，源文件扩展名为 `.h/.hpp/.c/.cc/.cpp/.cuh/.cu/.py`：

| Faiss 范围 | 文件数 | 物理行 | 非空行 |
|---|---:|---:|---:|
| CPU library | 423 | 130,438 | 112,448 |
| GPU library | 209 | 52,257 | 44,116 |
| `faiss/` 全部库代码 | 632 | 182,695 | 156,564 |
| benchmarks | 73 | 17,230 | 14,499 |
| tests | 124 | 47,245 | 38,481 |

仓颉全部实现的非空行约为 Faiss CPU library 的 **3.33%**，Violas 全部实现约为 **15.84%**。该比例只能说明工程规模，不能说明功能或质量等价。更有意义的差距是：Faiss 将索引、距离、训练、量化、CPU/GPU 后端和测试体系分层；当前仓颉版虽已拆出聚类、mixed scoring、HDMG 配置和 benchmark backend 边界，但 rep/single 仍是 exact scan，HDMG graph state 仍由 `VectorMap` 持有。

### 2.2 数据规模现状

当前 `dataset/precomputed` 共约 106 MB、7,100 条预计算向量：

| 数据集 | 类别/文件夹数 | 向量数 | 文件大小 |
|---|---:|---:|---:|
| 20 Newsgroups | 20 | 400 | 5.04 MB |
| OHSUMED | 23 | 460 | 6.66 MB |
| Yahoo Answers | 10 | 200 | 1.25 MB |
| Caltech-101 | 101 | 2,020 | 30.90 MB |
| CUB-200 | 200 | 4,000 | 61.15 MB |
| COCO | 11 | 20 | 1.39 MB |

这些文件可以作为中等规模回归集，但不代表六个原始数据集的全量规模。Python benchmark 默认通常只测试 200 个 query，Yahoo 默认 `sample_ratio=0.01`。因此“全量跑通”必须同时满足：原始数据全量输入、无采样预处理、全 query 评测，而不能只在仓颉菜单中选择 `full`。

### 2.3 基础健康状态

- `cjpm build` 当前成功。
- 已建立可执行核心回归和冻结 Python—仓颉对照；尚未迁入正式 `cjpm test` 目录。
- benchmark 已支持非交互命令、Result JSON、Recall/NDCG 和 mean/P50/P95；峰值内存、索引大小与五折数据库汇总尚未完成。
- 六套 Python benchmark 存在大量近似重复代码，但本次不修改；仓颉 benchmark 设计时不得复制同样结构。

## 3. Python 与仓颉接口差异

### 3.1 接口对齐矩阵

| 能力 | Python | 仓颉 | 结论/计划 |
|---|---|---|---|
| `VectorGroup` 构造、追加、统计 | 有 | 有 | 仓颉对齐 Python 的正常输入行为、维度校验和 ID 规则 |
| 基础 `insert/get/getAllKeys` | 有 | 有 | 基本对齐；内部状态不应全部公开可变 |
| 单向量、代表向量、组检索 | 统一 `search` 支持 `single/representative/group1/groupN` | `search` + `searchMulti` | 行为部分对齐；需要类型化配置和契约测试 |
| key 过滤 | 单个 key 或 key 列表 | 单个 `Option<String>` | 仓颉缺少多 key/candidate scope |
| 距离函数 | 字符串选择，支持 cosine/euclidean/manhattan | 字符串选择，同三种 | 数值行为接近，但字符串分支难扩展，应抽象 `DistanceMetric` |
| 自动聚类 | `n^(1-alpha)` + KMeans | 已实现 MT19937、KMeans++、`n_init=10` 和 Lloyd 首轮兼容 | sample 基本对齐；仍需量化 Yahoo/full 边界差异 |
| Flat 索引 | Python 可用 Faiss `IndexFlatIP` | rep/single 已有版本状态，但仍为 exact scan | 状态不再为空；真实 ANN 仍是 P1 |
| HDMG 构建/检索 | 有，含矩阵预计算和更多限流参数 | 已去除完整 G×G 距离矩阵并增加 trace | 内存风险降低，O(G²) 计算和 full profiling 仍待处理 |
| `create_group/create_cluster` | 有 | 无 | 补齐稳定的高层创建接口 |
| 对象插入、更新、迁移、删除 | `insert_object/update/assign/delete` | 无 | P1 核心契约，关系检索和 CodeAgent 场景的前提 |
| `VectorRef/VectorRelation` | 有 | 无 | P1 补齐类型化引用与关系模型 |
| 对象级关系增删 | 有 | 仅组级 pair/tree | P1 补齐；不要继续塞入任意 `HashMap<String, Any>` |
| entity/diverse/dependency/modal API | 有公开 wrapper | 无 | 先定义语义和验收用例，再补最小集合 |
| context/relation/multimodal search | 有 | 仅部分 context/pair 查询 | 按实际 benchmark 需求分批补齐 |
| 统计与诊断 | 有 | 部分有 | 统一结构化统计结果，移除核心库直接 `println` |
| 序列化/持久化 | 没有稳定统一契约 | 无 | 本周期只设计接口，不承诺完整实现 |

### 3.2 已发现问题及当前处理状态

1. Python `VectorGroup.group_id` 使用 32 个十六进制字符的 MD5 文本，仓颉使用 16 字符的本地哈希；仓颉应改为 Python 的 ID 规则。
2. Python HDMG fallback 中存在历史拼写 `gruop_expansion_factor`；只记录该 quirk，不修改 Python，仓颉保留正确命名并对齐计算行为。
3. `addVector` 的索引生命周期问题已通过 `dataVersion/builtFromVersion` 首轮修复；公开容器绕过 API 修改时仍需调用 `markDirty()`。
4. `buildRepIndex()` 和 `buildSingleIndex()` 已记录 exact-scan snapshot 状态，但尚未实现真实 ANN。
5. 顺序切块聚类已替换为兼容 KMeans；Yahoo 少量 query 仍有浮点/聚类边界差异，需要 full 对照继续验证。
6. 仓颉大量使用 `String` 表示 mode/metric/type，并以 `-1`、`false`、打印 warning 等多种方式表示错误，调用方难以可靠处理失败。
7. 仓颉 `VectorMap` 内部数据结构公开可变，外部修改可绕过维度检查和索引失效逻辑。
8. 已建立固定 fixture 和六 sample 的逐 query 对照；覆盖量仍不足以证明所有 sample/full query 完全一致。

## 4. 灵活性、可开发性与性能问题

### 4.1 结构问题

- Python `vectormap.py` 仅作为只读功能参照；仓颉 `vectormap.cj` 约 1,376 行，存储、关系、索引、HDMG、检索、统计和打印混在同一类，需要在仓颉侧拆分。
- Python 六套 benchmark 的重复结构不在本次修改范围；仓颉侧应从一开始使用共享 dataset/backend/reporter，避免复制。
- 仓颉 benchmark 硬编码数据路径和参数，入口依赖交互式 `readln()`，不适合自动运行、批处理和 CI。
- `HashMap<String, Any>` 承担元数据、关系和结果协议，编译期无法发现字段拼写或类型错误。

### 4.2 全量数据瓶颈

- 仓颉向量全部使用 `Float64`；常见向量检索和 Faiss 基线使用 `float32`，当前表示会增加内存和带宽压力。
- HDMG 已去除完整 group-to-group 距离矩阵，但构图计算量仍为 O(G²)。全量微簇数量增长后仍可能成为首要瓶颈。
- Flat search 目前收集全部候选再全局排序，可改为大小为 K 的有界堆，避免 O(N log N) 全排序。
- TXT 虽然流式读取，但最终仍把全部 `Array<Float64>` 对象保存在内存中；文本解析和对象开销不适合作为长期全量格式。
- 11 个 beta 值 × 5 folds 会重复构建相同的 fold 状态和 HDMG；应复用与 beta 无关的索引和数据切分。

### 4.3 benchmark 可信度缺口

- 需要冻结 ground truth 定义，避免不同 beta 或不同实现使用不同的 recall 语义。
- 当前已报告 build time 和 mean/P50/P95；仍缺统一 warm-up、重复次数、QPS、峰值内存和索引大小。
- 缺少统一记录：dataset 版本、样本数、维度、随机种子、硬件、commit、配置和失败原因。
- Faiss 应作为同一向量、同一 query、同一 metric 下的基线，而不是只比较代码行数。至少包括 `IndexFlatIP` 精确基线；若做近似检索，再加入一个与目标 recall 匹配的 Faiss 索引。

## 5. 修改计划与优先级

### P0：先保证可验证（7/20～7/23）

1. 只读运行 Python 生成 frozen golden fixtures：距离、插入、过滤、top-k、重复组等 expected output。
2. 新建仓颉自动化测试入口，验证仓颉结果与 frozen snapshot 对齐。
3. 在仓颉侧修复 group ID、索引失效和自动聚类语义等不一致；不修改 Python。
4. 把仓颉 benchmark 改为可传参数的非交互入口，保留交互模式作为外层 UI。
5. 输出结构化 JSON，包括配置、数据规模、指标、耗时和资源信息。

### P1：建立可扩展核心（7/22～7/28）

1. 定义并逐步引入 `DistanceMetric`、`ClusterStrategy`、`IndexBackend`、`BenchmarkReporter` 四个边界。
2. 将 `VectorMap` 拆分为存储/对象生命周期、索引/HDMG、查询/关系和统计模块；外部只通过受控方法修改状态。
3. 补齐仓颉最小对象 API：`createGroup`、`insertObject`、`updateObject`、`assignObject`、`deleteObject`、`VectorRef`。
4. 先实现一个明确命名的 Exact backend，再实现有实际加速效果的索引；空的 `build*Index` 不得继续作为“已完成”接口。
5. 全量性能改造优先级：复用 fold/图构建 → top-k 有界堆 → Float32/紧凑向量容器 → 避免 O(G²) 全矩阵。

### P2：全量评测与调优（7/24～7/30）

1. 采用三级规模阶梯：smoke（20/class）→ 当前 precomputed 回归集 → 原始全量数据。
2. 独立数据工具为六个数据集生成 manifest，不修改 Python 参考目录。
3. 每项实验至少比较：冻结的 Python Violas 基线、仓颉 Violas、Exact/Faiss，以及仓颉 benchmark 侧的 Milvus/Qdrant/Chroma；配置完全一致。
4. 指标：Recall@K、NDCG@K、p50/p95 latency、QPS、build time、peak RSS、index size；准确率调优与性能调优分开记录。
5. 优化必须有 before/after 数据，不接受只凭代码观感判断“更快”。

### P3：CodeAgent 可选探索（仅在门槛通过后）

进入条件：P0 核心测试通过；六个全量数据 pipeline 均可重复运行；主要指标和瓶颈有结论；没有阻断级正确性问题。

可选产出二选一：

- 设计稿：以 repository/module/file/symbol/chunk 为 VectorGroup 层级，定义依赖、调用、同名符号和历史修改关系。
- 小型 benchmark：在 2～3 个仓颉项目上评测 symbol lookup、跨文件依赖扩展、相关代码召回和 latency。

## 6. 两人分工与四次进展汇报

两名成员都不修改 `violas_python/`。两人分别负责一条同时覆盖目标（1）代码 review/修改
和目标（2）全量实验/Faiss 对比的垂直任务：

- 成员 A：基础检索、聚类、representative、mixed search；news20、ohsumed、yahoo 和文本 Faiss；
- 成员 B：HDMG、benchmark runner/指标/配置；caltech、cub、coco 及 Milvus/Qdrant/Chroma；
- 两人各自在负责的数据集上运行 Python、修改前后仓颉和 Faiss，并分别完成准确率/性能调优。

完整边界见 [`two-person-parallel-work-plan.md`](two-person-parallel-work-plan.md)。

| 日期 | 成员 A | 成员 B | 下班前汇报内容/验收物 |
|---|---|---|---|
| **7/21 周二** | 提交基础/mixed 计划并复核首轮修改、文本 sample | 提交 HDMG/benchmark 计划并复核首轮修改、数据库框架 | 已完成修改、P0/P1 遗留和公共实验契约 |
| **7/23 周四** | mixed/聚类测试、首批修复和文本 Python/仓颉/Faiss 基线 | HDMG/runner 测试、首批修复和图像 Python/仓颉/Faiss 基线 | 两人分别展示代码修改和对应实验闭环 |
| **7/28 周二** | 文本 full 状态、对齐结果和性能 before/after | 图像 full 状态、HDMG 对齐结果和性能 before/after | 每人至少跑通一个 full 数据集，两个分支达到 merge-ready |
| **7/30 周四** | 在合并 commit 上重跑三个文本集并提交结果 | 在同一 commit 上重跑三个图像集并提交结果 | 汇总六数据集、Faiss/数据库结果；满足门槛后再附 CodeAgent 设计/POC |

### 每日协作节奏

- 上午 10 分钟同步：昨天证据、今天目标、阻塞项。
- 下午只做只读交叉 review；修改意见由文件 owner 落地，避免两人同时编辑同一模块。
- 每次实验结果必须绑定 commit 和配置文件；禁止手工修改参数后只保留截图。
- 主分支保持可构建；大改动按“测试/契约 → 实现 → benchmark”拆分提交。

## 7. 验收标准

### 阶段（1）完成条件

- LOC 统计可一条命令复现，并固定 Faiss commit 与统计口径。
- 核心 API 有明确的 Python/仓颉映射和暂不支持列表。
- P0 行为有仓颉自动化测试；仓颉可构建并匹配冻结的 Python expected outputs。
- `git diff -- violas_python` 为空。
- 已知同名不同语义、陈旧索引、ID 不一致等问题已修复或显式禁用。
- 核心模块不再通过空索引接口或终端打印伪装成功。

### 阶段（2）完成条件

- 六个原始全量数据集均有 manifest，且无默认采样和 query 上限。
- 同一配置下得到 Exact、Faiss、Python、仓颉四组可比较结果。
- 至少报告 Recall@K、NDCG@K、p50/p95、QPS、build time、peak RSS、index size。
- 结果可由他人在相同环境用非交互命令复现；失败/OOM 也必须记录规模、位置和原因。
- 至少完成一轮有量化证据的性能优化和一轮准确率参数调优。

## 8. 风险与止损

- **数据/算力风险：** 7/21 前确认原始数据可用性、磁盘、内存、GPU 和最长运行窗口；缺失数据必须立即上报，不能等到 7/28。
- **O(G²) 风险：** 若 HDMG 在规模阶梯中出现明显二次增长，暂停六数据集盲跑，先改构建策略。
- **范围风险：** 优先对齐 Python 的核心能力；对象关系 API 只实现 benchmark/CodeAgent 必需闭环，序列化和 GPU 后端延后。
- **指标风险：** 先冻结 ground truth 和数据切分，再调参数；不得为了提高数字临时更换 recall 定义。
- **进度风险：** 7/28 若阶段（2）仍未形成全量闭环，自动取消阶段（3），两人共同处理数据与性能阻塞。

## 9. 当前提交与后续提交建议

当前提交包含首轮 Review 修复、跨语言测试、实验工具、数据库框架和计划文档，建议 commit message：

```text
feat: align core algorithms and add benchmark framework
```

后续提交按以下边界拆分：

1. `test: add cross-language storage contract fixtures`
2. `fix: align core storage semantics and index invalidation`
3. `refactor: introduce configurable benchmark runner`
4. `perf: optimize HDMG build and top-k search`
5. `bench: add full-dataset and Faiss comparison results`
