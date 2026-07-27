# 基于 Violas 四种检索范式的代码项目上下文案例

日期：2026 年 7 月 27 日

## 摘要

本案例研究如何将 Violas 的实体对齐检索、组内多样性检索、关系扩展检索和跨模态配对
应用于代码项目上下文组织。实验没有把代码片段视为彼此独立的向量，而是将当前仓库中的
源码、测试、说明文档、运行脚本和实验结果组织为 4 个项目实体、24 个成员对象、多个实现
侧面及 17 条显式关系。四类检索共享同一个仓颉 `VectorMap` 状态。

实验使用 `all-MiniLM-L6-v2` 生成 384 维归一化向量，并为四种范式分别设置一个小型任务。
实体对齐任务中，扁平检索和 Violas 的实体纯度均为 100%；组内多样性任务的方面覆盖率由
60% 提高到 100%；关系扩展任务的依赖覆盖率由 40% 提高到 100%；跨模态配对任务的
Pair Hit@1 由 0 提高到 100%。

四项指标的定义不同，不能汇总为一个平均准确率。实验结果表明的不是 Violas 在普通代码
相似度检索上必然优于其他系统，而是同一份 vector-group 状态能够支持四种不同的
CodeAgent 上下文需求。其中，实体对齐在本次简单查询上没有产生增益，多样性、依赖关系和
跨模态配对则提供了扁平向量排序无法稳定表达的约束。

## 1. 研究背景

Violas 论文第 6.5 节给出了四种原生实体中心检索能力：

| 缩写 | 论文中的能力 | 代码项目中的对应问题 |
| --- | --- | --- |
| EAR | Entity-aligned Retrieval | 先定位负责问题的子系统，再检索其成员 |
| DDR | Diversity-driven Retrieval | 从同一子系统的不同实现侧面选取上下文 |
| RER | Relation-expanded Retrieval | 从命中对象沿调用、测试或产物关系扩展 |
| CMP | Cross-modal Pairing | 从源码对象取得与其配对的文档、脚本或结果 |

初版案例仅在向量召回后执行关系广度优先展开，实质上只覆盖了 RER。它可以验证
`addPairRelation` 和 `getPairedVectors` 的作用，但不能反映 vector group 对实体成员、
组内多样性和跨模态对象的统一组织能力。本版因此将研究问题改为：

> 同一份 Violas 项目状态能否针对 CodeAgent 的不同上下文需求，分别执行实体路由、
> 组内多样化选择、依赖链扩展和源码—项目材料配对？

## 2. 项目上下文建模

### 2.1 项目实体

实验将仓库材料组织为四个实体 key：

| 实体 key | 成员范围 | 主要侧面 |
| --- | --- | --- |
| `hdmg_index` | HDMG 构建、查询、配置、测试和参数结果 | maintenance、construction、traversal、configuration、validation、evaluation |
| `vector_group` | VectorMap 插入、关系、上下文及测试 | mutation、relation、context、validation、documentation |
| `benchmark_pipeline` | 仓颉协议、运行脚本、最终结果和清单 | execution、result、provenance、documentation |
| `mixed_ranking` | mixed score、查询入口、测试和实验说明 | scoring、retrieval、validation、evaluation |

每个成员记录以下信息：

- 稳定 ID、所属 key 和实现侧面；
- `code`、`test`、`document`、`script` 或 `artifact` 模态；
- 仓库路径和行号；
- 符号名、类型和人工核对的功能描述；
- 由相同模型生成的成员向量。

每个 key 另有一个实体表示向量，用于 EAR 的软实体对齐。当前小型案例以一个成员
`VectorGroup` 表示一个微簇；因此能够验证按侧面选择的查询流程，但不评价自动聚类质量。

### 2.2 显式关系

17 条关系均从当前源码和项目材料中人工核对，示例如下：

```text
updateObject --calls--> _invalidateIndexes
_invalidateIndexes --affects--> searchHdmg
searchHdmg --rebuilds_with--> buildHdmg
_invalidateIndexes --tested_by--> lifecycle test
searchHdmgWithConfig --evaluation_artifact--> hdmg-parameter-scan.md
run_image_full_suite.ps1 --produces--> final-results.json
```

关系只连接同一项目实体内的成员，以符合当前仓颉 `addPairRelation` 对 key 和 group
规模的约束。`code` 与 `document` 或 `artifact` 的关系同时作为 CMP 的配对信息。

### 2.3 同一状态上的四类查询

四种范式不是四套独立数据。仓颉程序只创建一个 `VectorMap`：

1. 24 个成员按照所属实体写入 4 个 key；
2. 4 个实体表示通过 `setKeyVectors` 写入；
3. 17 条类型化关系通过 `addPairRelation` 写入；
4. EAR、DDR、RER 和 CMP 在同一状态上选择不同的查询策略。

这项复用是本案例区别于“向量库加外部关系表”的主要设计点：实体入口、成员向量、
组内侧面和对象关系由同一个受管理对象提供。

## 3. 实现与实验协议

### 3.1 数据准备

`tools/run_code_context_case_study.py` 完成以下工作：

1. 核对 24 个实体在当前仓库中的路径和行号；
2. 编码成员描述、实体 key 描述和四个查询；
3. 输出 `ENTITY`、`KEY`、`REL` 和 `QUERY` 四类 TSV 记录；
4. 调用仓颉 `codecase` 入口；
5. 解析仓颉输出并保存机器可读结果。

生成的输入和原始日志分别位于
`artifacts/code-context-case-study/` 与
`results/code-context-case-study/`，二者均不纳入版本控制。

### 3.2 对照原则

实验保持以下条件一致：

- 所有方法使用同一批 MiniLM 成员向量；
- Violas 策略和基线从同一个 `VectorMap` 读取对象；
- 每项任务的 Violas 方法与对应基线使用相同返回预算；
- 只改变查询是否使用实体、侧面或关系信息。

四种范式解决的问题不同，因此分别使用适合该任务的指标，不计算跨范式平均值。

## 4. Case Study 结果

### 4.1 EAR：实体对齐检索

查询要求定位同时负责 HDMG 构建、遍历、索引失效、候选配置和测试的子系统。扁平基线直接
在 24 个成员上执行 cosine Top-5；Violas 使用查询向量同时作为实体条件，并以
`beta=0.75` 执行 mixed entity/member ranking。

指标为 `Entity Purity@5`，即前五个结果中属于目标实体 `hdmg_index` 的比例。

| 方法 | Top-5 | Entity Purity@5 |
| --- | --- | ---: |
| 扁平向量 | search config、search HDMG、configured search、parameter scan、build HDMG | 5/5，100% |
| Violas EAR | search config、search HDMG、configured search、parameter scan、build HDMG | 5/5，100% |

该查询的文本直接包含 HDMG、构建和配置等高区分度词汇，MiniLM 本身已经完成正确路由，
因此 EAR 没有带来数值提升。这个持平结果仍有意义：实体路由在此处没有破坏成员相关性；
但若要证明 EAR 的普遍收益，需要增加包含跨模块同名函数或语义混淆对象的独立查询集。

### 4.2 DDR：组内多样性检索

任务要求为 HDMG 生成一个包含 maintenance、construction、traversal、configuration 和
validation 五个侧面的上下文包，返回预算为 5。

扁平基线在 `hdmg_index` 内直接取 Top-5。Violas DDR 先进入目标实体，再按照查询指定的
侧面从不同 `VectorGroup` 选择成员。

| 方法 | 返回对象 | Aspect Coverage@5 |
| --- | --- | ---: |
| 扁平向量 | search config、parameter scan、configured search、search HDMG、config test | 3/5，60% |
| Violas DDR | search config、search HDMG、config test、build HDMG、invalidate indexes | 5/5，100% |

扁平结果包含两个 configuration 成员，并选入不属于目标五方面的 evaluation 文档，因此遗漏
maintenance 和 construction。DDR 在不增加上下文预算的情况下覆盖五个所需侧面。这对应
CodeAgent 的“先建立模块全貌”任务：目标不是找到五个最相似片段，而是获得配置、执行、
维护和验证之间相对完整的切面。

### 4.3 RER：关系扩展检索

任务从 `_invalidateIndexes` 出发，恢复对象更新、索引失效、查询重建及生命周期测试组成的
依赖链。返回预算为 5。

| 方法 | 返回对象 | Dependency Coverage@5 |
| --- | --- | ---: |
| 扁平向量 | lifecycle test、config test、parameter scan、search HDMG、configured search | 2/5，40% |
| Violas RER | invalidate、update、search、lifecycle test、build | 5/5，100% |

扁平检索能够找到语义相近的测试和查询函数，但混入配置测试与参数扫描。RER 通过
`calls`、`affects`、`tested_by` 和 `rebuilds_with` 关系补齐完整链路。与初版案例相比，
这里的 RER 不再被当作 Violas 的全部能力，而只是统一查询状态中的一种策略。

### 4.4 CMP：跨模态配对

任务给定源码对象 `searchHdmgWithConfig`，要求取得与其配对的自然语言评估材料。这里的
“模态”指代码与项目文档/结果，而不是图像与文本。

基线使用该源码对象的向量检索最相似的另一个成员；Violas CMP 按
`evaluation_artifact` 关系取得配对对象。

| 方法 | Top-1 | Pair Hit@1 |
| --- | --- | ---: |
| 扁平近邻 | `searchHdmg` | 0/1，0% |
| Violas CMP | `hdmg-parameter-scan.md` | 1/1，100% |

向量近邻返回了语义相似的查询实现，但“语义相似”并不等于“由该接口产生或评价的材料”。
CMP 将这种确定性配对作为项目状态的一部分保存，适合 CodeAgent 在修改实现后同步定位
对应文档、测试、配置样例或实验结果。

## 5. 汇总与解释

| 范式 | 指标 | 基线 | Violas | 结果 |
| --- | --- | ---: | ---: | --- |
| EAR | Entity Purity@5 | 100% | 100% | 持平 |
| DDR | Aspect Coverage@5 | 60% | 100% | 提升 40 个百分点 |
| RER | Dependency Coverage@5 | 40% | 100% | 提升 60 个百分点 |
| CMP | Pair Hit@1 | 0% | 100% | 命中配对对象 |

四项结果共同说明：

1. 普通相似度查询已经足够明确时，实体对齐可能不会产生额外收益；
2. CodeAgent 需要模块全貌时，侧面约束能减少相似结果重复占用上下文；
3. 修改影响分析需要显式依赖，不能仅依赖语义相似；
4. 源码与说明、脚本或实验结果之间的确定性对应关系适合通过配对检索表达；
5. 四种行为可以复用同一个 vector-group 状态，不需要为每类任务建立独立索引和外部拼接流程。

## 6. 对 CodeAgent 的意义

本案例更适合作为 CodeAgent 的上下文规划层，而不是代替 Agent 自带的文件搜索工具。
Agent 可以根据任务类型选择策略：

| Agent 意图 | 建议策略 |
| --- | --- |
| 不确定问题属于哪个模块 | EAR |
| 首次阅读模块或准备重构 | EAR + DDR |
| 修改函数并检查影响范围 | EAR + RER |
| 修改实现后同步文档、测试或结果 | RER + CMP |

例如，“调整 HDMG 候选池并更新实验说明”可以先由 EAR 路由到 `hdmg_index`，再由 DDR
提供配置、遍历、验证和评估侧面，最后由 CMP 从 `searchHdmgWithConfig` 取得参数扫描文档。
这比一次全局 Top-K 更接近 CodeAgent 对结构化上下文的实际需求。

## 7. 限制

本实验仍属于探索性案例，不能解释为通用 benchmark：

- 只有 4 个项目实体、24 个成员和 4 个查询；
- 实体划分、侧面、关系及任务真值均经过人工整理；
- 查询与策略在同一开发过程中形成，没有独立留出测试集；
- 每个 `VectorGroup` 在本案例中只有一个成员，未评价自动微簇划分质量；
- CMP 将代码和项目文档视为两种模态，没有涉及图像—文本编码；
- EAR 查询区分度较高，尚未构成有挑战性的实体消歧实验；
- 当前关系受仓颉接口限制，只能连接同一 key 下的等规模 group；
- 没有直接比较真实 CodeAgent 的补丁成功率、工具调用次数或 token 消耗；
- 小规模数据使用精确 `VectorMap.search`，不评价 HDMG 的规模性能。

因此，本案例能够证明四种检索流程在仓颉实现中可运行，并展示它们对代码上下文的不同作用；
尚不能证明 Violas 会普遍提高 Codex 或 Claude Code 的任务成功率。

## 8. 复现

```powershell
python -m pip install -r tools\requirements-benchmark.txt
python tools\run_code_context_case_study.py
```

脚本生成 TSV 后调用仓颉入口：

```text
codecase ../artifacts/code-context-case-study/case-study.tsv
```

机器可读的冻结结果见
[`results-summary/code-context-case-study.json`](../../results-summary/code-context-case-study.json)。
