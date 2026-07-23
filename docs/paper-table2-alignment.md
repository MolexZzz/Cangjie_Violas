# 论文 Table 2 对齐说明

## 对齐原则

本项目以两类材料作为复现依据：

1. 检索算法和默认运行参数以仓库中的开源 Python 实现为可执行参考；
2. Table 2 的方法定义和 NDCG 计算方式以论文正文为准。

论文正文、伪代码和开源 Python 在个别实现细节上并不完全一致，因此不能同时保证“逐行复刻伪代码”和“逐查询复刻 Python”。当前实现优先保证仓颉 Violas 与开源 Python 的实际执行逻辑一致，同时按照论文公式生成主表指标。任何扩展实验都与论文主表分开保存。

## 当前统一配置

- 数据划分：每个类别 90% 建库、10% 测试，随机种子 42；
- 查询：使用每个数据集完整的 10% 测试池；当前规模应为 Caltech 911、CUB 1191、COCO 1033；
- 图片及类别文本向量：CLIP ViT-B/32，512 维；
- 返回数量：`topK=3`；
- 混合权重：β 从 0.0 到 1.0，步长 0.1；
- 微簇参数：`alpha=0.5`；
- HDMG：`embeddingK=16`、`semanticIntraK=8`、`semanticBridgeKeys=4`、`semanticBridgePerKey=2`、`topKeyCandidates=5`、`clusterPoolSize=9`、`extraHops=0`；
- 所有方法读取同一份冻结 artifact，不重新划分数据或重新生成 ground truth。

## Table 2 中各方法的实现

### Violas

使用仓颉 HDMG 图索引搜索微簇，再对候选微簇内的图片按 mixed score 重排并返回 Top-3。HDMG 的执行逻辑以开源 Python `search_hdmg` 为参考。

### w/o HDMG

保留实体类别信息和 mixed score，但不遍历 HDMG 图：先按类别文本向量选择 5 个实体组，再直接检查这些实体组内部的微簇，每组最多保留 9 个候选微簇，最后对其中图片按 mixed score 重排并返回 Top-3。

这与 `Representative-3` 不同。`Representative-3` 不先按实体类别路由，而是在全局给所有微簇代表向量计算 mixed score，只展开得分最好的 3 个微簇。该方法仅作为补充诊断，不进入论文 Table 2 主表。

使用同一冻结 artifact 调用开源 Python `search_with_representative_rerank` 后，β=0.5 的 Recall@3 分别为 Caltech `0.9617`、CUB `0.9700`、COCO `0.9300`。这证明 Caltech 和 COCO 先前看到的 `0.9617/0.9300` 是开源 Python 在当前冻结微簇上的实际结果，而不是仓颉实现偏差；不能把它们与另一实验定义下的 `1.0000` 直接比较。CUB 仓颉结果为 `0.9683`，相差一个 Top-3 命中，属于补充方法中的浮点边界/同分选择差异，后续可通过在 artifact 中冻结代表向量来消除，但不影响 Table 2 的五个主方法。

### Milvus、Qdrant、Chroma

论文主表中的数据库方法采用两阶段流程：Milvus、Qdrant、Chroma 先按图片 embedding 召回 `topK × 10 = 30` 个候选，Benchmark 再在本地计算 mixed score，重排后返回 Top-3。数据库自身没有执行 mixed search，但主表统计的是本地 mixed 重排后的结果。

数据库直接返回的 raw Top-3 仍作为辅助诊断数据保存，但只进入 raw 辅助表，不进入论文主表。

## 指标口径

- Recall@3：返回 Top-3 与当前 β 对应的精确 mixed Top-3 的重合比例；
- NDCG@3：按照论文公式，使用每个返回对象的 mixed relevance 作为分级相关性，增益为 `2^relevance - 1`；
- β=1.0 时，同一语义类别内的图片可能完全同分。此时结果正确性按类别 key 校验，NDCG 主表显示为 `—`，与论文一致；
- Violas 延迟只统计查询过程；
- 数据库论文列统计“30 候选数据库召回 + 本地 mixed 重排”的总延迟；raw Top-3 延迟只出现在辅助结果中。

## 协议与结果隔离

新结果统一标记为 `violas-paper-table2-v5`：

- 仓颉结果包含 `ndcgGain=mixed-score-graded` 和真实 `withoutHdmgMethod`；
- 数据库 `paperComparison.method=vector-candidate-plus-local-mixed-rerank`；
- raw Top-3 保存在 `rawComparison`，只用于辅助分析；
- `queryScope=full-10-percent-test-pool`；
- 汇总脚本拒绝读取旧协议结果和不完整的 200-query artifact。

因此，之前已经运行的 v4 及更早结果只能作为开发记录，不能直接用于最终论文对照，必须先重新生成包含完整 10% 查询池的 artifact，再用 v5 重新执行。

## 延迟复现限制

查询结果和准确率口径可以在当前环境对齐，但延迟数值还受硬件、操作系统、编译器和数据库部署方式影响，不能仅靠代码保证与论文数值完全相同。论文基线使用本地嵌入式/进程内数据库边界；当前工具提供：

- `paper-local`：Qdrant 内存模式、Chroma 临时模式、Milvus Lite，适合论文式延迟对照；
- `service`：连接 Docker 服务，适合验证准确率和真实服务部署，但网络/进程通信延迟不能与论文表中的本地延迟直接比较。

Milvus Lite 不支持当前原生 Windows 环境。要完成严格的三数据库本地延迟对照，需要在 Linux/WSL 环境运行 Milvus Lite；在 Windows Docker 模式得到的 Milvus 准确率仍有效，但延迟应明确标为 service-mode。
