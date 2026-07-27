# 第三阶段方案：基于 Violas 的代码项目上下文组织

日期：2026 年 7 月 28 日  
状态：四范式 MVP 已实施，结果见 [案例报告](reports/code-context-case-study.md)

## 1. 阶段目标

第三阶段不实现新的通用 CodeAgent，而是验证 Violas 能否作为 CodeAgent 的上下文规划层。
案例应体现论文中的四种实体中心检索能力，而不只验证关系数据结构：

1. Entity-aligned Retrieval（EAR）：定位问题所属项目实体；
2. Diversity-driven Retrieval（DDR）：选择同一实体内的不同实现侧面；
3. Relation-expanded Retrieval（RER）：补充调用、测试和产物依赖；
4. Cross-modal Pairing（CMP）：关联源码与文档、脚本或实验结果。

四类查询必须复用同一个仓颉 `VectorMap` 状态。

## 2. 数据组织

从当前仓库中人工核对 24 个项目成员，按功能组织为 4 个实体 key：

- `hdmg_index`
- `vector_group`
- `benchmark_pipeline`
- `mixed_ranking`

每个成员记录所属实体、实现侧面、模态、路径、行号、符号和向量。项目材料包括仓颉源码、
测试、Markdown 文档、PowerShell 脚本和 JSON 结果。另写入 17 条类型化关系。

Python 仅负责位置核对、MiniLM embedding 和结果汇总；数据组织及四种查询均由仓颉
`VectorMap` 执行。

## 3. 四个验证任务

| 范式 | 任务 | 指标 |
| --- | --- | --- |
| EAR | 从全项目定位负责 HDMG 生命周期与检索的实体 | Entity Purity@5 |
| DDR | 在固定预算内覆盖 HDMG 的五个实现侧面 | Aspect Coverage@5 |
| RER | 从索引失效函数恢复更新、查询、构建和测试链 | Dependency Coverage@5 |
| CMP | 从配置化查询接口取得配对的参数评估文档 | Pair Hit@1 |

各任务与扁平向量检索或向量近邻比较。不同指标不合并为一个平均准确率。

## 4. 完成标准

满足以下条件即可结束本阶段 MVP：

1. 四种范式均由仓颉入口实际运行并产生机器可读结果；
2. 四种范式共享同一份项目实体、成员向量和关系状态；
3. 每项任务给出对应基线、固定预算和独立指标；
4. 正式报告同时记录有效结果、持平结果和实验限制；
5. 复跑结果与提交的冻结结果一致；
6. 不宣称该小型案例能够证明真实 CodeAgent 的整体性能提升。

## 5. 后续可选工作

若继续扩展，可增加以下工作：

- 通过仓颉 AST 自动抽取函数、类型和调用关系；
- 扩充项目实体并设置独立留出查询集；
- 为同一侧面保留多个成员，评价微簇构建与组内多样化质量；
- 支持跨 key、非等长 group 的通用关系；
- 让同一 CodeAgent 在有无 Violas 上下文包的条件下执行真实修改任务；
- 比较补丁正确率、测试通过率、工具调用次数和上下文 token。
