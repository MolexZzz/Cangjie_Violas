# Faiss、数据维护与代码量对比

本工具组直接读取 `python-paper-90-10` 冻结实验包，不重新生成 embedding，也不重新划分
训练集和查询集。所有 JSON、Markdown 和日志应保存在已被 `.gitignore` 排除的 `results/` 中。

## 1. Faiss 检索基线

实现位于 `tools/run_faiss_baseline.py`，包括：

- `IndexFlatIP`：精确 cosine/IP 基线，用于检查输入和 ground truth；
- `IndexIVFFlat`：倒排近似索引，记录 `nlist` 和 `nprobe`；
- `IndexHNSWFlat`：图近似索引，记录 `M`、`efConstruction` 和 `efSearch`；
- Recall@3、集合相关性 NDCG@3、构建时间、Mean/P50/P95 latency；
- 序列化索引大小和进程 Peak RSS；
- Git commit、Faiss/NumPy 版本和操作系统。

该实验是单向量 cosine 检索，不包含 Violas 的类别语义距离或 mixed score。因此它是底层向量
索引基线，不应与完整 Violas 的功能范围混为一谈。

单数据集调试：

```powershell
python tools\run_faiss_baseline.py `
  --artifact artifacts\python-paper-90-10\caltech-full `
  --max-queries 20 `
  --repeats 1 `
  --output results\faiss-and-maintenance\faiss-caltech-smoke.json
```

正式运行时将 `--max-queries` 设为 `0`，使用完整 10% 查询池。

## 2. 数据与索引维护

`tools/run_maintenance_benchmark.py` 默认固定处理 200 个向量，记录整批插入、更新、删除、
初始构建和索引更新耗时。它不会修改冻结 artifact：

- 插入数据来自冻结的 10% 查询池；
- 更新数据是训练向量的确定性副本；
- 删除对象是本轮插入的对象；
- Faiss 原生支持 `add`，但为了兼容 HNSW，通用更新和删除统一按完整重建统计；
- Milvus、Qdrant 和 Chroma 使用稳定 record ID 执行 upsert/delete，服务端索引维护耗时包含在
  同步操作时间中。

只验证 Faiss：

```powershell
python tools\run_maintenance_benchmark.py `
  --artifact artifacts\python-paper-90-10\caltech-full `
  --mutation-count 200 `
  --backends cangjie,faiss `
  --output results\faiss-and-maintenance\maintenance-caltech.json
```

Docker 服务启动后加入三个数据库：

```powershell
python tools\run_maintenance_benchmark.py `
  --artifact artifacts\python-paper-90-10\caltech-full `
  --mutation-count 200 `
  --backends faiss,milvus,qdrant,chroma `
  --execution-mode service `
  --output results\faiss-and-maintenance\maintenance-caltech-all.json
```

仓颉侧已经增加按稳定 `recordId` 执行的对象插入、原位向量更新和删除接口。每次成功变更都会
使旧 HDMG 失效；维护 benchmark 在一批操作结束后完整重建 HDMG，并分别保存对象操作时间和
索引重建时间。这样不会把“更新数据”和“更新索引”混成一个数字。

## 3. 代码行数

`tools/count_source_lines.py` 默认只统计 Git 已跟踪文件，同时给出物理行、非空行和排除纯注释后的
源码行。数据、文档、构建产物、结果和第三方依赖不计入。

只统计当前仓库：

```powershell
python tools\count_source_lines.py
```

如果本地已有 Faiss 官方源码 checkout：

```powershell
python tools\count_source_lines.py `
  --faiss-root D:\source\faiss `
  --output results\faiss-and-maintenance\source-lines.json
```

正式报告必须同时保存 Faiss commit。代码行数只能说明工程规模，不能直接说明速度、正确性或
代码质量。

## 4. 三数据集统一入口

```powershell
.\tools\run_faiss_and_maintenance.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/faiss-and-maintenance/2026-07-23" `
  -MaxQueries 0 `
  -MutationCount 200 `
  -MaintenanceBackends "cangjie,faiss"
```

确认仓颉与 Faiss 结果正常后，再把 `MaintenanceBackends` 改成
`cangjie,faiss,milvus,qdrant,chroma`。所有输出都会持久化到指定 `RunRoot`。
