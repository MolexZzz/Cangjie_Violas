# Faiss 基线、维护基准与代码量统计

本组工具读取 `python-paper-90-10` 实验输入，不重新生成 embedding 或划分数据。原始输出
写入已被 `.gitignore` 排除的 `results/`。

## Faiss 基线

`tools/run_faiss_baseline.py` 运行 `IndexFlatIP`、`IndexIVFFlat` 和 `IndexHNSWFlat`，
记录 Recall@3、NDCG@3、构建时间、查询延迟、索引大小与进程峰值内存。

```powershell
python tools\run_faiss_baseline.py `
  --artifact artifacts\python-paper-90-10\caltech-full `
  --max-queries 0 `
  --repeats 3 `
  --output results\faiss-and-maintenance\faiss-caltech.json
```

该基线只计算 embedding cosine，相应真值也由精确 embedding 检索生成。它与 Violas 的
mixed search 目标不同。

## 维护基准

`tools/run_maintenance_benchmark.py` 测量固定批量的插入、更新、删除和索引重建。默认先预热
一次，再重复三次，并保存每轮数据。输入文件不会被修改。

```powershell
python tools\run_maintenance_benchmark.py `
  --artifact artifacts\python-paper-90-10\caltech-full `
  --mutation-count 200 `
  --backends cangjie,faiss `
  --repeats 3 `
  --warmup-runs 1 `
  --output results\faiss-and-maintenance\maintenance-caltech.json
```

Faiss 的通用更新和删除按完整重建计时；Milvus、Qdrant 和 Chroma 使用稳定 record ID
执行 upsert 与 delete。仓颉在对象变化后将旧 HDMG 标记为失效，并单独记录对象操作和重建时间。
这些数据作为工程诊断保存在本地，不列入项目汇总结果。

## 代码量

`tools/count_source_lines.py` 默认统计 Git 已跟踪源码，分别给出物理行、非空行和去除纯注释后的
源码行。数据、文档、构建产物、实验输出和第三方依赖不计入。

```powershell
python tools\count_source_lines.py `
  --faiss-root <path-to-faiss> `
  --output results\faiss-and-maintenance\source-lines.json
```

报告 Faiss 数据时必须同时记录其 commit。跨语言的代码行数只能说明选定模块的规模，不表示
算法复杂度、性能或代码质量。

## 批量运行

```powershell
.\tools\run_faiss_and_maintenance.ps1 `
  -Datasets caltech,cub,coco `
  -RunRoot "results/faiss-and-maintenance/reproduction" `
  -MaxQueries 0 `
  -MutationCount 200 `
  -MaintenanceBackends "cangjie,faiss"
```

数据库服务就绪后，可将 `MaintenanceBackends` 扩展为
`cangjie,faiss,milvus,qdrant,chroma`。
