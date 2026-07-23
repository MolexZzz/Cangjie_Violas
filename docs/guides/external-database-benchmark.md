# 外部向量数据库实验

仓颉 runner 通过 `ExternalVectorBackend` 调用 `tools/external_db_benchmark.py`，由后者适配
Milvus、Qdrant 和 Chroma。实验使用稳定 `recordId` 计算指标，不依赖数据库内部 ID。

适配器报告两种检索方式：

1. 数据库按 embedding cosine 直接返回 Top-K；
2. 数据库先召回候选，再由评测程序按 mixed score 重排。

第二种方式用于分析候选召回对混合检索的影响，不与数据库原生结果合并。

## 连接配置

| 后端 | 默认地址 | 环境变量 |
| --- | --- | --- |
| Qdrant | `http://127.0.0.1:6333` | `QDRANT_URL` |
| Milvus | `http://127.0.0.1:19530` | `MILVUS_URI`、`MILVUS_TOKEN` |
| Chroma | `127.0.0.1:8000` | `CHROMA_HOST`、`CHROMA_PORT` |

Python 依赖见 `tools/requirements-external-db.txt`。数据库服务需由运行者预先启动；
适配器不会下载镜像或把连接错误记录为零值。

## 运行

exact mock 可用于检查仓颉与 Python 之间的调用链：

```powershell
cd cj_core
"dbbench mock 1 smoke" | cjpm run
```

真实后端可从仓颉入口运行：

```powershell
"dbbench qdrant 1 smoke" | cjpm run
"dbbench milvus 1 smoke" | cjpm run
"dbbench chroma 1 smoke" | cjpm run
```

也可直接调用 Python 适配器：

```powershell
python tools\external_db_benchmark.py `
  --backend qdrant `
  --artifact artifacts/python-paper-90-10/caltech-full `
  --scale full
```

每次实验应新建 collection，以免不同数据划分相互污染。90/10 输入只在预处理阶段生成一次，
数据库端不得再次划分。

## 输出

输出 JSON 分别保存原生向量检索和 mixed rerank 的 Recall@K、NDCG@K 与延迟，同时记录
建库时间、后端配置、Git commit、操作系统和依赖版本。Docker service 模式下的延迟包含
SDK 调用及进程间通信，不应与进程内索引延迟直接比较。
