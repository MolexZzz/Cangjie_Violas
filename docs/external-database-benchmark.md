# 仓颉外部向量数据库 Benchmark 框架

## 目标

仓颉 runner 通过统一 `ExternalVectorBackend` 启动数据库 benchmark。Milvus、Qdrant、Chroma
的 SDK 差异封装在 `tools/external_db_benchmark.py`，结果统一返回 `DB_RESULT_JSON`。

该框架比较两种口径：

1. 数据库原始 embedding cosine Top-K；
2. 数据库取 `Top-K × 80` 候选，再按 Python 论文版本的 mixed score 在 benchmark 层 rerank。

所有 backend 使用稳定 `recordId`，数据库内部整数 ID 不进入指标计算。

## 快速验证

不启动真实数据库即可使用 exact mock 验证完整链路：

```powershell
cd cj_core
"dbbench mock 1 smoke" | cjpm run
```

真实 backend：

```powershell
"dbbench qdrant 1 smoke" | cjpm run
"dbbench milvus 1 smoke" | cjpm run
"dbbench chroma 1 smoke" | cjpm run
```

也可以直接运行适配器：

```powershell
python tools\external_db_benchmark.py --backend qdrant --dataset 1 --scale smoke
```

## 默认连接

| Backend | 默认地址 | 环境变量 |
|---|---|---|
| Qdrant | `http://127.0.0.1:6333` | `QDRANT_URL` |
| Milvus | `http://127.0.0.1:19530` | `MILVUS_URI`、`MILVUS_TOKEN` |
| Chroma | `127.0.0.1:8000` | `CHROMA_HOST`、`CHROMA_PORT` |

可选 Python 依赖见 `tools/requirements-external-db.txt`。数据库服务必须由实验环境显式启动；
runner 不会自动下载镜像、启动服务或把连接失败伪装成 N/A 结果。

## 输出字段

- `rawVector.recallAtK/ndcgAtK`：数据库原始向量结果；
- `mixedRerank.recallAtK/ndcgAtK`：候选经过统一 mixed rerank 后的结果；
- `buildMs`：建集合和批量写入时间；
- `latencyMs.database*`：数据库 SDK 调用时间；
- `latencyMs.rerankMean`：benchmark rerank 时间；
- `config`：数据库地址、索引、metric、beta、candidate multiplier；
- `provenance`：Git commit、系统和依赖环境。

正式实验时每个 fold 应重建独立 collection，避免训练集之间相互污染。当前框架先固定 fold-0，
用于搭建和调试；扩展 full 时再循环五个 fold。
