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

### 当前工作站真实服务验证

2026-07-21 已在 Docker 真实服务上完成 Caltech smoke：

| 后端 | 服务/镜像版本 | 本机端口 | 仓颉入口验证 |
|---|---|---|---|
| Milvus | server 2.3.15，`pymilvus` 2.6.9 | 19530 | 通过 |
| Qdrant | server 1.16.1，`qdrant-client` 1.17.0 | 6333 | 通过 |
| Chroma | `chromadb/chroma:1.5.5`，client 1.5.5 | 8000 | 通过 |

三者均使用同一 Caltech sample 的 fold-0、1616 条训练向量和 20 条查询，成功完成建库、插入、
向量查询、`TopK × 80` 候选和 Mixed Rerank。该结果证明真实连接链路可用，但仍是 smoke，不能
代替 full 数据和正式重复实验。

当前工作站后续可直接启动已有容器：

```powershell
docker start milvus-etcd milvus-minio milvus-standalone
docker start violas-qdrant
docker start violas-chroma
```

其中 Qdrant 与 Chroma 只绑定 `127.0.0.1`；Milvus 是工作站原有 Compose 服务。迁移到其他机器时，
应按官方 Docker/Compose 文档重新创建并固定镜像版本，不能依赖上述本机容器名称。

## 输出字段

- `rawVector.recallAtK/ndcgAtK`：数据库原始向量结果；
- `mixedRerank.recallAtK/ndcgAtK`：候选经过统一 mixed rerank 后的结果；
- `buildMs`：建集合和批量写入时间；
- `latencyMs.database*`：数据库 SDK 调用时间；
- `latencyMs.rerankMean`：benchmark rerank 时间；
- `config`：数据库地址、索引、metric、beta、candidate multiplier；
- `provenance`：Git commit、系统和依赖环境。

正式实验时每个 split/fold 应重建独立 collection，避免训练集之间相互污染。当前框架既保留
`precomputed-five-fold` 的 fold-0 调试入口，也支持直接读取冻结的 `python-paper-90-10` artifact。
90/10 只在预处理阶段按类别和 `random_state=42` 生成一次；数据库不再自行切分。五折仍作为另一个
明确命名的实验协议，不能与 Python 原指标混用。具体命令见 `docs/python-paper-90-10-protocol.md`。
