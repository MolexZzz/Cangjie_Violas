# Python 论文 90/10 复现实验协议

## 结论

仓库已经实现 `python-paper-90-10` 统一实验包。预处理只执行一次 90/10 划分，随后仓颉、Faiss、Milvus、Qdrant 和 Chroma 读取同一个目录，禁止各后端自行重新划分数据。

仓库的 precomputed 样例只能验证流程，生成结果会强制标为 `validation-only`。本地现已按正式规模下载三套图像数据，并生成 `reproduction-ready` 统一实验包；这表示输入协议已经具备复现条件，不表示所有后端的正式指标已经跑完。

当前已冻结的正式输入口径为：

1. 使用 Caltech 8,677 张有效图片（排除背景类）、CUB 11,788 张图片和固定 COCO-10k；
2. 用同一份 OpenAI CLIP `ViT-B/32` 权重生成并冻结三套向量；
3. 核对 90/10 划分、查询上限和 ground truth 后，生成时显式使用 `--full-verified`。

## 实验包内容

| 文件 | 内容 | 读取方 |
| --- | --- | --- |
| `manifest.json` | 协议名、状态、数量、CLIP 模型、随机种子、源数据哈希、每个文件哈希 | 全部后端/审计 |
| `records.jsonl` | 每张图片的固定 `recordId`、类别、源路径和 CLIP 图片向量 | Faiss、数据库 |
| `key_vectors.jsonl` | `a photo of a {类别}` 的 CLIP 文本向量 | Faiss、数据库 |
| `splits.jsonl` | 明确的 90% 训练 ID 和 10% 查询 ID | Faiss、数据库 |
| `queries.jsonl` | 查询 ID、图片向量、真实类别、文本 key vector | Faiss、数据库 |
| `ground_truth.jsonl` | 每个 query、每个 beta 的精确 Top-K ID | 全部后端 |
| `microclusters.jsonl` | Python 论文参数生成的 sklearn KMeans 微簇分配 | 仓颉 |
| `cangjie_input.txt` | 与上述文件同次导出、受 manifest 哈希保护的仓颉流式输入 | 仓颉 |

`recordId` 由数据集名、类别和规范化图片相对路径计算 SHA-256 后生成；它不依赖某次数据库插入顺序。Python 侧统一加载器会检查实验包内所有文件的 SHA-256，文件被修改后直接拒绝运行；仓颉输入文件的 SHA-256 也记录在 manifest 中，运行前由实验脚本审计。

图片 embedding 和微簇都只由冻结的 Python 协议生成一次。仓颉不重新运行 CLIP 或 KMeans，而是读取完全相同的 512 维向量和微簇分配。运行前执行：

```powershell
python tools/verify_shared_artifact.py artifacts/python-paper-90-10/caltech-full
```

审计器会逐行核对 Python JSONL 与 `cangjie_input.txt` 的 KEY、TRAIN、QUERY、GT，并验证 manifest 中全部 SHA-256。

## 预处理命令

先用现有 Caltech 样例和真实 CLIP 文本 key vector 验证完整链路（图片范围仍只是 sample）：

```powershell
python tools/paper_artifact.py `
  --dataset caltech `
  --source-kind precomputed-sample `
  --source dataset/precomputed/caltech_precomputed.txt `
  --output-dir artifacts/python-paper-90-10/caltech-validation `
  --key-vector-source clip-text
```

Caltech/CUB 完整原图的目录结构为“根目录/类别/图片”：

```powershell
python tools/paper_artifact.py `
  --dataset caltech `
  --source-kind folder-images `
  --source dataset/caltech-101/101_ObjectCategories `
  --output-dir artifacts/python-paper-90-10/caltech-full `
  --key-vector-source clip-text `
  --model ViT-B/32 `
  --seed 42 `
  --test-size 0.1 `
  --full-verified
```

COCO 使用固定的 10,000 张图片清单，不使用 40 张 smoke 样例：

```powershell
python tools/paper_artifact.py `
  --dataset coco `
  --source-kind coco-json `
  --source dataset/coco/coco_dataset_10000.json `
  --image-root dataset/coco `
  --output-dir artifacts/python-paper-90-10/coco-full `
  --key-vector-source clip-text `
  --full-verified
```

脚本复用了冻结 Python 版本的关键参数：OpenAI CLIP `ViT-B/32`、提示词 `a photo of a {}`，以及按类别执行的 `train_test_split(test_size=0.1, random_state=42, shuffle=True)`。manifest 同时记录 CLIP 包版本和本地 checkpoint SHA-256；如果模型尚未实际加载，checkpoint 哈希为空，实验包不能据此宣称完成正式复现。

## 五个后端的运行方式

```powershell
python tools/run_faiss_baseline.py --artifact artifacts/python-paper-90-10/caltech-full

python tools/external_db_benchmark.py --backend milvus --artifact artifacts/python-paper-90-10/caltech-full --scale full
python tools/external_db_benchmark.py --backend qdrant --artifact artifacts/python-paper-90-10/caltech-full --scale full
python tools/external_db_benchmark.py --backend chroma --artifact artifacts/python-paper-90-10/caltech-full --scale full

Set-Location cj_core
"paper ../artifacts/python-paper-90-10/caltech-full/cangjie_input.txt 1000000 0.3" | cjpm run
```

单个 beta 用于调试；正式实验使用 `all`，只加载一次数据、只构建一次 HDMG，然后依次运行
`0.0, 0.1, ..., 1.0`，最终输出与 Python 图像 benchmark 相同布局的 Recall、ms/query 和
HDMG 分阶段汇总表：

```powershell
Set-Location cj_core
"paper ../artifacts/python-paper-90-10/caltech-full/cangjie_input.txt 200 all" | cjpm run
```

Docker 服务未运行或尚未合并外部数据库结果时，Milvus、Qdrant、Chroma 列显示 `N/A`；
真实数据库实验完成后，由统一结果汇总步骤填入对应列，不能把 `N/A` 当作零值。

`--scale smoke` 只限制读取前 20 个冻结 query，不改变训练集，也不会重新划分。`--scale full` 使用实验包中的全部 query。

## 与 five-fold 的关系

`python-paper-90-10` 用于复现开源 Python 图像 benchmark；`five-fold` 是新增稳健性实验。两者分别报告，不把五折平均值写成论文指标。原有 `tools/precomputed_artifacts.py export` 继续生成五折 artifact，并在 manifest 中明确写入 `protocol: five-fold`。

## 已完成的 full 输入与链路验证

三套 full artifact 均使用 OpenAI CLIP `ViT-B/32`，checkpoint SHA-256 为 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。Caltech、CUB、COCO 的 Python JSONL—仓颉流输入逐行审计均已通过。

Caltech 已完成 200 个查询的仓颉 full 验证（beta=0.3）：Exact Recall@3=1.000000、Mixed Recall@3=0.998333、HDMG Recall@3=0.963333。原仓颉入口重复执行纯仓颉 KMeans，组构建约需 500 秒；冻结 Python 微簇后降至约 0.53 秒。当前主要瓶颈转为 HDMG 构建，Caltech 809 个微簇约需 90 秒。CUB、COCO 的仓颉 200-query 和三个外部数据库 full 指标仍需继续运行，不能据此宣称三套图像实验全部完成。
