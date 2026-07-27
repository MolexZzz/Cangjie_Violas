# 90/10 实验协议

`python-paper-90-10` 用于固定数据划分、向量、微簇和检索真值，使仓颉、Faiss、Milvus、
Qdrant 与 Chroma 在相同输入上运行。数据按类别划分，测试集比例为 0.1，随机种子为 42。
完整实验使用全部测试查询，即生成输入时设置 `--max-queries 0 --full-verified`。

## 文件组成

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 协议、数据规模、模型、随机种子和文件哈希 |
| `records.jsonl` | 稳定 `recordId`、类别、源路径和图像向量 |
| `key_vectors.jsonl` | 类别文本向量 |
| `splits.jsonl` | 训练集与查询集 ID |
| `queries.jsonl` | 查询向量、类别和 key vector |
| `ground_truth.jsonl` | 各查询、各 β 取值的精确 Top-K |
| `microclusters.jsonl` | KMeans 微簇分配 |
| `cangjie_input.txt` | 与上述数据对应的仓颉流式输入 |

`recordId` 由数据集、类别和规范化相对路径计算得到，不依赖数据库插入顺序。
`verify_shared_artifact.py` 校验 manifest 中的 SHA-256，并逐行对照 JSONL 与仓颉输入。

## 生成输入

Caltech-101 使用按类别组织的图片目录：

```powershell
python tools\paper_artifact.py `
  --dataset caltech `
  --source-kind folder-images `
  --source dataset/caltech-101/101_ObjectCategories `
  --output-dir artifacts/python-paper-90-10/caltech-full `
  --key-vector-source clip-text `
  --model ViT-B/32 `
  --seed 42 `
  --test-size 0.1 `
  --max-queries 0 `
  --full-verified
```

CUB-200-2011：

```powershell
python tools\paper_artifact.py `
  --dataset cub `
  --source-kind folder-images `
  --source dataset/CUB_200_2011/images `
  --output-dir artifacts/python-paper-90-10/cub-full `
  --key-vector-source clip-text `
  --model ViT-B/32 `
  --seed 42 `
  --test-size 0.1 `
  --max-queries 0 `
  --full-verified
```

COCO 使用项目中的固定 10,000 张图片清单：

```powershell
python tools\paper_artifact.py `
  --dataset coco `
  --source-kind coco-json `
  --source dataset/coco/coco_dataset_10000.json `
  --image-root dataset/coco `
  --output-dir artifacts/python-paper-90-10/coco-full `
  --key-vector-source clip-text `
  --model ViT-B/32 `
  --seed 42 `
  --test-size 0.1 `
  --max-queries 0 `
  --full-verified
```

生成后执行：

```powershell
python tools\verify_shared_artifact.py artifacts/python-paper-90-10/caltech-full
```

## 后端调用

```powershell
python tools\run_faiss_baseline.py --artifact artifacts/python-paper-90-10/caltech-full
python tools\external_db_benchmark.py --backend qdrant --artifact artifacts/python-paper-90-10/caltech-full --scale full

Set-Location cj_core
"paper ../artifacts/python-paper-90-10/caltech-full/cangjie_input.txt 0 all" | cjpm run
```

参数 `all` 依次运行 β=0.0 至 1.0；`--scale full` 使用完整查询集。小规模调试可以使用
`--scale smoke`，但其结果不能并入完整实验表。`five-fold` 是另一套实验协议，与本页的
90/10 结果分别记录。
