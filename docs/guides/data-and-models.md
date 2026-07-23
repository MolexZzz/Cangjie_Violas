# 数据、模型与实验输出

本仓库不提交大型数据、模型、Embedding、实验结果或 Python 参考实现。

## 本地目录

```text
dataset/
├── caltech-101/101_ObjectCategories/
├── CUB_200_2011/images/
└── coco/

model/                         # CLIP 等模型缓存
artifacts/python-paper-90-10/ # 冻结后的统一实验输入
results/                       # JSON、Markdown 和日志
violas_python/                 # 本地 Python 参考实现
```

这些目录均已由 `.gitignore` 排除；其中的本地文件不会因为 Git 整理而删除。

## 当前图像数据规模

| 数据集 | 原始记录数 | 主要用途 |
| --- | ---: | --- |
| Caltech-101 | 8,677 | 完整图像实验 |
| CUB-200-2011 | 11,788 | 完整图像实验 |
| COCO | 10,000 | 按项目清单构建的完整实验规模 |

统一预处理使用 CLIP `ViT-B/32` 生成 512 维图片向量和类别文本向量，按类别执行 90% 建库、10%
查询划分。更详细的随机种子、查询范围和 ground truth 规则见
[python-paper-90-10-protocol.md](python-paper-90-10-protocol.md)。

## 下载与生成

图像数据准备入口：

```powershell
python tools\download_image_datasets.py --help
```

COCO 清单下载工具位于：

```powershell
python photo-data\download_coco_dataset.py --help
```

冻结 artifact 的生成与验证工具：

```powershell
python tools\paper_artifact.py --help
python tools\verify_shared_artifact.py --help
```

## 注意

- 仓颉、Faiss、Milvus、Qdrant 和 Chroma 应读取同一份冻结 artifact，不能各自重新划分数据；
- `results/` 中的文件是实验产物，不应提交到 Git；
- `violas_python/` 只作为只读参考，不应修改或提交。
