# 数据与模型

原始图像、模型权重、embedding 和逐查询输出不纳入版本控制。仓库默认使用以下本地目录：

```text
dataset/
├── caltech-101/101_ObjectCategories/
├── CUB_200_2011/images/
└── coco/

model/                         CLIP 模型缓存
artifacts/python-paper-90-10/ 实验输入
results/                       JSON、Markdown 和日志
violas_python/                 Python 参考实现
```

本次实验包含 8,677 张 Caltech-101 图像、11,788 张 CUB-200-2011 图像和项目清单中的
10,000 张 COCO 图像。预处理使用 CLIP `ViT-B/32` 生成 512 维图像向量与类别文本向量，
再以随机种子 42 按类别作 90/10 划分。

数据下载和实验输入生成命令如下：

```powershell
python tools\download_image_datasets.py --help
python tools\paper_artifact.py --help
python tools\verify_shared_artifact.py --help
```

仓颉、Faiss 和外部数据库必须读取同一组 `artifacts/python-paper-90-10/` 输入，不得在后端内
重新划分数据。输入文件的具体结构和校验规则见
[90/10 实验协议](python-paper-90-10-protocol.md)。
