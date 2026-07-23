# 2026-07-22 上午：图像 full 数据、Embedding 与论文配置对齐说明

## 1. 文档结论

今天上午已经完成三套图像数据的下载、校验、CLIP embedding、90/10 划分、查询集、ground truth 和微簇冻结，并生成可同时供 Python、仓颉、Faiss、Milvus、Qdrant、Chroma 使用的统一实验包。

当前采用的实验边界是：

```text
原始图片
  ↓ Python + 固定 OpenAI CLIP ViT-B/32，只执行一次
图片向量、类别文本向量
  ↓ Python 按论文参数冻结
90/10 划分、前 200 个查询、ground truth、sklearn KMeans 微簇
  ↓ 生成带 SHA-256 的统一实验包
Python / 仓颉 / Faiss / Milvus / Qdrant / Chroma 读取同一份输入
```

仓颉当前不负责读取原始图片，也不重新运行 CLIP 或 KMeans。仓颉负责读取冻结后的向量和微簇，构建 VectorMap、HDMG 并执行检索。这种做法能够排除“两种语言分别预处理导致输入不同”的干扰，更适合比较两边的向量组织和检索实现。

需要注意：`reproduction-ready` 表示实验输入已经具备正式复现条件，不表示六个数据集和全部数据库实验均已完成。目前只完成了 Caltech 仓颉 full 200-query；CUB、COCO 的仓颉 full 指标以及三个外部数据库的 full 指标仍待运行。

## 2. 今天上午实际完成的工作

1. 下载并校验 Caltech-101、CUB-200-2011 和固定 COCO-10k。
2. 下载 OpenAI CLIP `ViT-B/32` 官方 checkpoint，并记录 SHA-256。
3. 将逐图片 CLIP 推理改为批量推理，增加每批落盘和中断恢复功能。
4. 为三个数据集生成全部图片的 512 维 embedding 缓存。
5. 按类别执行 90% 训练、10% 查询划分，并固定论文默认的前 200 个查询。
6. 生成每个查询在多个 beta 下的精确 Top-3 ground truth。
7. 使用与 Python 论文实现相同的 sklearn KMeans 参数生成并冻结微簇分配。
8. 生成 Python JSONL 和仓颉 `cangjie_input.txt` 两种读取格式，并记录每个文件的 SHA-256。
9. 新增共享输入审计器，逐条比较 KEY、TRAIN、QUERY、GT，三个数据集均通过 `verified-identical`。
10. 完成三套数据的 Faiss exact、IVF、HNSW 基线。
11. 完成 Caltech full 的仓颉 200-query 验证，并定位 full 运行的主要性能瓶颈。

Python 参考目录 `violas_python` 保持不变。

## 3. 三个图像数据集的具体情况

### 3.1 数据规模与划分结果

| 数据集 | 原始/有效图片数 | 类别数 | 90% 训练记录 | 10% 查询池 | 论文默认实际查询数 | 微簇数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Caltech-101 | 原始 9,144；排除 467 张 `BACKGROUND_Google` 后为 8,677 | 101 | 7,766 | 911 | 200 | 809 |
| CUB-200-2011 | 11,788 | 200 | 10,597 | 1,191 | 200 | 1,397 |
| COCO-10k | 10,000 | 80 个 CLIP 预测类别 | 8,967 | 1,033 | 200 | 787 |

训练数量并不简单等于总数乘以 0.9，因为 Python 实现是对每个类别分别调用 `train_test_split`，每个类别都会单独取整。查询池是完整的 10% 留出集合；论文开源 benchmark 默认只取按固定顺序排列后的前 200 个查询。

### 3.2 Caltech-101

- 本地路径：`dataset/caltech-101/101_ObjectCategories/`
- 官方压缩包：`caltech-101.zip`
- 官方/本地 MD5：`3138e1922a9193bfa496528edbbc45d0`
- 解压后共 9,144 张图片。
- 与 Python 开源逻辑相同，过滤 `BACKGROUND_Google` 的 467 张图片，正式输入为 8,677 张、101 个目标类别。
- 原始图片树 SHA-256：`44d689398f2862f5df4a7bedbf002b69f777ba4080f5081766339814afa8d746`

Caltech 的数据版本、有效数量和背景类过滤方式已经与仓库记录的论文配置对齐。

### 3.3 CUB-200-2011

- 本地路径：`dataset/CUB_200_2011/images/`
- 官方压缩包：`CUB_200_2011.tgz`
- 官方/本地 MD5：`97eceeb196236b17998738112f37df78`
- 图片数：11,788。
- 类别数：200；类别目录名保留原始形式，例如 `001.Black_footed_Albatross`。
- 原始图片树 SHA-256：`7dd4cc71f1c1b89c047b4a34c648d19256fa6b7d4538e37774ad487a2bad1119`

CUB 的官方版本、图片数量、类别数量和目录类别标识已经与仓库记录的论文配置对齐。

### 3.4 COCO-10k

- 本地路径：`dataset/coco/`
- 图片清单：`dataset/coco/coco_dataset_10000.json`
- 图片数：10,000。
- 唯一 URL 数：10,000，缺失图片数为 0。
- JSON SHA-256：`945ab695bc612f2652e428406aba430b7152ce428090d55e9f1e94edc8aeaba7`
- 图片树 SHA-256：`483c0a1609064f114e33245180ab72ca16ca7233e007c1bd7469e48fc15a8b9b`
- 数据来源：公开 `MS_COCO_2017_URL_TEXT` URL/caption 表。
- 选择规则：按公开表顺序收集前 10,000 个唯一图片 URL，同一图片的 caption 聚合在同一条记录中。
- 类别生成：使用固定的 80 个 COCO 类别文本，经同一 CLIP 模型计算文本向量；每张图片按余弦相似度最高的类别分桶。

仓库的《数据与模型放置说明》明确要求 `coco_dataset_10000.json` 和最多 10,000 张图片，因此当前规模、公开来源和确定性选择规则与该说明一致。

但是仍需保留一个审计边界：当前冻结的 Python 源码中，COCO benchmark 默认文件名仍是早期 smoke 用的 `coco_dataset_40.json`，而数据放置说明要求正式实验使用 COCO-10k。我们没有修改 Python 参考代码，而是在统一预处理工具中显式指定 COCO-10k。除非能够取得论文作者当时使用的原始 `coco_dataset_10000.json` 并比较哈希，否则目前只能确认“来源、规模和生成规则对齐”，不能证明 10,000 张图片的身份与作者本地文件逐张完全相同。

## 4. 与论文/Python 开源配置的对齐情况

| 配置项 | Python/论文配置 | 当前统一实验包 | 判断 |
| --- | --- | --- | --- |
| 图像模型 | OpenAI CLIP `ViT-B/32` | 相同 | 已对齐 |
| CLIP checkpoint | 官方 `ViT-B-32.pt` | SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` | 已冻结 |
| 图片向量维度 | 512 | 512 | 已对齐 |
| 类别提示词 | `a photo of a {}` | 相同 | 已对齐 |
| 图片向量归一化 | 提取时不归一化，余弦计算时归一化 | 相同 | 已对齐 |
| 类别文本向量 | L2 归一化 | 相同 | 已对齐 |
| 数据划分 | 每个类别单独 90/10 | 相同 | 已对齐 |
| 划分函数 | sklearn `train_test_split` | 相同 | 已对齐 |
| 随机种子 | `random_state=42` | 相同 | 已对齐 |
| Shuffle | `shuffle=True` | 相同 | 已对齐 |
| 默认查询上限 | 200 | 200 | 已对齐 |
| Top-K | 3 | 3 | 已对齐 |
| 微簇数量 | `round(n ** (1-alpha))` | 相同 | 已对齐 |
| alpha | 0.5 | 0.5 | 已对齐 |
| 微簇算法 | sklearn KMeans | 相同，并冻结分配结果 | 已对齐 |
| KMeans 参数 | `random_state=42, n_init=10` | 相同 | 已对齐 |
| Mixed 距离 | `(1-beta) × embedding distance + beta × semantic distance` | 相同 | 已对齐 |
| beta | 论文表格为 0.0 到 1.0、步长 0.1 | 当前冻结 0.0、0.3、0.5、0.8、1.0 | 尚未完整对齐 |
| 向量存储类型 | Python/CLIP 主要为 Float32 | 仓颉当前读入 Float64 | 数值来源相同，但内部类型尚未对齐 |
| 六数据集平均结果 | 三个图像 + 三个文本 | 当前只完成图像输入，且仅 Caltech 完成仓颉 full 查询 | 尚未完成 |
| 数据库版本与硬件 | 论文实验环境 | 当前本机 Docker/本机 CPU | 需要单独记录，延迟不能直接照搬论文数值 |

因此，当前可以声称的是：“图像实验输入协议和主要算法参数已经按 Python 论文版本冻结，三套图像输入可供各后端公平比较。”当前还不能声称：“已经完整复现论文表格中的六数据集平均指标。”

## 5. Embedding、模型和实验包缓存

所有缓存均已持久化到本地磁盘，重启电脑不会消失。

| 内容 | 路径 | 数量/大小 |
| --- | --- | ---: |
| CLIP checkpoint | `model/clip/ViT-B-32.pt` | 353,976,522 字节 |
| Caltech embedding 缓存 | `artifacts/python-paper-90-10/caltech-full/source_embeddings.jsonl` | 8,677 条，90,223,601 字节 |
| CUB embedding 缓存 | `artifacts/python-paper-90-10/cub-full/source_embeddings.jsonl` | 11,788 条，123,072,973 字节 |
| COCO embedding 缓存 | `artifacts/python-paper-90-10/coco-full/source_embeddings.jsonl` | 10,000 条，103,888,108 字节 |

每个 embedding 缓存旁都有 `.meta.json`，记录模型名、checkpoint SHA-256、schema 版本和向量维度。如果模型哈希发生变化，工具会拒绝误用旧缓存。

每个 full artifact 还包含：

- `records.jsonl`：全部图片的 recordId、路径、类别和 embedding；
- `key_vectors.jsonl`：类别文本向量；
- `splits.jsonl`：训练 ID、完整查询池 ID、实际 200 个查询 ID；
- `queries.jsonl`：查询图片向量和 query key vector；
- `ground_truth.jsonl`：不同 beta 对应的精确 Top-3；
- `microclusters.jsonl`：Python sklearn 生成的微簇分配；
- `cangjie_input.txt`：仓颉直接读取的流式输入；
- `manifest.json`：所有输入参数、数量和文件 SHA-256。

模型、原始数据和 artifact 都已加入 `.gitignore`，不会被误提交到 Git。它们只存在于当前电脑；换电脑或删除这些目录后需要重新准备。

## 6. 如何证明 Python 与仓颉输入相同

新增工具：

```powershell
python tools/verify_shared_artifact.py artifacts/python-paper-90-10/caltech-full
python tools/verify_shared_artifact.py artifacts/python-paper-90-10/cub-full
python tools/verify_shared_artifact.py artifacts/python-paper-90-10/coco-full
```

审计分为四层：

1. 检查 manifest 中每个文件的 SHA-256；
2. 检查 CLIP checkpoint SHA-256；
3. 逐行比较 Python JSONL 和仓颉输入中的 KEY、TRAIN、QUERY、GT；
4. 通过稳定 recordId/queryId 比较各后端逐查询 Top-3，不依赖数据库内部插入编号。

三套数据均已通过 `verified-identical`。当前仓颉输入文件为：

| 数据集 | `cangjie_input.txt` 大小 | SHA-256 |
| --- | ---: | --- |
| Caltech | 86,363,049 字节 | `035c30cdbbca32dd716d7c6259fb7ae17d7c176fc07b9799fc60624ddcca2407` |
| CUB | 117,390,420 字节 | `a5870ddbc682be47d86f4ddc2e51baef5a06d19ea7587f2e73035395dacdfe7f` |
| COCO | 98,603,685 字节 | `57b23e1567627e9fa71be5aaad1657175ec105296976be79c99810beba564d35` |

## 7. 今天得到的初步实验结果

### 7.1 Faiss 基线

以下结果使用相同 artifact、200 个查询和 Top-K=3：

| 数据集 | 方法 | Recall@3 | NDCG@3 | Mean latency (ms/query) |
| --- | --- | ---: | ---: | ---: |
| Caltech | Exact | 1.000000 | 1.000000 | 1.5085 |
| Caltech | IVF | 1.000000 | 1.000000 | 0.1122 |
| Caltech | HNSW | 1.000000 | 1.000000 | 0.0363 |
| CUB | Exact | 1.000000 | 1.000000 | 1.7846 |
| CUB | IVF | 0.980000 | 0.985614 | 0.1489 |
| CUB | HNSW | 0.993333 | 0.995307 | 0.1822 |
| COCO | Exact | 1.000000 | 1.000000 | 1.4653 |
| COCO | IVF | 0.985000 | 0.989134 | 0.3491 |
| COCO | HNSW | 0.996667 | 0.997654 | 0.0544 |

这些 Faiss 数字用于验证统一输入、ground truth 和近似索引基线，不能直接与论文中不同机器上的延迟作数值等同。

### 7.2 仓颉 Caltech full

参数：200 个查询、Top-K=3、beta=0.3。

| 方法 | Recall@3 |
| --- | ---: |
| Exact | 1.000000 |
| Mixed | 0.998333 |
| HDMG | 0.963333 |

阶段耗时：

| 阶段 | 优化前 | 冻结 Python 微簇后 |
| --- | ---: | ---: |
| 读取统一输入 | 约 9 秒 | 约 9 秒 |
| KMeans/组构建 | 约 500 秒 | 约 0.52 秒 |
| HDMG 构建 | 约 90 秒 | 约 90 秒 |

组构建大幅下降的原因是：原入口在仓颉中重新执行一次 KMeans；新协议直接读取 Python/sklearn 已冻结的同一微簇分配。现在真正的主要瓶颈是 809 个微簇的 HDMG 构建。

当前仓颉 full 入口只汇总 Recall，正式论文表格还需要补 NDCG、mean/P50/P95 查询延迟及结构化 JSON 输出。论文截图中的表 2 是六个数据集的平均值，Caltech 单数据集结果不能直接与该平均值逐项比较。

## 8. 明天再次运行时是否需要重新计算

如果直接使用现有 artifact，不需要重新读取图片或计算 CLIP embedding。建议先执行输入审计，再运行后端：

```powershell
python tools/verify_shared_artifact.py artifacts/python-paper-90-10/caltech-full

python tools/verify_shared_artifact.py `
  artifacts/python-paper-90-10/caltech-full `
  --run-cangjie `
  --max-queries 200 `
  --beta 0.3
```

如果重新执行 `paper_artifact.py`，日志应显示 `cached=总数, pending=0`，不会重新计算图片 embedding；但仍会加载模型、读取缓存、校验图片树并重新生成 sklearn 微簇和实验包，因此会花费几十秒到约两分钟。

仓颉当前没有持久化 HDMG 图索引，所以每次启动新的仓颉进程仍会重新构建 HDMG。Caltech 目前约需 90 秒。后续可增加 HDMG 索引序列化，进一步缩短重复实验时间。

## 9. 下午复核时建议重点检查

1. 是否认可“Python 统一预处理一次，仓颉读取冻结 embedding/微簇”的实验边界。
2. 是否能够找到论文作者原始 COCO-10k JSON 或其哈希，用于完成 COCO 图片身份的最终核验。
3. 是否将 beta 扫描补齐为 `0.0, 0.1, ..., 1.0`，以完整对应论文表 2。
4. 是否优先优化/缓存 HDMG 构建，因为 KMeans 已不再是主要瓶颈。
5. 是否先完成 CUB、COCO 的仓颉 200-query，再启动三个外部数据库 full 实验。
6. 正式报告中应分别展示单数据集结果和六数据集平均值，不能混写。

## 10. 当前尚未完成的事项

- CUB、COCO 的仓颉 full 200-query 指标；
- Caltech/CUB/COCO 的完整 beta 0.0–1.0 步长 0.1 扫描；
- 仓颉 full 的 NDCG、mean/P50/P95 延迟和 Result JSON；
- Milvus、Qdrant、Chroma 的三套图像 full 实验；
- 与冻结 Python Violas 实现的完整逐查询结果文件对照；
- Float32 仓颉数据通路；
- HDMG 索引序列化和进一步构图优化；
- 三个文本数据集的 full 输入与最终六数据集平均结果。

完成上述事项后，才能严谨地判断仓颉版本是否复现出与论文 Python 版本接近的完整实验结果。
