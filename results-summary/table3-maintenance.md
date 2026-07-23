# Table 3-style data and index maintenance

论文主表列为 200 条向量插入、200 条向量更新、初始索引构建和索引更新。删除、内存和索引状态放在辅助表，避免改变论文主表含义。

| Dataset | Insert(ms) Violas | Insert(ms) Milvus | Insert(ms) Qdrant | Insert(ms) Chroma | Update(ms) Violas | Update(ms) Milvus | Update(ms) Qdrant | Update(ms) Chroma | Build(s) Violas | Build(s) Milvus | Build(s) Qdrant | Build(s) Chroma | Index update(s) Violas | Index update(s) Milvus | Index update(s) Qdrant | Index update(s) Chroma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| caltech | 0.20 | 2952.70 | 129.06 | 106.89 | 2.20 | 3117.64 | 117.35 | 119.84 | 0.421 | 8.383 | 4.345 | 4.040 | 0.444 | N/A | N/A | N/A |
| cub | 0.41 | 3094.57 | 127.69 | 111.35 | 1.40 | 8984.51 | 130.47 | 104.96 | 1.720 | 6.864 | 6.324 | 5.710 | 2.732 | N/A | N/A | N/A |
| coco | 0.21 | 2963.39 | 122.84 | 135.10 | 1.50 | 9152.77 | 109.59 | 182.36 | 0.440 | 7.082 | 5.207 | 5.091 | 0.438 | N/A | N/A | N/A |
| Three-image Avg. | 0.27 | 3003.55 | 126.53 | 117.78 | 1.70 | 7084.97 | 119.14 | 135.72 | 0.860 | 7.443 | 5.292 | 4.947 | 1.205 | N/A | N/A | N/A |

## Auxiliary maintenance diagnostics

| Dataset | Method | Delete 200 (ms) | Repeats | Peak RSS (MiB) | Index size (MiB) | Index state |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| caltech | Violas | 4.79 | 3 | 311.41 | N/A | records=N/A, indexed=N/A, segments=N/A |
| caltech | Milvus | 5460.99 | 3 | N/A | N/A | records=7966, indexed=N/A, segments=N/A |
| caltech | Qdrant | 9.27 | 3 | N/A | N/A | records=7766, indexed=0, segments=7 |
| caltech | Chroma | 176.89 | 3 | N/A | N/A | records=7766, indexed=N/A, segments=N/A |
| cub | Violas | 11.17 | 3 | 385.50 | N/A | records=N/A, indexed=N/A, segments=N/A |
| cub | Milvus | 1032.33 | 3 | N/A | N/A | records=10997, indexed=N/A, segments=N/A |
| cub | Qdrant | 17.04 | 3 | N/A | N/A | records=10597, indexed=0, segments=7 |
| cub | Chroma | 126.70 | 3 | N/A | N/A | records=10597, indexed=N/A, segments=N/A |
| coco | Violas | 10.73 | 3 | 318.48 | N/A | records=N/A, indexed=N/A, segments=N/A |
| coco | Milvus | 1034.22 | 3 | N/A | N/A | records=9367, indexed=N/A, segments=N/A |
| coco | Qdrant | 9.38 | 3 | N/A | N/A | records=8967, indexed=0, segments=7 |
| coco | Chroma | 128.84 | 3 | N/A | N/A | records=8967, indexed=N/A, segments=N/A |

## Measurement boundary

- 每次重复都从重新创建后端状态开始；主表报告算术平均值，JSON 保留每轮样本和样本标准差。
- Violas 的 index update 是数据更新后完整重建 HDMG。
- 外部数据库的同步 upsert 同时触发内部索引维护，但开源 Violas 未发布论文 Table 3 所用的独立 index-update 计时代码，因此该列保留 N/A。
- Milvus、Qdrant、Chroma 当前通过 Docker 服务访问；延迟包含本机进程间通信，与论文机器上的绝对值不应直接比较。
- 三个数据库客户端没有提供统一、可移植的单 collection 索引字节数，因此不虚构字节数。
