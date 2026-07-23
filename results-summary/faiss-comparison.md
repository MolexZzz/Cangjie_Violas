# 仓颉 Violas 与 Faiss 检索实现对比

## 可比代码范围

| 实现 | 主实现文件数 | 主实现源码行 | 本列包含 | 未计入的共享依赖 |
| --- | ---: | ---: | --- | --- |
| 仓颉 Violas HDMG/mixed | 3 | 1453 | HDMG 构建/遍历、w/o HDMG、候选重排和 mixed score | VectorGroup、向量距离工具、KMeans 聚类 |
| Faiss IndexFlatIP | 4 | 1204 | Flat 精确索引及其连续向量编码存储 | Index 基类、公共距离计算和 SIMD 内核 |
| Faiss IndexIVFFlat | 4 | 1980 | IVF 基础流程和 IVFFlat 编码/扫描 | Flat 量化器、Clustering、InvertedLists、距离内核 |
| Faiss IndexHNSWFlat | 4 | 2680 | HNSW 索引封装、建图和图搜索 | IndexFlat 向量存储、公共距离计算和 SIMD 内核 |

这里统计的是本实验实际使用索引的主实现模块，而不是整个仓库。所有数字均为去除空行和纯注释后的源码行。公共依赖不重复计入某一种索引，因此这些数字用于比较实现规模，不表示完整可独立编译的代码量。

仓颉 `vectormap.cj` 同时包含部分 CRUD、关系和上下文接口；Faiss 的索引文件也包含同系列变体。因此主实现源码行仍是模块级近似值，不应被解释为算法复杂度或代码质量排名。Faiss 使用 C++，表中同时统计 `.h` 接口声明和 `.cpp` 实现；仓颉没有同样的头文件分离方式。

## 功能边界

| 项目 | 仓颉 Violas HDMG | Faiss Flat / IVF / HNSW |
| --- | --- | --- |
| 检索目标 | β 加权的语义距离与 embedding 距离 | embedding 向量相似度 |
| 候选获取 | 实体路由、微簇和 HDMG 图遍历 | 精确扫描或 ANN 索引 |
| Mixed score | 原生完成候选 mixed 重排 | 索引本身不支持，需应用层增加 |
| 动态维护 | 数据变化后使 HDMG 失效并重建 | 能力随索引而异，统一实验采用重建 |
| 当前向量精度 | Float64 | Float32 |

## 实验性能

### 仓颉 Violas：β=0.5 的三图像数据集平均 Mixed Search

| Method | Mixed Recall@3 | Mixed NDCG@3 | Mean latency (ms/query) |
| --- | ---: | ---: | ---: |
| Violas | 0.9983 | 1.0000 | 1.137 |
| w/o HDMG | 0.9980 | 1.0000 | 1.300 |

### Faiss：三图像数据集平均纯 embedding cosine 检索

| Index | Recall@3 | NDCG@3 | Mean latency (ms/query) | Build (ms) | Index (MiB) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IndexFlatIP | 1.0000 | 1.0000 | 1.675 | 10.66 | 17.79 | 499.34 |
| IndexIVFFlat | 0.9890 | 0.9920 | 0.179 | 138.72 | 17.99 | 530.67 |
| IndexHNSWFlat | 0.9946 | 0.9960 | 0.086 | 316.13 | 19.04 | 538.03 |

Violas 表使用 mixed ground truth，Faiss 表使用纯 embedding exact ground truth，二者目标函数不同，不能把两张表的 Recall 或 latency 直接解释成同一算法的胜负。Faiss 在这里承担底层向量索引基线；严格端到端比较需要让 Faiss 召回候选后使用相同 mixed score 重排。

## 主实现文件清单

### 仓颉 Violas HDMG/mixed

- `cj_core/src/storage/vectormap.cj`
- `cj_core/src/storage/hdmg.cj`
- `cj_core/src/storage/mixed_scoring.cj`

### Faiss IndexFlatIP

- `faiss/IndexFlat.cpp`
- `faiss/IndexFlat.h`
- `faiss/IndexFlatCodes.cpp`
- `faiss/IndexFlatCodes.h`

### Faiss IndexIVFFlat

- `faiss/IndexIVF.cpp`
- `faiss/IndexIVF.h`
- `faiss/IndexIVFFlat.cpp`
- `faiss/IndexIVFFlat.h`

### Faiss IndexHNSWFlat

- `faiss/IndexHNSW.cpp`
- `faiss/IndexHNSW.h`
- `faiss/impl/HNSW.cpp`
- `faiss/impl/HNSW.h`

## 可复现信息

- 仓颉实验 commit：`44e14ef10a9380de9011d5b46b9bedfad9bb093f`
- Faiss commit：`7d4bb39f7eb3e9bb4d160aa38ec821ee1a407afc`
- LOC 规则：`non-empty after removing comment-only content`
- 数据集：Caltech-101、CUB-200-2011、COCO
- Faiss 重复次数：3；预热查询：20
