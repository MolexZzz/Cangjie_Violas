# Average image-dataset paper-style comparison

Average over Caltech-101, CUB-200-2011, and COCO. Each dataset contributes equally; each run uses its complete 10% test pool.

<table>
  <thead>
    <tr><th rowspan="2">β</th><th colspan="5">Mixed Recall@3</th><th colspan="5">Mixed NDCG@3</th><th colspan="5">Latency (ms/query)</th></tr>
    <tr><th>Violas</th><th>w/o HDMG</th><th>Milvus</th><th>Qdrant</th><th>Chroma</th><th>Violas</th><th>w/o HDMG</th><th>Milvus</th><th>Qdrant</th><th>Chroma</th><th>Violas</th><th>w/o HDMG</th><th>Milvus</th><th>Qdrant</th><th>Chroma</th></tr>
  </thead>
  <tbody>
    <tr><td>0.0</td><td>0.911</td><td>0.793</td><td>0.977</td><td><strong>1.000</strong></td><td>1.000</td><td>0.998</td><td>0.992</td><td>1.000</td><td><strong>1.000</strong></td><td>1.000</td><td><strong>0.77</strong></td><td>1.48</td><td>6.10</td><td>25.22</td><td>15.10</td></tr>
    <tr><td>0.1</td><td><strong>0.954</strong></td><td>0.910</td><td>0.831</td><td>0.837</td><td>0.837</td><td><strong>0.999</strong></td><td>0.997</td><td>0.997</td><td>0.997</td><td>0.997</td><td><strong>1.01</strong></td><td>1.52</td><td>6.12</td><td>25.47</td><td>14.97</td></tr>
    <tr><td>0.2</td><td><strong>0.970</strong></td><td>0.963</td><td>0.752</td><td>0.755</td><td>0.755</td><td><strong>0.999</strong></td><td>0.999</td><td>0.990</td><td>0.990</td><td>0.990</td><td><strong>0.93</strong></td><td>1.41</td><td>6.11</td><td>25.42</td><td>14.96</td></tr>
    <tr><td>0.3</td><td>0.978</td><td><strong>0.986</strong></td><td>0.715</td><td>0.716</td><td>0.716</td><td>0.999</td><td><strong>0.999</strong></td><td>0.981</td><td>0.982</td><td>0.982</td><td><strong>0.96</strong></td><td>1.44</td><td>6.08</td><td>25.32</td><td>15.01</td></tr>
    <tr><td>0.4</td><td><strong>0.997</strong></td><td>0.994</td><td>0.701</td><td>0.702</td><td>0.702</td><td><strong>1.000</strong></td><td>1.000</td><td>0.973</td><td>0.973</td><td>0.973</td><td><strong>1.15</strong></td><td>1.28</td><td>6.06</td><td>25.48</td><td>14.98</td></tr>
    <tr><td>0.5</td><td><strong>0.998</strong></td><td>0.998</td><td>0.694</td><td>0.695</td><td>0.695</td><td><strong>1.000</strong></td><td>1.000</td><td>0.964</td><td>0.964</td><td>0.964</td><td><strong>1.14</strong></td><td>1.30</td><td>6.08</td><td>25.34</td><td>14.93</td></tr>
    <tr><td>0.6</td><td>0.999</td><td><strong>0.999</strong></td><td>0.692</td><td>0.693</td><td>0.693</td><td>1.000</td><td><strong>1.000</strong></td><td>0.955</td><td>0.955</td><td>0.955</td><td><strong>1.01</strong></td><td>1.19</td><td>6.07</td><td>25.29</td><td>14.96</td></tr>
    <tr><td>0.7</td><td><strong>0.999</strong></td><td><strong>0.999</strong></td><td>0.692</td><td>0.693</td><td>0.693</td><td><strong>1.000</strong></td><td><strong>1.000</strong></td><td>0.947</td><td>0.947</td><td>0.947</td><td><strong>1.00</strong></td><td>1.17</td><td>6.03</td><td>25.84</td><td>15.03</td></tr>
    <tr><td>0.8</td><td><strong>0.999</strong></td><td><strong>0.999</strong></td><td>0.692</td><td>0.693</td><td>0.693</td><td><strong>1.000</strong></td><td><strong>1.000</strong></td><td>0.939</td><td>0.939</td><td>0.939</td><td><strong>0.89</strong></td><td>1.01</td><td>6.05</td><td>25.71</td><td>15.05</td></tr>
    <tr><td>0.9</td><td><strong>0.999</strong></td><td><strong>0.999</strong></td><td>0.692</td><td>0.693</td><td>0.693</td><td><strong>1.000</strong></td><td><strong>1.000</strong></td><td>0.931</td><td>0.931</td><td>0.931</td><td><strong>0.91</strong></td><td>1.04</td><td>6.08</td><td>25.28</td><td>14.95</td></tr>
    <tr><td>1.0</td><td><strong>1.000</strong></td><td><strong>1.000</strong></td><td>0.693</td><td>0.693</td><td>0.693</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td><strong>1.00</strong></td><td>1.37</td><td>6.08</td><td>25.45</td><td>15.10</td></tr>
  </tbody>
</table>

Notes:

- Violas is the Cangjie HDMG result. `w/o HDMG` first routes entity groups by entity score and directly searches micro-clusters inside those groups.
- In the main paper-style table, Milvus, Qdrant, and Chroma rank instance embeddings only and directly return vector Top-3, without entity representations or local mixed reranking.
- NDCG uses mixed score as the graded gain, following Equation 14.
- NDCG at β=1.0 is shown as `—`, matching the paper table, because all records under the winning semantic key are tied.
- This is a three-image-dataset average, not the paper's six-dataset average.

## Auxiliary database two-stage mixed-rerank results

Each database first retrieves 30 candidates by embedding similarity, then the benchmark locally reranks them by mixed score. These enhanced results are not used in the main paper-style table.

| β | Milvus Recall | Qdrant Recall | Chroma Recall | Milvus NDCG | Qdrant NDCG | Chroma NDCG | Milvus latency | Qdrant latency | Chroma latency |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.977 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7.76 | 27.03 | 17.90 |
| 0.1 | 0.978 | 0.998 | 0.998 | 1.000 | 1.000 | 1.000 | 7.83 | 27.18 | 17.92 |
| 0.2 | 0.965 | 0.983 | 0.983 | 0.999 | 1.000 | 1.000 | 7.82 | 27.52 | 18.15 |
| 0.3 | 0.952 | 0.969 | 0.969 | 0.998 | 0.999 | 0.999 | 7.76 | 27.26 | 17.96 |
| 0.4 | 0.944 | 0.961 | 0.961 | 0.997 | 0.998 | 0.998 | 7.76 | 27.31 | 17.95 |
| 0.5 | 0.939 | 0.956 | 0.955 | 0.996 | 0.997 | 0.997 | 7.79 | 27.01 | 18.01 |
| 0.6 | 0.937 | 0.954 | 0.953 | 0.995 | 0.996 | 0.996 | 7.77 | 27.50 | 17.86 |
| 0.7 | 0.937 | 0.954 | 0.953 | 0.994 | 0.995 | 0.995 | 7.76 | 27.40 | 17.91 |
| 0.8 | 0.937 | 0.954 | 0.953 | 0.993 | 0.994 | 0.994 | 7.79 | 26.98 | 17.88 |
| 0.9 | 0.937 | 0.954 | 0.953 | 0.992 | 0.993 | 0.993 | 7.73 | 26.88 | 17.96 |
| 1.0 | 0.951 | 0.954 | 0.953 | 0.991 | 0.992 | 0.992 | 7.78 | 26.96 | 18.04 |

Source: `results/python-paper-90-10/2026-07-23-table2-v5-service/summary.json`
