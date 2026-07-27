# Entity-Centric Code Context Retrieval

This case study applies the four retrieval capabilities described by Violas to
source-code context retrieval:

- Entity-aligned retrieval (EAR)
- Diversity-driven retrieval (DDR)
- Relation-expanded retrieval (RER)
- Cross-modal pairing (CMP)

The goal is to evaluate whether one vector-group state can serve different
context requests without treating source files, tests, documentation, scripts,
and result artifacts as unrelated vectors.

## Data model

The dataset contains 24 repository objects organized under four entity keys:

| Entity key | Contents |
| --- | --- |
| `hdmg_index` | HDMG construction, traversal, configuration, tests, and parameter results |
| `vector_group` | storage operations, context lookup, relations, and tests |
| `benchmark_pipeline` | evaluation code, scripts, frozen results, and provenance |
| `mixed_ranking` | mixed scoring, retrieval, tests, and metric documentation |

Each object records its entity key, implementation aspect, modality, repository
location, symbol, and a short functional description. The modalities are
`code`, `test`, `document`, `script`, and `artifact`. Seventeen manually
verified relations describe calls, tests, index effects, documentation, and
generated artifacts.

Member descriptions, entity descriptions, and queries use normalized
384-dimensional `all-MiniLM-L6-v2` embeddings. All workloads read the same
Cangjie `VectorMap`.

## Workloads

| Workload | Query behavior | Baseline | Metric |
| --- | --- | --- | --- |
| EAR | Route an HDMG question to the owning entity before member ranking | global vector Top-5 | Entity Purity@5 |
| DDR | Select five requested implementation aspects within `hdmg_index` | entity-filtered vector Top-5 | Aspect Coverage@5 |
| RER | Recover the update, invalidation, search, rebuild, and test chain | entity-filtered vector Top-5 | Dependency Coverage@5 |
| CMP | Retrieve the evaluation artifact paired with a configured search API | nearest neighbor of the source object | Pair Hit@1 |

The baseline and Violas variants use identical member embeddings and return
the same number of objects. The four metrics measure different behaviors and
are therefore reported separately.

## Results

| Capability | Baseline | Violas |
| --- | ---: | ---: |
| EAR — Entity Purity@5 | 100% | 100% |
| DDR — Aspect Coverage@5 | 60% | 100% |
| RER — Dependency Coverage@5 | 40% | 100% |
| CMP — Pair Hit@1 | 0% | 100% |

EAR is unchanged because the query contains sufficiently distinctive HDMG
terms for the embedding model to route all five results correctly.

For DDR, vector Top-5 contains two configuration objects and an evaluation
document, leaving maintenance and construction uncovered. Selecting one member
from each requested aspect covers configuration, traversal, validation,
construction, and maintenance within the same context budget.

For RER, the vector baseline retrieves two of the five required objects.
Expansion from `_invalidateIndexes` follows `calls`, `affects`, `tested_by`,
and `rebuilds_with` relations to recover the complete lifecycle chain.

For CMP, the nearest vector to `searchHdmgWithConfig` is another search
implementation. The explicit `evaluation_artifact` relation instead returns
`hdmg-parameter-scan.md`, which is the associated evaluation document.

These observations show that entity membership, implementation aspects, and
explicit relations express retrieval constraints that are not equivalent to
embedding proximity. They do not establish an overall performance improvement
for a downstream coding agent.

## Implementation

The experiment has two entry points:

- `tools/run_code_context_case_study.py` locates repository objects, generates
  embeddings, writes the experiment input, invokes Cangjie, and freezes the
  result.
- `cj_core/src/examples/code_context_case.cj` loads one `VectorMap` and executes
  EAR, DDR, RER, and CMP.

Generated inputs and raw logs are written below ignored `artifacts/` and
`results/` directories. The tracked result is
[`results-summary/code-context-case-study.json`](../results-summary/code-context-case-study.json).

## Reproduction

From the repository root:

```powershell
python -m pip install -r tools\requirements-benchmark.txt
python tools\run_code_context_case_study.py
```

The script verifies repository locations at runtime and fails if a referenced
symbol or file is missing.

## Scope

This is a small functional case study rather than a general code-retrieval
benchmark. Entity assignments, aspects, relations, and relevance targets are
manually curated. Each group contains one member, the four queries are not a
held-out evaluation set, and the experiment does not measure patch success,
tool calls, or token usage. The small collection also uses exact vector search
rather than evaluating HDMG scalability.
