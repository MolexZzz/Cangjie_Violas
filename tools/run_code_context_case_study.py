"""Run a four-paradigm Violas case study over this code repository.

The case uses one managed VectorMap state to evaluate entity-aligned retrieval
(EAR), diversity-driven retrieval (DDR), relation-expanded retrieval (RER),
and cross-modal pairing (CMP). Generated vectors and raw logs stay in ignored
``artifacts/`` and ``results/`` directories; the frozen summary is tracked.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
from sentence_transformers import SentenceTransformer


ARTIFACT_DIR = ROOT / "artifacts" / "code-context-case-study"
RESULT_DIR = ROOT / "results" / "code-context-case-study"
INPUT_PATH = ARTIFACT_DIR / "case-study.tsv"
SUMMARY_PATH = RESULT_DIR / "summary.json"
RAW_OUTPUT_PATH = RESULT_DIR / "cangjie-output.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ProjectEntity:
    id: str
    key: str
    aspect: str
    modality: str
    path: str
    symbol: str
    kind: str
    description: str


@dataclass(frozen=True)
class ProjectKey:
    id: str
    description: str


@dataclass(frozen=True)
class Relation:
    key: str
    source: str
    target: str
    type: str


@dataclass(frozen=True)
class Query:
    id: str
    paradigm: str
    text: str
    target_key: str
    seed: str
    relation_type: str
    gold_ids: tuple[str, ...] = ()
    requested_aspects: tuple[str, ...] = ()


PROJECT_KEYS = (
    ProjectKey(
        "hdmg_index",
        "HDMG approximate graph index construction, traversal, candidate configuration, "
        "index invalidation, rebuilding, and correctness validation.",
    ),
    ProjectKey(
        "vector_group",
        "Vector group storage, entity membership, object CRUD, metadata, context, and explicit relations.",
    ),
    ProjectKey(
        "benchmark_pipeline",
        "Full benchmark execution, frozen results, artifact provenance, and reproducibility workflow.",
    ),
    ProjectKey(
        "mixed_ranking",
        "Beta-weighted entity alignment and embedding similarity scoring, reranking, and evaluation.",
    ),
)


ENTITIES = (
    ProjectEntity(
        "update_object", "hdmg_index", "maintenance", "code",
        "cj_core/src/storage/vectormap.cj", "updateObject", "function",
        "Update an object by stable ID and invalidate every derived search index.",
    ),
    ProjectEntity(
        "invalidate_indexes", "hdmg_index", "maintenance", "code",
        "cj_core/src/storage/vectormap.cj", "_invalidateIndexes", "function",
        "Advance the data version and mark HDMG and representative indexes stale.",
    ),
    ProjectEntity(
        "build_hdmg", "hdmg_index", "construction", "code",
        "cj_core/src/storage/vectormap.cj", "buildHdmg", "function",
        "Construct the hierarchical diversified micro-cluster graph and record its source version.",
    ),
    ProjectEntity(
        "search_hdmg", "hdmg_index", "traversal", "code",
        "cj_core/src/storage/vectormap.cj", "searchHdmg", "function",
        "Traverse HDMG using entity and embedding signals and rerank candidate members.",
    ),
    ProjectEntity(
        "search_with_config", "hdmg_index", "configuration", "code",
        "cj_core/src/storage/vectormap.cj", "searchHdmgWithConfig", "function",
        "Run HDMG retrieval with an explicit candidate-pool and traversal configuration.",
    ),
    ProjectEntity(
        "search_config", "hdmg_index", "configuration", "code",
        "cj_core/src/storage/hdmg.cj", "HdmgSearchConfig", "configuration",
        "Define maximum steps, cluster pool multiplier, top key candidates, and extra hops.",
    ),
    ProjectEntity(
        "lifecycle_test", "hdmg_index", "validation", "test",
        "cj_core/src/storage/storage_test.cj", "invalidatesAndRebuildsHdmgAfterMutation", "test",
        "Verify that mutation invalidates HDMG and rebuilding restores a current index.",
    ),
    ProjectEntity(
        "config_validation_test", "hdmg_index", "validation", "test",
        "cj_core/src/storage/storage_test.cj", "rejectsInvalidBuildAndSearchParameters", "test",
        "Verify rejection of invalid HDMG build and search budgets.",
    ),
    ProjectEntity(
        "parameter_scan_result", "hdmg_index", "evaluation", "document",
        "results-summary/hdmg-parameter-scan.md", "HDMG 准确率", "result",
        "Report the measured accuracy and latency of alternative HDMG candidate-pool settings.",
    ),
    ProjectEntity(
        "insert_group", "vector_group", "mutation", "code",
        "cj_core/src/storage/vectormap.cj", "public func insert(", "function",
        "Insert a vector group under an entity key and update stored type metadata.",
    ),
    ProjectEntity(
        "add_pair_relation", "vector_group", "relation", "code",
        "cj_core/src/storage/vectormap.cj", "addPairRelation", "function",
        "Store a typed pair relation between two compatible groups.",
    ),
    ProjectEntity(
        "get_paired_vectors", "vector_group", "relation", "code",
        "cj_core/src/storage/vectormap.cj", "getPairedVectors", "function",
        "Resolve the groups connected to a retrieved group through explicit pair relations.",
    ),
    ProjectEntity(
        "get_contextual_vectors", "vector_group", "context", "code",
        "cj_core/src/storage/vectormap.cj", "getContextualVectors", "function",
        "Recover neighboring ordered chunks using stored context identifiers.",
    ),
    ProjectEntity(
        "vector_group_test", "vector_group", "validation", "test",
        "cj_core/src/storage/storage_test.cj", "keepsVectorsAndDescriptionsAligned", "test",
        "Verify that vector-group mutations keep member vectors and descriptions aligned.",
    ),
    ProjectEntity(
        "architecture_doc", "vector_group", "documentation", "document",
        "docs/architecture.md", "架构与设计", "documentation",
        "Explain the vector-group data model, modules, mixed distance, and index lifecycle.",
    ),
    ProjectEntity(
        "paper_protocol", "benchmark_pipeline", "execution", "code",
        "cj_core/src/bench/paper_protocol.cj", "runPaperArtifact(", "function",
        "Execute the frozen Cangjie paper evaluation protocol over a prepared artifact.",
    ),
    ProjectEntity(
        "full_suite", "benchmark_pipeline", "execution", "script",
        "tools/run_image_full_suite.ps1", "run_image_full_suite", "tool",
        "Run the complete image benchmark suite and collect the backend outputs.",
    ),
    ProjectEntity(
        "final_results", "benchmark_pipeline", "result", "artifact",
        "results-summary/final-results.json", "final-results", "result",
        "Freeze recall, NDCG, latency, source hashes, dataset sizes, and experiment commit.",
    ),
    ProjectEntity(
        "release_manifest", "benchmark_pipeline", "provenance", "artifact",
        "manifests/release-artifacts.json", "release-artifacts", "manifest",
        "Record artifact locations, byte sizes, query counts, and SHA-256 hashes.",
    ),
    ProjectEntity(
        "reproducibility_doc", "benchmark_pipeline", "documentation", "document",
        "docs/reproducibility.md", "复现说明", "documentation",
        "Describe environment setup, verification, full-run commands, and result validation.",
    ),
    ProjectEntity(
        "mixed_distance", "mixed_ranking", "scoring", "code",
        "cj_core/src/storage/mixed_scoring.cj", "mixedDistance", "function",
        "Combine embedding distance and semantic entity distance using query weight beta.",
    ),
    ProjectEntity(
        "mixed_search", "mixed_ranking", "retrieval", "code",
        "cj_core/src/storage/vectormap.cj", "searchWithMixedKeyRepVec", "function",
        "Route groups and rank members with the beta-weighted mixed objective.",
    ),
    ProjectEntity(
        "mixed_scoring_test", "mixed_ranking", "validation", "test",
        "cj_core/src/storage/storage_test.cj", "clampsBetaAndCombinesDistances", "test",
        "Verify beta clamping and mixed semantic plus embedding distance calculation.",
    ),
    ProjectEntity(
        "experiments_doc", "mixed_ranking", "evaluation", "document",
        "docs/experiments.md", "评价指标", "documentation",
        "Define mixed Recall, mixed NDCG, full-query results, and measurement conditions.",
    ),
)


RELATIONS = (
    Relation("hdmg_index", "update_object", "invalidate_indexes", "calls"),
    Relation("hdmg_index", "invalidate_indexes", "search_hdmg", "affects"),
    Relation("hdmg_index", "search_hdmg", "build_hdmg", "rebuilds_with"),
    Relation("hdmg_index", "invalidate_indexes", "lifecycle_test", "tested_by"),
    Relation("hdmg_index", "search_config", "search_with_config", "used_by"),
    Relation("hdmg_index", "search_config", "config_validation_test", "tested_by"),
    Relation("hdmg_index", "search_with_config", "parameter_scan_result", "evaluation_artifact"),
    Relation("vector_group", "insert_group", "vector_group_test", "tested_by"),
    Relation("vector_group", "add_pair_relation", "get_paired_vectors", "read_by"),
    Relation("vector_group", "add_pair_relation", "architecture_doc", "documented_by"),
    Relation("benchmark_pipeline", "paper_protocol", "full_suite", "invoked_by"),
    Relation("benchmark_pipeline", "full_suite", "final_results", "produces"),
    Relation("benchmark_pipeline", "final_results", "release_manifest", "uses_artifact"),
    Relation("benchmark_pipeline", "paper_protocol", "reproducibility_doc", "documented_by"),
    Relation("mixed_ranking", "mixed_distance", "mixed_search", "used_by"),
    Relation("mixed_ranking", "mixed_distance", "mixed_scoring_test", "tested_by"),
    Relation("mixed_ranking", "mixed_search", "experiments_doc", "documented_by"),
)


QUERIES = (
    Query(
        "entity_route", "EAR",
        "Which subsystem owns HDMG graph construction, traversal, stale-index rebuilding, "
        "candidate configuration, and its correctness tests?",
        "hdmg_index", "-", "-",
    ),
    Query(
        "balanced_hdmg_context", "DDR",
        "Build a balanced HDMG context package covering maintenance, construction, "
        "traversal, configuration, and validation.",
        "hdmg_index", "-", "-",
        requested_aspects=("maintenance", "construction", "traversal", "configuration", "validation"),
    ),
    Query(
        "index_dependency_chain", "RER",
        "Explain the implementation and test chain by which an object update makes HDMG stale "
        "and causes a later search to rebuild it.",
        "hdmg_index", "invalidate_indexes", "*",
        gold_ids=("update_object", "invalidate_indexes", "search_hdmg", "build_hdmg", "lifecycle_test"),
    ),
    Query(
        "code_to_evaluation", "CMP",
        "Given the configured HDMG search API, retrieve its paired natural-language evaluation artifact.",
        "hdmg_index", "search_with_config", "evaluation_artifact",
        gold_ids=("parameter_scan_result",),
    ),
)


def source_line(entity: ProjectEntity) -> int:
    path = ROOT / entity.path
    if not path.exists():
        raise FileNotFoundError(f"case-study entity path is missing: {entity.path}")
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if entity.symbol in line:
            return number
    if entity.modality in {"document", "artifact", "script"}:
        return 1
    raise ValueError(f"symbol not found in {entity.path}: {entity.symbol}")


def clean(text: str) -> str:
    return " ".join(text.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def vector_text(vector: np.ndarray) -> str:
    return ",".join(f"{float(value):.8f}" for value in vector)


def load_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except OSError:
        return SentenceTransformer(MODEL_NAME)


def build_input() -> dict[str, object]:
    model = load_model()
    entity_texts = [
        f"{entity.kind} {entity.symbol}. {entity.description} File {entity.path}."
        for entity in ENTITIES
    ]
    key_texts = [key.description for key in PROJECT_KEYS]
    query_texts = [query.text for query in QUERIES]
    embeddings = model.encode(
        entity_texts + key_texts + query_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    entity_end = len(ENTITIES)
    key_end = entity_end + len(PROJECT_KEYS)
    entity_vectors = embeddings[:entity_end]
    key_vectors = embeddings[entity_end:key_end]
    query_vectors = embeddings[key_end:]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"META\tcode-context-four-paradigm-v2\t{embeddings.shape[1]}"]
    locations: dict[str, int] = {}
    for entity, vector in zip(ENTITIES, entity_vectors, strict=True):
        line = source_line(entity)
        locations[entity.id] = line
        lines.append(
            "\t".join(
                (
                    "ENTITY", entity.id, entity.key, entity.aspect, entity.modality,
                    entity.path, str(line), entity.kind, clean(entity.description),
                    vector_text(vector),
                )
            )
        )
    for key, vector in zip(PROJECT_KEYS, key_vectors, strict=True):
        lines.append("\t".join(("KEY", key.id, clean(key.description), vector_text(vector))))
    for relation in RELATIONS:
        lines.append(
            "\t".join(("REL", relation.key, relation.source, relation.target, relation.type))
        )
    for query, vector in zip(QUERIES, query_vectors, strict=True):
        lines.append(
            "\t".join(
                (
                    "QUERY", query.id, query.paradigm, clean(query.text), query.target_key,
                    query.seed, query.relation_type, ",".join(query.gold_ids),
                    ",".join(query.requested_aspects), vector_text(vector),
                )
            )
        )
    INPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "model": MODEL_NAME,
        "dimension": int(embeddings.shape[1]),
        "projectKeys": len(PROJECT_KEYS),
        "entities": len(ENTITIES),
        "relations": len(RELATIONS),
        "queries": len(QUERIES),
        "locations": locations,
    }


def parse_cangjie_output(stdout: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in stdout.splitlines():
        marker = raw.find("CODE_CASE|RESULT|")
        if marker < 0:
            continue
        raw = raw[marker:]
        parts = raw.split("|")
        if len(parts) != 10:
            raise ValueError(f"invalid Cangjie result line: {raw}")
        rows.append(
            {
                "query": parts[2],
                "paradigm": parts[3],
                "method": parts[4],
                "metric": parts[5],
                "ranked": [item for item in parts[6].split(",") if item],
                "hits": int(parts[7]),
                "total": int(parts[8]),
                "score": float(parts[9]),
            }
        )
    expected_rows = len(QUERIES) * 2
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} Cangjie result rows, got {len(rows)}")
    return rows


def main() -> int:
    metadata = build_input()
    relative_input = INPUT_PATH.relative_to(ROOT).as_posix()
    env = os.environ.copy()
    env.setdefault("cjHeapSize", "1GB")
    completed = subprocess.run(
        ["cjpm", "run"],
        cwd=ROOT / "cj_core",
        input=f"codecase ../{relative_input}\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_PATH.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Cangjie code case failed ({completed.returncode}); see {RAW_OUTPUT_PATH}")

    rows = parse_cangjie_output(completed.stdout)
    paradigm_summary: dict[str, dict[str, float]] = {}
    for row in rows:
        paradigm_summary.setdefault(str(row["paradigm"]), {})[str(row["method"])] = float(row["score"])
    payload = {
        "schemaVersion": 2,
        "protocol": "code-context-four-paradigm-v2",
        **metadata,
        "results": rows,
        "paradigmSummary": paradigm_summary,
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
