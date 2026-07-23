"""Compare candidate NDCG definitions against frozen image artifacts.

This is a read-only diagnostic. It evaluates the exact embedding Top-3 baseline
against the beta-specific exact mixed Top-3 stored in ground_truth.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


BETAS = tuple(round(value / 10, 1) for value in range(10))
DISCOUNTS = np.asarray([1.0 / math.log2(rank + 2) for rank in range(3)], dtype=np.float64)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def binary_ndcg(ideal_ids: list[str], actual_ids: list[str]) -> float:
    relevant = set(ideal_ids[:3])
    dcg = sum(DISCOUNTS[rank] for rank, item_id in enumerate(actual_ids[:3]) if item_id in relevant)
    return float(dcg / DISCOUNTS.sum())


def graded_ndcg(
    ideal_indices: list[int],
    actual_indices: list[int],
    embedding_scores: np.ndarray,
    entity_scores: np.ndarray,
    beta: float,
    shifted_cosine: bool,
) -> float:
    def dcg(indices: list[int]) -> float:
        emb = embedding_scores[indices].astype(np.float64)
        ent = entity_scores[indices].astype(np.float64)
        if shifted_cosine:
            emb = (emb + 1.0) / 2.0
            ent = (ent + 1.0) / 2.0
        else:
            emb = np.clip(emb, 0.0, 1.0)
            ent = np.clip(ent, 0.0, 1.0)
        relevance = beta * ent + (1.0 - beta) * emb
        gains = np.power(2.0, relevance) - 1.0
        return float(np.sum(gains * DISCOUNTS[: len(indices)]))

    ideal = dcg(ideal_indices[:3])
    return dcg(actual_indices[:3]) / ideal if ideal else 0.0


def diagnose(artifact: Path) -> None:
    all_records = load_jsonl(artifact / "records.jsonl")
    split = load_jsonl(artifact / "splits.jsonl")[0]
    train_ids = set(split["trainRecordIds"])
    records = [row for row in all_records if row["recordId"] in train_ids]
    queries = load_jsonl(artifact / "queries.jsonl")
    ground_truth_rows = load_jsonl(artifact / "ground_truth.jsonl")
    key_rows = load_jsonl(artifact / "key_vectors.jsonl")

    record_ids = [row["recordId"] for row in records]
    id_to_index = {item_id: index for index, item_id in enumerate(record_ids)}
    record_keys = [row["key"] for row in records]
    record_vectors = normalize(np.asarray([row["vector"] for row in records], dtype=np.float32))
    query_vectors = normalize(np.asarray([row["vector"] for row in queries], dtype=np.float32))
    key_vectors = {
        row.get("key", row.get("name")): normalize(np.asarray([row["vector"]], dtype=np.float32))[0]
        for row in key_rows
    }
    ground_truth = {
        (row["queryId"], round(float(row["beta"]), 1)): row["recordIds"]
        for row in ground_truth_rows
    }

    totals = {
        beta: {"recall": [], "binary": [], "graded_clamp": [], "graded_shift": []}
        for beta in BETAS
    }
    batch_size = 64
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        similarities = query_vectors[start:stop] @ record_vectors.T
        for offset, query in enumerate(queries[start:stop]):
            embedding_scores = similarities[offset]
            raw_indices = np.argpartition(-embedding_scores, 2)[:3]
            raw_indices = raw_indices[np.argsort(-embedding_scores[raw_indices], kind="stable")].tolist()
            raw_ids = [record_ids[index] for index in raw_indices]
            query_key = normalize(np.asarray([query["keyVector"]], dtype=np.float32))[0]
            entity_by_key = {key: float(query_key @ vector) for key, vector in key_vectors.items()}
            entity_scores = np.asarray([entity_by_key[key] for key in record_keys], dtype=np.float32)

            for beta in BETAS:
                ideal_ids = ground_truth[(query["queryId"], beta)]
                ideal_indices = [id_to_index[item_id] for item_id in ideal_ids]
                totals[beta]["recall"].append(len(set(ideal_ids) & set(raw_ids)) / 3.0)
                totals[beta]["binary"].append(binary_ndcg(ideal_ids, raw_ids))
                totals[beta]["graded_clamp"].append(
                    graded_ndcg(ideal_indices, raw_indices, embedding_scores, entity_scores, beta, False)
                )
                totals[beta]["graded_shift"].append(
                    graded_ndcg(ideal_indices, raw_indices, embedding_scores, entity_scores, beta, True)
                )

    print(f"dataset={artifact.name} queries={len(queries)} records={len(records)}")
    print("beta  recall  binary_ndcg  graded_clamp  graded_shift")
    for beta in BETAS:
        row = totals[beta]
        print(
            f"{beta:>3.1f}  {np.mean(row['recall']):.4f}  {np.mean(row['binary']):.4f}"
            f"       {np.mean(row['graded_clamp']):.4f}        {np.mean(row['graded_shift']):.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    diagnose(args.artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
