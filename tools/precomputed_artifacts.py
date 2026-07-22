"""Audit and export deterministic benchmark artifacts from precomputed TXT files.

This tool is intentionally outside ``violas_python``.  The open-source Python
implementation remains frozen and is only used later as a reference backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DATASET_FILES = {
    "news20": "news20_precomputed.txt",
    "ohsumed": "ohsumed_precomputed.txt",
    "yahoo": "yahoo_precomputed.txt",
    "caltech": "caltech_precomputed.txt",
    "cub": "cub_precomputed.txt",
    "coco": "coco_precomputed.txt",
}


@dataclass
class Folder:
    name: str
    key: str
    data_type: str
    descriptions: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    global_reps: dict[str, list[float]] = field(default_factory=dict)
    fold_reps: dict[str, list[tuple[list[float], list[float]]]] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_dimension(raw: str) -> int:
    return 0 if not raw else raw.count(",") + 1


def audit_file(dataset: str, path: Path) -> dict:
    folders: dict[str, dict] = {}
    current: dict | None = None
    dimensions: set[int] = set()
    global_rep_count = 0
    fold_rep_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("|")
            command = parts[0]
            if command == "FOLDER":
                if len(parts) < 3:
                    raise ValueError(f"{path}:{line_no}: malformed FOLDER")
                current = {
                    "folder": parts[1],
                    "key": parts[2],
                    "dataType": parts[3] if len(parts) > 3 else "DATA",
                    "vectors": 0,
                    "dimensions": set(),
                }
                if parts[1] in folders:
                    raise ValueError(f"{path}:{line_no}: duplicate folder {parts[1]!r}")
                folders[parts[1]] = current
            elif current is None:
                raise ValueError(f"{path}:{line_no}: {command} before FOLDER")
            elif command == "VECTOR":
                if len(parts) < 3:
                    raise ValueError(f"{path}:{line_no}: malformed VECTOR")
                dim = vector_dimension(parts[2])
                if dim <= 0:
                    raise ValueError(f"{path}:{line_no}: empty VECTOR")
                dimensions.add(dim)
                current["dimensions"].add(dim)
                current["vectors"] += 1
            elif command == "GLOBAL_REP":
                if len(parts) < 3:
                    raise ValueError(f"{path}:{line_no}: malformed GLOBAL_REP")
                dimensions.add(vector_dimension(parts[2]))
                global_rep_count += 1
            elif command == "FOLD_REP":
                if len(parts) < 5:
                    raise ValueError(f"{path}:{line_no}: malformed FOLD_REP")
                if parts[3]:
                    dimensions.add(vector_dimension(parts[3]))
                if parts[4]:
                    dimensions.add(vector_dimension(parts[4]))
                fold_rep_count += 1
            else:
                raise ValueError(f"{path}:{line_no}: unknown command {command!r}")

    folder_rows = []
    for name in sorted(folders):
        row = folders[name]
        dims = sorted(row.pop("dimensions"))
        if len(dims) > 1:
            raise ValueError(f"{path}: inconsistent dimensions in folder {name}: {dims}")
        row["dimension"] = dims[0] if dims else 0
        folder_rows.append(row)

    return {
        "dataset": dataset,
        "source": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "artifactLevel": "sample",
        "fullVerified": False,
        "folderCount": len(folder_rows),
        "vectorCount": sum(row["vectors"] for row in folder_rows),
        "dimensions": sorted(dimensions),
        "globalRepCount": global_rep_count,
        "foldRepCount": fold_rep_count,
        "folders": folder_rows,
        "note": "Current repository artifact; must not be labelled full until source/count/hash are independently verified.",
    }


def parse_vector(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",")]


def load_file(path: Path) -> list[Folder]:
    folders: list[Folder] = []
    current: Folder | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("|")
            command = parts[0]
            if command == "FOLDER":
                if len(parts) < 3:
                    raise ValueError(f"{path}:{line_no}: malformed FOLDER")
                current = Folder(parts[1], parts[2], parts[3] if len(parts) > 3 else "DATA")
                folders.append(current)
                continue
            if current is None:
                raise ValueError(f"{path}:{line_no}: {command} before FOLDER")
            if command == "VECTOR":
                current.descriptions.append(parts[1])
                current.vectors.append(parse_vector(parts[2]))
            elif command == "GLOBAL_REP":
                current.global_reps[parts[1]] = parse_vector(parts[2])
            elif command == "FOLD_REP":
                current.fold_reps.setdefault(parts[1], []).append(
                    (parse_vector(parts[3]), parse_vector(parts[4]))
                )
    return sorted(folders, key=lambda folder: folder.name)


def cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return max(0.0, 1.0 - dot / (left_norm * right_norm))


def record_id(dataset: str, folder: str, index: int) -> str:
    return f"{dataset}/{folder}/{index:08d}"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def export_artifact(
    dataset: str,
    input_path: Path,
    output_dir: Path,
    folds: int,
    rep_method: str,
    top_k: int,
    betas: list[float],
    max_queries: int,
) -> None:
    if folds < 2 or top_k <= 0 or max_queries < 0:
        raise ValueError("folds>=2, top_k>0 and max_queries>=0 are required")
    folders = load_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    by_id: dict[str, dict] = {}
    for folder in folders:
        for index, (description, vector) in enumerate(zip(folder.descriptions, folder.vectors)):
            row = {
                "recordId": record_id(dataset, folder.name, index),
                "folder": folder.name,
                "key": folder.key,
                "index": index,
                "description": description,
                "vector": vector,
            }
            records.append(row)
            by_id[row["recordId"]] = row
    write_jsonl(output_dir / "records.jsonl", records)

    queries: list[dict] = []
    ground_truth: list[dict] = []
    query_count = 0
    for fold in range(folds):
        training: list[dict] = []
        held_out: list[tuple[Folder, int, dict]] = []
        key_vectors: dict[str, list[float]] = {}
        query_key_vectors: dict[str, list[float]] = {}
        for folder in folders:
            reps = folder.fold_reps.get(rep_method, [])
            if fold >= len(reps):
                raise ValueError(f"missing {rep_method!r} fold {fold} for {folder.name}")
            train_rep, test_rep = reps[fold]
            key_vectors[folder.key] = train_rep
            query_key_vectors[folder.key] = test_rep
            fold_size = len(folder.vectors) // folds
            start = fold * fold_size
            end = start + fold_size if fold < folds - 1 else len(folder.vectors)
            for index in range(len(folder.vectors)):
                item = by_id[record_id(dataset, folder.name, index)]
                if start <= index < end:
                    held_out.append((folder, index, item))
                else:
                    training.append(item)

        for folder, index, item in held_out:
            if max_queries and query_count >= max_queries:
                break
            query_id = f"{dataset}/fold-{fold}/{folder.name}/{index:08d}"
            query_key = query_key_vectors[folder.key]
            queries.append(
                {
                    "queryId": query_id,
                    "fold": fold,
                    "sourceRecordId": item["recordId"],
                    "trueKey": folder.key,
                    "vector": item["vector"],
                    "keyVector": query_key,
                }
            )
            for beta in betas:
                scored: list[tuple[float, str]] = []
                for candidate in training:
                    emb = cosine_distance(item["vector"], candidate["vector"])
                    candidate_key_vector = key_vectors[candidate["key"]]
                    semantic = cosine_distance(query_key, candidate_key_vector)
                    score = beta * semantic + (1.0 - beta) * emb
                    scored.append((score, candidate["recordId"]))
                scored.sort(key=lambda pair: (pair[0], pair[1]))
                ground_truth.append(
                    {
                        "queryId": query_id,
                        "beta": beta,
                        "recordIds": [item_id for _, item_id in scored[:top_k]],
                    }
                )
            query_count += 1
        if max_queries and query_count >= max_queries:
            break

    write_jsonl(output_dir / "queries.jsonl", queries)
    write_jsonl(output_dir / "ground_truth.jsonl", ground_truth)
    manifest = audit_file(dataset, input_path)
    manifest["schemaVersion"] = 1
    manifest["protocol"] = "five-fold"
    manifest["recordIdFormat"] = "<dataset>/<folder>/<zero-based-index:08d>"
    manifest["folds"] = folds
    manifest["repMethod"] = rep_method
    manifest["topK"] = top_k
    manifest["betas"] = betas
    manifest["exportedQueries"] = len(queries)
    manifest["recordsSha256"] = sha256_file(output_dir / "records.jsonl")
    manifest["queriesSha256"] = sha256_file(output_dir / "queries.jsonl")
    manifest["groundTruthSha256"] = sha256_file(output_dir / "ground_truth.jsonl")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_betas(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",")]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("betas must be a non-empty comma-separated list in [0,1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--input-dir", type=Path, default=Path("dataset/precomputed"))
    audit_parser.add_argument("--output", type=Path, default=Path("manifests/current-artifacts.json"))

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--input-dir", type=Path, default=Path("dataset/precomputed"))
    export_parser.add_argument("--dataset", choices=sorted(DATASET_FILES), required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--folds", type=int, default=5)
    export_parser.add_argument("--rep-method", default="simple")
    export_parser.add_argument("--top-k", type=int, default=3)
    export_parser.add_argument("--betas", type=parse_betas, default=parse_betas("0.0,0.3,0.5,0.8,1.0"))
    export_parser.add_argument("--max-queries", type=int, default=20)

    args = parser.parse_args()
    if args.command == "audit":
        rows = []
        for dataset, filename in DATASET_FILES.items():
            path = args.input_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(audit_file(dataset, path))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"schemaVersion": 1, "datasets": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    else:
        export_artifact(
            args.dataset,
            args.input_dir / DATASET_FILES[args.dataset],
            args.output_dir,
            args.folds,
            args.rep_method,
            args.top_k,
            args.betas,
            args.max_queries,
        )
        print(f"wrote artifact to {args.output_dir}")


if __name__ == "__main__":
    main()
