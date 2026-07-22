"""Build and read the frozen ``python-paper-90-10`` benchmark artifact.

The artifact is the experiment boundary: Cangjie, Faiss and external database
adapters consume it and must never perform their own train/query split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from precomputed_artifacts import DATASET_FILES, Folder, load_file, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "python-paper-90-10"
SCHEMA_VERSION = 1
DEFAULT_MODEL = "ViT-B/32"
DEFAULT_PROMPT = "a photo of a {}"


@dataclass
class SourceRecord:
    folder: str
    key: str
    source_path: str
    vector: list[float]


def normalize(vector: list[float]) -> list[float]:
    row = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(row))
    return (row / norm if norm else row).tolist()


def stable_record_id(dataset: str, folder: str, source_path: str) -> str:
    normalized_path = source_path.replace("\\", "/")
    identity = f"{dataset}\0{folder}\0{normalized_path}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{dataset}/{folder}/{suffix}"


def jsonl_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_precomputed_source(dataset: str, path: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for folder in load_file(path):
        for description, vector in zip(folder.descriptions, folder.vectors):
            records.append(SourceRecord(folder.name, folder.name, str(description), vector))
    return records


def load_folder_images(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "BACKGROUND_Google"):
        for image in sorted(path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            rows.append((folder.name, image))
    return rows


def clip_checkpoint(model_name: str) -> Path | None:
    configured = os.environ.get("CLIP_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CLIP_MODEL_PATH does not exist: {path}")
        return path
    local = ROOT / "model" / "clip" / "ViT-B-32.pt"
    if model_name == DEFAULT_MODEL and local.is_file():
        return local.resolve()
    return None


def load_clip(model_name: str):
    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = clip_checkpoint(model_name)
    model_ref = str(checkpoint) if checkpoint else model_name
    download_root = os.environ.get("CLIP_DOWNLOAD_ROOT", str(ROOT / "model" / "clip"))
    model, preprocess = clip.load(model_ref, device=device, download_root=download_root)
    if checkpoint is None:
        checkpoint = clip_checkpoint(model_name)
    return clip, torch, device, model, preprocess, checkpoint


def clip_model_provenance(model_name: str) -> dict:
    try:
        import clip
        checkpoint = clip_checkpoint(model_name)
        if checkpoint is None:
            model_urls = getattr(clip, "_MODELS", None) or getattr(clip.clip, "_MODELS", {})
            url = model_urls.get(model_name)
            root = Path(os.environ.get("CLIP_DOWNLOAD_ROOT", ROOT / "model" / "clip"))
            checkpoint = root / Path(url).name if url else None
        return {
            "name": model_name,
            "package": "openai-clip",
            "packageVersion": metadata.version("clip"),
            "checkpointFile": checkpoint.name if checkpoint and checkpoint.is_file() else None,
            "checkpointSha256": sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None,
        }
    except (ImportError, ModuleNotFoundError, metadata.PackageNotFoundError):
        return {"name": model_name, "package": "openai-clip", "packageVersion": None,
                "checkpointFile": None, "checkpointSha256": None}


def encode_images(
    rows: list[tuple[str, Path]],
    root: Path,
    model_name: str,
    cache_path: Path,
    batch_size: int,
) -> list[SourceRecord]:
    from PIL import Image

    _, torch, device, model, preprocess, checkpoint = load_clip(model_name)
    checkpoint_hash = sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    expected_meta = {
        "schemaVersion": 1,
        "model": model_name,
        "checkpointSha256": checkpoint_hash,
        "dimension": 512,
    }
    cached: dict[str, SourceRecord] = {}
    if cache_path.exists():
        if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != expected_meta:
            raise ValueError(f"embedding cache provenance mismatch: {cache_path}")
        for item in jsonl_rows(cache_path):
            cached[item["sourcePath"]] = SourceRecord(
                item["folder"], item["key"], item["sourcePath"], item["vector"]
            )
    else:
        meta_path.write_text(json.dumps(expected_meta, indent=2) + "\n", encoding="utf-8")

    pending = [
        (folder, path, path.relative_to(root).as_posix())
        for folder, path in rows
        if path.relative_to(root).as_posix() not in cached
    ]
    print(
        f"CLIP embedding: total={len(rows)}, cached={len(rows) - len(pending)}, "
        f"pending={len(pending)}, device={device}, batch={batch_size}",
        flush=True,
    )
    with cache_path.open("a", encoding="utf-8", newline="\n") as cache:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            tensors = []
            for _, path, _ in batch:
                with Image.open(path) as image:
                    tensors.append(preprocess(image).unsqueeze(0))
            images = torch.cat(tensors, dim=0).to(device)
            with torch.no_grad():
                vectors = model.encode_image(images).cpu().numpy().astype(np.float32)
            for (folder, _, relative), vector in zip(batch, vectors):
                record = SourceRecord(folder, folder, relative, vector.tolist())
                cached[relative] = record
                cache.write(json.dumps({
                    "folder": record.folder,
                    "key": record.key,
                    "sourcePath": record.source_path,
                    "vector": record.vector,
                }, separators=(",", ":")) + "\n")
            cache.flush()
            completed = min(offset + len(batch), len(pending))
            if completed % (batch_size * 10) == 0 or completed == len(pending):
                print(f"  encoded {completed}/{len(pending)} pending images", flush=True)

    output: list[SourceRecord] = []
    for folder, path in rows:
        relative = path.relative_to(root).as_posix()
        record = cached[relative]
        if record.folder != folder:
            record = SourceRecord(folder, folder, relative, record.vector)
        output.append(record)
    return output


def encode_coco_json(
    root: Path,
    json_path: Path,
    model_name: str,
    cache_path: Path,
    batch_size: int,
) -> tuple[list[SourceRecord], list[Path]]:
    categories = sorted([
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("the paper COCO JSON must be a list")
    rows: list[tuple[str, Path]] = []
    used_paths: list[Path] = [json_path]
    for item in payload:
        relative = item.get("path") or item.get("file_name")
        if not relative:
            continue
        image_path = root / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO JSON references missing image: {image_path}")
        rows.append(("", image_path))
        used_paths.append(image_path)
    raw = encode_images(rows, root, model_name, cache_path, batch_size)
    key_vectors = encode_text_keys(categories, model_name, DEFAULT_PROMPT)
    text_matrix = np.asarray([key_vectors[key] for key in categories], dtype=np.float32)
    output: list[SourceRecord] = []
    for item in raw:
        vector = np.asarray(normalize(item.vector), dtype=np.float32)
        category = categories[int(np.argmax(vector @ text_matrix.T))]
        output.append(SourceRecord(category, category, item.source_path, item.vector))
    return output, used_paths


def encode_text_keys(keys: list[str], model_name: str, prompt: str) -> dict[str, list[float]]:
    clip, torch, device, model, _, _ = load_clip(model_name)
    tokens = clip.tokenize([prompt.format(key) for key in keys]).to(device)
    with torch.no_grad():
        vectors = model.encode_text(tokens)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    return {key: vectors[index].cpu().numpy().tolist() for index, key in enumerate(keys)}


def representative_key_vectors(records: list[SourceRecord]) -> dict[str, list[float]]:
    grouped: dict[str, list[list[float]]] = {}
    for record in records:
        grouped.setdefault(record.key, []).append(record.vector)
    return {key: normalize(np.mean(np.asarray(rows, dtype=np.float32), axis=0).tolist()) for key, rows in grouped.items()}


def split_ids(records: list[dict], seed: int, test_size: float) -> tuple[list[str], list[str]]:
    by_folder: dict[str, list[str]] = {}
    for record in records:
        by_folder.setdefault(record["folder"], []).append(record["recordId"])
    training: list[str] = []
    queries: list[str] = []
    for folder in sorted(by_folder):
        ids = by_folder[folder]
        if len(ids) < 2:
            training.extend(ids)
            continue
        train, test = train_test_split(ids, test_size=test_size, random_state=seed, shuffle=True)
        training.extend(train)
        queries.extend(test)
    return training, queries


def freeze_microclusters(train_records: list[dict], alpha: float = 0.5) -> list[dict]:
    """Freeze the exact sklearn clustering used by the Python paper implementation."""
    by_folder: dict[str, list[dict]] = {}
    for record in train_records:
        by_folder.setdefault(record["folder"], []).append(record)
    output: list[dict] = []
    for folder in sorted(by_folder):
        rows = by_folder[folder]
        count = len(rows)
        clusters = max(1, min(count, round(count ** (1.0 - alpha))))
        if clusters == 1 or count < 5:
            labels = np.zeros(count, dtype=np.int64)
        else:
            matrix = np.asarray([row["vector"] for row in rows], dtype=np.float32)
            labels = KMeans(
                n_clusters=clusters, random_state=42, n_init=10
            ).fit_predict(matrix)
        # Python creates subgroups by sorted label and numbers only non-empty labels.
        label_order = {label: index + 1 for index, label in enumerate(sorted(set(labels.tolist())))}
        for record, label in zip(rows, labels.tolist()):
            suffix = label_order[label]
            output.append({
                "recordId": record["recordId"],
                "baseKey": record["key"],
                "clusterKey": f"{record['key']}-{suffix:04d}",
                "clusterGroup": f"{folder}_cluster_{suffix:04d}",
            })
    return output


def cosine_distance(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    return 1.0 if not an or not bn else max(0.0, 1.0 - float(np.dot(a, b) / (an * bn)))


def tree_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def build_artifact(
    dataset: str,
    source_records: list[SourceRecord],
    output_dir: Path,
    source: dict,
    key_vector_source: str,
    model_name: str,
    prompt: str,
    seed: int,
    test_size: float,
    top_k: int,
    betas: list[float],
    max_queries: int | None,
) -> dict:
    if not source_records:
        raise ValueError("source contains no records")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in source_records:
        records.append({
            "recordId": stable_record_id(dataset, item.folder, item.source_path),
            "folder": item.folder,
            "key": item.key,
            "sourcePath": item.source_path.replace("\\", "/"),
            "vector": [float(value) for value in item.vector],
        })
    records.sort(key=lambda row: (row["folder"], row["sourcePath"], row["recordId"]))
    dimensions = {len(row["vector"]) for row in records}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent vector dimensions: {sorted(dimensions)}")
    if len({row["recordId"] for row in records}) != len(records):
        raise ValueError("recordId collision")

    keys = sorted({row["key"] for row in records})
    if key_vector_source == "clip-text":
        key_vectors = encode_text_keys(keys, model_name, prompt)
        reproduction_ready = bool(source.get("fullVerified")) and source.get("kind") != "precomputed-sample"
    elif key_vector_source == "representative-validation":
        key_vectors = representative_key_vectors(source_records)
        reproduction_ready = False
    else:
        raise ValueError(key_vector_source)

    train_ids, query_pool_ids = split_ids(records, seed, test_size)
    query_ids = query_pool_ids[:max_queries] if max_queries else query_pool_ids
    by_id = {row["recordId"]: row for row in records}
    train_records = [by_id[item_id] for item_id in train_ids]
    microclusters = freeze_microclusters(train_records, alpha=0.5)
    cluster_by_id = {row["recordId"]: row for row in microclusters}
    queries = [{
        "queryId": f"{PROTOCOL}/{record_id}",
        "sourceRecordId": record_id,
        "trueKey": by_id[record_id]["key"],
        "vector": by_id[record_id]["vector"],
        "keyVector": key_vectors[by_id[record_id]["key"]],
    } for record_id in query_ids]
    ground_truth = []
    train_matrix = np.asarray([row["vector"] for row in train_records], dtype=np.float32)
    train_matrix /= np.maximum(np.linalg.norm(train_matrix, axis=1, keepdims=True), np.finfo(np.float32).eps)
    candidate_key_matrix = np.asarray([key_vectors[row["key"]] for row in train_records], dtype=np.float32)
    candidate_key_matrix /= np.maximum(np.linalg.norm(candidate_key_matrix, axis=1, keepdims=True), np.finfo(np.float32).eps)
    candidate_ids = np.asarray([row["recordId"] for row in train_records])
    for query in queries:
        query_vector = np.asarray(normalize(query["vector"]), dtype=np.float32)
        query_key_vector = np.asarray(normalize(query["keyVector"]), dtype=np.float32)
        embedding_distances = np.maximum(0.0, 1.0 - train_matrix @ query_vector)
        semantic_distances = np.maximum(0.0, 1.0 - candidate_key_matrix @ query_key_vector)
        for beta in betas:
            scores = beta * semantic_distances + (1.0 - beta) * embedding_distances
            order = np.lexsort((candidate_ids, scores))[:top_k]
            ground_truth.append({"queryId": query["queryId"], "beta": beta,
                                 "recordIds": [str(candidate_ids[index]) for index in order]})

    write_jsonl(output_dir / "records.jsonl", records)
    write_jsonl(output_dir / "key_vectors.jsonl", [
        {"key": key, "prompt": prompt.format(key), "vector": key_vectors[key]} for key in keys
    ])
    write_jsonl(output_dir / "splits.jsonl", [{
        "protocol": PROTOCOL, "seed": seed, "testSize": test_size,
        "trainRecordIds": train_ids, "queryPoolRecordIds": query_pool_ids,
        "queryRecordIds": query_ids,
    }])
    write_jsonl(output_dir / "queries.jsonl", queries)
    write_jsonl(output_dir / "ground_truth.jsonl", ground_truth)
    write_jsonl(output_dir / "microclusters.jsonl", microclusters)

    with (output_dir / "cangjie_input.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"META\t{PROTOCOL}\t{dataset}\t{top_k}\n")
        for key in keys:
            handle.write(f"KEY\t{key}\t{','.join(map(str, key_vectors[key]))}\n")
        for record in train_records:
            cluster = cluster_by_id[record["recordId"]]
            handle.write(
                f"TRAIN\t{record['recordId']}\t{record['folder']}\t{record['key']}\t"
                f"{','.join(map(str, record['vector']))}\t{cluster['clusterKey']}\t"
                f"{cluster['clusterGroup']}\n"
            )
        for query in queries:
            handle.write(f"QUERY\t{query['queryId']}\t{query['sourceRecordId']}\t{query['trueKey']}\t{','.join(map(str, query['vector']))}\t{','.join(map(str, query['keyVector']))}\n")
        for row in ground_truth:
            handle.write(f"GT\t{row['queryId']}\t{row['beta']}\t{','.join(row['recordIds'])}\n")

    files = ["records.jsonl", "key_vectors.jsonl", "splits.jsonl", "queries.jsonl",
             "ground_truth.jsonl", "microclusters.jsonl", "cangjie_input.txt"]
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "artifactStatus": "reproduction-ready" if reproduction_ready else "validation-only",
        "reproductionReady": reproduction_ready,
        "blockingReason": None if reproduction_ready else (
            "Current input is a repository sample and/or key vectors are representative fallbacks; raw full data is required."
        ),
        "source": source,
        "preprocessing": {
            "reference": "frozen violas_python image benchmark logic",
            "clipModel": clip_model_provenance(model_name),
            "promptTemplate": prompt,
            "imageVectorNormalization": "none (cosine consumers normalize)",
            "keyVectorNormalization": "L2",
            "split": {"scope": "per-class", "testSize": test_size, "randomState": seed, "shuffle": True,
                      "implementation": "sklearn.model_selection.train_test_split"},
            "querySelection": {"order": "class then source path", "maxQueries": max_queries},
            "microclustering": {"implementation": "sklearn.cluster.KMeans",
                                "alpha": 0.5, "randomState": 42, "nInit": 10},
            "keyVectorSource": key_vector_source,
        },
        "counts": {"records": len(records), "training": len(train_ids),
                   "queryPool": len(query_pool_ids), "queries": len(query_ids), "keys": len(keys)},
        "dimension": dimensions.pop(),
        "topK": top_k,
        "betas": betas,
        "recordId": "sha256(dataset, folder, normalized source path), first 20 hex characters",
        "files": {name: {"sha256": sha256_file(output_dir / name), "bytes": (output_dir / name).stat().st_size} for name in files},
        "consumerContract": "Cangjie, Faiss, Milvus, Qdrant and Chroma read this directory; consumers must not resplit records.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_artifact(path: Path) -> tuple[dict, list[dict], list[dict], dict[str, list[float]], dict[tuple[str, float], list[str]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"expected protocol {PROTOCOL!r}, got {manifest.get('protocol')!r}")
    for name, metadata in manifest["files"].items():
        actual = sha256_file(path / name)
        if actual != metadata["sha256"]:
            raise ValueError(f"artifact hash mismatch for {name}: {actual} != {metadata['sha256']}")
    all_records = {row["recordId"]: row for row in jsonl_rows(path / "records.jsonl")}
    split = jsonl_rows(path / "splits.jsonl")[0]
    records = [all_records[item_id] for item_id in split["trainRecordIds"]]
    queries = jsonl_rows(path / "queries.jsonl")
    key_vectors = {row["key"]: row["vector"] for row in jsonl_rows(path / "key_vectors.jsonl")}
    ground_truth = {(row["queryId"], float(row["beta"])): row["recordIds"] for row in jsonl_rows(path / "ground_truth.jsonl")}
    return manifest, records, queries, key_vectors, ground_truth


def parse_betas(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",")]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("betas must be in [0,1]")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("caltech", "cub", "coco"), required=True)
    parser.add_argument("--source-kind", choices=("precomputed-sample", "folder-images", "coco-json"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="image root required with --source-kind coco-json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-vector-source", choices=("clip-text", "representative-validation"), default="clip-text")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="CLIP image batch size; lower this if memory is insufficient")
    parser.add_argument("--embedding-cache", type=Path,
                        help="resumable JSONL cache (default: OUTPUT_DIR/source_embeddings.jsonl)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=200,
                        help="match Python paper benchmark default; 0 means the full 10%% query pool")
    parser.add_argument("--betas", type=parse_betas,
                        default=parse_betas("0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"))
    parser.add_argument("--full-verified", action="store_true",
                        help="assert that source/version/scope match the intended full paper dataset")
    args = parser.parse_args()
    if (not 0.0 < args.test_size < 1.0 or args.top_k <= 0
            or args.batch_size <= 0 or args.max_queries < 0):
        parser.error("test-size must be in (0,1), top-k and batch-size must be positive")
    if args.source_kind == "precomputed-sample" and args.full_verified:
        parser.error("precomputed-sample cannot be marked --full-verified")
    source = args.source.resolve()
    cache_path = (args.embedding_cache or (args.output_dir / "source_embeddings.jsonl")).resolve()
    if args.source_kind == "precomputed-sample":
        source_records = load_precomputed_source(args.dataset, source)
        source_meta = {"kind": args.source_kind, "path": str(source), "sha256": sha256_file(source),
                       "fullVerified": args.full_verified}
    elif args.source_kind == "folder-images":
        image_rows = load_folder_images(source)
        source_records = encode_images(image_rows, source, args.model, cache_path, args.batch_size)
        source_meta = {"kind": args.source_kind, "path": str(source),
                       "treeSha256": tree_hash((path for _, path in image_rows), source),
                       "fullVerified": args.full_verified}
    else:
        if args.dataset != "coco" or args.image_root is None:
            parser.error("coco-json requires --dataset coco and --image-root")
        image_root = args.image_root.resolve()
        source_records, used_paths = encode_coco_json(
            image_root, source, args.model, cache_path, args.batch_size
        )
        source_meta = {"kind": args.source_kind, "jsonPath": str(source), "jsonSha256": sha256_file(source),
                       "imageRoot": str(image_root), "treeSha256": tree_hash(used_paths[1:], image_root),
                       "fullVerified": args.full_verified,
                       "requiredPaperInput": "fixed coco_dataset_10000.json (COCO-10k)"}
    expected_full_counts = {"caltech": 8677, "cub": 11788, "coco": 10000}
    if args.full_verified and len(source_records) != expected_full_counts[args.dataset]:
        parser.error(
            f"--full-verified requires {expected_full_counts[args.dataset]} records for "
            f"{args.dataset}, got {len(source_records)}"
        )
    manifest = build_artifact(args.dataset, source_records, args.output_dir, source_meta,
                              args.key_vector_source, args.model, args.prompt, args.seed,
                              args.test_size, args.top_k, args.betas,
                              args.max_queries or None)
    print(json.dumps({"artifact": str(args.output_dir), "status": manifest["artifactStatus"],
                      "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
