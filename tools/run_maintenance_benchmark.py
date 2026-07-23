"""Benchmark reproducible 200-vector data and index maintenance operations.

The artifact is never modified. Query-pool vectors are used as deterministic
insertions, and copies of training vectors are perturbed for updates.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from external_db_benchmark import make_backend, normalize
from paper_artifact import load_artifact
from run_faiss_baseline import build_index, normalized_matrix


ROOT = Path(__file__).resolve().parents[1]


def elapsed_ms(start: int) -> float:
    return (time.perf_counter_ns() - start) / 1_000_000.0


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def updated_vector(vector: list[float], offset: int) -> list[float]:
    row = np.asarray(vector, dtype=np.float32)
    changed = 0.99 * row + 0.01 * np.roll(row, 1 + offset % 7)
    norm = float(np.linalg.norm(changed))
    return (changed / norm if norm else changed).tolist()


def faiss_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        nlist=args.nlist,
        nprobe=args.nprobe,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )


def run_faiss(method: str, base_records: list[dict], insert_records: list[dict],
              update_records: list[dict], args: argparse.Namespace) -> dict:
    base = normalized_matrix([row["vector"] for row in base_records])
    inserted = normalized_matrix([row["vector"] for row in insert_records])
    options = faiss_args(args)

    started = time.perf_counter_ns()
    index, config = build_index(method, base, options)
    construction_ms = elapsed_ms(started)

    started = time.perf_counter_ns()
    index.add(inserted)
    insertion_ms = elapsed_ms(started)

    updated = normalized_matrix([
        updated_vector(row["vector"], i) for i, row in enumerate(update_records)
    ])
    full_after_update = np.ascontiguousarray(np.vstack([
        updated,
        base[len(updated):],
        inserted,
    ]))
    started = time.perf_counter_ns()
    updated_index, _ = build_index(method, full_after_update, options)
    update_ms = elapsed_ms(started)

    # Faiss HNSW has no general in-place deletion. Use the same portable
    # full-rebuild strategy for all three indexes so the operation is comparable.
    after_delete = full_after_update[:-len(inserted)]
    started = time.perf_counter_ns()
    deleted_index, _ = build_index(method, after_delete, options)
    deletion_ms = elapsed_ms(started)

    return {
        "backend": "faiss",
        "method": config["indexType"],
        "maintenanceStrategy": {
            "insert": "native index.add",
            "update": "full rebuild after deterministic vector replacement",
            "delete": "full rebuild after record removal",
        },
        "operations": {
            "vectorInsertion": {"count": len(insert_records), "batchMs": insertion_ms,
                                "perVectorMs": insertion_ms / len(insert_records)},
            "vectorUpdate": {"count": len(update_records), "batchMs": update_ms,
                             "perVectorMs": update_ms / len(update_records)},
            "vectorDelete": {"count": len(insert_records), "batchMs": deletion_ms,
                             "perVectorMs": deletion_ms / len(insert_records)},
            "indexConstruction": {"records": len(base_records), "ms": construction_ms},
            "indexUpdate": {"records": len(full_after_update), "ms": update_ms,
                            "strategy": "full-rebuild"},
        },
        "finalIndexBytes": len(__import__("faiss").serialize_index(deleted_index)),
        "postUpdateIndexBytes": len(__import__("faiss").serialize_index(updated_index)),
    }


def backend_namespace(name: str, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        backend=name,
        execution_mode=args.execution_mode,
        local_state_dir=args.local_state_dir / name,
        qdrant_url=args.qdrant_url,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
    )


def run_database(name: str, dataset: str, dimension: int, base_records: list[dict],
                 insert_records: list[dict], update_records: list[dict],
                 args: argparse.Namespace) -> dict:
    backend = make_backend(backend_namespace(name, args))
    collection = f"violas_maintenance_{dataset}_{name}".lower().replace("-", "_")
    try:
        started = time.perf_counter_ns()
        backend.reset(collection, dimension)
        reset_ms = elapsed_ms(started)

        started = time.perf_counter_ns()
        backend.upsert(base_records)
        initial_write_ms = elapsed_ms(started)

        started = time.perf_counter_ns()
        backend.upsert(insert_records)
        insertion_ms = elapsed_ms(started)

        replacements = [
            {**row, "vector": updated_vector(row["vector"], i)}
            for i, row in enumerate(update_records)
        ]
        started = time.perf_counter_ns()
        backend.upsert(replacements)
        update_ms = elapsed_ms(started)

        started = time.perf_counter_ns()
        backend.delete([row["recordId"] for row in insert_records])
        delete_ms = elapsed_ms(started)
        return {
            "backend": name,
            "method": backend.config.get("index", "engine-default"),
            "config": backend.config,
            "maintenanceStrategy": {
                "insert": "synchronous backend upsert",
                "update": "synchronous backend upsert with existing ID",
                "delete": "synchronous backend delete",
                "indexUpdate": "managed by database and included in operation latency",
            },
            "operations": {
                "vectorInsertion": {"count": len(insert_records), "batchMs": insertion_ms,
                                    "perVectorMs": insertion_ms / len(insert_records)},
                "vectorUpdate": {"count": len(update_records), "batchMs": update_ms,
                                 "perVectorMs": update_ms / len(update_records)},
                "vectorDelete": {"count": len(insert_records), "batchMs": delete_ms,
                                 "perVectorMs": delete_ms / len(insert_records)},
                "indexConstruction": {
                    "records": len(base_records),
                    "ms": reset_ms + initial_write_ms,
                    "resetMs": reset_ms,
                    "initialWriteMs": initial_write_ms,
                    "boundary": "collection reset plus synchronous initial ingestion",
                },
                "indexUpdate": {
                    "records": len(update_records),
                    "ms": update_ms,
                    "strategy": "engine-managed; included in vectorUpdate",
                },
            },
        }
    finally:
        backend.close()


def run_cangjie(artifact: Path, count: int, args: argparse.Namespace) -> dict:
    input_path = (artifact / "cangjie_input.txt").resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    environment = os.environ.copy()
    environment["cjHeapSize"] = args.cangjie_heap_size
    completed = subprocess.run(
        ["cjpm", "run"],
        cwd=ROOT / "cj_core",
        input=f"maintenance {input_path} {count}\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cangjie maintenance failed ({completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    summary = next(
        (line for line in completed.stdout.splitlines()
         if line.startswith("MAINTENANCE_SUMMARY|")),
        None,
    )
    if summary is None:
        raise RuntimeError(f"Cangjie maintenance summary is missing:\n{completed.stdout}")
    fields = {}
    for token in summary.split("|")[1:]:
        key, value = token.split("=", 1)
        fields[key] = value
    return {
        "backend": "cangjie-violas",
        "method": "HDMG",
        "maintenanceStrategy": {
            "insert": "stable-ID append to an existing micro-cluster",
            "update": "stable-ID in-place vector replacement",
            "delete": "stable-ID object removal",
            "indexUpdate": "full HDMG rebuild after each mutation batch",
        },
        "operations": {
            "vectorInsertion": {"count": count, "batchMs": float(fields["vectorInsertionMs"]),
                                "perVectorMs": float(fields["vectorInsertionMs"]) / count},
            "vectorUpdate": {"count": count, "batchMs": float(fields["vectorUpdateMs"]),
                             "perVectorMs": float(fields["vectorUpdateMs"]) / count},
            "vectorDelete": {"count": count, "batchMs": float(fields["vectorDeleteMs"]),
                             "perVectorMs": float(fields["vectorDeleteMs"]) / count},
            "indexConstruction": {"records": int(fields["initialRecords"]),
                                  "ms": float(fields["indexConstructionMs"])},
            "indexUpdate": {"records": int(fields["initialRecords"]),
                            "ms": float(fields["indexAfterUpdateMs"]),
                            "afterInsertMs": float(fields["indexAfterInsertMs"]),
                            "afterUpdateMs": float(fields["indexAfterUpdateMs"]),
                            "afterDeleteMs": float(fields["indexAfterDeleteMs"]),
                            "strategy": "full-hdmg-rebuild"},
        },
    }


def markdown(payload: dict) -> str:
    lines = [
        f"# Data and index maintenance: {payload['dataset']}",
        "",
        f"每项数据变更固定处理 {payload['scale']['mutationVectors']} 个向量。时间是整批操作耗时，不是单条耗时。",
        "",
        "| Backend/index | Insert batch (ms) | Update batch (ms) | Delete batch (ms) | Initial index/build (ms) | Index update (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        ops = row["operations"]
        lines.append(
            f"| {row['backend']} / {row['method']} | "
            f"{ops['vectorInsertion']['batchMs']:.3f} | "
            f"{ops['vectorUpdate']['batchMs']:.3f} | "
            f"{ops['vectorDelete']['batchMs']:.3f} | "
            f"{ops['indexConstruction']['ms']:.3f} | "
            f"{ops['indexUpdate']['ms']:.3f} |"
        )
    lines.extend([
        "",
        "## 口径说明",
        "",
        "- Faiss 插入使用原生 `add`；通用更新和删除采用完整重建，因为 HNSW 不支持通用原位删除。",
        "- 仓颉 Violas 使用稳定 record ID 完成对象插入、原位更新和删除；每批变更后完整重建 HDMG。",
        "- Milvus、Qdrant、Chroma 的更新和索引维护由服务端同步完成，index update 已包含在 update 时间内。",
        "- 数据库的 initial build 包含建集合和初始数据写入，不能与 Faiss 的纯内存索引构建时间直接等同。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mutation-count", type=int, default=200)
    parser.add_argument("--backends", default="cangjie,faiss",
                        help="comma list: cangjie,faiss,mock,milvus,qdrant,chroma")
    parser.add_argument("--execution-mode", choices=("paper-local", "service"),
                        default="service")
    parser.add_argument("--local-state-dir", type=Path,
                        default=ROOT / "results" / "maintenance" / "database-state")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=int, default=32)
    parser.add_argument("--cangjie-heap-size", default="2GB")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN", "root:Milvus"))
    parser.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "127.0.0.1"))
    parser.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    requested = [item.strip() for item in args.backends.split(",") if item.strip()]
    valid = {"cangjie", "faiss", "mock", "milvus", "qdrant", "chroma"}
    if not requested or set(requested) - valid:
        parser.error(f"--backends must contain only {sorted(valid)}")

    manifest, records, queries, _, _ = load_artifact(args.artifact)
    count = min(args.mutation_count, len(records), len(queries))
    if count <= 0:
        parser.error("artifact does not contain enough records and queries")
    base_records = [{**row, "vector": normalize(row["vector"])} for row in records]
    insert_records = [
        {
            "recordId": row["sourceRecordId"],
            "key": row["trueKey"],
            "vector": normalize(row["vector"]),
        }
        for row in queries[:count]
    ]
    update_records = base_records[:count]

    results = []
    for backend in requested:
        print(f"MAINTENANCE_STAGE|{backend}|start", flush=True)
        if backend == "cangjie":
            results.append(run_cangjie(args.artifact, count, args))
        elif backend == "faiss":
            for method in ("exact", "ivf", "hnsw"):
                results.append(run_faiss(
                    method, base_records, insert_records, update_records, args
                ))
        else:
            results.append(run_database(
                backend, manifest["dataset"], manifest["dimension"], base_records,
                insert_records, update_records, args
            ))
        print(f"MAINTENANCE_STAGE|{backend}|done", flush=True)

    payload = {
        "schemaVersion": 1,
        "protocol": "violas-maintenance-v1",
        "dataset": manifest["dataset"],
        "artifact": str(args.artifact),
        "scale": {
            "initialRecords": len(base_records),
            "mutationVectors": count,
            "dimension": manifest["dimension"],
        },
        "cangjieViolas": {
            "status": "measured" if "cangjie" in requested else "not-requested",
            "objectIdentity": "stable record ID",
            "indexLifecycle": "every successful mutation invalidates HDMG; benchmark rebuilds it",
        },
        "results": results,
        "provenance": {
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "gitCommit": git_commit(),
            "runner": "tools/run_maintenance_benchmark.py",
            "os": platform.platform(),
            "numpyVersion": np.__version__,
        },
    }
    output = args.output or ROOT / "results" / "maintenance" / f"{manifest['dataset']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = output.with_suffix(".md")
    md.write_text(markdown(payload), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
