"""Count reproducible source lines for Cangjie Violas, Python Violas and Faiss."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def tracked_files(root: Path) -> set[Path] | None:
    result = subprocess.run(
        ["git", "ls-files"], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        return None
    return {(root / line).resolve() for line in result.stdout.splitlines() if line}


def source_line_count(path: Path) -> tuple[int, int, int]:
    physical = non_empty = source = 0
    in_block = False
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        physical += 1
        stripped = raw.strip()
        if stripped:
            non_empty += 1
        code = stripped
        while code:
            if in_block:
                end = code.find("*/")
                if end < 0:
                    code = ""
                    break
                code = code[end + 2:].lstrip()
                in_block = False
                continue
            if code.startswith("//") or (path.suffix.lower() in {".py", ".ps1"} and code.startswith("#")):
                code = ""
                break
            block = code.find("/*")
            line_comment = code.find("//")
            hash_comment = code.find("#") if path.suffix.lower() in {".py", ".ps1"} else -1
            comment_positions = [item for item in (block, line_comment, hash_comment) if item >= 0]
            if not comment_positions:
                break
            first = min(comment_positions)
            if first == block:
                before = code[:first].strip()
                end = code.find("*/", first + 2)
                if end < 0:
                    code = before
                    in_block = True
                    break
                code = (before + " " + code[end + 2:]).strip()
            else:
                code = code[:first].strip()
                break
        if code:
            source += 1
    return physical, non_empty, source


def collect(root: Path, relative: str, extensions: set[str],
            tracked: set[Path] | None, excludes: tuple[str, ...] = ()) -> dict:
    base = (root / relative).resolve()
    files = []
    if base.exists():
        for path in base.rglob("*"):
            resolved = path.resolve()
            rel = resolved.relative_to(root.resolve()).as_posix()
            if (path.is_file() and path.suffix.lower() in extensions
                    and not any(rel.startswith(prefix) for prefix in excludes)
                    and (tracked is None or resolved in tracked)):
                files.append(resolved)
    totals = [0, 0, 0]
    for path in files:
        values = source_line_count(path)
        totals = [left + right for left, right in zip(totals, values)]
    return {
        "files": len(files),
        "physicalLines": totals[0],
        "nonEmptyLines": totals[1],
        "sourceLines": totals[2],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faiss-root", type=Path,
                        help="checkout of facebookresearch/faiss; omit when unavailable")
    parser.add_argument("--include-untracked", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "loc" / "source-lines.json")
    args = parser.parse_args()

    repo_tracked = None if args.include_untracked else tracked_files(ROOT)
    scopes = {
        "Cangjie storage core": collect(ROOT, "cj_core/src/storage", {".cj"}, repo_tracked),
        "Cangjie benchmark": collect(ROOT, "cj_core/src/bench", {".cj"}, repo_tracked),
        "Cangjie all src": collect(ROOT, "cj_core/src", {".cj"}, repo_tracked),
        "Python Violas core": collect(ROOT, "violas_python/violas", {".py"}, repo_tracked),
        "Python benchmarks": collect(ROOT, "violas_python/benchmarks", {".py"}, repo_tracked),
        "Repository benchmark tools": collect(ROOT, "tools", {".py", ".ps1"}, repo_tracked),
    }
    faiss_info = None
    if args.faiss_root:
        faiss_root = args.faiss_root.resolve()
        faiss_tracked = None if args.include_untracked else tracked_files(faiss_root)
        extensions = {".h", ".hpp", ".c", ".cc", ".cpp", ".cuh", ".cu", ".py"}
        scopes.update({
            "Faiss CPU library": collect(
                faiss_root, "faiss", extensions, faiss_tracked, excludes=("faiss/gpu/",)
            ),
            "Faiss GPU library": collect(faiss_root, "faiss/gpu", extensions, faiss_tracked),
            "Faiss all library": collect(faiss_root, "faiss", extensions, faiss_tracked),
            "Faiss benchmarks": collect(faiss_root, "benchs", extensions, faiss_tracked),
            "Faiss tests": collect(faiss_root, "tests", extensions, faiss_tracked),
        })
        faiss_info = {"path": str(faiss_root), "gitCommit": git_commit(faiss_root)}

    payload = {
        "schemaVersion": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "trackedOnly": not args.include_untracked,
        "rules": {
            "physicalLines": "all text lines",
            "nonEmptyLines": "trimmed line is non-empty",
            "sourceLines": "non-empty after removing comment-only content",
            "exclusions": "datasets, artifacts, results, docs, build output and dependencies",
            "warning": "LOC describes implementation scope, not speed, correctness or quality",
        },
        "repository": {"path": str(ROOT), "gitCommit": git_commit(ROOT)},
        "faiss": faiss_info,
        "scopes": scopes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    lines = [
        "# Source line comparison",
        "",
        f"- tracked files only: {payload['trackedOnly']}",
        f"- Violas commit: `{payload['repository']['gitCommit']}`",
        f"- Faiss commit: `{faiss_info['gitCommit'] if faiss_info else 'not supplied'}`",
        "",
        "| Scope | Files | Physical | Non-empty | Source (comments excluded) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in scopes.items():
        lines.append(
            f"| {name} | {row['files']} | {row['physicalLines']} | "
            f"{row['nonEmptyLines']} | {row['sourceLines']} |"
        )
    lines.extend([
        "",
        "代码行数只表示工程规模。Faiss 是包含大量索引、量化、SIMD、CPU/GPU "
        "后端和测试的成熟底层库，不能用 LOC 直接判断两者性能或代码质量。",
        "",
    ])
    md = args.output.with_suffix(".md")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
