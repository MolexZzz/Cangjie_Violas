"""Run one non-interactive Cangjie benchmark and persist its structured result."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CJ_CORE = ROOT / "cj_core"


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def command_version(command: list[str]) -> str | None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    return "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("smoke", "partial", "full"), default="smoke")
    parser.add_argument("--dataset", choices=("1", "2", "3", "4", "5", "6"), default="1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    command = f"bench {args.scale} {args.dataset}"
    completed = subprocess.run(
        ["cjpm", "run"],
        cwd=CJ_CORE,
        input=command + "\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stdout + completed.stderr)

    prefix = "RESULT_JSON|"
    payloads = [line[len(prefix) :] for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if len(payloads) != 1:
        raise RuntimeError(f"expected one RESULT_JSON line, found {len(payloads)}")

    result = json.loads(payloads[0])
    result["provenance"] = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "gitCommit": git_commit(),
        "runner": "tools/run_cangjie_benchmark.py",
        "command": command,
        "environment": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "cjc": command_version(["cjc", "--version"]),
            "cjpm": command_version(["cjpm", "--version"]),
        },
    }
    output = args.output or ROOT / "results" / "cangjie" / f"dataset-{args.dataset}-{args.scale}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
