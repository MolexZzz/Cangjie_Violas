#!/usr/bin/env python3
"""Download the paper-aligned Caltech-101, CUB-200-2011 and COCO-10k data.

Dataset payloads are written below ``dataset/`` and are intentionally ignored by
Git.  Caltech and CUB archives are verified against the hashes published by
CaltechDATA.  COCO-10k is a deterministic prefix of unique images in the
MS_COCO_2017_URL_TEXT table; all captions belonging to a selected URL are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from PIL import Image


CALTECH_URL = (
    "https://data.caltech.edu/records/mzrjq-6wc02/files/"
    "caltech-101.zip?download=1"
)
CALTECH_MD5 = "3138e1922a9193bfa496528edbbc45d0"
CUB_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/"
    "CUB_200_2011.tgz?download=1"
)
CUB_MD5 = "97eceeb196236b17998738112f37df78"
COCO_TABLE_URL = (
    "https://huggingface.co/datasets/ChristophSchuhmann/"
    "MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet"
)


def fresh_request_url(url: str) -> str:
    if "data.caltech.edu/" not in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}cache_bust={time.time_ns()}"


def stream_download(
    url: str,
    destination: Path,
    *,
    timeout: int = 60,
    max_retries: int = 30,
) -> None:
    """Download with HTTP range resume and atomically publish the result."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        # CaltechDATA redirects to a short-lived signed object-store URL.  A
        # unique query prevents an intermediary returning an expired redirect.
        request_url = fresh_request_url(url)
        try:
            with requests.get(
                request_url,
                headers=headers,
                stream=True,
                timeout=(30, timeout),
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                mode = "ab" if offset and response.status_code == 206 else "wb"
                total = response.headers.get("Content-Length")
                expected = offset + int(total) if total else None
                written = offset
                next_report = written + 64 * 1024 * 1024
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        written += len(chunk)
                        if written >= next_report:
                            suffix = f"/{expected}" if expected else ""
                            print(
                                f"  {destination.name}: {written}{suffix} bytes",
                                flush=True,
                            )
                            next_report = written + 64 * 1024 * 1024
            if expected is not None and written != expected:
                raise RuntimeError(
                    f"incomplete download: {written} != {expected}: {url}"
                )
            partial.replace(destination)
            return
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt == max_retries:
                break
            current = partial.stat().st_size if partial.exists() else 0
            print(
                f"  connection interrupted at {current} bytes; "
                f"resuming ({attempt}/{max_retries})",
                flush=True,
            )
            time.sleep(min(attempt, 5))
    raise RuntimeError(f"download failed after {max_retries} attempts: {url}") from last_error


def remote_size(url: str) -> int:
    with requests.get(
        fresh_request_url(url),
        headers={"Range": "bytes=0-0"},
        stream=True,
        timeout=(30, 30),
    ) as response:
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        if response.status_code != 206 or "/" not in content_range:
            raise RuntimeError(f"server does not support byte ranges: {url}")
        return int(content_range.rsplit("/", 1)[1])


def download_range(
    url: str,
    output: Path,
    start: int,
    end: int,
    *,
    max_retries: int = 30,
) -> None:
    expected = end - start + 1
    partial = output.with_suffix(".part")
    if output.exists() and output.stat().st_size == expected:
        return
    if output.exists():
        output.unlink()
    for attempt in range(1, max_retries + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if have > expected:
            partial.unlink()
            have = 0
        if have == expected:
            partial.replace(output)
            return
        try:
            with requests.get(
                fresh_request_url(url),
                headers={"Range": f"bytes={start + have}-{end}"},
                stream=True,
                timeout=(30, 30),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError("range request unexpectedly returned full content")
                with partial.open("ab") as target:
                    for block in response.iter_content(chunk_size=64 * 1024):
                        if block:
                            target.write(block)
            if partial.stat().st_size == expected:
                partial.replace(output)
                return
        except (requests.RequestException, RuntimeError, OSError):
            if attempt == max_retries:
                raise
            time.sleep(min(attempt, 3))
    raise RuntimeError(f"failed range {start}-{end}: {url}")


def parallel_archive_download(
    url: str,
    destination: Path,
    *,
    workers: int = 12,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Fetch unstable CaltechDATA objects as independently retryable ranges."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    chunk_dir = destination.parent / f".{destination.name}.chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + chunk_size - 1, total - 1))
        for start in range(0, total, chunk_size)
    ]
    print(
        f"  {destination.name}: {total} bytes in {len(ranges)} ranges",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_range,
                url,
                chunk_dir / f"{index:05d}.chunk",
                start,
                end,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 8 == 0 or completed == len(ranges):
                print(
                    f"  {destination.name}: ranges {completed}/{len(ranges)}",
                    flush=True,
                )
    assembling = destination.with_suffix(destination.suffix + ".assembling")
    with assembling.open("wb") as target:
        for index in range(len(ranges)):
            with (chunk_dir / f"{index:05d}.chunk").open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    if assembling.stat().st_size != total:
        raise RuntimeError(
            f"assembled size mismatch: {assembling.stat().st_size} != {total}"
        )
    assembling.replace(destination)
    old_partial = destination.with_suffix(destination.suffix + ".part")
    old_partial.unlink(missing_ok=True)
    shutil.rmtree(chunk_dir)


def digest(path: Path, algorithm: str = "md5") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ensure_archive(url: str, archive: Path, expected_md5: str) -> None:
    if not archive.exists() or digest(archive) != expected_md5:
        print(f"Downloading {archive.name}", flush=True)
        if "data.caltech.edu/" in url:
            parallel_archive_download(url, archive)
        else:
            stream_download(url, archive)
    actual = digest(archive)
    if actual != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {archive}: expected {expected_md5}, got {actual}"
        )
    print(f"Verified {archive.name}: md5={actual}", flush=True)


def safe_extract_zip(archive: Path, output: Path) -> None:
    root = output.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (output / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        bundle.extractall(output)


def safe_extract_tar(archive: Path, output: Path) -> None:
    root = output.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (output / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe tar member: {member.name}")
        bundle.extractall(output, members=members, filter="data")


def image_count(path: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in extensions)


def prepare_caltech(root: Path, archives: Path) -> None:
    target = root / "caltech-101" / "101_ObjectCategories"
    if target.exists() and image_count(target) >= 9_000:
        print(f"Caltech-101 already present: {image_count(target)} files", flush=True)
        return
    archive = archives / "caltech-101.zip"
    ensure_archive(CALTECH_URL, archive, CALTECH_MD5)
    print("Extracting Caltech-101", flush=True)
    safe_extract_zip(archive, root)
    inner_archive = root / "caltech-101" / "101_ObjectCategories.tar.gz"
    if not target.exists() and inner_archive.exists():
        safe_extract_tar(inner_archive, root / "caltech-101")
    if not target.exists():
        raise RuntimeError(f"archive did not create expected path: {target}")
    total = image_count(target)
    if total < 9_000:
        raise RuntimeError(f"Caltech extraction looks incomplete: {total} images")
    print(f"Caltech-101 ready: {total} images ({total - image_count(target / 'BACKGROUND_Google')} excluding background)", flush=True)


def prepare_cub(root: Path, archives: Path) -> None:
    target = root / "CUB_200_2011" / "images"
    if target.exists() and image_count(target) == 11_788:
        print("CUB-200-2011 already present: 11788 images", flush=True)
        return
    archive = archives / "CUB_200_2011.tgz"
    ensure_archive(CUB_URL, archive, CUB_MD5)
    print("Extracting CUB-200-2011", flush=True)
    safe_extract_tar(archive, root)
    total = image_count(target)
    if total != 11_788:
        raise RuntimeError(f"CUB extraction looks incomplete: {total} != 11788")
    print("CUB-200-2011 ready: 11788 images", flush=True)


def coco_records(table: Path, max_items: int) -> list[dict[str, object]]:
    frame = pd.read_parquet(table, columns=["URL", "TEXT"])
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for url, text in frame.itertuples(index=False, name=None):
        url = str(url)
        if url not in grouped:
            if len(grouped) >= max_items:
                continue
            grouped[url] = []
        if text is not None:
            grouped[url].append(str(text))
    if len(grouped) != max_items:
        raise RuntimeError(f"only found {len(grouped)} unique COCO URLs")
    records: list[dict[str, object]] = []
    for index, (url, captions) in enumerate(grouped.items()):
        records.append(
            {
                "image_id": Path(url).stem,
                "url": url,
                "path": f"images/image_{index:05d}.jpg",
                "text": captions[:5],
            }
        )
    return records


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_coco_image(record: dict[str, object], output: Path) -> str | None:
    relative = Path(str(record["path"]))
    destination = output / relative
    if valid_image(destination):
        return None
    try:
        stream_download(str(record["url"]), destination, timeout=30)
        if not valid_image(destination):
            destination.unlink(missing_ok=True)
            return "invalid image"
        return None
    except Exception as error:  # keep the full fixed selection; never substitute
        return str(error)


def prepare_coco(root: Path, archives: Path, max_items: int, workers: int) -> None:
    output = root / "coco"
    output.mkdir(parents=True, exist_ok=True)
    table = archives / "mscoco.parquet"
    if not table.exists():
        print("Downloading COCO URL/caption table", flush=True)
        stream_download(COCO_TABLE_URL, table)
    records = coco_records(table, max_items)
    failures: list[dict[str, str]] = []
    print(f"Downloading/verifying {max_items} deterministic COCO images", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_coco_image, record, output): record
            for record in records
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            error = future.result()
            if error:
                record = futures[future]
                failures.append({"url": str(record["url"]), "error": error})
            if completed % 500 == 0 or completed == max_items:
                print(f"  COCO: {completed}/{max_items}, failures={len(failures)}", flush=True)
    failure_path = output / "download_failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(
            f"COCO selection is incomplete ({len(failures)} failures); rerun to retry. "
            f"Details: {failure_path}"
        )
    manifest = output / f"coco_dataset_{max_items}.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest)
    print(f"COCO ready: {max_items} images; manifest={manifest}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=("caltech", "cub", "coco"),
        default=("caltech", "cub", "coco"),
    )
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--archives", type=Path, default=Path("dataset/.downloads"))
    parser.add_argument("--coco-items", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4) * 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected: Iterable[str] = args.datasets or ("caltech", "cub", "coco")
    args.root.mkdir(parents=True, exist_ok=True)
    args.archives.mkdir(parents=True, exist_ok=True)
    for name in selected:
        if name == "caltech":
            prepare_caltech(args.root, args.archives)
        elif name == "cub":
            prepare_cub(args.root, args.archives)
        else:
            prepare_coco(args.root, args.archives, args.coco_items, args.workers)


if __name__ == "__main__":
    main()
