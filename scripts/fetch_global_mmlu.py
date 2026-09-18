#!/usr/bin/env python3
"""Fetch the Global-MMLU language files this project evaluates on.

Global-MMLU ships 42 languages; we use English and Nepali. Unlike the Belebele
files, these are *not* committed -- ~28 MB of JSONL is an input we can refetch,
not an artifact worth carrying in git. What is committed is the frozen split
manifests under `data/splits/`, each pinned to the SHA256 this script prints.
That is the reproducibility contract: the splits are fixed, the source is
verifiable, the bytes are downloadable.

The dataset is published at `CohereLabs/Global-MMLU`. The `CohereForAI` id used
in older references 307-redirects to it. Apache-2.0.

Usage:
    python -m scripts.fetch_global_mmlu
    python -m scripts.fetch_global_mmlu --check   # verify what is on disk
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict

from edge_slm_ace.data.global_mmlu import (
    EXPECTED_ROWS,
    GLOBAL_MMLU_LANGUAGES,
    SPLITS,
    assert_parallel,
    global_mmlu_path,
    load_global_mmlu,
    subjects,
)

SOURCE = (
    "https://huggingface.co/datasets/CohereLabs/Global-MMLU/resolve/main/"
    "{config}/{split}-00000-of-00001.parquet"
)


def sha256(path: Path) -> str:
    """Checksum of a file on disk, read in chunks so a large split is fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(language: str, split: str) -> int:
    """
    Download one language/split and write it out as JSONL.

    The published format is parquet; we convert on the way in so that the file
    on disk is the same line-per-record format as Belebele, greppable by eye
    when a split manifest has to be audited.

    Returns:
        The row count written.
    """
    import pyarrow.parquet as pq

    config = GLOBAL_MMLU_LANGUAGES[language]
    dest = global_mmlu_path(language, split)
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = SOURCE.format(config=config, split=split)
    tmp = dest.with_suffix(".parquet.tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        rows = pq.read_table(tmp).to_pylist()
    finally:
        tmp.unlink(missing_ok=True)

    with open(dest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def check() -> int:
    """
    Verify the files on disk are the parallel corpus the protocol assumes.

    Returns:
        Process exit code.
    """
    problems = []
    loaded: Dict[str, list] = {}

    for language in GLOBAL_MMLU_LANGUAGES:
        path = global_mmlu_path(language, "test")
        if not path.exists():
            problems.append(f"{language}: missing ({path})")
            continue
        try:
            rows = load_global_mmlu(language, "test")
        except ValueError as exc:
            problems.append(f"{language}: {exc}")
            continue
        loaded[language] = rows

        gold = Counter(r["gold_option_idx"] for r in rows)
        print(
            f"  {language:3} {len(rows):6} items  "
            f"{len(subjects(rows)):3} subjects  "
            f"gold slot {dict(sorted(gold.items()))}"
        )
        print(f"      sha256 {sha256(path)}")

    if len(loaded) == len(GLOBAL_MMLU_LANGUAGES):
        try:
            assert_parallel(loaded)
        except ValueError as exc:
            problems.append(str(exc))

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1

    print("  parallel across languages, gold agrees on every item.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify, do not download")
    args = parser.parse_args()

    if not args.check:
        for language in GLOBAL_MMLU_LANGUAGES:
            for split in SPLITS:
                rows = fetch(language, split)
                flag = "" if rows == EXPECTED_ROWS[split] else "  <-- UNEXPECTED ROW COUNT"
                print(
                    f"  {language}/{split}: {rows} rows -> {global_mmlu_path(language, split)}{flag}"
                )

    print("Checking:")
    return check()


if __name__ == "__main__":
    sys.exit(main())
