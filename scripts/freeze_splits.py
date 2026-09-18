#!/usr/bin/env python3
"""Regenerate the frozen split manifests in data/splits/.

This is not part of any normal workflow. The manifests are committed, and every
run loads them from disk; rerunning this is how you would *change* the study's
splits, which invalidates the pre-registration and every result already
collected against them.

It is wired to `make splits-REGENERATE` rather than `make splits` for that
reason, and it refuses to overwrite an existing manifest without --force.

Usage:
    python -m scripts.freeze_splits --check    # verify what is committed
    python -m scripts.freeze_splits --force    # rewrite the manifests
"""

import argparse
import sys
from typing import Dict, List

from edge_slm_ace.data.belebele import belebele_path, load_belebele, passage_of, study_split
from edge_slm_ace.data.global_mmlu import GLOBAL_MMLU_LANGUAGES, global_mmlu_path, load_global_mmlu
from edge_slm_ace.data.splits import (
    ADAPT_SUBSAMPLE_SIZE,
    BELEBELE_SPLIT,
    EVAL_SUBSAMPLE_SIZE,
    GLOBAL_MMLU_ADAPT,
    GLOBAL_MMLU_EVAL,
    SUBSAMPLE_SEED,
    file_checksum,
    ids_checksum,
    load_manifest,
    manifest_path,
    stratified_sample,
    write_manifest,
)
from edge_slm_ace.utils import BELEBELE_ADAPTATION_PASSAGES, DEFAULT_SEED


def source_checksums(task: str) -> Dict[str, str]:
    """SHA256 of each language file the ids index into."""
    if task == "global_mmlu":
        return {
            lang: file_checksum(global_mmlu_path(lang, "test"))
            for lang in sorted(GLOBAL_MMLU_LANGUAGES)
        }
    return {lang: file_checksum(belebele_path(lang)) for lang in ("en", "ne")}


def build_global_mmlu() -> List[str]:
    """
    Freeze the Global-MMLU evaluation subsample and the adaptation split.

    The eval subsample is drawn first and the adaptation split from what is
    left, so disjointness is a property of the construction rather than a
    check bolted on after it.

    Returns:
        Human-readable lines describing what was written.
    """
    examples = load_global_mmlu("ne", "test")

    eval_ids = stratified_sample(
        examples, EVAL_SUBSAMPLE_SIZE, SUBSAMPLE_SEED, stratum_key="subject"
    )
    adapt_ids = stratified_sample(
        examples,
        ADAPT_SUBSAMPLE_SIZE,
        SUBSAMPLE_SEED,
        stratum_key="subject",
        exclude=eval_ids,
    )

    overlap = set(eval_ids) & set(adapt_ids)
    if overlap:  # pragma: no cover - construction makes this unreachable
        raise AssertionError(f"{len(overlap)} ids are in both splits")

    checksums = source_checksums("global_mmlu")
    common = {
        "task": "global_mmlu",
        "split": "test",
        "seed": SUBSAMPLE_SEED,
        "stratified_by": "subject",
        # The ids are sample_ids, which are identical across language configs.
        # English evaluates the same ids; that is what keeps McNemar paired.
        "languages": sorted(GLOBAL_MMLU_LANGUAGES),
        "source_sha256": checksums,
    }

    write_manifest(
        GLOBAL_MMLU_EVAL,
        {
            **common,
            "role": "evaluation",
            "size": len(eval_ids),
            "item_ids": eval_ids,
        },
    )
    write_manifest(
        GLOBAL_MMLU_ADAPT,
        {
            **common,
            "role": "adaptation",
            "size": len(adapt_ids),
            "disjoint_from": GLOBAL_MMLU_EVAL,
            "item_ids": adapt_ids,
        },
    )
    return [
        f"  {GLOBAL_MMLU_EVAL}: {len(eval_ids)} items",
        f"  {GLOBAL_MMLU_ADAPT}: {len(adapt_ids)} items, disjoint from the eval subsample",
    ]


def build_belebele() -> List[str]:
    """Freeze the Belebele passage-level split."""
    examples = load_belebele("ne")
    adapt_ids, eval_ids = study_split([e["id"] for e in examples], DEFAULT_SEED)

    write_manifest(
        BELEBELE_SPLIT,
        {
            "task": "belebele",
            "split": "test",
            "seed": DEFAULT_SEED,
            "stratified_by": None,
            "unit": "passage",
            "adaptation_passages": BELEBELE_ADAPTATION_PASSAGES,
            "languages": ["en", "ne"],
            "source_sha256": source_checksums("belebele"),
            "role": "evaluation",
            "size": len(eval_ids),
            "adaptation_item_ids": sorted(adapt_ids),
            "adaptation_ids_sha256": ids_checksum(adapt_ids),
            "item_ids": eval_ids,
        },
    )
    return [
        f"  {BELEBELE_SPLIT}: {len(adapt_ids)} adaptation "
        f"({len({passage_of(i) for i in adapt_ids})} passages) / "
        f"{len(eval_ids)} evaluation "
        f"({len({passage_of(i) for i in eval_ids})} passages)"
    ]


def check() -> int:
    """
    Verify every committed manifest loads, verifies, and still matches its source.

    Returns:
        Process exit code.
    """
    problems = []
    for name in (GLOBAL_MMLU_EVAL, GLOBAL_MMLU_ADAPT, BELEBELE_SPLIT):
        try:
            body = load_manifest(name)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{name}: {exc}")
            continue

        print(f"  {name}: {len(body['item_ids'])} items, ids_sha256 {body['ids_sha256'][:16]}...")

        try:
            actual = source_checksums(body["task"])
        except FileNotFoundError:
            print("      source corpus absent; cannot verify source_sha256 (run `make data`)")
            continue
        if actual != body["source_sha256"]:
            problems.append(
                f"{name}: the corpus on disk is not the one these ids were "
                f"drawn from. Expected {body['source_sha256']}, got {actual}."
            )

    disjoint = set(load_manifest(GLOBAL_MMLU_EVAL)["item_ids"]) & set(
        load_manifest(GLOBAL_MMLU_ADAPT)["item_ids"]
    )
    if disjoint:
        problems.append(f"{len(disjoint)} ids are in both Global-MMLU splits")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="Verify, do not write")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite manifests that already exist. Invalidates the pre-registration.",
    )
    args = parser.parse_args(argv)

    if args.check:
        print("Checking frozen splits:")
        return check()

    existing = [
        n
        for n in (GLOBAL_MMLU_EVAL, GLOBAL_MMLU_ADAPT, BELEBELE_SPLIT)
        if manifest_path(n).exists()
    ]
    if existing and not args.force:
        print(
            "Refusing to overwrite committed split manifests: "
            f"{', '.join(existing)}.\n"
            "Every result already collected is keyed to these ids. Pass "
            "--force if you really mean to change the study's splits.",
            file=sys.stderr,
        )
        return 1

    print("Rewriting frozen splits. This invalidates the pre-registration.\n")
    lines = build_global_mmlu() + build_belebele()
    print("\n".join(lines))
    print("\nChecking what was written:")
    return check()


if __name__ == "__main__":
    sys.exit(main())
