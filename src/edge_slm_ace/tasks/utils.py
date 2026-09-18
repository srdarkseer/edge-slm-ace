"""Dataset builders and `process_docs` hooks for this project's harness tasks.

The task YAMLs next to this file are the only place a reported number is
produced, so everything they do is here rather than spread between a loader, a
prompt builder and a scorer.

Two jobs.

**Building the dataset.** The YAMLs use `custom_dataset` rather than
`dataset_path`, so the harness is handed the output of `load_global_mmlu` /
`load_belebele` -- the same functions the splits were drawn with, resolving the
same `REPO_ROOT`. A `dataset_path: json` with a relative `data_files` would
have worked only from the repo root, and would have let the harness score one
copy of the corpus while the splits indexed another.

**Selecting and permuting.** `process_docs` filters to a frozen split and
applies the seeded option permutation. Both belong here, together, because the
harness must see exactly the items the manifest names, in exactly the option
order adaptation saw. Selecting by *position* -- the previous approach, via
`simple_evaluate(samples=...)` -- depended on file row order matching between
languages, which upstream does not promise.

The permutation key is the language-invariant id; see `data/shuffle.py` for why
that is the one thing this pipeline must not get wrong.
"""

from typing import Callable, Dict, List, Optional, Sequence

from edge_slm_ace.data.belebele import load_belebele
from edge_slm_ace.data.global_mmlu import load_global_mmlu
from edge_slm_ace.data.shuffle import shuffle_example
from edge_slm_ace.data.splits import BELEBELE_SPLIT, GLOBAL_MMLU_EVAL, load_manifest

# The letters the tasks score over. Kept here as well as in harness/prompts.py
# because a task whose choices disagreed with the adaptation renderer would
# score continuations the playbook was never tuned against.
CHOICE_LETTERS = ["A", "B", "C", "D"]


def _to_dataset(rows: List[Dict]):
    """Wrap normalised examples as the `DatasetDict` the harness expects."""
    import datasets

    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def _render_fields(example: Dict) -> Dict:
    """
    Flatten one normalised example into the fields the YAML templates reference.

    `options` is a list, and a jinja template indexing into it would silently
    render "None" for a missing slot rather than fail. Explicit `option_a`..
    `option_d` keys make a shape change an error at this boundary instead.
    """
    options = example["options"]
    if len(options) != len(CHOICE_LETTERS):
        raise ValueError(f"{example['id']}: {len(options)} options, expected {len(CHOICE_LETTERS)}")
    fields = {
        "id": example["id"],
        "question": example["question"],
        "context": example.get("context", ""),
        "answer": CHOICE_LETTERS[example["gold_option_idx"]],
        "gold_option_idx": example["gold_option_idx"],
        "shuffle_perm": example.get("shuffle_perm", []),
        "subject": example.get("subject", ""),
    }
    for letter, option in zip(CHOICE_LETTERS, options):
        fields[f"option_{letter.lower()}"] = option
    return fields


def prepare(
    examples: Sequence[Dict],
    keep_ids: Optional[Sequence[str]] = None,
    shuffle: bool = True,
) -> List[Dict]:
    """
    Select, permute and flatten examples for the harness.

    Args:
        examples: Normalised examples from a loader.
        keep_ids: Restrict to these ids. Selection is by id, never by position.
        shuffle: Apply the seeded option permutation.

    Returns:
        Rows in the flat shape the YAML templates render, ordered by id so a
        run is reproducible and two languages line up.

    Raises:
        ValueError: If `keep_ids` names an item the corpus does not hold. A
            silently dropped id would shrink the evaluation set while every
            reported accuracy still computed.
    """
    rows = {e["id"]: e for e in examples}
    if keep_ids is not None:
        missing = sorted(set(keep_ids) - rows.keys())
        if missing:
            raise ValueError(
                f"{len(missing)} frozen split id(s) are not in the loaded "
                f"corpus, e.g. {missing[:3]}. The corpus and the manifest have "
                f"drifted apart."
            )
        rows = {i: rows[i] for i in keep_ids}

    prepared = [shuffle_example(e) if shuffle else e for e in rows.values()]
    return [_render_fields(e) for e in sorted(prepared, key=lambda e: e["id"])]


def _global_mmlu_builder(language: str) -> Callable:
    def build(**_kwargs):
        return _to_dataset(
            prepare(
                load_global_mmlu(language, "test"),
                keep_ids=load_manifest(GLOBAL_MMLU_EVAL)["item_ids"],
            )
        )

    return build


def _belebele_builder(language: str) -> Callable:
    def build(**_kwargs):
        return _to_dataset(
            prepare(
                load_belebele(language),
                keep_ids=load_manifest(BELEBELE_SPLIT)["item_ids"],
            )
        )

    return build


# Referenced from the YAMLs by `!function utils.<name>`.
global_mmlu_en = _global_mmlu_builder("en")
global_mmlu_ne = _global_mmlu_builder("ne")
belebele_en = _belebele_builder("en")
belebele_ne = _belebele_builder("ne")
