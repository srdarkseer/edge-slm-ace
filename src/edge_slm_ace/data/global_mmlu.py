"""Global-MMLU knowledge MCQ, in the shape the rest of the package expects.

Global-MMLU is 14,042 four-option questions over 57 MMLU subjects, released in
42 languages. Like Belebele the variants are *parallel* -- the same questions,
the same correct option, translated -- so an en/ne comparison is paired and
McNemar applies to it.

This is the study's **primary** task. Belebele is the negative control: ACE's
own paper names passage-answer tasks as the family its method does not help
(§5), while knowledge MCQ is where it reports headroom (DDXPlus +15.0). A study
that ran only Belebele would be predicting its own null.

Four properties of the raw files drive the design here, each verified against
the published parquet rather than assumed:

1. **`sample_id` is language-invariant.** It looks like
   "abstract_algebra/test/0" and is byte-identical across language configs. It
   is therefore the join key, the shuffle key, and the split key -- nothing
   here is ever derived from row position or from question text. A translated
   question hashes differently in every language; an id keyed on text would
   give English and Nepali different option permutations and silently break the
   item-matched pairing the whole design rests on.
2. **`answer` is a letter**, "A".."D", not an index and not the option text.
3. **Gold position is not uniform**: A 3222 / B 3462 / C 3582 / D 3776 across
   the test split, a 3.9-point spread that a letter-biased model can ride. The
   harness-side `process_docs` permutes options per item to remove it; see
   `data/shuffle.py`. This module reports gold as it is stored and leaves the
   permutation to the layer that also renders the prompt, so that adaptation
   and evaluation cannot disagree about option order.
4. **The dataset moved orgs.** It is published at `CohereLabs/Global-MMLU`;
   the `CohereForAI/Global-MMLU` id in older references 307-redirects there.
   Apache-2.0 either way.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from edge_slm_ace.utils.config import REPO_ROOT

# Short language code -> the HuggingFace config name. Global-MMLU already names
# its configs by ISO-639-1, so the mapping is the identity; it exists so that
# callers can use the same "en"/"ne" vocabulary they use for Belebele, where the
# file names are FLORES codes and the mapping is not the identity.
GLOBAL_MMLU_LANGUAGES: Dict[str, str] = {
    "en": "en",
    "ne": "ne",
}

GLOBAL_MMLU_DIR = Path("data/tasks/global_mmlu")

# `test` is what the study evaluates and adapts on, from disjoint frozen
# subsets. `dev` is held back entirely, as an independent sanity set that no
# split, no screening run and no grid cell ever touches.
SPLITS = ("test", "dev")

# Row counts, checked on load. An upstream reshuffle that changed these would
# invalidate every frozen split keyed to them, and we would rather fail than
# score a different dataset under the same name.
EXPECTED_ROWS: Dict[str, int] = {
    "test": 14042,
    "dev": 285,
}

_N_OPTIONS = 4

# A tuple, not the string "ABCD", and the distinction is load-bearing: `"" in
# "ABCD"` is True and `"ABCD".index("")` is 0, so a blank answer field would
# pass the membership check and silently become gold = A. So would "AB". Exact
# membership over a sequence of single characters is the only form that
# rejects both.
_ANSWER_LETTERS = ("A", "B", "C", "D")
_OPTION_FIELDS = ("option_a", "option_b", "option_c", "option_d")

# Fields every record must carry. Checked on load, because a silently renamed
# upstream field would mismap the gold answer rather than fail.
_REQUIRED_FIELDS = (
    "sample_id",
    "subject",
    "subject_category",
    "question",
    "answer",
) + _OPTION_FIELDS


def global_mmlu_path(language: str, split: str = "test") -> Path:
    """
    Path to one Global-MMLU language/split file.

    Args:
        language: Short code from GLOBAL_MMLU_LANGUAGES.
        split: One of SPLITS.

    Returns:
        Absolute path, resolved against the data root.

    Raises:
        KeyError: If the language is not one this project uses.
        ValueError: If the split is not one this dataset has.
    """
    if language not in GLOBAL_MMLU_LANGUAGES:
        raise KeyError(
            f"Unknown Global-MMLU language '{language}'. " f"Known: {sorted(GLOBAL_MMLU_LANGUAGES)}"
        )
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Known: {sorted(SPLITS)}")
    config = GLOBAL_MMLU_LANGUAGES[language]
    return REPO_ROOT / GLOBAL_MMLU_DIR / config / f"{split}.jsonl"


def item_id(record: Dict) -> str:
    """
    Stable identifier for a question, shared across every language variant.

    Unlike Belebele -- where the id has to be synthesised from `(link,
    question_number)` -- Global-MMLU ships one: `sample_id` is already
    byte-identical across language configs, so it is returned unchanged.

    It is returned rather than hashed on purpose. The id is read by humans in
    frozen split manifests and in error messages, and "abstract_algebra/test/0"
    says which subject a leak came from where a hex digest would not.

    Args:
        record: A raw Global-MMLU record.

    Returns:
        The record's `sample_id`.

    Raises:
        ValueError: If `sample_id` is absent or blank. This never falls back to
            hashing the question text: translated text hashes differently in
            every language, so a text-derived id would hand English and Nepali
            different ids for the same question, and every paired comparison
            downstream would quietly compare unrelated items.
    """
    raw = record.get("sample_id")
    if raw is None or not str(raw).strip():
        raise ValueError(
            "Global-MMLU record has no usable 'sample_id'. Refusing to "
            "substitute a text-derived id: it would differ between languages "
            "and break the paired design."
        )
    return str(raw)


def _normalize(record: Dict, language: str, domain: str) -> Dict:
    """Convert one raw Global-MMLU record into the package's example schema."""
    missing = [f for f in _REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Global-MMLU record missing fields: {', '.join(missing)}")

    letter = str(record["answer"]).strip().upper()
    if letter not in _ANSWER_LETTERS:
        raise ValueError(f"answer is not one of {_ANSWER_LETTERS}: {record['answer']!r}")
    gold = _ANSWER_LETTERS.index(letter)

    options = [record[f] for f in _OPTION_FIELDS]
    blank = [f for f, o in zip(_OPTION_FIELDS, options) if o is None or not str(o).strip()]
    if blank:
        raise ValueError(
            f"{item_id(record)}: blank option(s) {', '.join(blank)}. A blank "
            f"option is scored as an empty continuation, which is not a "
            f"four-way choice."
        )

    return {
        "id": item_id(record),
        "language": language,
        "domain": domain,
        # Knowledge MCQ has no passage. The key is present and empty so that
        # one prompt renderer can serve both tasks without branching on task
        # name -- see harness/prompts.py.
        "context": "",
        "question": record["question"],
        "options": [str(o) for o in options],
        "gold_option_idx": gold,
        "answer": str(options[gold]),
        # Carried for stratification (the frozen eval subsample is stratified
        # by subject) and for per-subject error analysis.
        "subject": record["subject"],
        "subject_category": record["subject_category"],
    }


def load_global_mmlu(
    language: str,
    split: str = "test",
    path: Optional[Path] = None,
    domain: Optional[str] = None,
) -> List[Dict]:
    """
    Load one Global-MMLU language/split, normalised and in a language-independent
    order.

    Rows are returned sorted by `id`, so `load_global_mmlu("en")[i]` and
    `load_global_mmlu("ne")[i]` are the same question. The published files
    happen to share a row order today, but nothing upstream guarantees that and
    a future re-release could change it, so order is imposed here rather than
    trusted.

    The sort is lexicographic on `sample_id`, which orders "…/test/10" before
    "…/test/2". That is fine and deliberate: the only requirement is that the
    order is deterministic and identical across languages. Nothing downstream
    reads the numeric suffix.

    Args:
        language: Short code ("en", "ne").
        split: One of SPLITS. Defaults to "test"; `dev` is the untouched sanity
            set and no study code should load it.
        path: Override the file location.
        domain: Playbook domain to file lessons under. Defaults to
            "global_mmlu_<language>", which keeps an English playbook and a
            Nepali playbook separate. Pass an explicit shared value to study
            cross-lingual playbook transfer.

    Returns:
        Examples in the package's schema, sorted by id.

    Raises:
        FileNotFoundError: If the file is absent, naming `make data`.
        ValueError: On a duplicate id or an unexpected row count.
    """
    source = path or global_mmlu_path(language, split)
    domain = domain or f"global_mmlu_{language}"

    if not source.exists():
        raise FileNotFoundError(
            f"{source} is missing. Run `make data` to fetch Global-MMLU "
            f"(it is not committed; only the frozen split manifests are)."
        )

    with open(source, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Only enforced when reading a real split file. An explicit `path` is a
    # test fixture or a debugging slice, and holding it to 14,042 rows would
    # make the loader untestable without the full dataset.
    if path is None and len(records) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{source.name}: {len(records)} rows, expected "
            f"{EXPECTED_ROWS[split]}. The upstream dataset has changed; every "
            f"frozen split keyed to it is now invalid."
        )

    examples = [_normalize(r, language, domain) for r in records]

    ids = [e["id"] for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source} contains duplicate item ids")

    return sorted(examples, key=lambda e: e["id"])


def assert_parallel(by_language: Dict[str, Sequence[Dict]]) -> None:
    """
    Verify two loaded languages really are the same questions.

    The paired design assumes en item *i* and ne item *i* are a translation
    pair with the same correct option. Both halves are checked, because they
    fail differently: a mismatched id set means the languages were built from
    different releases, while an agreeing id set with disagreeing gold means a
    translation reordered the options -- which looks like a model error and is
    a data error.

    Args:
        by_language: Mapping of short language code to loaded examples.

    Raises:
        ValueError: If the id sets differ, or if any shared id disagrees on
            `gold_option_idx`.
    """
    if len(by_language) < 2:
        raise ValueError("assert_parallel needs at least two languages")

    gold_by_language = {
        lang: {e["id"]: e["gold_option_idx"] for e in rows} for lang, rows in by_language.items()
    }
    reference_lang, reference = next(iter(gold_by_language.items()))

    for lang, gold in gold_by_language.items():
        if lang == reference_lang:
            continue
        only_ref = set(reference) - set(gold)
        only_other = set(gold) - set(reference)
        if only_ref or only_other:
            raise ValueError(
                f"{reference_lang} and {lang} are not parallel: "
                f"{len(only_ref)} id(s) only in {reference_lang} "
                f"(e.g. {sorted(only_ref)[:3]}), "
                f"{len(only_other)} only in {lang} (e.g. {sorted(only_other)[:3]})"
            )
        disagree = sorted(i for i, g in gold.items() if reference[i] != g)
        if disagree:
            raise ValueError(
                f"{reference_lang} and {lang} disagree on the correct option "
                f"for {len(disagree)} item(s), e.g. {disagree[:3]}. The "
                f"translation reordered options; the pairing is unusable."
            )


def subjects(examples: Iterable[Dict]) -> Dict[str, int]:
    """
    Count examples per subject.

    Args:
        examples: Loaded examples.

    Returns:
        Subject -> count, sorted by subject so the output is diff-stable.
    """
    counts: Dict[str, int] = {}
    for example in examples:
        counts[example["subject"]] = counts.get(example["subject"], 0) + 1
    return dict(sorted(counts.items()))
