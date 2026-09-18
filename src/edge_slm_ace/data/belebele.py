"""Belebele reading comprehension, in the shape the rest of the package expects.

Belebele is 900 multiple-choice questions over FLORES passages, released in 122
language variants. The variants are *parallel*: the same 900 questions, the same
correct option, translated. That is what makes an en/ne comparison paired rather
than merely matched, so a difference between the two can be tested with McNemar
on the same items.

Four properties of the raw files drive the design here, all verified against
`facebook/belebele` rather than assumed:

1. **The language files are not in the same row order.** They contain the same
   `(link, question_number)` keys, but positionally row *i* of `eng_Latn` is not
   row *i* of `npi_Deva`. Splitting or joining by index silently pairs unrelated
   questions -- so every id here is derived from the content key, never from
   position.
2. **`correct_answer_num` is a 1-based string**, not a 0-based int.
3. **Gold position is close to uniform already** (206/251/246/197 across the
   four slots), unlike the SciQ files this repo used to carry, where gold was
   first in 1000/1000 rows. This module therefore does NOT permute option
   order, and cannot: the harness loads its own copy of the split and renders
   the options itself, so a permutation applied here would change what
   adaptation sees without changing what is scored -- which is the arm
   asymmetry the whole prompt layer exists to prevent. The residual position
   bias is a limitation to state, not something the loader can correct.
4. **The 900 questions cover only 488 passages**, 412 of which carry two. The
   split unit is therefore the passage, not the question: a question-level
   split adapts the playbook on a passage and then scores it on the sibling
   question about the same 79 words. That leak inflates the adapted arm and
   only the adapted arm -- exactly the direction that would manufacture the
   result this study is trying to test. See `passage_split`.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edge_slm_ace.utils.config import BELEBELE_ADAPTATION_PASSAGES, REPO_ROOT

# Short language code -> the FLORES-200 code Belebele files are named by.
BELEBELE_LANGUAGES: Dict[str, str] = {
    "en": "eng_Latn",
    "ne": "npi_Deva",
}

BELEBELE_DIR = Path("data/tasks/belebele")

# Fields every Belebele record must carry. Checked on load, because a silently
# renamed upstream field would mismap the gold answer rather than fail.
_REQUIRED_FIELDS = (
    "link",
    "question_number",
    "flores_passage",
    "question",
    "mc_answer1",
    "mc_answer2",
    "mc_answer3",
    "mc_answer4",
    "correct_answer_num",
)

_N_OPTIONS = 4


def belebele_path(language: str) -> Path:
    """
    Path to a Belebele language file.

    Args:
        language: Short code from BELEBELE_LANGUAGES, or a FLORES code.

    Returns:
        Absolute path, resolved against the data root.

    Raises:
        KeyError: If the language is not one this project uses.
    """
    if language in BELEBELE_LANGUAGES:
        flores = BELEBELE_LANGUAGES[language]
    elif language in BELEBELE_LANGUAGES.values():
        flores = language
    else:
        raise KeyError(
            f"Unknown Belebele language '{language}'. "
            f"Known: {sorted(BELEBELE_LANGUAGES)} or {sorted(BELEBELE_LANGUAGES.values())}"
        )
    return REPO_ROOT / BELEBELE_DIR / f"{flores}.jsonl"


def item_id(record: Dict) -> str:
    """
    Stable identifier for a question, shared across every language variant.

    `(link, question_number)` is unique across all 900 rows and identical
    between language files, which makes it the only safe join key. The link is
    a long URL, so it is hashed rather than embedded.

    Do not shorten the link to its last path segment: six different passages end
    in ".../Introduction", and collapsing them collides. The uniqueness check in
    `load_belebele` exists because that mistake mismaps gold answers rather than
    failing.

    Args:
        record: A raw Belebele record.

    Returns:
        An id of the form "bel-<10 hex chars>-q<question_number>".
    """
    return f"{passage_id(record)}-q{record['question_number']}"


def passage_id(record: Dict) -> str:
    """
    Identifier for the passage a question is asked about.

    412 of Belebele's 488 passages carry two questions. Splitting by question
    therefore puts a passage's two questions on opposite sides of the split
    roughly half the time -- the playbook adapts on a passage, and is then
    evaluated on its sibling question about the same 79 words. That is a leak
    that inflates the adapted arm and only the adapted arm, which is the
    direction that would manufacture our headline result.

    Args:
        record: A raw Belebele record.

    Returns:
        An id of the form "bel-<10 hex chars>".
    """
    digest = hashlib.blake2s(str(record["link"]).encode("utf-8"), digest_size=5).hexdigest()
    return f"bel-{digest}"


def passage_of(item: str) -> str:
    """
    The passage id an item id belongs to.

    Args:
        item: An id from `item_id`, "bel-<digest>-q<n>".

    Returns:
        The "bel-<digest>" prefix.

    Raises:
        ValueError: If the id is not in that form. It raises rather than
            returning the input, because a silently passed-through id would
            make every item look like its own passage -- and the
            passage-overlap assertion would then pass on a split that leaks.
    """
    head, sep, tail = str(item).rpartition("-q")
    if not sep or not head or not tail.isdigit():
        raise ValueError(
            f"{item!r} is not a Belebele item id of the form 'bel-<digest>-q<n>', "
            f"so its passage cannot be determined."
        )
    return head


def assert_zero_passage_overlap(adaptation_ids: Sequence[str], eval_ids: Sequence[str]) -> None:
    """
    Verify no passage appears on both sides of a split.

    Args:
        adaptation_ids: Ids the playbook is built on.
        eval_ids: Ids the study reports over.

    Raises:
        ValueError: If any passage is shared, naming how many and giving
            examples. Also if the two sides share an item id outright, which
            would be a stronger version of the same failure.
    """
    shared_items = set(adaptation_ids) & set(eval_ids)
    if shared_items:
        raise ValueError(
            f"{len(shared_items)} item(s) are in both splits, " f"e.g. {sorted(shared_items)[:3]}"
        )

    adaptation_passages = {passage_of(i) for i in adaptation_ids}
    eval_passages = {passage_of(i) for i in eval_ids}
    shared = adaptation_passages & eval_passages
    if shared:
        raise ValueError(
            f"{len(shared)} passage(s) appear in both the adaptation and the "
            f"evaluation split, e.g. {sorted(shared)[:3]}. The playbook would "
            f"be adapted on a passage and then scored on another question "
            f"about it."
        )


def _normalize(record: Dict, language: str, domain: str) -> Dict:
    """Convert one raw Belebele record into the package's example schema."""
    missing = [f for f in _REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Belebele record missing fields: {', '.join(missing)}")

    gold = int(record["correct_answer_num"]) - 1  # 1-based in the file
    if not 0 <= gold < _N_OPTIONS:
        raise ValueError(f"correct_answer_num out of range: {record['correct_answer_num']!r}")

    options = [record[f"mc_answer{i}"] for i in range(1, _N_OPTIONS + 1)]

    return {
        "id": item_id(record),
        "language": language,
        "domain": domain,
        # The passage is the context. It is long -- 79 words on average in
        # English, up to 217 -- which is why prompt-length bounds and the
        # truncation health check matter more here than they did on SciQ.
        "context": record["flores_passage"],
        "question": record["question"],
        "options": options,
        "gold_option_idx": gold,
        "answer": options[gold],
    }


def load_belebele(
    language: str,
    path: Optional[Path] = None,
    domain: Optional[str] = None,
) -> List[Dict]:
    """
    Load one Belebele language, normalised and in a language-independent order.

    Rows are returned sorted by `id`, so `load_belebele("en")[i]` and
    `load_belebele("ne")[i]` are the same question. The files themselves are not
    in the same order, so this is the only safe way to line them up.

    Args:
        language: Short code ("en", "ne") or FLORES code.
        path: Override the file location.
        domain: Playbook domain to file lessons under. Defaults to
            "belebele_<language>", which keeps an English playbook and a Nepali
            playbook separate. Pass an explicit shared value to study
            cross-lingual playbook transfer.

    Returns:
        Examples in the package's schema, sorted by id.
    """
    source = path or belebele_path(language)
    short = next((k for k, v in BELEBELE_LANGUAGES.items() if v == source.stem), language)
    domain = domain or f"belebele_{short}"

    with open(source, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    examples = [_normalize(r, short, domain) for r in records]

    ids = [e["id"] for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source} contains duplicate item ids")

    return sorted(examples, key=lambda e: e["id"])


def passage_split(
    item_ids: Sequence[str],
    adaptation_passages: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split ids into an adaptation set and a frozen evaluation set, by passage.

    The unit of randomisation is the passage, not the question, because a
    passage is the unit of leakage: 412 of 488 passages carry two questions,
    and a question-level split routinely hands the playbook one of a pair and
    then scores it on the other.

    Deterministic in `(seed, ids)` and independent of the order ids arrive in,
    so every language and every arm gets the identical split. If the split were
    derived from row order, English and Nepali would adapt on different
    passages and the paired comparison would be meaningless.

    Args:
        item_ids: All ids in the language.
        adaptation_passages: How many passages to reserve for the playbook.
        seed: Run seed.

    Returns:
        (adaptation_ids, eval_ids), both sorted, passage-disjoint. The item
        counts are not exactly proportional to the passage counts, because 76
        passages carry one question and 412 carry two.

    Raises:
        ValueError: If the adaptation set would take every passage or none.
    """
    by_passage: Dict[str, List[str]] = {}
    for item in sorted(set(item_ids)):
        by_passage.setdefault(passage_of(item), []).append(item)

    passages = sorted(by_passage)
    if not 0 < adaptation_passages < len(passages):
        raise ValueError(
            f"adaptation_passages must be in (0, {len(passages)}); got {adaptation_passages}"
        )

    shuffled = list(passages)
    random.Random(seed).shuffle(shuffled)
    chosen = set(shuffled[:adaptation_passages])

    adaptation = sorted(i for p in chosen for i in by_passage[p])
    evaluation = sorted(i for p in passages if p not in chosen for i in by_passage[p])

    # Belt and braces: the construction above cannot leak, so this catches a
    # future edit to it rather than a bug in it.
    assert_zero_passage_overlap(adaptation, evaluation)
    return adaptation, evaluation


def study_split(
    item_ids: Sequence[str],
    seed: int,
    adaptation_passages: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    The study's one split of Belebele into adaptation and evaluation.

    Every entrypoint must call this rather than `passage_split` directly.
    Screening and the grid each chose their own size -- 400 and 200 -- and
    since the split was one shuffle sliced at that index, the 200 items between
    them were screened on *and* scored on. Model selection was therefore made
    on items the study reports over. One function, one split, no second
    constant to disagree with the first.

    Args:
        item_ids: All ids in the language.
        seed: Run seed.
        adaptation_passages: Override, for tests and debugging only. A run that
            passes one is no longer on the study's split.

    Returns:
        (adaptation_ids, eval_ids), both sorted, disjoint, and passage-disjoint.
    """
    return passage_split(item_ids, adaptation_passages or BELEBELE_ADAPTATION_PASSAGES, seed)


# lm-evaluation-harness task names for the language variants we use.
HARNESS_TASKS: Dict[str, str] = {
    "en": "belebele_eng_Latn",
    "ne": "belebele_npi_Deva",
}


def harness_indices(
    language: str, item_ids: Sequence[str], path: Optional[Path] = None
) -> List[int]:
    """
    Positional indices the harness needs in order to evaluate `item_ids`.

    `lm_eval.simple_evaluate(samples={task: [...]})` selects documents by their
    position in the loaded dataset, which is raw file order. The English and
    Nepali files hold the same 900 questions in *different* orders, so passing
    one index list to both tasks would evaluate two disjoint question sets while
    looking exactly like a matched comparison.

    This maps a language-independent id set onto the positions that language
    actually stores them at.

    Args:
        language: Short code ("en", "ne") or FLORES code.
        item_ids: Ids to evaluate, from `study_split`.
        path: Override the file location.

    Returns:
        Sorted positional indices, one per id.

    Raises:
        KeyError: If an id is absent from this language file.
    """
    source = path or belebele_path(language)
    # Number the records, not the lines. `enumerate(f)` counted blank lines that
    # the filter then dropped, so one blank line anywhere in the file shifted
    # every position after it -- and the harness, which skips blanks when it
    # loads the split, would have scored the wrong document while reporting a
    # clean accuracy. That is the failure this whole function exists to prevent.
    positions = {}
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                positions[item_id(json.loads(line))] = len(positions)

    wanted = set(item_ids)
    missing = wanted - positions.keys()
    if missing:
        raise KeyError(
            f"{len(missing)} id(s) not present in {source.name}, " f"e.g. {sorted(missing)[:3]}"
        )
    return sorted(positions[i] for i in wanted)
