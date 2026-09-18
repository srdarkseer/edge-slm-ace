"""Seeded per-item option permutation, keyed so English and Nepali agree.

Both tasks put gold in a non-uniform position -- Global-MMLU's test split is
A 3222 / B 3462 / C 3582 / D 3776 -- and a small model with a letter bias
converts that into accuracy it has not earned. Permuting the options per item
removes the bias without touching the questions.

The permutation key is the whole point of this module, and it is the guard G1
names.

A permutation seeded on *text* -- the question, the passage, a hash of the
rendered prompt -- is a different permutation in every language, because a
translation hashes differently. English item *i* and Nepali item *i* would then
be scored with their options in different orders. Every number downstream still
computes: accuracies come out, McNemar runs, the paired design reports a delta.
It is simply no longer paired, and nothing in the pipeline says so.

So the key is the language-invariant id and nothing else:

    Global-MMLU   `sample_id`, shipped identical across configs
    Belebele      `(link, question_number)`, via `item_id`

Both are ids the *dataset* guarantees are the same across languages, not values
this project derives from content. `permutation_for` therefore takes an id, not
an example, so there is no path by which text can reach the seed.

One more constraint: the permutation must be applied by the layer that renders
the prompt for scoring, not by a loader. Belebele's loader deliberately does
not permute, because the harness loads its own copy of the split -- a
permutation applied loader-side would change what adaptation sees without
changing what is scored, which is the arm asymmetry the prompt layer exists to
prevent. These functions are called from `process_docs` (harness side) and from
the adaptation loop through the same helper, so both see one order.
"""

import hashlib
import random
from typing import Dict, List, Sequence, Tuple

# Fixed at pre-registration, like the subsample seed and for the same reason:
# the option order must not move when the run seed moves, or the three study
# seeds would be scoring three different renderings of the same item.
SHUFFLE_SEED = 20260915


def permutation_for(item_id: str, n_options: int = 4, seed: int = SHUFFLE_SEED) -> List[int]:
    """
    The permutation for one item, as source indices in destination order.

    `perm[j] == i` means "destination slot j holds the option that was at
    source index i".

    Deterministic in `(item_id, n_options, seed)` and in nothing else. It takes
    an id rather than an example precisely so that no caller can accidentally
    seed it on translated text -- see the module docstring.

    Args:
        item_id: A language-invariant item id.
        n_options: How many options the item has.
        seed: Fixed at SHUFFLE_SEED for the study.

    Returns:
        A permutation of range(n_options).

    Raises:
        ValueError: If `item_id` is blank, which would give every such item the
            same permutation, or if `n_options` is not positive.
    """
    if not str(item_id).strip():
        raise ValueError(
            "permutation_for needs a non-empty item id; a blank key would give "
            "every affected item the same option order."
        )
    if n_options < 1:
        raise ValueError(f"n_options must be positive; got {n_options}")

    # Hash the key rather than feeding the string to Random() directly:
    # `Random(str)` is seeded by the string's hash, and `PYTHONHASHSEED`
    # randomises that per process. The permutation would then differ between
    # the adaptation run and the evaluation run of the same item.
    digest = hashlib.blake2s(f"{seed}:{item_id}".encode("utf-8"), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "big"))

    order = list(range(n_options))
    rng.shuffle(order)
    return order


def apply_permutation(
    options: Sequence[str], gold_option_idx: int, permutation: Sequence[int]
) -> Tuple[List[str], int]:
    """
    Reorder options and follow the gold answer to its new slot.

    Args:
        options: Options in source order.
        gold_option_idx: Index of the correct option, in source order.
        permutation: From `permutation_for`.

    Returns:
        (reordered options, new gold index).

    Raises:
        ValueError: If the permutation does not match the options, or if gold
            is out of range. Both are silent-mislabel bugs otherwise: a short
            permutation would drop options, and an out-of-range gold would
            either raise deep inside indexing or, worse, wrap around.
    """
    if sorted(permutation) != list(range(len(options))):
        raise ValueError(
            f"permutation {list(permutation)} is not a permutation of " f"range({len(options)})"
        )
    if not 0 <= gold_option_idx < len(options):
        raise ValueError(
            f"gold_option_idx {gold_option_idx} is out of range for {len(options)} options"
        )

    reordered = [options[i] for i in permutation]
    new_gold = permutation.index(gold_option_idx)
    return reordered, new_gold


def shuffle_example(example: Dict, seed: int = SHUFFLE_SEED) -> Dict:
    """
    Return a copy of an example with its options permuted.

    The permutation is recorded on the result as `shuffle_perm`, so a rendered
    prompt can always be traced back to the source ordering during error
    analysis -- without it, "the model picked C" cannot be mapped to an option
    in the published dataset.

    Args:
        example: A normalised example, with "id", "options" and
            "gold_option_idx".
        seed: Fixed at SHUFFLE_SEED for the study.

    Returns:
        A new dict; the input is not mutated.
    """
    permutation = permutation_for(example["id"], len(example["options"]), seed)
    options, gold = apply_permutation(example["options"], example["gold_option_idx"], permutation)

    shuffled = dict(example)
    shuffled["options"] = options
    shuffled["gold_option_idx"] = gold
    shuffled["answer"] = options[gold]
    shuffled["shuffle_perm"] = list(permutation)
    return shuffled


def assert_shuffle_parity(examples_by_language: Dict[str, Sequence[Dict]]) -> None:
    """
    Verify every language gives the same item the same option order.

    This is the assertion G1 asks for. It is cheap, and it is the only thing
    standing between a text-seeded permutation and a paired comparison that
    silently compares unrelated renderings.

    Args:
        examples_by_language: Short language code -> loaded examples.

    Raises:
        ValueError: If any shared id gets different permutations, naming the
            ids. Also if the languages do not share an id set, since parity is
            meaningless without that.
    """
    if len(examples_by_language) < 2:
        raise ValueError("assert_shuffle_parity needs at least two languages")

    perms = {
        lang: {e["id"]: tuple(shuffle_example(e)["shuffle_perm"]) for e in rows}
        for lang, rows in examples_by_language.items()
    }
    reference_lang, reference = next(iter(perms.items()))

    for lang, by_id in perms.items():
        if lang == reference_lang:
            continue
        if set(by_id) != set(reference):
            raise ValueError(
                f"{reference_lang} and {lang} do not share an id set, so their "
                f"option orders cannot be compared"
            )
        differing = sorted(i for i, p in by_id.items() if reference[i] != p)
        if differing:
            raise ValueError(
                f"{len(differing)} item(s) get a different option order in "
                f"{lang} than in {reference_lang}, e.g. {differing[:3]}. The "
                f"permutation is being seeded on something language-dependent; "
                f"the paired comparison is invalid."
            )
