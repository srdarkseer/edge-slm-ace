"""Tests for the seeded option permutation.

The headline test is `test_shuffle_parity_across_languages`, which is the guard
G1 names: English and Nepali must give the same item the same option order.
Everything else here exists to make that guarantee hard to break by accident.
"""

import pytest

from edge_slm_ace.data.belebele import belebele_path, load_belebele
from edge_slm_ace.data.global_mmlu import GLOBAL_MMLU_LANGUAGES, global_mmlu_path, load_global_mmlu
from edge_slm_ace.data.shuffle import (
    SHUFFLE_SEED,
    apply_permutation,
    assert_shuffle_parity,
    permutation_for,
    shuffle_example,
)

BELEBELE_PRESENT = belebele_path("en").exists() and belebele_path("ne").exists()
GLOBAL_MMLU_PRESENT = all(global_mmlu_path(lang, "test").exists() for lang in GLOBAL_MMLU_LANGUAGES)


def example(item_id="a/test/0", gold=1):
    return {
        "id": item_id,
        "options": ["w", "x", "y", "z"],
        "gold_option_idx": gold,
        "answer": ["w", "x", "y", "z"][gold],
    }


class TestPermutationFor:
    def test_is_a_permutation(self):
        assert sorted(permutation_for("a/test/0")) == [0, 1, 2, 3]

    def test_is_deterministic(self):
        assert permutation_for("a/test/0") == permutation_for("a/test/0")

    def test_depends_only_on_the_id(self):
        """The whole point: nothing language-dependent may reach the seed."""
        assert permutation_for("a/test/0") == permutation_for("a/test/0", 4, SHUFFLE_SEED)

    def test_different_items_get_different_orders(self):
        orders = {tuple(permutation_for(f"a/test/{i}")) for i in range(40)}
        assert len(orders) > 10, "the permutation is barely varying across items"

    def test_a_different_seed_gives_a_different_order(self):
        different = [
            permutation_for(f"a/test/{i}", 4, 1) != permutation_for(f"a/test/{i}", 4, 2)
            for i in range(40)
        ]
        assert any(different)

    def test_gold_position_is_spread_across_slots(self):
        """What the permutation is for: no slot may stay preferred."""
        slots = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(4000):
            perm = permutation_for(f"a/test/{i}")
            slots[perm.index(0)] += 1
        assert min(slots.values()) > 800, slots

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_a_blank_id_raises(self, bad):
        """A blank key would give every affected item the same order."""
        with pytest.raises(ValueError, match="non-empty item id"):
            permutation_for(bad)

    def test_a_nonpositive_option_count_raises(self):
        with pytest.raises(ValueError, match="n_options"):
            permutation_for("a/test/0", 0)

    def test_is_stable_across_processes(self):
        """
        Regression guard. `random.Random(some_string)` seeds from the string's
        hash, which PYTHONHASHSEED randomises per process -- adaptation and
        evaluation would then permute the same item differently. The id is
        hashed to an int first, so the order is fixed forever.

        These values were produced by this implementation; if they change, the
        option order of every already-scored item changed with them.
        """
        assert permutation_for("abstract_algebra/test/0") == [2, 3, 0, 1]
        assert permutation_for("bel-0123456789-q1") == [1, 2, 3, 0]


class TestApplyPermutation:
    def test_gold_follows_its_option(self):
        options, gold = apply_permutation(["w", "x", "y", "z"], 1, [2, 1, 3, 0])
        assert options == ["y", "x", "z", "w"]
        assert options[gold] == "x"

    def test_the_identity_permutation_changes_nothing(self):
        options, gold = apply_permutation(["w", "x", "y", "z"], 3, [0, 1, 2, 3])
        assert (options, gold) == (["w", "x", "y", "z"], 3)

    def test_a_short_permutation_raises(self):
        """Otherwise it would silently drop options."""
        with pytest.raises(ValueError, match="not a permutation"):
            apply_permutation(["w", "x", "y", "z"], 0, [0, 1, 2])

    def test_a_repeated_index_raises(self):
        with pytest.raises(ValueError, match="not a permutation"):
            apply_permutation(["w", "x", "y", "z"], 0, [0, 0, 1, 2])

    @pytest.mark.parametrize("gold", [-1, 4])
    def test_an_out_of_range_gold_raises(self, gold):
        with pytest.raises(ValueError, match="out of range"):
            apply_permutation(["w", "x", "y", "z"], gold, [0, 1, 2, 3])


class TestShuffleExample:
    def test_the_answer_text_survives_the_permutation(self):
        """The reordering must move gold, not relabel it."""
        source = example(gold=2)
        shuffled = shuffle_example(source)
        assert shuffled["answer"] == source["answer"] == "y"
        assert shuffled["options"][shuffled["gold_option_idx"]] == "y"

    def test_the_input_is_not_mutated(self):
        source = example()
        shuffle_example(source)
        assert source["options"] == ["w", "x", "y", "z"]
        assert "shuffle_perm" not in source

    def test_the_permutation_is_recorded_for_error_analysis(self):
        """Without it, 'the model picked C' maps to nothing in the dataset."""
        shuffled = shuffle_example(example())
        assert shuffled["shuffle_perm"] == permutation_for(shuffled["id"])

    def test_the_option_set_is_unchanged(self):
        shuffled = shuffle_example(example())
        assert sorted(shuffled["options"]) == ["w", "x", "y", "z"]


class TestParityAssertion:
    def test_identical_permutations_pass(self):
        rows = [example("a/test/0"), example("a/test/1")]
        assert_shuffle_parity({"en": rows, "ne": rows})

    def test_a_translated_item_keeps_its_order(self):
        """
        The realistic case: same ids, different text. If the permutation ever
        reaches for the text, this is what catches it.
        """
        en = [{**example("a/test/0"), "question": "What is the degree?"}]
        ne = [{**example("a/test/0"), "question": "डिग्री के हो?"}]
        assert_shuffle_parity({"en": en, "ne": ne})

    def test_a_text_seeded_permutation_would_be_caught(self):
        """
        Simulate the G1 bug by handing the two languages different ids for the
        same item, which is exactly what hashing translated text produces.
        """
        with pytest.raises(ValueError, match="do not share an id set"):
            assert_shuffle_parity({"en": [example("a/test/0")], "ne": [example("2d1f9c4e8a")]})

    def test_differing_orders_on_a_shared_id_are_caught(self, monkeypatch):
        """
        Same ids, but the permutation is rigged to depend on the question text
        -- the exact shape of the G1 bug. The assertion must catch it.
        """
        import edge_slm_ace.data.shuffle as shuffle_module

        real = shuffle_module.permutation_for

        def text_seeded(item_id, n_options=4, seed=SHUFFLE_SEED):
            return real(f"{item_id}:{text_seeded.current}", n_options, seed)

        def rigged(ex, seed=SHUFFLE_SEED):
            text_seeded.current = ex["question"]
            monkeypatch.setattr(shuffle_module, "permutation_for", text_seeded)
            try:
                return real_shuffle(ex, seed)
            finally:
                monkeypatch.setattr(shuffle_module, "permutation_for", real)

        real_shuffle = shuffle_module.shuffle_example
        monkeypatch.setattr(shuffle_module, "shuffle_example", rigged)

        en = [{**example("a/test/0"), "question": "What is the degree?"}]
        ne = [{**example("a/test/0"), "question": "डिग्री के हो?"}]
        with pytest.raises(ValueError, match="different option order"):
            shuffle_module.assert_shuffle_parity({"en": en, "ne": ne})

    def test_one_language_is_not_a_parity_claim(self):
        with pytest.raises(ValueError, match="at least two languages"):
            assert_shuffle_parity({"en": [example()]})


@pytest.mark.skipif(not GLOBAL_MMLU_PRESENT, reason="Global-MMLU not fetched")
def test_shuffle_parity_across_languages_global_mmlu():
    """G1, on the primary task's full 14,042 items."""
    assert_shuffle_parity({lang: load_global_mmlu(lang, "test") for lang in ("en", "ne")})


@pytest.mark.skipif(not BELEBELE_PRESENT, reason="Belebele not fetched")
def test_shuffle_parity_across_languages_belebele():
    """G1, on the negative control's 900 items."""
    assert_shuffle_parity({lang: load_belebele(lang) for lang in ("en", "ne")})


@pytest.mark.skipif(not GLOBAL_MMLU_PRESENT, reason="Global-MMLU not fetched")
def test_the_shuffle_flattens_gold_position_on_real_data():
    """
    A 3222 / B 3462 / C 3582 / D 3776 going in; near-uniform coming out. This
    is the bias the permutation exists to remove, measured rather than assumed.
    """
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for row in load_global_mmlu("ne", "test"):
        counts[shuffle_example(row)["gold_option_idx"]] += 1
    spread = max(counts.values()) - min(counts.values())
    assert spread < 200, f"gold position is still skewed after shuffling: {counts}"
