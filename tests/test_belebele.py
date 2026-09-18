"""Tests for the Belebele data layer.

The whole en/ne claim rests on the two language files being the *same* 900
questions. They are -- but not in the same row order, which makes every
position-based shortcut a silent methodology bug rather than an error.
"""

import json

import pytest

from edge_slm_ace.utils import BELEBELE_ADAPTATION_PASSAGES
from edge_slm_ace.data.belebele import (
    BELEBELE_LANGUAGES,
    HARNESS_TASKS,
    assert_zero_passage_overlap,
    belebele_path,
    harness_indices,
    item_id,
    load_belebele,
    passage_id,
    passage_of,
    passage_split,
    study_split,
)

pytestmark = pytest.mark.skipif(
    not belebele_path("en").exists() or not belebele_path("ne").exists(),
    reason="Belebele not fetched; run scripts/fetch_belebele.py",
)


@pytest.fixture(scope="module")
def languages():
    return load_belebele("en"), load_belebele("ne")


class TestParallelism:
    def test_both_languages_have_the_same_items(self, languages):
        en, ne = languages
        assert [e["id"] for e in en] == [e["id"] for e in ne]

    def test_gold_agrees_across_languages(self, languages):
        """A translated question keeps its answer; if not, the pairing is wrong."""
        en, ne = languages
        assert all(a["gold_option_idx"] == b["gold_option_idx"] for a, b in zip(en, ne))

    def test_files_are_not_in_the_same_order(self):
        """The premise for every id-based join in this module."""
        raw = {}
        for lang in ("en", "ne"):
            with open(belebele_path(lang), encoding="utf-8") as f:
                raw[lang] = [item_id(json.loads(line)) for line in f if line.strip()]
        assert set(raw["en"]) == set(raw["ne"])
        assert raw["en"] != raw["ne"], "if these ever align, the trap is dormant, not gone"


class TestItemIds:
    def test_ids_are_unique(self, languages):
        en, _ = languages
        ids = [e["id"] for e in en]
        assert len(set(ids)) == len(ids) == 900

    def test_truncating_the_link_would_collide(self):
        """Why the id hashes the whole link: six passages end in '/Introduction'."""
        with open(belebele_path("en"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        naive = {(str(r["link"]).rsplit("/", 1)[-1], r["question_number"]) for r in rows}
        assert len(naive) < len(rows), "the naive key is expected to collide"
        assert len({item_id(r) for r in rows}) == len(rows)


class TestPassageIds:
    def test_an_item_id_carries_its_passage_as_a_prefix(self, languages):
        en, _ = languages
        for example in en:
            assert example["id"].startswith(passage_of(example["id"]) + "-q")

    def test_the_900_questions_cover_488_passages(self, languages):
        """The premise of the whole passage-level split."""
        en, _ = languages
        assert len({passage_of(e["id"]) for e in en}) == 488

    def test_most_passages_carry_two_questions(self, languages):
        """412 of 488. That is how often a question-level split would leak."""
        en, _ = languages
        counts = {}
        for example in en:
            key = passage_of(example["id"])
            counts[key] = counts.get(key, 0) + 1
        assert sorted(counts.values())[-1] == 2
        assert sum(1 for c in counts.values() if c == 2) == 412

    def test_passage_id_matches_the_prefix_of_item_id(self):
        record = {"link": "https://example.org/a/Introduction", "question_number": 2}
        assert item_id(record) == f"{passage_id(record)}-q2"

    @pytest.mark.parametrize("bad", ["bel-abcdef0123", "", "-q1", "bel-abc-qX"])
    def test_an_id_without_a_question_suffix_raises(self, bad):
        """
        Returning the input unchanged would make every item its own passage,
        and `assert_zero_passage_overlap` would then pass on a leaking split.
        """
        with pytest.raises(ValueError, match="not a Belebele item id"):
            passage_of(bad)


class TestPassageOverlapAssertion:
    def test_a_shared_passage_raises(self):
        with pytest.raises(ValueError, match="passage"):
            assert_zero_passage_overlap(["bel-aaaaaaaaaa-q1"], ["bel-aaaaaaaaaa-q2"])

    def test_a_shared_item_raises(self):
        with pytest.raises(ValueError, match="in both splits"):
            assert_zero_passage_overlap(["bel-aaaaaaaaaa-q1"], ["bel-aaaaaaaaaa-q1"])

    def test_disjoint_passages_pass(self):
        assert_zero_passage_overlap(["bel-aaaaaaaaaa-q1"], ["bel-bbbbbbbbbb-q1"])


class TestSplit:
    def test_split_is_identical_across_languages(self, languages):
        en, ne = languages
        a_en, e_en = passage_split([e["id"] for e in en], 100, seed=42)
        a_ne, e_ne = passage_split([e["id"] for e in ne], 100, seed=42)
        assert (a_en, e_en) == (a_ne, e_ne)

    def test_split_is_disjoint_and_complete(self, languages):
        en, _ = languages
        ids = [e["id"] for e in en]
        adapt, evaluate = passage_split(ids, 100, seed=42)
        assert not set(adapt) & set(evaluate)
        assert set(adapt) | set(evaluate) == set(ids)

    def test_no_passage_straddles_the_split(self, languages):
        """D17. The one property a question-level split cannot give us."""
        en, _ = languages
        adapt, evaluate = passage_split([e["id"] for e in en], 100, seed=42)
        assert_zero_passage_overlap(adapt, evaluate)
        assert len({passage_of(i) for i in adapt}) == 100

    def test_a_question_level_split_would_have_leaked(self, languages):
        """
        The bug this module now prevents, asserted rather than assumed: slice
        the shuffled *questions* the way the old code did and passages land on
        both sides.
        """
        import random

        en, _ = languages
        ids = sorted(e["id"] for e in en)
        shuffled = list(ids)
        random.Random(42).shuffle(shuffled)
        adapt, evaluate = sorted(shuffled[:185]), sorted(shuffled[185:])
        with pytest.raises(ValueError, match="passage"):
            assert_zero_passage_overlap(adapt, evaluate)

    def test_split_is_independent_of_input_order(self, languages):
        """Order-dependence would make the split differ per language."""
        en, _ = languages
        ids = [e["id"] for e in en]
        assert passage_split(ids, 100, 42) == passage_split(list(reversed(ids)), 100, 42)

    def test_different_seeds_give_different_splits(self, languages):
        en, _ = languages
        ids = [e["id"] for e in en]
        assert passage_split(ids, 100, 42) != passage_split(ids, 100, 43)

    @pytest.mark.parametrize("bad", [0, 488, 900])
    def test_rejects_a_degenerate_passage_count(self, languages, bad):
        en, _ = languages
        with pytest.raises(ValueError, match="adaptation_passages"):
            passage_split([e["id"] for e in en], bad, seed=42)

    def test_study_split_uses_the_registered_passage_count(self, languages):
        en, _ = languages
        ids = [e["id"] for e in en]
        assert study_split(ids, 42) == passage_split(ids, BELEBELE_ADAPTATION_PASSAGES, 42)

    def test_the_study_split_sizes_are_what_the_protocol_says(self, languages):
        """~185 adaptation / ~715 evaluation. Pinned so a silent drift shows."""
        en, _ = languages
        adapt, evaluate = study_split([e["id"] for e in en], 42)
        assert len(adapt) + len(evaluate) == 900
        assert 150 <= len(adapt) <= 220, len(adapt)
        assert 680 <= len(evaluate) <= 750, len(evaluate)


class TestHarnessIndices:
    """
    `simple_evaluate(samples=...)` selects documents by position. Reusing one
    language's index list for the other silently evaluates a different question
    set -- and because the split is 200/700, part of that set is the adaptation
    split the playbook was built on.
    """

    def test_indices_select_exactly_the_requested_items(self, languages):
        en, _ = languages
        _, evaluate = passage_split([e["id"] for e in en], 100, seed=42)
        for lang in ("en", "ne"):
            with open(belebele_path(lang), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            selected = {item_id(rows[i]) for i in harness_indices(lang, evaluate)}
            assert selected == set(evaluate)

    def test_reusing_one_languages_indices_leaks_adaptation_items(self, languages):
        """The bug this function exists to prevent, asserted rather than assumed."""
        en, _ = languages
        adapt, evaluate = passage_split([e["id"] for e in en], 100, seed=42)
        with open(belebele_path("ne"), encoding="utf-8") as f:
            ne_rows = [json.loads(line) for line in f if line.strip()]

        wrong = {item_id(ne_rows[i]) for i in harness_indices("en", evaluate)}
        assert wrong != set(evaluate)
        assert wrong & set(adapt), "expected adaptation items to leak into the eval set"

    def test_unknown_id_is_an_error_not_a_silent_drop(self):
        with pytest.raises(KeyError):
            harness_indices("en", ["bel-deadbeefff-q1"])

    def test_a_blank_line_does_not_shift_the_positions(self, tmp_path):
        """Positions must number records, the way the harness loads them.

        `enumerate(f)` counted blank lines the filter then dropped, so one blank
        line shifted every position after it and the run scored a different
        document while reporting a clean accuracy.
        """
        rows = [
            {
                "link": f"http://example.invalid/passage-{i}",
                "question_number": 1,
                "flores_passage": "p",
                "question": "q",
                "mc_answer1": "a",
                "mc_answer2": "b",
                "mc_answer3": "c",
                "mc_answer4": "d",
                "correct_answer_num": "1",
            }
            for i in range(3)
        ]
        path = tmp_path / "gappy.jsonl"
        path.write_text(
            "\n".join([json.dumps(rows[0]), "", json.dumps(rows[1]), json.dumps(rows[2])]) + "\n",
            encoding="utf-8",
        )
        ids = [item_id(r) for r in rows]
        assert harness_indices("en", ids, path=path) == [0, 1, 2]


class TestSchema:
    def test_examples_carry_what_the_runner_needs(self, languages):
        en, _ = languages
        example = en[0]
        assert {"id", "question", "context", "options", "gold_option_idx", "answer"} <= set(example)
        assert len(example["options"]) == 4
        assert example["answer"] == example["options"][example["gold_option_idx"]]

    def test_domain_separates_the_languages_by_default(self, languages):
        en, ne = languages
        assert en[0]["domain"] == "belebele_en"
        assert ne[0]["domain"] == "belebele_ne"

    def test_domain_can_be_shared_for_cross_lingual_transfer(self):
        shared = load_belebele("ne", domain="belebele")
        assert shared[0]["domain"] == "belebele"

    def test_harness_task_names_cover_both_languages(self):
        assert set(HARNESS_TASKS) == set(BELEBELE_LANGUAGES)


class TestStudySplit:
    """One split for the whole study, so screening cannot leak into evaluation.

    `screen_models` split at 400 and `run_arm` at 200. The split was one
    shuffle sliced at that index, so the 200 items between them were screened
    on *and* scored on: models were selected on items their own results are
    reported over. Screening now has no size of its own -- it takes the whole
    adaptation split -- so there is no second number left to disagree.
    """

    @staticmethod
    def _ids(n_passages=300):
        """Synthetic ids in the real shape: two questions per passage."""
        return [f"bel-{i:010d}-q{q}" for i in range(n_passages) for q in (1, 2)]

    def test_screening_takes_the_whole_adaptation_split(self):
        adapt, evaluate = study_split(self._ids(), seed=42)
        assert not set(adapt) & set(evaluate)

    def test_screening_cannot_see_an_evaluation_passage(self):
        """What the old item-level split could not promise."""
        adapt, evaluate = study_split(self._ids(), seed=42)
        assert_zero_passage_overlap(adapt, evaluate)

    def test_every_entrypoint_gets_the_same_split(self):
        ids = self._ids()
        assert study_split(ids, seed=42) == study_split(list(reversed(ids)), seed=42)

    def test_an_override_is_still_available_for_debugging(self):
        adapt, _ = study_split(self._ids(), seed=42, adaptation_passages=10)
        assert len({passage_of(i) for i in adapt}) == 10
