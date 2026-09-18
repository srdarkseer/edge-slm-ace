"""Tests for this project's harness tasks.

These four tasks are where every reported number comes from, so what is checked
here is not "does it run" but the three things that would corrupt a result
while still producing one:

  - the harness scores exactly the frozen split, selected by id;
  - the option permutation moves gold rather than relabelling it;
  - English and Nepali render the same item in the same option order.
"""

import pytest

from edge_slm_ace.data.global_mmlu import GLOBAL_MMLU_LANGUAGES, global_mmlu_path
from edge_slm_ace.data.splits import GLOBAL_MMLU_EVAL, manifest_ids
from edge_slm_ace.harness.tasks import (
    HARNESS_TASKS,
    harness_task,
    load_tasks,
    task_manager,
)
from edge_slm_ace.tasks.utils import CHOICE_LETTERS, prepare

pytest.importorskip("lm_eval", reason="lm-eval not installed")

CORPUS_PRESENT = all(global_mmlu_path(lang, "test").exists() for lang in GLOBAL_MMLU_LANGUAGES)
needs_corpus = pytest.mark.skipif(not CORPUS_PRESENT, reason="Global-MMLU not fetched")


def example(item_id="a/test/0", gold=0, n_options=4):
    options = ["w", "x", "y", "z"][:n_options]
    return {
        "id": item_id,
        "question": "Q?",
        "context": "",
        "options": options,
        "gold_option_idx": gold,
        "answer": options[gold],
        "subject": "algebra",
    }


class TestTaskNames:
    def test_every_task_and_language_resolves(self):
        for task, languages in HARNESS_TASKS.items():
            for language in languages:
                assert harness_task(task, language) == HARNESS_TASKS[task][language]

    def test_an_unknown_task_raises_with_the_options(self):
        with pytest.raises(KeyError, match="Unknown task"):
            harness_task("hellaswag", "en")

    def test_an_unknown_language_raises(self):
        """
        Better here than as an empty result set after a checkpoint has loaded.
        """
        with pytest.raises(KeyError, match="Unknown language"):
            harness_task("belebele", "hi")


class TestPrepare:
    def test_selection_is_by_id_not_position(self):
        rows = [example("a/test/0"), example("a/test/1"), example("a/test/2")]
        prepared = prepare(rows, keep_ids=["a/test/2", "a/test/0"])
        assert [r["id"] for r in prepared] == ["a/test/0", "a/test/2"]

    def test_a_missing_id_raises_rather_than_shrinking_the_set(self):
        """
        A silently dropped id would shrink the evaluation set while every
        reported accuracy still computed.
        """
        with pytest.raises(ValueError, match="not in the loaded corpus"):
            prepare([example("a/test/0")], keep_ids=["a/test/0", "a/test/9"])

    def test_rows_come_out_sorted_by_id(self):
        rows = [example("b/test/0"), example("a/test/0")]
        assert [r["id"] for r in prepare(rows)] == ["a/test/0", "b/test/0"]

    def test_options_are_flattened_to_lettered_fields(self):
        (row,) = prepare([example()], shuffle=False)
        assert [row[f"option_{c.lower()}"] for c in CHOICE_LETTERS] == ["w", "x", "y", "z"]

    def test_the_answer_field_is_the_letter_of_the_gold_slot(self):
        (row,) = prepare([example(gold=2)], shuffle=False)
        assert row["answer"] == "C"

    def test_shuffling_moves_gold_rather_than_relabelling_it(self):
        source = example(gold=0)
        (row,) = prepare([source], shuffle=True)
        letter = row["answer"]
        assert row[f"option_{letter.lower()}"] == "w", "gold text changed under the permutation"

    def test_a_wrong_option_count_raises(self):
        """
        A jinja template indexing a short list renders "None" rather than
        failing, so the shape is checked at this boundary instead.
        """
        with pytest.raises(ValueError, match="options, expected 4"):
            prepare([example(n_options=3)], shuffle=False)


@needs_corpus
class TestRegisteredTasks:
    @pytest.fixture(scope="class")
    def docs(self):
        names = [harness_task("global_mmlu", lang) for lang in ("en", "ne")]
        loaded = load_tasks(names)
        return {
            lang: list(loaded[harness_task("global_mmlu", lang)].test_docs())
            for lang in ("en", "ne")
        }

    def test_the_tasks_register(self):
        assert task_manager() is not None

    def test_an_unregistered_name_is_refused(self):
        with pytest.raises(Exception):
            load_tasks(["tinyace_not_a_task"])

    def test_belebele_registers_too(self):
        name = harness_task("belebele", "ne")
        assert len(list(load_tasks([name])[name].test_docs())) == 720

    def test_the_harness_sees_exactly_the_frozen_split(self, docs):
        frozen = set(manifest_ids(GLOBAL_MMLU_EVAL))
        for lang, rows in docs.items():
            assert {d["id"] for d in rows} == frozen, lang

    def test_both_languages_render_the_same_items_in_the_same_order(self, docs):
        assert [d["id"] for d in docs["en"]] == [d["id"] for d in docs["ne"]]

    def test_both_languages_permute_identically(self, docs):
        """G1, through the layer that actually renders the scored prompt."""
        for a, b in zip(docs["en"], docs["ne"]):
            assert a["shuffle_perm"] == b["shuffle_perm"], a["id"]

    def test_both_languages_agree_on_the_answer_letter(self, docs):
        for a, b in zip(docs["en"], docs["ne"]):
            assert a["answer"] == b["answer"], a["id"]

    def test_the_permutation_flattens_gold_position(self, docs):
        """Going in: A 3222 / B 3462 / C 3582 / D 3776. Coming out: flat."""
        counts = {letter: 0 for letter in CHOICE_LETTERS}
        for doc in docs["ne"]:
            counts[doc["answer"]] += 1
        spread = max(counts.values()) - min(counts.values())
        assert spread < 60, counts

    def test_the_rendered_prompt_carries_all_four_options(self, docs):
        name = harness_task("global_mmlu", "ne")
        task = load_tasks([name])[name]
        doc = docs["ne"][0]
        text = task.doc_to_text(doc)
        for letter in CHOICE_LETTERS:
            assert f"{letter}. {doc[f'option_{letter.lower()}']}" in text

    def test_the_target_is_the_gold_letter(self, docs):
        name = harness_task("global_mmlu", "ne")
        task = load_tasks([name])[name]
        for doc in docs["ne"][:50]:
            assert task.doc_to_target(doc) == doc["answer"]
            assert task.doc_to_choice(doc) == CHOICE_LETTERS
