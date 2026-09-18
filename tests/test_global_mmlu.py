"""Tests for the Global-MMLU data layer.

Global-MMLU is the study's primary task, so the guarantees it has to carry are
the same ones Belebele carries -- the languages are the same questions, the
join key is language-invariant, gold agrees -- plus one Belebele does not: the
id is *given* rather than synthesised, and the loader must refuse to invent a
substitute when it is missing.

Most tests here run against synthetic records so that the suite works without
the ~31 MB download. The handful that need the real corpus are marked and skip
cleanly when it is absent.
"""

import json

import pytest

from edge_slm_ace.data.global_mmlu import (
    EXPECTED_ROWS,
    GLOBAL_MMLU_LANGUAGES,
    SPLITS,
    assert_parallel,
    global_mmlu_path,
    item_id,
    load_global_mmlu,
    subjects,
)

CORPUS_PRESENT = all(global_mmlu_path(lang, "test").exists() for lang in GLOBAL_MMLU_LANGUAGES)
needs_corpus = pytest.mark.skipif(
    not CORPUS_PRESENT,
    reason="Global-MMLU not fetched; run scripts/fetch_global_mmlu.py",
)


def raw_record(sample_id="abstract_algebra/test/0", answer="B", **overrides):
    """One syntactically valid raw record, before normalisation."""
    record = {
        "sample_id": sample_id,
        "subject": "abstract_algebra",
        "subject_category": "STEM",
        "question": "Find the degree of the field extension.",
        "option_a": "0",
        "option_b": "4",
        "option_c": "2",
        "option_d": "6",
        "answer": answer,
        "cultural_sensitivity_label": "-",
        "is_annotated": False,
    }
    record.update(overrides)
    return record


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class TestItemIds:
    def test_the_given_sample_id_is_used_unchanged(self):
        """It is already language-invariant; hashing it would only hide it."""
        assert item_id(raw_record(sample_id="anatomy/test/17")) == "anatomy/test/17"

    @pytest.mark.parametrize("bad", [None, "", "   "])
    def test_a_missing_id_raises_rather_than_falling_back_to_text(self, bad):
        """
        G1. A text-derived id would differ between languages, handing English
        and Nepali different ids for the same question -- every paired
        comparison downstream would then silently compare unrelated items.
        """
        with pytest.raises(ValueError, match="sample_id"):
            item_id(raw_record(sample_id=bad))

    def test_a_record_with_no_id_field_at_all_raises(self):
        record = raw_record()
        del record["sample_id"]
        with pytest.raises(ValueError, match="sample_id"):
            item_id(record)


class TestNormalisation:
    def test_letter_answer_becomes_a_zero_based_index(self, tmp_path):
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record(answer="C")])
        (example,) = load_global_mmlu("en", path=path)
        assert example["gold_option_idx"] == 2
        assert example["answer"] == "2"  # option_c

    def test_options_are_kept_in_stored_order(self, tmp_path):
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record()])
        (example,) = load_global_mmlu("en", path=path)
        assert example["options"] == ["0", "4", "2", "6"]

    def test_context_is_empty_because_knowledge_mcq_has_no_passage(self, tmp_path):
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record()])
        (example,) = load_global_mmlu("en", path=path)
        assert example["context"] == ""

    def test_subject_is_carried_for_stratification(self, tmp_path):
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record()])
        (example,) = load_global_mmlu("en", path=path)
        assert example["subject"] == "abstract_algebra"
        assert example["subject_category"] == "STEM"

    def test_domain_separates_the_languages_by_default(self, tmp_path):
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record()])
        assert load_global_mmlu("ne", path=path)[0]["domain"] == "global_mmlu_ne"
        assert load_global_mmlu("en", path=path)[0]["domain"] == "global_mmlu_en"

    @pytest.mark.parametrize("bad", ["E", "1", "-", "None"])
    def test_an_answer_outside_ABCD_raises(self, tmp_path, bad):
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record(answer=bad)])
        with pytest.raises(ValueError, match="answer is not one of"):
            load_global_mmlu("en", path=path)

    @pytest.mark.parametrize("bad", ["", "   ", "AB", "ABCD"])
    def test_an_answer_that_is_a_substring_of_ABCD_raises(self, tmp_path, bad):
        """
        Regression guard. `_ANSWER_LETTERS` was the string "ABCD", and `"" in
        "ABCD"` is True while `"ABCD".index("")` is 0 -- so a blank answer
        passed the check and became gold = A, turning missing data into a
        confidently wrong label on every affected item. Only exact membership
        over single characters rejects these.
        """
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record(answer=bad)])
        with pytest.raises(ValueError, match="answer is not one of"):
            load_global_mmlu("en", path=path)

    def test_a_lowercase_padded_answer_is_accepted(self, tmp_path):
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record(answer=" b ")])
        (example,) = load_global_mmlu("en", path=path)
        assert example["gold_option_idx"] == 1

    def test_a_blank_option_raises(self, tmp_path):
        """A blank option is an empty continuation, not a fourth choice."""
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record(option_c="  ")])
        with pytest.raises(ValueError, match="blank option"):
            load_global_mmlu("en", path=path)

    def test_a_missing_field_raises_rather_than_defaulting(self, tmp_path):
        record = raw_record()
        del record["subject_category"]
        path = write_jsonl(tmp_path / "test.jsonl", [record])
        with pytest.raises(ValueError, match="missing fields"):
            load_global_mmlu("en", path=path)


class TestLoad:
    def test_duplicate_ids_raise(self, tmp_path):
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record(), raw_record()])
        with pytest.raises(ValueError, match="duplicate item ids"):
            load_global_mmlu("en", path=path)

    def test_rows_come_back_sorted_by_id(self, tmp_path):
        ids = ["anatomy/test/1", "abstract_algebra/test/0", "virology/test/9"]
        path = write_jsonl(tmp_path / "test.jsonl", [raw_record(sample_id=i) for i in ids])
        assert [e["id"] for e in load_global_mmlu("en", path=path)] == sorted(ids)

    def test_load_order_is_independent_of_file_order(self, tmp_path):
        """
        The published files share a row order today. Nothing upstream promises
        that, so the loader imposes one rather than trusting it.
        """
        ids = ["anatomy/test/1", "abstract_algebra/test/0", "virology/test/9"]
        forward = write_jsonl(tmp_path / "a.jsonl", [raw_record(sample_id=i) for i in ids])
        reverse = write_jsonl(
            tmp_path / "b.jsonl", [raw_record(sample_id=i) for i in reversed(ids)]
        )
        assert [e["id"] for e in load_global_mmlu("en", path=forward)] == [
            e["id"] for e in load_global_mmlu("en", path=reverse)
        ]

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text(
            json.dumps(raw_record()) + "\n\n" + json.dumps(raw_record(sample_id="x/test/1")) + "\n",
            encoding="utf-8",
        )
        assert len(load_global_mmlu("en", path=path)) == 2

    def test_a_missing_corpus_names_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="make data"):
            load_global_mmlu("en", path=tmp_path / "absent.jsonl")

    def test_an_unknown_language_raises(self):
        with pytest.raises(KeyError, match="Unknown Global-MMLU language"):
            global_mmlu_path("hi")

    def test_an_unknown_split_raises(self):
        with pytest.raises(ValueError, match="Unknown split"):
            global_mmlu_path("en", "train")


class TestParallelAssertion:
    def test_agreeing_languages_pass(self):
        rows = [raw_record(sample_id="a/test/0"), raw_record(sample_id="a/test/1")]
        loaded = {
            lang: [{"id": r["sample_id"], "gold_option_idx": 1} for r in rows]
            for lang in ("en", "ne")
        }
        assert_parallel(loaded)

    def test_a_differing_id_set_raises(self):
        loaded = {
            "en": [{"id": "a/test/0", "gold_option_idx": 1}],
            "ne": [{"id": "a/test/9", "gold_option_idx": 1}],
        }
        with pytest.raises(ValueError, match="not parallel"):
            assert_parallel(loaded)

    def test_disagreeing_gold_raises(self):
        """
        Same ids, different correct option: a translation reordered the
        options. That looks like a model error and is a data error.
        """
        loaded = {
            "en": [{"id": "a/test/0", "gold_option_idx": 1}],
            "ne": [{"id": "a/test/0", "gold_option_idx": 3}],
        }
        with pytest.raises(ValueError, match="disagree on the correct option"):
            assert_parallel(loaded)

    def test_one_language_is_not_a_parallel_claim(self):
        with pytest.raises(ValueError, match="at least two languages"):
            assert_parallel({"en": []})


class TestSubjects:
    def test_counts_are_sorted_for_a_stable_diff(self):
        examples = [
            {"subject": "virology"},
            {"subject": "anatomy"},
            {"subject": "virology"},
        ]
        assert subjects(examples) == {"anatomy": 1, "virology": 2}


@needs_corpus
class TestRealCorpus:
    @pytest.fixture(scope="class")
    def languages(self):
        return {lang: load_global_mmlu(lang, "test") for lang in GLOBAL_MMLU_LANGUAGES}

    def test_row_count_is_what_every_frozen_split_assumes(self, languages):
        for lang, rows in languages.items():
            assert len(rows) == EXPECTED_ROWS["test"], lang

    def test_the_languages_are_parallel(self, languages):
        assert_parallel(languages)

    def test_all_57_subjects_are_present(self, languages):
        for lang, rows in languages.items():
            assert len(subjects(rows)) == 57, lang

    def test_gold_position_is_skewed(self, languages):
        """
        Why the harness-side shuffle exists. If this ever comes back uniform,
        the debiasing is dormant rather than unnecessary -- check before
        removing it.
        """
        counts = {}
        for example in languages["en"]:
            counts[example["gold_option_idx"]] = counts.get(example["gold_option_idx"], 0) + 1
        spread = max(counts.values()) - min(counts.values())
        assert spread > 300, f"expected a skewed gold position, got {counts}"

    def test_dev_is_loadable_and_small(self):
        """`dev` stays out of every split; it is the independent sanity set."""
        assert len(load_global_mmlu("en", "dev")) == EXPECTED_ROWS["dev"]

    def test_every_split_this_project_names_exists(self):
        for lang in GLOBAL_MMLU_LANGUAGES:
            for split in SPLITS:
                assert global_mmlu_path(lang, split).exists()
