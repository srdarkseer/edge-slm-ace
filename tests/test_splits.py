"""Tests for the frozen split manifests.

Two separate claims are under test here.

The first is that the *mechanism* is sound: stratified sampling is
deterministic and proportional, checksums catch an edited manifest, and
`assert_not_in_eval` refuses an evaluation item. Those run on synthetic data.

The second is that the *committed manifests* are the ones the study
pre-registered -- they load, they verify, they are disjoint, and they are the
sizes the protocol names. Those read `data/splits/` and are the reason the
files are in git rather than regenerated per run.
"""

import json

import pytest

from edge_slm_ace.data.splits import (
    ADAPT_SUBSAMPLE_SIZE,
    BELEBELE_SPLIT,
    EVAL_SUBSAMPLE_SIZE,
    GLOBAL_MMLU_ADAPT,
    GLOBAL_MMLU_EVAL,
    SUBSAMPLE_SEED,
    assert_not_in_eval,
    clear_manifest_cache,
    ids_checksum,
    load_manifest,
    manifest_ids,
    manifest_path,
    stratified_sample,
    write_manifest,
)


def population(per_subject=None):
    """Synthetic examples with an uneven subject distribution."""
    per_subject = per_subject or {"algebra": 100, "anatomy": 40, "virology": 10}
    return [
        {"id": f"{subject}/test/{i}", "subject": subject}
        for subject, n in per_subject.items()
        for i in range(n)
    ]


class TestStratifiedSample:
    def test_is_deterministic_in_the_seed(self):
        pop = population()
        assert stratified_sample(pop, 30, 7) == stratified_sample(pop, 30, 7)

    def test_a_different_seed_gives_a_different_sample(self):
        pop = population()
        assert stratified_sample(pop, 30, 7) != stratified_sample(pop, 30, 8)

    def test_is_independent_of_input_order(self):
        """Otherwise the split would depend on how the loader happened to sort."""
        pop = population()
        assert stratified_sample(pop, 30, 7) == stratified_sample(list(reversed(pop)), 30, 7)

    def test_draws_exactly_the_requested_size(self):
        for size in (1, 15, 30, 149, 150):
            assert len(stratified_sample(population(), size, 7)) == size

    def test_allocation_is_proportional(self):
        """
        150 items split 100/40/10; a 30-item draw should be about 20/8/2.
        Simple random sampling would leave the 10-item subject swinging
        between 0 and 5 on seed alone.
        """
        chosen = stratified_sample(population(), 30, 7)
        counts = {}
        for item in chosen:
            subject = item.split("/")[0]
            counts[subject] = counts.get(subject, 0) + 1
        assert counts == {"algebra": 20, "anatomy": 8, "virology": 2}

    def test_a_tiny_draw_can_drop_a_tiny_stratum(self):
        """
        The boundary of proportional allocation, pinned rather than papered
        over. Drawing 3 from 100/40/10 gives exact shares 2.0/0.8/0.2, so the
        one remainder seat goes to the 40 and the 10 gets nothing.

        No minimum-per-stratum floor is imposed, because that would distort the
        proportions the stratification exists to preserve. It is safe here
        because the study's draws are nowhere near this boundary: at n=2000 the
        smallest subject takes 14 items and at n=400 it takes 3. The committed
        manifests are checked for all 57 subjects below, which is the claim
        that actually matters.
        """
        chosen = stratified_sample(population(), 3, 7)
        assert {c.split("/")[0] for c in chosen} == {"algebra", "anatomy"}

    def test_the_studys_draw_sizes_keep_every_stratum(self):
        """57 subjects at >=100 items each, drawn down to 400: none vanish."""
        pop = population({f"s{i}": 100 for i in range(57)})
        for size in (EVAL_SUBSAMPLE_SIZE, ADAPT_SUBSAMPLE_SIZE):
            chosen = stratified_sample(pop, size, SUBSAMPLE_SEED)
            assert len({c.split("/")[0] for c in chosen}) == 57, size

    def test_no_stratum_is_allocated_more_than_it_holds(self):
        """
        Largest-remainder rounding can hand a seat to a stratum whose exact
        share was already a whole number. Drawing the whole population is the
        case that exposes it.
        """
        pop = population({"a": 3, "b": 3, "c": 4})
        assert sorted(stratified_sample(pop, 10, 7)) == sorted(p["id"] for p in pop)

    def test_excluded_ids_are_never_drawn(self):
        pop = population()
        first = stratified_sample(pop, 30, 7)
        second = stratified_sample(pop, 30, 7, exclude=first)
        assert not set(first) & set(second)

    def test_exclusion_shrinks_the_eligible_pool(self):
        pop = population()
        everything = [p["id"] for p in pop]
        with pytest.raises(ValueError, match="eligible pool"):
            stratified_sample(pop, 1, 7, exclude=everything)

    @pytest.mark.parametrize("size", [0, -1, 151])
    def test_rejects_a_size_that_does_not_fit(self, size):
        with pytest.raises(ValueError, match="cannot draw"):
            stratified_sample(population(), size, 7)


class TestManifestRoundTrip:
    def test_checksum_is_order_independent(self):
        assert ids_checksum(["b", "a"]) == ids_checksum(["a", "b"])

    def test_different_ids_hash_differently(self):
        assert ids_checksum(["a", "b"]) != ids_checksum(["a", "c"])

    def test_write_then_load_verifies(self, tmp_path, monkeypatch):
        monkeypatch.setattr("edge_slm_ace.data.splits.REPO_ROOT", tmp_path)
        clear_manifest_cache()
        write_manifest("fixture", {"task": "t", "item_ids": ["b/test/1", "a/test/0"]})
        body = load_manifest("fixture")
        assert body["item_ids"] == ["a/test/0", "b/test/1"]
        assert body["name"] == "fixture"

    def test_an_edited_id_list_fails_the_checksum(self, tmp_path, monkeypatch):
        """
        The tripwire. It catches a truncated file, a hand edit and a
        half-written regeneration; it cannot stop someone regenerating both
        the ids and the checksum, which is what git history is for.
        """
        monkeypatch.setattr("edge_slm_ace.data.splits.REPO_ROOT", tmp_path)
        clear_manifest_cache()
        path = write_manifest("fixture", {"task": "t", "item_ids": ["a/test/0", "b/test/1"]})

        body = json.loads(path.read_text(encoding="utf-8"))
        body["item_ids"].append("c/test/2")
        path.write_text(json.dumps(body), encoding="utf-8")

        clear_manifest_cache()
        with pytest.raises(ValueError, match="ids_sha256"):
            load_manifest("fixture")

    def test_a_missing_manifest_names_the_fix(self, tmp_path, monkeypatch):
        monkeypatch.setattr("edge_slm_ace.data.splits.REPO_ROOT", tmp_path)
        clear_manifest_cache()
        with pytest.raises(FileNotFoundError, match="REGENERATE"):
            load_manifest("absent")

    def test_a_manifest_without_ids_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("edge_slm_ace.data.splits.REPO_ROOT", tmp_path)
        with pytest.raises(ValueError, match="item_ids"):
            write_manifest("fixture", {"task": "t"})


class TestAssertNotInEval:
    def test_an_evaluation_item_is_refused(self):
        with pytest.raises(ValueError, match="frozen evaluation split"):
            assert_not_in_eval(manifest_ids(GLOBAL_MMLU_EVAL)[0], GLOBAL_MMLU_EVAL)

    def test_an_adaptation_item_passes(self):
        assert_not_in_eval(manifest_ids(GLOBAL_MMLU_ADAPT)[0], GLOBAL_MMLU_EVAL)

    def test_an_unknown_item_passes(self):
        """It guards the eval split, not membership in the corpus."""
        assert_not_in_eval("not/an/id", GLOBAL_MMLU_EVAL)

    def test_every_belebele_evaluation_item_is_refused(self):
        for item in manifest_ids(BELEBELE_SPLIT)[:20]:
            with pytest.raises(ValueError):
                assert_not_in_eval(item, BELEBELE_SPLIT)


class TestCommittedManifests:
    """The pre-registration, as a thing a reviewer can diff."""

    def test_all_three_load_and_verify(self):
        for name in (GLOBAL_MMLU_EVAL, GLOBAL_MMLU_ADAPT, BELEBELE_SPLIT):
            assert load_manifest(name)["item_ids"]

    def test_sizes_are_what_the_protocol_names(self):
        assert len(manifest_ids(GLOBAL_MMLU_EVAL)) == EVAL_SUBSAMPLE_SIZE
        assert len(manifest_ids(GLOBAL_MMLU_ADAPT)) == ADAPT_SUBSAMPLE_SIZE

    def test_the_global_mmlu_splits_are_disjoint(self):
        assert not set(manifest_ids(GLOBAL_MMLU_EVAL)) & set(manifest_ids(GLOBAL_MMLU_ADAPT))

    def test_the_belebele_split_is_passage_disjoint(self):
        from edge_slm_ace.data.belebele import assert_zero_passage_overlap

        body = load_manifest(BELEBELE_SPLIT)
        assert_zero_passage_overlap(body["adaptation_item_ids"], body["item_ids"])

    def test_the_belebele_split_covers_all_900_questions(self):
        body = load_manifest(BELEBELE_SPLIT)
        assert len(set(body["adaptation_item_ids"]) | set(body["item_ids"])) == 900

    def test_the_eval_subsample_keeps_every_subject(self):
        """A dropped subject would narrow the task without narrowing the claim."""
        subjects = {i.rsplit("/test/", 1)[0] for i in manifest_ids(GLOBAL_MMLU_EVAL)}
        assert len(subjects) == 57

    def test_the_adaptation_subsample_keeps_every_subject(self):
        subjects = {i.rsplit("/test/", 1)[0] for i in manifest_ids(GLOBAL_MMLU_ADAPT)}
        assert len(subjects) == 57

    def test_the_seed_is_recorded_so_the_draw_is_reproducible(self):
        for name in (GLOBAL_MMLU_EVAL, GLOBAL_MMLU_ADAPT):
            assert load_manifest(name)["seed"] == SUBSAMPLE_SEED

    def test_each_manifest_pins_the_corpus_it_was_drawn_from(self):
        """
        Global-MMLU is not committed, so without this "the ids are frozen"
        would say nothing about which 14,042 rows they index into.
        """
        for name in (GLOBAL_MMLU_EVAL, GLOBAL_MMLU_ADAPT, BELEBELE_SPLIT):
            checksums = load_manifest(name)["source_sha256"]
            assert set(checksums) == {"en", "ne"}
            assert all(len(v) == 64 for v in checksums.values())

    def test_english_evaluates_the_same_ids(self):
        """What keeps the en/ne comparison paired rather than merely matched."""
        assert load_manifest(GLOBAL_MMLU_EVAL)["languages"] == ["en", "ne"]

    def test_the_manifests_are_committed(self):
        for name in (GLOBAL_MMLU_EVAL, GLOBAL_MMLU_ADAPT, BELEBELE_SPLIT):
            assert manifest_path(name).exists()
