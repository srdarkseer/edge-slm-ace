"""Dataset loaders."""

from edge_slm_ace.data.belebele import (
    BELEBELE_LANGUAGES,
    assert_zero_passage_overlap,
    belebele_path,
    load_belebele,
    passage_id,
    passage_of,
    passage_split,
    study_split,
)
from edge_slm_ace.data.global_mmlu import (
    GLOBAL_MMLU_LANGUAGES,
    assert_parallel,
    global_mmlu_path,
    load_global_mmlu,
    subjects,
)

__all__ = [
    "BELEBELE_LANGUAGES",
    "GLOBAL_MMLU_LANGUAGES",
    "assert_parallel",
    "assert_zero_passage_overlap",
    "belebele_path",
    "global_mmlu_path",
    "load_belebele",
    "load_global_mmlu",
    "passage_id",
    "passage_of",
    "passage_split",
    "study_split",
    "subjects",
]
