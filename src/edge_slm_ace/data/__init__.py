"""Dataset loaders."""

from edge_slm_ace.data.belebele import (
    BELEBELE_LANGUAGES,
    belebele_path,
    load_belebele,
    parallel_split,
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
    "belebele_path",
    "global_mmlu_path",
    "load_belebele",
    "load_global_mmlu",
    "parallel_split",
    "study_split",
    "subjects",
]
