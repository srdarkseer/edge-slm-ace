"""Registration of this project's harness tasks.

Every reported number comes from lm-evaluation-harness scoring one of the four
tasks defined in `edge_slm_ace/tasks/`. This module is how the harness finds
them, and the only place that names them.

The tasks are ours rather than upstream's for two reasons, one forced and one
chosen. Forced: lm-eval ships no `global_mmlu_ne` at all, and the
`global_mmlu_<lang>` tasks it does ship read Global-MMLU *Lite*, a different
and much smaller corpus. Chosen: a project-owned task can select documents by
frozen-split id and permute options per item, neither of which an upstream task
will do for us.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Sequence

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"

# Short (task, language) -> harness task name. Every entrypoint resolves through
# here, so a task can be renamed in one place and a typo is a KeyError at the
# top of a run rather than an empty result set at the bottom.
HARNESS_TASKS: Dict[str, Dict[str, str]] = {
    "global_mmlu": {
        "en": "tinyace_global_mmlu_en",
        "ne": "tinyace_global_mmlu_ne",
    },
    "belebele": {
        "en": "tinyace_belebele_en",
        "ne": "tinyace_belebele_ne",
    },
}


def harness_task(task: str, language: str) -> str:
    """
    The harness task name for one task/language pair.

    Args:
        task: "global_mmlu" or "belebele".
        language: "en" or "ne".

    Returns:
        The registered task name.

    Raises:
        KeyError: Naming what is available, rather than letting an unknown name
            reach the harness -- which reports it as an empty result set after
            the model has loaded.
    """
    if task not in HARNESS_TASKS:
        raise KeyError(f"Unknown task '{task}'. Known: {sorted(HARNESS_TASKS)}")
    languages = HARNESS_TASKS[task]
    if language not in languages:
        raise KeyError(f"Unknown language '{language}' for {task}. Known: {sorted(languages)}")
    return languages[language]


@lru_cache(maxsize=1)
def task_manager():
    """
    A `TaskManager` that can see this project's tasks as well as the built-in ones.

    Cached: constructing one indexes every task lm-eval ships, which is slow
    enough to notice when a grid builds one per cell.

    Returns:
        `lm_eval.tasks.TaskManager` with our task directory on its include path.

    Raises:
        RuntimeError: If our tasks did not register, which otherwise surfaces
            much later as "task not found" once a checkpoint is already loaded.
    """
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_path=str(TASKS_DIR))

    available = set(manager.all_tasks)
    expected = {name for langs in HARNESS_TASKS.values() for name in langs.values()}
    missing = sorted(expected - available)
    if missing:
        raise RuntimeError(
            f"{len(missing)} project task(s) failed to register from "
            f"{TASKS_DIR}: {missing}. The YAMLs are present but the harness "
            f"did not accept them."
        )
    return manager


def load_tasks(names: Sequence[str]) -> Dict:
    """
    Load registered tasks by name.

    Wraps `TaskManager.load`, which returns `{"tasks": ..., "groups": ...}`, and
    hands back just the flat task mapping. The older `load_task_or_group`
    returned tasks at the top level and is deprecated; going through one helper
    means that shape change is absorbed here rather than in every caller.

    Args:
        names: Registered task names, from `harness_task`.

    Returns:
        {task name: Task}.

    Raises:
        KeyError: If a requested task did not come back.
    """
    loaded = task_manager().load(list(names))
    tasks = loaded["tasks"] if "tasks" in loaded else loaded

    missing = sorted(set(names) - set(tasks))
    if missing:
        raise KeyError(f"the harness returned no task for {missing}")
    return {name: tasks[name] for name in names}
