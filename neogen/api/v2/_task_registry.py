"""In-memory async task registry for long-running V2 API endpoints.

The V1.1 validation harness has a synchronous ``/v2/validation/full``
endpoint that blocks an HTTP worker for 3-5 minutes on a full cohort
run. For interactive use (CI pipelines, dashboards, the frontend) we
need an async alternative that:

  1. Accepts a POST, starts the work in a background thread, returns
     a task_id immediately.
  2. Exposes ``GET /status/{task_id}`` to poll for completion,
     progress, and the final result.
  3. Saves persistent reports to disk regardless of whether the
     caller polls for the result.

Design
------
Single-process, in-memory dict guarded by a lock. Backgrounded via
:mod:`threading` — the worker simply runs the
``ValidationRunner.run_all`` inside a daemon thread. No Celery /
RQ / Dramatiq dependency. This is adequate for single-worker
deployments (the current Railway setup) and scales horizontally
via process isolation rather than shared state.

Task states:

    pending   — created but not started
    running   — worker thread active
    done      — finished successfully, ``result`` populated
    failed    — worker raised, ``error`` populated

Results are kept for 1 hour after completion then garbage-collected.
Pending/running tasks are never garbage-collected.

Thread safety
-------------
Every mutation of the registry goes through ``_LOCK``. Worker
threads write to their own task record but never touch others,
so contention is limited to create / status / sweep paths.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    """Mutable task record stored in the registry."""

    task_id: str
    kind: str  # e.g. "validation_full"
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress_stage: str = ""
    progress_percent: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    # Arbitrary metadata the caller set when creating the task.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def runtime_s(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return end - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_s": self.runtime_s,
            "progress_stage": self.progress_stage,
            "progress_percent": round(self.progress_percent, 1),
            "error": self.error,
            "metadata": self.metadata,
            # Result is returned only when status == done. Serialized
            # as a plain dict so it passes through FastAPI's JSON
            # encoder without custom converters.
            "result": self.result if self.status == TaskStatus.DONE else None,
        }


# ════════════════════════════════════════════════════════════════
# REGISTRY STATE
# ════════════════════════════════════════════════════════════════


_LOCK = threading.Lock()
_TASKS: Dict[str, Task] = {}
_RESULT_TTL_S = 3600.0  # 1 hour — how long completed results linger


def _new_task_id() -> str:
    return uuid.uuid4().hex[:16]


def create_task(kind: str, metadata: Optional[Dict[str, Any]] = None) -> Task:
    """Allocate a new :class:`Task` record in the PENDING state."""
    with _LOCK:
        _sweep_locked()
        task = Task(
            task_id=_new_task_id(),
            kind=kind,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            metadata=dict(metadata or {}),
        )
        _TASKS[task.task_id] = task
        return task


def get_task(task_id: str) -> Optional[Task]:
    """Return the task record, or ``None`` if not present or expired."""
    with _LOCK:
        _sweep_locked()
        return _TASKS.get(task_id)


def list_tasks(include_done: bool = False) -> List[Task]:
    """Snapshot of every known task. Mainly for debugging."""
    with _LOCK:
        _sweep_locked()
        return [
            t
            for t in _TASKS.values()
            if include_done or t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        ]


def start_in_background(
    task: Task,
    work: Callable[["_Progress"], Any],
) -> None:
    """Launch ``work`` in a daemon thread, updating task state on
    completion or failure.

    ``work`` receives a :class:`_Progress` helper it can use to
    update progress_stage / progress_percent on the task. Return
    value becomes ``task.result``; exceptions become ``task.error``.
    """

    def _runner() -> None:
        progress = _Progress(task.task_id)
        with _LOCK:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
        try:
            result = work(progress)
            with _LOCK:
                task.result = result
                task.progress_stage = "done"
                task.progress_percent = 100.0
                task.status = TaskStatus.DONE
                task.finished_at = time.time()
        except Exception as exc:
            logger.exception("Background task %s failed", task.task_id)
            with _LOCK:
                task.error = f"{type(exc).__name__}: {exc}"
                task.status = TaskStatus.FAILED
                task.finished_at = time.time()

    thread = threading.Thread(
        target=_runner, name=f"v2-task-{task.task_id}", daemon=True
    )
    thread.start()


class _Progress:
    """Callback helper passed to background workers for progress updates."""

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id

    def set(self, stage: str, percent: float) -> None:
        with _LOCK:
            task = _TASKS.get(self._task_id)
            if task is not None:
                task.progress_stage = stage
                task.progress_percent = max(0.0, min(percent, 100.0))


def _sweep_locked() -> None:
    """Garbage-collect completed tasks older than ``_RESULT_TTL_S``.

    Caller must hold ``_LOCK``.
    """
    now = time.time()
    expired = [
        tid
        for tid, t in _TASKS.items()
        if t.status in (TaskStatus.DONE, TaskStatus.FAILED)
        and t.finished_at is not None
        and (now - t.finished_at) > _RESULT_TTL_S
    ]
    for tid in expired:
        del _TASKS[tid]


# Test-only entry point — not part of the public API.
def _reset_for_tests() -> None:  # pragma: no cover — used in tests only
    with _LOCK:
        _TASKS.clear()
