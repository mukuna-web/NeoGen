"""Validation harness endpoints.

Exposes four synchronous endpoints:

    POST /v2/validation/tier4   — ~1s  : mechanistic invariants only
    POST /v2/validation/tier2   — ~30s : DDI pair AUCR only
    POST /v2/validation/tier3   — ~2s  : bioavailability only
    POST /v2/validation/full    — 3-5 min : all four tiers
    GET  /v2/validation/latest  — read most recent saved report

Synchronous is a deliberate V1 choice. A background-task queue
with 202+polling is in V1.1 scope along with Celery/RQ wiring.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from neogen.api.v2._task_registry import (
    TaskStatus,
    create_task,
    get_task,
    start_in_background,
)
from neogen.api.v2.schemas import (
    ValidationFullRequest,
    ValidationResponse,
    ValidationTierRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/validation", tags=["validation"])


_REPORTS_DIR = (
    Path(__file__).resolve().parents[4] / "reports" / "validation"
)


# ════════════════════════════════════════════════════════════════
# SINGLE-TIER ENDPOINTS
# ════════════════════════════════════════════════════════════════


@router.post(
    "/tier4",
    response_model=ValidationResponse,
    summary="Run Tier 4 mechanistic invariants (~1s)",
)
async def run_tier4(
    request: ValidationTierRequest,
) -> ValidationResponse:
    from neogen.engine.orchestrator_v2 import ENGINE_VERSION
    from neogen.validation.report import ValidationReport
    from neogen.validation.tier4 import Tier4Harness

    try:
        tier4_report = Tier4Harness().run()
    except Exception as exc:
        logger.exception("Tier 4 failed")
        raise HTTPException(
            status_code=500, detail=f"tier4: {exc}"
        ) from exc

    report = ValidationReport(
        engine_version=ENGINE_VERSION,
        run_timestamp_utc=ValidationReport.now_timestamp(),
        tier4=tier4_report,
    )
    if request.save_report:
        _save_report(report)
    return _wrap(report)


@router.post(
    "/tier2",
    response_model=ValidationResponse,
    summary="Run Tier 2 DDI pair validation (~30s)",
)
async def run_tier2(
    request: ValidationTierRequest,
) -> ValidationResponse:
    from neogen.engine.orchestrator_v2 import ENGINE_VERSION
    from neogen.validation.report import ValidationReport
    from neogen.validation.tier2 import Tier2Harness

    try:
        tier2_report = Tier2Harness().run()
    except Exception as exc:
        logger.exception("Tier 2 failed")
        raise HTTPException(
            status_code=500, detail=f"tier2: {exc}"
        ) from exc

    report = ValidationReport(
        engine_version=ENGINE_VERSION,
        run_timestamp_utc=ValidationReport.now_timestamp(),
        tier2=tier2_report,
    )
    if request.save_report:
        _save_report(report)
    return _wrap(report)


@router.post(
    "/tier3",
    response_model=ValidationResponse,
    summary="Run Tier 3 bioavailability validation",
)
async def run_tier3(
    request: ValidationTierRequest,
) -> ValidationResponse:
    from neogen.engine.orchestrator_v2 import ENGINE_VERSION
    from neogen.validation.report import ValidationReport
    from neogen.validation.tier3 import Tier3Harness

    try:
        tier3_report = Tier3Harness().run()
    except Exception as exc:
        logger.exception("Tier 3 failed")
        raise HTTPException(
            status_code=500, detail=f"tier3: {exc}"
        ) from exc

    report = ValidationReport(
        engine_version=ENGINE_VERSION,
        run_timestamp_utc=ValidationReport.now_timestamp(),
        tier3=tier3_report,
    )
    if request.save_report:
        _save_report(report)
    return _wrap(report)


# ════════════════════════════════════════════════════════════════
# FULL HARNESS
# ════════════════════════════════════════════════════════════════


@router.post(
    "/full",
    response_model=ValidationResponse,
    summary="Run the full validation harness (synchronous, 3-5 min)",
)
async def run_full(request: ValidationFullRequest) -> ValidationResponse:
    """Run every selected tier synchronously.

    Warning: this endpoint blocks the HTTP worker for 3-5 minutes
    on a full cohort run. Prefer Tier 4 or Tier 2 alone for
    interactive use; reserve ``/v2/validation/full`` for CI jobs
    or overnight validation.
    """
    from neogen.validation.runner import ValidationRunner

    try:
        runner = ValidationRunner()
        report = runner.run_all(
            include_tier1=request.include_tier1,
            include_tier2=request.include_tier2,
            include_tier3=request.include_tier3,
            include_tier4=request.include_tier4,
        )
    except Exception as exc:
        logger.exception("Full validation failed")
        raise HTTPException(
            status_code=500, detail=f"full validation: {exc}"
        ) from exc

    if request.save_report:
        _save_report(report)
    return _wrap(report)


# ════════════════════════════════════════════════════════════════
# ASYNC FULL HARNESS (background task)
# ════════════════════════════════════════════════════════════════


@router.post(
    "/full/async",
    summary="Kick off the full validation harness in a background task",
)
async def run_full_async(
    request: ValidationFullRequest,
) -> Dict[str, Any]:
    """Start the full validation harness without blocking the worker.

    Returns immediately with::

        {"task_id": "ab12cd34...", "status": "pending", "status_url": "/v2/validation/status/ab12cd34"}

    Poll the status URL to monitor progress. Completed results are
    persisted to ``reports/validation/`` regardless of whether the
    caller retrieves them via the registry. The in-memory task
    entry is garbage-collected one hour after completion.
    """
    task = create_task(
        kind="validation_full",
        metadata={
            "include_tier1": request.include_tier1,
            "include_tier2": request.include_tier2,
            "include_tier3": request.include_tier3,
            "include_tier4": request.include_tier4,
            "save_report": request.save_report,
        },
    )

    def _work(progress) -> Dict[str, Any]:
        from neogen.engine.orchestrator_v2 import ENGINE_VERSION
        from neogen.validation.report import ValidationReport
        from neogen.validation.tier1 import Tier1Harness
        from neogen.validation.tier2 import Tier2Harness
        from neogen.validation.tier3 import Tier3Harness
        from neogen.validation.tier4 import Tier4Harness

        report_kwargs: Dict[str, Any] = dict(
            engine_version=ENGINE_VERSION,
            run_timestamp_utc=ValidationReport.now_timestamp(),
            notes=(),
        )

        total_tiers = sum(
            1
            for flag in (
                request.include_tier4,
                request.include_tier2,
                request.include_tier3,
                request.include_tier1,
            )
            if flag
        )
        if total_tiers == 0:
            progress.set("no tiers selected", 100.0)
            final = ValidationReport(**report_kwargs)
            if request.save_report:
                _save_report(final)
            return {
                "report": final.to_dict(),
                "overall_pass": final.overall_pass,
            }

        completed = 0

        def step(stage: str) -> None:
            progress.set(stage, 100.0 * completed / total_tiers)

        step("starting")

        if request.include_tier4:
            progress.set("tier4: mechanistic invariants", 100.0 * completed / total_tiers)
            report_kwargs["tier4"] = Tier4Harness().run()
            completed += 1

        if request.include_tier2:
            progress.set("tier2: DDI pairs", 100.0 * completed / total_tiers)
            report_kwargs["tier2"] = Tier2Harness().run()
            completed += 1

        if request.include_tier3:
            progress.set("tier3: bioavailability", 100.0 * completed / total_tiers)
            report_kwargs["tier3"] = Tier3Harness().run()
            completed += 1

        if request.include_tier1:
            progress.set("tier1: single-drug PK (slowest)", 100.0 * completed / total_tiers)
            report_kwargs["tier1"] = Tier1Harness().run()
            completed += 1

        progress.set("assembling report", 99.0)
        report = ValidationReport(**report_kwargs)

        if request.save_report:
            _save_report(report)

        return {
            "engine_version": report.engine_version,
            "run_timestamp_utc": report.run_timestamp_utc,
            "overall_pass": report.overall_pass,
            "report": report.to_dict(),
        }

    start_in_background(task, _work)

    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "status": task.status.value,
        "created_at": task.created_at,
        "status_url": f"/v2/validation/status/{task.task_id}",
    }


@router.get(
    "/status/{task_id}",
    summary="Poll the status of an async validation task",
)
async def task_status(task_id: str) -> Dict[str, Any]:
    """Return the current state of a background validation task.

    Returns 404 if the task_id is unknown (never existed or already
    garbage-collected one hour after completion).
    """
    task = get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Task {task_id!r} not found. It may have expired "
                f"after 1 hour of retention."
            ),
        )
    return task.to_dict()


# ════════════════════════════════════════════════════════════════
# LATEST-REPORT RETRIEVAL
# ════════════════════════════════════════════════════════════════


@router.get(
    "/latest",
    summary="Read the most recent persisted validation report",
)
async def latest_report() -> dict:
    """Return the JSON body of the most recent report in reports/validation/."""
    if not _REPORTS_DIR.exists():
        raise HTTPException(
            status_code=404, detail="No validation reports on disk"
        )
    json_files = sorted(_REPORTS_DIR.glob("validation_*.json"), reverse=True)
    if not json_files:
        raise HTTPException(
            status_code=404, detail="No validation reports on disk"
        )
    latest = json_files[0]
    try:
        return json.loads(latest.read_text())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"failed to read {latest.name}: {exc}",
        ) from exc


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _wrap(report) -> ValidationResponse:
    return ValidationResponse(
        engine_version=report.engine_version,
        run_timestamp_utc=report.run_timestamp_utc,
        overall_pass=report.overall_pass,
        report=report.to_dict(),
    )


def _save_report(report) -> Optional[tuple]:
    try:
        return report.save(_REPORTS_DIR)
    except Exception as exc:
        logger.warning("Failed to save validation report: %s", exc)
        return None
