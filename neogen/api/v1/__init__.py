"""NeoGen V1 mechanistic simulation API (Phase P7).

This package exposes the V1 engine (OrchestratorV2) as a thin, stateless
HTTP surface. It is intentionally separate from the legacy ML-predict
API in ``neogen/api/predict.py`` — that one returns XGBoost-fitted PK
predictions from a SMILES string, whereas this one runs a real
14-compartment PBPK + P2 absorption + P3 DDI + P4 PD simulation for a
literature drug, patient, and schedule.

Public entry points
-------------------
* :func:`build_app` — factory that wires the FastAPI app, CORS,
  routers, and error handlers. Use it in tests and ASGI servers alike.
* :data:`app` — a pre-built default instance for uvicorn::

      uvicorn neogen.api.v1:app --reload

The default app has CORS locked to ``localhost`` origins; production
deployments should call :func:`build_app` with an explicit allow-list.
"""
from __future__ import annotations

from neogen.api.v1.app import build_app

app = build_app()

__all__ = ["app", "build_app"]
