"""NeoGen V1.1 ML Property Layer.

Lean wrappers around the trained ADME pipeline models that expose
single-property predictors as pure functions. The V1 engine
(:class:`neogen.engine.drug_factory.DrugFactory`) calls these when
a drug has no literature override, so novel compounds get
ML-predicted ADME parameters instead of hard defaults.

Design principles
-----------------

  * **Lazy loading.** Each predictor loads its joblib model on
    first call and caches the handle. Importing ``neogen.ml`` has
    near-zero cost when no prediction is needed.
  * **Pure functions.** Each exposed predictor is a module-level
    function that takes a SMILES string and returns a single
    scalar (or ``None`` on failure). No state between calls.
  * **Zero dependency on ``neogen.simulation``.** The old ML
    pipeline code in ``neogen.simulation.adme_pipeline`` loads
    ~20 models at construction and couples tightly to the legacy
    API. The ``neogen.ml`` layer duplicates only the minimum
    feature-computation code needed per predictor in
    :mod:`neogen.ml._features`.
  * **Graceful degradation.** If a model file is missing, RDKit
    is unavailable, or the SMILES does not parse, the predictor
    returns ``None`` and the caller falls back to literature /
    RDKit / defaults.
  * **XGBoost version safety.** All loaded models are patched via
    :func:`apply_xgboost_compat_shim` in :mod:`neogen.ml._features`
    so joblibs pickled with older XGBoost versions still work.

Current predictors
------------------

  * :func:`clint_predictor.predict_clint_microsomal` — hepatic
    microsomal intrinsic clearance (µL/min/mg). 1044-dim features.
    Closed GA-9 in session 3.
  * :func:`fu_predictor.predict_fu_plasma` — fraction unbound in
    plasma. 1044-dim features. Highest-leverage fallback because
    fu feeds every clearance and Kp calculation.
  * :func:`renal_predictor.predict_fraction_excreted_unchanged`
    — renal elimination fraction (mapped from a categorical
    classifier). 2083-dim features.
  * :func:`pka_predictor.predict_pka_acidic` /
    :func:`pka_predictor.predict_pka_basic` — physiological pKa
    values for ionization class + Kp partitioning. 2083-dim
    features.

Future predictors (V1.2+)
-------------------------

  * ``vd_predictor`` — direct Vss (1045-dim, needs fu appended).
  * ``cl_predictor`` — direct total CL (1053-dim, needs
    fu + CYP classifier flags).
  * ``papp_predictor`` — Caco-2 apparent permeability for P2
    absorption.
"""

from neogen.ml.clint_predictor import predict_clint_microsomal
from neogen.ml.cyp_classifier import predict_cyp_substrate_profile
from neogen.ml.fu_predictor import predict_fu_plasma
from neogen.ml.pka_predictor import (
    predict_pka_acidic,
    predict_pka_basic,
    predict_pkas,
)
from neogen.ml.renal_predictor import predict_fraction_excreted_unchanged
from neogen.ml.consistency_checker import check_pk_consistency, reconcile_pk_params
from neogen.ml.phase2_predictor import predict_phase2_profile

__all__ = [
    "predict_clint_microsomal",
    "predict_cyp_substrate_profile",
    "predict_fu_plasma",
    "predict_pka_acidic",
    "predict_pka_basic",
    "predict_pkas",
    "predict_fraction_excreted_unchanged",
    "check_pk_consistency",
    "reconcile_pk_params",
    "predict_phase2_profile",
]
