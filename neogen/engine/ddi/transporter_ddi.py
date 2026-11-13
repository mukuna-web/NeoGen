"""Transporter-mediated DDI — V1.2 Track B.

Steady-state IC50 modulation of hepatic active-uptake (OATP1B1)
at the hepatic inlet. Extends the V1 ``HepaticClearanceEngine``'s
flat 15x OATP1B1 uptake multiplier into a mechanism-aware path
that responds to co-administered inhibitors.

Scientific background
---------------------
OATP1B1 (SLCO1B1) is a sinusoidal uptake transporter expressed on
the basolateral membrane of hepatocytes. For its substrates
(statins, valsartan, repaglinide, bosentan, and many others),
active uptake concentrates the drug in hepatocytes so that
hepatocyte concentrations exceed portal-inlet plasma
concentrations by a factor of 5-30x at steady state. The
effective CLint seen by the body is therefore the intrinsic
microsomal CLint *amplified* by this uptake enhancement — the
Extended Clearance Model (Shitara 2006, Kusuhara 2013).

When an OATP1B1 inhibitor is co-dosed (cyclosporine, rifampicin,
gemfibrozil, lopinavir/ritonavir), the active uptake is
suppressed. Hepatocyte concentrations fall, the effective CLint
drops, and the victim's systemic AUC rises. Clinically important
pairs include:

  * atorvastatin + single-dose rifampicin (AUCR ~7x, Lau 2007)
  * pravastatin + cyclosporine (AUCR ~10x, Hedman 2004)
  * atorvastatin + cyclosporine (AUCR ~9x, Lemahieu 2005)
  * rosuvastatin + cyclosporine (AUCR ~7x, Simonson 2004)

Model
-----
V1.2 Track B uses the standard steady-state IC50 form::

    uptake_factor(t) = uptake_base / (1 + Sigma_k [I]_u_k / IC50_k)

where:

  * ``uptake_base`` — baseline active uptake amplification for the
    victim drug. From :attr:`TransporterAffinities.oatp1b1_uptake_factor`
    if set; otherwise the V1.1 default of 15.0.
  * ``[I]_u_k`` — unbound perpetrator concentration at the hepatic
    inlet (uM) during steady-state co-administration. This is the
    *sinusoidal* concentration, which for oral perpetrators is
    typically higher than systemic plasma because it includes the
    first-pass effect.
  * ``IC50_k`` — the perpetrator's OATP1B1 IC50 from
    :attr:`TransporterAffinities.oatp1b1_ic50`.
  * Sum is over all OATP1B1 inhibitors in the exposure list
    (additive in ratio-to-IC50 space, which is the standard
    static-cocktail assumption; V2+ will model mixed-type
    inhibition more carefully).

The modulated uptake factor is then applied to the well-stirred
model inside :class:`HepaticClearanceEngine`::

    CLint_effective = CLint_microsomal_scaled * uptake_factor
    CLh             = Qh * fu * CLint_effective / (Qh + fu * CLint_effective)

No uptake inhibitors → full baseline enhancement (matches V1.1).
Complete inhibition ([I] >> IC50) → uptake_factor -> 0, so
CLh -> Qh * fu * CLint_base / (Qh + fu * CLint_base) which is
the passive-diffusion-only CLh and the largest possible AUC shift
from transporter inhibition alone.

Scope
-----
V1.2 Track B models OATP1B1 at **steady state** only, with an
**IC50-based** inhibition model (no Ki vs Km distinction, no
saturable Vmax). This matches every clinical DDI study protocol
(perpetrator dosed for days before victim) and uses parameters
that are available in published literature for most clinically
relevant pairs.

Not yet modelled (V1.3+):
  * Time-varying transporter expression (analogous to enzyme pool
    evolver for CYPs — transporter protein pools turn over but
    OATP1B1 is not a clinically important MBI target).
  * Mixed competitive / non-competitive inhibition with explicit
    Ki/Km. The IC50 form is adequate for most literature pairs.
  * Other hepatic uptake transporters (OATP1B3, OCT1, NTCP).
    V1.2 wires only OATP1B1; the module structure allows adding
    siblings in V1.3.
  * Efflux transporter DDIs on the sinusoidal (MRP3, MRP4) or
    canalicular (MRP2, BCRP) side. These would need a second
    module with different kinetics.

References
----------
  * Shitara Y, Horie T, Sugiyama Y. Transporters as a determinant
    of drug clearance and tissue distribution. Eur J Pharm Sci
    2006; 27:425-446.
  * Kusuhara H, Sugiyama Y. In vitro-in vivo extrapolation of
    transporter-mediated clearance in the liver and kidney.
    Drug Metab Pharmacokinet 2013; 28:57-68.
  * Varma MV et al. Mechanism-based approach for prediction of
    hepatic clearance of drugs. Drug Metab Dispos 2012; 40:1527.
  * Lau YY, Huang Y, Frassetto L, Benet LZ. Effect of OATP1B
    transporter inhibition on the pharmacokinetics of
    atorvastatin in healthy volunteers. Clin Pharmacol Ther
    2007; 81:194-204. (Atorvastatin + rifampicin AUCR 7x)
  * Hedman A et al. Digoxin-verapamil interaction: reduction of
    biliary but not renal digoxin clearance in humans. Clin
    Pharmacol Ther 1991; 49:256-262.
  * Simonson SG et al. Rosuvastatin pharmacokinetics in heart
    transplant recipients administered an antirejection regimen
    including cyclosporine. Clin Pharmacol Ther 2004; 76:167-177.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from neogen.core.entities.drug import Drug

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════

#: V1.1 default OATP1B1 active uptake amplification factor.
#: Used as the fallback when a drug's entity does not set
#: ``TransporterAffinities.oatp1b1_uptake_factor``. Matches the
#: flat multiplier used by V1.1 ``HepaticClearanceEngine``.
DEFAULT_OATP1B1_UPTAKE_FACTOR: float = 15.0


# ════════════════════════════════════════════════════════════════
# EXPOSURE RECORD
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OATP1B1InhibitorExposure:
    """Steady-state OATP1B1 inhibitor exposure at the hepatic inlet.

    Attributes:
        perpetrator: The V1 :class:`Drug` whose
            :class:`TransporterAffinities` carries the
            ``oatp1b1_ic50`` parameter. Drugs without
            ``is_oatp1b1_inhibitor=True`` or with ``oatp1b1_ic50=None``
            are silently ignored — the exposure contributes zero
            to the inhibition sum.
        unbound_portal_conc_um: Unbound inhibitor concentration at
            the hepatic inlet (sinusoidal blood) in micromolar.
            For oral perpetrators this is typically higher than
            systemic unbound Cmax because it includes first-pass
            residence time in the portal vein. For IV perpetrators
            it is approximately the unbound plasma Cmax.
            Typical clinical values:

              rifampicin 600 mg PO single   :  ~2 uM unbound portal
              cyclosporine 100 mg PO BID    :  ~1 uM unbound portal
              gemfibrozil 600 mg PO BID     :  ~10 uM unbound portal
              lopinavir/ritonavir 400/100   :  ~3 uM unbound portal

            These are the *unbound* concentrations; total plasma
            is much higher for highly bound inhibitors.
    """

    perpetrator: Drug
    unbound_portal_conc_um: float

    def __post_init__(self) -> None:
        if self.unbound_portal_conc_um < 0.0:
            raise ValueError(
                f"unbound_portal_conc_um must be >= 0 (got "
                f"{self.unbound_portal_conc_um})"
            )

    @property
    def has_oatp_ic50(self) -> bool:
        """True when the perpetrator is flagged as an OATP1B1
        inhibitor AND has an IC50 value set on its transporter
        affinities. Other drugs contribute zero to the inhibition
        sum even if they somehow land in an exposure list."""
        t = self.perpetrator.transporters
        return bool(
            getattr(t, "is_oatp1b1_inhibitor", False)
            and t.oatp1b1_ic50 is not None
            and t.oatp1b1_ic50 > 0.0
        )


# ════════════════════════════════════════════════════════════════
# CORE MATH
# ════════════════════════════════════════════════════════════════


def _sum_inhibition_ratios(
    exposures: Sequence[OATP1B1InhibitorExposure],
) -> float:
    """Sum of ``[I]_u / IC50`` across valid OATP1B1 inhibitors.

    Additive across perpetrators per the static-cocktail
    convention used in FDA DDI guidance — treats each inhibitor
    as contributing independently to the inhibition sum. Drugs
    without OATP1B1 inhibition parameters contribute zero.
    """
    total = 0.0
    for exp in exposures:
        if not exp.has_oatp_ic50:
            continue
        ic50 = exp.perpetrator.transporters.oatp1b1_ic50
        total += exp.unbound_portal_conc_um / ic50
    return total


def compute_oatp1b1_uptake_factor(
    victim: Drug,
    exposures: Sequence[OATP1B1InhibitorExposure] = (),
) -> float:
    """Return the modulated OATP1B1 uptake factor for the victim.

    Formula::

        uptake_effective = uptake_base / (1 + Sigma_k [I]_u_k / IC50_k)

    where ``uptake_base`` is the victim's baseline active-uptake
    enhancement (per-drug override or the default 15.0), and the
    sum is over all valid OATP1B1 inhibitors in ``exposures``.

    When ``victim.transporters.is_oatp1b1_substrate`` is False,
    or the victim has ``oatp1b1_uptake_factor=1.0`` explicitly,
    the result is 1.0 regardless of inhibitors (no uptake
    enhancement to modulate).

    When ``exposures`` is empty (or contains no valid OATP1B1
    inhibitors), the result is ``uptake_base`` unchanged —
    matching V1.1 byte-for-byte for the no-DDI path.

    Returns:
        float in [1.0, uptake_base]. Always >= 1.0 because the
        denominator is >= 1.0. The lower bound of 1.0 represents
        "complete OATP1B1 inhibition leaves only passive uptake",
        which is the maximum AUCR shift from OATP1B1 DDI alone.
    """
    if not getattr(victim.transporters, "is_oatp1b1_substrate", False):
        return 1.0

    uptake_base = _victim_uptake_base(victim)
    if uptake_base <= 1.0:
        # Drug has uptake_factor=1.0 or smaller set — explicit
        # declaration that active uptake is negligible. Don't
        # modulate.
        return max(uptake_base, 1.0)

    denom = 1.0 + _sum_inhibition_ratios(exposures)
    if denom <= 0.0:  # pragma: no cover — defensive
        return uptake_base
    effective = uptake_base / denom
    # Floor at 1.0: even with total uptake suppression the drug
    # can still enter hepatocytes via passive diffusion. The
    # "effective CLint = CLint_base × 1.0" regime corresponds to
    # the well-stirred model with no active uptake, which is
    # the correct physical limit.
    return max(effective, 1.0)


def _victim_uptake_base(victim: Drug) -> float:
    """Return the victim's baseline OATP1B1 uptake factor.

    Order of precedence:
      1. ``victim.transporters.oatp1b1_uptake_factor`` if set
         (per-drug literature override).
      2. ``DEFAULT_OATP1B1_UPTAKE_FACTOR`` (V1.1 flat 15x).

    Non-OATP1B1 substrates should not call this — it returns the
    default even for non-substrates, which is why
    :func:`compute_oatp1b1_uptake_factor` guards with
    ``is_oatp1b1_substrate`` first.
    """
    override = getattr(
        victim.transporters, "oatp1b1_uptake_factor", None
    )
    if override is not None and override >= 0.0:
        return float(override)
    return DEFAULT_OATP1B1_UPTAKE_FACTOR


# ════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OATP1B1DDIDiagnostic:
    """Audit snapshot of the OATP1B1 DDI computation.

    Emitted alongside :class:`HepaticClearanceResult` when a V1.2
    Track B DDI run is active. Lets tests and reports check
    exactly which inhibitors contributed and by how much.
    """

    victim_name: str
    uptake_base: float
    n_inhibitors: int
    inhibitor_names: tuple
    total_inhibition_ratio: float
    uptake_effective: float

    @property
    def suppression_percent(self) -> float:
        """How much the uptake enhancement was suppressed, as
        a percent of the baseline. 0% = no effect, 100% = complete
        passive-only regime."""
        if self.uptake_base <= 1.0:
            return 0.0
        dropped = self.uptake_base - self.uptake_effective
        total = self.uptake_base - 1.0
        return 100.0 * max(0.0, min(dropped / total, 1.0))


def diagnose_oatp1b1_ddi(
    victim: Drug,
    exposures: Sequence[OATP1B1InhibitorExposure] = (),
) -> OATP1B1DDIDiagnostic:
    """Return a frozen audit record of the OATP1B1 DDI computation."""
    valid = [e for e in exposures if e.has_oatp_ic50]
    base = _victim_uptake_base(victim) if getattr(
        victim.transporters, "is_oatp1b1_substrate", False
    ) else 1.0
    effective = compute_oatp1b1_uptake_factor(victim, exposures)
    return OATP1B1DDIDiagnostic(
        victim_name=victim.name,
        uptake_base=base,
        n_inhibitors=len(valid),
        inhibitor_names=tuple(e.perpetrator.name for e in valid),
        total_inhibition_ratio=_sum_inhibition_ratios(exposures),
        uptake_effective=effective,
    )
