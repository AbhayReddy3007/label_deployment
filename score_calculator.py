"""Label Expansion Score Calculation — Module 3: score_calculator.

Ported from the reference ``cal3.py``: takes the TA-I summary rows
produced by ``trial_selector`` (one row per therapy_area / OT-disease
combination — the highest-weighted trial for each) and derives the full
scoring chain, ending in a single ``Final Score`` per drug.

Steps (numbered to match the reference):
    1.  prior                    - from association_score
    2.  maturity_weight          - from phase
    3.  effective_indications / effective_therapy_areas
    4.  w_geo                    - from primary_region tier
    5.  w_dose                   - dosage rank within (TA-I, dosage)
    6.  w_sample                 - from drug_arm_size_n
    6b. Non-trial overrides      - w_geo/w_dose/w_sample forced to 1.00
    7.  Q_i = w_geo x w_sample x w_dose
    8b. e_phase_i                - phase x association bucket lookup
    8c. Non-trial override       - e_phase_i forced to 0.05
    8.  e_i = Q_i x e_phase_i
    8d. Approved override        - e_i forced to 1.00
    9.  Link = 1 - (1 - prior) x (1 - e_i)
    10. Link_TA                  - mean of Link per therapy_area
    11. L_ind / B_raw_ind / B_ind   - indication-breadth logistic curve
    12. L_TA / B_raw_TA / B_TA      - therapy-area-breadth logistic curve
    13. B = B_ind x B_TA
    14. Overall Coherence
    15. C = 0.1 + 0.9 x (Overall Coherence)^1.75
    16. Final Score = 1 + 4 x B x C
"""

from __future__ import annotations

import logging
import math
import re

import pandas as pd

from medical_potential.config import DRUG_NAME

from ..bq_utils import push_score_calculation
from .trial_selector import phase_rank, select_trials

logger = logging.getLogger(__name__)

# ==============================
# CONSTANTS
# ==============================
_N0_IND, _A_IND = 9, 0.40      # indication-breadth logistic curve
_N0_TA, _A_TA = 3, 0.9         # therapy-area-breadth logistic curve

_E_PHASE_TABLE = {
    ("phase1", "obvious"): 0.10, ("phase1", "indirect"): 0.10, ("phase1", "novel"): 0.10,
    ("phase2", "obvious"): 0.40, ("phase2", "indirect"): 0.35, ("phase2", "novel"): 0.30,
    ("phase3", "obvious"): 0.80, ("phase3", "indirect"): 0.65, ("phase3", "novel"): 0.55,
    ("approved", "obvious"): 1.00, ("approved", "indirect"): 1.00, ("approved", "novel"): 1.00,
}


def _is_missing(val) -> bool:
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (ValueError, TypeError):
        pass
    if isinstance(val, str) and val.strip() in ("", "nan", "None"):
        return True
    return False


def _phase_bucket(val) -> "str | None":
    if _is_missing(val):
        return None
    text = str(val).strip().lower()
    if re.search(r"\b(approved|approv|marketed|market)\b", text):
        return "approved"
    if re.search(r"\biv\b", text) or re.search(r"\b4\b", text):
        return "approved"
    if re.search(r"\biii\b", text) or re.search(r"\b3\b", text):
        return "phase3"
    if re.search(r"\bii\b", text) or re.search(r"\b2\b", text):
        return "phase2"
    if re.search(r"\bi\b", text) or re.search(r"\b1\b", text):
        return "phase1"
    return None


def _assoc_bucket(val) -> str:
    if _is_missing(val):
        return "novel"
    try:
        score = float(val)
    except (ValueError, TypeError):
        return "novel"
    if score > 0.40:
        return "obvious"
    if score >= 0.10:
        return "indirect"
    return "novel"


def _non_trial_mask(df: pd.DataFrame):
    if "data_source" not in df.columns:
        return None

    def _is_trials(val) -> bool:
        return not _is_missing(val) and str(val).strip().lower() == "trials"

    return ~df["data_source"].apply(_is_trials)


# ===========================================================================
# 1. prior
# ===========================================================================
def add_prior(df: pd.DataFrame) -> pd.DataFrame:
    """association_score > 0.40 -> 0.8 | 0.10-0.40 -> 0.4 | <0.10/missing -> 0.0"""
    def _prior(val):
        if _is_missing(val):
            return 0.0
        try:
            score = float(val)
        except (ValueError, TypeError):
            return 0.0
        if score > 0.40:
            return 0.8
        if score >= 0.10:
            return 0.4
        return 0.0

    df["prior"] = df.get("association_score", pd.Series(dtype=float)).apply(_prior) \
        if "association_score" in df.columns else 0.0
    return df


# ===========================================================================
# 2. maturity_weight
# ===========================================================================
def add_maturity_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Preclinical/missing -> 0.05 | P1 -> 0.10 | P2 -> 0.30 | P3 -> 0.60 | Approved/P4 -> 1.00"""
    def _maturity(val):
        if _is_missing(val):
            return 0.05
        text = str(val).strip().lower()
        if re.search(r"\b(approved|approv|marketed|market)\b", text):
            return 1.00
        if re.search(r"\bpreclinical\b", text):
            return 0.05
        if re.search(r"\biv\b", text) or re.search(r"\b4\b", text):
            return 1.00
        if re.search(r"\biii\b", text) or re.search(r"\b3\b", text):
            return 0.60
        if re.search(r"\bii\b", text) or re.search(r"\b2\b", text):
            return 0.30
        if re.search(r"\bi\b", text) or re.search(r"\b1\b", text):
            return 0.10
        return 0.05

    df["maturity_weight"] = df["phase"].apply(_maturity) if "phase" in df.columns else 0.05
    return df


# ===========================================================================
# 3. effective_indications / effective_therapy_areas
# ===========================================================================
def add_effective_indications(df: pd.DataFrame, drug_col: str) -> pd.DataFrame:
    """effective_indications: drug-level sum of maturity_weight.
    effective_therapy_areas: sum over each TA of mean(maturity_weight within TA)."""
    drug_sum = df.groupby(drug_col, sort=False)["maturity_weight"].sum().rename("_drug_sum")
    df = df.join(drug_sum, on=drug_col)
    df["effective_indications"] = df["_drug_sum"]
    df = df.drop(columns=["_drug_sum"])

    if "therapy_area" in df.columns:
        eff_ta = df.groupby("therapy_area", sort=False)["maturity_weight"].mean().sum()
        df["effective_therapy_areas"] = eff_ta
    else:
        df["effective_therapy_areas"] = float("nan")
    return df


# ===========================================================================
# 4. w_geo
# ===========================================================================
_EU_COUNTRY_NAMES = {
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic",
    "czechia", "denmark", "estonia", "finland", "france", "germany", "greece",
    "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta",
    "netherlands", "poland", "portugal", "romania", "slovakia", "slovenia",
    "spain", "sweden",
}
_TIER2_NAMES = {"canada", "switzerland", "australia", "japan"}


def _region_tier(region_val) -> int:
    if _is_missing(region_val):
        return 3
    text = str(region_val).strip().lower()
    if re.search(r"\b(us|usa|united states|u\.s\.a?\.?)\b", text):
        return 1
    if re.search(r"\b(uk|u\.k\.|united kingdom|great britain|gb)\b", text):
        return 1
    if re.search(r"\b(europe|eu|european union|e\.u\.)\b", text):
        return 1
    if text in _EU_COUNTRY_NAMES:
        return 1
    if text in _TIER2_NAMES or re.search(r"\b(canada|switzerland|australia|japan)\b", text):
        return 2
    return 3


def add_w_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 1 (US/UK/EU) -> 1.00 | Tier 2 (CA/CH/AU/JP) -> 0.85 | Tier 3 -> 0.65.
    Falls back to the pre-computed geo_score from trial_selector if primary_region is absent."""
    if "primary_region" in df.columns:
        weight_map = {1: 1.00, 2: 0.85, 3: 0.65}
        df["w_geo"] = df["primary_region"].apply(lambda r: weight_map[_region_tier(r)])
    elif "geo_score" in df.columns:
        df["w_geo"] = df["geo_score"]
    else:
        df["w_geo"] = 0.65
    return df


# ===========================================================================
# 5. w_dose
# ===========================================================================
def add_w_dose(df: pd.DataFrame, drug_col: str) -> pd.DataFrame:
    """Dense-rank rows within (ta_i, dosage) by phase descending.
    Rank 1 -> 1.00; rank 2+ -> 0.75. Missing dosage -> 1.00."""
    tai_col = "ta_i" if "ta_i" in df.columns else drug_col
    required = {"dosage", "phase", tai_col}
    if required - set(df.columns):
        df["w_dose"] = df["dosage_score"] if "dosage_score" in df.columns else 1.0
        return df

    _MISSING = "__missing__"
    df["_dose_key"] = df["dosage"].apply(lambda v: _MISSING if _is_missing(v) else str(v).strip().lower())
    df["_phase_rank_num"] = df["phase"].apply(phase_rank)
    df["_dose_rank"] = (
        df.groupby([tai_col, "_dose_key"], sort=False, dropna=False)["_phase_rank_num"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df["w_dose"] = df["_dose_rank"].apply(lambda r: 1.00 if r == 1 else 0.75)
    df.loc[df["_dose_key"] == _MISSING, "w_dose"] = 1.0
    df = df.drop(columns=["_dose_key", "_phase_rank_num", "_dose_rank"])
    return df


# ===========================================================================
# 6. w_sample
# ===========================================================================
def add_w_sample(df: pd.DataFrame) -> pd.DataFrame:
    """>=500 -> 1.00 | 200-499 -> 0.85 | 50-199 -> 0.65 | <50/missing -> 0.40.
    Falls back to the pre-computed sample_score if drug_arm_size_n is absent."""
    if "drug_arm_size_n" in df.columns:
        def _w_sample(val):
            if _is_missing(val):
                return 0.40
            try:
                n = float(val)
            except (ValueError, TypeError):
                return 0.40
            if n >= 500:
                return 1.00
            if n >= 200:
                return 0.85
            if n >= 50:
                return 0.65
            return 0.40

        df["w_sample"] = df["drug_arm_size_n"].apply(_w_sample)
    elif "sample_score" in df.columns:
        df["w_sample"] = df["sample_score"]
    else:
        df["w_sample"] = 0.40
    return df


# ===========================================================================
# 6b / 8c. Non-trial overrides
# ===========================================================================
def add_non_trial_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Forces w_geo/w_dose/w_sample to 1.00 for rows where data_source != 'Trials'."""
    mask = _non_trial_mask(df)
    if mask is not None:
        df.loc[mask, ["w_geo", "w_dose", "w_sample"]] = 1.00
    return df


def add_non_trial_e_phase_override(df: pd.DataFrame) -> pd.DataFrame:
    """Forces e_phase_i to 0.05 for rows where data_source != 'Trials'."""
    mask = _non_trial_mask(df)
    if mask is not None:
        df.loc[mask, "e_phase_i"] = 0.05
    return df


# ===========================================================================
# 7. Q_i
# ===========================================================================
def add_Q_i(df: pd.DataFrame) -> pd.DataFrame:
    """Q_i = w_geo x w_sample x w_dose"""
    df["q_i"] = df["w_geo"] * df["w_sample"] * df["w_dose"]
    return df


# ===========================================================================
# 8b. e_phase_i
# ===========================================================================
def add_e_phase_i(df: pd.DataFrame) -> pd.DataFrame:
    """Phase x association-bucket lookup table."""
    def _lookup(row):
        pb = _phase_bucket(row.get("phase"))
        ab = _assoc_bucket(row.get("association_score"))
        if pb is None:
            return float("nan")
        return _E_PHASE_TABLE[(pb, ab)]

    df["e_phase_i"] = df.apply(_lookup, axis=1)
    return df


# ===========================================================================
# 8. e_i
# ===========================================================================
def add_e_i(df: pd.DataFrame) -> pd.DataFrame:
    """e_i = Q_i x e_phase_i"""
    df["e_i"] = df["q_i"] * df["e_phase_i"]
    return df


# ===========================================================================
# 8d. Approved-phase override for e_i
# ===========================================================================
def add_approved_e_i_override(df: pd.DataFrame) -> pd.DataFrame:
    """Forces e_i to 1.00 for rows where phase is Approved / Phase IV."""
    if "phase" in df.columns:
        approved_mask = df["phase"].apply(_phase_bucket) == "approved"
        df.loc[approved_mask, "e_i"] = 1.00
    return df


# ===========================================================================
# 9. Link
# ===========================================================================
def add_link(df: pd.DataFrame) -> pd.DataFrame:
    """Link = 1 - (1 - prior) x (1 - e_i)"""
    df["link"] = 1 - (1 - df["prior"]) * (1 - df["e_i"])
    return df


# ===========================================================================
# 10. Link_TA
# ===========================================================================
def add_link_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Link_TA = maturity-weighted average of Link per therapy_area.

    weighted_link_ta = SUM(maturity_weight * link)
    maturity_weight_sum = SUM(maturity_weight)
    link_ta = weighted_link_ta / maturity_weight_sum

    Falls back to an unweighted mean for a therapy_area whose rows all
    have maturity_weight == 0 (weighted average is undefined - division
    by zero), so a TA with no mature evidence doesn't just disappear.
    """
    if "therapy_area" not in df.columns:
        df["link_ta"] = float("nan")
        return df

    weighted_link = df["maturity_weight"] * df["link"]
    weighted_sum = weighted_link.groupby(df["therapy_area"], sort=False, dropna=False).transform("sum")
    weight_sum = df["maturity_weight"].groupby(df["therapy_area"], sort=False, dropna=False).transform("sum")

    with pd.option_context("mode.use_inf_as_na", True):
        link_ta = weighted_sum / weight_sum

    # weight_sum == 0 -> undefined weighted average; fall back to the
    # plain (unweighted) mean of link for that therapy_area.
    zero_weight_mask = weight_sum == 0
    if zero_weight_mask.any():
        unweighted_mean = df.groupby("therapy_area", sort=False, dropna=False)["link"].transform("mean")
        link_ta = link_ta.where(~zero_weight_mask, unweighted_mean)

    df["link_ta"] = link_ta
    return df


# ===========================================================================
# 11. L_ind / B_raw_ind / B_ind
# ===========================================================================
def _l_ind(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-_A_IND * (x - _N0_IND)))


def _b_raw_ind(x: float, l_ind_0: float) -> float:
    return (_l_ind(x) - l_ind_0) / (1.0 - l_ind_0)


def add_indication_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level constants derived from effective_indications via a
    logistic curve, normalised against a benchmark of 15 effective indications."""
    n_eff = df["effective_indications"].iloc[0]
    l0 = _l_ind(0)
    l_x = _l_ind(n_eff)
    b_raw_x = _b_raw_ind(n_eff, l0)
    b_raw_15 = _b_raw_ind(15, l0)

    b_ind = float("nan") if abs(b_raw_15) < 1e-12 else min(1.0, b_raw_x / b_raw_15)

    df["l_ind"] = l_x
    df["b_raw_ind"] = b_raw_x
    df["b_ind"] = b_ind
    return df


# ===========================================================================
# 12. L_TA / B_raw_TA / B_TA
# ===========================================================================
def _l_ta(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-_A_TA * (x - _N0_TA)))


def _b_raw_ta(x: float, l_ta_0: float) -> float:
    return (_l_ta(x) - l_ta_0) / (1.0 - l_ta_0)


def add_therapy_area_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level constants derived from effective_therapy_areas via a
    logistic curve, normalised against a benchmark of 5 effective therapy areas."""
    n_eff_ta = df["effective_therapy_areas"].iloc[0]
    l0 = _l_ta(0)
    l_x = _l_ta(n_eff_ta)
    b_raw_x = _b_raw_ta(n_eff_ta, l0)
    b_raw_5 = _b_raw_ta(5, l0)

    b_ta = float("nan") if abs(b_raw_5) < 1e-12 else min(1.0, b_raw_x / b_raw_5)

    df["l_ta"] = l_x
    df["b_raw_ta"] = b_raw_x
    df["b_ta"] = b_ta
    return df


# ===========================================================================
# 13. B
# ===========================================================================
def add_B(df: pd.DataFrame) -> pd.DataFrame:
    """B = B_ind x B_TA"""
    df["b"] = df["b_ind"] * df["b_ta"]
    return df


# ===========================================================================
# 14. Overall Coherence
# ===========================================================================
def add_overall_coherence(df: pd.DataFrame) -> pd.DataFrame:
    """Overall Coherence = (sum(W_i * sqrt(L_i)) / sum(W_i))^2
    W_i = unique ot_disease_name count within therapy area i; L_i = Link_TA."""
    ta_summary = (
        df.groupby("therapy_area", sort=False)
        .agg(W=("ot_disease_name", "nunique"), L=("link_ta", "first"))
        .reset_index()
    )
    ta_summary["W_sqrt_L"] = ta_summary["W"] * ta_summary["L"].clip(lower=0).pow(0.5)
    sum_w_sqrt_l = ta_summary["W_sqrt_L"].sum()
    sum_w = ta_summary["W"].sum()

    overall_coherence = float("nan") if sum_w == 0 else (sum_w_sqrt_l / sum_w) ** 2
    df["overall_coherence"] = overall_coherence
    return df


# ===========================================================================
# 15. C
# ===========================================================================
def add_C(df: pd.DataFrame) -> pd.DataFrame:
    """C = 0.1 + 0.9 x (Overall Coherence)^1.75"""
    overall_coherence = df["overall_coherence"].iloc[0]
    df["c"] = 0.1 + 0.9 * (overall_coherence ** 1.75)
    return df


# ===========================================================================
# 16. Final Score
# ===========================================================================
def add_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """Final Score = 1 + 4 x B x C"""
    b = df["b"].iloc[0]
    c = df["c"].iloc[0]
    df["final_score"] = 1 + 4 * b * c
    return df


# ==============================
# MAIN PIPELINE
# ==============================
def run_score_calculation(drug_name: str = DRUG_NAME, push: bool = True, secondary_only: bool = False) -> list[dict]:
    """Full score-calculation pipeline for one drug.

    1. Select the best trial per TA-I (via ``trial_selector``).
    2. Run all 16 derived-column calculations in order.
    3. Optionally push the result to ``LE_SCORE_CALCULATION_TABLE``.

    Args:
        drug_name: the drug/molecule name.
        push: whether to push results to BigQuery.
        secondary_only: if ``True``, only processes Secondary indications.

    Returns a list of dicts, one per TA-I row, with every computed column.
    """
    rows = select_trials(drug_name, secondary_only=secondary_only)
    if not rows:
        logger.warning("[SCORE_CALC] No TA-I rows to score for '%s'", drug_name)
        return []

    df = pd.DataFrame(rows)
    drug_col = "drug_name" if "drug_name" in df.columns else df.columns[0]

    logger.info("[SCORE_CALC] Running score calculations for '%s' (%d TA-I row(s))", drug_name, len(df))

    df = add_prior(df)                           # 1
    df = add_maturity_weight(df)                 # 2
    df = add_effective_indications(df, drug_col) # 3
    df = add_w_geo(df)                           # 4
    df = add_w_dose(df, drug_col)                # 5
    df = add_w_sample(df)                        # 6
    df = add_non_trial_overrides(df)             # 6b
    df = add_Q_i(df)                             # 7
    df = add_e_phase_i(df)                       # 8b
    df = add_non_trial_e_phase_override(df)      # 8c
    df = add_e_i(df)                             # 8
    df = add_approved_e_i_override(df)           # 8d
    df = add_link(df)                            # 9
    df = add_link_ta(df)                         # 10
    df = add_indication_breadth(df)              # 11
    df = add_therapy_area_breadth(df)            # 12
    df = add_B(df)                               # 13
    df = add_overall_coherence(df)               # 14
    df = add_C(df)                                # 15
    df = add_final_score(df)                     # 16

    df = df.astype(object).where(pd.notnull(df), None)
    result_rows = df.to_dict(orient="records")

    logger.info(
        "[SCORE_CALC] Completed for '%s'. Final Score = %.4f",
        drug_name, result_rows[0]["final_score"] if result_rows else float("nan"),
    )

    if push:
        push_score_calculation(result_rows)

    return result_rows
