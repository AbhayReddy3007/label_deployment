"""Label Expansion Score Calculation — Module 2: trial_selector.

Ported from the reference ``trial_weight.py`` calculator: computes a
composite ``trial_weight`` for every enriched row from ``data_fetcher``,
then selects the single highest-weighted row per TA-I (therapy_area +
Open Targets disease name) group — mirroring the "TA-I Summary" sheet
that ``trial_weight.py`` produces.

Weight components (trial-sourced rows only; non-trial rows get a fixed
``trial_weight = 0.05``):
    phase_weight : Phase 3/Approved -> 1.00 | Phase 2 -> 0.75 | Phase 1 -> 0.50 | NA -> 0.05
    geo_score    : US/UK/EU -> 1.00 | CA/CH/AU/JP -> 0.85 | Other -> 0.65
    sample_score : >=500 -> 1.00 | 300-499 -> 0.85 | 50-299 -> 0.65 | <50 -> 0.40
    dosage_score : ranked within TA-I by phase then arm size -> 1.00 / 0.75 / 0.50
    trial_weight = phase_weight x geo_score x sample_score x dosage_score
                   (Approved trials always get trial_weight = 1.0)
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from medical_potential.config import DRUG_NAME

from ..ot_mapping.ot_utils import OT_DISEASE_TABLE, fetch_existing_mappings
from .data_fetcher import fetch_and_enrich_trial_data

logger = logging.getLogger(__name__)

# ==============================
# EU / TIER-2 REGION SETS
# ==============================
_EU_COUNTRY_NAMES = {
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic",
    "czechia", "denmark", "estonia", "finland", "france", "germany", "greece",
    "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta",
    "netherlands", "poland", "portugal", "romania", "slovakia", "slovenia",
    "spain", "sweden",
}
_TIER2 = {"canada", "switzerland", "australia", "japan"}


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


# ==============================
# PHASE HELPERS
# ==============================
def normalize_phase(phase_value):
    if _is_missing(phase_value):
        return phase_value
    text = str(phase_value).strip()
    lower = text.lower()
    if "approved" in lower or "approv" in lower or "market" in lower or "submitted" in lower:
        return "Approved"
    if re.search(r"3b", lower) or re.search(r"iiib", lower):
        return "Phase 3"
    return text


def phase_rank(phase_value) -> int:
    if _is_missing(phase_value):
        return -1
    text = str(phase_value).strip().lower()
    if "approved" in text or "approv" in text or "market" in text or "submitted" in text:
        return 4
    if re.search(r"3b", text) or re.search(r"iiib", text):
        return 3
    roman = {"iii": 3, "ii": 2, "i": 1, "iv": 4}
    for numeral, val in roman.items():
        if re.search(rf"\b{numeral}\b", text):
            return val
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return int(m.group(1))
    return -1


# ==============================
# WEIGHT COMPONENTS
# ==============================
def compute_phase_weight(phase_value) -> float:
    rank = phase_rank(phase_value)
    if rank >= 3:
        return 1.00
    if rank == 2:
        return 0.75
    if rank == 1:
        return 0.50
    return 0.05


def compute_geo_score(primary_region) -> float:
    if _is_missing(primary_region):
        return 0.65
    text = str(primary_region).strip().lower()
    if re.search(r"\b(us|usa|united states|u\.s\.a?\.?)\b", text):
        return 1.00
    if re.search(r"\b(uk|u\.k\.|united kingdom|great britain|gb)\b", text):
        return 1.00
    if re.search(r"\b(europe|eu|european union|e\.u\.)\b", text):
        return 1.00
    if text in _EU_COUNTRY_NAMES:
        return 1.00
    if text in _TIER2 or re.search(r"\b(canada|switzerland|australia|japan)\b", text):
        return 0.85
    return 0.65


def compute_sample_score(drug_arm_size_n) -> float:
    if _is_missing(drug_arm_size_n):
        return 0.40
    try:
        n = float(drug_arm_size_n)
    except (ValueError, TypeError):
        return 0.40
    if n >= 500:
        return 1.00
    if n >= 300:
        return 0.85
    if n >= 50:
        return 0.65
    return 0.40


def assign_dosage_scores(df: pd.DataFrame) -> pd.Series:
    """Rank dosages within each TA-I group. Priority: highest phase_rank ->
    highest arm size. Rank 1 -> 1.00, Rank 2 -> 0.75, Rank 3+ -> 0.50."""
    scores = pd.Series(0.50, index=df.index, dtype=float)
    tmp = df.copy()
    tmp["_phase_rank_tmp"] = tmp["phase"].apply(phase_rank)
    tmp["_arm_size_tmp"] = pd.to_numeric(tmp.get("drug_arm_size_n"), errors="coerce").fillna(0)

    for _tai, group in tmp.groupby("ta_i", sort=False):
        sorted_idx = group.sort_values(
            ["_phase_rank_tmp", "_arm_size_tmp"], ascending=[False, False]
        ).index
        for rank_pos, idx in enumerate(sorted_idx):
            scores.at[idx] = 1.00 if rank_pos == 0 else (0.75 if rank_pos == 1 else 0.50)
    return scores


# ==============================
# TA-I GROUPING KEY
# ==============================
def _build_ta_i_column(df: pd.DataFrame, drug_name: str) -> pd.DataFrame:
    """Adds ``ot_disease_name`` (looked up from ``OT_DISEASE_TABLE``, falling
    back to the raw indication if unresolved) and ``ta_i`` = therapy_area +
    " - " + ot_disease_name."""
    existing_disease_map = fetch_existing_mappings(OT_DISEASE_TABLE, "indication")

    def _ot_name(indication: str) -> str:
        entry = existing_disease_map.get((indication or "").strip().lower())
        if entry and entry.get("ot_disease"):
            return entry["ot_disease"]
        return indication or ""

    df["ot_disease_name"] = df["indication"].apply(_ot_name)
    df["ta_i"] = df["therapy_area"].astype(str) + " - " + df["ot_disease_name"].astype(str)
    return df


# ==============================
# WEIGHT CALCULATION
# ==============================
def compute_trial_weights(rows: list[dict], drug_name: str) -> pd.DataFrame:
    """Computes phase_weight/geo_score/sample_score/dosage_score/trial_weight
    for every row. Trial-sourced rows use the full formula; non-trial rows
    get a fixed ``trial_weight = 0.05``."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _build_ta_i_column(df, drug_name)

    is_trial = df["data_source"].astype(str).str.strip().str.lower() == "trials"
    df_trial = df[is_trial].copy().reset_index(drop=True)
    df_other = df[~is_trial].copy().reset_index(drop=True)

    if not df_trial.empty:
        df_trial["phase"] = df_trial["phase"].apply(normalize_phase)
        df_trial["phase_weight"] = df_trial["phase"].apply(compute_phase_weight)
        df_trial["geo_score"] = df_trial["primary_region"].apply(compute_geo_score)
        df_trial["sample_score"] = df_trial["drug_arm_size_n"].apply(compute_sample_score)
        df_trial["dosage_score"] = assign_dosage_scores(df_trial)
        df_trial["trial_weight"] = (
            df_trial["phase_weight"] * df_trial["geo_score"]
            * df_trial["sample_score"] * df_trial["dosage_score"]
        ).round(4)

        approved_mask = df_trial["phase"].apply(lambda p: phase_rank(p) == 4)
        df_trial.loc[approved_mask, "trial_weight"] = 1.0

    for col in ("phase_weight", "geo_score", "sample_score", "dosage_score"):
        df_other[col] = float("nan")
    df_other["trial_weight"] = 0.05

    df_final = pd.concat([df_trial, df_other], ignore_index=True)
    logger.info(
        "[TRIAL_SELECTOR] Computed weights for %d row(s) (%d trial + %d non-trial) for '%s'",
        len(df_final), len(df_trial), len(df_other), drug_name,
    )
    return df_final


# ==============================
# TA-I SUMMARY SELECTION
# ==============================
def select_best_trial_per_tai(df: pd.DataFrame) -> pd.DataFrame:
    """Selects one row per TA-I: the row with the highest ``trial_weight``,
    ties broken by highest phase_rank then first occurrence."""
    if df.empty:
        return df

    tmp = df.copy()
    tmp["_pr"] = tmp["phase"].apply(phase_rank)
    tmp = tmp.sort_values(["ta_i", "trial_weight", "_pr"], ascending=[True, False, False])
    summary = tmp.drop_duplicates(subset=["ta_i"], keep="first").drop(columns=["_pr"])
    summary = summary.reset_index(drop=True)

    logger.info("[TRIAL_SELECTOR] Selected %d TA-I summary row(s)", len(summary))
    return summary


# ==============================
# ENTRY POINT
# ==============================
def select_trials(drug_name: str = DRUG_NAME, secondary_only: bool = False) -> list[dict]:
    """Full trial-selection pipeline for one drug:

    1. Fetch + enrich all LE_TABLE rows (via ``data_fetcher``).
    2. Compute trial_weight and its components for every row.
    3. Select the single best (highest trial_weight) row per TA-I group.

    Args:
        drug_name: the drug/molecule name.
        secondary_only: if ``True``, only processes Secondary indications.

    Returns a list of dicts — one per TA-I — ready for ``score_calculator``.
    """
    rows = fetch_and_enrich_trial_data(drug_name, secondary_only=secondary_only)
    if not rows:
        logger.warning("[TRIAL_SELECTOR] No rows to select from for '%s'", drug_name)
        return []

    df = compute_trial_weights(rows, drug_name)
    summary_df = select_best_trial_per_tai(df)

    summary_df = summary_df.astype(object).where(pd.notnull(summary_df), None)
    return summary_df.to_dict(orient="records")
