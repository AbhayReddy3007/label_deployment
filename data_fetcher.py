"""Label Expansion Score Calculation — Module 1: data_fetcher.

Fetches every LE_TABLE row for a drug, then enriches the trial-sourced
rows with the fields the scoring model needs (``primary_region``,
``drug_arm_size_n`` / ``size``, ``dosage``, ``association_score``):

  1. Values already present on the LE_TABLE row (``dosage``, ``trial_size``,
     ``trial_location`` — aliased to ``primary_region``) are used first.
  2. Anything still missing is looked up by ``trial_id`` in
     ``CLINICAL_TRIALS_SERIOUS_SAFETY_DATA``.
  3. Anything still missing after that is filled via Gemini + Google
     Search grounding.

Non-trial (``data_source == "Web"``) rows are returned as-is — they have
no trial to enrich against.
"""

from __future__ import annotations

import logging

from google.cloud import bigquery

from medical_potential.config import (
    BQ_DATASET_ID,
    CLINICAL_TRIALS_SERIOUS_SAFETY_DATA,
    DRUG_NAME,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client

from ..bq_utils import LE_TABLE
from ..indication_extractor.utils import extract_json, gemini_generate

logger = logging.getLogger(__name__)

# ==============================
# CONSTANTS
# ==============================
ENRICHMENT_FIELDS = ("primary_region", "drug_arm_size_n", "dosage", "association_score")
GEMINI_TRIALS_PER_CALL = 5


# ==============================
# HELPERS
# ==============================
def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "nan", "none", "n/a"):
        return True
    return False


def _normalize_trial_id(trial_id) -> str:
    import re
    s = str(trial_id or "").strip()
    paren_idx = s.find("(")
    if paren_idx != -1:
        s = s[:paren_idx].strip()
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


# ==============================
# STEP 1: FETCH LE_TABLE ROWS
# ==============================
def fetch_le_rows(drug_name: str = DRUG_NAME) -> list[dict]:
    """Fetches every ``LE_TABLE`` row for this drug (trial- and web-sourced)."""
    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LE_TABLE}"

    query = f"""
        SELECT drug_name, indication, indication_type, therapy_area, rationale,
               trial_id, trial_title, phase, dosage, trial_size, trial_location,
               source_url, data_source
        FROM `{table_id}`
        WHERE LOWER(drug_name) = LOWER(@drug_name)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name)]
    )
    results = bq_client.query(query, job_config=job_config).result()
    rows = [dict(row) for row in results]
    logger.info("[DATA_FETCHER] Fetched %d %s row(s) for '%s'", len(rows), LE_TABLE, drug_name)
    return rows


# ==============================
# STEP 2: BQ ENRICHMENT FROM CLINICAL_TRIALS_SERIOUS_SAFETY_DATA
# ==============================
def fetch_serious_safety_data(trial_ids: list[str]) -> dict[str, dict]:
    """Looks up enrichment fields for the given trial IDs from
    ``CLINICAL_TRIALS_SERIOUS_SAFETY_DATA``.

    Returns ``{normalized_trial_id: {primary_region, drug_arm_size_n, dosage,
    association_score}}``. Missing table/columns degrade gracefully to an
    empty dict rather than raising.
    """
    if not trial_ids:
        return {}

    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_SERIOUS_SAFETY_DATA}"

    query = f"""
        SELECT trial_id, primary_region, drug_arm_size_n, dosage, association_score
        FROM `{table_id}`
        WHERE trial_id IS NOT NULL
    """
    try:
        results = bq_client.query(query).result()
    except Exception as exc:
        logger.warning(
            "[DATA_FETCHER] Could not query %s (table/columns may not exist): %s",
            table_id, exc,
        )
        return {}

    wanted = {_normalize_trial_id(t) for t in trial_ids}
    enrichment: dict[str, dict] = {}
    for row in results:
        norm_id = _normalize_trial_id(row.get("trial_id"))
        if norm_id in wanted:
            enrichment[norm_id] = {
                "primary_region": row.get("primary_region"),
                "drug_arm_size_n": row.get("drug_arm_size_n"),
                "dosage": row.get("dosage"),
                "association_score": row.get("association_score"),
            }

    logger.info(
        "[DATA_FETCHER] %s: matched %d/%d trial(s)",
        CLINICAL_TRIALS_SERIOUS_SAFETY_DATA, len(enrichment), len(wanted),
    )
    return enrichment


# ==============================
# STEP 3: GEMINI FALLBACK
# ==============================
def _build_gemini_prompt(trial_ids: list[str]) -> str:
    trials_list = "\n".join(f"- {t}" for t in trial_ids)
    return f"""You are a clinical trial data expert. Search for each trial ID below.

Trial IDs:
{trials_list}

For each trial find:
- primary_region: Primary geographic region of trial sites (e.g. "United States", "Europe").
- drug_arm_size_n: Number of patients in the drug/treatment arm(s) only
  (use total enrollment if arm-level data is unavailable).
- dosage: Dose/regimen of the investigational drug arm.

Return ONLY a JSON array, no explanation, no markdown fences:
[
  {{"trial_id": "NCT12345678", "primary_region": "United States", "drug_arm_size_n": 250, "dosage": "10 mg once daily"}}
]
Use null only if genuinely not findable. Do not guess.
"""


def gemini_fill_missing(trial_ids: list[str], batch_size: int = GEMINI_TRIALS_PER_CALL) -> dict[str, dict]:
    """Uses Gemini + Google Search grounding to fill in enrichment fields
    for trials that BQ couldn't resolve. Returns ``{normalized_trial_id: {...}}``."""
    if not trial_ids:
        return {}

    logger.info("[DATA_FETCHER] Gemini fallback for %d trial(s)", len(trial_ids))
    batches = [trial_ids[i : i + batch_size] for i in range(0, len(trial_ids), batch_size)]
    results: dict[str, dict] = {}

    for i, batch in enumerate(batches, 1):
        try:
            raw = gemini_generate(
                _build_gemini_prompt(batch),
                system_instruction=(
                    "You are a clinical trial data assistant. Search for each trial to find "
                    "its geographic region, drug-arm sample size, and dosage. Return ONLY valid JSON."
                ),
                use_search=True,
            )
            parsed = extract_json(raw)
            entries = parsed if isinstance(parsed, list) else parsed.get("trials", []) if isinstance(parsed, dict) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                tid = _normalize_trial_id(entry.get("trial_id"))
                if tid:
                    results[tid] = {
                        "primary_region": entry.get("primary_region"),
                        "drug_arm_size_n": entry.get("drug_arm_size_n"),
                        "dosage": entry.get("dosage"),
                    }
        except Exception as exc:
            logger.warning("[DATA_FETCHER] Gemini fallback failed for batch %d/%d: %s", i, len(batches), exc)

    logger.info("[DATA_FETCHER] Gemini resolved %d/%d trial(s)", len(results), len(trial_ids))
    return results


# ==============================
# ENTRY POINT
# ==============================
def fetch_and_enrich_trial_data(drug_name: str = DRUG_NAME) -> list[dict]:
    """Fetches all LE_TABLE rows for a drug and enriches trial-sourced rows
    with ``primary_region``, ``drug_arm_size_n``, ``dosage``, and
    ``association_score``.

    Enrichment priority per field:
        1. Already present on the LE_TABLE row (dosage / trial_size /
           trial_location, the latter aliased to primary_region).
        2. ``CLINICAL_TRIALS_SERIOUS_SAFETY_DATA`` lookup by trial_id.
        3. Gemini + Google Search fallback.

    Returns a flat list of dicts, each with the original LE_TABLE fields
    plus ``primary_region``, ``drug_arm_size_n``, ``dosage`` (overwritten
    with the enriched value), and ``association_score``.
    """
    rows = fetch_le_rows(drug_name)
    if not rows:
        logger.warning("[DATA_FETCHER] No %s rows found for '%s'", LE_TABLE, drug_name)
        return []

    trial_rows = [r for r in rows if (r.get("data_source") or "").strip().lower() == "trials"]
    non_trial_rows = [r for r in rows if (r.get("data_source") or "").strip().lower() != "trials"]

    # Seed enrichment fields from what's already on the LE_TABLE row.
    for r in trial_rows:
        r["primary_region"] = r.get("trial_location")
        r["drug_arm_size_n"] = r.get("trial_size")
        r["association_score"] = None  # not present on LE_TABLE; filled below

    def _needs_fill(r: dict) -> bool:
        return any(_is_missing(r.get(f)) for f in ENRICHMENT_FIELDS)

    trials_needing_fill = sorted({
        _normalize_trial_id(r["trial_id"])
        for r in trial_rows
        if r.get("trial_id") and _needs_fill(r)
    })

    if trials_needing_fill:
        logger.info(
            "[DATA_FETCHER] %d/%d trial row(s) need enrichment: %s",
            len(trials_needing_fill), len(trial_rows), trials_needing_fill,
        )

        # Step 2: CLINICAL_TRIALS_SERIOUS_SAFETY_DATA
        bq_enrichment = fetch_serious_safety_data(trials_needing_fill)
        for r in trial_rows:
            norm_id = _normalize_trial_id(r.get("trial_id"))
            entry = bq_enrichment.get(norm_id)
            if not entry:
                continue
            for field in ENRICHMENT_FIELDS:
                if _is_missing(r.get(field)) and not _is_missing(entry.get(field)):
                    r[field] = entry[field]

        # Step 3: Gemini fallback for whatever's still missing
        still_missing = sorted({
            _normalize_trial_id(r["trial_id"])
            for r in trial_rows
            if r.get("trial_id") and _needs_fill(r)
        })
        if still_missing:
            gemini_enrichment = gemini_fill_missing(still_missing)
            for r in trial_rows:
                norm_id = _normalize_trial_id(r.get("trial_id"))
                entry = gemini_enrichment.get(norm_id)
                if not entry:
                    continue
                for field in ("primary_region", "drug_arm_size_n", "dosage"):
                    if _is_missing(r.get(field)) and not _is_missing(entry.get(field)):
                        r[field] = entry[field]
    else:
        logger.info("[DATA_FETCHER] All trial rows already fully populated - no enrichment needed")

    for r in non_trial_rows:
        r.setdefault("primary_region", None)
        r.setdefault("drug_arm_size_n", None)
        r.setdefault("association_score", None)

    all_rows = trial_rows + non_trial_rows
    logger.info("[DATA_FETCHER] Completed. %d row(s) ready for scoring for '%s'", len(all_rows), drug_name)
    return all_rows
