"""Label Expansion Score Calculation — Module 1: data_fetcher.

Fetches every LE_TABLE row for a drug, then enriches it with the fields
the scoring model needs:

  ``association_score`` (every row, trial- and web-sourced alike):
      Resolved via the Open Targets API — the target-disease association
      score between this drug's resolved MOA target(s) (``OT_MOA_TABLE``)
      and each indication's resolved OT disease (``OT_DISEASE_TABLE``).

  ``primary_region``, ``drug_arm_size_n``, ``dosage`` (trial-sourced rows only):
      1. Values already present on the LE_TABLE row (``dosage``,
         ``trial_size``, ``trial_location`` — aliased to ``primary_region``).
      2. Anything still missing is looked up by ``trial_id`` in
         ``CLINICAL_TRIALS_SERIOUS_SAFETY_DATA``.
      3. Anything still missing after that is filled via Gemini + Google
         Search grounding.
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
from ..ot_mapping.moa_mapping import fetch_moa_for_drug
from ..ot_mapping.ot_utils import OT_DISEASE_TABLE, OT_MOA_TABLE, fetch_existing_mappings, ot_post

logger = logging.getLogger(__name__)

# ==============================
# CONSTANTS
# ==============================
TRIAL_ENRICHMENT_FIELDS = ("primary_region", "drug_arm_size_n", "dosage")
GEMINI_TRIALS_PER_CALL = 5
OT_DISEASE_PAGE_SIZE = 50


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
# STEP 2: ASSOCIATION SCORE FROM OPEN TARGETS
# ==============================
def fetch_target_ensembl_ids(drug_name: str, drug_details_table: str = "drug_details") -> list[str]:
    """Resolves this drug's Ensembl target ID(s) by looking up its MOA(s)
    (fetched fresh from the drug_details table) against ``OT_MOA_TABLE``,
    the mapping table ``moa_mapping`` already populated in Step 4 of the
    main pipeline. Returns an empty list if nothing resolves."""
    try:
        moas = fetch_moa_for_drug(drug_name, drug_details_table)
    except Exception as exc:
        logger.warning("[DATA_FETCHER] Could not fetch MOA(s) for '%s': %s", drug_name, exc)
        return []

    if not moas:
        return []

    moa_map = fetch_existing_mappings(OT_MOA_TABLE, "moa")
    ensembl_ids = []
    for moa in moas:
        entry = moa_map.get(moa.strip().lower())
        if entry and entry.get("ensembl_id"):
            ensembl_ids.append(entry["ensembl_id"])

    ensembl_ids = sorted(set(ensembl_ids))
    logger.info(
        "[DATA_FETCHER] Resolved %d target Ensembl ID(s) for '%s' from %s: %s",
        len(ensembl_ids), drug_name, OT_MOA_TABLE, ensembl_ids,
    )
    return ensembl_ids


def fetch_ot_association_scores(target_ensembl_ids: list[str]) -> dict[str, float]:
    """Queries the Open Targets Platform GraphQL API for every disease
    associated with the given target(s), returning ``{disease_id: score}``
    (the OT target-disease association score, 0-1). When a disease is
    associated with more than one target, the highest score is kept."""
    if not target_ensembl_ids:
        return {}

    query = """
    query TargetDiseaseScores($targetId: String!, $index: Int!, $size: Int!) {
      target(ensemblId: $targetId) {
        associatedDiseases(page: { index: $index, size: $size }) {
          count
          rows { disease { id } score }
        }
      }
    }
    """
    scores: dict[str, float] = {}
    for tid in target_ensembl_ids:
        page_index = 0
        total = None
        fetched = 0
        while True:
            data = ot_post(
                query,
                {"targetId": tid, "index": page_index, "size": OT_DISEASE_PAGE_SIZE},
                context=f"association-score:{tid}:p{page_index}",
            )
            if not data:
                break
            assoc = data.get("target", {}).get("associatedDiseases", {})
            if total is None:
                total = assoc.get("count", 0)
            rows = assoc.get("rows", [])
            if not rows:
                break
            for row in rows:
                disease_id = row.get("disease", {}).get("id")
                score = row.get("score")
                if disease_id is not None and score is not None:
                    scores[disease_id] = max(score, scores.get(disease_id, 0.0))
            fetched += len(rows)
            if fetched >= (total or 0):
                break
            page_index += 1

    logger.info(
        "[DATA_FETCHER] Fetched %d target-disease association score(s) from Open Targets for %d target(s)",
        len(scores), len(target_ensembl_ids),
    )
    return scores


def apply_association_scores(rows: list[dict], drug_name: str, drug_details_table: str = "drug_details") -> None:
    """Fills ``association_score`` on every row (trial- and web-sourced
    alike, since it's an indication-level target-disease score, not a
    trial-level one) using the Open Targets API. Mutates ``rows`` in place."""
    target_ids = fetch_target_ensembl_ids(drug_name, drug_details_table)
    if not target_ids:
        logger.warning(
            "[DATA_FETCHER] No target Ensembl ID(s) resolved for '%s' - "
            "association_score will be unavailable for all rows. "
            "Ensure Step 4 (moa_mapping) has run for this drug first.",
            drug_name,
        )
        for r in rows:
            r["association_score"] = None
        return

    disease_scores = fetch_ot_association_scores(target_ids)
    disease_map = fetch_existing_mappings(OT_DISEASE_TABLE, "indication")

    resolved = 0
    for r in rows:
        indication_key = (r.get("indication") or "").strip().lower()
        disease_entry = disease_map.get(indication_key)
        disease_id = disease_entry.get("ot_disease_id") if disease_entry else None
        score = disease_scores.get(disease_id) if disease_id else None
        r["association_score"] = score
        if score is not None:
            resolved += 1

    if not resolved:
        logger.warning(
            "[DATA_FETCHER] association_score could not be resolved for any row of '%s' - "
            "check that Step 5 (indication_mapping) has populated %s with ot_disease_id values.",
            drug_name, OT_DISEASE_TABLE,
        )
    else:
        logger.info(
            "[DATA_FETCHER] association_score resolved for %d/%d row(s) via Open Targets",
            resolved, len(rows),
        )


# ==============================
# STEP 3: BQ ENRICHMENT FROM CLINICAL_TRIALS_SERIOUS_SAFETY_DATA
# ==============================
def fetch_serious_safety_data(trial_ids: list[str]) -> dict[str, dict]:
    """Looks up trial-level enrichment fields for the given trial IDs from
    ``CLINICAL_TRIALS_SERIOUS_SAFETY_DATA``.

    Returns ``{normalized_trial_id: {primary_region, drug_arm_size_n,
    dosage}}``. Missing table/columns degrade gracefully to an empty dict
    rather than raising.
    """
    if not trial_ids:
        return {}

    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_SERIOUS_SAFETY_DATA}"

    query = f"""
        SELECT trial_id, primary_region, drug_arm_size_n, dosage
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
            }

    logger.info(
        "[DATA_FETCHER] %s: matched %d/%d trial(s)",
        CLINICAL_TRIALS_SERIOUS_SAFETY_DATA, len(enrichment), len(wanted),
    )
    return enrichment


# ==============================
# STEP 4: GEMINI FALLBACK
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
def fetch_and_enrich_trial_data(drug_name: str = DRUG_NAME, drug_details_table: str = "drug_details") -> list[dict]:
    """Fetches all LE_TABLE rows for a drug and enriches them for scoring.

    ``association_score`` (Step 2) is resolved for every row — trial- and
    web-sourced alike, since it's an indication-level target-disease score —
    via the Open Targets API, using this drug's already-resolved MOA
    target(s) (from ``OT_MOA_TABLE``) and each indication's resolved OT
    disease (from ``OT_DISEASE_TABLE``).

    Trial-level fields (``primary_region``, ``drug_arm_size_n``, ``dosage``)
    are enriched for trial-sourced rows only, with this priority:
        1. Already present on the LE_TABLE row (dosage / trial_size /
           trial_location, the latter aliased to primary_region).
        2. ``CLINICAL_TRIALS_SERIOUS_SAFETY_DATA`` lookup by trial_id.
        3. Gemini + Google Search fallback.

    Returns a flat list of dicts, each with the original LE_TABLE fields
    plus ``primary_region``, ``drug_arm_size_n``, ``dosage``, and
    ``association_score``.
    """
    rows = fetch_le_rows(drug_name)
    if not rows:
        logger.warning("[DATA_FETCHER] No %s rows found for '%s'", LE_TABLE, drug_name)
        return []

    # Step 2: association_score for every row, via Open Targets.
    apply_association_scores(rows, drug_name, drug_details_table)

    trial_rows = [r for r in rows if (r.get("data_source") or "").strip().lower() == "trials"]
    non_trial_rows = [r for r in rows if (r.get("data_source") or "").strip().lower() != "trials"]

    # Seed trial-level fields from what's already on the LE_TABLE row.
    for r in trial_rows:
        r["primary_region"] = r.get("trial_location")
        r["drug_arm_size_n"] = r.get("trial_size")

    def _needs_fill(r: dict) -> bool:
        return any(_is_missing(r.get(f)) for f in TRIAL_ENRICHMENT_FIELDS)

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

        # Step 3: CLINICAL_TRIALS_SERIOUS_SAFETY_DATA
        bq_enrichment = fetch_serious_safety_data(trials_needing_fill)
        for r in trial_rows:
            norm_id = _normalize_trial_id(r.get("trial_id"))
            entry = bq_enrichment.get(norm_id)
            if not entry:
                continue
            for field in TRIAL_ENRICHMENT_FIELDS:
                if _is_missing(r.get(field)) and not _is_missing(entry.get(field)):
                    r[field] = entry[field]

        # Step 4: Gemini fallback for whatever's still missing
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
                for field in TRIAL_ENRICHMENT_FIELDS:
                    if _is_missing(r.get(field)) and not _is_missing(entry.get(field)):
                        r[field] = entry[field]
    else:
        logger.info("[DATA_FETCHER] All trial rows already fully populated - no trial-level enrichment needed")

    for r in non_trial_rows:
        r.setdefault("primary_region", None)
        r.setdefault("drug_arm_size_n", None)

    all_rows = trial_rows + non_trial_rows
    logger.info("[DATA_FETCHER] Completed. %d row(s) ready for scoring for '%s'", len(all_rows), drug_name)
    return all_rows
