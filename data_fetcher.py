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
      2. Anything still missing is looked up in ``DATA_FETCHER_TABLE`` —
         a cache of everything this module has already fetched/searched
         for on a previous run, keyed by ``trial_id``. A cache hit means
         no lookup or search is repeated for that trial.
      3. Anything still missing is looked up by ``trial_id`` in
         ``CLINICAL_TRIALS_SERIOUS_SAFETY_DATA``.
      4. Anything still missing after that is filled via Gemini + Google
         Search grounding.
      Whatever gets resolved in steps 3-4 is written back into
      ``DATA_FETCHER_TABLE`` so future runs hit the cache instead.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from google.cloud import bigquery

from medical_potential.config import (
    BQ_DATASET_ID,
    CLINICAL_TRIALS_SERIOUS_SAFETY_DATA,
    DRUG_NAME,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client

from ..bq_utils import LE_TABLE
from ..indication_extractor.utils import extract_json, gemini_generate_with_timeout
from ..ot_mapping.moa_mapping import fetch_moa_for_drug
from ..ot_mapping.ot_utils import OT_DISEASE_TABLE, OT_MOA_TABLE, fetch_existing_mappings, ot_post

logger = logging.getLogger(__name__)

# ==============================
# TABLE NAMES
# ==============================
DATA_FETCHER_TABLE = "data_fetched_le"

# ==============================
# CONSTANTS
# ==============================
TRIAL_ENRICHMENT_FIELDS = ("primary_region", "drug_arm_size_n", "dosage")
GEMINI_TRIALS_PER_CALL = 5
MAX_WORKERS = 10
GEMINI_FILL_TIMEOUT_SECONDS = 90  # scaled by batch size, same as trial_analyser
GEMINI_FILL_MAX_ATTEMPTS = 2  # retry once on timeout before giving up on a batch
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
def fetch_le_rows(drug_name: str = DRUG_NAME, secondary_only: bool = False) -> list[dict]:
    """Fetches ``LE_TABLE`` rows for this drug (trial- and web-sourced).

    Args:
        drug_name: the drug/molecule name.
        secondary_only: if ``True``, only fetches rows where
            ``indication_type = 'Secondary'``. Used for scoring, which
            should only run on label-expansion candidates.
    """
    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LE_TABLE}"

    secondary_filter = "AND LOWER(indication_type) = 'secondary'" if secondary_only else ""

    query = f"""
        SELECT drug_name, indication, indication_type, therapy_area, rationale,
               trial_id, trial_title, phase, dosage, trial_size, trial_location,
               source_url, data_source
        FROM `{table_id}`
        WHERE LOWER(drug_name) = LOWER(@drug_name)
          AND indication IS NOT NULL
          AND indication != 'Unknown (extraction failed)'
          {secondary_filter}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name)]
    )
    results = bq_client.query(query, job_config=job_config).result()
    rows = [dict(row) for row in results]
    label = "Secondary" if secondary_only else "all"
    logger.info("[DATA_FETCHER] Fetched %d %s %s row(s) for '%s'", len(rows), label, LE_TABLE, drug_name)
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
# STEP 2.5: DATA_FETCHER_TABLE CACHE (avoid re-fetching/re-searching)
# ==============================
DATA_FETCHER_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("trial_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("primary_region", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("drug_arm_size_n", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("dosage", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
]


def _ensure_data_fetcher_table_exists(bq_client: bigquery.Client, table_id: str) -> None:
    """Creates ``DATA_FETCHER_TABLE`` if missing, and patches in any columns
    from ``DATA_FETCHER_SCHEMA`` that an already-existing table lacks."""
    table = bigquery.Table(table_id, schema=DATA_FETCHER_SCHEMA)
    table = bq_client.create_table(table, exists_ok=True)

    existing_field_names = {f.name for f in table.schema}
    missing_fields = [f for f in DATA_FETCHER_SCHEMA if f.name not in existing_field_names]
    if missing_fields:
        logger.info(
            "[DATA_FETCHER] Table %s is missing column(s) %s - adding them now.",
            table_id, ", ".join(f.name for f in missing_fields),
        )
        table.schema = list(table.schema) + missing_fields
        bq_client.update_table(table, ["schema"])


def fetch_cached_enrichment(trial_ids: list[str]) -> dict[str, dict]:
    """Looks up trial-level enrichment fields already fetched on a
    previous run, from ``DATA_FETCHER_TABLE``. Returns
    ``{normalized_trial_id: {primary_region, drug_arm_size_n, dosage}}``.

    Any trial found here does not need to be looked up again in
    ``CLINICAL_TRIALS_SERIOUS_SAFETY_DATA`` or re-searched via Gemini.
    """
    if not trial_ids:
        return {}

    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{DATA_FETCHER_TABLE}"

    query = f"""
        SELECT trial_id, primary_region, drug_arm_size_n, dosage
        FROM `{table_id}`
        WHERE trial_id IN UNNEST(@trial_ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("trial_ids", "STRING", trial_ids)]
    )
    try:
        results = bq_client.query(query, job_config=job_config).result()
    except Exception:
        logger.info(
            "[DATA_FETCHER] %s does not exist yet or has no rows - treating all trials as new",
            DATA_FETCHER_TABLE,
        )
        return {}

    cached = {
        row["trial_id"]: {
            "primary_region": row.get("primary_region"),
            "drug_arm_size_n": row.get("drug_arm_size_n"),
            "dosage": row.get("dosage"),
        }
        for row in results
        if row.get("trial_id")
    }
    logger.info(
        "[DATA_FETCHER] %s: found %d/%d trial(s) already cached - skipping re-fetch for those",
        DATA_FETCHER_TABLE, len(cached), len(trial_ids),
    )
    return cached


def _as_str(value):
    """Coerces a value to str for a BigQuery STRING parameter, preserving
    None (BigQuery accepts None for a NULLABLE parameter)."""
    if value is None:
        return None
    return str(value)


def save_enrichment_cache(enrichment: dict[str, dict]) -> None:
    """Upserts newly-fetched trial-level enrichment fields into
    ``DATA_FETCHER_TABLE`` (keyed on ``trial_id``), so a future run can
    reuse them instead of looking them up or searching for them again."""
    if not enrichment:
        return

    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{DATA_FETCHER_TABLE}"
    _ensure_data_fetcher_table_exists(bq_client, table_id)

    now = datetime.now(timezone.utc).isoformat()
    struct_params = [
        bigquery.StructQueryParameter(
            None,
            bigquery.ScalarQueryParameter("trial_id", "STRING", tid),
            bigquery.ScalarQueryParameter("primary_region", "STRING", _as_str(entry.get("primary_region"))),
            bigquery.ScalarQueryParameter("drug_arm_size_n", "STRING", _as_str(entry.get("drug_arm_size_n"))),
            bigquery.ScalarQueryParameter("dosage", "STRING", _as_str(entry.get("dosage"))),
            bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", now),
        )
        for tid, entry in enrichment.items()
        if tid
    ]
    if not struct_params:
        return

    merge_query = f"""
        MERGE `{table_id}` T
        USING (SELECT * FROM UNNEST(@rows)) S
        ON T.trial_id = S.trial_id
        WHEN MATCHED THEN
            UPDATE SET
                primary_region = S.primary_region,
                drug_arm_size_n = S.drug_arm_size_n,
                dosage = S.dosage,
                updated_at = S.updated_at
        WHEN NOT MATCHED THEN
            INSERT (trial_id, primary_region, drug_arm_size_n, dosage, created_at, updated_at)
            VALUES (S.trial_id, S.primary_region, S.drug_arm_size_n, S.dosage, S.updated_at, S.updated_at)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("rows", "STRUCT", struct_params)]
    )
    bq_client.query(merge_query, job_config=job_config).result()
    logger.info(
        "[DATA_FETCHER] Cached enrichment for %d trial(s) into %s for future runs",
        len(struct_params), DATA_FETCHER_TABLE,
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


def _fill_missing_batch(batch: list[str]) -> dict[str, dict]:
    """Runs one enrichment batch through Gemini with a hard timeout,
    retrying on timeout before giving up on this batch."""
    # Larger batches legitimately need more time, so the timeout scales
    # with batch size (same approach as trial_analyser).
    timeout = GEMINI_FILL_TIMEOUT_SECONDS * len(batch)

    try:
        raw = gemini_generate_with_timeout(
            _build_gemini_prompt(batch),
            system_instruction=(
                "You are a clinical trial data assistant. Search for each trial to find "
                "its geographic region, drug-arm sample size, and dosage. Return ONLY valid JSON."
            ),
            use_search=True,
            timeout_seconds=timeout,
            max_attempts=GEMINI_FILL_MAX_ATTEMPTS,
            log_context=f"enrichment batch {batch}",
        )
    except Exception as exc:  # noqa: BLE001 - includes TimeoutError
        logger.warning("[DATA_FETCHER] Gemini fallback failed for batch %s: %s", batch, exc)
        return {}

    batch_results: dict[str, dict] = {}
    parsed = extract_json(raw)
    entries = parsed if isinstance(parsed, list) else parsed.get("trials", []) if isinstance(parsed, dict) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tid = _normalize_trial_id(entry.get("trial_id"))
        if tid:
            batch_results[tid] = {
                "primary_region": entry.get("primary_region"),
                "drug_arm_size_n": entry.get("drug_arm_size_n"),
                "dosage": entry.get("dosage"),
            }
    return batch_results


def gemini_fill_missing(trial_ids: list[str], batch_size: int = GEMINI_TRIALS_PER_CALL) -> dict[str, dict]:
    """Uses Gemini + Google Search grounding to fill in enrichment fields
    for trials that BQ couldn't resolve. Returns ``{normalized_trial_id: {...}}``.

    Batches run in parallel (up to ``MAX_WORKERS`` at a time), each with its
    own timeout so one slow/hung batch can't block the rest."""
    if not trial_ids:
        return {}

    logger.info(
        "[DATA_FETCHER] Gemini fallback for %d trial(s) in batches of %d, using %d workers",
        len(trial_ids), batch_size, MAX_WORKERS,
    )
    batches = [trial_ids[i : i + batch_size] for i in range(0, len(trial_ids), batch_size)]
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fill_missing_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                results.update(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("[DATA_FETCHER] Unexpected error for batch %s: %s", batch, exc)

    logger.info("[DATA_FETCHER] Gemini resolved %d/%d trial(s)", len(results), len(trial_ids))
    return results


# ==============================
# ENTRY POINT
# ==============================
def fetch_and_enrich_trial_data(drug_name: str = DRUG_NAME, drug_details_table: str = "drug_details", secondary_only: bool = False) -> list[dict]:
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
    rows = fetch_le_rows(drug_name, secondary_only=secondary_only)
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

        # Step 2.5: DATA_FETCHER_TABLE cache - reuse anything already fetched
        # on a previous run instead of looking it up or searching again.
        cached_enrichment = fetch_cached_enrichment(trials_needing_fill)
        for r in trial_rows:
            norm_id = _normalize_trial_id(r.get("trial_id"))
            entry = cached_enrichment.get(norm_id)
            if not entry:
                continue
            for field in TRIAL_ENRICHMENT_FIELDS:
                if _is_missing(r.get(field)) and not _is_missing(entry.get(field)):
                    r[field] = entry[field]

        newly_fetched: dict[str, dict] = {}

        # Step 3: CLINICAL_TRIALS_SERIOUS_SAFETY_DATA - only for trials not
        # already resolved by the cache above.
        still_needs_bq = sorted({
            _normalize_trial_id(r["trial_id"])
            for r in trial_rows
            if r.get("trial_id") and _needs_fill(r)
        })
        if still_needs_bq:
            bq_enrichment = fetch_serious_safety_data(still_needs_bq)
            for r in trial_rows:
                norm_id = _normalize_trial_id(r.get("trial_id"))
                entry = bq_enrichment.get(norm_id)
                if not entry:
                    continue
                for field in TRIAL_ENRICHMENT_FIELDS:
                    if _is_missing(r.get(field)) and not _is_missing(entry.get(field)):
                        r[field] = entry[field]
                if norm_id in still_needs_bq:
                    newly_fetched.setdefault(norm_id, {}).update(entry)

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
                if norm_id in still_missing:
                    newly_fetched.setdefault(norm_id, {}).update(entry)

        # Save whatever was newly resolved this run (BQ safety-data lookup
        # or Gemini) into DATA_FETCHER_TABLE, so a future run can reuse it
        # directly instead of fetching/searching for it again.
        save_enrichment_cache(newly_fetched)
    else:
        logger.info("[DATA_FETCHER] All trial rows already fully populated - no trial-level enrichment needed")

    for r in non_trial_rows:
        r.setdefault("primary_region", None)
        r.setdefault("drug_arm_size_n", None)

    all_rows = trial_rows + non_trial_rows
    logger.info("[DATA_FETCHER] Completed. %d row(s) ready for scoring for '%s'", len(all_rows), drug_name)
    return all_rows
