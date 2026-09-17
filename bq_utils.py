"""Label Expansion Opportunity — BigQuery utilities: merge module results and push to BigQuery.

Merges the rows produced by ``trial_analyser`` (Module 1) and
``web_analyser`` (Module 2) and upserts the merged, de-duplicated
rows into the configured BigQuery table.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone

from google.cloud import bigquery

from medical_potential.config import BQ_DATASET_ID, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client

logger = logging.getLogger(__name__)

# ==============================
# TABLE NAMES
# ==============================
LE_TABLE = "label_expansion_opportunity_results"
LE_SCORE_CALCULATION_TABLE = "label_expansion_score_calculation"


# ==============================
# FETCH EXISTING TRIAL IDS (for incremental runs)
# ==============================
def fetch_existing_trial_ids(drug_name: str) -> set[str]:
    """Returns the set of ``trial_id`` values already present in ``LE_TABLE``
    for this drug (normalized: stripped of whitespace/punctuation, upper-cased,
    parenthetical annotations removed). Used so a re-run only processes trials
    that haven't been extracted before.

    Returns an empty set if the table doesn't exist yet or has no rows for
    this drug.
    """
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LE_TABLE}"
    bq_client = get_bq_client()

    query = f"""
        SELECT DISTINCT trial_id
        FROM `{table_id}`
        WHERE LOWER(drug_name) = LOWER(@drug_name)
          AND trial_id IS NOT NULL
          AND trial_id != ''
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name)]
    )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        normalized_ids = {_normalize_trial_id(row["trial_id"]) for row in results if row["trial_id"]}
        logger.info(
            "[LE_BQ] Found %d existing trial_id(s) already in %s for '%s'",
            len(normalized_ids), LE_TABLE, drug_name,
        )
        return normalized_ids
    except Exception:
        logger.info(
            "[LE_BQ] %s does not exist yet or has no rows for '%s' — treating all trials as new",
            LE_TABLE, drug_name,
        )
        return set()


def _normalize_trial_id(trial_id) -> str:
    """Strips whitespace/punctuation, a parenthetical annotation, and
    upper-cases a trial_id so it can be compared reliably regardless of
    minor formatting differences, e.g. 'nct06929156 (BGM0504-305)' and
    'NCT06929156' both normalize to 'NCT06929156'."""
    s = str(trial_id or "").strip()
    paren_idx = s.find("(")
    if paren_idx != -1:
        s = s[:paren_idx].strip()
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


# ==============================
# MERGE
# ==============================
def merge_results(trial_rows: list[dict], web_rows: list[dict]) -> list[dict]:
    """Merges Module 1 (trial) and Module 2 (web) rows.

    De-duplicates on ``(drug_name, indication, trial_id)`` — every
    trial/indication pair is kept as a separate row.  When the same
    ``(indication, trial_id)`` appears from both modules, the
    trial-sourced row is preferred (it carries phase/trial_title).
    Web-only rows (``trial_id`` is None) are de-duplicated on
    indication alone so the same web-sourced indication isn't repeated.
    """
    merged: dict[tuple, dict] = {}

    for row in trial_rows:
        key = (
            (row.get("drug_name") or "").strip().lower(),
            row["indication"].strip().lower(),
            (row.get("trial_id") or ""),
        )
        merged[key] = dict(row)

    for row in web_rows:
        # Web rows have no trial_id — key on indication alone (with a
        # sentinel) so we keep one web row per indication at most.
        key = (
            (row.get("drug_name") or "").strip().lower(),
            row["indication"].strip().lower(),
            "__web__",
        )
        if key in merged:
            existing = merged[key]
            if not existing.get("source_url") and row.get("source_url"):
                existing["source_url"] = row["source_url"]
        else:
            # Also skip if a trial row already covers this indication
            # (any trial_id) — trial evidence is stronger.
            indication_key = row["indication"].strip().lower()
            has_trial_row = any(
                k[1] == indication_key and k[2] != "__web__"
                for k in merged
            )
            if not has_trial_row:
                merged[key] = dict(row)

    merged_rows = list(merged.values())
    logger.info(
        "[LE_MERGE] Merged %d trial row(s) + %d web row(s) -> %d row(s) "
        "(unique on drug_name + indication + trial_id)",
        len(trial_rows),
        len(web_rows),
        len(merged_rows),
    )
    return merged_rows


# ==============================
# BIGQUERY SCHEMA + PUSH
# ==============================
LE_RESULTS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("drug_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("indication", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("llm_ot_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("indication_type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("therapy_area", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("rationale", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("trial_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("trial_title", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("phase", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("dosage", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("trial_size", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("trial_location", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("source_url", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("data_source", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
]


def _ensure_table_exists(bq_client: bigquery.Client, table_id: str) -> None:
    """Creates the LE results table if missing, and patches in any columns
    from ``LE_RESULTS_SCHEMA`` that an already-existing table lacks."""
    table = bigquery.Table(table_id, schema=LE_RESULTS_SCHEMA)
    table = bq_client.create_table(table, exists_ok=True)

    existing_field_names = {f.name for f in table.schema}
    missing_fields = [f for f in LE_RESULTS_SCHEMA if f.name not in existing_field_names]
    if missing_fields:
        logger.info(
            "[LE_PUSH] Table %s is missing column(s) %s - adding them now.",
            table_id,
            ", ".join(f.name for f in missing_fields),
        )
        table.schema = list(table.schema) + missing_fields
        bq_client.update_table(table, ["schema"])


def push_to_bigquery(rows: list[dict]) -> None:
    """Upserts merged Label Expansion Opportunity rows into BigQuery.

    ``(drug_name, indication, trial_id)`` is the unique key: each
    trial/indication pair is a separate row. Re-running the pipeline
    updates existing rows instead of appending duplicates.
    """
    if not rows:
        logger.info("[LE_PUSH] No rows to push - skipping.")
        return

    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LE_TABLE}"
    now = datetime.now(timezone.utc).isoformat()

    bq_client = get_bq_client()
    _ensure_table_exists(bq_client, table_id)

    struct_params = [
        bigquery.StructQueryParameter(
            None,
            bigquery.ScalarQueryParameter("drug_name", "STRING", r.get("drug_name")),
            bigquery.ScalarQueryParameter("indication", "STRING", r.get("indication")),
            bigquery.ScalarQueryParameter("llm_ot_name", "STRING", r.get("llm_ot_name")),
            bigquery.ScalarQueryParameter("indication_type", "STRING", r.get("indication_type")),
            bigquery.ScalarQueryParameter("therapy_area", "STRING", r.get("therapy_area")),
            bigquery.ScalarQueryParameter("rationale", "STRING", r.get("rationale")),
            bigquery.ScalarQueryParameter("trial_id", "STRING", r.get("trial_id")),
            bigquery.ScalarQueryParameter("trial_title", "STRING", r.get("trial_title")),
            bigquery.ScalarQueryParameter("phase", "STRING", r.get("phase")),
            bigquery.ScalarQueryParameter("dosage", "STRING", r.get("dosage")),
            bigquery.ScalarQueryParameter("trial_size", "STRING", r.get("trial_size")),
            bigquery.ScalarQueryParameter("trial_location", "STRING", r.get("trial_location")),
            bigquery.ScalarQueryParameter("source_url", "STRING", r.get("source_url")),
            bigquery.ScalarQueryParameter("data_source", "STRING", r.get("data_source")),
            bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", now),
        )
        for r in rows
    ]

    merge_query = f"""
        MERGE `{table_id}` T
        USING (SELECT * FROM UNNEST(@rows)) S
        ON T.drug_name = S.drug_name
           AND T.indication = S.indication
           AND IFNULL(T.trial_id, '') = IFNULL(S.trial_id, '')
        WHEN MATCHED THEN
            UPDATE SET
                llm_ot_name = S.llm_ot_name,
                indication_type = S.indication_type,
                therapy_area = S.therapy_area,
                rationale = S.rationale,
                trial_title = S.trial_title,
                phase = S.phase,
                dosage = S.dosage,
                trial_size = S.trial_size,
                trial_location = S.trial_location,
                source_url = S.source_url,
                data_source = S.data_source,
                updated_at = S.updated_at
        WHEN NOT MATCHED THEN
            INSERT (drug_name, indication, llm_ot_name, indication_type, therapy_area, rationale,
                    trial_id, trial_title, phase, dosage, trial_size, trial_location,
                    source_url, data_source, created_at, updated_at)
            VALUES (S.drug_name, S.indication, S.llm_ot_name, S.indication_type, S.therapy_area, S.rationale,
                    S.trial_id, S.trial_title, S.phase, S.dosage, S.trial_size, S.trial_location,
                    S.source_url, S.data_source, S.updated_at, S.updated_at)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("rows", "STRUCT", struct_params)]
    )
    query_job = bq_client.query(merge_query, job_config=job_config)
    query_job.result()
    logger.info(
        "[LE_PUSH] Upserted %d row(s) into %s (unique on drug_name + indication + trial_id)",
        len(rows),
        table_id,
    )


# ==============================
# SCORE CALCULATION TABLE: SCHEMA + PUSH
# ==============================
LE_SCORE_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("drug_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("indication", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("therapy_area", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ta_i", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ot_disease_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("trial_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("phase", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("primary_region", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("dosage", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("drug_arm_size_n", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("data_source", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("association_score", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("trial_weight", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("phase_weight", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("geo_score", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("sample_score", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("dosage_score", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("prior", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("maturity_weight", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("effective_indications", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("effective_therapy_areas", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("w_geo", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("w_dose", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("w_sample", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("q_i", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("e_phase_i", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("e_i", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("link", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("link_ta", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("l_ind", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("b_raw_ind", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("b_ind", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("l_ta", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("b_raw_ta", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("b_ta", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("b", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("overall_coherence", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("c", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("final_score", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
]


def _ensure_score_table_exists(bq_client: bigquery.Client, table_id: str) -> None:
    """Creates the score-calculation table if missing, and patches in any
    columns from ``LE_SCORE_SCHEMA`` that an already-existing table lacks."""
    table = bigquery.Table(table_id, schema=LE_SCORE_SCHEMA)
    table = bq_client.create_table(table, exists_ok=True)

    existing_field_names = {f.name for f in table.schema}
    missing_fields = [f for f in LE_SCORE_SCHEMA if f.name not in existing_field_names]
    if missing_fields:
        logger.info(
            "[LE_SCORE_PUSH] Table %s is missing column(s) %s - adding them now.",
            table_id,
            ", ".join(f.name for f in missing_fields),
        )
        table.schema = list(table.schema) + missing_fields
        bq_client.update_table(table, ["schema"])


def _json_safe(value, field_type: str = "STRING"):
    """Converts values that BigQuery's ``insertAll`` REST API would reject
    for the given ``field_type`` into ``None``:

    * ``float('nan')`` → ``None`` (NaN is not valid JSON; only ``null`` is).
    * For ``FLOAT64`` fields: any non-numeric string (e.g. ``"N/A"``,
      ``"n/a"``, ``"None"``, ``""``) → ``None``.  Numeric strings are
      coerced to ``float`` so BQ can ingest them correctly.

    This is the last line of defense before serialization, even if an
    upstream step should have already cleaned the value.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if field_type == "FLOAT64":
        if isinstance(value, (int, float)):
            return float(value)
        # String value in a FLOAT64 column — try to parse, else null out
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in ("", "nan", "none", "n/a", "na", "null", "-"):
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
    return value


def push_score_calculation(rows: list[dict]) -> None:
    """Replaces the score-calculation rows for the drug(s) present in
    ``rows`` inside ``LE_SCORE_CALCULATION_TABLE``.

    One row per TA-I (therapy_area + indication) combination. Since scores
    are recomputed holistically from the full drug dataset on every run
    (breadth/coherence metrics are dataset-level constants), a drug's prior
    rows are deleted and replaced rather than merged field-by-field.
    """
    if not rows:
        logger.info("[LE_SCORE_PUSH] No rows to push - skipping.")
        return

    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LE_SCORE_CALCULATION_TABLE}"
    now = datetime.now(timezone.utc).isoformat()

    bq_client = get_bq_client()
    _ensure_score_table_exists(bq_client, table_id)

    drug_names = sorted({r.get("drug_name") for r in rows if r.get("drug_name")})
    if drug_names:
        delete_query = f"DELETE FROM `{table_id}` WHERE drug_name IN UNNEST(@drug_names)"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("drug_names", "STRING", drug_names)]
        )
        bq_client.query(delete_query, job_config=job_config).result()

    insert_rows = []
    field_type_map = {field.name: field.field_type for field in LE_SCORE_SCHEMA}
    for r in rows:
        row = {
            field.name: _json_safe(r.get(field.name), field_type=field_type_map.get(field.name, "STRING"))
            for field in LE_SCORE_SCHEMA
            if field.name not in ("created_at", "updated_at")
        }
        row["created_at"] = now
        row["updated_at"] = now
        insert_rows.append(row)

    errors = bq_client.insert_rows_json(table_id, insert_rows)
    if errors:
        logger.error("[LE_SCORE_PUSH] Errors inserting rows into %s: %s", table_id, errors)
    else:
        logger.info(
            "[LE_SCORE_PUSH] Inserted %d row(s) into %s for drug(s): %s",
            len(insert_rows), table_id, ", ".join(drug_names),
        )
