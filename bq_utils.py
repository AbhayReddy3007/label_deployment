"""Label Expansion Opportunity — BigQuery utilities: merge module results and push to BigQuery.

Merges the rows produced by ``trial_analyser`` (Module 1) and
``web_analyser`` (Module 2) and upserts the merged, de-duplicated
rows into the configured BigQuery table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from google.cloud import bigquery

from medical_potential.config import BQ_DATASET_ID, LE_TABLE, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client

logger = logging.getLogger(__name__)


# ==============================
# MERGE
# ==============================
def merge_results(trial_rows: list[dict], web_rows: list[dict]) -> list[dict]:
    """Merges Module 1 (trial) and Module 2 (web) rows.

    De-duplicates on indication (case-insensitive). When the same
    indication is found by both modules, the trial-sourced row (which
    carries registered-trial evidence like phase/trial_id) is kept.
    """
    merged: dict[str, dict] = {}

    for row in trial_rows:
        key = row["indication"].strip().lower()
        merged[key] = dict(row)

    for row in web_rows:
        key = row["indication"].strip().lower()
        if key in merged:
            existing = merged[key]
            if not existing.get("source_url") and row.get("source_url"):
                existing["source_url"] = row["source_url"]
        else:
            merged[key] = dict(row)

    merged_rows = list(merged.values())
    logger.info(
        "[LE_MERGE] Merged %d trial row(s) + %d web row(s) -> %d unique indication row(s)",
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
    bigquery.SchemaField("indication_type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("therapy_area", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("rationale", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("trial_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("trial_title", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("phase", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("source_url", "STRING", mode="NULLABLE"),
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

    ``(drug_name, indication)`` is the unique key: re-running the
    pipeline updates the existing row instead of appending a duplicate.
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
            bigquery.ScalarQueryParameter("indication_type", "STRING", r.get("indication_type")),
            bigquery.ScalarQueryParameter("therapy_area", "STRING", r.get("therapy_area")),
            bigquery.ScalarQueryParameter("rationale", "STRING", r.get("rationale")),
            bigquery.ScalarQueryParameter("trial_id", "STRING", r.get("trial_id")),
            bigquery.ScalarQueryParameter("trial_title", "STRING", r.get("trial_title")),
            bigquery.ScalarQueryParameter("phase", "STRING", r.get("phase")),
            bigquery.ScalarQueryParameter("source_url", "STRING", r.get("source_url")),
            bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", now),
        )
        for r in rows
    ]

    merge_query = f"""
        MERGE `{table_id}` T
        USING (SELECT * FROM UNNEST(@rows)) S
        ON T.drug_name = S.drug_name AND T.indication = S.indication
        WHEN MATCHED THEN
            UPDATE SET
                indication_type = S.indication_type,
                therapy_area = S.therapy_area,
                rationale = S.rationale,
                trial_id = S.trial_id,
                trial_title = S.trial_title,
                phase = S.phase,
                source_url = S.source_url,
                updated_at = S.updated_at
        WHEN NOT MATCHED THEN
            INSERT (drug_name, indication, indication_type, therapy_area, rationale,
                    trial_id, trial_title, phase, source_url,
                    created_at, updated_at)
            VALUES (S.drug_name, S.indication, S.indication_type, S.therapy_area, S.rationale,
                    S.trial_id, S.trial_title, S.phase, S.source_url,
                    S.updated_at, S.updated_at)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("rows", "STRUCT", struct_params)]
    )
    query_job = bq_client.query(merge_query, job_config=job_config)
    query_job.result()
    logger.info(
        "[LE_PUSH] Upserted %d row(s) into %s (unique on drug_name + indication)",
        len(rows),
        table_id,
    )
