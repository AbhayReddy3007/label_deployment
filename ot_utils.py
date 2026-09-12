"""Open Targets mapping — shared utilities.

Shared constants, OT GraphQL helpers, Gemini call wrapper, JSON parsing,
and BigQuery read/write functions used by both ``moa_mapping`` and
``indication_mapping``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import requests
from google.cloud import bigquery

from medical_potential.config import BQ_DATASET_ID, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client

logger = logging.getLogger(__name__)

# ==============================
# TABLE NAMES
# ==============================
OT_MOA_TABLE = "ot_moa_mapping"
OT_DISEASE_TABLE = "ot_disease_mapping"

# ==============================
# OT API
# ==============================
OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
HEADERS_JSON = {"Content-Type": "application/json"}


# ──────────────────────────────────────────────────────────────────────
# OT GraphQL POST with retry
# ──────────────────────────────────────────────────────────────────────
def ot_post(
    query: str,
    variables: dict,
    context: str = "",
    max_retries: int = 3,
) -> dict | None:
    """POST to the OT GraphQL endpoint with retry + exponential backoff.
    Returns the ``data`` dict on success, ``None`` on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                OT_GRAPHQL,
                json={"query": query, "variables": variables},
                headers=HEADERS_JSON,
                timeout=30,
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "[OT_UTILS] OT HTTP %s [%s] — retrying in %ds (%d/%d)",
                    r.status_code, context, wait, attempt, max_retries,
                )
                time.sleep(wait)
                continue
            if not r.ok:
                logger.warning("[OT_UTILS] OT HTTP %s [%s]: %s", r.status_code, context, r.text[:300])
                return None
            body = r.json()
            if "errors" in body:
                logger.warning("[OT_UTILS] OT GraphQL error [%s]: %s", context, body["errors"])
                return None
            return body.get("data")
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning("[OT_UTILS] OT timeout [%s] — retrying (%d/%d)", context, attempt, max_retries)
                time.sleep(2 ** attempt)
                continue
            logger.warning("[OT_UTILS] OT timeout [%s] — giving up", context)
            return None
        except Exception as exc:
            logger.warning("[OT_UTILS] OT request failed [%s]: %s", context, exc)
            return None
    return None


# ──────────────────────────────────────────────────────────────────────
# Gemini call (raw REST, with Google Search grounding)
# ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


def gemini_call(prompt: str, max_retries: int = 2, max_tokens: int = 2048) -> str:
    """Send a prompt to Gemini with Google Search grounding.
    Returns the text response, or an empty string on total failure."""
    if not GEMINI_API_KEY:
        logger.error("[OT_UTILS] GEMINI_API_KEY not set — cannot call Gemini")
        return ""

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens},
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=payload, headers=HEADERS_JSON, timeout=60)
            r.raise_for_status()
            parts = (
                r.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            return " ".join(p.get("text", "") for p in parts).strip()
        except Exception as exc:
            logger.warning("[OT_UTILS] Gemini attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return ""


# ──────────────────────────────────────────────────────────────────────
# JSON parsing (robust extraction from free-text Gemini responses)
# ──────────────────────────────────────────────────────────────────────
def parse_json_response(text: str) -> list[dict]:
    """Robustly extract a JSON array from Gemini's free-text output.

    Strategies (in order):
      1. Strip markdown fences → find outermost ``[…]`` by bracket counting.
      2. Fix trailing commas → retry ``json.loads``.
      3. Object-by-object extraction via ``{…}`` regex.
    """
    cleaned = re.sub(r"```(?:json|JSON)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # Find outermost [...]
    def _find_array(s: str) -> str | None:
        start = s.find("[")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None

    array_str = _find_array(cleaned)
    if array_str:
        fixed = re.sub(r",\s*([\]\}])", r"\1", array_str)
        try:
            result = json.loads(fixed)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Object-by-object fallback
    objects = re.findall(r"\{[^{}]+\}", cleaned, re.DOTALL)
    results = []
    for obj_str in objects:
        obj_str = re.sub(r",\s*([\]\}])", r"\1", obj_str).strip()
        try:
            results.append(json.loads(obj_str))
        except json.JSONDecodeError:
            pass
    return results


# ──────────────────────────────────────────────────────────────────────
# BigQuery helpers
# ──────────────────────────────────────────────────────────────────────
def ensure_table_exists(
    table_name: str,
    schema: list[bigquery.SchemaField],
) -> str:
    """Creates the table if missing; patches any new columns.
    Returns the fully-qualified table ID."""
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"
    bq_client = get_bq_client()

    table = bigquery.Table(table_id, schema=schema)
    table = bq_client.create_table(table, exists_ok=True)

    existing_field_names = {f.name for f in table.schema}
    missing_fields = [f for f in schema if f.name not in existing_field_names]
    if missing_fields:
        logger.info(
            "[OT_UTILS] Table %s is missing column(s) %s — adding them now.",
            table_id,
            ", ".join(f.name for f in missing_fields),
        )
        table.schema = list(table.schema) + missing_fields
        bq_client.update_table(table, ["schema"])

    return table_id


def fetch_existing_mappings(
    table_name: str,
    source_column: str,
) -> dict[str, str]:
    """Returns a dict of ``{source_value_lower: resolved_value}`` from the
    given mapping table. Returns an empty dict if the table doesn't exist."""
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"
    bq_client = get_bq_client()

    target_column = "ot_moa" if source_column == "moa" else "ot_disease"
    query = f"SELECT {source_column}, {target_column} FROM `{table_id}`"
    try:
        results = bq_client.query(query).result()
        return {
            row[source_column].strip().lower(): row[target_column]
            for row in results
            if row[source_column]
        }
    except Exception:
        logger.info(
            "[OT_UTILS] Table %s does not exist yet or is empty — treating all values as new",
            table_id,
        )
        return {}


def push_mappings(
    table_name: str,
    schema: list[bigquery.SchemaField],
    rows: list[dict],
) -> None:
    """Inserts new mapping rows into the given table (append-only — caller
    ensures only genuinely new values are passed)."""
    if not rows:
        logger.info("[OT_UTILS] No new mappings to push to %s — skipping.", table_name)
        return

    table_id = ensure_table_exists(table_name, schema)
    bq_client = get_bq_client()

    now = datetime.now(timezone.utc).isoformat()
    for r in rows:
        r.setdefault("created_at", now)
        r.setdefault("updated_at", now)

    errors = bq_client.insert_rows_json(table_id, rows)
    if errors:
        logger.error("[OT_UTILS] Errors inserting rows into %s: %s", table_id, errors)
    else:
        logger.info("[OT_UTILS] Inserted %d new mapping(s) into %s", len(rows), table_id)
