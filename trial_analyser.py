"""Label Expansion Opportunity — Module 1: trial_analyser.

Reads the configured drug's clinical trial rows from the BigQuery
clinical-efficacy table, extracts every indication each trial is
studying (via Gemini + Google Search grounding), and classifies each
indication as Primary/Secondary with a therapy area.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from indication_standardizer import standardize_indication

from medical_potential.config import (
    BQ_DATASET_ID,
    CLINICAL_EFFICACY_TABLE,
    DRUG_NAME,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client
from medical_potential.label_expansion_opportunity._llm_client import (
    extract_json,
    gemini_generate,
)

logger = logging.getLogger(__name__)

MAX_WORKERS = 10
TRIAL_EXTRACTION_TIMEOUT_SECONDS = 90

SECONDARY_INDICATION_CRITERIA = """
A secondary indication qualifies ONLY if ALL of the following are true:

- The indication represents a true expansion - i.e., it is not part of the
  primary indication (for clinical assets) or currently approved label
  (for commercial assets)
- The indication is described at a clear disease-level definition, avoiding
  vague, overlapping, or synonymous representations
- The source must describe observed or measured outcomes in that specific
  indication (e.g., trial results, endpoint readouts, biomarker response),
  not just planned evaluation or exploratory intent

The following must NOT be considered secondary indications:
* Indications mentioned only as hypothesis, targets, or exploratory possibilities
* Pipeline indications without any data or outcomes
* Mechanism based assumptions without clinical or empirical validation
* Early discovery or preclinical signals without human data
* Any indication lacking traceable, verifiable evidence of results
"""


# ==============================
# BIGQUERY: FETCH TRIAL ROWS
# ==============================
def fetch_trial_rows(drug_name: str = DRUG_NAME) -> list[dict]:
    """Fetches this drug's clinical trial rows from the configured BQ table.

    ``drug_name`` must be a single drug name (str), not a list.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise TypeError(f"fetch_trial_rows() accepts exactly one drug name (str), got: {drug_name!r}")

    bq_client = get_bq_client()
    table_id = f"`{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}`"

    query = f"""
        SELECT molecule_name, company_name, source_url, phase, trial_id
        FROM {table_id}
        WHERE LOWER(molecule_name) = LOWER(@drug_name)
    """
    import google.cloud.bigquery as bigquery  # local import keeps gcp_utils as the single client owner

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name)]
    )
    results = bq_client.query(query, job_config=job_config).result()
    rows = [dict(row) for row in results]
    logger.info("[TRIAL_ANALYSER] Retrieved %d trial row(s) for '%s' from %s", len(rows), drug_name, table_id)
    return rows


# ==============================
# GEMINI: EXTRACT INDICATIONS PER TRIAL
# ==============================
def _extract_single_trial(row: dict) -> tuple[str, list[dict], str, str]:
    """Extracts indications, exact trial title and phase for a single trial."""
    trial_id = row.get("trial_id")

    prompt = f"""
You are a clinical trial data assistant.

Trial details:
Molecule: {row.get('molecule_name')}
Company: {row.get('company_name')}
Trial ID: {trial_id}
Phase: {row.get('phase')}
Source URL: {row.get('source_url')}

STEP 1 - Look up the trial.
Search for this trial using the Trial ID "{trial_id}" on ClinicalTrials.gov
or other clinical trial registries (e.g. EudraCT, WHO ICTRP). Find the
EXACT official trial title as registered. If the source URL is provided,
also check that URL for the trial title.

STEP 2 - Extract indications.
From the trial record, extract ALL disease indications being studied,
including both the primary indication and any secondary/exploratory
indications that have documented outcomes. Look at the official trial
title, the "Conditions"/"Diseases" field, primary and secondary outcome
measures, and the trial description.

Return ONLY valid JSON - no markdown fences, no explanation:
{{
  "conditions": [
    {{"indication": "<disease or condition>", "rationale": "<why - cite the trial record field>"}}
  ],
  "trial_title": "<EXACT official trial title as registered on the clinical trial registry>",
  "phase": "<Phase from the registry, e.g. Phase 1, Phase 2, Phase 3, Phase 4, Phase 2/3>"
}}

Rules:
- trial_title must be the EXACT title from the registry, not a summary or guess
- phase must match what the registry lists
- Include ALL indications the trial is evaluating
- Always extract at least the primary indication
"""

    result_holder: dict = {}
    error_holder: dict = {}

    def _call_gemini():
        try:
            result_holder["text"] = gemini_generate(
                prompt,
                system_instruction=(
                    "You are a clinical trial data assistant. Search for the trial on "
                    "ClinicalTrials.gov or other registries to get the exact title. "
                    "Return ONLY valid JSON."
                ),
                use_search=True,
            )
        except Exception as exc:  # noqa: BLE001
            error_holder["error"] = exc

    thread = threading.Thread(target=_call_gemini, daemon=True)
    thread.start()
    thread.join(timeout=TRIAL_EXTRACTION_TIMEOUT_SECONDS)

    if thread.is_alive():
        logger.warning("[TRIAL_ANALYSER] Timeout (>%ss) - skipping trial %s", TRIAL_EXTRACTION_TIMEOUT_SECONDS, trial_id)
        return trial_id, [], "Skipped (timeout)", ""

    if "error" in error_holder:
        logger.warning("[TRIAL_ANALYSER] Extraction failed for trial %s: %s", trial_id, error_holder["error"])
        return trial_id, [], str(error_holder["error"]), ""

    try:
        data = extract_json(result_holder.get("text", ""))
        raw_conditions = data.get("conditions", [])
        trial_title = data.get("trial_title", "N/A")
        extracted_phase = (data.get("phase") or "").strip()

        conditions = []
        for c in raw_conditions if isinstance(raw_conditions, list) else []:
            if isinstance(c, dict):
                indication = (c.get("indication") or "").strip()
                rationale = (c.get("rationale") or "").strip()
            elif isinstance(c, str):
                indication, rationale = c.strip(), ""
            else:
                continue
            if indication:
                conditions.append({"indication": indication, "rationale": rationale})

        seen: set[str] = set()
        deduped = []
        for c in conditions:
            key = c["indication"].lower()
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        logger.info(
            "[TRIAL_ANALYSER] %s: %s",
            trial_id,
            ", ".join(c["indication"] for c in deduped) if deduped else "no indications",
        )
        return trial_id, deduped, trial_title, extracted_phase
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TRIAL_ANALYSER] Failed to parse extraction for trial %s: %s", trial_id, exc)
        return trial_id, [], str(exc), ""


# ==============================
# GEMINI: CLASSIFY INDICATIONS (Primary/Secondary + therapy area)
# ==============================
def _classify_indications(drug_name: str, unique_indications: list[str]) -> dict[str, dict]:
    """Classifies each indication as Primary/Secondary and assigns a therapy area."""
    if not unique_indications:
        return {}

    import json as _json

    indications_json = _json.dumps(unique_indications, indent=2)
    prompt = f"""
You are a pharmaceutical analyst. Research the drug "{drug_name}" and
classify each of the following indications.

Indications to classify:
{indications_json}

STEP 1 - Research the drug: what it is primarily approved/developed for,
FDA/EMA approved labels, and the originator's pipeline.

STEP 2 - Classify each indication.
  indication_type:
    "Primary"   - one of the drug's main approved or originally intended indications.
    "Secondary" - a label expansion beyond the primary use.
                  {SECONDARY_INDICATION_CRITERIA}
  therapy_area:
    Choose from: Metabolic, Cardiovascular, Oncology, Neuroscience,
    Immunology, Respiratory, Nephrology, Hepatology, Ophthalmology,
    Musculoskeletal, Gastroenterology, Infectious Disease, Dermatology,
    Hematology, Endocrinology, Rare Disease, or another appropriate area.

Return ONLY valid JSON:
{{
  "classifications": [
    {{"indication": "<exact indication name from input list>",
      "indication_type": "Primary" or "Secondary",
      "therapy_area": "<therapy area>",
      "rationale": "<why, citing the specific evidence you found>"}}
  ]
}}
"""
    try:
        text = gemini_generate(
            prompt,
            system_instruction=(
                "You are a pharmaceutical analyst. Search the web to find what this drug "
                "is approved for. Return ONLY valid JSON."
            ),
            use_search=True,
        )
        data = extract_json(text)
        classifications = data.get("classifications", [])
        result = {
            (c.get("indication") or "").strip().lower(): {
                "indication_type": c.get("indication_type", "Secondary"),
                "therapy_area": c.get("therapy_area", "Other"),
                "rationale": c.get("rationale", ""),
            }
            for c in classifications
            if (c.get("indication") or "").strip()
        }
        for ind in unique_indications:
            result.setdefault(ind.lower(), {"indication_type": "Secondary", "therapy_area": "Other", "rationale": ""})
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TRIAL_ANALYSER] Bulk classification failed for '%s': %s", drug_name, exc)
        return {
            ind.lower(): {"indication_type": "Secondary", "therapy_area": "Other", "rationale": f"Classification failed: {exc}"}
            for ind in unique_indications
        }


# ==============================
# ENTRY POINT FOR THIS MODULE
# ==============================
def analyse(drug_name: str = DRUG_NAME) -> list[dict]:
    """Runs the full trial-analysis pipeline for exactly one drug.

    ``drug_name`` must be a single drug name (str) - not a list. To
    analyse multiple drugs, call this once per drug from the caller.

    Returns a flat list of row dicts, one per indication/trial.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise TypeError(
            f"trial_analyser.analyse() accepts exactly one drug name (str), got: {drug_name!r}"
        )

    logger.info("[TRIAL_ANALYSER] Starting trial analysis for '%s'", drug_name)
    trial_rows = fetch_trial_rows(drug_name)
    if not trial_rows:
        logger.warning("[TRIAL_ANALYSER] No trial rows found for '%s' - skipping module", drug_name)
        return []

    logger.info("[TRIAL_ANALYSER] Extracting indications from %d trial(s) using %d workers", len(trial_rows), MAX_WORKERS)
    extractions = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_extract_single_trial, row): row for row in trial_rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                extractions.append((*future.result(), row))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[TRIAL_ANALYSER] Unexpected error for %s: %s", row.get("trial_id"), exc)
                extractions.append((row.get("trial_id"), [], str(exc), "", row))

    flat_rows: list[dict] = []
    for trial_id, conditions, trial_title, extracted_phase, row in extractions:
        phase = row.get("phase") or extracted_phase or ""
        if not conditions:
            continue

        seen: set[str] = set()
        for c in conditions:
            std = standardize_indication(c.get("indication", ""))
            if not std or std.lower() in ("error", "n/a", "no indication found", "none"):
                continue
            if std.lower() in seen:
                continue
            seen.add(std.lower())
            flat_rows.append(
                {
                    "drug_name": row.get("molecule_name") or drug_name,
                    "indication": std,
                    "rationale": c.get("rationale", ""),
                    "trial_title": trial_title,
                    "trial_id": trial_id,
                    "phase": phase,
                    "source_url": row.get("source_url"),
                }
            )

    unique_indications = sorted({r["indication"] for r in flat_rows})
    classification_map = _classify_indications(drug_name, unique_indications)
    for row in flat_rows:
        cls = classification_map.get(row["indication"].lower(), {})
        row["indication_type"] = cls.get("indication_type", "")
        row["therapy_area"] = cls.get("therapy_area", "")
        row["rationale"] = row["rationale"] or cls.get("rationale", "")

    logger.info("[TRIAL_ANALYSER] Completed. %d row(s) for '%s'", len(flat_rows), drug_name)
    return flat_rows
