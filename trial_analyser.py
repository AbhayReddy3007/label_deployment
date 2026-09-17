"""Label Expansion Opportunity — Module 1: trial_analyser.

Reads the configured drug's clinical trial rows from the BigQuery
clinical-efficacy table, extracts every indication each trial is
studying (via Gemini + Google Search grounding), and classifies each
indication as Primary/Secondary with a therapy area.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from medical_potential.config import (
    BQ_DATASET_ID,
    CLINICAL_EFFICACY_TABLE,
    DRUG_NAME,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client
from medical_potential.label_expansion_opportunity.indication_extractor.utils import (
    INDICATIONS_PER_CALL,
    SECONDARY_INDICATION_CRITERIA,
    TRIALS_PER_CALL,
    extract_json,
    gemini_generate,
    gemini_generate_with_timeout,
)

from ..bq_utils import LE_TABLE, fetch_existing_trial_ids

logger = logging.getLogger(__name__)

MAX_WORKERS = 10
TRIAL_EXTRACTION_TIMEOUT_SECONDS = 90
TRIAL_EXTRACTION_MAX_ATTEMPTS = 2  # retry once on timeout before giving up on a batch

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
        SELECT molecule_name, company_name, source_url, phase, trial_id,
               dosage, trial_size, trial_location
        FROM {table_id}
        WHERE LOWER(molecule_name) = LOWER(@drug_name)
          AND llm_confidence > 0.6
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
# GEMINI: EXTRACT INDICATIONS PER BATCH OF TRIALS
# ==============================
def _clean_trial_id(trial_id) -> str:
    """Strips a trailing parenthetical annotation from a trial ID, e.g.
    'NCT06929156 (BGMxxxx)' -> 'NCT06929156'. Only the registry ID (before
    the parenthesis) should be used for searching ClinicalTrials.gov - the
    parenthetical part is an internal/company code, not part of the ID.
    """
    s = str(trial_id or "").strip()
    paren_idx = s.find("(")
    if paren_idx != -1:
        s = s[:paren_idx].strip()
    return s


def _normalize_trial_id(trial_id) -> str:
    """Strips whitespace/punctuation, a parenthetical annotation, and
    upper-cases a trial_id so it can be compared reliably regardless of
    minor formatting differences. Matches the normalization used by
    ``bq_utils.fetch_existing_trial_ids`` so the two stay in sync."""
    return re.sub(r"[^A-Za-z0-9]", "", _clean_trial_id(trial_id)).upper()


def _extract_trial_batch(rows: list[dict]) -> list[tuple[str, list[dict], str, str]]:
    """Extracts indications, exact trial title and phase for a batch of trials.

    Sends up to ``TRIALS_PER_CALL`` trials in a single Gemini call. Returns
    one ``(trial_id, conditions, trial_title, extracted_phase)`` tuple per
    input row, in the same order as ``rows``.
    """
    trial_ids = [str(r.get("trial_id")) for r in rows]

    trials_block = "\n\n".join(
        f"Trial {i + 1}:\n"
        f"Trial ID: {_clean_trial_id(r.get('trial_id'))}\n"
        f"Molecule: {r.get('molecule_name')}\n"
        f"Company: {r.get('company_name')}\n"
        f"Phase: {r.get('phase')}\n"
        f"Source URL: {r.get('source_url')}"
        for i, r in enumerate(rows)
    )

    prompt = f"""
You are a clinical trial data assistant.

Below are {len(rows)} clinical trials. For EACH trial, independently:

STEP 1 - Look up the trial.
Search for the trial using its Trial ID on ClinicalTrials.gov or other
clinical trial registries (e.g. EudraCT, WHO ICTRP). Find the EXACT
official trial title as registered. If a source URL is given for that
trial, also check that URL for the trial title.

STEP 2 - Extract indications.
From the trial record, extract ALL disease indications being studied,
including both the primary indication and any secondary/exploratory
indications that have documented outcomes. Look at the official trial
title, the "Conditions"/"Diseases" field, primary and secondary outcome
measures, and the trial description.

Trials:
{trials_block}

Return ONLY valid JSON - no markdown fences, no explanation. Include
exactly one entry per trial above, in the same order, each carrying its
exact Trial ID:
{{
  "trials": [
    {{
      "trial_id": "<exact Trial ID from input>",
      "conditions": [
        {{"indication": "<disease or condition>", "rationale": "<why - cite the trial record field>"}}
      ],
      "trial_title": "<EXACT official trial title as registered on the clinical trial registry>",
      "phase": "<Phase from the registry, e.g. Phase 1, Phase 2, Phase 3, Phase 4, Phase 2/3>"
    }}
  ]
}}

Rules:
- Return one object per trial provided above, even if extraction fails for
  one of them (use an empty "conditions" list and "N/A" fields for that trial only)
- trial_title must be the EXACT title from the registry, not a summary or guess
- phase must match what the registry lists
- Include ALL indications each trial is evaluating
- Always extract at least the primary indication for each trial
"""

    system_instruction = (
        "You are a clinical trial data assistant. Search for each trial on "
        "ClinicalTrials.gov or other registries to get the exact title. "
        "Return ONLY valid JSON."
    )

    # Larger batches legitimately need more time, so the timeout scales with batch size.
    timeout = TRIAL_EXTRACTION_TIMEOUT_SECONDS * len(rows)

    try:
        text = gemini_generate_with_timeout(
            prompt,
            system_instruction=system_instruction,
            use_search=True,
            timeout_seconds=timeout,
            max_attempts=TRIAL_EXTRACTION_MAX_ATTEMPTS,
            log_context=f"trial batch {trial_ids}",
        )
    except TimeoutError:
        return [(tid, [], "Skipped (timeout)", "") for tid in trial_ids]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TRIAL_ANALYSER] Extraction failed for trial batch %s: %s", trial_ids, exc)
        return [(tid, [], str(exc), "") for tid in trial_ids]

    try:
        data = extract_json(text)
        raw_trials = data.get("trials", []) if isinstance(data, dict) else []
        raw_trials = [t for t in raw_trials if isinstance(t, dict)]

        def _norm(tid) -> str:
            # Tolerates whitespace, case, punctuation, and a parenthetical
            # company-code suffix (e.g. "NCT01234567 (BGMxxxx)") so a
            # reformatted or annotated trial_id still matches the one we sent.
            return _normalize_trial_id(tid)

        by_trial_id = {_norm(t.get("trial_id")): t for t in raw_trials}

        if len(raw_trials) < len(rows):
            logger.warning(
                "[TRIAL_ANALYSER] Batch %s: requested %d trial(s) but Gemini returned only %d entr(y/ies) - "
                "likely truncated (batch too large) or partially dropped",
                trial_ids, len(rows), len(raw_trials),
            )

        results: list[tuple[str, list[dict], str, str]] = []
        for i, row in enumerate(rows):
            trial_id = row.get("trial_id")
            entry = by_trial_id.get(_norm(trial_id))

            # Fallback: if there's no normalized-ID match but the response has
            # an entry at the same position, use it (the prompt requires
            # Gemini to preserve input order even if it garbles the ID text).
            if not entry and i < len(raw_trials):
                candidate = raw_trials[i]
                logger.warning(
                    "[TRIAL_ANALYSER] Trial %s: no ID match in Gemini's response (got trial_id=%r at "
                    "position %d) - using positional fallback",
                    trial_id, candidate.get("trial_id"), i,
                )
                entry = candidate

            if not entry:
                logger.warning("[TRIAL_ANALYSER] No result returned for trial %s in batch", trial_id)
                results.append((trial_id, [], "N/A (missing from batch response)", ""))
                continue

            raw_conditions = entry.get("conditions", [])
            trial_title = entry.get("trial_title", "N/A")
            extracted_phase = (entry.get("phase") or "").strip()

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
            results.append((trial_id, deduped, trial_title, extracted_phase))

        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TRIAL_ANALYSER] Failed to parse extraction for trial batch %s: %s", trial_ids, exc)
        return [(tid, [], str(exc), "") for tid in trial_ids]


# ==============================
# GEMINI: CLASSIFY INDICATIONS (Primary/Secondary + therapy area)
# ==============================
def _classify_indication_batch(drug_name: str, indication_batch: list[str]) -> dict[str, dict]:
    """Classifies up to ``INDICATIONS_PER_CALL`` indications in a single Gemini call."""
    if not indication_batch:
        return {}

    import json as _json

    indications_json = _json.dumps(indication_batch, indent=2)
    prompt = f"""
You are a pharmaceutical analyst. Research the drug "{drug_name}" and
classify each of the following indications.

Indications to classify:
{indications_json}

STEP 1 - Research the drug: what it is primarily approved/developed for,
FDA/EMA approved labels, and the originator's pipeline. Check ALL of its
current approved indications - a drug can be approved for MORE THAN ONE
indication at the same time (e.g. tirzepatide is approved for both type 2
diabetes AND obesity/weight management - both are Primary, not just the
first one approved).

STEP 2 - Classify each indication.
  indication_type:
    "Primary"   - ANY indication that is currently part of the drug's
                  approved label (FDA/EMA/etc.), or, for clinical-stage
                  assets, one of its originally intended indications.
                  Do not restrict this to a single indication - list every
                  currently-approved use as Primary.
    "Secondary" - a genuine label-expansion candidate: NOT currently part
                  of the drug's approved label and not one of its
                  originally intended indications.
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
        for ind in indication_batch:
            result.setdefault(ind.lower(), {"indication_type": "Secondary", "therapy_area": "Other", "rationale": ""})
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TRIAL_ANALYSER] Classification failed for '%s' batch %s: %s", drug_name, indication_batch, exc)
        return {
            ind.lower(): {"indication_type": "Secondary", "therapy_area": "Other", "rationale": f"Classification failed: {exc}"}
            for ind in indication_batch
        }


def _classify_indications(drug_name: str, unique_indications: list[str]) -> dict[str, dict]:
    """Classifies all unique indications, ``INDICATIONS_PER_CALL`` at a time.

    Splits ``unique_indications`` into batches of size ``INDICATIONS_PER_CALL``
    and runs one Gemini call per batch (in parallel), then merges the
    per-batch classification maps into one.
    """
    if not unique_indications:
        return {}

    batches = [
        unique_indications[i : i + INDICATIONS_PER_CALL]
        for i in range(0, len(unique_indications), INDICATIONS_PER_CALL)
    ]
    logger.info(
        "[TRIAL_ANALYSER] Classifying %d unique indication(s) in %d batch(es) of up to %d",
        len(unique_indications),
        len(batches),
        INDICATIONS_PER_CALL,
    )

    classification_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_classify_indication_batch, drug_name, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                classification_map.update(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("[TRIAL_ANALYSER] Unexpected error classifying batch %s: %s", batch, exc)
                for ind in batch:
                    classification_map.setdefault(
                        ind.lower(),
                        {"indication_type": "Secondary", "therapy_area": "Other", "rationale": f"Classification failed: {exc}"},
                    )

    return classification_map


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

    # Incremental run: skip trials whose trial_id is already in LE_TABLE for
    # this drug, so re-running the pipeline doesn't re-extract (and
    # re-append/re-generate slightly different text for) trials already
    # processed. Only genuinely new trials hit Gemini.
    existing_trial_ids = fetch_existing_trial_ids(drug_name)
    if existing_trial_ids:
        total_before = len(trial_rows)
        trial_rows = [
            r for r in trial_rows
            if _normalize_trial_id(r.get("trial_id")) not in existing_trial_ids
        ]
        skipped = total_before - len(trial_rows)
        if skipped:
            logger.info(
                "[TRIAL_ANALYSER] Skipping %d trial(s) already in %s for '%s' - processing %d new trial(s)",
                skipped, LE_TABLE, drug_name, len(trial_rows),
            )

    if not trial_rows:
        logger.info(
            "[TRIAL_ANALYSER] No new trials to process for '%s' - all trials already in %s",
            drug_name, LE_TABLE,
        )
        return []

    logger.info(
        "[TRIAL_ANALYSER] Extracting indications from %d trial(s) in batches of %d, using %d workers",
        len(trial_rows),
        TRIALS_PER_CALL,
        MAX_WORKERS,
    )
    trial_batches = [trial_rows[i : i + TRIALS_PER_CALL] for i in range(0, len(trial_rows), TRIALS_PER_CALL)]

    extractions = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_extract_trial_batch, batch): batch for batch in trial_batches}
        for future in as_completed(futures):
            batch = futures[future]
            rows_by_id = {r.get("trial_id"): r for r in batch}
            try:
                batch_results = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[TRIAL_ANALYSER] Unexpected error for batch %s: %s", [r.get("trial_id") for r in batch], exc)
                batch_results = [(r.get("trial_id"), [], str(exc), "") for r in batch]

            for trial_id, conditions, trial_title, extracted_phase in batch_results:
                row = rows_by_id.get(trial_id, {})
                extractions.append((trial_id, conditions, trial_title, extracted_phase, row))

    flat_rows: list[dict] = []
    skipped_trials: list[str] = []
    for trial_id, conditions, trial_title, extracted_phase, row in extractions:
        phase = row.get("phase") or extracted_phase or ""
        if not conditions:
            # Every trial should produce at least one row. If extraction
            # returned nothing (timeout, parse error, Gemini omission),
            # create a placeholder so the trial still appears in the output.
            logger.warning(
                "[TRIAL_ANALYSER] Trial %s returned no indications - creating placeholder row",
                trial_id,
            )
            skipped_trials.append(str(trial_id))
            flat_rows.append(
                {
                    "drug_name": row.get("molecule_name") or drug_name,
                    "indication": "Unknown (extraction failed)",
                    "rationale": f"No indications could be extracted. Trial title: {trial_title}",
                    "trial_title": trial_title,
                    "trial_id": trial_id,
                    "phase": phase,
                    "dosage": row.get("dosage"),
                    "trial_size": str(row.get("trial_size")) if row.get("trial_size") is not None else None,
                    "trial_location": row.get("trial_location"),
                    "source_url": row.get("source_url"),
                    "data_source": "Trials",
                }
            )
            continue

        seen: set[str] = set()
        for c in conditions:
            std = (c.get("indication", "") or "").strip()
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
                    "dosage": row.get("dosage"),
                    "trial_size": str(row.get("trial_size")) if row.get("trial_size") is not None else None,
                    "trial_location": row.get("trial_location"),
                    "source_url": row.get("source_url"),
                    "data_source": "Trials",
                }
            )

    if skipped_trials:
        logger.warning(
            "[TRIAL_ANALYSER] %d trial(s) had no indications extracted: %s",
            len(skipped_trials), ", ".join(skipped_trials),
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
