"""Label Expansion Opportunity — Module 2: web_analyser.

Uses Gemini + Google Search grounding to look directly at public web
sources (regulatory filings, company pipeline pages, conference
readouts, news) for label-expansion signals for the configured drug -
independent of, and complementary to, the registered-trial evidence
mined by ``trial_analyser``. This catches expansion signals that are
public but not yet reflected as a registered clinical trial.
"""

from __future__ import annotations

import logging

from medical_potential.config import DRUG_NAME
from medical_potential.label_expansion_opportunity.indication_extractor.utils import (
    INDICATIONS_PER_CALL,
    PROCESS_INDICATIONS,
    SECONDARY_INDICATION_CRITERIA,
    extract_json,
    gemini_generate,
)

from ..bq_utils import fetch_existing_indication_rows

logger = logging.getLogger(__name__)


def _research_prompt(drug_name: str) -> str:
    return f"""
You are a pharmaceutical analyst researching label-expansion opportunities
for the drug "{drug_name}".

Search the web for:
- The drug's current FDA/EMA approved label / primary indication(s)
- Company pipeline pages, press releases, and investor updates that
  mention new indications being pursued
- Conference readouts, publications, or regulatory filings (e.g.
  Breakthrough Therapy, Priority Review, Orphan Drug designations)
  describing observed or measured outcomes in a new indication

For every additional (non-primary) indication you find evidence for,
capture it as a candidate label-expansion opportunity.

Do NOT capture as an indication:
- Trial endpoints or outcome measures (e.g. "Exercise Capacity",
  "Waist Circumference", "Postprandial Glucose")
- Biomarkers or lab values (e.g. "Lipid Profile", "Hepatocyte Ballooning")
- Pharmacokinetic parameters (e.g. "Pharmacokinetics")
- Procedures or interventions (e.g. "Bariatric Surgery")
Only capture things that could plausibly appear as an approved FDA
indication - a disease or condition name.

A candidate only qualifies as a secondary/label-expansion indication if:
{SECONDARY_INDICATION_CRITERIA}

Also give your best guess of each indication's standardized Open Targets
(EFO/MONDO) disease name - the canonical disease term as it would appear
in the Open Targets Platform, not a synonym or colloquial phrasing. If
you are not confident, leave this null rather than guessing.

Return ONLY valid JSON - no markdown fences, no explanation:
{{
  "primary_indications": ["<the drug's current approved/primary indication(s)>"],
  "opportunities": [
    {{
      "indication": "<disease or condition>",
      "indication_type": "Primary" or "Secondary",
      "therapy_area": "<Metabolic, Cardiovascular, Oncology, Neuroscience, Immunology, Respiratory, Nephrology, Hepatology, Ophthalmology, Musculoskeletal, Gastroenterology, Infectious Disease, Dermatology, Hematology, Endocrinology, Rare Disease, or another appropriate area>",
      "rationale": "<why - cite the specific source/evidence you found>",
      "source_url": "<the URL of the source that supports this, if available>",
      "ot_disease_name": "<your best guess of the Open Targets disease name, or null>"
    }}
  ]
}}
"""


def _fetch_opportunities(drug_name: str) -> list[dict]:
    prompt = _research_prompt(drug_name)
    try:
        text = gemini_generate(
            prompt,
            system_instruction=(
                "You are a pharmaceutical analyst. Search the web thoroughly for label-"
                "expansion evidence. Return ONLY valid JSON."
            ),
            use_search=True,
        )
        data = extract_json(text)
        return data.get("opportunities", []) if isinstance(data, dict) else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("[WEB_ANALYSER] Web research failed for '%s': %s", drug_name, exc)
        return []


# ==============================
# PROCESS_INDICATIONS MODE: RECLASSIFY WITHOUT RE-SEARCHING THE WEB
# ==============================
def _classify_web_indication_batch(drug_name: str, indication_batch: list[str]) -> dict[str, dict]:
    """Re-classifies up to ``INDICATIONS_PER_CALL`` already-known web
    indications in a single Gemini call - indication_type, therapy_area,
    and ot_disease_name - WITHOUT re-running the full web-research prompt.
    Still uses Search grounding (to check current approval status), but
    is far cheaper than ``_fetch_opportunities``'s open-ended research
    pass since it only classifies indications already given to it rather
    than discovering new ones from scratch.
    """
    if not indication_batch:
        return {}

    import json as _json

    indications_json = _json.dumps(indication_batch, indent=2)
    prompt = f"""You are a pharmaceutical analyst. Research the drug "{drug_name}" and
re-classify each of the following previously-identified indications.

Indications to classify:
{indications_json}

STEP 1 - Research the drug: what it is primarily approved/developed for,
FDA/EMA approved labels, and the originator's pipeline.

STEP 2 - Classify each indication.
  indication_type — you MUST choose exactly one of these two values:
    "Primary"   - one of the drug's main approved or originally intended indications.
    "Secondary" - a label expansion beyond the primary use.
                  {SECONDARY_INDICATION_CRITERIA}
  Do NOT use any other value. If unsure, classify as "Secondary". Never
  use "None", "Not Applicable", "Not Classified", null, or any other label.
  therapy_area:
    Choose from: Metabolic, Cardiovascular, Oncology, Neuroscience,
    Immunology, Respiratory, Nephrology, Hepatology, Ophthalmology,
    Musculoskeletal, Gastroenterology, Infectious Disease, Dermatology,
    Hematology, Endocrinology, Rare Disease, or another appropriate area.
  ot_disease_name: your best guess of this indication's standardized Open
    Targets (EFO/MONDO) disease name. If not confident, use null.

Return ONLY valid JSON:
{{
  "classifications": [
    {{"indication": "<exact indication name from input list>",
      "indication_type": "Primary" or "Secondary",
      "therapy_area": "<therapy area>",
      "ot_disease_name": "<Open Targets disease name, or null>",
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
        _VALID_TYPES = {"primary", "secondary"}
        result = {
            (c.get("indication") or "").strip().lower(): {
                "indication_type": c.get("indication_type", "Secondary")
                    if (c.get("indication_type") or "").strip().lower() in _VALID_TYPES
                    else "Secondary",
                "therapy_area": c.get("therapy_area", "Other"),
                "ot_disease_name": c.get("ot_disease_name"),
                "rationale": c.get("rationale", ""),
            }
            for c in classifications
            if (c.get("indication") or "").strip()
        }
        for ind in indication_batch:
            result.setdefault(ind.lower(), {"indication_type": "Secondary", "therapy_area": "Other", "ot_disease_name": None, "rationale": ""})
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[WEB_ANALYSER] Reclassification failed for '%s' batch %s: %s", drug_name, indication_batch, exc)
        return {
            ind.lower(): {"indication_type": "Secondary", "therapy_area": "Other", "ot_disease_name": None, "rationale": f"Classification failed: {exc}"}
            for ind in indication_batch
        }


def _reprocess_existing_indications(drug_name: str) -> list[dict]:
    """Re-classifies web-sourced indications already sitting in
    ``LE_TABLE`` for this drug instead of re-running the full web-search
    research pass. One (or a few) batched classify call(s) over the
    unique indications, versus one open-ended research call - much
    cheaper, at the cost of not discovering any new web-sourced
    opportunities that aren't already stored.
    """
    logger.info(
        "[WEB_ANALYSER] PROCESS_INDICATIONS=True — reprocessing existing web indications for '%s' "
        "instead of re-running web research",
        drug_name,
    )
    existing_rows = fetch_existing_indication_rows(drug_name, source="web")
    if not existing_rows:
        logger.warning(
            "[WEB_ANALYSER] No existing web-sourced rows found in LE_TABLE for '%s' to reprocess",
            drug_name,
        )
        return []

    unique_indications = sorted({
        (r.get("indication") or "").strip()
        for r in existing_rows
        if (r.get("indication") or "").strip()
    })
    batches = [
        unique_indications[i : i + INDICATIONS_PER_CALL]
        for i in range(0, len(unique_indications), INDICATIONS_PER_CALL)
    ]
    classification_map: dict[str, dict] = {}
    for batch in batches:
        classification_map.update(_classify_web_indication_batch(drug_name, batch))

    for row in existing_rows:
        cls = classification_map.get((row.get("indication") or "").strip().lower(), {})
        if cls:
            row["indication_type"] = cls.get("indication_type", row.get("indication_type", ""))
            row["therapy_area"] = cls.get("therapy_area", row.get("therapy_area", ""))
            row["llm_ot_name"] = cls.get("ot_disease_name") or row.get("llm_ot_name")
            row["rationale"] = row.get("rationale") or cls.get("rationale", "")

    logger.info(
        "[WEB_ANALYSER] Reprocessed %d existing row(s) for '%s'",
        len(existing_rows), drug_name,
    )
    return existing_rows


# ==============================
# ENTRY POINT FOR THIS MODULE
# ==============================
def analyse(drug_name: str = DRUG_NAME) -> list[dict]:
    """Runs the web-research pipeline for exactly one drug.

    ``drug_name`` must be a single drug name (str) - not a list. To
    analyse multiple drugs, call this once per drug from the caller.

    If ``PROCESS_INDICATIONS`` (in ``utils.py``) is ``True``, skips the
    full web-research pass entirely, and instead re-classifies the
    web-sourced indications already sitting in ``LE_TABLE`` for this
    drug - see ``_reprocess_existing_indications``.

    Returns a flat list of row dicts. Rows have no ``trial_id``/
    ``trial_title``/``phase`` since they aren't sourced from a
    registered trial.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise TypeError(
            f"web_analyser.analyse() accepts exactly one drug name (str), got: {drug_name!r}"
        )

    if PROCESS_INDICATIONS:
        return _reprocess_existing_indications(drug_name)

    logger.info("[WEB_ANALYSER] Starting web research for '%s'", drug_name)
    opportunities = _fetch_opportunities(drug_name)
    if not opportunities:
        logger.warning("[WEB_ANALYSER] No web-research opportunities found for '%s'", drug_name)
        return []

    flat_rows: list[dict] = []
    seen: set[str] = set()
    for opp in opportunities:
        indication = (opp.get("indication") or "").strip()
        if not indication or indication.lower() in seen:
            continue
        seen.add(indication.lower())
        flat_rows.append(
            {
                "drug_name": drug_name,
                "indication": indication,
                "llm_ot_name": opp.get("ot_disease_name") or None,
                "rationale": (opp.get("rationale") or "").strip(),
                "trial_title": None,
                "trial_id": None,
                "phase": None,
                "source_url": opp.get("source_url") or None,
                "indication_type": opp.get("indication_type", "Secondary"),
                "therapy_area": opp.get("therapy_area", "Other"),
                "data_source": "Web",
            }
        )

    logger.info("[WEB_ANALYSER] Completed. %d row(s) for '%s'", len(flat_rows), drug_name)
    return flat_rows
