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
from medical_potential.label_expansion_opportunity._llm_client import (
    extract_json,
    gemini_generate,
)
from medical_potential.label_expansion_opportunity.trial_analyser import (
    SECONDARY_INDICATION_CRITERIA,
)

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

A candidate only qualifies as a secondary/label-expansion indication if:
{SECONDARY_INDICATION_CRITERIA}

Return ONLY valid JSON - no markdown fences, no explanation:
{{
  "primary_indications": ["<the drug's current approved/primary indication(s)>"],
  "opportunities": [
    {{
      "indication": "<disease or condition>",
      "indication_type": "Primary" or "Secondary",
      "therapy_area": "<Metabolic, Cardiovascular, Oncology, Neuroscience, Immunology, Respiratory, Nephrology, Hepatology, Ophthalmology, Musculoskeletal, Gastroenterology, Infectious Disease, Dermatology, Hematology, Endocrinology, Rare Disease, or another appropriate area>",
      "rationale": "<why - cite the specific source/evidence you found>",
      "source_url": "<the URL of the source that supports this, if available>"
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
# ENTRY POINT FOR THIS MODULE
# ==============================
def analyse(drug_name: str = DRUG_NAME) -> list[dict]:
    """Runs the web-research pipeline for ``drug_name``.

    Returns a flat list of row dicts. Rows have no ``trial_id``/
    ``trial_title``/``phase`` since they aren't sourced from a
    registered trial.
    """
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
                "rationale": (opp.get("rationale") or "").strip(),
                "trial_title": None,
                "trial_id": None,
                "phase": None,
                "source_url": opp.get("source_url") or None,
                "indication_type": opp.get("indication_type", "Secondary"),
                "therapy_area": opp.get("therapy_area", "Other"),
            }
        )

    logger.info("[WEB_ANALYSER] Completed. %d row(s) for '%s'", len(flat_rows), drug_name)
    return flat_rows
