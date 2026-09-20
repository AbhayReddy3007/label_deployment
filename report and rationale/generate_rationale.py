"""Prompt-based rationale generator for label expansion opportunity scoring.

Mirrors ``serious_safety_profile``'s ``ssp_rationale.py`` pattern (reduce
input -> build a short prompt -> call Gemini -> return text + payload), but
uses this pipeline's own ``gemini_generate`` helper (already handles
retries/backoff - see ``indication_extractor/utils.py``) instead of a raw
``genai.Client`` call, and is a plain synchronous function rather than
``async``, matching how the rest of this pipeline calls Gemini (no
``asyncio`` is used anywhere else in this codebase - parallelism here is
done with ``ThreadPoolExecutor`` instead).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..indication_extractor.utils import gemini_generate

logger = logging.getLogger(__name__)

NOT_GENERATED_MESSAGE = "Rationale has not been generated."

# How many of the drug's highest-scoring opportunities to include in the
# prompt payload - keeps the prompt short, same idea as ssp_rationale.py's
# top_trials[:10].
_TOP_N_OPPORTUNITIES = 10


def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Select and reduce input data to the minimal fields needed for a
    concise rationale.

    ``data`` is expected to be the dict ``label_expansion()`` returns for
    one drug (see ``label_expansion_opportunity.py``):
    ``{"drug_name": ..., "score_rows": [LE_SCORE_CALCULATION_TABLE row, ...]}``.
    Each row in ``score_rows`` is one TA-I (therapy_area + OT disease)
    opportunity, as pushed by ``scoring/score_calculator.py``.

    Returns a small dictionary suitable for embedding in the rationale prompt.
    """
    if not isinstance(data, dict):
        return {}

    drug_name = data.get("drug_name")
    score_rows = data.get("score_rows") or []

    # Condense each opportunity to the fields that actually explain WHY it
    # scored the way it did, without exposing internal field names in the
    # prompt output itself (that's enforced in the prompt instructions below).
    opportunities = []
    for row in score_rows:
        if not isinstance(row, dict):
            continue
        try:
            opportunities.append({
                "indication": row.get("indication"),
                "therapy_area": row.get("therapy_area"),
                "ot_disease_name": row.get("ot_disease_name"),
                "final_score": row.get("final_score"),
                "phase": row.get("phase"),
                "association_score": row.get("association_score"),
                "link": row.get("link"),
                "maturity_weight": row.get("maturity_weight"),
                "trial_weight": row.get("trial_weight"),
            })
        except Exception:
            continue

    try:
        sorted_opportunities = sorted(
            opportunities, key=lambda x: float(x.get("final_score") or 0), reverse=True
        )
        top_opportunities = sorted_opportunities[:_TOP_N_OPPORTUNITIES]
    except Exception:
        top_opportunities = opportunities[:_TOP_N_OPPORTUNITIES]

    scores = [o.get("final_score") for o in opportunities if o.get("final_score") is not None]

    return {
        "drug_name": drug_name,
        "num_opportunities": len(opportunities),
        "top_opportunities": top_opportunities,
        "score_range": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        },
    }


def generate_label_expansion_rationale(
    data: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Generate a concise qualitative rationale for a drug's label
    expansion opportunities.

    Returns ``(rationale_text, payload)`` - same shape as
    ``ssp_rationale.generate_serious_safety_rationale``. This function does
    NOT persist anything itself; the caller is responsible for storing the
    rationale (e.g. as a column alongside the drug's score rows) - matching
    how ``ssp_rationale.py`` also just returns a tuple rather than writing
    to BigQuery itself.

    Note: unlike ``ssp_rationale.py``, there is no ``model_name`` parameter
    here - ``gemini_generate`` (this pipeline's shared Gemini call helper)
    always uses ``GEMINI_FLASH_PREVIEW_MODEL`` internally and doesn't accept
    a model override, so exposing one here would be misleading.
    """
    payload = _prepare_prompt_payload(data)

    if not payload.get("top_opportunities"):
        logger.warning(
            "[LABEL_EXPANSION][RATIONALE] No score rows for '%s' - nothing to explain",
            payload.get("drug_name"),
        )
        return NOT_GENERATED_MESSAGE, payload

    prompt = f"""
You are a concise medical insights writer. Produce a short, plain-text rationale (one sentence or a very short paragraph) that explains the main label expansion opportunities for this drug and what is driving their scores.

Input (JSON):
{json.dumps(payload, indent=2)}

Requirements:
- Explain only the primary indications driving the highest scores (e.g. strong clinical trial evidence, advanced trial phase, a strong known link between the drug's target and the disease).
- Mention uncertainty when evidence is limited (few trials, early phase, weak target-disease link).
- Avoid methodology, scoring framework names, internal field names (e.g. "final_score", "link", "prior", "TA-I", "maturity_weight"), or tables.
- Do not mention or cite any specific trial ID.
- Use plain language appropriate for a Medical Affairs / business development audience.
- Return plain text only.

Note:
Strictly limit to 50 words, anything longer will be penalised
""".strip()

    try:
        rationale = gemini_generate(
            prompt,
            system_instruction="You are a concise medical insights writer. Return plain text only.",
            use_search=False,
        ).strip()
        if rationale:
            return rationale, payload
        logger.warning(
            "[LABEL_EXPANSION][RATIONALE] Prompt output was empty for '%s'; rationale was not generated.",
            payload.get("drug_name"),
        )
    except Exception:
        logger.exception(
            "[LABEL_EXPANSION][RATIONALE] Prompt generation failed for '%s'; rationale was not generated.",
            payload.get("drug_name"),
        )

    return NOT_GENERATED_MESSAGE, payload
