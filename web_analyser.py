"""Label Expansion Opportunity — Module 2: web_analyser.

Uses Gemini + Google Search grounding to look directly at public web
sources (regulatory filings, company pipeline pages, conference
readouts, news) for label-expansion signals for the configured drug -
independent of, and complementary to, the registered-trial evidence
mined by ``trial_analyser``. This catches expansion signals that are
public but not yet reflected as a registered clinical trial.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from medical_potential.config import DRUG_NAME, GEMINI_FLASH_PREVIEW_MODEL
from medical_potential.label_expansion_opportunity.trial_analyser import (
    SECONDARY_INDICATION_CRITERIA,
)

logger = logging.getLogger(__name__)

# ==============================
# GEMINI CLIENT
# ==============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MAX_RETRIES = 4
GEMINI_BASE_DELAY_SECONDS = 5


def _safe_response_text(resp) -> str:
    """Safely extract text from a Gemini response.

    With Google Search grounding the simple ``resp.text`` property can be
    None or raise, so this walks the full candidates -> parts tree.
    """
    try:
        if resp.text is not None:
            return resp.text.strip()
    except Exception:
        pass

    texts: list[str] = []
    try:
        for candidate in resp.candidates or []:
            try:
                parts = candidate.content.parts
            except Exception:
                continue
            for part in parts or []:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text.strip())
    except Exception:
        pass

    return "\n".join(texts)


def _is_transient(exc: Exception) -> bool:
    err = str(exc).lower()
    return any(
        k in err
        for k in (
            "503", "429", "unavailable", "overloaded", "resource exhausted",
            "rate limit", "deadline exceeded", "connection", "timeout", "502", "500",
        )
    )


def gemini_generate(prompt: str, *, system_instruction: str = "", use_search: bool = True) -> str:
    """Calls Gemini with optional Google Search grounding.

    Retries transient errors with exponential backoff. If search grounding
    returns nothing, falls back to a plain (non-grounded) call.
    """
    configs = []
    if use_search:
        configs.append(
            types.GenerateContentConfig(
                temperature=0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=system_instruction or "Return ONLY valid JSON.",
            )
        )
    configs.append(
        types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_instruction or "Return ONLY valid JSON.",
        )
    )

    last_err: Exception | None = None
    for i, cfg in enumerate(configs):
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                resp = gemini_client.models.generate_content(
                    model=GEMINI_FLASH_PREVIEW_MODEL, contents=prompt, config=cfg,
                )
                text = _safe_response_text(resp)
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                if text:
                    return text
                break  # empty response — try next config
            except Exception as exc:
                last_err = exc
                if _is_transient(exc) and attempt < GEMINI_MAX_RETRIES - 1:
                    delay = GEMINI_BASE_DELAY_SECONDS * (2**attempt)
                    logger.warning("[WEB_ANALYSER] %s — retrying in %ss (%d/%d)", exc, delay, attempt + 1, GEMINI_MAX_RETRIES)
                    time.sleep(delay)
                elif i == 0 and len(configs) > 1:
                    logger.info("[WEB_ANALYSER] Error with Search grounding (%s) — trying without grounding", exc)
                    break
                else:
                    raise
        if i == 0 and len(configs) > 1:
            logger.info("[WEB_ANALYSER] Empty/failed response with Search grounding — retrying without grounding")

    if last_err:
        raise last_err
    return ""


def extract_json(text: str) -> dict | list:
    """Extracts and parses the first JSON object or array from *text*."""
    if not text:
        raise ValueError("Cannot extract JSON from empty text")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        if start == -1:
            continue
        depth, end, in_str = 0, start, False
        while end < len(stripped):
            ch = stripped[end]
            if ch == '"' and (end == 0 or stripped[end - 1] != "\\"):
                in_str = not in_str
            elif not in_str:
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(stripped[start : end + 1])
                        except json.JSONDecodeError:
                            break
            end += 1

    raise ValueError(f"No valid JSON found in response (first 200 chars): {text[:200]}")


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
    """Runs the web-research pipeline for exactly one drug.

    ``drug_name`` must be a single drug name (str) - not a list. To
    analyse multiple drugs, call this once per drug from the caller.

    Returns a flat list of row dicts. Rows have no ``trial_id``/
    ``trial_title``/``phase`` since they aren't sourced from a
    registered trial.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise TypeError(
            f"web_analyser.analyse() accepts exactly one drug name (str), got: {drug_name!r}"
        )

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
