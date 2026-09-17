"""Label Expansion Opportunity — shared utilities.

Holds everything ``trial_analyser.py`` (Module 1) and ``web_analyser.py``
(Module 2) both need but that isn't specific to either one:

- the Gemini client and a resilient ``gemini_generate()`` wrapper
  (Search grounding, retries with backoff, fallback to non-grounded calls)
- ``extract_json()`` to reliably parse JSON out of a model response
- ``SECONDARY_INDICATION_CRITERIA``, the shared prompt text defining what
  counts as a genuine label-expansion indication

Neither module talks to Gemini directly outside of what's defined here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL

logger = logging.getLogger(__name__)

# ==============================
# BATCHING CONSTANTS
# ==============================
TRIALS_PER_CALL = 1
INDICATIONS_PER_CALL = 20

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
                    logger.warning("[UTILS] %s — retrying in %ss (%d/%d)", exc, delay, attempt + 1, GEMINI_MAX_RETRIES)
                    time.sleep(delay)
                elif i == 0 and len(configs) > 1:
                    logger.info("[UTILS] Error with Search grounding (%s) — trying without grounding", exc)
                    break
                else:
                    raise
        if i == 0 and len(configs) > 1:
            logger.info("[UTILS] Empty/failed response with Search grounding — retrying without grounding")

    if last_err:
        raise last_err
    return ""


def gemini_generate_with_timeout(
    prompt: str,
    *,
    system_instruction: str = "",
    use_search: bool = True,
    timeout_seconds: float,
    max_attempts: int = 2,
    log_context: str = "",
) -> str:
    """Runs ``gemini_generate`` in a background thread with a hard wall-clock
    timeout, retrying up to ``max_attempts`` times if it doesn't come back
    in time.

    Shared by any caller that batches Gemini calls (e.g. trial_analyser's
    per-batch extraction, data_fetcher's enrichment fallback) so batches
    that legitimately need more time get it, without one hung batch
    blocking a whole run.

    Raises ``TimeoutError`` if every attempt times out, or re-raises
    whatever ``gemini_generate`` itself raised.
    """
    result_holder: dict = {}
    error_holder: dict = {}
    context_suffix = f" for {log_context}" if log_context else ""

    for attempt in range(1, max_attempts + 1):
        result_holder.clear()
        error_holder.clear()

        def _call_gemini():
            try:
                result_holder["text"] = gemini_generate(
                    prompt, system_instruction=system_instruction, use_search=use_search,
                )
            except Exception as exc:  # noqa: BLE001
                error_holder["error"] = exc

        thread = threading.Thread(target=_call_gemini, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if not thread.is_alive():
            break  # got a response (success or error) within the timeout

        if attempt < max_attempts:
            logger.warning(
                "[UTILS] Timeout (>%ss) on attempt %d/%d%s - retrying",
                timeout_seconds, attempt, max_attempts, context_suffix,
            )
        else:
            logger.warning(
                "[UTILS] Timeout (>%ss) on attempt %d/%d%s - giving up",
                timeout_seconds, attempt, max_attempts, context_suffix,
            )
            raise TimeoutError(f"Gemini call timed out after {max_attempts} attempt(s){context_suffix}")

    if "error" in error_holder:
        raise error_holder["error"]

    return result_holder.get("text", "")


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


# ==============================
# SHARED PROMPT CONSTANTS
# ==============================
SECONDARY_INDICATION_CRITERIA = """
A secondary indication qualifies ONLY if ALL of the following are true:

- The indication represents a true expansion - i.e., it is NOT one of the
  drug's primary indications (for clinical assets) and is NOT part of its
  currently approved label (for commercial assets). Note: a drug can have
  MULTIPLE primary/approved indications at once (e.g. tirzepatide is
  approved for both type 2 diabetes AND obesity/weight management - both
  are Primary, regardless of which one was approved first). Do not treat
  a later-approved indication as "Secondary" just because it came after
  an earlier one - check whether it is CURRENTLY part of the approved
  label, not the order in which approvals happened.
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
