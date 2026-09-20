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
# PIPELINE MODE TOGGLE
# ==============================
# False (default): run the full pipeline from scratch - trial_analyser
# fetches trials and extracts indications via Gemini + Google Search per
# trial, fda_fetcher fetches FDA brands and label text and extracts
# disease names, then both classify what they found. This is the
# expensive path (one search-grounded Gemini call per trial/brand).
#
# True: skip re-discovery and re-extraction entirely. Instead,
# trial_analyser, fda_fetcher, and web_analyser pull the indications
# already sitting in LE_TABLE for the drug and only re-run classification
# (indication_type / therapy_area / ot_disease_name) on them - one (or a
# few) batched Gemini call(s) over the unique indications, not one call
# per trial/brand or one open-ended web-research pass. Use this after a
# classification prompt/logic change (e.g. fixing how Primary/Secondary
# is decided) when you want existing rows re-classified with the fixed
# logic without paying to re-discover indications that are already
# correctly identified.
PROCESS_INDICATIONS = False

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
# On an empty (non-erroring) response, retry the SAME config this many
# times in total before giving up on it (2 = one retry, as instructed).
# Bounded well under GEMINI_MAX_RETRIES since transient-error backoff and
# empty-response retry are different concerns with different costs.
GEMINI_EMPTY_RETRY_ATTEMPTS = 2


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


def gemini_generate(
    prompt: str,
    *,
    system_instruction: str = "",
    use_search: bool = True,
    allow_ungrounded_fallback: bool = True,
) -> str:
    """Calls Gemini with optional Google Search grounding.

    Retries transient errors with exponential backoff. An empty (but
    non-erroring) response is retried once more with the SAME config
    before moving on - e.g. an empty grounded response is retried with
    grounding still in place, not immediately treated as failed or
    switched to a different config. Only after that retry is also empty
    does this move to the next config (falling back to a plain,
    non-grounded call) - unless ``allow_ungrounded_fallback`` is False,
    in which case an empty/failed grounded response is left empty rather
    than retried without grounding.
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
    if not use_search or allow_ungrounded_fallback:
        configs.append(
            types.GenerateContentConfig(
                temperature=0,
                system_instruction=system_instruction or "Return ONLY valid JSON.",
            )
        )

    last_err: Exception | None = None
    for i, cfg in enumerate(configs):
        config_label = "Search grounding" if i == 0 and use_search else "config"
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                resp = gemini_client.models.generate_content(
                    model=GEMINI_FLASH_PREVIEW_MODEL, contents=prompt, config=cfg,
                )
                text = _safe_response_text(resp)
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                if text:
                    return text
                # Empty (non-erroring) response - retry the SAME config
                # once before giving up on it.
                if attempt < GEMINI_EMPTY_RETRY_ATTEMPTS - 1:
                    logger.warning(
                        "[UTILS] Empty response with %s (attempt %d/%d) — retrying same config",
                        config_label, attempt + 1, GEMINI_EMPTY_RETRY_ATTEMPTS,
                    )
                    continue
                break  # empty-response retries exhausted — try next config (if any)
            except Exception as exc:
                last_err = exc
                if _is_transient(exc) and attempt < GEMINI_MAX_RETRIES - 1:
                    delay = GEMINI_BASE_DELAY_SECONDS * (2**attempt)
                    logger.warning("[UTILS] %s — retrying in %ss (%d/%d)", exc, delay, attempt + 1, GEMINI_MAX_RETRIES)
                    time.sleep(delay)
                elif i == 0 and len(configs) > 1:
                    logger.info("[UTILS] Error with Search grounding (%s) — trying without grounding", exc)
                    break
                elif i == 0 and not allow_ungrounded_fallback:
                    logger.warning(
                        "[UTILS] Error with Search grounding (%s) and ungrounded fallback disabled - leaving empty",
                        exc,
                    )
                    return ""
                else:
                    raise
        if i == 0 and len(configs) > 1:
            logger.info("[UTILS] Empty/failed response with Search grounding — retrying without grounding")
        elif i == 0 and not allow_ungrounded_fallback:
            logger.info("[UTILS] Empty response with Search grounding and ungrounded fallback disabled - leaving empty")
            return ""

    if last_err and allow_ungrounded_fallback:
        raise last_err
    return ""


def gemini_generate_with_timeout(
    prompt: str,
    *,
    system_instruction: str = "",
    use_search: bool = True,
    allow_ungrounded_fallback: bool = True,
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

    ``allow_ungrounded_fallback`` is forwarded to ``gemini_generate`` as-is
    (see there): set it to ``False`` to leave an empty/failed grounded
    response empty instead of retrying it without Search grounding.

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
                    prompt,
                    system_instruction=system_instruction,
                    use_search=use_search,
                    allow_ungrounded_fallback=allow_ungrounded_fallback,
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
# SCORABLE-INDICATION FILTER (shared safety net)
# ==============================
# Used by the main orchestrator, after merging trial_analyser/fda_fetcher/
# web_analyser output, to drop rows whose "indication" is actually a trial
# endpoint, biomarker, PK parameter, or procedure rather than a genuine
# disease/condition. These can never become an approved label indication,
# so they shouldn't be scored as label-expansion opportunities. This is a
# safety net for cases the extraction prompts themselves didn't already
# exclude.
SCORABLE_BATCH_SIZE = 20


def _classify_scorable_batch(indications: list[str]) -> dict[str, bool]:
    """One Gemini call: classifies each indication as scorable (a genuine
    disease/condition) or not (an endpoint/biomarker/PK parameter/
    procedure). Returns ``{indication_lower: is_scorable}``."""
    if not indications:
        return {}

    ind_list = "\n".join(f"{i + 1}. {ind}" for i, ind in enumerate(indications))
    prompt = f"""You are a pharmaceutical analyst.

Below is a numbered list of strings extracted from clinical trial and FDA
label data. For EACH one, decide whether it is a disease or medical
condition that could plausibly appear as an FDA-approved drug indication
(scorable), or whether it is something else that should NOT be scored as
an indication - a trial endpoint or outcome measure, a biomarker or lab
value, a pharmacokinetic parameter, a physiological measurement, or a
procedure/intervention (not scorable).

Examples of NOT scorable: "Exercise Capacity", "Waist Circumference",
"Postprandial Glucose", "Pharmacokinetics", "Lipid Profile",
"Hepatocyte Ballooning", "Bariatric Surgery".
Examples of scorable: "Type 2 Diabetes", "Obesity", "Heart Failure".

Items:
{ind_list}

Return ONLY a JSON array, one object per item, in the same order:
[
  {{"item": "<exact item text>", "scorable": true or false}}
]
"""
    try:
        text = gemini_generate(
            prompt,
            system_instruction=(
                "Classify each item as a scorable disease/condition or not. "
                "Return ONLY valid JSON."
            ),
            use_search=False,
        )
        parsed = extract_json(text)
        entries = parsed if isinstance(parsed, list) else parsed.get("items", []) if isinstance(parsed, dict) else []
        result: dict[str, bool] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            item = (e.get("item") or "").strip().lower()
            if item:
                result[item] = bool(e.get("scorable", True))
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[UTILS] Scorable classification failed for batch %s: %s", indications, exc)
        # Fail open: on error, keep everything rather than silently
        # dropping potentially-valid indications.
        return {ind.lower(): True for ind in indications}


def filter_scorable_indications(rows: list[dict]) -> list[dict]:
    """Drops rows whose ``indication`` is a trial endpoint, biomarker,
    pharmacokinetic parameter, or procedure rather than an actual
    disease/condition.

    Runs one batched Gemini call per ``SCORABLE_BATCH_SIZE`` unique
    indications (not per row) to keep cost down, then filters ``rows``
    using the result. On any classification failure, the affected
    indications are kept (fail open) rather than silently dropped.
    """
    if not rows:
        return rows

    unique_indications = sorted({
        (r.get("indication") or "").strip()
        for r in rows
        if (r.get("indication") or "").strip()
    })
    if not unique_indications:
        return rows

    batches = [
        unique_indications[i : i + SCORABLE_BATCH_SIZE]
        for i in range(0, len(unique_indications), SCORABLE_BATCH_SIZE)
    ]

    scorable_map: dict[str, bool] = {}
    for batch in batches:
        scorable_map.update(_classify_scorable_batch(batch))

    kept: list[dict] = []
    dropped: list[str] = []
    for r in rows:
        ind = (r.get("indication") or "").strip().lower()
        if not ind or scorable_map.get(ind, True):
            kept.append(r)
        else:
            dropped.append(r.get("indication"))

    if dropped:
        logger.info(
            "[UTILS] Filtered out %d row(s) with non-scorable indication(s): %s",
            len(dropped), sorted(set(dropped)),
        )
    return kept
