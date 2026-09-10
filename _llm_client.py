"""Shared Gemini call helper for the Label Expansion Opportunity pillar.

Kept as a small internal module so ``trial_analyser.py`` and
``web_analyser.py`` don't duplicate the same retry / JSON-extraction
logic. Not part of ``gcp_utils`` because it talks to the Gemini
generative API, not BigQuery/GCS.
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

from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL

logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=GEMINI_API_KEY)

MAX_RETRIES = 4
BASE_DELAY_SECONDS = 5


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
    """Call Gemini with optional Google Search grounding.

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
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=GEMINI_FLASH_PREVIEW_MODEL, contents=prompt, config=cfg,
                )
                text = _safe_response_text(resp)
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                if text:
                    return text
                break  # empty response — try next config
            except Exception as exc:
                last_err = exc
                if _is_transient(exc) and attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY_SECONDS * (2**attempt)
                    logger.warning("[LE_LLM] %s — retrying in %ss (%d/%d)", exc, delay, attempt + 1, MAX_RETRIES)
                    time.sleep(delay)
                elif i == 0 and len(configs) > 1:
                    logger.info("[LE_LLM] Error with Search grounding (%s) — trying without grounding", exc)
                    break
                else:
                    raise
        if i == 0 and len(configs) > 1:
            logger.info("[LE_LLM] Empty/failed response with Search grounding — retrying without grounding")

    if last_err:
        raise last_err
    return ""


def extract_json(text: str) -> dict | list:
    """Extract and parse the first JSON object or array from *text*."""
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
