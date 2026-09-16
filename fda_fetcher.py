"""Label Expansion Opportunity — Module 3: fda_fetcher.

Fetches FDA-approved indications for a drug by querying the openFDA
Drugs@FDA and Drug Label APIs, then using Gemini to extract clean,
structured disease/condition names from the raw label text.

All FDA-sourced rows get:
    - ``phase``: ``"Approved"`` (FDA-approved = post-market)
    - ``trial_id``: ``"<indication> + fda"``
    - ``data_source``: ``"Trials"``
    - ``indication_type``: ``"Primary"`` (FDA-approved indications are
      by definition the drug's primary/approved uses)

After extraction, a single Gemini call classifies each indication's
``therapy_area`` and marks any that are genuinely secondary (label
expansion) vs primary.

Ported from the reference ``fda_indications_gemini.py``.
"""

from __future__ import annotations

import logging
import re

import requests

from medical_potential.config import DRUG_NAME

from .utils import extract_json, gemini_generate

logger = logging.getLogger(__name__)

# ==============================
# FDA API ENDPOINTS
# ==============================
DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"
LABEL_URL = "https://api.fda.gov/drug/label.json"

FDA_LIMIT = 100

# ==============================
# FDA API HELPERS
# ==============================
def _fetch_json(url: str, params: dict) -> list:
    try:
        resp = requests.get(url, params=params, timeout=15)
    except requests.RequestException as exc:
        logger.warning("[FDA_FETCHER] Request error: %s", exc)
        return []
    if resp.status_code == 200:
        return resp.json().get("results", [])
    if resp.status_code == 404:
        return []
    logger.warning("[FDA_FETCHER] API returned status %s", resp.status_code)
    return []


def _find_brands(drug_name: str, limit: int = FDA_LIMIT) -> list[dict]:
    """Finds all FDA brand entries for a drug via Drugs@FDA."""
    upper = drug_name.upper()
    queries = [
        f'products.active_ingredients.name:"{upper}"',
        f'openfda.substance_name:"{upper}"',
        f'openfda.generic_name:"{drug_name}"',
        f'openfda.brand_name:"{drug_name}"',
    ]
    seen: set[tuple[str, str]] = set()
    brands: list[dict] = []
    for q in queries:
        results = _fetch_json(DRUGSFDA_URL, {"search": q, "limit": min(limit, 1000)})
        for app in results:
            app_number = app.get("application_number", "N/A")
            for product in app.get("products", []):
                brand = product.get("brand_name", "N/A")
                key = (brand.upper(), app_number)
                if key not in seen:
                    seen.add(key)
                    brands.append({"brand": brand, "app_number": app_number})

    logger.info("[FDA_FETCHER] Found %d unique brand(s) for '%s'", len(brands), drug_name)
    return brands


def _fetch_raw_indications(brand_name: str, app_number: str) -> list[str]:
    """Fetches raw indications_and_usage text(s) from the FDA label endpoint."""
    bare_app = re.sub(r"^[A-Za-z]+", "", app_number).strip()
    queries = [
        f'openfda.brand_name:"{brand_name}"',
        f'openfda.application_number:"{bare_app}"' if bare_app else None,
    ]
    seen_texts: set[str] = set()
    indications: list[str] = []
    for q in queries:
        if not q:
            continue
        results = _fetch_json(LABEL_URL, {"search": q, "limit": 10})
        for record in results:
            for text in record.get("indications_and_usage", []):
                text = text.strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    indications.append(text)
    return indications


# ==============================
# GEMINI EXTRACTION
# ==============================
_EXTRACTION_SYSTEM = """You are a precise medical text extractor.
Your ONLY job is to extract approved indications from the raw FDA label text provided to you.

STRICT RULES:
- Use ONLY the text provided. Do NOT use any external knowledge.
- Do NOT infer, assume, or add anything not explicitly stated in the text.
- List each distinct approved indication as a separate numbered item.
- If no clear indication is found in the text, respond with exactly: NOT FOUND
- Do not add any preamble, explanation, or closing remarks.

Output format:
Approved Indications:
1. <indication>
2. <indication>
..."""


def _extract_per_brand(brand: str, raw_texts: list[str]) -> str:
    """Sends raw FDA label text for one brand to Gemini for extraction."""
    if not raw_texts:
        return "NOT FOUND"

    combined = "\n\n--- LABEL RECORD SEPARATOR ---\n\n".join(raw_texts)
    prompt = (
        f"Below is the raw FDA label text for the brand '{brand}'.\n"
        f"Extract all approved indications strictly from this text.\n\n"
        f"--- BEGIN FDA LABEL TEXT ---\n{combined}\n--- END FDA LABEL TEXT ---"
    )
    try:
        text = gemini_generate(
            _EXTRACTION_SYSTEM + "\n\n" + prompt,
            system_instruction="Extract approved indications from FDA label text. Return structured text only.",
            use_search=False,
        )
        return text.strip()
    except Exception as exc:
        logger.warning("[FDA_FETCHER] Gemini extraction failed for brand '%s': %s", brand, exc)
        return f"Gemini API error: {exc}"


def _parse_indications_from_gemini(gemini_text: str) -> list[str]:
    """Parses numbered indication lines from Gemini's structured output."""
    if not gemini_text or "NOT FOUND" in gemini_text or "error" in gemini_text.lower():
        return []

    indications: list[str] = []
    in_section = False
    for line in gemini_text.splitlines():
        stripped = line.strip()
        if re.match(r"^approved indications\s*:", stripped, re.IGNORECASE):
            in_section = True
            continue
        if re.match(r"^limitations of use\s*:", stripped, re.IGNORECASE):
            in_section = False
            continue
        if in_section and stripped:
            m = re.match(r"^[\d]+[.)]\s+(.+)$", stripped) or re.match(r"^[-*]\s+(.+)$", stripped)
            if m:
                indications.append(m.group(1).strip())
    return indications


_DISEASE_EXTRACTION_PROMPT = """You are a precise medical text extractor.
Given the following list of FDA-approved indication statements, extract ONLY
the unique disease or condition names being treated or managed.

STRICT RULES:
- Return ONLY unique disease/condition names, one per line.
- Do NOT include treatment context (e.g. "adjunct to diet", "reduce risk of").
- Do NOT include population qualifiers (e.g. "in adults", "in pediatric patients").
- Remove all duplicates.
- Use standard, recognised medical terminology.
- Return ONLY the disease/condition names — no numbering, no bullets, no preamble.
"""


def _extract_disease_names(all_indications: list[str]) -> list[str]:
    """Single Gemini call: deduplicates and extracts clean disease names."""
    if not all_indications:
        return []

    indications_text = "\n".join(f"- {ind}" for ind in all_indications)
    prompt = f"{_DISEASE_EXTRACTION_PROMPT}\n--- INDICATIONS ---\n{indications_text}\n--- END ---"
    try:
        text = gemini_generate(
            prompt,
            system_instruction="Extract unique disease/condition names. Return one per line, nothing else.",
            use_search=False,
        )
        return [line.strip() for line in text.strip().splitlines() if line.strip()]
    except Exception as exc:
        logger.warning("[FDA_FETCHER] Gemini disease extraction failed: %s", exc)
        return []


# ==============================
# CLASSIFY THERAPY AREA
# ==============================
def _classify_indications(drug_name: str, disease_names: list[str]) -> list[dict]:
    """Classifies each FDA indication's therapy_area and indication_type
    (Primary vs Secondary) via a single Gemini call."""
    if not disease_names:
        return []

    diseases_list = "\n".join(f"- {d}" for d in disease_names)
    prompt = f"""You are a pharmaceutical analyst.

The drug "{drug_name}" has the following FDA-approved indications:
{diseases_list}

For each indication, determine:
1. therapy_area: Metabolic, Cardiovascular, Oncology, Neuroscience, Immunology,
   Respiratory, Nephrology, Hepatology, Ophthalmology, Musculoskeletal,
   Gastroenterology, Infectious Disease, Dermatology, Hematology, Endocrinology,
   Rare Disease, or another appropriate area.
2. indication_type: "Primary" if this is one of the drug's original/main approved
   indications, or "Secondary" if it represents a label expansion (a later addition
   to the drug's approved uses).

Return ONLY a JSON array:
[
  {{"indication": "<disease>", "therapy_area": "<area>", "indication_type": "Primary" or "Secondary"}}
]
"""
    try:
        text = gemini_generate(
            prompt,
            system_instruction="Classify FDA indications by therapy area and primary/secondary status. Return ONLY valid JSON.",
            use_search=True,
        )
        parsed = extract_json(text)
        entries = parsed if isinstance(parsed, list) else parsed.get("indications", []) if isinstance(parsed, dict) else []
        return [e for e in entries if isinstance(e, dict)]
    except Exception as exc:
        logger.warning("[FDA_FETCHER] Gemini classification failed: %s", exc)
        return []


# ==============================
# ENTRY POINT
# ==============================
def analyse(drug_name: str = DRUG_NAME) -> list[dict]:
    """Fetches FDA-approved indications for one drug.

    Returns a flat list of row dicts, one per unique disease/condition,
    with ``phase = "Approved"``, ``data_source = "Trials"``, and
    ``trial_id = "<indication> + fda"``.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise TypeError(
            f"fda_fetcher.analyse() accepts exactly one drug name (str), got: {drug_name!r}"
        )

    logger.info("[FDA_FETCHER] Starting FDA indication fetch for '%s'", drug_name)

    # Step 1: Find all brands
    brands = _find_brands(drug_name)
    if not brands:
        logger.warning("[FDA_FETCHER] No FDA brands found for '%s'", drug_name)
        return []

    # Step 2: Fetch raw label text and extract per brand
    all_indication_statements: list[str] = []
    seen: set[str] = set()
    for entry in brands:
        raw_texts = _fetch_raw_indications(entry["brand"], entry["app_number"])
        if not raw_texts:
            continue
        gemini_result = _extract_per_brand(entry["brand"], raw_texts)
        for ind in _parse_indications_from_gemini(gemini_result):
            key = " ".join(ind.lower().split())
            if key not in seen:
                seen.add(key)
                all_indication_statements.append(ind)

    logger.info(
        "[FDA_FETCHER] %d unique indication statement(s) across %d brand(s)",
        len(all_indication_statements), len(brands),
    )

    # Step 3: Extract clean disease names
    disease_names = _extract_disease_names(all_indication_statements)
    if not disease_names:
        logger.warning("[FDA_FETCHER] No disease names extracted for '%s'", drug_name)
        return []

    logger.info("[FDA_FETCHER] %d unique disease/condition name(s) extracted", len(disease_names))

    # Step 4: Classify therapy area and indication type
    classifications = _classify_indications(drug_name, disease_names)
    classified_map = {
        (c.get("indication") or "").strip().lower(): c
        for c in classifications
        if isinstance(c, dict)
    }

    # Step 5: Build output rows
    flat_rows: list[dict] = []
    for disease in disease_names:
        classification = classified_map.get(disease.lower(), {})
        flat_rows.append({
            "drug_name": drug_name,
            "indication": disease,
            "indication_type": classification.get("indication_type", "Primary"),
            "therapy_area": classification.get("therapy_area", "Other"),
            "rationale": "FDA-approved indication",
            "trial_title": None,
            "trial_id": f"{disease} + fda",
            "phase": "Approved",
            "dosage": None,
            "trial_size": None,
            "trial_location": None,
            "source_url": None,
            "data_source": "Trials",
        })

    logger.info("[FDA_FETCHER] Completed. %d row(s) for '%s'", len(flat_rows), drug_name)
    return flat_rows
