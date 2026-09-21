"""Open Targets mapping — MOA → OT target resolution.

Fetches the Mechanism_of_Action for a drug from BigQuery, resolves each
MOA to its Open Targets target name using a two-phase approach:

  Phase 1 (deterministic): extract candidate gene search terms from the
      MOA string and query the OT target search API.
  Phase 2 (LLM fallback):  ask Gemini + Google Search for the Ensembl ID
      or HGNC symbol, then verify against OT.

Already-resolved MOAs (present in ``OT_MOA_TABLE``) are skipped so only
new values hit the API.

Resolved mappings are pushed to ``PROJECT_ID.BQ_DATASET_ID.OT_MOA_TABLE``
with columns ``moa`` and ``ot_moa``.
"""

from __future__ import annotations

import logging
import re

from google.cloud import bigquery

from medical_potential.config import BQ_DATASET_ID, DRUG_NAME, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client

from .ot_utils import (
    OT_MOA_TABLE,
    fetch_existing_mappings,
    gemini_call,
    ot_post,
    push_mappings,
)

logger = logging.getLogger(__name__)

# ==============================
# BQ SCHEMA
# ==============================
OT_MOA_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("moa", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ot_moa", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ensembl_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
]

# ==============================
# MOA TERM EXTRACTION
# ==============================
ACTION_WORDS = {
    "Agonist", "Antagonist", "Activator", "Inhibitor",
    "Modulator", "Blocker", "Stimulator", "Suppressor",
}


def extract_gene_search_terms(moa: str) -> list[str]:
    """Extract candidate gene/target search terms from a MOA description.

    Strategies (in order):
      1. Symbol in parentheses — ``Calcitonin Receptor (CALCR) Agonist`` → ``["CALCR"]``
      2. Hyphenated/numeric acronym — ``GLP-1 Receptor Agonist`` → ``["GLP-1"]``
      3. All-caps standalone acronym — ``GIPR Agonist`` → ``["GIPR"]``
      4. Strip action word from end → protein description
      5. Raw MOA string as last resort
    """
    terms: list[str] = []

    # Strategy 1: symbol inside parentheses
    m = re.search(r"\(([A-Z][A-Z0-9\-]+)\)", moa)
    if m:
        terms.append(m.group(1))

    # Strategy 2: hyphenated/numeric acronym
    m = re.search(r"\b([A-Z]{2,}[\-]\d+[A-Z]*)\b", moa)
    if m:
        terms.append(m.group(1))

    # Strategy 3: all-caps standalone acronym (3+ chars)
    for word in moa.split():
        clean = re.sub(r"[^A-Z0-9]", "", word)
        if re.fullmatch(r"[A-Z]{3,}\d*", clean) and word.rstrip(".,;") not in ACTION_WORDS:
            terms.append(clean)
            break

    # Strategy 4: strip action word from end
    stripped = re.sub(
        r"\s+(" + "|".join(ACTION_WORDS) + r")\s*$", "", moa, flags=re.IGNORECASE,
    ).strip()
    stripped = re.sub(r"\s*\(.*?\)", "", stripped).strip()
    if stripped and stripped != moa:
        terms.append(stripped)

    # Strategy 5: raw MOA string
    if moa not in terms:
        terms.append(moa)

    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ==============================
# OT TARGET SEARCH (Phase 1)
# ==============================
def ot_search_target(term: str) -> tuple[str | None, str | None, str | None]:
    """Query OT target search for a given term.
    Returns ``(ensembl_id, approved_symbol, approved_name)`` or ``(None, None, None)``."""
    query = """
    query SearchTarget($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 3}) {
        hits {
          id
          object { ... on Target { approvedSymbol approvedName } }
        }
      }
    }
    """
    data = ot_post(query, {"q": term}, context=f"target-search:{term}")
    if data:
        hits = data.get("search", {}).get("hits", [])
        if hits:
            h = hits[0]
            return (
                h["id"],
                h["object"].get("approvedSymbol"),
                h["object"].get("approvedName"),
            )
    return None, None, None


# ==============================
# GEMINI FALLBACK (Phase 2)
# ==============================
def _extract_ensg(text: str) -> str | None:
    m = re.search(r"ENSG\d{11}", text)
    return m.group() if m else None


def _extract_symbol(text: str) -> str | None:
    m = re.search(r"(?:symbol|gene)[:\s]+([A-Z][A-Z0-9]{1,9})\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    noise = {"ENSG", "HGNC", "URL", "API", "FDA", "EMA", "THE", "AND", "FOR"}
    for c in re.findall(r"\b([A-Z][A-Z0-9]{2,9})\b", text):
        if c not in noise and not c.startswith("ENSG"):
            return c
    return None


def gemini_resolve_moa(moa: str) -> tuple[str | None, str | None, str | None]:
    """Use Gemini + Google Search to find the primary gene target for a MOA.

    1. Ask for the Ensembl ID → verify against OT.
    2. If that fails, ask for the HGNC gene symbol → search OT.

    Returns ``(ensembl_id, approved_symbol, approved_name)`` or ``(None, None, None)``.
    """
    logger.info("[MOA_MAPPING] Gemini fallback for '%s'", moa)

    # Ask 1: ENSG ID
    prompt_ensg = (
        f'Search OpenTargets Platform (platform.opentargets.org) for the primary '
        f'gene target of the mechanism of action: "{moa}".\n'
        f'Return ONLY the Ensembl gene ID in the format ENSGXXXXXXXXXXX '
        f'(11 digits after ENSG). No other text.'
    )
    text = gemini_call(prompt_ensg, max_tokens=512)
    ensg = _extract_ensg(text)
    if ensg:
        eid, sym, name = ot_search_target(ensg)
        if eid:
            logger.info("[MOA_MAPPING] Gemini→OT verified: %s | %s | %s", eid, sym, name)
            return eid, sym, name
        logger.warning("[MOA_MAPPING] OT could not verify ENSG %s — trying symbol fallback", ensg)

    # Ask 2: gene symbol
    prompt_sym = (
        f'What is the official HGNC gene symbol of the primary protein target '
        f'for the drug mechanism: "{moa}"?\n'
        f'Examples: GLP1R for GLP-1 Receptor Agonist, GIPR for GIP Receptor Agonist.\n'
        f'Return ONLY the gene symbol. No other text.'
    )
    text2 = gemini_call(prompt_sym, max_tokens=512)
    symbol = _extract_symbol(text2)
    if symbol:
        eid, sym, name = ot_search_target(symbol)
        if eid:
            logger.info("[MOA_MAPPING] Gemini→Symbol→OT: %s | %s | %s", eid, sym, name)
            return eid, sym, name
        logger.warning("[MOA_MAPPING] OT could not find symbol '%s'", symbol)

    logger.warning("[MOA_MAPPING] Gemini fallback exhausted for '%s'", moa)
    return None, None, None


# ==============================
# MASTER RESOLVER
# ==============================
def resolve_single_moa(moa: str) -> dict:
    """Resolve one MOA string to an OT target via Phase 1 then Phase 2.

    Returns a dict with keys: ``moa``, ``ot_moa``, ``ensembl_id``,
    ``approved_symbol``, ``resolution_path``.
    """
    search_terms = extract_gene_search_terms(moa)
    logger.info("[MOA_MAPPING] Resolving '%s' — search terms: %s", moa, search_terms)

    # Phase 1: deterministic OT search
    for term in search_terms:
        ensembl_id, symbol, name = ot_search_target(term)
        if ensembl_id:
            logger.info("[MOA_MAPPING] OT search hit for '%s' → %s | %s | %s", moa, ensembl_id, symbol, name)
            return {
                "moa": moa,
                "ot_moa": name,
                "ensembl_id": ensembl_id,
                "approved_symbol": symbol,
                "resolution_path": "ot_search",
            }

    # Phase 2: Gemini fallback
    ensembl_id, symbol, name = gemini_resolve_moa(moa)
    if ensembl_id:
        return {
            "moa": moa,
            "ot_moa": name,
            "ensembl_id": ensembl_id,
            "approved_symbol": symbol,
            "resolution_path": "gemini_fallback",
        }

    logger.warning("[MOA_MAPPING] Unresolved: '%s'", moa)
    return {
        "moa": moa,
        "ot_moa": None,
        "ensembl_id": None,
        "approved_symbol": None,
        "resolution_path": "unresolved",
    }


# ==============================
# FETCH MOA FROM BQ
# ==============================
def fetch_moa_for_drug(drug_name: str, drug_details_table: str) -> list[str]:
    """Fetches the distinct Mechanism_of_Action values for a drug from BQ.

    Returns a list of individual MOA strings (split on ``'; '``).
    """
    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{drug_details_table}"

    query = f"""
        SELECT
            Cleaned_Generic_Name,
            STRING_AGG(DISTINCT Mechanism_of_Action, '; ') AS Mechanisms_Of_Action
        FROM `{table_id}`
        WHERE Cleaned_Generic_Name = @drug_name
        GROUP BY Cleaned_Generic_Name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name),
        ]
    )
    results = bq_client.query(query, job_config=job_config).result()

    moas: list[str] = []
    for row in results:
        raw = row.get("Mechanisms_Of_Action") or ""
        for part in raw.split("; "):
            part = part.strip()
            if part:
                moas.append(part)

    logger.info("[MOA_MAPPING] Fetched %d MOA(s) for '%s': %s", len(moas), drug_name, moas)
    return moas


# ==============================
# ENTRY POINT
# ==============================
def run_moa_mapping(drug_name: str = DRUG_NAME, drug_details_table: str = "drug_details") -> list[dict]:
    """Full MOA mapping pipeline for one drug.

    1. Fetch MOAs from BQ drug_details table.
    2. Check which MOAs are already resolved in ``OT_MOA_TABLE``.
    3. Resolve only the new ones (Phase 1 + Phase 2).
    4. Push new mappings to ``OT_MOA_TABLE``.
    5. Return all resolved mappings (existing + new).
    """
    logger.info("[MOA_MAPPING] Starting MOA mapping for '%s'", drug_name)

    # Step 1: Fetch MOAs
    moas = fetch_moa_for_drug(drug_name, drug_details_table)
    if not moas:
        logger.warning("[MOA_MAPPING] No MOAs found for '%s' — nothing to resolve", drug_name)
        return []

    # Step 2: Check existing mappings
    existing = fetch_existing_mappings(OT_MOA_TABLE, "moa")
    new_moas = [m for m in moas if m.strip().lower() not in existing]
    logger.info(
        "[MOA_MAPPING] %d MOA(s) total, %d already resolved, %d new to resolve",
        len(moas), len(moas) - len(new_moas), len(new_moas),
    )

    # Step 3: Resolve new MOAs
    new_mappings: list[dict] = []
    for moa in new_moas:
        result = resolve_single_moa(moa)
        new_mappings.append({
            "moa": result["moa"],
            "ot_moa": result["ot_moa"],
            "ensembl_id": result["ensembl_id"],
        })

    # Step 4: Push new mappings to BQ
    push_mappings(OT_MOA_TABLE, OT_MOA_SCHEMA, new_mappings, key_column="moa")

    # Step 5: Return all mappings (existing + new)
    all_mappings: list[dict] = []
    for moa in moas:
        key = moa.strip().lower()
        if key in existing:
            all_mappings.append({
                "moa": moa,
                "ot_moa": existing[key].get("ot_moa"),
                "ensembl_id": existing[key].get("ensembl_id"),
            })
        else:
            match = next((m for m in new_mappings if m["moa"] == moa), None)
            all_mappings.append(match or {"moa": moa, "ot_moa": None, "ensembl_id": None})

    logger.info("[MOA_MAPPING] Completed. %d mapping(s) for '%s'", len(all_mappings), drug_name)
    return all_mappings
