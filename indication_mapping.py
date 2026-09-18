"""Open Targets mapping — Indication → OT disease resolution.

Fetches indications from the Label Expansion ``LE_TABLE`` in BigQuery,
resolves each to its Open Targets disease entity (EFO/MONDO ID + name)
using a multi-path approach:

  Path A (Gemini semantic matching): if ``target_ensembl_ids`` are
      provided, fetches the full OT disease list for those targets and
      asks Gemini to semantically match indications against it.
  Path B (OT text search fallback): progressive search with synonym
      expansion, parenthetical stripping, and term shortening.
      Includes retry-with-backoff for rate-limited responses.
  Path C (Gemini + Google Search grounding): web-search-grounded
      resolution for anything Path A and B miss, with OT API
      verification of returned IDs to reject hallucinations.

Already-resolved indications (present in ``OT_DISEASE_TABLE``) are
skipped so only new values hit the API.

Resolved mappings are pushed to ``PROJECT_ID.BQ_DATASET_ID.OT_DISEASE_TABLE``
with columns ``indication``, ``ot_disease``, ``ot_disease_id``, and
``resolution_path`` (tracks which path resolved each mapping for auditing).
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import bigquery

from medical_potential.config import BQ_DATASET_ID, DRUG_NAME, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client

from ..bq_utils import LE_TABLE
from ..indication_extractor.utils import extract_json as extract_json_utils, gemini_generate

from .ot_utils import (
    OT_DISEASE_TABLE,
    fetch_existing_mappings,
    gemini_call,
    ot_post,
    parse_json_response,
    push_mappings,
)

logger = logging.getLogger(__name__)

# ==============================
# BQ SCHEMA
# ==============================
OT_DISEASE_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("indication", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ot_disease", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ot_disease_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("resolution_path", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
]

# ==============================
# CONSTANTS
# ==============================
_DISEASE_CHUNK_SIZE = 100
_INDICATION_BATCH_SIZE = 15  # max indications per Gemini Path A call
_WORKERS = 3
_OT_SEARCH_PAGE_SIZE = 15  # was 5 — larger window catches better matches
_OT_MAX_RETRIES = 3        # retry OT API calls with exponential backoff

_MEASUREMENT_PREFIXES = ("measurement", "process", "risk measurement", "risk factor")

# Valid Open Targets disease ID prefixes — these are real disease entities
# on the OT Platform.  HP_ (Human Phenotype Ontology) and GO_ (Gene
# Ontology) terms show up in OT API search results but are NOT disease
# pages on the platform and can't be found by users.
_VALID_DISEASE_PREFIXES = ("EFO_", "MONDO_", "Orphanet_", "OTAR_")
_NON_DISEASE_PREFIXES = ("HP_", "GO_")


def _is_valid_disease_id(disease_id: str | None) -> bool:
    """Return True if the ID is a proper OT disease entity (EFO/MONDO/Orphanet).

    HP_ (phenotype) and GO_ (gene ontology) IDs exist in the OT API but
    are NOT disease entries — they can't be found on the OT Platform UI
    and shouldn't be used as disease mappings.
    """
    if not disease_id:
        return False
    return disease_id.startswith(_VALID_DISEASE_PREFIXES)

MEDICAL_SYNONYMS: dict[str, list[str]] = {
    # --- Cardiovascular ---
    "hfpef":         ["heart failure with preserved ejection fraction", "heart failure"],
    "hfref":         ["heart failure with reduced ejection fraction", "heart failure"],
    "chf":           ["congestive heart failure", "heart failure"],
    "mace":          ["major adverse cardiovascular event", "cardiovascular disease"],
    "mi":            ["myocardial infarction"],
    "af":            ["atrial fibrillation"],
    "pad":           ["peripheral artery disease", "peripheral arterial disease"],
    "htn":           ["hypertension"],
    "cad":           ["coronary artery disease"],
    "cvd":           ["cardiovascular disease"],
    "acs":           ["acute coronary syndrome"],
    "dvt":           ["deep vein thrombosis"],
    "pe":            ["pulmonary embolism"],
    "vte":           ["venous thromboembolism"],
    "aaa":           ["abdominal aortic aneurysm"],
    "svt":           ["supraventricular tachycardia"],
    "pah":           ["pulmonary arterial hypertension"],
    "tia":           ["transient ischemic attack", "transient ischaemic attack"],
    "dcm":           ["dilated cardiomyopathy"],
    "hcm":           ["hypertrophic cardiomyopathy"],
    # --- Metabolic / Endocrine ---
    "t2dm":          ["type 2 diabetes mellitus"],
    "t1dm":          ["type 1 diabetes mellitus"],
    "t2d":           ["type 2 diabetes mellitus"],
    "t1d":           ["type 1 diabetes mellitus"],
    "dm":            ["diabetes mellitus"],
    "dka":           ["diabetic ketoacidosis"],
    "bmi":           ["body mass index", "obesity"],
    "dyslipidemia":  ["dyslipidaemia", "hyperlipidemia", "hyperlipidaemia"],
    "dyslipidaemia": ["dyslipidemia", "hyperlipidaemia", "hyperlipidemia"],
    "hyperlipidemia": ["hyperlipidaemia", "dyslipidemia"],
    "hyperlipidaemia": ["hyperlipidemia", "dyslipidaemia"],
    "pcos":          ["polycystic ovary syndrome", "polycystic ovarian syndrome"],
    # --- Liver ---
    "nafld":         ["non-alcoholic fatty liver disease",
                      "metabolic dysfunction-associated steatotic liver disease"],
    "nash":          ["non-alcoholic steatohepatitis",
                      "metabolic dysfunction-associated steatohepatitis"],
    "mash":          ["metabolic dysfunction-associated steatohepatitis",
                      "non-alcoholic steatohepatitis"],
    "masld":         ["metabolic dysfunction-associated steatotic liver disease",
                      "non-alcoholic fatty liver disease"],
    "hcc":           ["hepatocellular carcinoma"],
    "psc":           ["primary sclerosing cholangitis"],
    "pbc":           ["primary biliary cholangitis", "primary biliary cirrhosis"],
    "ald":           ["alcohol-related liver disease", "alcoholic liver disease"],
    # --- Renal ---
    "ckd":           ["chronic kidney disease"],
    "aki":           ["acute kidney injury"],
    "esrd":          ["end-stage renal disease", "end-stage kidney disease"],
    "rcc":           ["renal cell carcinoma"],
    "dkd":           ["diabetic kidney disease", "diabetic nephropathy"],
    "fsgs":          ["focal segmental glomerulosclerosis"],
    # --- Respiratory ---
    "copd":          ["chronic obstructive pulmonary disease"],
    "osa":           ["obstructive sleep apnea", "obstructive sleep apnoea"],
    "ards":          ["acute respiratory distress syndrome"],
    "ipf":           ["idiopathic pulmonary fibrosis"],
    "cf":            ["cystic fibrosis"],
    # --- GI ---
    "gerd":          ["gastroesophageal reflux disease", "gastro-oesophageal reflux disease"],
    "ibd":           ["inflammatory bowel disease"],
    "uc":            ["ulcerative colitis"],
    "cd":            ["Crohn disease", "Crohn's disease"],
    "ibs":           ["irritable bowel syndrome"],
    "sbs":           ["short bowel syndrome"],
    # --- Neurological ---
    "ad":            ["Alzheimer disease", "Alzheimer's disease"],
    "pd":            ["Parkinson disease", "Parkinson's disease"],
    "ms":            ["multiple sclerosis"],
    "als":           ["amyotrophic lateral sclerosis"],
    "mdd":           ["major depressive disorder"],
    "adhd":          ["attention deficit hyperactivity disorder"],
    "gad":           ["generalized anxiety disorder", "generalised anxiety disorder"],
    "ptsd":          ["post-traumatic stress disorder"],
    "ocd":           ["obsessive-compulsive disorder"],
    "tbi":           ["traumatic brain injury"],
    # --- Oncology ---
    "nsclc":         ["non-small cell lung cancer", "non-small cell lung carcinoma"],
    "sclc":          ["small cell lung cancer", "small cell lung carcinoma"],
    "crc":           ["colorectal cancer", "colorectal carcinoma"],
    "aml":           ["acute myeloid leukemia", "acute myeloid leukaemia"],
    "all":           ["acute lymphoblastic leukemia", "acute lymphoblastic leukaemia"],
    "cml":           ["chronic myeloid leukemia", "chronic myeloid leukaemia"],
    "cll":           ["chronic lymphocytic leukemia", "chronic lymphocytic leukaemia"],
    "dlbcl":         ["diffuse large B-cell lymphoma"],
    "nhl":           ["non-Hodgkin lymphoma"],
    "mm":            ["multiple myeloma"],
    "mds":           ["myelodysplastic syndrome"],
    "tnbc":          ["triple-negative breast cancer"],
    "gist":          ["gastrointestinal stromal tumor", "gastrointestinal stromal tumour"],
    # --- Musculoskeletal / Autoimmune ---
    "ra":            ["rheumatoid arthritis"],
    "sle":           ["systemic lupus erythematosus"],
    "oa":            ["osteoarthritis"],
    "as":            ["ankylosing spondylitis"],
    "psa":           ["psoriatic arthritis"],
    "jia":           ["juvenile idiopathic arthritis"],
    "gca":           ["giant cell arteritis"],
    "ssc":           ["systemic sclerosis", "scleroderma"],
    # --- Dermatological ---
    "csu":           ["chronic spontaneous urticaria"],
    "aa":            ["alopecia areata"],
    "hs":            ["hidradenitis suppurativa"],
    # --- Hematological ---
    "itp":           ["immune thrombocytopenia", "idiopathic thrombocytopenic purpura"],
    "ttp":           ["thrombotic thrombocytopenic purpura"],
    "hit":           ["heparin-induced thrombocytopenia"],
    "pnh":           ["paroxysmal nocturnal hemoglobinuria", "paroxysmal nocturnal haemoglobinuria"],
    "scd":           ["sickle cell disease"],
    # --- Other ---
    "aud":           ["alcohol use disorder"],
    "sud":           ["substance use disorder"],
    "uti":           ["urinary tract infection"],
    "bph":           ["benign prostatic hyperplasia"],
    "eds":           ["excessive daytime sleepiness"],
    "hiv":           ["human immunodeficiency virus infection"],
    "tb":            ["tuberculosis"],
    "iga":           ["immunoglobulin A nephropathy", "IgA nephropathy"],
    "mg":            ["myasthenia gravis"],
    "nmo":           ["neuromyelitis optica"],
}

# ==============================
# BRITISH ↔ AMERICAN SPELLING NORMALIZATION
# ==============================
# We normalize TO American English so both forms map to one key.
_BRIT_TO_AMERICAN: dict[str, str] = {
    "oedema": "edema",
    "tumour": "tumor",
    "anaemia": "anemia",
    "leukaemia": "leukemia",
    "haemorrhage": "hemorrhage",
    "haemophilia": "hemophilia",
    "haemoglobin": "hemoglobin",
    "haemolytic": "hemolytic",
    "haematological": "hematological",
    "haematopoietic": "hematopoietic",
    "oestrogen": "estrogen",
    "foetus": "fetus",
    "foetal": "fetal",
    "coeliac": "celiac",
    "paediatric": "pediatric",
    "orthopaedic": "orthopedic",
    "gynaecological": "gynecological",
    "diarrhoea": "diarrhea",
    "apnoea": "apnea",
    "ischaemic": "ischemic",
    "ischaemia": "ischemia",
    "oesophageal": "esophageal",
    "oesophagus": "esophagus",
    "hypoglycaemia": "hypoglycemia",
    "hyperglycaemia": "hyperglycemia",
    "dyslipidaemia": "dyslipidemia",
    "hyperlipidaemia": "hyperlipidemia",
    "septicaemia": "septicemia",
    "bacteraemia": "bacteremia",
    "uraemia": "uremia",
    "thalassaemia": "thalassemia",
    "fibre": "fiber",
    "labelling": "labeling",
    "modelling": "modeling",
}

# Build regex for fast British→American replacement
_BRIT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_BRIT_TO_AMERICAN, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)


def _to_american(text: str) -> str:
    """Replace British medical spellings with American equivalents (case-insensitive)."""
    def _replace(m: re.Match) -> str:
        word = m.group(0)
        replacement = _BRIT_TO_AMERICAN.get(word.lower(), word)
        # Preserve original casing style
        if word[0].isupper():
            return replacement.capitalize()
        return replacement
    return _BRIT_PATTERN.sub(_replace, text)


# ==============================
# GENERIC MEDICAL WORDS (for IDF-like scoring)
# ==============================
# High-frequency words that carry less weight in overlap scoring.
# "renal" + "disease" overlap is weaker than "steatohepatitis" overlap.
_GENERIC_MEDICAL_WORDS: frozenset[str] = frozenset({
    "disease", "disorder", "syndrome", "condition", "chronic", "acute",
    "type", "stage", "grade", "risk", "primary", "secondary", "severe",
    "mild", "moderate", "progressive", "recurrent", "advanced", "early",
    "late", "unspecified", "other", "related", "associated", "induced",
    "major", "minor", "generalized", "generalised", "systemic", "local",
    "idiopathic", "acquired", "congenital", "hereditary", "familial",
    "benign", "malignant", "metastatic", "refractory", "resistant",
    "impairment", "insufficiency", "failure", "injury", "infection",
    "inflammation", "neoplasm", "tumor", "tumour", "carcinoma", "cancer",
})

# Specific clinical terms score higher (not in the generic set).
_GENERIC_WORD_WEIGHT = 2    # weight for generic words
_SPECIFIC_WORD_WEIGHT = 8   # weight for specific clinical terms
_EXTRA_WORD_PENALTY = 3     # penalty per extra word in hit not in query


def _weighted_word_score(query_words: set[str], hit_words: set[str]) -> float:
    """IDF-like scoring: specific clinical terms count more than generic ones."""
    overlap = query_words & hit_words
    extra = hit_words - query_words
    score = 0.0
    for w in overlap:
        score += _GENERIC_WORD_WEIGHT if w in _GENERIC_MEDICAL_WORDS else _SPECIFIC_WORD_WEIGHT
    for w in extra:
        score -= _EXTRA_WORD_PENALTY if w not in _GENERIC_MEDICAL_WORDS else 1
    return score


# ==============================
# CLINICALLY MEANINGFUL PARENTHETICALS
# ==============================
# Parenthetical content matching these patterns is KEPT during normalization
# (e.g. "Type 2", "Stage III", "Grade 2") because it distinguishes
# clinically different conditions.
_CLINICAL_PAREN_RE = re.compile(
    r"\b(type\s*\d|stage\s*[ivxIVX\d]+|grade\s*\d|class\s*[ivxIVX\d]+|"
    r"group\s*\d|phase\s*[ivxIVX\d]+|category\s*\d|variant\s*\d|"
    r"subtype\s*\w+|genotype\s*\d)\b",
    flags=re.IGNORECASE,
)


def _is_abbreviation_only_paren(content: str) -> bool:
    """Returns True if the parenthetical is just an abbreviation/acronym
    (e.g. 'NAFLD', 'CKD') and NOT clinically meaningful content."""
    content = content.strip()
    # Pure abbreviation: all uppercase, optionally with digits/hyphens
    if re.fullmatch(r"[A-Z][A-Z0-9\-]{0,10}", content):
        return True
    # Check if it contains clinically meaningful info
    if _CLINICAL_PAREN_RE.search(content):
        return False
    # Short annotations that are just labels
    if len(content.split()) <= 2 and content.isupper():
        return True
    return False


# ==============================
# NORMALIZATION
# ==============================
def normalize_indication(ind: str) -> str:
    """Canonical form so spelling variants map to the same key.

    ``'Pre-diabetes'``, ``'Prediabetes'`` → ``'prediabetes'``.
    ``'Non-alcoholic Fatty Liver Disease (NAFLD)'`` →
    ``'nonalcoholic fatty liver disease'`` (same as ``'Non-alcoholic
    fatty liver disease'``).

    Preserves clinically meaningful parentheticals (e.g. ``(Type 2)``,
    ``(Stage III)``) while stripping abbreviation-only ones (e.g.
    ``(NAFLD)``, ``(CKD)``).

    Normalizes British→American spelling so ``oedema`` and ``edema``
    map to the same key.

    Strips: casing, intra-word hyphens, abbreviation-only parentheticals,
    and collapses whitespace.
    """
    s = ind.strip()
    # Selectively strip parentheticals: only abbreviation-only ones,
    # preserve clinically meaningful content like (Type 2), (Stage III)
    def _strip_paren(m: re.Match) -> str:
        content = m.group(1)
        if _is_abbreviation_only_paren(content):
            return ""
        # Keep the parenthetical content but remove the parens themselves
        return " " + content.strip()

    s = re.sub(r"\s*\(([^)]*)\)", _strip_paren, s).strip()
    s = re.sub(r"\s+", " ", s).lower()
    # Strip intra-word hyphens (pre-diabetes → prediabetes)
    s = re.sub(r"(?<=\w)-(?=\w)", "", s)
    # Normalize British → American spelling
    s = _to_american(s)
    return re.sub(r"\s+", " ", s).strip()


# ==============================
# OT API CALL WITH RETRY
# ==============================
def _ot_post_with_retry(
    query: str,
    variables: dict,
    context: str = "",
    max_retries: int = _OT_MAX_RETRIES,
) -> dict | None:
    """Wraps ``ot_post`` with exponential backoff for transient failures
    (rate limits, timeouts). Returns None only after all retries exhausted."""
    for attempt in range(1, max_retries + 1):
        data = ot_post(query, variables, context=context)
        if data is not None:
            return data
        if attempt < max_retries:
            wait = 2 ** attempt  # 2s, 4s, 8s
            logger.warning(
                "[IND_MAPPING] OT API returned None for '%s' (attempt %d/%d) — "
                "retrying in %ds",
                context, attempt, max_retries, wait,
            )
            time.sleep(wait)
    logger.warning("[IND_MAPPING] OT API failed after %d attempts for '%s'", max_retries, context)
    return None


# ==============================
# OT DISEASE SEARCH
# ==============================
def _is_disease_hit(name: str | None) -> bool:
    if not name:
        return False
    lower = name.lower()
    return not any(prefix in lower for prefix in _MEASUREMENT_PREFIXES)


def ot_search_disease(name: str) -> tuple[str | None, str | None]:
    """Search OT for a disease by name. Returns ``(disease_id, disease_name)``.

    Uses a page size of ``_OT_SEARCH_PAGE_SIZE`` (15) to catch better
    matches that would be missed at size 5. Scoring uses IDF-like
    weighting so generic words (disease, syndrome) count less than
    specific clinical terms.
    """
    graphql = f"""
    query SearchDisease($q: String!) {{
      search(queryString: $q, entityNames: ["disease"], page: {{index: 0, size: {_OT_SEARCH_PAGE_SIZE}}}) {{
        hits {{
          id
          object {{ ... on Disease {{ name }} }}
        }}
      }}
    }}
    """
    search_queries = [name]
    name_lower = name.lower()
    # British/American spelling variant generation for search
    american = _to_american(name)
    if american.lower() != name_lower:
        search_queries.append(american)
    if "emia" in name_lower:
        search_queries.append(re.sub(r"emia\b", "aemia", name, flags=re.IGNORECASE))
    elif "aemia" in name_lower:
        search_queries.append(re.sub(r"aemia\b", "emia", name, flags=re.IGNORECASE))
    if len(name.split()) == 1 and not name.startswith('"'):
        search_queries.append(f'"{name}"')

    all_candidates: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for sq in search_queries:
        data = _ot_post_with_retry(graphql, {"q": sq}, context=f"disease-search:{sq}")
        if data:
            for h in data.get("search", {}).get("hits", []):
                hid = h["id"]
                if hid not in seen_ids:
                    seen_ids.add(hid)
                    all_candidates.append((hid, h["object"].get("name", "")))

    if not all_candidates:
        return None, None

    # Filter out non-disease IDs (HP_, GO_) — these aren't real disease
    # entries on the OT Platform
    disease_candidates = [(cid, cname) for cid, cname in all_candidates if _is_valid_disease_id(cid)]
    if not disease_candidates:
        logger.warning(
            "[IND_MAPPING] OT search for '%s' returned only non-disease IDs (HP_/GO_) — skipping",
            name,
        )
        return None, None

    # Score candidates with IDF-like weighting
    query_words = set(re.findall(r"[a-z]{3,}", name.lower()))
    best_id, best_name, best_score = None, None, -999.0

    for rank, (cid, cname) in enumerate(disease_candidates):
        score = 0.0
        cname_lower = (cname or "").lower()
        query_lower = name.lower().strip()

        if _is_disease_hit(cname):
            score += 100
        if cname_lower == query_lower:
            score += 50
        elif re.sub(r"aemia\b", "emia", cname_lower) == query_lower:
            score += 50
        elif _to_american(cname_lower) == _to_american(query_lower):
            score += 50

        hit_words = set(re.findall(r"[a-z]{3,}", cname_lower))
        score += _weighted_word_score(query_words, hit_words)
        score -= rank * 0.1

        if score > best_score:
            best_score = score
            best_id, best_name = cid, cname

    return best_id, best_name


# ==============================
# FALLBACK SEARCH TERMS
# ==============================
def _fallback_search_terms(ind: str) -> list[str]:
    """Generate progressively simpler OT search terms for an indication."""
    terms: list[str] = []
    ind_lower = ind.strip().lower()
    paren_match = re.search(r"\(([^)]+)\)", ind)
    paren_base = re.sub(r"\s*\(.*?\)", "", ind).strip()

    # Synonym expansion
    for candidate in [ind_lower, paren_base.lower()]:
        if candidate in MEDICAL_SYNONYMS:
            terms.extend(MEDICAL_SYNONYMS[candidate])
    if paren_match:
        acronym = paren_match.group(1).strip().lower()
        if acronym in MEDICAL_SYNONYMS:
            terms.extend(MEDICAL_SYNONYMS[acronym])

    # British/American spelling variant
    american = _to_american(ind)
    if american.lower() != ind_lower:
        terms.append(american)

    # Strip parentheticals
    if paren_base and paren_base != ind:
        terms.append(paren_base)

    base = paren_base if paren_base else ind

    # Strip action words
    action_re = r"\b(reduction|risk\s+reduction|outcomes|risk|prevention|increased|decreased)\b"
    cleaned = re.sub(action_re, "", base, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*/\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned.lower() != base.lower() and len(cleaned) > 1:
        terms.append(cleaned)
        if cleaned.lower() in MEDICAL_SYNONYMS:
            terms.extend(MEDICAL_SYNONYMS[cleaned.lower()])

    # Slash handling
    slashed = re.sub(r"\s*/\s*", " ", base).strip()
    slashed = re.sub(r"\s+", " ", slashed).strip()
    if slashed != base:
        terms.append(slashed)
        base = slashed

    # Dehyphenate
    dehyphen = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", base).strip()
    if dehyphen != base:
        terms.append(dehyphen)
        base = dehyphen

    # Progressive shortening
    words = base.split()
    if len(words) > 3:
        terms.append(" ".join(words[:3]))
    if len(words) > 2:
        terms.append(" ".join(words[:2]))
    if len(words) > 1:
        terms.append(words[0])

    # Deduplicate, exclude original
    seen = {ind.lower()}
    return [t for t in terms if t and t.lower() not in seen and not seen.add(t.lower())]


# ==============================
# OT TEXT SEARCH FALLBACK (Path B)
# ==============================
def _resolve_via_ot_search(ind: str, llm_ot_name: str | None = None) -> tuple[str | None, str | None]:
    """Resolve one indication via progressive OT text search.

    ``llm_ot_name``, when available (the LLM's own guess of this
    indication's standardized Open Targets name, captured at extraction
    time), is searched FIRST - it's usually a more accurate query than
    the raw extracted indication text, since it's already phrased in
    standard disease terminology.

    Uses IDF-like scoring to downweight generic medical terms.
    """
    search_terms = ([llm_ot_name] if llm_ot_name else []) + [ind] + _fallback_search_terms(ind)
    candidates: list[tuple[str, str, str]] = []

    for term in search_terms:
        ot_id, ot_name = ot_search_disease(term)
        if not ot_id:
            continue
        candidates.append((term, ot_id, ot_name))
        if (ot_name or "").lower().strip() == ind.lower().strip():
            break

    if not candidates:
        return None, None

    # Score candidates with IDF-like weighting
    ind_words = set(re.findall(r"[a-z]{3,}", ind.lower()))
    best_id, best_name, best_score = None, None, -999.0

    for term_used, cid, cname in candidates:
        score = 0.0
        cname_lower = (cname or "").lower().strip()
        ind_lower_s = ind.lower().strip()

        if cname_lower == ind_lower_s:
            score += 200
        elif re.sub(r"aemia\b", "emia", cname_lower) == ind_lower_s:
            score += 200
        elif _to_american(cname_lower) == _to_american(ind_lower_s):
            score += 200

        if _is_disease_hit(cname):
            score += 100

        hit_words = set(re.findall(r"[a-z]{3,}", cname_lower))
        score += _weighted_word_score(ind_words, hit_words)

        if score > best_score:
            best_score = score
            best_id, best_name = cid, cname

    return best_id, best_name


# ==============================
# GEMINI SEMANTIC MATCHING (Path A)
# ==============================
def fetch_all_target_diseases(
    target_ids: list[str],
    page_size: int = 50,
) -> list[tuple[str, str]]:
    """Paginate through OT to get ALL diseases associated with the given targets."""
    query = """
    query TargetDiseases($targetId: String!, $index: Int!, $size: Int!) {
      target(ensemblId: $targetId) {
        associatedDiseases(page: { index: $index, size: $size }) {
          count
          rows { disease { id name } }
        }
      }
    }
    """
    seen: set[str] = set()
    all_diseases: list[tuple[str, str]] = []

    for tid in target_ids:
        if not tid:
            continue
        page_index = 0
        total = None
        fetched = 0
        while True:
            data = _ot_post_with_retry(
                query,
                {"targetId": tid, "index": page_index, "size": page_size},
                context=f"target-diseases:{tid}:p{page_index}",
            )
            if not data:
                break
            assoc = data.get("target", {}).get("associatedDiseases", {})
            if total is None:
                total = assoc.get("count", 0)
                logger.info("[IND_MAPPING] Target %s: %d associated diseases in OT", tid, total)
            rows = assoc.get("rows", [])
            if not rows:
                break
            for row in rows:
                d = row.get("disease", {})
                did = d.get("id")
                name = d.get("name", "")
                if did and did not in seen:
                    seen.add(did)
                    # Only keep valid disease IDs (EFO_/MONDO_/Orphanet_),
                    # skip HP_/GO_ phenotype/ontology entries
                    if _is_valid_disease_id(did):
                        all_diseases.append((did, name))
            fetched += len(rows)
            if fetched >= (total or 0):
                break
            page_index += 1

    logger.info(
        "[IND_MAPPING] Fetched %d unique diseases across %d target(s)",
        len(all_diseases), len(target_ids),
    )
    return all_diseases


def _gemini_match_batch(
    indications: list[str],
    ot_diseases: list[tuple[str, str]],
) -> list[tuple[str, str | None, str | None]]:
    """One Gemini call to semantically match indications against an OT disease chunk.

    Filters out non-disease IDs (HP_, GO_) from the disease list before
    sending to Gemini, so only valid EFO_/MONDO_ entries are matchable.
    """
    # Filter to valid disease IDs only — don't show HP_/GO_ to Gemini
    valid_diseases = [(did, dname) for did, dname in ot_diseases if _is_valid_disease_id(did)]
    if not valid_diseases:
        return [(ind, None, None) for ind in indications]

    disease_lines = "\n".join(
        f"{i + 1}. {dname} | {did}" for i, (did, dname) in enumerate(valid_diseases)
    )
    ind_numbered = "\n".join(f"{i + 1}. {ind}" for i, ind in enumerate(indications))

    prompt = (
        "You are a biomedical terminology expert.\n\n"
        "Below is a numbered list of diseases from the OpenTargets Platform, "
        "followed by a list of clinical indications.\n\n"
        "Your task: for each indication, find the BEST matching disease "
        "from the disease list. Consider synonyms, acronyms, spelling "
        "variants (British/American), and clinical shorthand.\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY a valid JSON array. No prose, no markdown, no ```json fences.\n"
        "- One object per indication, in the same order as the input.\n"
        "- Each object must have exactly these keys:\n"
        '  {"indication": "<exact indication text>", '
        '"id": "<EFO_/MONDO_ ID from the disease list, or null if no confident match>", '
        '"name": "<disease name from the disease list, or null>"}\n'
        "- Use JSON null (not the string \"null\") when no confident match exists.\n"
        "- ONLY pick from the provided disease list — do NOT invent IDs.\n\n"
        f"DISEASE LIST:\n{disease_lines}\n\n"
        f"INDICATIONS TO MATCH:\n{ind_numbered}\n\n"
        "JSON array output:"
    )

    text = gemini_call(prompt)
    parsed = parse_json_response(text)
    if not parsed:
        return [(ind, None, None) for ind in indications]

    valid_ids = {did: dname for did, dname in valid_diseases}
    results: list[tuple[str, str | None, str | None]] = []

    for item in parsed:
        ind = item.get("indication", "")
        did = item.get("id") or None
        name = item.get("name") or None

        if did and did not in valid_ids:
            logger.warning("[IND_MAPPING] Gemini returned unknown ID '%s' for '%s' — discarding", did, ind)
            did, name = None, None
        if did and did in valid_ids:
            name = valid_ids[did]

        results.append((ind, did, name))

    while len(results) < len(indications):
        results.append((indications[len(results)], None, None))

    return results


def _score_match(indication: str, disease_id: str, disease_name: str) -> float:
    """Score how well a disease matches an indication.

    Used to pick the BEST match across all disease chunks rather than
    taking the first one Gemini returns.
    """
    score = 0.0
    ind_lower = indication.lower().strip()
    name_lower = (disease_name or "").lower().strip()

    # Strong bonus for valid disease IDs (EFO_/MONDO_), reject HP_/GO_
    if _is_valid_disease_id(disease_id):
        score += 100
    else:
        # Non-disease IDs should never win over a real disease match
        score -= 500

    # Exact name match is ideal
    if name_lower == ind_lower:
        score += 200
    elif _to_american(name_lower) == _to_american(ind_lower):
        score += 200
    elif re.sub(r"aemia\b", "emia", name_lower) == ind_lower:
        score += 200

    # IDF-weighted word overlap
    ind_words = set(re.findall(r"[a-z]{3,}", ind_lower))
    name_words = set(re.findall(r"[a-z]{3,}", name_lower))
    score += _weighted_word_score(ind_words, name_words)

    # Bonus if disease name is a proper superset or subset match
    if _is_disease_hit(disease_name):
        score += 50

    return score


def _match_all_against_ot_list(
    indications: list[str],
    ot_diseases: list[tuple[str, str]],
) -> dict[str, tuple[str | None, str | None]]:
    """Match all indications against the full OT disease list using Gemini,
    chunking the disease list into groups of ``_DISEASE_CHUNK_SIZE`` AND
    batching indications into groups of ``_INDICATION_BATCH_SIZE``.

    Every indication is checked against EVERY disease chunk. All candidate
    matches are collected, then the best match per indication is selected
    using ``_score_match`` scoring. This ensures the best match wins even
    if a weaker match is found in an earlier chunk.

    Non-disease IDs (HP_, GO_) returned by Gemini are filtered out.
    """
    # Collect ALL candidate matches: {indication: [(did, name, score), ...]}
    all_candidates: dict[str, list[tuple[str, str, float]]] = {ind: [] for ind in indications}
    total_chunks = (len(ot_diseases) + _DISEASE_CHUNK_SIZE - 1) // _DISEASE_CHUNK_SIZE

    logger.info(
        "[IND_MAPPING] Gemini matching: %d indication(s) × %d diseases → "
        "%d disease chunk(s), indication batches of %d",
        len(indications), len(ot_diseases), total_chunks, _INDICATION_BATCH_SIZE,
    )

    for chunk_idx in range(total_chunks):
        disease_chunk = ot_diseases[chunk_idx * _DISEASE_CHUNK_SIZE : (chunk_idx + 1) * _DISEASE_CHUNK_SIZE]

        # Sub-batch indications so each Gemini call is small enough to
        # produce reliable results
        ind_batches = [
            indications[i : i + _INDICATION_BATCH_SIZE]
            for i in range(0, len(indications), _INDICATION_BATCH_SIZE)
        ]

        for batch_idx, ind_batch in enumerate(ind_batches):
            batch_results = None
            for attempt in range(1, 3):
                try:
                    batch_results = _gemini_match_batch(ind_batch, disease_chunk)
                    if len(batch_results) == len(ind_batch):
                        break
                except Exception as exc:
                    logger.warning(
                        "[IND_MAPPING] Disease chunk %d, ind batch %d, attempt %d failed: %s",
                        chunk_idx + 1, batch_idx + 1, attempt, exc,
                    )
                time.sleep(2)

            if not batch_results:
                continue

            for ind, did, name in batch_results:
                if did and name:
                    score = _score_match(ind, did, name)
                    all_candidates[ind].append((did, name, score))
                    logger.debug(
                        "[IND_MAPPING] Gemini candidate: '%s' → %s (%s) [score=%.1f]",
                        ind, did, name, score,
                    )

    # Pick the best match per indication
    result: dict[str, tuple[str | None, str | None]] = {}
    for ind in indications:
        candidates = all_candidates[ind]
        if not candidates:
            result[ind] = (None, None)
            continue

        # Filter to valid disease IDs only
        valid = [(did, name, sc) for did, name, sc in candidates if _is_valid_disease_id(did)]
        if not valid:
            logger.warning(
                "[IND_MAPPING] Gemini matched '%s' only to non-disease IDs (%s) — discarding",
                ind,
                ", ".join(f"{did}" for did, _, _ in candidates),
            )
            result[ind] = (None, None)
            continue

        # Sort by score descending, pick the best
        valid.sort(key=lambda x: x[2], reverse=True)
        best_did, best_name, best_score = valid[0]
        result[ind] = (best_did, best_name)
        logger.info("[IND_MAPPING] Gemini best: '%s' → %s (%s) [score=%.1f]", ind, best_did, best_name, best_score)

    matched = sum(1 for did, _ in result.values() if did)
    logger.info(
        "[IND_MAPPING] Gemini matched %d/%d indication(s) across %d disease chunk(s)",
        matched, len(indications), total_chunks,
    )
    return result


# ==============================
# FETCH INDICATIONS FROM LE_TABLE
# ==============================
def fetch_indications_for_drug(drug_name: str, secondary_only: bool = False) -> list[dict]:
    """Fetches distinct indications for a drug from the LE_TABLE, along with
    each indication's LLM-suggested Open Targets disease name (``llm_ot_name``,
    populated at extraction time by trial_analyser/fda_fetcher/web_analyser),
    when available - used as a resolution hint by Path B search and by
    semantic clustering.

    Args:
        drug_name: the drug/molecule name.
        secondary_only: if ``True``, only fetches indications where
            ``indication_type = 'Secondary'``. Used for OT mapping and
            scoring, which should only run on label-expansion candidates.

    Returns:
        List of dicts: ``{"indication": str, "llm_ot_name": str | None}``.
    """
    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LE_TABLE}"

    secondary_filter = "AND LOWER(indication_type) = 'secondary'" if secondary_only else ""

    query = f"""
        SELECT indication, MAX(llm_ot_name) AS llm_ot_name
        FROM `{table_id}`
        WHERE drug_name = @drug_name
          AND indication IS NOT NULL
          AND indication != ''
          AND indication != 'Unknown (extraction failed)'
          {secondary_filter}
        GROUP BY indication
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name),
        ]
    )
    results = bq_client.query(query, job_config=job_config).result()
    indications = [
        {
            "indication": row["indication"].strip(),
            "llm_ot_name": (row.get("llm_ot_name") or "").strip() or None,
        }
        for row in results
        if row["indication"]
    ]

    label = "Secondary" if secondary_only else "all"
    logger.info("[IND_MAPPING] Fetched %d %s indication(s) for '%s' from %s", len(indications), label, drug_name, LE_TABLE)
    return indications


# ==============================
# SEMANTIC CLUSTERING (groups synonyms that don't share spelling)
# ==============================
_CLUSTER_CHUNK_SIZE = 60


def _cluster_chunk(indications: list[str], llm_ot_name_map: dict[str, str]) -> dict[str, str]:
    """One Gemini call: groups a chunk of indications describing the same
    underlying clinical concept (synonyms, abbreviations, anatomical
    qualifiers, old vs. new nomenclature - e.g. NAFLD and MASLD are the
    same disease) and returns ``{raw_indication: canonical_indication}``.
    Every input indication is guaranteed a mapping, even if that maps to
    itself (unclustered)."""
    if not indications:
        return {}

    lines = []
    for i, ind in enumerate(indications):
        hint = llm_ot_name_map.get(ind)
        lines.append(f"{i + 1}. {ind}" + (f" (suggested OT disease: {hint})" if hint else ""))
    ind_block = "\n".join(lines)

    prompt = f"""You are a biomedical terminology expert.

Below is a list of clinical indications extracted from trials and FDA
labels for one drug. Some entries describe the SAME underlying disease
or condition using different wording - synonyms, abbreviations,
anatomical/severity qualifiers, or old vs. current nomenclature (for
example, NAFLD and MASLD are the same disease under different names).
Where given, a "suggested OT disease" hint shows another model's guess
at that indication's standardized Open Targets name - use it as a
signal, but rely on your own clinical judgment.

Group these into clusters where each cluster represents ONE underlying
clinical concept. For each cluster, pick ONE canonical name (prefer the
most standard/current medical term). Do NOT merge indications that are
only superficially similar but refer to different conditions or
different severities that are tracked separately.

Indications:
{ind_block}

Return ONLY a JSON array - no markdown fences, no explanation:
[
  {{"canonical": "<canonical name>", "members": ["<indication 1>", "<indication 2>", ...]}}
]
Every indication listed above must appear in exactly one cluster's
"members" list, even if its cluster contains only itself.
"""
    try:
        text = gemini_call(prompt)
        clusters = parse_json_response(text)
        seen_inputs = {ind.strip().lower(): ind for ind in indications}
        mapping: dict[str, str] = {}
        for c in clusters if isinstance(clusters, list) else []:
            if not isinstance(c, dict):
                continue
            canonical = (c.get("canonical") or "").strip()
            if not canonical:
                continue
            for member in c.get("members", []) if isinstance(c.get("members"), list) else []:
                orig = seen_inputs.get((member or "").strip().lower())
                if orig:
                    mapping[orig] = canonical
        for ind in indications:
            mapping.setdefault(ind, ind)  # unclustered -> maps to itself
        return mapping
    except Exception as exc:  # noqa: BLE001
        logger.warning("[IND_MAPPING] Clustering failed for chunk %s: %s - treating each as its own cluster", indications, exc)
        return {ind: ind for ind in indications}


def _reconcile_cross_chunk_canonicals(
    canonical_names: list[str],
    llm_ot_name_map: dict[str, str],
) -> dict[str, str]:
    """Cross-chunk reconciliation: takes canonical names from all chunks
    and merges any that are synonyms of each other. This catches synonyms
    that were split across different clustering chunks.

    Returns ``{canonical: merged_canonical}`` — identity for names that
    don't merge with anything.
    """
    if len(canonical_names) <= 1:
        return {c: c for c in canonical_names}

    # If the list is small enough, no reconciliation needed
    # (all fit in one chunk, so they were already clustered together)
    if len(canonical_names) <= _CLUSTER_CHUNK_SIZE:
        # Still run reconciliation — these are from DIFFERENT chunks
        pass

    lines = []
    for i, name in enumerate(canonical_names):
        hint = llm_ot_name_map.get(name)
        lines.append(f"{i + 1}. {name}" + (f" (OT hint: {hint})" if hint else ""))
    name_block = "\n".join(lines)

    prompt = f"""You are a biomedical terminology expert.

Below is a list of canonical disease/condition names that were produced
by clustering indications in separate batches. Some of these canonical
names may STILL refer to the same underlying disease under different
wording (e.g. "Non-alcoholic fatty liver disease" and "Metabolic
dysfunction-associated steatotic liver disease" are the same disease).

Merge any that refer to the same condition. For each merged group, pick
the most standard/current name as the single canonical.

Names:
{name_block}

Return ONLY a JSON array - no markdown fences:
[
  {{"canonical": "<merged canonical>", "members": ["<name 1>", "<name 2>", ...]}}
]
Every name above must appear in exactly one group's "members", even if
the group has only one member (no merge needed).
"""
    try:
        text = gemini_call(prompt)
        parsed = parse_json_response(text)
        mapping: dict[str, str] = {}
        input_lower = {n.strip().lower(): n for n in canonical_names}
        for c in parsed if isinstance(parsed, list) else []:
            if not isinstance(c, dict):
                continue
            canonical = (c.get("canonical") or "").strip()
            if not canonical:
                continue
            for member in c.get("members", []) if isinstance(c.get("members"), list) else []:
                orig = input_lower.get((member or "").strip().lower())
                if orig:
                    mapping[orig] = canonical
        for n in canonical_names:
            mapping.setdefault(n, n)
        merged_count = len(canonical_names) - len(set(mapping.values()))
        if merged_count > 0:
            logger.info(
                "[IND_MAPPING] Cross-chunk reconciliation merged %d canonical name(s)",
                merged_count,
            )
        return mapping
    except Exception as exc:  # noqa: BLE001
        logger.warning("[IND_MAPPING] Cross-chunk reconciliation failed: %s — skipping", exc)
        return {c: c for c in canonical_names}


def cluster_indications(indications: list[str], llm_ot_name_map: dict[str, str] | None = None) -> dict[str, str]:
    """Groups indications describing the same clinical concept so they
    resolve to (and score as) a single TA-I instead of several duplicates.

    Chunks large lists (``_CLUSTER_CHUNK_SIZE`` at a time), then runs a
    cross-chunk reconciliation pass to merge canonical names from different
    chunks that are synonyms of each other.

    Returns ``{raw_indication: canonical_indication}`` - every input maps
    to something, even if only to itself.
    """
    if not indications:
        return {}
    llm_ot_name_map = llm_ot_name_map or {}

    chunks = [
        indications[i : i + _CLUSTER_CHUNK_SIZE]
        for i in range(0, len(indications), _CLUSTER_CHUNK_SIZE)
    ]
    logger.info(
        "[IND_MAPPING] Clustering %d indication(s) in %d chunk(s) of up to %d",
        len(indications), len(chunks), _CLUSTER_CHUNK_SIZE,
    )

    mapping: dict[str, str] = {}
    for chunk in chunks:
        mapping.update(_cluster_chunk(chunk, llm_ot_name_map))

    # Cross-chunk reconciliation: merge canonical names from different
    # chunks that refer to the same disease
    if len(chunks) > 1:
        canonical_names = sorted(set(mapping.values()))
        if len(canonical_names) > 1:
            reconciled = _reconcile_cross_chunk_canonicals(canonical_names, llm_ot_name_map)
            # Re-map through the reconciliation
            mapping = {raw: reconciled.get(canonical, canonical) for raw, canonical in mapping.items()}

    return mapping


# ==============================
# PATH C: GEMINI + GOOGLE SEARCH GROUNDING FALLBACK
# ==============================
_PATH_C_BATCH_SIZE = 10


def _resolve_via_search_grounding_batch(indications: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """Uses Gemini with Google Search grounding to find the correct Open
    Targets disease name and EFO/MONDO ID for indications that both Path A
    (Gemini semantic match against the target's disease list) and Path B
    (OT text search) failed to resolve.

    This is the most expensive resolution path (one search-grounded Gemini
    call per batch), but it has access to the full web — so it can find
    the correct OT disease name even for indications the OT text search
    API returns nothing useful for.

    Returns ``{indication: (ot_disease_id, ot_disease_name)}`` for every
    indication in the batch. Indications the search can't resolve get
    ``(None, None)``.
    """
    if not indications:
        return {}

    ind_list = "\n".join(f"{i + 1}. {ind}" for i, ind in enumerate(indications))
    prompt = f"""You are a biomedical terminology expert specializing in the Open Targets
Platform (https://platform.opentargets.org/).

Below is a list of clinical indications (disease/condition names). For each
one, search the web to find its correct Open Targets disease entry — the
standardized EFO or MONDO disease name and ID as used by the Open Targets
Platform.

Indications:
{ind_list}

For each indication, return:
- ot_disease_name: the EXACT disease name as it appears on the Open Targets
  Platform (e.g. "non-alcoholic steatohepatitis", "prediabetes syndrome",
  "obesity"). This must be the canonical name from the EFO/MONDO ontology,
  not a synonym or colloquial phrasing.
- ot_disease_id: the EFO or MONDO ID (e.g. "EFO_0004268", "MONDO_0005148").
  If you cannot find a confident match, use null for both fields.

Return ONLY a JSON array — no markdown fences, no explanation:
[
  {{"indication": "<exact indication from input>",
    "ot_disease_name": "<Open Targets disease name, or null>",
    "ot_disease_id": "<EFO/MONDO ID, or null>"}}
]
"""
    try:
        text = gemini_generate(
            prompt,
            system_instruction=(
                "You are a biomedical terminology expert. Search the Open Targets "
                "Platform and EFO/MONDO ontologies to find exact disease mappings. "
                "Return ONLY valid JSON."
            ),
            use_search=True,
        )
        parsed = extract_json_utils(text)
        entries = parsed if isinstance(parsed, list) else parsed.get("results", []) if isinstance(parsed, dict) else []
        result: dict[str, tuple[str | None, str | None]] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            ind = (e.get("indication") or "").strip()
            did = (e.get("ot_disease_id") or "").strip() or None
            name = (e.get("ot_disease_name") or "").strip() or None
            if ind:
                result[ind] = (did, name)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[IND_MAPPING] Path C search-grounded resolution failed for batch %s: %s", indications, exc)
        return {}


def _verify_ot_id(disease_id: str) -> tuple[str | None, str | None]:
    """Verify an EFO/MONDO ID actually exists in the OT API AND is a
    real disease entity (not an HP_/GO_ phenotype/ontology term).

    Returns ``(disease_id, disease_name)`` if valid, ``(None, None)`` if
    the ID doesn't exist or isn't a proper disease entry.
    """
    # Reject non-disease prefixes before even calling the API
    if not _is_valid_disease_id(disease_id):
        logger.warning(
            "[IND_MAPPING] Rejecting non-disease ID '%s' (not EFO_/MONDO_/Orphanet_)",
            disease_id,
        )
        return None, None

    graphql = """
    query VerifyDisease($id: String!) {
      disease(efoId: $id) {
        id
        name
      }
    }
    """
    data = _ot_post_with_retry(graphql, {"id": disease_id}, context=f"verify-id:{disease_id}")
    if not data:
        return None, None
    disease = data.get("disease")
    if not disease or not disease.get("id"):
        return None, None
    return disease["id"], disease.get("name", "")


def resolve_via_search_grounding(unresolved: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """Batched Path C: resolves unmatched indications via Gemini + Google
    Search grounding, ``_PATH_C_BATCH_SIZE`` at a time.

    After Gemini returns IDs, each is **verified against the OT API** to
    reject hallucinated IDs that don't actually exist in Open Targets.

    Called after Path A and Path B, for any indications they left with
    ``(None, None)``. Returns ``{indication: (ot_disease_id, ot_disease_name)}``.
    """
    if not unresolved:
        return {}

    logger.info(
        "[IND_MAPPING] Path C: Gemini + Google Search grounding for %d unresolved indication(s)",
        len(unresolved),
    )
    batches = [
        unresolved[i : i + _PATH_C_BATCH_SIZE]
        for i in range(0, len(unresolved), _PATH_C_BATCH_SIZE)
    ]
    raw_results: dict[str, tuple[str | None, str | None]] = {}
    for batch in batches:
        raw_results.update(_resolve_via_search_grounding_batch(batch))

    # Verify every ID returned by Path C against the OT API
    verified_results: dict[str, tuple[str | None, str | None]] = {}
    ids_to_verify: dict[str, list[str]] = {}  # disease_id -> [indications]
    for ind, (did, name) in raw_results.items():
        if did and name:
            ids_to_verify.setdefault(did, []).append(ind)
        else:
            verified_results[ind] = (None, None)

    verified_cache: dict[str, tuple[str | None, str | None]] = {}
    rejected_count = 0
    for did, inds in ids_to_verify.items():
        if did not in verified_cache:
            verified_cache[did] = _verify_ot_id(did)
        verified_id, verified_name = verified_cache[did]
        for ind in inds:
            if verified_id:
                # Use the verified name from OT (more authoritative than Gemini's)
                verified_results[ind] = (verified_id, verified_name)
            else:
                logger.warning(
                    "[IND_MAPPING] Path C: rejected hallucinated ID '%s' for '%s'",
                    did, ind,
                )
                verified_results[ind] = (None, None)
                rejected_count += 1

    resolved_count = sum(1 for did, name in verified_results.values() if did and name)
    logger.info(
        "[IND_MAPPING] Path C resolved %d/%d indication(s) via search grounding "
        "(%d hallucinated ID(s) rejected)",
        resolved_count, len(unresolved), rejected_count,
    )
    return verified_results


# ==============================
# MATCH VALIDATION (rejects word-overlap false positives)
# ==============================
_VALIDATE_BATCH_SIZE = 20
_VALIDATE_MAX_RETRIES = 2  # retry validation before failing closed


def _validate_match_batch(pairs: list[tuple[str, str]]) -> dict[str, bool]:
    """One Gemini call: for each (indication, proposed OT disease name)
    pair, decides whether the proposed disease is a clinically valid
    match - not just a word-overlap false positive (e.g. "renal
    impairment" matched to "renal carcinoma" shares a word but is NOT a
    valid match; they must refer to the same or a closely related
    disease). Returns ``{indication_lower: is_valid}``.

    **Fail-closed**: on error after retries, all matches in the batch
    are marked INVALID (False) rather than silently accepted. A null
    mapping is preferable to a confidently wrong one.
    """
    if not pairs:
        return {}

    lines = "\n".join(
        f'{i + 1}. Indication: "{ind}" | Proposed match: "{name}"'
        for i, (ind, name) in enumerate(pairs)
    )
    prompt = f"""You are a biomedical terminology expert reviewing automated
disease-name matches for quality control.

For each pair below, decide whether the "Proposed match" is a clinically
valid match for the "Indication" - i.e. they refer to the same disease,
or a very closely related form of it (e.g. a synonym, a renamed term, or
a specific subtype). A match based only on shared words (e.g. "renal
impairment" vs. "renal carcinoma", "ectopic fat" vs. "ectopic posterior
pituitary") is NOT valid unless the underlying condition is genuinely
the same or closely related.

Pairs:
{lines}

Return ONLY a JSON array - no markdown fences, no explanation:
[
  {{"indication": "<exact indication text>", "valid": true or false}}
]
"""
    last_exc = None
    for attempt in range(1, _VALIDATE_MAX_RETRIES + 1):
        try:
            text = gemini_call(prompt)
            parsed = parse_json_response(text)
            result: dict[str, bool] = {}
            for e in parsed if isinstance(parsed, list) else []:
                if not isinstance(e, dict):
                    continue
                ind = (e.get("indication") or "").strip().lower()
                if ind:
                    result[ind] = bool(e.get("valid", True))
            if result:
                return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "[IND_MAPPING] Match validation attempt %d/%d failed: %s",
                attempt, _VALIDATE_MAX_RETRIES, exc,
            )
            if attempt < _VALIDATE_MAX_RETRIES:
                time.sleep(2)

    # Fail CLOSED: on error, reject all matches in this batch rather than
    # silently accepting potentially wrong mappings. A null mapping is
    # preferable to a confidently wrong one — the indication simply won't
    # get an association_score and can be re-attempted later.
    logger.warning(
        "[IND_MAPPING] Match validation failed after %d attempts (last error: %s) "
        "— REJECTING all %d match(es) in this batch (fail-closed)",
        _VALIDATE_MAX_RETRIES, last_exc, len(pairs),
    )
    return {ind.lower(): False for ind, _ in pairs}


def revalidate_existing_mappings(drug_name: str, secondary_only: bool = False) -> dict:
    """Re-checks ALL of this drug's already-resolved (non-null) mappings
    in ``OT_DISEASE_TABLE`` against the current QC validation logic, and
    nulls out (upserts) any that fail.

    ``validate_resolved_mappings`` normally only checks mappings resolved
    in the SAME run it's called from - it never re-examines a mapping
    that was already sitting in the table, however wrong. That means a
    bad match resolved before validation existed (or before an
    improvement to it) stays wrong forever unless something re-checks it
    explicitly. This is that explicit re-check: call it on demand (e.g.
    after improving the validation prompt, or when a wrong mapping like
    "Malignant Solid Neoplasm" -> "malignant endocrine neoplasm" is
    spotted) to clean up the whole table for one drug in one pass.

    Nulled-out indications will be re-attempted by Path A/B/C on the
    next ``run_indication_mapping`` call (nulls are excluded from the
    "already resolved" check there).

    Returns a dict summary: ``{"checked": int, "rejected": [indication, ...]}``.
    """
    logger.info("[IND_MAPPING] Re-validating existing mappings for '%s'", drug_name)

    fetched = fetch_indications_for_drug(drug_name, secondary_only=secondary_only)
    if not fetched:
        logger.warning("[IND_MAPPING] No indications found for '%s' - nothing to re-validate", drug_name)
        return {"checked": 0, "rejected": []}

    existing_raw = fetch_existing_mappings(OT_DISEASE_TABLE, "indication")
    indications = [f["indication"] for f in fetched]

    to_check: dict[str, tuple[str | None, str | None]] = {}
    for ind in indications:
        entry = existing_raw.get(ind.strip().lower())
        if entry and entry.get("ot_disease") and entry.get("ot_disease_id"):
            to_check[ind] = (entry["ot_disease_id"], entry["ot_disease"])

    if not to_check:
        logger.info("[IND_MAPPING] No non-null existing mappings for '%s' to re-validate", drug_name)
        return {"checked": 0, "rejected": []}

    revalidated = validate_resolved_mappings(to_check)
    rejected = [ind for ind, (did, _name) in revalidated.items() if not did]

    if rejected:
        logger.warning(
            "[IND_MAPPING] Re-validation rejected %d previously-accepted mapping(s) for '%s': %s",
            len(rejected), drug_name, rejected,
        )
        rejected_rows = [{"indication": ind, "ot_disease": None, "ot_disease_id": None} for ind in rejected]
        push_mappings(OT_DISEASE_TABLE, OT_DISEASE_SCHEMA, rejected_rows)
    else:
        logger.info("[IND_MAPPING] Re-validation: all %d checked mapping(s) for '%s' still hold up", len(to_check), drug_name)

    return {"checked": len(to_check), "rejected": rejected}


def validate_resolved_mappings(
    resolved: dict[str, tuple[str | None, str | None]],
) -> dict[str, tuple[str | None, str | None]]:
    """Re-checks every resolved mapping with a Gemini QC pass, nulling
    out any match that isn't a genuine clinical match (rather than a
    word-overlap false positive like "renal impairment" → "renal carcinoma").

    Fail-CLOSED: a null mapping is preferable to a confidently wrong one.
    """
    to_check = [(ind, did, name) for ind, (did, name) in resolved.items() if did and name]
    if not to_check:
        return resolved

    logger.info("[IND_MAPPING] Validating %d resolved mapping(s)", len(to_check))
    validated = dict(resolved)

    batches = [
        to_check[i : i + _VALIDATE_BATCH_SIZE]
        for i in range(0, len(to_check), _VALIDATE_BATCH_SIZE)
    ]
    rejected = 0
    for batch in batches:
        pairs = [(ind, name) for ind, _did, name in batch]
        results = _validate_match_batch(pairs)
        for ind, did, name in batch:
            if not results.get(ind.lower(), True):
                logger.warning(
                    "[IND_MAPPING] Rejected low-confidence match: '%s' -> '%s' (%s)",
                    ind, name, did,
                )
                validated[ind] = (None, None)
                rejected += 1

    if rejected:
        logger.info("[IND_MAPPING] Validation rejected %d/%d resolved mapping(s)", rejected, len(to_check))
    return validated


# ==============================
# POST-VALIDATION RE-RESOLUTION FALLBACK
# ==============================
def _re_resolve_rejected(
    rejected_indications: list[str],
    canonical_hint_map: dict[str, str],
) -> tuple[dict[str, tuple[str | None, str | None]], dict[str, str]]:
    """Re-attempt resolution for indications whose initial match was
    rejected by validation (false positive).

    Uses a more targeted approach:
    1. First tries Path B (OT text search) with the raw indication text —
       the initial match may have come from Path A which matched against
       a disease list, while Path B does a direct text search that may
       find a better (broader) match.
    2. For anything still unresolved, tries Path C (Gemini + Google Search
       grounding with OT API verification).

    Returns ``(resolved_map, resolution_paths)`` — same format as the
    main pipeline uses.
    """
    if not rejected_indications:
        return {}, {}

    logger.info(
        "[IND_MAPPING] Re-resolution fallback: attempting %d rejected indication(s)",
        len(rejected_indications),
    )

    resolved_map: dict[str, tuple[str | None, str | None]] = {}
    resolution_paths: dict[str, str] = {}

    # Round 1: Path B — targeted OT text search
    still_unresolved = []
    with ThreadPoolExecutor(
        max_workers=min(_WORKERS, len(rejected_indications)),
        thread_name_prefix="re-resolve",
    ) as exe:
        def _resolve_one(ind: str) -> tuple[str, str | None, str | None]:
            did, name = _resolve_via_ot_search(ind, llm_ot_name=canonical_hint_map.get(ind))
            return ind, did, name

        futures = {exe.submit(_resolve_one, ind): ind for ind in rejected_indications}
        for fut in as_completed(futures):
            ind, did, name = fut.result()
            if did and name:
                resolved_map[ind] = (did, name)
                resolution_paths[ind] = "path_b_retry"
                logger.info("[IND_MAPPING] Re-resolve Path B: '%s' → %s (%s)", ind, did, name)
            else:
                still_unresolved.append(ind)

    # Round 2: Path C — Gemini + Google Search grounding
    if still_unresolved:
        logger.info(
            "[IND_MAPPING] Re-resolve Path C for %d still-unresolved indication(s)",
            len(still_unresolved),
        )
        path_c_results = resolve_via_search_grounding(still_unresolved)
        for ind, (did, name) in path_c_results.items():
            if did and name:
                resolved_map[ind] = (did, name)
                resolution_paths[ind] = "path_c_retry"
                logger.info("[IND_MAPPING] Re-resolve Path C: '%s' → %s (%s)", ind, did, name)

    re_resolved_count = len(resolved_map)
    total = len(rejected_indications)
    logger.info(
        "[IND_MAPPING] Re-resolution fallback: recovered %d/%d rejected indication(s)",
        re_resolved_count, total,
    )
    return resolved_map, resolution_paths


# ==============================
# ENTRY POINT
# ==============================
def run_indication_mapping(
    drug_name: str = DRUG_NAME,
    target_ensembl_ids: list[str] | None = None,
    secondary_only: bool = False,
) -> list[dict]:
    """Full indication mapping pipeline for one drug.

    1. Fetch indications (+ their LLM-suggested OT name hint) from ``LE_TABLE``.
    2. Check which indications are already resolved in ``OT_DISEASE_TABLE``,
       matching on normalized text so spelling/hyphenation variants of an
       already-resolved indication (e.g. "Prediabetes" vs. "Pre-diabetes")
       reuse the existing mapping instead of being re-resolved from scratch.
    3. Normalize and deduplicate new indications by spelling.
    4. Cluster the normalized representatives semantically (Gemini), with
       cross-chunk reconciliation to merge synonyms split across chunks.
    5. Resolve each canonical cluster via Path A (Gemini + OT disease list,
       with indication batching for reliable results) if targets given,
       then Path B (OT text search with retry, seeded with the
       LLM-suggested OT name when available) for any remaining, then
       Path C (Gemini + Google Search grounding, with OT API verification
       to reject hallucinated IDs) for anything still unresolved.
    6. Validate every resolved mapping with a dedicated Gemini QC pass
       (fail-closed: API errors reject the batch rather than accepting),
       nulling out matches that are word-overlap false positives.
    6.5 Re-resolution fallback: indications rejected by validation are
        re-attempted via Path B (targeted search) and Path C (search
        grounding) to find a correct match.
    7. Push ONLY correctly resolved mappings to ``OT_DISEASE_TABLE`` —
       null/unresolved mappings are NOT pushed, so the table contains
       only verified matches. Unresolved indications will be re-attempted
       on the next run.
    8. Return all resolved mappings (existing + new).

    Args:
        drug_name: the drug/molecule name.
        target_ensembl_ids: Ensembl IDs for the drug's gene targets
            (enables Path A Gemini matching). Pass ``None`` to skip Path A.

    Returns:
        List of dicts with keys ``indication``, ``ot_disease``,
        ``ot_disease_id``, ``resolution_path``.
    """
    logger.info("[IND_MAPPING] Starting indication mapping for '%s'", drug_name)

    # Step 1: Fetch indications + their LLM-suggested OT name hints
    fetched = fetch_indications_for_drug(drug_name, secondary_only=secondary_only)
    if not fetched:
        logger.warning("[IND_MAPPING] No indications found for '%s' — nothing to resolve", drug_name)
        return []

    indications = [f["indication"] for f in fetched]
    llm_ot_name_map: dict[str, str] = {
        f["indication"]: f["llm_ot_name"] for f in fetched if f.get("llm_ot_name")
    }

    # Step 2: Check existing mappings — keyed by NORMALIZED text so a
    # spelling/hyphenation variant of an already-resolved indication
    # (e.g. "Pre-diabetes" showing up after "Prediabetes" was already
    # resolved in a prior run) reuses that mapping instead of being
    # treated as brand new and re-resolved independently.
    #
    # IMPORTANT: null mappings (ot_disease is None/empty — previously
    # rejected or unresolved) are NOT treated as "already resolved".
    # They are skipped here so the indication gets re-attempted with the
    # current (possibly improved) resolution logic, rather than staying
    # permanently frozen as unresolved from a prior run.
    existing_raw = fetch_existing_mappings(OT_DISEASE_TABLE, "indication")
    existing_norm: dict[str, dict] = {}
    null_mapped_count = 0
    for raw_key, entry in existing_raw.items():
        # Skip entries where ot_disease is null/empty — these are
        # previously-failed resolutions that should be re-attempted.
        if not (entry.get("ot_disease") or "").strip():
            null_mapped_count += 1
            continue
        nk = normalize_indication(raw_key)
        existing_norm.setdefault(nk, entry)  # first writer wins on collision

    if null_mapped_count:
        logger.info(
            "[IND_MAPPING] Skipping %d null-mapped row(s) in %s — these will be re-attempted",
            null_mapped_count, OT_DISEASE_TABLE,
        )

    new_indications = [ind for ind in indications if normalize_indication(ind) not in existing_norm]
    logger.info(
        "[IND_MAPPING] %d indication(s) total, %d already resolved (non-null), %d to resolve (new + previously null)",
        len(indications), len(indications) - len(new_indications), len(new_indications),
    )

    def _existing_lookup(ind: str) -> dict:
        return existing_norm.get(normalize_indication(ind), {})

    if not new_indications:
        logger.info("[IND_MAPPING] All indications already resolved — skipping resolution")
        return [
            {
                "indication": ind,
                "ot_disease": _existing_lookup(ind).get("ot_disease"),
                "ot_disease_id": _existing_lookup(ind).get("ot_disease_id"),
                "resolution_path": _existing_lookup(ind).get("resolution_path", "existing"),
            }
            for ind in indications
        ]

    # Step 3: Normalize and deduplicate by spelling
    norm_to_raw: dict[str, str] = {}
    raw_to_norm: dict[str, str] = {}
    for raw in new_indications:
        nk = normalize_indication(raw)
        raw_to_norm[raw] = nk
        if nk not in norm_to_raw:
            norm_to_raw[nk] = raw
    unique_reps = list(norm_to_raw.values())

    # Step 4: Semantic clustering — groups synonyms that don't share
    # spelling (e.g. "Visceral Adipose Tissue (VAT)" / "Visceral fat")
    # onto one canonical name, with cross-chunk reconciliation.
    cluster_map = cluster_indications(unique_reps, llm_ot_name_map) if len(unique_reps) > 1 else {r: r for r in unique_reps}
    canonical_reps = sorted(set(cluster_map.values()))
    logger.info(
        "[IND_MAPPING] Clustering collapsed %d indication(s) into %d canonical cluster(s)",
        len(unique_reps), len(canonical_reps),
    )
    # A canonical cluster's own LLM-name hint: prefer the hint attached to
    # whichever raw rep happens to share the canonical's exact text, else
    # the first available hint among the cluster's members.
    canonical_hint_map: dict[str, str] = {}
    for rep, canonical in cluster_map.items():
        hint = llm_ot_name_map.get(rep)
        if hint and canonical not in canonical_hint_map:
            canonical_hint_map[canonical] = hint

    # Step 5a: Path A — Gemini semantic matching (against the target's own
    # OT disease list — generally reliable, run on canonical clusters)
    gemini_unresolved = list(canonical_reps)
    # Track resolution path: {indication: (did, name, path_label)}
    resolved_map: dict[str, tuple[str | None, str | None]] = {}
    resolution_paths: dict[str, str] = {}  # indication -> "path_a" / "path_b" / "path_c"

    if target_ensembl_ids:
        logger.info("[IND_MAPPING] Path A: Fetching OT diseases for %d target(s)", len(target_ensembl_ids))
        ot_diseases = fetch_all_target_diseases(target_ensembl_ids)
        if ot_diseases:
            gemini_results = _match_all_against_ot_list(canonical_reps, ot_diseases)
            gemini_unresolved = []
            for ind in canonical_reps:
                did, name = gemini_results.get(ind, (None, None))
                if did:
                    resolved_map[ind] = (did, name)
                    resolution_paths[ind] = "path_a"
                else:
                    gemini_unresolved.append(ind)
            if gemini_unresolved:
                logger.info(
                    "[IND_MAPPING] %d indication(s) unmatched by Gemini — falling back to OT text search",
                    len(gemini_unresolved),
                )
    else:
        logger.info("[IND_MAPPING] Path A skipped (no target_ensembl_ids provided)")

    # Step 5b: Path B — OT text search with retry, seeded with the
    # LLM-suggested OT name hint (searched first) when available.
    if gemini_unresolved:
        logger.info("[IND_MAPPING] Path B: OT text search for %d indication(s)", len(gemini_unresolved))

        def _resolve_one(ind: str) -> tuple[str, str | None, str | None]:
            did, name = _resolve_via_ot_search(ind, llm_ot_name=canonical_hint_map.get(ind))
            if did:
                logger.info("[IND_MAPPING] OT search: '%s' → %s (%s)", ind, did, name)
            else:
                logger.warning("[IND_MAPPING] Could not resolve '%s'", ind)
            return ind, did, name

        with ThreadPoolExecutor(
            max_workers=min(_WORKERS, len(gemini_unresolved)),
            thread_name_prefix="ind-resolve",
        ) as exe:
            futures = {exe.submit(_resolve_one, ind): ind for ind in gemini_unresolved}
            for fut in as_completed(futures):
                ind, did, name = fut.result()
                resolved_map[ind] = (did, name)
                if did:
                    resolution_paths[ind] = "path_b"

    # Step 5c: Path C — Gemini + Google Search grounding with OT API
    # verification for anything Path A and Path B both failed to resolve.
    still_unresolved = [
        ind for ind in canonical_reps
        if not resolved_map.get(ind, (None, None))[0]
    ]
    if still_unresolved:
        path_c_results = resolve_via_search_grounding(still_unresolved)
        for ind, (did, name) in path_c_results.items():
            if did and name:
                resolved_map[ind] = (did, name)
                resolution_paths[ind] = "path_c"

    # Step 6: Validate every resolved mapping (Path A, B, and C alike) —
    # rejects word-overlap false positives (e.g. "renal impairment" ->
    # "renal carcinoma") rather than storing wrong matches.
    # Fail-CLOSED: API errors reject the batch rather than accepting.
    pre_validation_resolved = {
        ind: (did, name) for ind, (did, name) in resolved_map.items() if did and name
    }
    resolved_map = validate_resolved_mappings(resolved_map)

    # Step 6.5: Re-resolution fallback — collect indications that were
    # resolved but then REJECTED by validation, and re-attempt them via
    # Path B (targeted search) and Path C (search grounding). This
    # recovers mappings that were rejected because Path A found an
    # false-positive match, but a better match exists.
    rejected_by_validation = [
        ind for ind in canonical_reps
        if ind in pre_validation_resolved  # was resolved before validation
        and not resolved_map.get(ind, (None, None))[0]  # now null after validation
    ]
    if rejected_by_validation:
        logger.info(
            "[IND_MAPPING] %d mapping(s) rejected by validation — attempting re-resolution",
            len(rejected_by_validation),
        )
        re_resolved, re_paths = _re_resolve_rejected(rejected_by_validation, canonical_hint_map)

        # Validate the re-resolved mappings too (same QC standard)
        if re_resolved:
            re_resolved = validate_resolved_mappings(re_resolved)

        # Merge successful re-resolutions back into the main maps
        for ind, (did, name) in re_resolved.items():
            if did and name:
                resolved_map[ind] = (did, name)
                resolution_paths[ind] = re_paths.get(ind, "retry")

        re_recovered = sum(1 for did, name in re_resolved.values() if did and name)
        logger.info(
            "[IND_MAPPING] Re-resolution recovered %d/%d rejected mapping(s)",
            re_recovered, len(rejected_by_validation),
        )

    # Map canonical cluster results back to every raw variant that fed it
    final_resolved: dict[str, tuple[str | None, str | None]] = {}
    final_paths: dict[str, str] = {}
    for raw in new_indications:
        nk = raw_to_norm[raw]
        rep = norm_to_raw[nk]
        canonical = cluster_map.get(rep, rep)
        final_resolved[raw] = resolved_map.get(canonical, (None, None))
        final_paths[raw] = resolution_paths.get(canonical, "unresolved")

    # Step 7: Push ONLY correctly resolved mappings to BQ — null/unresolved
    # mappings are NOT pushed so the table only contains verified matches.
    # Unresolved indications will be re-attempted on the next run.
    new_rows: list[dict] = []
    pushed_keys: set[str] = set()
    skipped_null = 0
    for raw, (did, name) in final_resolved.items():
        key = raw.strip().lower()
        if key in pushed_keys:
            continue
        pushed_keys.add(key)
        if did and name:
            new_rows.append({
                "indication": raw,
                "ot_disease": name,
                "ot_disease_id": did,
                "resolution_path": final_paths.get(raw, "unresolved"),
            })
        else:
            skipped_null += 1
    if skipped_null:
        logger.info(
            "[IND_MAPPING] Skipping %d unresolved mapping(s) — only pushing %d verified match(es) to BQ",
            skipped_null, len(new_rows),
        )
    if new_rows:
        push_mappings(OT_DISEASE_TABLE, OT_DISEASE_SCHEMA, new_rows)
    else:
        logger.info("[IND_MAPPING] No new verified mappings to push to BQ")

    # Step 8: Return all mappings
    all_mappings: list[dict] = []
    for ind in indications:
        existing_entry = _existing_lookup(ind)
        if existing_entry:
            all_mappings.append({
                "indication": ind,
                "ot_disease": existing_entry.get("ot_disease"),
                "ot_disease_id": existing_entry.get("ot_disease_id"),
                "resolution_path": existing_entry.get("resolution_path", "existing"),
            })
        elif ind in final_resolved:
            did, name = final_resolved[ind]
            all_mappings.append({
                "indication": ind,
                "ot_disease": name,
                "ot_disease_id": did,
                "resolution_path": final_paths.get(ind, "unresolved"),
            })
        else:
            all_mappings.append({
                "indication": ind,
                "ot_disease": None,
                "ot_disease_id": None,
                "resolution_path": "unresolved",
            })

    logger.info("[IND_MAPPING] Completed. %d mapping(s) for '%s'", len(all_mappings), drug_name)
    return all_mappings
