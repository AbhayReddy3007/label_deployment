"""Label Expansion Opportunity PDF report generator.

Mirrors ``serious_safety_profile``'s ``ssp_report.py`` pattern:
- Uses prompt-based section extraction (LLM), with a deterministic fallback
  if the LLM output is unavailable or incomplete.
- Keeps PDF rendering simple and readable (reportlab, bold section titles,
  separator lines).

Structure (adapted from a business-analyst report template): a HEADLINE,
an INDICATION LANDSCAPE paragraph, KEY INSIGHTS (Insight: / Why it
matters:), EXPANSION INDICATIONS (Indication: / Rationale:), EVIDENCE
GAPS & RISKS (bullets), and a BOTTOM LINE - all written in plain business
language with no scores or technical jargon. A summary box up top shows
therapy area count, secondary indication count, and the top Final Score.
A final page ("How This Score Was Calculated") explains the scoring
formula using the drug's own top-scoring opportunity as a worked example
- this page is deliberately the one place scores/formula terms ARE shown,
since explaining them is its entire purpose.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from io import BytesIO
from typing import Any

from google.cloud import bigquery, storage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from medical_potential.config import (
    BQ_DATASET_ID,
    GCS_BUCKET,
    GCS_PIPELINE_CACHE_BASE_PATH,
    GCS_REPORT_BASE_PATH,
    LABEL_EXPANSION_OPPORTUNITY_TABLE,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client

from ..indication_extractor.utils import gemini_generate

logger = logging.getLogger(__name__)

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E75B6")
GREY = colors.HexColor("#666666")
DARK_TEXT = colors.HexColor("#1A1A2E")
LIGHT_BG = colors.HexColor("#F5F7FA")

# This pillar's name, used as the last path segment under both GCS base
# paths: {BASE_PATH}/{drug_name}/{PILLAR_NAME}/...
PILLAR_NAME = "label_expansion_opportunity"

# Columns pulled from LABEL_EXPANSION_OPPORTUNITY_TABLE (config.py) when a
# report is generated standalone (no in-memory score_rows available).
# NOTE: "l_raw_ta" as given did not match this pipeline's actual schema
# (score_calculator.py / LE_SCORE_SCHEMA uses "b_raw_ta", following the
# established b_raw_ind/b_ind, b_raw_ta/b_ta naming pattern) - corrected
# here. Flag if this table's real column is actually named differently.
LE_OPPORTUNITY_COLUMNS = [
    "drug_name", "indication", "therapy_area", "ta_i", "ot_disease_name",
    "trial_id", "phase", "primary_region", "dosage", "drug_arm_size_n",
    "prior", "maturity_weight", "effective_indications", "effective_therapy_areas",
    "w_geo", "w_dose", "w_sample", "q_i", "e_phase_i", "e_i", "link", "link_ta",
    "l_ind", "b_raw_ind", "b_ind", "l_ta", "b_raw_ta", "b_ta", "b",
    "overall_coherence", "c", "final_score", "created_at", "updated_at",
]

# How many of the drug's highest-scoring opportunities to send to the LLM
# and show in the report - keeps the prompt (and the report) readable.
_TOP_N_OPPORTUNITIES = 15

_SECTION_HEADINGS = [
    "HEADLINE",
    "INDICATION LANDSCAPE",
    "KEY INSIGHTS",
    "EXPANSION INDICATIONS",
    "EVIDENCE GAPS & RISKS",
    "BOTTOM LINE",
]


# ==============================
# GCS UPLOAD
# ==============================
_storage_client: storage.Client | None = None


def _get_storage_client() -> storage.Client:
    """Lazily creates a single shared ``storage.Client`` for this process."""
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _upload_bytes(blob_path: str, data: bytes, content_type: str) -> str:
    """Uploads raw bytes to ``gs://{GCS_BUCKET}/{blob_path}``. Returns the
    ``gs://`` URI of the uploaded object. Overwrites any existing object at
    that path."""
    client = _get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
    uri = f"gs://{GCS_BUCKET}/{blob_path}"
    logger.info("[GCS_STORAGE] Uploaded %d byte(s) to %s", len(data), uri)
    return uri


def upload_report_pdf(drug_name: str, pdf_bytes: bytes, filename: str = "report.pdf") -> str:
    """Uploads a PDF report to
    ``{GCS_REPORT_BASE_PATH}/{drug_name}/{PILLAR_NAME}/{filename}``.

    Returns the ``gs://`` URI of the uploaded file.
    """
    blob_path = f"{GCS_REPORT_BASE_PATH}/{drug_name}/{PILLAR_NAME}/{filename}"
    return _upload_bytes(blob_path, pdf_bytes, content_type="application/pdf")


def upload_json_payload(drug_name: str, payload: dict[str, Any], filename: str = "payload.json") -> str:
    """Uploads a JSON payload to
    ``{GCS_PIPELINE_CACHE_BASE_PATH}/{drug_name}/{PILLAR_NAME}/{filename}``.

    Returns the ``gs://`` URI of the uploaded file. Shared by both
    ``generate_report.py`` (its report payload/sections) and
    ``generate_rationale.py`` (its rationale payload), so it lives here
    rather than being duplicated in both files.
    """
    blob_path = f"{GCS_PIPELINE_CACHE_BASE_PATH}/{drug_name}/{PILLAR_NAME}/{filename}"
    data = json.dumps(payload, indent=2, default=str).encode("utf-8")
    return _upload_bytes(blob_path, data, content_type="application/json")


# ==============================
# FETCH (standalone report generation, no in-memory score_rows)
# ==============================
def fetch_score_rows(drug_name: str) -> list[dict]:
    """Fetches this drug's rows directly from ``LABEL_EXPANSION_OPPORTUNITY_TABLE``
    (see ``LE_OPPORTUNITY_COLUMNS`` for exactly which columns), so a report
    can be generated for a drug that was already scored in a previous run,
    without needing a live ``score_rows`` list in memory."""
    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LABEL_EXPANSION_OPPORTUNITY_TABLE}"
    cols = ", ".join(LE_OPPORTUNITY_COLUMNS)

    query = f"""
        SELECT {cols}
        FROM `{table_id}`
        WHERE LOWER(drug_name) = LOWER(@drug_name)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name)]
    )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        rows = [dict(row) for row in results]
    except Exception:
        logger.exception(
            "[GENERATE_REPORT] Failed to fetch '%s' from %s", drug_name, LABEL_EXPANSION_OPPORTUNITY_TABLE,
        )
        return []
    logger.info("[GENERATE_REPORT] Fetched %d row(s) for '%s' from %s", len(rows), drug_name, LABEL_EXPANSION_OPPORTUNITY_TABLE)
    return rows


# ==============================
# STYLES
# ==============================
def escape_html(text: str | None) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=18, leading=22, textColor=NAVY,
        fontName="Helvetica-Bold", spaceAfter=2, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontSize=9.5, leading=12, textColor=GREY,
        fontName="Helvetica", spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader", fontSize=11.5, leading=14, textColor=colors.white,
        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=0, backColor=NAVY,
        alignment=TA_LEFT, borderPadding=(6, 8, 6, 8),
    ))
    styles.add(ParagraphStyle(
        name="Headline", fontSize=12, leading=16, textColor=NAVY,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=8, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="BodyProse", fontSize=9.5, leading=13, textColor=DARK_TEXT,
        fontName="Helvetica", spaceAfter=5, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="InsightHeadline", fontSize=9.5, leading=12, textColor=NAVY,
        fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        name="InsightBody", fontSize=9, leading=12, textColor=DARK_TEXT,
        fontName="Helvetica", spaceAfter=3, leftIndent=10, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="BulletText", fontSize=9, leading=12, textColor=DARK_TEXT,
        fontName="Helvetica", spaceAfter=3, leftIndent=12,
    ))
    styles.add(ParagraphStyle(
        name="SnapshotLabel", fontSize=8, leading=10, textColor=GREY,
        fontName="Helvetica-Bold", spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="SnapshotValue", fontSize=14, leading=16, textColor=NAVY,
        fontName="Helvetica-Bold", spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="FooterText", fontSize=7, leading=9, textColor=GREY, fontName="Helvetica",
    ))
    return styles


def _build_summary_box(styles, num_therapy_areas: int, num_secondary_indications: int, final_score_text: str) -> Table:
    """The box at the top of the report: therapy area count, secondary
    indication count, and the top Final Score."""
    cells = [[
        [
            Paragraph(str(num_therapy_areas), styles["SnapshotValue"]),
            Paragraph("Therapy Areas", styles["SnapshotLabel"]),
        ],
        [
            Paragraph(str(num_secondary_indications), styles["SnapshotValue"]),
            Paragraph("Secondary Indications", styles["SnapshotLabel"]),
        ],
        [
            Paragraph(final_score_text, styles["SnapshotValue"]),
            Paragraph("Final Score", styles["SnapshotLabel"]),
        ],
    ]]
    table = Table(cells, colWidths=[2.17 * inch] * 3, rowHeights=[0.5 * inch])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E0E0E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


# ==============================
# PAYLOAD PREPARATION
# ==============================
def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Builds the full payload used both for the narrative prompt and the
    methodology page.

    ``data`` is expected to be (at minimum) ``{"drug_name": ...}``, plus
    either:
      - ``score_rows``: the drug's LABEL_EXPANSION_OPPORTUNITY_TABLE rows,
        already in memory (e.g. passed straight from ``label_expansion()``'s
        Step 6 output) - used as-is if present; OR fetched live via
        ``fetch_score_rows()`` if absent.
      - ``merged_rows`` (optional): the drug's full LE_TABLE rows (primary
        AND secondary), used only to identify primary indications and to
        look up each secondary indication's ``rationale`` text (neither
        of which exist in LABEL_EXPANSION_OPPORTUNITY_TABLE - see the
        note on ``rationale`` below). If omitted, primary indications and
        rationale text are simply left blank in the narrative.
    """
    if not isinstance(data, dict):
        return {}

    drug_name = data.get("drug_name", "Unknown")

    score_rows = [r for r in (data.get("score_rows") or []) if isinstance(r, dict)]
    if not score_rows and drug_name and drug_name != "Unknown":
        score_rows = fetch_score_rows(drug_name)

    merged_rows = [r for r in (data.get("merged_rows") or []) if isinstance(r, dict)]

    secondary_indications = sorted({r.get("indication") for r in score_rows if r.get("indication")})
    primary_indications = sorted({
        r.get("indication") for r in merged_rows
        if r.get("indication") and (r.get("indication_type") or "").strip().lower() == "primary"
    })
    therapy_areas = sorted({r.get("therapy_area") for r in score_rows if r.get("therapy_area")})

    trial_ids = {r.get("trial_id") for r in score_rows if r.get("trial_id")}
    total_trials = len(trial_ids) if trial_ids else len(score_rows)

    has_regulatory_label = any((r.get("trial_id") or "").strip().endswith("+ fda") for r in score_rows)

    phase_distribution: dict[str, int] = {}
    for r in score_rows:
        p = r.get("phase") or "Unknown"
        phase_distribution[p] = phase_distribution.get(p, 0) + 1

    # NOTE: "rationale" is captured at extraction time (trial_analyser /
    # fda_fetcher / web_analyser) and lives in LE_TABLE, but is NOT one of
    # the columns carried through to LABEL_EXPANSION_OPPORTUNITY_TABLE /
    # score_rows (see LE_OPPORTUNITY_COLUMNS above). It's only available
    # here if the caller also passed merged_rows.
    rationale_by_indication: dict[str, str] = {}
    data_sources_from_merged: set[str] = set()
    for r in merged_rows:
        ind = r.get("indication")
        if ind and ind not in rationale_by_indication and r.get("rationale"):
            rationale_by_indication[ind] = r["rationale"]
        if r.get("data_source"):
            data_sources_from_merged.add(r["data_source"])

    data_sources = sorted({r.get("data_source") for r in score_rows if r.get("data_source")}) or sorted(data_sources_from_merged)

    # Best (highest final_score) row per indication - one entry per
    # opportunity, used for both the narrative and the methodology page.
    best_by_indication: dict[str, dict] = {}
    for r in score_rows:
        ind = r.get("indication")
        if not ind:
            continue
        fs = r.get("final_score")
        existing = best_by_indication.get(ind)
        if existing is None or (fs or 0) > (existing.get("final_score") or 0):
            best_by_indication[ind] = r

    try:
        opportunities = sorted(
            best_by_indication.values(), key=lambda r: float(r.get("final_score") or 0), reverse=True
        )
    except Exception:
        opportunities = list(best_by_indication.values())

    top_final_score = opportunities[0].get("final_score") if opportunities else None

    return {
        "drug_name": drug_name,
        "company_name": data.get("company_name"),  # not collected elsewhere in this pipeline yet
        "primary_indications": primary_indications,
        "secondary_indications": secondary_indications,
        "total_indications": len(primary_indications) + len(secondary_indications),
        "therapy_areas": therapy_areas,
        "num_therapy_areas": len(therapy_areas),
        "num_secondary_indications": len(secondary_indications),
        "total_trials": total_trials,
        "data_sources": data_sources,
        "has_regulatory_label": has_regulatory_label,
        "phase_distribution": phase_distribution,
        "top_final_score": top_final_score,
        "opportunities": opportunities[:_TOP_N_OPPORTUNITIES],
        "all_opportunities": opportunities,  # full list, used by the methodology page
        "rationale_by_indication": rationale_by_indication,
    }


# ==============================
# PROMPT
# ==============================
def _build_narrative_prompt(payload: dict[str, Any]) -> str:
    drug_name = payload.get("drug_name", "Unknown")

    primary_list = ", ".join(payload.get("primary_indications") or []) or "Not separately identified in the available data"
    secondary_list = ", ".join(payload.get("secondary_indications") or []) or "None identified"
    ta_list = ", ".join(payload.get("therapy_areas") or []) or "None identified"
    sources_list = ", ".join(payload.get("data_sources") or []) or "Not specified"

    secondary_rationale_lines = [
        f"- {o.get('indication')} ({o.get('therapy_area')}): "
        f"{payload.get('rationale_by_indication', {}).get(o.get('indication')) or 'No rationale captured in the available data.'}"
        for o in payload.get("opportunities") or []
    ]

    indication_lines = [
        f"- {o.get('indication')} | Therapy Area: {o.get('therapy_area')} | Phase: {o.get('phase') or 'N/A'} | "
        f"Region: {o.get('primary_region') or 'N/A'}"
        for o in (payload.get("all_opportunities") or [])[:30]
    ]

    return f"""You are a senior business analyst preparing a detailed analytical report
on the label expansion dimension of the pharmaceutical molecule "{drug_name}"
for senior business decision-makers.

This dimension evaluates the breadth and depth of the drug's indication landscape —
how many distinct indications it is approved for or being studied in, whether it has
expanded beyond its primary use, how many therapy areas it spans, and what this means
for the drug's commercial and strategic potential.

YOUR OUTPUT MUST FOLLOW THIS EXACT STRUCTURE (use these exact headings):

## HEADLINE
Write ONE impactful sentence summarizing the overall business implication of the
label expansion landscape for this molecule. This should be the single most important
takeaway a decision-maker needs.

## INDICATION LANDSCAPE
Write 3-5 sentences providing the quantitative context a decision-maker needs.
Cover: how many distinct indications exist (primary vs. secondary/expansion),
how many therapy areas the drug spans, what the primary indication(s) are and
what expansion indications are being pursued, what data sources corroborate the
findings (clinical trials, regulatory labels, investor materials, SEC filings),
and the phase maturity of indication-level evidence. Cite specific numbers from
the data. This section sets the stage — it should tell the reader the size and
shape of the indication portfolio before diving into insights.

## KEY INSIGHTS

Provide 4-6 key insights. For EACH insight, write EXACTLY two lines using this format:
Line 1: "Insight: " followed by a short, specific finding (ONE sentence, max 20 words).
         This is the bold headline of the insight.
Line 2: "Why it matters: " followed by the business implication (2-3 sentences,
         ~40-60 words). This is the explanatory body text.

IMPORTANT: You MUST use exactly the labels "Insight: " and "Why it matters: " —
these labels are required for formatting.

Example format:
Insight: Drug spans 4 therapy areas beyond its original metabolic indication.
Why it matters: Multi-therapy-area reach transforms the commercial model from a single-franchise asset to a platform molecule. Each new therapy area unlocks distinct prescriber networks, payer segments, and revenue pools — compounding lifecycle value.

Insight: 3 secondary indications are already in Phase 3, signaling near-term label expansion.
Why it matters: Phase 3 secondary indications with active enrollment represent 12-24 month catalysts for label expansion. Successful readouts would broaden the addressable market and strengthen payer negotiation leverage with real-world evidence of multi-indication utility.

Be specific. Reference indication counts, therapy area breadth, primary vs. secondary
classification, geographic reach, or data source corroboration. Do NOT write generic
statements like "the label is expanding" without concrete data.

Prioritize insights that address:
1. Breadth of indication portfolio and what it enables (platform potential, lifecycle value)
2. Therapy area diversification and cross-specialty commercial opportunity
3. Regulatory label status — approved indications vs. investigational expansions
4. Quality and variety of evidence sources (clinical trials, regulatory filings, investor disclosures)
5. Label expansion gaps that create risk or delay
6. Pipeline maturity of secondary indications (how close to approval)

## EXPANSION INDICATIONS
For EACH secondary (expansion) indication listed in the data below, write EXACTLY
two lines using this format:
Line 1: "Indication: " followed by the indication name and its therapy area in parentheses.
Line 2: "Rationale: " followed by 1-2 sentences explaining WHY this is classified as
         a secondary/expansion indication — i.e., what makes it distinct from the primary
         label, what evidence supports it, and its current development status. Use the
         rationale data provided but rewrite it in clear business language.

IMPORTANT: You MUST use exactly the labels "Indication: " and "Rationale: " —
these labels are required for formatting. Cover ALL secondary indications listed.

## EVIDENCE GAPS & RISKS
Write 3-5 bullet points (each starting with "- ") identifying the most material
gaps in the label expansion profile and the business risk each creates. Focus ONLY
on indication and expansion gaps — for example, missing indications in large
addressable markets, limited expansion beyond primary therapy area, over-reliance
on a single indication for revenue, absence of real-world evidence for newer
indications, or lack of confirmatory data for pipeline expansions.
CRITICAL: Do NOT mention peer-reviewed journals, publications, published literature,
academic publishing, or the need for more published studies. These are NOT relevant
gaps for this report. Every gap must be about missing DATA or missing INDICATIONS.

## BOTTOM LINE
Write 3-4 sentences stating what a decision-maker should infer from this dimension.
Be direct and actionable — state whether the label expansion profile supports
investment, partnership, or market entry decisions, and flag any conditions or
watchpoints. Focus on the strategic value of the indication breadth.

STRICT RULES:
- Total length: 700-1000 words (the report should comfortably fill ~2 pages)
- NO technical jargon (no "Ep", "Et", "scoring", "model", "pipeline page API",
  "BigQuery", "ClinicalTrials.gov API", "Gemini", "LLM")
- Do NOT mention scores of any kind — no Ep, Et, numerical scores, or scoring methodology
- Do NOT mention peer-reviewed journals, publications, or academic publishing anywhere
- You MUST use "Insight: " and "Why it matters: " labels exactly in KEY INSIGHTS
- Every statement must add insight or implication — no restating obvious facts
- Use clear, natural business language that a non-scientific executive can follow
- Do not use markdown bold (**text**) — use plain text only
- Reference specific numbers, indication counts, and therapy areas wherever possible
- Keep paragraphs short (2-4 sentences max)

DATA FOR YOUR ANALYSIS:
======================

Molecule: {drug_name}
Company: {payload.get('company_name') or 'Unknown'}

Total Unique Indications: {payload.get('total_indications', 0)}
Primary Indications: {primary_list}
Secondary (Expansion) Indications: {secondary_list}
Number of Therapy Areas: {payload.get('num_therapy_areas', 0)}
Therapy Areas: {ta_list}
Total Trials / Sources: {payload.get('total_trials', 0)}
Data Sources: {sources_list}
Has Regulatory Label Data: {'Yes' if payload.get('has_regulatory_label') else 'No'}

Phase Distribution: {json.dumps(payload.get('phase_distribution', {}))}

Secondary Indication Details (with rationale for each):
{chr(10).join(secondary_rationale_lines) if secondary_rationale_lines else 'No secondary indications identified'}

Indication Details (first 30):
{chr(10).join(indication_lines) if indication_lines else 'No indication details available'}

Now write the report. Remember: business language, specific numbers, no jargon,
no scores, 700-1000 words.
CRITICAL: In KEY INSIGHTS, every insight headline MUST start with "Insight: "
and every body line MUST start with "Why it matters: "."""


# ==============================
# PARSING (Gemini's raw text -> structured sections)
# ==============================
def _split_into_sections(text: str) -> dict[str, str]:
    """Splits Gemini's raw ``## HEADING`` text into ``{heading: body_text}``."""
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    for line in (text or "").splitlines():
        stripped = line.strip()
        matched = None
        if stripped.startswith("##"):
            heading_text = stripped.lstrip("#").strip().upper()
            for h in _SECTION_HEADINGS:
                if heading_text == h or heading_text.startswith(h):
                    matched = h
                    break
        if matched:
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = matched
            buffer = []
        elif current is not None:
            buffer.append(line)

    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    return sections


def _parse_insights(text: str) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("insight:"):
            if current:
                insights.append(current)
            current = {"insight": stripped.split(":", 1)[1].strip(), "why_it_matters": ""}
        elif stripped.lower().startswith("why it matters:") and current is not None:
            current["why_it_matters"] = stripped.split(":", 1)[1].strip()
    if current:
        insights.append(current)
    return insights


def _parse_expansion_indications(text: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("indication:"):
            if current:
                items.append(current)
            current = {"indication": stripped.split(":", 1)[1].strip(), "rationale": ""}
        elif stripped.lower().startswith("rationale:") and current is not None:
            current["rationale"] = stripped.split(":", 1)[1].strip()
    if current:
        items.append(current)
    return items


def _parse_bullets(text: str) -> list[str]:
    return [
        line.strip().lstrip("-").strip()
        for line in (text or "").splitlines()
        if line.strip().startswith("-")
    ]


# ==============================
# FALLBACK (deterministic, if the LLM call fails)
# ==============================
def _fallback_report_data(payload: dict[str, Any]) -> dict[str, Any]:
    drug_name = payload.get("drug_name", "Unknown")
    num_ind = payload.get("num_secondary_indications", 0)
    num_ta = payload.get("num_therapy_areas", 0)
    opportunities = payload.get("opportunities") or []
    ta_text = ", ".join(payload.get("therapy_areas") or []) or "none identified"

    return {
        "HEADLINE": (
            f"{drug_name} shows {num_ind} candidate label-expansion indication(s) "
            f"across {num_ta} therapy area(s), based on the available evidence."
        ),
        "INDICATION LANDSCAPE": (
            f"{drug_name} has {num_ind} distinct secondary (expansion) indication(s) under "
            f"investigation or reflected in regulatory data, spanning {num_ta} therapy area(s): "
            f"{ta_text}. This assessment draws on {payload.get('total_trials', 0)} underlying "
            f"trial/source record(s)."
        ),
        "KEY INSIGHTS": (
            [{
                "insight": f"{num_ta} therapy area(s) represented among expansion indications.",
                "why_it_matters": (
                    "Broader therapy area reach can diversify commercial exposure across "
                    "prescriber networks and payer segments."
                ),
            }] if num_ta else []
        ),
        "EXPANSION INDICATIONS": [
            {
                "indication": f"{o.get('indication')} ({o.get('therapy_area')})",
                "rationale": (
                    payload.get("rationale_by_indication", {}).get(o.get("indication"))
                    or "No rationale captured in the available data."
                ),
            }
            for o in opportunities
        ],
        "EVIDENCE GAPS & RISKS": [
            "Limited data corroboration is available for some indications; confidence should be weighed accordingly.",
        ],
        "BOTTOM LINE": (
            f"The label expansion profile for {drug_name} reflects {num_ind} candidate "
            f"indication(s) at varying stages of maturity. Further validation of the "
            f"highest-potential opportunities is recommended before committing significant resources."
        ),
    }


def _extract_report_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Generates the narrative report content via Gemini, falling back to a
    deterministic version if the call fails or the required sections/labels
    are missing."""
    prompt = _build_narrative_prompt(payload)
    try:
        text = gemini_generate(
            prompt,
            system_instruction=(
                "You are a senior business analyst. Follow the requested structure "
                "and exact labels precisely. Return plain text only."
            ),
            use_search=False,
        )
        sections = _split_into_sections(text)
        insights = _parse_insights(sections.get("KEY INSIGHTS", ""))
        if not sections.get("HEADLINE") or not insights:
            raise ValueError("Missing required sections/labels in LLM output")
        return {
            "HEADLINE": sections.get("HEADLINE", "").strip(),
            "INDICATION LANDSCAPE": sections.get("INDICATION LANDSCAPE", "").strip(),
            "KEY INSIGHTS": insights,
            "EXPANSION INDICATIONS": _parse_expansion_indications(sections.get("EXPANSION INDICATIONS", "")),
            "EVIDENCE GAPS & RISKS": _parse_bullets(sections.get("EVIDENCE GAPS & RISKS", "")),
            "BOTTOM LINE": sections.get("BOTTOM LINE", "").strip(),
        }
    except Exception:
        logger.exception(
            "[GENERATE_REPORT] Narrative generation failed for '%s' - using fallback",
            payload.get("drug_name"),
        )
        return _fallback_report_data(payload)


# ==============================
# METHODOLOGY PAGE (deterministic - explains the score, not drug-specific narrative)
# ==============================
def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"


def _build_methodology_flowables(payload: dict[str, Any], styles) -> list:
    """A dedicated final page explaining how Final Score is calculated,
    using the drug's top-scoring opportunity as a worked example. Unlike
    the rest of the report, this page IS meant to reference the scoring
    components by name - explaining them is its entire purpose."""
    flowables = [
        PageBreak(),
        Paragraph("HOW THIS SCORE WAS CALCULATED", styles["SectionHeader"]),
        Spacer(1, 8),
    ]

    all_opportunities = payload.get("all_opportunities") or []
    if not all_opportunities:
        flowables.append(Paragraph(
            "No scored opportunities were available to illustrate the calculation.",
            styles["BodyProse"],
        ))
        return flowables

    example = all_opportunities[0]  # highest final_score
    drug_name = payload.get("drug_name", "the drug")

    intro = (
        f"The Final Score for each candidate indication is built from three underlying "
        f"components: how strong the known biological link is between {drug_name}'s "
        f"mechanism and the disease (the association prior), how mature and well-supported "
        f"the clinical evidence is (trial quality and phase), and how broad the opportunity "
        f"is across indications and therapy areas (breadth). These combine into a single "
        f"score from 1 to 5, where higher scores reflect stronger, more mature, and broader "
        f"opportunities."
    )
    flowables.append(Paragraph(escape_html(intro), styles["BodyProse"]))
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        escape_html(f"Worked example: {example.get('indication', 'N/A')} ({example.get('therapy_area', 'N/A')})"),
        ParagraphStyle("worked-example-title", parent=styles["BodyProse"], fontName="Helvetica-Bold", alignment=TA_LEFT),
    ))
    flowables.append(Spacer(1, 4))

    steps = [
        ("1. Association strength (prior)", example.get("prior"),
         "How strong the known link is between the drug's target and this disease."),
        ("2. Evidence maturity (maturity_weight)", example.get("maturity_weight"),
         "How advanced the supporting trial(s) are - later-phase trials score higher."),
        ("3. Trial quality factor (Q)", example.get("q_i"),
         "Combines geographic reach, sample size, and dosing confidence of the supporting trial(s)."),
        ("4. Evidence strength (e_i)", example.get("e_i"),
         "Trial quality combined with how advanced the evidence is."),
        ("5. Combined link", example.get("link"),
         "Association strength and evidence strength combined for this indication."),
        ("6. Therapy area link (link_ta)", example.get("link_ta"),
         "The combined link, averaged across this indication's therapy area, weighted by evidence maturity."),
        ("7. Indication breadth (B_ind)", example.get("b_ind"),
         "Credit for the number of distinct indications this drug has evidence for."),
        ("8. Therapy area breadth (B_ta)", example.get("b_ta"),
         "Credit for the number of distinct therapy areas this drug spans."),
        ("9. Overall breadth (B)", example.get("b"),
         "Indication and therapy area breadth combined."),
        ("10. Overall coherence", example.get("overall_coherence"),
         "How consistently strong the evidence is across this drug's full opportunity set."),
        ("11. Coherence factor (C)", example.get("c"),
         "Overall coherence converted into a scaling factor."),
        ("12. Final Score", example.get("final_score"),
         "Breadth (B) combined with the coherence factor (C), scaled to a 1-5 range."),
    ]

    table_data = [["Step", "Value", "What it represents"]]
    for label, value, desc in steps:
        table_data.append([label, _fmt(value), desc])

    tbl = Table(table_data, colWidths=[1.7 * inch, 0.7 * inch, 4.1 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    flowables.append(tbl)
    flowables.append(Spacer(1, 8))

    final_score = example.get("final_score")
    closing = (
        f"This worked example resulted in a Final Score of {final_score:.2f} out of 5 "
        f"for {example.get('indication', 'this indication')}. The same calculation is "
        f"applied independently to every candidate indication."
        if isinstance(final_score, (int, float)) else
        "The same calculation is applied independently to every candidate indication."
    )
    flowables.append(Paragraph(escape_html(closing), styles["BodyProse"]))

    return flowables


# ==============================
# PDF ASSEMBLY
# ==============================
def _build_narrative_flowables(report_data: dict[str, Any], styles) -> list:
    flowables = []

    # HEADLINE
    flowables.append(Paragraph(escape_html(report_data.get("HEADLINE", "")), styles["Headline"]))
    flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    flowables.append(Spacer(1, 6))

    # INDICATION LANDSCAPE
    flowables.append(Paragraph("INDICATION LANDSCAPE", styles["SectionHeader"]))
    flowables.append(Spacer(1, 8))
    flowables.append(Paragraph(escape_html(report_data.get("INDICATION LANDSCAPE", "")), styles["BodyProse"]))

    # KEY INSIGHTS
    flowables.append(Paragraph("KEY INSIGHTS", styles["SectionHeader"]))
    flowables.append(Spacer(1, 8))
    for item in report_data.get("KEY INSIGHTS", []):
        flowables.append(Paragraph(escape_html(item.get("insight", "")), styles["InsightHeadline"]))
        flowables.append(Paragraph(
            f"<b>Why it matters:</b> {escape_html(item.get('why_it_matters', ''))}", styles["InsightBody"],
        ))

    # EXPANSION INDICATIONS
    flowables.append(Paragraph("EXPANSION INDICATIONS", styles["SectionHeader"]))
    flowables.append(Spacer(1, 8))
    for item in report_data.get("EXPANSION INDICATIONS", []):
        flowables.append(Paragraph(escape_html(item.get("indication", "")), styles["InsightHeadline"]))
        flowables.append(Paragraph(
            f"<b>Rationale:</b> {escape_html(item.get('rationale', ''))}", styles["InsightBody"],
        ))

    # EVIDENCE GAPS & RISKS
    flowables.append(Paragraph("EVIDENCE GAPS &amp; RISKS", styles["SectionHeader"]))
    flowables.append(Spacer(1, 8))
    for bullet in report_data.get("EVIDENCE GAPS & RISKS", []):
        flowables.append(Paragraph(f"&bull; {escape_html(bullet)}", styles["BulletText"]))

    # BOTTOM LINE
    flowables.append(Paragraph("BOTTOM LINE", styles["SectionHeader"]))
    flowables.append(Spacer(1, 8))
    flowables.append(Paragraph(escape_html(report_data.get("BOTTOM LINE", "")), styles["BodyProse"]))

    return flowables


def generate_label_expansion_report_bytes(
    data: dict[str, Any],
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Generate a Label Expansion Opportunity PDF report and return PDF
    bytes plus structured content.

    Returns ``(pdf_bytes, report_content, payload)`` - same shape as
    ``ssp_report.generate_prompt_safety_report_bytes``. This function does
    NOT upload the PDF itself; the caller uploads it via ``upload_report_pdf``.
    """
    payload = _prepare_prompt_payload(data)
    report_data = _extract_report_data(payload)

    drug_name = payload.get("drug_name", "Unknown")
    dimension = "Label Expansion Opportunity"

    buffer = BytesIO()
    styles = build_styles()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        title=f"{drug_name} - {dimension}",
    )

    top_final_score = payload.get("top_final_score")
    final_score_text = f"{top_final_score:.2f}/5" if isinstance(top_final_score, (int, float)) else "N/A"

    story = [
        Paragraph(escape_html(dimension), styles["ReportTitle"]),
        Paragraph(
            f"Drug: <b>{escape_html(drug_name)}</b>&nbsp;&nbsp;|&nbsp;&nbsp;{escape_html(date.today().isoformat())}",
            styles["ReportSubtitle"],
        ),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1, color=NAVY),
        Spacer(1, 6),
        _build_summary_box(
            styles,
            num_therapy_areas=payload.get("num_therapy_areas", 0),
            num_secondary_indications=payload.get("num_secondary_indications", 0),
            final_score_text=final_score_text,
        ),
        Spacer(1, 8),
    ]

    story.extend(_build_narrative_flowables(report_data, styles))
    story.extend(_build_methodology_flowables(payload, styles))

    story.extend([
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")),
        Paragraph(
            f"Report generated {escape_html(date.today().isoformat())}  |  Analytical narrative generated by Gemini",
            styles["FooterText"],
        ),
    ])

    doc.build(story)
    pdf_bytes = buffer.getvalue()

    report_content: dict[str, Any] = {
        "summary_box": {
            "num_therapy_areas": payload.get("num_therapy_areas", 0),
            "num_secondary_indications": payload.get("num_secondary_indications", 0),
            "final_score": top_final_score,
        },
        "sections": report_data,
    }

    return pdf_bytes, report_content, payload
