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

from google.cloud import bigquery
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

from medical_potential.config import BQ_DATASET_ID, LABEL_EXPANSION_OPPORTUNITY_TABLE, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client

from ..indication_extractor.utils import gemini_generate
from .score_calculator import SCORING_FORMULAS, get_methodology_text

logger = logging.getLogger(__name__)

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E75B6")
GREY = colors.HexColor("#666666")
DARK_TEXT = colors.HexColor("#1A1A2E")
LIGHT_BG = colors.HexColor("#F5F7FA")

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
    "EVIDENCE GAPS & RISKS",
    "BOTTOM LINE",
]


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
        fontName="Helvetica-Bold", spaceBefore=7, spaceAfter=0, backColor=NAVY,
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

    indication_lines = [
        f"- {o.get('indication')} | Therapy Area: {o.get('therapy_area')} | Phase: {o.get('phase') or 'N/A'} | "
        f"Region: {o.get('primary_region') or 'N/A'}"
        for o in (payload.get("all_opportunities") or [])[:15]
    ]

    return f"""You are a senior business analyst preparing a concise analytical report
on the label expansion dimension of the pharmaceutical molecule "{drug_name}"
for senior business decision-makers. This report MUST fit within 2 pages of
narrative content (a separate scoring-methodology page follows after yours,
so keep this tight and avoid padding).

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
Write 3-4 sentences providing the quantitative context a decision-maker needs.
Cover: how many distinct indications exist (primary vs. secondary/expansion),
how many therapy areas the drug spans, what the primary indication(s) are and
what expansion indications are being pursued, what data sources corroborate the
findings (clinical trials, regulatory labels, investor materials, SEC filings),
and the phase maturity of indication-level evidence. Cite specific numbers from
the data. This section sets the stage — it should tell the reader the size and
shape of the indication portfolio before diving into insights.

## KEY INSIGHTS

Provide 3-5 key insights. For EACH insight, write EXACTLY two lines using this format:
Line 1: "Insight: " followed by a short, specific finding (ONE sentence, max 20 words).
         This is the bold headline of the insight.
Line 2: The business implication, written directly as plain prose (1-2 sentences,
         ~30-40 words) — do NOT prefix this line with any label at all, just
         state the implication directly.

IMPORTANT: You MUST use exactly the label "Insight: " on line 1 of each insight —
this label is required for formatting. Line 2 must NOT have any label or prefix.

Example format:
Insight: Drug spans 4 therapy areas beyond its original metabolic indication.
Multi-therapy-area reach transforms the commercial model into a platform play, unlocking distinct prescriber networks, payer segments, and revenue pools.

Insight: 3 secondary indications are already in Phase 3, signaling near-term label expansion.
Phase 3 secondary indications with active enrollment represent 12-24 month catalysts for label expansion and broader payer leverage.

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

## EVIDENCE GAPS & RISKS
Write 3-4 bullet points (each starting with "- ") identifying the most material
gaps in the label expansion profile and the business risk each creates. Focus ONLY
on indication and expansion gaps — for example, missing indications in large
addressable markets, limited expansion beyond primary therapy area, over-reliance
on a single indication for revenue, absence of real-world evidence for newer
indications, or lack of confirmatory data for pipeline expansions.
CRITICAL: Do NOT mention peer-reviewed journals, publications, published literature,
academic publishing, or the need for more published studies. These are NOT relevant
gaps for this report. Every gap must be about missing DATA or missing INDICATIONS.

## BOTTOM LINE
Write 2-3 sentences stating what a decision-maker should infer from this dimension.
Be direct and actionable — state whether the label expansion profile supports
investment, partnership, or market entry decisions, and flag any conditions or
watchpoints. Focus on the strategic value of the indication breadth.

STRICT RULES:
- Total length: 400-550 words (this report must fit in 2 pages of narrative content)
- NO technical jargon (no "Ep", "Et", "scoring", "model", "pipeline page API",
  "BigQuery", "ClinicalTrials.gov API", "Gemini", "LLM")
- Do NOT mention scores of any kind — no Ep, Et, numerical scores, or scoring methodology
- Do NOT mention peer-reviewed journals, publications, or academic publishing anywhere
- You MUST use the "Insight: " label exactly on line 1 of each KEY INSIGHTS item;
  line 2 must be plain prose with no label
- Every statement must add insight or implication — no restating obvious facts
- Use clear, natural business language that a non-scientific executive can follow
- Do not use markdown bold (**text**) — use plain text only
- Reference specific numbers, indication counts, and therapy areas wherever possible
- Keep paragraphs short (2-3 sentences max)
- Do NOT include a section on individual expansion indications — that is handled
  separately as a table; do not restate the full indication list in prose

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

Indication Details (first 15):
{chr(10).join(indication_lines) if indication_lines else 'No indication details available'}

Now write the report. Remember: business language, specific numbers, no jargon,
no scores, 400-550 words.
CRITICAL: In KEY INSIGHTS, every insight headline MUST start with "Insight: " and
the following line must be plain prose with no "Why it matters" or any other label."""


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
    """Parses ``Insight: <headline>`` lines, where everything following an
    "Insight:" line (up to the next "Insight:" line or end of block) is
    that insight's plain-prose explanation - no "Why it matters:" or any
    other label required on the explanation line(s)."""
    insights: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("insight:"):
            if current:
                current["explanation"] = current["explanation"].strip()
                insights.append(current)
            current = {"insight": stripped.split(":", 1)[1].strip(), "explanation": ""}
        elif current is not None and stripped:
            current["explanation"] = f"{current['explanation']} {stripped}".strip()
    if current:
        current["explanation"] = current["explanation"].strip()
        insights.append(current)
    return insights


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
                "explanation": (
                    "Broader therapy area reach can diversify commercial exposure across "
                    "prescriber networks and payer segments."
                ),
            }] if num_ta else []
        ),
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
    """A dedicated set of pages explaining how Final Score is calculated,
    showing every intermediate calculation stage with the drug's actual
    numbers worked out step-by-step. Unlike the rest of the report, these
    pages ARE meant to reference the scoring components by name - explaining
    them is their entire purpose."""
    flowables = [
        PageBreak(),
        Paragraph("HOW THIS SCORE WAS CALCULATED", styles["SectionHeader"]),
        Spacer(1, 8),
    ]

    opportunities = payload.get("opportunities") or []
    if not opportunities:
        flowables.append(Paragraph(
            "No scored opportunities were available to illustrate the calculation.",
            styles["BodyProse"],
        ))
        return flowables

    drug_name = payload.get("drug_name", "the drug")

    methodology = get_methodology_text(drug_name)
    flowables.append(Paragraph(escape_html(methodology["intro"]), styles["BodyProse"]))
    flowables.append(Spacer(1, 6))

    # Use the top-scoring opportunity as the worked example
    top = opportunities[0]
    top_ind = top.get("indication") or "N/A"

    # Helper styles for formula display
    formula_style = ParagraphStyle(
        "FormulaText", parent=styles["BodyProse"], fontSize=9, leading=12,
        textColor=NAVY, fontName="Helvetica-Bold", leftIndent=10, spaceAfter=2,
    )
    calc_style = ParagraphStyle(
        "CalcText", parent=styles["BodyProse"], fontSize=8.5, leading=11,
        textColor=DARK_TEXT, fontName="Helvetica", leftIndent=15, spaceAfter=4,
    )
    stage_header_style = ParagraphStyle(
        "StageHeader", parent=styles["InsightHeadline"], fontSize=10, leading=13,
        textColor=BLUE, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4,
    )

    flowables.append(Paragraph(
        f"Worked example using the top-scoring indication: <b>{escape_html(top_ind)}</b>",
        styles["InsightHeadline"],
    ))
    flowables.append(Spacer(1, 4))

    # ------------------------------------------------------------------
    # STAGE 2: Evidence Strength (e_i)
    # ------------------------------------------------------------------
    flowables.append(Paragraph("Stage 2: Evidence Strength (e_i)", stage_header_style))

    q_i = top.get("q_i")
    w_geo = top.get("w_geo")
    w_dose = top.get("w_dose")
    w_sample = top.get("w_sample")
    e_phase_i = top.get("e_phase_i")
    e_i = top.get("e_i")

    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["q_i"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"= {_fmt(w_geo)} x {_fmt(w_sample)} x {_fmt(w_dose)} = <b>{_fmt(q_i)}</b>",
        calc_style,
    ))

    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["e_phase_i"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"e_phase_i = <b>{_fmt(e_phase_i)}</b> "
        f"(Phase: {escape_html(str(top.get('phase') or 'N/A'))})",
        calc_style,
    ))

    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["e_i"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"= {_fmt(q_i)} x {_fmt(e_phase_i)} = <b>{_fmt(e_i)}</b>",
        calc_style,
    ))

    # ------------------------------------------------------------------
    # STAGE 3: Link
    # ------------------------------------------------------------------
    flowables.append(Paragraph("Stage 3: Link", stage_header_style))

    prior = top.get("prior")
    link = top.get("link")

    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["link"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"= 1 - (1 - {_fmt(prior)}) x (1 - {_fmt(e_i)}) = <b>{_fmt(link)}</b>",
        calc_style,
    ))
    flowables.append(Paragraph(
        f"Prior = {_fmt(prior)} (from association score); "
        f"e_i = {_fmt(e_i)} (evidence strength computed above)",
        ParagraphStyle("calc-note", parent=calc_style, fontSize=7.5, textColor=GREY),
    ))

    # ------------------------------------------------------------------
    # STAGE 4: Breadth (B)
    # ------------------------------------------------------------------
    flowables.append(Paragraph("Stage 4: Breadth (B)", stage_header_style))

    eff_ind = top.get("effective_indications")
    eff_ta = top.get("effective_therapy_areas")
    b_ind = top.get("b_ind")
    b_ta = top.get("b_ta")
    l_ind = top.get("l_ind")
    b_raw_ind = top.get("b_raw_ind")
    l_ta = top.get("l_ta")
    b_raw_ta = top.get("b_raw_ta")
    b = top.get("b")

    # Indication Breadth
    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["b_ind"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"Effective Indications = {_fmt(eff_ind)}",
        calc_style,
    ))
    flowables.append(Paragraph(
        f"L_ind({_fmt(eff_ind)}) = {_fmt(l_ind)}, "
        f"B_raw_ind = {_fmt(b_raw_ind)}, "
        f"B_ind = <b>{_fmt(b_ind)}</b>",
        calc_style,
    ))

    # Therapy Area Breadth
    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["b_ta"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"Effective Therapy Areas = {_fmt(eff_ta)}",
        calc_style,
    ))
    flowables.append(Paragraph(
        f"L_TA({_fmt(eff_ta)}) = {_fmt(l_ta)}, "
        f"B_raw_TA = {_fmt(b_raw_ta)}, "
        f"B_TA = <b>{_fmt(b_ta)}</b>",
        calc_style,
    ))

    # Combined Breadth
    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["b"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"= {_fmt(b_ind)} x {_fmt(b_ta)} = <b>{_fmt(b)}</b>",
        calc_style,
    ))

    # ------------------------------------------------------------------
    # STAGE 5: Coherence (C)
    # ------------------------------------------------------------------
    flowables.append(Paragraph("Stage 5: Coherence (C)", stage_header_style))

    overall_coherence = top.get("overall_coherence")
    c = top.get("c")

    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["overall_coherence"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"Overall Coherence = <b>{_fmt(overall_coherence)}</b>",
        calc_style,
    ))

    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["c"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"= 0.1 + 0.9 x ({_fmt(overall_coherence)})^1.75 = <b>{_fmt(c)}</b>",
        calc_style,
    ))

    # ------------------------------------------------------------------
    # FINAL SCORE
    # ------------------------------------------------------------------
    flowables.append(Paragraph("Final Score", stage_header_style))

    final_score = top.get("final_score")

    flowables.append(Paragraph(
        escape_html(SCORING_FORMULAS["final_score"]["formula"]),
        formula_style,
    ))
    flowables.append(Paragraph(
        f"= 1 + 4 x {_fmt(b)} x {_fmt(c)} = <b>{f'{final_score:.2f}' if isinstance(final_score, (int, float)) else 'N/A'}</b>",
        calc_style,
    ))
    flowables.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # FULL TABLE: all indications with every intermediate value
    # ------------------------------------------------------------------
    flowables.append(Paragraph("All Scored Indications", stage_header_style))
    flowables.append(Spacer(1, 4))

    table_data = [["Indication", "Prior", "Maturity", "Q_i", "e_phase_i", "e_i", "Link", "Breadth", "Coherence", "Final"]]
    for o in opportunities:
        fs = o.get("final_score")
        table_data.append([
            o.get("indication") or "N/A",
            _fmt(o.get("prior")),
            _fmt(o.get("maturity_weight")),
            _fmt(o.get("q_i")),
            _fmt(o.get("e_phase_i")),
            _fmt(o.get("e_i")),
            _fmt(o.get("link")),
            _fmt(o.get("b")),
            _fmt(o.get("c")),
            f"{fs:.2f}" if isinstance(fs, (int, float)) else "N/A",
        ])

    tbl = Table(
        table_data,
        colWidths=[1.35 * inch, 0.5 * inch, 0.55 * inch, 0.5 * inch, 0.55 * inch, 0.5 * inch, 0.5 * inch, 0.55 * inch, 0.6 * inch, 0.5 * inch],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    flowables.append(tbl)
    flowables.append(Spacer(1, 6))

    legend = methodology["legend"]
    flowables.append(Paragraph(
        escape_html(legend),
        ParagraphStyle("methodology-legend", parent=styles["BodyProse"], fontSize=7.5, leading=10, textColor=GREY),
    ))

    return flowables


# ==============================
# PDF ASSEMBLY
# ==============================
def _build_expansion_indications_table(opportunities: list[dict], styles) -> list:
    """Expansion indications grouped by Therapy Area.

    Layout: Therapy Area | Indications
    Each therapy area row lists all its indications (comma-separated),
    so the reader sees the portfolio organised by therapeutic domain."""
    flowables = [Paragraph("EXPANSION INDICATIONS", styles["SectionHeader"]), Spacer(1, 8)]

    if not opportunities:
        flowables.append(Paragraph("No secondary indications identified.", styles["BodyProse"]))
        return flowables

    # Group indications under each therapy area, preserving order
    from collections import OrderedDict
    ta_to_indications: OrderedDict[str, list[str]] = OrderedDict()
    for o in opportunities:
        ta = o.get("therapy_area") or "N/A"
        ind = o.get("indication") or "N/A"
        if ta not in ta_to_indications:
            ta_to_indications[ta] = []
        if ind not in ta_to_indications[ta]:
            ta_to_indications[ta].append(ind)

    table_data = [["Therapy Area", "Indications"]]
    for ta, indications in ta_to_indications.items():
        table_data.append([
            ta,
            ", ".join(indications),
        ])

    tbl = Table(table_data, colWidths=[2.2 * inch, 4.3 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    flowables.append(tbl)
    return flowables


def _build_narrative_flowables(report_data: dict[str, Any], payload: dict[str, Any], styles) -> list:
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
        flowables.append(Paragraph(escape_html(item.get("explanation", "")), styles["InsightBody"]))

    # EXPANSION INDICATIONS - deterministic table, no LLM/rationale
    flowables.extend(_build_expansion_indications_table(payload.get("opportunities") or [], styles))
    flowables.append(Spacer(1, 8))

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
    NOT upload the PDF itself; the caller uploads it via
    ``medical_potential.gcp_utils.upload_dimension_report_pdf_to_gcs``.
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

    story.extend(_build_narrative_flowables(report_data, payload, styles))
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
