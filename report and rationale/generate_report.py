"""Label Expansion Opportunity PDF report generator.

Mirrors ``serious_safety_profile``'s ``ssp_report.py`` pattern:
- Uses prompt-based section extraction (LLM), with a deterministic fallback
  if the LLM output is unavailable or incomplete.
- Keeps PDF rendering simple and readable (reportlab, bold section titles,
  separator lines).

Structure (adapted from a business-analyst report template): a HEADLINE,
an INDICATION LANDSCAPE paragraph, KEY INSIGHTS (Insight: / plain
explanation), EXPANSION INDICATIONS (grouped by therapy area, as a
table), EVIDENCE GAPS & RISKS (bullets), and a BOTTOM LINE - all written
in plain business language with no scores or technical jargon. A summary
box up top shows therapy area count, secondary indication count, and the
top Final Score. A final set of pages ("How This Score Was Calculated")
walks through the scoring formula step-by-step using the drug's own
top-scoring opportunity as a worked example, plus a full table of every
scored indication's intermediate values - this page is deliberately the
one place scores/formula terms ARE shown, since explaining them is its
entire purpose.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from datetime import date
from io import BytesIO
from typing import Any

from google.cloud import bigquery
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
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

from medical_potential.config import BQ_DATASET_ID, LE_SCORE_CALCULATION_TABLE, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client

from ..indication_extractor.utils import gemini_generate
from ..scoring.trial_selector import phase_rank

logger = logging.getLogger(__name__)

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E75B6")
GREY = colors.HexColor("#666666")
DARK_TEXT = colors.HexColor("#1A1A2E")
LIGHT_BG = colors.HexColor("#F5F7FA")

# Columns actually present in LABEL_EXPANSION_OPPORTUNITY_TABLE (see
# LABEL_EXPANSION_OPPORTUNITY_SCHEMA in bq_utils.py, filled by
# score_calculator.py's push_label_expansion_opportunity() immediately
# after LE_SCORE_CALCULATION_TABLE - a curated subset of that fuller
# table). Used only when a report is generated standalone (no in-memory
# score_rows available). Note this subset does NOT include trial-level
# detail (trial_id, primary_region, dosage, drug_arm_size_n) or the raw
# trial-quality weights (w_geo, w_dose, w_sample) - those are only
# available when score_rows comes from a live pipeline run in memory.
LE_OPPORTUNITY_COLUMNS = [
    "drug_name", "indication", "ot_disease_name", "therapy_area", "ta_i",
    "phase", "association_score", "prior", "maturity_weight",
    "effective_indications", "effective_therapy_areas", "q_i", "e_phase_i",
    "e_i", "link_ta", "b_ind", "b_ta", "b", "overall_coherence", "c",
    "final_score",
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
    """Fetches this drug's rows from the latest run in ``LE_SCORE_CALCULATION_TABLE``
    (see ``LE_OPPORTUNITY_COLUMNS`` for exactly which columns), so a report
    can be generated for a drug that was already scored in a previous run,
    without needing a live ``score_rows`` list in memory.

    Since ``LE_SCORE_CALCULATION_TABLE`` is append-only (historical runs
    accumulate), rows are filtered to the latest ``created_at`` timestamp
    for this drug so only the most recent pipeline run is used."""
    bq_client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LE_SCORE_CALCULATION_TABLE}"
    cols = ", ".join(LE_OPPORTUNITY_COLUMNS)

    query = f"""
        SELECT {cols}
        FROM `{table_id}`
        WHERE LOWER(drug_name) = LOWER(@drug_name)
          AND created_at = (
              SELECT MAX(created_at)
              FROM `{table_id}`
              WHERE LOWER(drug_name) = LOWER(@drug_name)
          )
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("drug_name", "STRING", drug_name)]
    )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        rows = [dict(row) for row in results]
    except Exception:
        logger.exception(
            "[GENERATE_REPORT] Failed to fetch '%s' from %s", drug_name, LE_SCORE_CALCULATION_TABLE,
        )
        return []
    logger.info("[GENERATE_REPORT] Fetched %d row(s) for '%s' from %s", len(rows), drug_name, LE_SCORE_CALCULATION_TABLE)
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
    """A dedicated final page explaining what drove the Final Score, in
    plain business language with no formulas or step-by-step math.
    Identifies which therapy area is pulling the score up and which is
    pulling it down, using each therapy area's Link value (the only
    quantity in the model that actually varies by therapy area - Breadth,
    Coherence, and the Final Score itself are single drug-level numbers)."""
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

    drug_name = payload.get("drug_name", "the drug")

    # Link value per therapy area (uniform within each therapy area already).
    ta_link: dict[str, float] = {}
    for o in all_opportunities:
        ta = o.get("therapy_area")
        lt = o.get("link_ta")
        if ta and isinstance(lt, (int, float)) and ta not in ta_link:
            ta_link[ta] = lt

    # Breadth, Coherence, and Final Score are single drug-level numbers -
    # identical on every row - so any row carries the values for the whole drug.
    reference_row = all_opportunities[0]
    breadth = reference_row.get("b")
    coherence = reference_row.get("overall_coherence")
    final_score = reference_row.get("final_score")
    final_score_text = f"{final_score:.2f} out of 5" if isinstance(final_score, (int, float)) else "N/A"

    sub_header_style = ParagraphStyle(
        "MethodologySubHeader", parent=styles["InsightHeadline"], fontSize=10, leading=13,
        textColor=BLUE, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4,
    )
    cell_style = ParagraphStyle(
        "MethodologyCellText", parent=styles["BodyProse"], fontSize=8.5, leading=11,
        alignment=TA_LEFT, spaceAfter=0,
    )

    intro = (
        f"{drug_name}'s Final Score comes down to two things: how strong and consistent "
        f"the clinical evidence is across its candidate indications, and how broad the "
        f"opportunity is across different therapy areas. The two combine into a single "
        f"score from 1 to 5 - the stronger and broader the evidence, the higher the score."
    )
    flowables.append(Paragraph(escape_html(intro), styles["BodyProse"]))
    flowables.append(Spacer(1, 6))

    if len(ta_link) >= 2:
        best_ta = max(ta_link, key=ta_link.get)
        worst_ta = min(ta_link, key=ta_link.get)
        if best_ta != worst_ta:
            driver_text = (
                f"The score is being pulled up mainly by <b>{escape_html(best_ta)}</b>, "
                f"where the clinical evidence is strongest and most consistent. It is being "
                f"held back by <b>{escape_html(worst_ta)}</b>, where the evidence is comparatively "
                f"weaker or less mature. Strengthening the evidence in {escape_html(worst_ta)} - "
                f"through further trials or regulatory progress - would have the biggest impact "
                f"on raising the overall score."
            )
        else:
            driver_text = f"Evidence strength is broadly consistent across all of {escape_html(drug_name)}'s therapy areas."
        flowables.append(Paragraph(driver_text, styles["BodyProse"]))
        flowables.append(Spacer(1, 6))
    elif len(ta_link) == 1:
        only_ta = next(iter(ta_link))
        flowables.append(Paragraph(
            escape_html(
                f"All of {drug_name}'s scored evidence currently sits within a single "
                f"therapy area, {only_ta}."
            ),
            styles["BodyProse"],
        ))
        flowables.append(Spacer(1, 6))

    breadth_text = (
        f"Two other factors set the ceiling on the score. Breadth reflects how many "
        f"indications and therapy areas {escape_html(drug_name)} has credible evidence for - "
        f"the wider the reach, the more this contributes. Coherence reflects how "
        f"consistently strong that evidence is across the board - a few very strong "
        f"therapy areas can lift coherence even if others are weaker, but scattered, "
        f"low-confidence evidence pulls it down. Combined, these put {escape_html(drug_name)}'s "
        f"Final Score at <b>{final_score_text}</b>."
    )
    flowables.append(Paragraph(breadth_text, styles["BodyProse"]))
    flowables.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # Therapy-area-level summary table
    # ------------------------------------------------------------------
    flowables.append(Paragraph("Score By Therapy Area", sub_header_style))
    flowables.append(Spacer(1, 4))

    table_data = [["Therapy Area", "Link (TA)", "Breadth", "Coherence", "Final Score"]]
    for ta in sorted(ta_link.keys()):
        table_data.append([
            Paragraph(escape_html(ta), cell_style),
            _fmt(ta_link[ta]),
            _fmt(breadth),
            _fmt(coherence),
            f"{final_score:.2f}" if isinstance(final_score, (int, float)) else "N/A",
        ])

    tbl = Table(
        table_data,
        colWidths=[2.4 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    flowables.append(tbl)
    flowables.append(Spacer(1, 6))

    flowables.append(Paragraph(
        escape_html(
            "Link (TA) is the strength of the clinical evidence within that therapy area. "
            "Breadth and Coherence are calculated once for the drug as a whole (not per "
            "therapy area), which is why they repeat across every row - they're included "
            "here so each therapy area's evidence strength can be read alongside the "
            "factors that turn it into the overall Final Score."
        ),
        ParagraphStyle("methodology-legend", parent=styles["BodyProse"], fontSize=7.5, leading=10, textColor=GREY),
    ))

    return flowables


# ==============================
# PDF ASSEMBLY
# ==============================
# Fixed phase buckets every therapy area is broken into - one row each,
# always in this order, using the same phase_rank(...) bucketing the
# scoring model itself uses (1=Phase 1 ... 4=Approved/Phase 4).
_PHASE_BUCKETS = [(1, "Phase 1"), (2, "Phase 2"), (3, "Phase 3"), (4, "Approved")]


def _build_expansion_indications_table(opportunities: list[dict], styles) -> list:
    """Expansion indications grouped by Therapy Area, then by phase.

    Layout: Therapy Area | Phase | Indications
    Each therapy area gets one row per phase bucket that actually HAS at
    least one indication - a phase with no indications for that therapy
    area is skipped entirely rather than shown as a blank/dashed row, so
    a therapy area may end up with anywhere from 1 to 4 rows. The Therapy
    Area cell is merged (SPAN) and vertically + horizontally centered
    across however many rows that therapy area ends up with, so it's
    shown once rather than repeated. Indication cells are wrapped in
    Paragraphs so long lists wrap within the column instead of
    overflowing it."""
    flowables = [Paragraph("EXPANSION INDICATIONS", styles["SectionHeader"]), Spacer(1, 8)]

    if not opportunities:
        flowables.append(Paragraph("No secondary indications identified.", styles["BodyProse"]))
        return flowables

    cell_style = ParagraphStyle(
        "TableCellText", parent=styles["BodyProse"], fontSize=8.5, leading=11,
        alignment=TA_LEFT, spaceAfter=0,
    )
    ta_cell_style = ParagraphStyle(
        "TATableCellText", parent=cell_style, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    # Group indications under each (therapy area, phase bucket), preserving
    # therapy-area order of first appearance. Every therapy area present in
    # `opportunities` gets its own block - this function is expected to be
    # called with the FULL (uncapped) opportunity list, not a top-N slice,
    # so no therapy area is silently dropped.
    ta_phase_to_indications: OrderedDict[str, dict[int, list[str]]] = OrderedDict()
    for o in opportunities:
        ta = o.get("therapy_area") or "N/A"
        ind = o.get("indication") or "N/A"
        rank = phase_rank(o.get("phase"))
        if rank not in (1, 2, 3, 4):
            continue  # phase not recognized - excluded from the phase-bucketed table
        if ta not in ta_phase_to_indications:
            ta_phase_to_indications[ta] = {1: [], 2: [], 3: [], 4: []}
        if ind not in ta_phase_to_indications[ta][rank]:
            ta_phase_to_indications[ta][rank].append(ind)

    table_data = [["Therapy Area", "Phase", "Indications"]]
    span_commands: list[tuple] = []
    background_commands: list[tuple] = []
    row_idx = 1  # row 0 is the header
    striped = False

    for ta, phase_map in ta_phase_to_indications.items():
        # Only phases with at least one indication get a row - an empty
        # phase for this therapy area is skipped entirely.
        populated_phases = [(rank, label) for rank, label in _PHASE_BUCKETS if phase_map.get(rank)]
        if not populated_phases:
            continue

        start_row = row_idx
        for rank, label in populated_phases:
            indications = phase_map[rank]
            table_data.append([
                Paragraph(escape_html(ta), ta_cell_style) if rank == populated_phases[0][0] else "",
                label,
                Paragraph(escape_html(", ".join(indications)), cell_style),
            ])
            row_idx += 1
        end_row = row_idx - 1
        # Merge the Therapy Area cell across however many phase rows this
        # therapy area ended up with (a no-op if there's only one row).
        span_commands.append(("SPAN", (0, start_row), (0, end_row)))
        # Stripe by TA block (not by individual row) so the alternating
        # background reads cleanly across the merged cell.
        background_commands.append(
            ("BACKGROUND", (0, start_row), (-1, end_row), LIGHT_BG if striped else colors.white)
        )
        striped = not striped

    tbl = Table(table_data, colWidths=[1.6 * inch, 0.9 * inch, 4.0 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        *span_commands,
        *background_commands,
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
    flowables.extend(_build_expansion_indications_table(payload.get("all_opportunities") or [], styles))
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
