"""Label Expansion Opportunity PDF report generator.

- Uses prompt-based section extraction (LLM), with a deterministic fallback
  if the LLM output is unavailable or incomplete.
- Keeps PDF rendering simple and readable (reportlab, bold section titles,
  separator lines).

Difference from ``ssp_report.py``: that report summarizes a single safety
score for one molecule. This report summarizes a LIST of label-expansion
opportunities (one row per therapy_area + OT disease combination, i.e. one
per TA-I) for one drug, so the payload and section content are built around
a ranked list rather than a single score.
"""

from __future__ import annotations

import json
from datetime import date
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..indication_extractor.utils import extract_json, gemini_generate

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E75B6")
GREY = colors.HexColor("#666666")
DARK_TEXT = colors.HexColor("#1A1A2E")

SECTION_ORDER = [
    "label_expansion_landscape",
    "top_opportunities",
    "clinical_evidence_summary",
    "therapy_area_breakdown",
]

SECTION_TITLES = {
    "label_expansion_landscape": "Label Expansion Opportunity Landscape",
    "top_opportunities": "Top Opportunities",
    "clinical_evidence_summary": "Clinical Evidence Summary",
    "therapy_area_breakdown": "Therapy Area Breakdown",
}

# How many of the drug's highest-scoring opportunities to send to the LLM
# and show in the report - keeps the prompt (and the report) readable.
_TOP_N_OPPORTUNITIES = 15


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
        name="ReportTitle",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        fontName="Helvetica-Bold",
        spaceAfter=2,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontSize=9.5,
        leading=12,
        textColor=GREY,
        fontName="Helvetica",
        spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=11.5,
        leading=14,
        textColor=colors.white,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=0,
        backColor=NAVY,
        alignment=TA_LEFT,
        leftIndent=0,
        rightIndent=0,
        firstLineIndent=0,
        borderPadding=(6, 8, 6, 8),
    ))
    styles.add(ParagraphStyle(
        name="BodyProse",
        fontSize=9.5,
        leading=13,
        textColor=DARK_TEXT,
        fontName="Helvetica",
        spaceAfter=5,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="SnapshotLabel",
        fontSize=8,
        leading=10,
        textColor=GREY,
        fontName="Helvetica-Bold",
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="SnapshotValue",
        fontSize=10,
        leading=11,
        textColor=DARK_TEXT,
        fontName="Helvetica-Bold",
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="FooterText",
        fontSize=7,
        leading=9,
        textColor=GREY,
        fontName="Helvetica",
    ))
    return styles


def _build_summary_table(styles, top_score_text: str, num_opportunities_text: str) -> Table:
    snap_cells = [[
        [
            Paragraph(top_score_text, ParagraphStyle("score-value", parent=styles["SnapshotValue"], textColor=NAVY)),
            Paragraph("Top Score", styles["SnapshotLabel"]),
        ],
        [
            Paragraph(num_opportunities_text, ParagraphStyle("opp-value", parent=styles["SnapshotValue"], textColor=NAVY)),
            Paragraph("Opportunities Found", styles["SnapshotLabel"]),
        ],
    ]]

    table = Table(snap_cells, colWidths=[3.25 * inch, 3.25 * inch], rowHeights=[0.4 * inch])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E0E0E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _build_section_flowables(section_title: str, section_body: str, styles) -> list:
    flowables = [Paragraph(escape_html(section_title.upper()), styles["SectionHeader"]), Spacer(1, 8)]

    for paragraph in (section_body or "").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        flowables.append(Paragraph(escape_html(paragraph), styles["BodyProse"]))

    flowables.append(Spacer(1, 8))
    return flowables


def _humanize_key(key: str) -> str:
    return str(key).replace("_", " ").strip().capitalize()


def _normalize_section_value(value) -> str:
    """Convert LLM section value to plain descriptive text."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = []
        for item in value:
            item_text = _normalize_section_value(item)
            if item_text:
                parts.append(item_text)
        return " ".join(parts).strip()

    if isinstance(value, dict):
        sentences = []
        for k, v in value.items():
            v_text = _normalize_section_value(v)
            if not v_text:
                continue
            if isinstance(v, (dict, list)):
                sentences.append(v_text)
            else:
                sentences.append(f"{_humanize_key(str(k))}: {v_text}.")
        return " ".join(sentences).strip()

    return str(value).strip()


def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Reduce a drug's label_expansion() result to only what's needed for
    narrative section generation.

    ``data`` is expected to be the dict ``label_expansion()`` returns for
    one drug: ``{"drug_name": ..., "score_rows": [LE_SCORE_CALCULATION_TABLE row, ...]}``.
    """
    if not isinstance(data, dict):
        return {}

    drug_name = data.get("drug_name", "Unknown")
    score_rows = [r for r in (data.get("score_rows") or []) if isinstance(r, dict)]

    opportunities = []
    for row in score_rows:
        try:
            opportunities.append({
                "indication": row.get("indication"),
                "therapy_area": row.get("therapy_area"),
                "ot_disease_name": row.get("ot_disease_name"),
                "final_score": row.get("final_score"),
                "phase": row.get("phase"),
                "primary_region": row.get("primary_region"),
                "association_score": row.get("association_score"),
            })
        except Exception:
            continue

    try:
        sorted_opportunities = sorted(
            opportunities, key=lambda x: float(x.get("final_score") or 0), reverse=True
        )
    except Exception:
        sorted_opportunities = opportunities
    top_opportunities = sorted_opportunities[:_TOP_N_OPPORTUNITIES]

    scores = [o.get("final_score") for o in opportunities if o.get("final_score") is not None]

    # Group by therapy_area for the breakdown section: count + average score.
    ta_groups: dict[str, list[float]] = {}
    for o in opportunities:
        ta = o.get("therapy_area") or "Other"
        if o.get("final_score") is not None:
            ta_groups.setdefault(ta, []).append(float(o["final_score"]))
        else:
            ta_groups.setdefault(ta, [])
    therapy_area_summary = [
        {
            "therapy_area": ta,
            "num_opportunities": sum(1 for o in opportunities if (o.get("therapy_area") or "Other") == ta),
            "avg_final_score": (sum(vals) / len(vals)) if vals else None,
        }
        for ta, vals in ta_groups.items()
    ]

    return {
        "drug_name": drug_name,
        "num_opportunities": len(opportunities),
        "top_opportunities": top_opportunities,
        "score_range": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        },
        "therapy_area_summary": therapy_area_summary,
    }


def _fallback_sections(payload: dict) -> dict:
    """Fallback section content if LLM output is unavailable."""
    drug_name = payload.get("drug_name", "Unknown")
    num_opportunities = payload.get("num_opportunities", 0)
    top_opportunities = payload.get("top_opportunities") or []
    score_range = payload.get("score_range") or {}
    ta_summary = payload.get("therapy_area_summary") or []

    top_names = ", ".join(
        f"{o.get('indication')} ({o.get('final_score')})"
        for o in top_opportunities[:5]
        if o.get("indication")
    ) or "No opportunities were identified in the available evidence."

    ta_lines = "\n".join(
        f"{t.get('therapy_area')}: {t.get('num_opportunities')} opportunity(ies), "
        f"average score {t.get('avg_final_score'):.2f}" if t.get("avg_final_score") is not None
        else f"{t.get('therapy_area')}: {t.get('num_opportunities')} opportunity(ies)"
        for t in ta_summary
    ) or "No therapy area breakdown is available."

    return {
        "label_expansion_landscape": (
            f"This section provides the overall label expansion opportunity landscape for {drug_name} "
            f"based on the available evidence. A total of {num_opportunities} candidate indication(s) "
            f"were identified, with scores ranging from {score_range.get('min', 'N/A')} to {score_range.get('max', 'N/A')}."
        ),
        "top_opportunities": (
            f"The highest-scoring candidate indications for {drug_name} are: {top_names}."
        ),
        "clinical_evidence_summary": (
            "Clinical evidence for these opportunities is summarized from the extracted trial and "
            "regulatory data, considering trial phase, region, and the strength of the known "
            "connection between the drug's mechanism and each disease."
        ),
        "therapy_area_breakdown": ta_lines,
    }


def _extract_sections_via_prompt(payload: dict) -> dict:
    """Generate section narratives with Gemini using the same section
    structure as ``_fallback_sections``."""
    prompt = f"""
You are a business-focused medical insights analyst.
Goal:
    - Create a concise, 2-page report for a given drug focusing on Label Expansion Opportunities, highlighting key findings and insights derived from the provided input json data.
    - The report is intended for a Medical Affairs / business development audience.
Context:
    - The data comes from structured json data listing candidate indications (disease areas) this drug could potentially be developed or approved for, each with a composite opportunity score.
    - The audience is non-technical and not familiar with internal analytical frameworks, scoring methodologies, or internal jargon.

Source:
    - Use only the provided json data as the source of truth.
    - Focus specifically on the label expansion opportunity data points.
    - Do not introduce external assumptions unless clearly derived from the data

Input data (JSON):
{json.dumps(payload, indent=2)}

Return strictly valid JSON with exactly these keys:
- label_expansion_landscape
    - Provide the overall landscape of label expansion opportunities for this drug based on the given data
    - Add details on how many candidate indications were found and the range of scores observed
    - Give a concise brief on how promising the overall opportunity landscape is for this drug
- top_opportunities
    - Highlight the handful of highest-scoring candidate indications and, in plain language, what makes them stand out (e.g. more advanced trial stage, stronger evidence)
    - Do not present this as a list or bullet points; write it as flowing prose
- clinical_evidence_summary
    - Summarize, in plain language, the strength and maturity of the clinical evidence behind the leading opportunities
    - Mention where evidence is still early-stage or limited
- therapy_area_breakdown
    - Describe which therapy areas (disease categories) show the most opportunities and how they compare

Critical output constraints:
- Each of the 4 keys must map to a single paragraph string value.
- Do NOT return nested JSON objects, arrays, or key-value maps for these section keys.
- Do NOT wrap the output in markdown code fences.

Writing rules:
- Be more descriptive for the top_opportunities and clinical_evidence_summary sections

Language and Style Guidelines:
    - Do NOT use internal jargon, scoring framework names, or technical modeling terms (e.g. "final_score", "TA-I", "prior", "link", "maturity_weight").
    - Avoid methodological explanations of how the score was derived.
    - Use clear, simple, business-friendly language.
    - Translate clinical findings into plain-language impact (e.g. what stronger evidence means for development timelines).

Formatting Requirements:
    - Limit the report (combining all sections) to approximately 2 pages of content.
    - Use clear headings and paragraphs.

Tone:
    - Professional, objective, and insight-driven.
    - Focus on clarity, relevance, and business impact
""".strip()

    try:
        response = gemini_generate(
            prompt,
            system_instruction=(
                "You are a business-focused medical insights analyst. Return ONLY valid JSON."
            ),
            use_search=False,
        )
        parsed = extract_json(response)
        if isinstance(parsed, dict):
            sections = {}
            for key in SECTION_ORDER:
                value = parsed.get(key)
                sections[key] = _normalize_section_value(value)
            if all(sections.get(k) for k in SECTION_ORDER):
                return sections
    except Exception:
        pass
    return _fallback_sections(payload)


def generate_label_expansion_report_bytes(
    data: dict[str, Any],
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Generate a Label Expansion Opportunity PDF report and return PDF
    bytes plus structured content.

    Returns ``(pdf_bytes, report_content, payload)`` - same shape as
    ``ssp_report.generate_prompt_safety_report_bytes`` (minus the
    ``model_name`` argument - see the note in ``generate_rationale.py``
    for why). This function does NOT persist/upload the PDF itself; the
    caller is responsible for storing ``pdf_bytes`` wherever reports are
    kept, matching how ``ssp_report.py`` also just returns bytes.
    """
    payload = _prepare_prompt_payload(data)
    sections = _extract_sections_via_prompt(payload)

    drug_name = payload.get("drug_name", "Unknown")
    dimension = "Label Expansion Opportunity"

    buffer = BytesIO()
    styles = build_styles()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        title=f"{drug_name} - {dimension}",
    )

    story = [
        Paragraph(escape_html(dimension), styles["ReportTitle"]),
        Paragraph(
            f"Drug: <b>{escape_html(drug_name)}</b>&nbsp;&nbsp;|&nbsp;&nbsp;{escape_html(date.today().isoformat())}",
            styles["ReportSubtitle"],
        ),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1, color=NAVY),
        Spacer(1, 6),
    ]

    top_score = payload.get("score_range", {}).get("max")
    num_opportunities = payload.get("num_opportunities", 0)
    top_score_text = f"{top_score:.2f}/5" if top_score is not None else "N/A"

    summary_table = {
        "top_score": top_score,
        "num_opportunities": num_opportunities,
    }

    story.append(_build_summary_table(styles, top_score_text=top_score_text, num_opportunities_text=str(num_opportunities)))
    story.append(Spacer(1, 8))

    for key in SECTION_ORDER:
        section_title = SECTION_TITLES[key]
        section_body = sections.get(key, "")
        story.extend(_build_section_flowables(section_title, section_body, styles))

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
        "summary_table": summary_table,
        "sections": sections,
    }

    return pdf_bytes, report_content, payload
