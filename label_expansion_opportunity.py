"""Label Expansion Opportunity: Main Orchestrator.

Runs the full label-expansion pipeline for exactly one drug:

**Indication Discovery (Steps 1-3):**
    1.  Extract indications from registered clinical trials (trial_analyser).
    1b. Fetch FDA-approved indications (fda_fetcher).
    2.  Extract indications from public web sources (web_analyser).
    3.  Filter non-scorable indications, merge and de-duplicate results,
        push merged rows to BigQuery.

**Open Targets Mapping (Steps 4-5) — Secondary indications only:**
    4.  Resolve the drug's Mechanism(s) of Action to OT target names.
    5.  Resolve Secondary indications to OT disease names.

**Score Calculation (Step 6) — Secondary indications only:**
    6.  Select the best trial per TA-I, compute Final Score, push to BQ.

``label_expansion(molecule_name)`` takes only the drug/molecule name.

Two options come from ``medical_potential/config.py``:

    START_FROM: str              — which stage to start at (see below).
    LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME: str
                                  — dimension name used for GCS uploads and
                                    the DIM_SCORES_TABLE append in Step 7.

(The BQ table holding Mechanism_of_Action is ``DRUG_DETAILS_FULL_TABLE_ID``
in config.py - a fully-qualified ``project.dataset.table`` id used directly
by ``ot_mapping/moa_mapping.py``; this file doesn't need to import it.)

The remaining pipeline options are local constants defined just below the
imports in this file (``LE_TARGET_ENSEMBL_IDS``, ``LE_RUN_OT_MAPPING``,
``LE_GENERATE_REPORT``) — edit them there directly.

There is no module-level default drug name anywhere in this package.
Every function that needs a drug name requires it as an explicit
argument; ``label_expansion(molecule_name)`` is the only place a drug name
is provided, and it is threaded down through every submodule call from
there.

Set ``START_FROM`` to skip earlier stages entirely (e.g. re-run only
indication mapping + scoring against a ``LE_TABLE`` that's already
populated, without re-paying for discovery's search-grounded Gemini calls).

Module responsibilities:
  - ``bq_utils.py``                  — merge module results, push to BigQuery
  - ``indication_extractor/``        — trial_analyser, fda_fetcher, web_analyser
  - ``ot_mapping/``                  — MOA + indication -> Open Targets mapping
  - ``scoring/``                     — trial selection, Final Score calculation
  - ``generate_report_and_rationale/`` — PDF report + rationale generation

This module takes no command-line input and has no default drug. To run
it - from a notebook, a script, or anywhere else in Python - import
``label_expansion`` and call it with a drug name:

    from medical_potential.label_expansion_opportunity.label_expansion_opportunity import label_expansion
    result = label_expansion("Semaglutide")
"""

from __future__ import annotations

import logging

from medical_potential.config import (
    LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME,
    OT_MOA_TABLE,
    START_FROM,
)
from medical_potential.gcp_utils import (
    append_dimension_score_to_bigquery,
    upload_dimension_payload_cache_to_gcs,
    upload_dimension_report_pdf_to_gcs,
)

from .bq_utils import merge_results, push_to_bigquery
from .generate_report_and_rationale import (
    generate_label_expansion_rationale,
    generate_label_expansion_report_bytes,
)
from .indication_extractor import analyse_trials, analyse_web, analyse_fda
from .indication_extractor.utils import filter_scorable_indications
from .ot_mapping import run_moa_mapping, run_indication_mapping
from .ot_mapping.ot_utils import fetch_existing_mappings
from .scoring import run_score_calculation
from .scoring.data_fetcher import fetch_le_rows

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

logger = logging.getLogger("medical_potential.label_expansion_opportunity")
logger.propagate = False
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.handlers = [handler]

# ==============================
# PIPELINE STAGES (for START_FROM)
# ==============================
# Ordered so index comparison ("has stage X already happened by the time
# we START_FROM Y") is a simple list-index lookup.
PIPELINE_STAGES = ("discovery", "moa_mapping", "indication_mapping", "scoring")

# ==============================
# PIPELINE OPTIONS (local — not in config.py)
# ==============================
# Optional target Ensembl IDs for Path A semantic matching in indication
# mapping. If None, these are derived automatically from the MOA mapping
# resolved in Step 4 (or, if Step 4 is skipped via START_FROM, read
# directly from the already-populated OT_MOA_TABLE).
LE_TARGET_ENSEMBL_IDS: list[str] | None = None
# Whether to run Steps 4-6 (Open Targets mapping + scoring).
LE_RUN_OT_MAPPING = True
# Whether to run Step 7 (rationale + PDF report generation, then upload to
# GCS/BigQuery).
LE_GENERATE_REPORT = True


def label_expansion(molecule_name: str | None = None) -> dict:
    """Run the full Label Expansion Opportunity pipeline for one drug.

    Takes only ``molecule_name``. ``START_FROM`` comes from
    ``medical_potential/config.py``; ``LE_TARGET_ENSEMBL_IDS``,
    ``LE_RUN_OT_MAPPING``, and ``LE_GENERATE_REPORT`` are local constants
    defined near the top of this file (see module docstring).

    Usage:
        from medical_potential.label_expansion_opportunity.label_expansion_opportunity import label_expansion
        label_expansion("Semaglutide")

    Executes a 7-step pipeline:

    **Indication Discovery (Steps 1-3):**
        1.  Module 1 — trial_analyser: mines registered clinical trials
            from BigQuery for indications and their trial phase.
        1b. Module 3 — fda_fetcher: fetches FDA-approved indications via
            the openFDA API, with phase = Approved and data_source = Trials.
        2.  Module 2 — web_analyser: uses Gemini + Google Search to find
            label-expansion signals from public web sources.
        3.  Filter & merge & push: drops rows whose "indication" is really a
            trial endpoint/biomarker/PK parameter/procedure, de-duplicates on
            (drug_name, indication, trial_id) preferring trial-sourced rows,
            and upserts into BigQuery.

    **Open Targets Mapping (Steps 4-5) — Secondary indications only:**
        4.  MOA mapping: fetches Mechanism_of_Action from
            ``DRUG_DETAILS_FULL_TABLE_ID`` (config), resolves each to an
            OT target name, pushes to OT_MOA_TABLE.
        5.  Indication mapping: reads Secondary indications from LE_TABLE,
            resolves each to an OT disease name, pushes to OT_DISEASE_TABLE.

    **Score Calculation (Step 6) — Secondary indications only:**
        6.  Selects the best trial per TA-I, computes trial weights and
            a composite Final Score, pushes to LE_SCORE_CALCULATION_TABLE.

    **Report Generation (Step 7):**
        7.  Generates a short plain-text rationale and a 2-page PDF report
            summarizing the drug's label-expansion opportunities. Uploads
            the PDF and a combined JSON cache (rationale + report
            payloads/sections) via the shared ``medical_potential.gcp_utils``
            helpers (``upload_dimension_report_pdf_to_gcs`` /
            ``upload_dimension_payload_cache_to_gcs`` - each also writes a
            timestamped archived copy), and appends the top score + rationale
            to ``DIM_SCORES_TABLE`` via ``append_dimension_score_to_bigquery``.
            Dimension name: ``LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME``
            (from config). Runs only if score rows exist.

    Args:
        molecule_name: The drug / molecule name (e.g. "Semaglutide"). Required -
            raises ``ValueError`` if omitted/``None``, ``TypeError`` if it's
            not a non-empty string.

    ``START_FROM`` (in ``medical_potential/config.py``) controls which stage
    to start at - skips every stage before it, reusing whatever is already
    in BigQuery from a prior run instead of re-discovering/re-resolving it.
    One of:
      - ``"discovery"`` (default): run everything, Steps 1-6.
      - ``"moa_mapping"``: skip Steps 1/1b/2/3 (discovery) entirely.
        Assumes ``LE_TABLE`` already has this drug's indications.
        Starts at Step 4.
      - ``"indication_mapping"``: skip discovery AND Step 4 (MOA
        mapping). Target Ensembl IDs are read directly from
        ``OT_MOA_TABLE`` (already populated by a prior run) unless
        ``LE_TARGET_ENSEMBL_IDS`` is set explicitly. Starts at Step 5.
      - ``"scoring"``: skip everything except Step 6. Assumes both
        ``LE_TABLE`` and ``OT_DISEASE_TABLE`` are already populated
        for this drug.
    Invalid values raise ``ValueError``. Step 7 always runs last
    (governed only by ``LE_GENERATE_REPORT``), regardless of ``START_FROM``.

    Returns:
        dict with keys: ``drug_name``, ``merged_rows``, ``moa_mappings``,
        ``indication_mappings``, ``score_rows``, ``rationale``,
        ``rationale_payload``, ``report_content``, ``report_payload``,
        ``top_final_score``, ``report_gcs_uri`` (the ``gs://`` URI of the
        uploaded PDF, or ``None``), ``report_archive_gcs_uri`` (the
        ``gs://`` URI of the timestamped archived PDF copy, or ``None``),
        and ``cache_gcs_uri`` (the ``gs://`` URI of the uploaded final output
        payload, or ``None``). The raw PDF bytes are not included - fetch the
        PDF from ``report_gcs_uri`` if needed.
    """
    if molecule_name is None:
        raise ValueError(
            "label_expansion() requires a molecule_name argument - none was provided. "
            'Call it as label_expansion("Semaglutide").'
        )
    if not isinstance(molecule_name, str) or not molecule_name.strip():
        raise TypeError(
            f"label_expansion() accepts exactly one molecule name (str), got: {molecule_name!r}"
        )
    if START_FROM not in PIPELINE_STAGES:
        raise ValueError(
            f"START_FROM must be one of {PIPELINE_STAGES}, got: {START_FROM!r}"
        )
    stage_index = PIPELINE_STAGES.index(START_FROM)

    logger.info(
        "[LABEL_EXPANSION] Starting Label Expansion Opportunity pipeline for '%s' (START_FROM=%r)",
        molecule_name, START_FROM,
    )

    # ── Steps 1-3: Indication Discovery ─────────────────────────────────────
    if stage_index > PIPELINE_STAGES.index("discovery"):
        logger.info(
            "[LABEL_EXPANSION] Steps 1-3: Skipped (START_FROM=%r) — reusing indications "
            "already in LE_TABLE for '%s'",
            START_FROM, molecule_name,
        )
        merged_rows = fetch_le_rows(molecule_name)
        n_primary = sum(1 for r in merged_rows if (r.get("indication_type") or "").lower() == "primary")
        n_secondary = sum(1 for r in merged_rows if (r.get("indication_type") or "").lower() == "secondary")
        logger.info(
            "[LABEL_EXPANSION] Found %d existing row(s) in LE_TABLE for '%s' (%d Primary + %d Secondary)",
            len(merged_rows), molecule_name, n_primary, n_secondary,
        )
    else:
        # ── Step 1: Module 1 — Trial Analyser ──────────────────────────────
        logger.info("[LABEL_EXPANSION] Step 1: Extract indications from clinical trials (trial_analyser)")
        trial_rows = analyse_trials(molecule_name)
        logger.info("[LABEL_EXPANSION] Step 1 complete: %d trial-sourced row(s)", len(trial_rows))

        # ── Step 1b: Module 3 — FDA Fetcher ────────────────────────────────
        logger.info("[LABEL_EXPANSION] Step 1b: Fetch FDA-approved indications (fda_fetcher)")
        try:
            fda_rows = analyse_fda(molecule_name)
            logger.info("[LABEL_EXPANSION] Step 1b complete: %d FDA row(s)", len(fda_rows))
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 1b failed for '%s'", molecule_name)
            fda_rows = []

        # Combine trial + FDA rows — both are data_source="Trials"
        all_trial_rows = trial_rows + fda_rows

        # ── Step 2: Module 2 — Web Analyser ────────────────────────────────
        logger.info("[LABEL_EXPANSION] Step 2: Extract indications from web sources (web_analyser)")
        web_rows = analyse_web(molecule_name)
        logger.info("[LABEL_EXPANSION] Step 2 complete: %d web-sourced row(s)", len(web_rows))

        # ── Step 3: Filter non-scorable indications, merge & push to BigQuery ──
        if not all_trial_rows and not web_rows:
            logger.warning(
                "[LABEL_EXPANSION] All modules returned no results for '%s' — nothing to push",
                molecule_name,
            )
            merged_rows = []
        else:
            logger.info("[LABEL_EXPANSION] Step 3: Filter non-scorable indications, merge and push results to BigQuery")
            combined_rows = all_trial_rows + web_rows
            before_count = len(combined_rows)
            scorable_rows = filter_scorable_indications(combined_rows)
            if len(scorable_rows) != before_count:
                logger.info(
                    "[LABEL_EXPANSION] Filtered out %d non-scorable row(s) (endpoints/biomarkers/PK/procedures)",
                    before_count - len(scorable_rows),
                )
            trial_rows_scorable = [r for r in scorable_rows if (r.get("data_source") or "").strip().lower() == "trials"]
            web_rows_scorable = [r for r in scorable_rows if (r.get("data_source") or "").strip().lower() != "trials"]

            merged_rows = merge_results(trial_rows_scorable, web_rows_scorable)
            push_to_bigquery(merged_rows)
            logger.info("[LABEL_EXPANSION] Step 3 complete: %d merged row(s) pushed", len(merged_rows))

        # Count primary vs secondary for logging
        n_primary = sum(1 for r in merged_rows if (r.get("indication_type") or "").lower() == "primary")
        n_secondary = sum(1 for r in merged_rows if (r.get("indication_type") or "").lower() == "secondary")
        logger.info(
            "[LABEL_EXPANSION] %d Primary + %d Secondary indication(s) in LE_TABLE",
            n_primary, n_secondary,
        )

    # ── Step 4: MOA → Open Targets mapping ─────────────────────────────────
    moa_mappings = []
    if LE_RUN_OT_MAPPING and stage_index <= PIPELINE_STAGES.index("moa_mapping"):
        logger.info("[LABEL_EXPANSION] Step 4: Resolve MOA(s) to Open Targets target names")
        try:
            moa_mappings = run_moa_mapping(drug_name=molecule_name)
            logger.info("[LABEL_EXPANSION] Step 4 complete: %d MOA mapping(s)", len(moa_mappings))
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 4 failed for '%s'", molecule_name)
    elif LE_RUN_OT_MAPPING:
        logger.info(
            "[LABEL_EXPANSION] Step 4: Skipped (START_FROM=%r) — reading target Ensembl ID(s) "
            "directly from OT_MOA_TABLE instead",
            START_FROM,
        )
        existing_moas = fetch_existing_mappings(OT_MOA_TABLE, "moa")
        moa_mappings = [
            {"moa": moa, "ot_moa": entry.get("ot_moa"), "ensembl_id": entry.get("ensembl_id")}
            for moa, entry in existing_moas.items()
        ]
        if not moa_mappings:
            logger.warning(
                "[LABEL_EXPANSION] OT_MOA_TABLE has no existing mapping(s) for '%s' — "
                "Step 5 will have no target Ensembl IDs for Path A unless "
                "LE_TARGET_ENSEMBL_IDS is set explicitly",
                molecule_name,
            )
    else:
        logger.info("[LABEL_EXPANSION] Step 4: Skipped (LE_RUN_OT_MAPPING=False)")

    # ── Step 5: Indication → OT disease mapping (Secondary only) ───────────
    indication_mappings = []
    if LE_RUN_OT_MAPPING and stage_index <= PIPELINE_STAGES.index("indication_mapping"):
        effective_target_ids = LE_TARGET_ENSEMBL_IDS
        if effective_target_ids is None:
            effective_target_ids = [m["ensembl_id"] for m in moa_mappings if m.get("ensembl_id")]
            if effective_target_ids:
                logger.info(
                    "[LABEL_EXPANSION] Derived %d target Ensembl ID(s) from MOA mapping for Path A: %s",
                    len(effective_target_ids), effective_target_ids,
                )
            else:
                logger.warning(
                    "[LABEL_EXPANSION] No Ensembl ID resolved from MOA mapping — "
                    "Step 5 will fall back to OT text search only (Path B)"
                )

        logger.info("[LABEL_EXPANSION] Step 5: Resolve Secondary indications to OT disease names")
        try:
            indication_mappings = run_indication_mapping(
                drug_name=molecule_name,
                target_ensembl_ids=effective_target_ids,
                secondary_only=True,
            )
            logger.info(
                "[LABEL_EXPANSION] Step 5 complete: %d indication mapping(s)",
                len(indication_mappings),
            )
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 5 failed for '%s'", molecule_name)
    elif LE_RUN_OT_MAPPING:
        logger.info(
            "[LABEL_EXPANSION] Step 5: Skipped (START_FROM=%r) — reusing indications "
            "already in OT_DISEASE_TABLE",
            START_FROM,
        )
    else:
        logger.info("[LABEL_EXPANSION] Step 5: Skipped (LE_RUN_OT_MAPPING=False)")

    # ── Step 6: Score calculation (Secondary only) ───────────────────────────
    score_rows = []
    if LE_RUN_OT_MAPPING:
        logger.info("[LABEL_EXPANSION] Step 6: Compute label-expansion scores (Secondary only)")
        try:
            score_rows = run_score_calculation(drug_name=molecule_name, push=True, secondary_only=True)
            logger.info("[LABEL_EXPANSION] Step 6 complete: %d TA-I score row(s)", len(score_rows))
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 6 failed for '%s'", molecule_name)
    else:
        logger.info("[LABEL_EXPANSION] Step 6: Skipped (LE_RUN_OT_MAPPING=False)")

    # ── Step 7: Generate rationale and PDF report, upload to GCS ────────────
    rationale = None
    rationale_payload = None
    pdf_bytes = None
    report_content = None
    report_payload = None
    report_gcs_uri = None
    report_archive_gcs_uri = None
    cache_gcs_uri = None
    top_final_score = None
    if LE_GENERATE_REPORT and LE_RUN_OT_MAPPING and score_rows:
        logger.info("[LABEL_EXPANSION] Step 7: Generate rationale and PDF report")
        try:
            report_data = {"drug_name": molecule_name, "score_rows": score_rows, "merged_rows": merged_rows}
            rationale, rationale_payload = generate_label_expansion_rationale(report_data)
            pdf_bytes, report_content, report_payload = generate_label_expansion_report_bytes(report_data)
            logger.info(
                "[LABEL_EXPANSION] Step 7: Generated rationale (%d char(s)) and PDF (%d bytes)",
                len(rationale or ""), len(pdf_bytes or b""),
            )

            try:
                report_gcs_uri, report_archive_gcs_uri = upload_dimension_report_pdf_to_gcs(
                    pdf_bytes, molecule_name, LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME,
                )
                logger.info(
                    "[LABEL_EXPANSION] Step 7: report -> %s (archived: %s)",
                    report_gcs_uri, report_archive_gcs_uri,
                )
            except Exception:
                logger.exception(
                    "[LABEL_EXPANSION] Step 7: report GCS upload failed for '%s' (report/rationale were still generated)",
                    molecule_name,
                )

            try:
                final_scores = [
                    r.get("final_score")
                    for r in score_rows
                    if isinstance(r, dict) and r.get("final_score") is not None
                ]
                top_final_score = max(final_scores) if final_scores else None
                append_dimension_score_to_bigquery(
                    molecule_name=molecule_name,
                    dimension_name=LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME,
                    score=top_final_score,
                    rationale=rationale,
                )
                logger.info(
                    "[LABEL_EXPANSION] Step 7 complete: appended score %s + rationale to DIM_SCORES_TABLE",
                    top_final_score,
                )
            except Exception:
                logger.exception(
                    "[LABEL_EXPANSION] Step 7: failed to append dimension score for '%s'", molecule_name,
                )
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 7 failed for '%s'", molecule_name)
    elif LE_GENERATE_REPORT and LE_RUN_OT_MAPPING and not score_rows:
        logger.info("[LABEL_EXPANSION] Step 7: Skipped — no score rows to report on")
    else:
        logger.info("[LABEL_EXPANSION] Step 7: Skipped (LE_GENERATE_REPORT=False)")

    # ── Final output payload and cache upload ───────────────────────────────
    output = {
        "drug_name": molecule_name,
        "merged_rows": merged_rows,
        "moa_mappings": moa_mappings,
        "indication_mappings": indication_mappings,
        "score_rows": score_rows,
        "report_data": report_data,
        "rationale": rationale,
        "rationale_payload": rationale_payload,
        "report_content": report_content,
        "report_payload": report_payload,
        "top_final_score": top_final_score,
        "report_gcs_uri": report_gcs_uri,
        "report_archive_gcs_uri": report_archive_gcs_uri,
        "cache_gcs_uri": cache_gcs_uri,
    }

    try:
        logger.info("[LABEL_EXPANSION] Uploading final output payload to GCS")
        cache_gcs_uri = upload_dimension_payload_cache_to_gcs(
            output,
            molecule_name,
            LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME,
        )
        output["cache_gcs_uri"] = cache_gcs_uri
        logger.info("[LABEL_EXPANSION] Output payload cache -> %s", cache_gcs_uri)
    except Exception:
        logger.exception(
            "[LABEL_EXPANSION] Output payload GCS upload failed for '%s'",
            molecule_name,
        )

    logger.info(
        "[LABEL_EXPANSION] Pipeline complete for '%s': "
        "%d indication row(s), %d MOA mapping(s), %d indication mapping(s), %d score row(s)",
        molecule_name,
        len(merged_rows),
        len(moa_mappings),
        len(indication_mappings),
        len(score_rows),
    )
    return output
