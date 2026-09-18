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

Use ``start_from`` to skip earlier stages entirely (e.g. re-run only
indication mapping + scoring against a ``LE_TABLE`` that's already
populated, without re-paying for discovery's search-grounded Gemini calls).

Run with:
    python -m medical_potential.label_expansion_opportunity.label_expansion_opportunity
"""

from __future__ import annotations

import logging

from medical_potential.config import DRUG_NAME

from .bq_utils import merge_results, push_to_bigquery
from .indication_extractor import analyse_trials, analyse_web, analyse_fda
from .indication_extractor.utils import filter_scorable_indications
from .ot_mapping import run_moa_mapping, run_indication_mapping
from .ot_mapping.ot_utils import OT_MOA_TABLE, fetch_existing_mappings
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
# PIPELINE STAGES (for start_from)
# ==============================
# Ordered so index comparison ("has stage X already happened by the time
# we start_from Y") is a simple list-index lookup.
PIPELINE_STAGES = ("discovery", "moa_mapping", "indication_mapping", "scoring")


def label_expansion(
    drug_name: str = DRUG_NAME,
    drug_details_table: str = "drug_details",
    target_ensembl_ids: list[str] | None = None,
    run_ot_mapping: bool = True,
    start_from: str = "discovery",
) -> dict:
    """Run the full Label Expansion Opportunity pipeline for one drug.

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
        4.  MOA mapping: fetches Mechanism_of_Action from drug_details table,
            resolves each to an OT target name, pushes to OT_MOA_TABLE.
        5.  Indication mapping: reads Secondary indications from LE_TABLE,
            resolves each to an OT disease name, pushes to OT_DISEASE_TABLE.

    **Score Calculation (Step 6) — Secondary indications only:**
        6.  Selects the best trial per TA-I, computes trial weights and
            a composite Final Score, pushes to LE_SCORE_CALCULATION_TABLE.

    Args:
        drug_name: The drug / molecule name (e.g. "semaglutide").
        drug_details_table: BQ table name holding drug details with
            Mechanism_of_Action column.
        target_ensembl_ids: Optional Ensembl IDs for the drug's gene targets,
            to enable Gemini semantic matching (Path A) in indication mapping.
            If ``None`` (default), these are derived automatically - from the
            MOA mapping resolved in Step 4, or (if Step 4 is skipped via
            ``start_from``) read directly from the already-populated
            ``OT_MOA_TABLE``.
        run_ot_mapping: Whether to run Steps 4-6. Set ``False`` to only
            discover indications without OT resolution or scoring.
        start_from: Which stage to start at - skips every stage before it,
            reusing whatever is already in BigQuery from a prior run
            instead of re-discovering/re-resolving it. One of:
              - ``"discovery"`` (default): run everything, Steps 1-6.
              - ``"moa_mapping"``: skip Steps 1/1b/2/3 (discovery) entirely.
                Assumes ``LE_TABLE`` already has this drug's indications.
                Starts at Step 4.
              - ``"indication_mapping"``: skip discovery AND Step 4 (MOA
                mapping). Target Ensembl IDs are read directly from
                ``OT_MOA_TABLE`` (already populated by a prior run) unless
                ``target_ensembl_ids`` is passed explicitly. Starts at Step 5.
              - ``"scoring"``: skip everything except Step 6. Assumes both
                ``LE_TABLE`` and ``OT_DISEASE_TABLE`` are already populated
                for this drug.
            Invalid values raise ``ValueError``.

    Returns:
        dict with keys: ``drug_name``, ``merged_rows``, ``moa_mappings``,
        ``indication_mappings``, ``score_rows``.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise TypeError(
            f"label_expansion() accepts exactly one drug name (str), got: {drug_name!r}"
        )
    if start_from not in PIPELINE_STAGES:
        raise ValueError(
            f"start_from must be one of {PIPELINE_STAGES}, got: {start_from!r}"
        )
    stage_index = PIPELINE_STAGES.index(start_from)

    logger.info(
        "[LABEL_EXPANSION] Starting Label Expansion Opportunity pipeline for '%s' (start_from=%r)",
        drug_name, start_from,
    )

    # ── Steps 1-3: Indication Discovery ─────────────────────────────────────
    if stage_index > PIPELINE_STAGES.index("discovery"):
        logger.info(
            "[LABEL_EXPANSION] Steps 1-3: Skipped (start_from=%r) — reusing indications "
            "already in LE_TABLE for '%s'",
            start_from, drug_name,
        )
        merged_rows = fetch_le_rows(drug_name)
        n_primary = sum(1 for r in merged_rows if (r.get("indication_type") or "").lower() == "primary")
        n_secondary = sum(1 for r in merged_rows if (r.get("indication_type") or "").lower() == "secondary")
        logger.info(
            "[LABEL_EXPANSION] Found %d existing row(s) in LE_TABLE for '%s' (%d Primary + %d Secondary)",
            len(merged_rows), drug_name, n_primary, n_secondary,
        )
    else:
        # ── Step 1: Module 1 — Trial Analyser ──────────────────────────────
        logger.info("[LABEL_EXPANSION] Step 1: Extract indications from clinical trials (trial_analyser)")
        trial_rows = analyse_trials(drug_name)
        logger.info("[LABEL_EXPANSION] Step 1 complete: %d trial-sourced row(s)", len(trial_rows))

        # ── Step 1b: Module 3 — FDA Fetcher ────────────────────────────────
        logger.info("[LABEL_EXPANSION] Step 1b: Fetch FDA-approved indications (fda_fetcher)")
        try:
            fda_rows = analyse_fda(drug_name)
            logger.info("[LABEL_EXPANSION] Step 1b complete: %d FDA row(s)", len(fda_rows))
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 1b failed for '%s'", drug_name)
            fda_rows = []

        # Combine trial + FDA rows — both are data_source="Trials"
        all_trial_rows = trial_rows + fda_rows

        # ── Step 2: Module 2 — Web Analyser ────────────────────────────────
        logger.info("[LABEL_EXPANSION] Step 2: Extract indications from web sources (web_analyser)")
        web_rows = analyse_web(drug_name)
        logger.info("[LABEL_EXPANSION] Step 2 complete: %d web-sourced row(s)", len(web_rows))

        # ── Step 3: Filter non-scorable indications, merge & push to BigQuery ──
        if not all_trial_rows and not web_rows:
            logger.warning(
                "[LABEL_EXPANSION] All modules returned no results for '%s' — nothing to push",
                drug_name,
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
    if run_ot_mapping and stage_index <= PIPELINE_STAGES.index("moa_mapping"):
        logger.info("[LABEL_EXPANSION] Step 4: Resolve MOA(s) to Open Targets target names")
        try:
            moa_mappings = run_moa_mapping(
                drug_name=drug_name,
                drug_details_table=drug_details_table,
            )
            logger.info("[LABEL_EXPANSION] Step 4 complete: %d MOA mapping(s)", len(moa_mappings))
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 4 failed for '%s'", drug_name)
    elif run_ot_mapping:
        logger.info(
            "[LABEL_EXPANSION] Step 4: Skipped (start_from=%r) — reading target Ensembl ID(s) "
            "directly from OT_MOA_TABLE instead",
            start_from,
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
                "target_ensembl_ids is passed explicitly",
                drug_name,
            )
    else:
        logger.info("[LABEL_EXPANSION] Step 4: Skipped (run_ot_mapping=False)")

    # ── Step 5: Indication → OT disease mapping (Secondary only) ───────────
    indication_mappings = []
    if run_ot_mapping and stage_index <= PIPELINE_STAGES.index("indication_mapping"):
        effective_target_ids = target_ensembl_ids
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
                drug_name=drug_name,
                target_ensembl_ids=effective_target_ids,
                secondary_only=True,
            )
            logger.info(
                "[LABEL_EXPANSION] Step 5 complete: %d indication mapping(s)",
                len(indication_mappings),
            )
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 5 failed for '%s'", drug_name)
    elif run_ot_mapping:
        logger.info(
            "[LABEL_EXPANSION] Step 5: Skipped (start_from=%r) — reusing indications "
            "already in OT_DISEASE_TABLE",
            start_from,
        )
    else:
        logger.info("[LABEL_EXPANSION] Step 5: Skipped (run_ot_mapping=False)")

    # ── Step 6: Score calculation (Secondary only) ───────────────────────────
    score_rows = []
    if run_ot_mapping:
        logger.info("[LABEL_EXPANSION] Step 6: Compute label-expansion scores (Secondary only)")
        try:
            score_rows = run_score_calculation(drug_name=drug_name, push=True, secondary_only=True)
            logger.info("[LABEL_EXPANSION] Step 6 complete: %d TA-I score row(s)", len(score_rows))
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 6 failed for '%s'", drug_name)
    else:
        logger.info("[LABEL_EXPANSION] Step 6: Skipped (run_ot_mapping=False)")

    # ── Done ───────────────────────────────────────────────────────────────
    output = {
        "drug_name": drug_name,
        "merged_rows": merged_rows,
        "moa_mappings": moa_mappings,
        "indication_mappings": indication_mappings,
        "score_rows": score_rows,
    }

    logger.info(
        "[LABEL_EXPANSION] Pipeline complete for '%s': "
        "%d indication row(s), %d MOA mapping(s), %d indication mapping(s), %d score row(s)",
        drug_name,
        len(merged_rows),
        len(moa_mappings),
        len(indication_mappings),
        len(score_rows),
    )
    return output


if __name__ == "__main__":
    label_expansion()
