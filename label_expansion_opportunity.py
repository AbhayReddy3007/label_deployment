"""Label Expansion Opportunity: Main Orchestrator.

Runs the full label-expansion pipeline for exactly one drug:

**Indication Discovery (Steps 1-3):**
    1. Extract indications from registered clinical trials (Module 1 — trial_analyser).
    2. Extract indications from public web sources (Module 2 — web_analyser).
    3. Merge and de-duplicate results, push merged rows to BigQuery.

**Open Targets Mapping (Steps 4-5):**
    4. Resolve the drug's Mechanism(s) of Action to OT target names.
    5. Resolve discovered indications to OT disease names.

**Score Calculation (Step 6):**
    6. Select the best trial per therapy_area/OT-disease combination,
       compute trial weights, and derive a composite Final Score, pushed
       to LE_SCORE_CALCULATION_TABLE.

Run with:
    python -m medical_potential.label_expansion_opportunity.label_expansion_opportunity
"""

from __future__ import annotations

import logging

from medical_potential.config import DRUG_NAME

from .bq_utils import merge_results, push_to_bigquery
from .indication_extractor import analyse_trials, analyse_web
from .ot_mapping import run_moa_mapping, run_indication_mapping
from .scoring import run_score_calculation

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

logger = logging.getLogger("medical_potential.label_expansion_opportunity")
logger.propagate = False
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.handlers = [handler]


def label_expansion(
    drug_name: str = DRUG_NAME,
    drug_details_table: str = "drug_details",
    target_ensembl_ids: list[str] | None = None,
    run_ot_mapping: bool = True,
) -> dict:
    """Run the full Label Expansion Opportunity pipeline for one drug.

    Executes a 5-step pipeline:

    **Indication Discovery (Steps 1-3):**
        1. Module 1 — trial_analyser: mines registered clinical trials from
           BigQuery for indications and their trial phase.
        2. Module 2 — web_analyser: uses Gemini + Google Search to find
           label-expansion signals from public web sources.
        3. Merge & push: de-duplicates on (drug_name, indication, trial_id),
           preferring trial-sourced rows, and upserts into BigQuery.

    **Open Targets Mapping (Steps 4-5):**
        4. MOA mapping: fetches Mechanism_of_Action from drug_details table,
           resolves each to an OT target name (deterministic OT search →
           Gemini fallback), pushes to OT_MOA_TABLE.
        5. Indication mapping: reads the indications just pushed to LE_TABLE,
           resolves each to an OT disease name (Gemini semantic matching →
           OT text search fallback), pushes to OT_DISEASE_TABLE.

    Args:
        drug_name: The drug / molecule name (e.g. "semaglutide").
        drug_details_table: BQ table name holding drug details with
            Mechanism_of_Action column.
        target_ensembl_ids: Optional Ensembl IDs for the drug's gene targets,
            to enable Gemini semantic matching (Path A) in indication mapping.
            If ``None`` (default), these are derived automatically from the
            MOA mapping resolved in Step 4 — you normally don't need to pass
            this explicitly. Pass an explicit list only to override that.
        run_ot_mapping: Whether to run Steps 4-5. Set ``False`` to only
            discover indications without OT resolution.

    Returns:
        dict with keys: ``drug_name``, ``merged_rows``, ``moa_mappings``,
        ``indication_mappings``.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise TypeError(
            f"label_expansion() accepts exactly one drug name (str), got: {drug_name!r}"
        )

    logger.info("[LABEL_EXPANSION] Starting Label Expansion Opportunity pipeline for '%s'", drug_name)

    # ── Step 1: Module 1 — Trial Analyser ──────────────────────────────────
    logger.info("[LABEL_EXPANSION] Step 1: Extract indications from clinical trials (trial_analyser)")
    trial_rows = analyse_trials(drug_name)
    logger.info("[LABEL_EXPANSION] Step 1 complete: %d trial-sourced row(s)", len(trial_rows))

    # ── Step 2: Module 2 — Web Analyser ────────────────────────────────────
    logger.info("[LABEL_EXPANSION] Step 2: Extract indications from web sources (web_analyser)")
    web_rows = analyse_web(drug_name)
    logger.info("[LABEL_EXPANSION] Step 2 complete: %d web-sourced row(s)", len(web_rows))

    # ── Step 3: Merge & push to BigQuery ───────────────────────────────────
    if not trial_rows and not web_rows:
        logger.warning(
            "[LABEL_EXPANSION] Both modules returned no results for '%s' — nothing to push",
            drug_name,
        )
        merged_rows = []
    else:
        logger.info("[LABEL_EXPANSION] Step 3: Merge and push results to BigQuery")
        merged_rows = merge_results(trial_rows, web_rows)
        push_to_bigquery(merged_rows)
        logger.info("[LABEL_EXPANSION] Step 3 complete: %d merged row(s) pushed", len(merged_rows))

    # ── Step 4: MOA → Open Targets mapping ─────────────────────────────────
    moa_mappings = []
    if run_ot_mapping:
        logger.info("[LABEL_EXPANSION] Step 4: Resolve MOA(s) to Open Targets target names")
        try:
            moa_mappings = run_moa_mapping(
                drug_name=drug_name,
                drug_details_table=drug_details_table,
            )
            logger.info("[LABEL_EXPANSION] Step 4 complete: %d MOA mapping(s)", len(moa_mappings))
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 4 failed for '%s'", drug_name)
    else:
        logger.info("[LABEL_EXPANSION] Step 4: Skipped (run_ot_mapping=False)")

    # ── Step 5: Indication → Open Targets disease mapping ──────────────────
    indication_mappings = []
    if run_ot_mapping:
        # Derive target Ensembl IDs from the MOA mapping we just resolved in
        # Step 4, so Path A (Gemini matching against the actual OT diseases
        # linked to this drug's target) runs automatically — no need for the
        # caller to supply target_ensembl_ids by hand. An explicit
        # target_ensembl_ids argument still overrides this.
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

        logger.info("[LABEL_EXPANSION] Step 5: Resolve indications to Open Targets disease names")
        try:
            indication_mappings = run_indication_mapping(
                drug_name=drug_name,
                target_ensembl_ids=effective_target_ids,
            )
            logger.info(
                "[LABEL_EXPANSION] Step 5 complete: %d indication mapping(s)",
                len(indication_mappings),
            )
        except Exception:
            logger.exception("[LABEL_EXPANSION] Step 5 failed for '%s'", drug_name)
    else:
        logger.info("[LABEL_EXPANSION] Step 5: Skipped (run_ot_mapping=False)")

    # ── Step 6: Score calculation ────────────────────────────────────────────
    score_rows = []
    if run_ot_mapping:
        logger.info("[LABEL_EXPANSION] Step 6: Compute label-expansion scores")
        try:
            score_rows = run_score_calculation(drug_name=drug_name, push=True)
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
