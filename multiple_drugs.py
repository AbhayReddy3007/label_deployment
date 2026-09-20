"""Run the Label Expansion Opportunity pipeline for multiple drugs.

``label_expansion_opportunity.label_expansion()`` only accepts one drug
name at a time. This script is a thin wrapper around it: edit
``DRUG_NAMES`` below, then run this file, and it calls ``label_expansion()``
once per drug, in sequence.

Nothing in the rest of the package is modified - this script only imports
and calls the existing, unmodified pipeline entry point.

Run with:
    python -m medical_potential.label_expansion_opportunity.multiple_drugs
"""

from __future__ import annotations

import logging

from .label_expansion_opportunity import PIPELINE_STAGES, label_expansion, logger

# ==============================
# DRUGS TO RUN
# ==============================
# Add/remove drug names here. Each one is run through the full pipeline,
# one at a time, in this order.
DRUG_NAMES: list[str] = [
    "Tirzepatide",
    "Semaglutide",
]

# ==============================
# PIPELINE OPTIONS
# ==============================
# Passed through to label_expansion() for every drug in DRUG_NAMES.
# See label_expansion_opportunity.py's docstring for what each one does.
DRUG_DETAILS_TABLE = "drug_details"
RUN_OT_MAPPING = True
START_FROM = "discovery"  # one of PIPELINE_STAGES: "discovery" | "moa_mapping" | "indication_mapping" | "scoring"
GENERATE_REPORT = True  # Step 7: rationale + PDF report generation and storage

# If True, a drug that raises an exception is logged and skipped so the
# rest of the batch still runs. If False, the first failure stops the
# whole batch immediately.
CONTINUE_ON_ERROR = True


def run_multiple_drugs(
    drug_names: list[str] = DRUG_NAMES,
    drug_details_table: str = DRUG_DETAILS_TABLE,
    run_ot_mapping: bool = RUN_OT_MAPPING,
    start_from: str = START_FROM,
    generate_report: bool = GENERATE_REPORT,
    continue_on_error: bool = CONTINUE_ON_ERROR,
) -> dict[str, dict | None]:
    """Runs ``label_expansion()`` once per drug in ``drug_names``, in order.

    Args:
        drug_names: list of drug/molecule names to run the pipeline for.
        drug_details_table: forwarded to ``label_expansion()`` for every drug.
        run_ot_mapping: forwarded to ``label_expansion()`` for every drug.
        start_from: forwarded to ``label_expansion()`` for every drug - one
            of ``PIPELINE_STAGES`` (see label_expansion_opportunity.py).
        generate_report: forwarded to ``label_expansion()`` for every drug -
            whether to generate + store the rationale and PDF report (Step 7).
        continue_on_error: if ``True`` (default), a drug that raises is
            logged and skipped rather than stopping the whole batch.

    Returns:
        ``{drug_name: result_dict_or_None}`` - the same dict
        ``label_expansion()`` returns for each drug, or ``None`` for any
        drug that failed (only reachable when ``continue_on_error=True``).
    """
    if not drug_names:
        logger.warning("[MULTIPLE_DRUGS] DRUG_NAMES is empty — nothing to run")
        return {}

    logger.info(
        "[MULTIPLE_DRUGS] Running Label Expansion Opportunity for %d drug(s): %s",
        len(drug_names), drug_names,
    )

    results: dict[str, dict | None] = {}
    succeeded: list[str] = []
    failed: list[str] = []

    for i, drug_name in enumerate(drug_names, 1):
        logger.info(
            "[MULTIPLE_DRUGS] (%d/%d) Starting '%s'",
            i, len(drug_names), drug_name,
        )
        try:
            result = label_expansion(
                drug_name=drug_name,
                drug_details_table=drug_details_table,
                run_ot_mapping=run_ot_mapping,
                start_from=start_from,
                generate_report=generate_report,
            )
            results[drug_name] = result
            succeeded.append(drug_name)
            logger.info(
                "[MULTIPLE_DRUGS] (%d/%d) Finished '%s': %d indication row(s), "
                "%d MOA mapping(s), %d indication mapping(s), %d score row(s), "
                "report=%s, cache=%s",
                i, len(drug_names), drug_name,
                len(result.get("merged_rows", [])),
                len(result.get("moa_mappings", [])),
                len(result.get("indication_mappings", [])),
                len(result.get("score_rows", [])),
                result.get("report_gcs_uri") or "not uploaded",
                result.get("cache_gcs_uri") or "not uploaded",
            )
        except Exception:
            logger.exception(
                "[MULTIPLE_DRUGS] (%d/%d) Failed for '%s'",
                i, len(drug_names), drug_name,
            )
            results[drug_name] = None
            failed.append(drug_name)
            if not continue_on_error:
                logger.error(
                    "[MULTIPLE_DRUGS] continue_on_error=False — stopping batch after failure on '%s'",
                    drug_name,
                )
                raise

    logger.info(
        "[MULTIPLE_DRUGS] Batch complete: %d/%d succeeded%s",
        len(succeeded), len(drug_names),
        f", {len(failed)} failed: {failed}" if failed else "",
    )
    return results


if __name__ == "__main__":
    run_multiple_drugs()
