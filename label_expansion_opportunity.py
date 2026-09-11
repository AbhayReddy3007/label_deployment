"""Label Expansion Opportunity — entry point.

Runs Module 1 (``trial_analyser``) and Module 2 (``web_analyser``)
for exactly one drug, merges their results, and pushes the merged
rows to BigQuery. Defaults to the drug configured in
``medical_potential.config`` (``DRUG_NAME``), but can be called with
a different single drug name explicitly.

Run with:
    python -m medical_potential.label_expansion_opportunity.label_expansion_opportunity
"""

from __future__ import annotations

import logging

from medical_potential.config import DRUG_NAME
from medical_potential.label_expansion_opportunity.bq_utils import (
    merge_results,
    push_to_bigquery,
)
from medical_potential.label_expansion_opportunity.indication_extractor.trial_analyser import (
    analyse as analyse_trials,
)
from medical_potential.label_expansion_opportunity.indication_extractor.web_analyser import (
    analyse as analyse_web,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def label_expansion(drug_name: str = DRUG_NAME) -> list[dict]:
    """Runs the full Label Expansion Opportunity pipeline for exactly one drug.

    ``drug_name`` must be a single drug name (str) - not a list. To
    process multiple drugs, call this once per drug (e.g. loop over
    drug names at the call site and invoke ``label_expansion`` each time).

    Returns the merged rows that were pushed to BigQuery.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise TypeError(
            f"label_expansion() accepts exactly one drug name (str), got: {drug_name!r}"
        )

    logger.info("[LABEL_EXPANSION] Starting Label Expansion Opportunity pipeline for '%s'", drug_name)

    logger.info("[LABEL_EXPANSION] Running Module 1: trial_analyser")
    trial_rows = analyse_trials(drug_name)

    logger.info("[LABEL_EXPANSION] Running Module 2: web_analyser")
    web_rows = analyse_web(drug_name)

    if not trial_rows and not web_rows:
        logger.warning("[LABEL_EXPANSION] Both modules returned no results for '%s' - nothing to push", drug_name)
        return []

    logger.info("[LABEL_EXPANSION] Merging module results")
    merged_rows = merge_results(trial_rows, web_rows)

    logger.info("[LABEL_EXPANSION] Pushing %d merged row(s) to BigQuery", len(merged_rows))
    push_to_bigquery(merged_rows)

    logger.info("[LABEL_EXPANSION] Done. '%s': %d indication(s) processed", drug_name, len(merged_rows))
    return merged_rows


if __name__ == "__main__":
    label_expansion()
