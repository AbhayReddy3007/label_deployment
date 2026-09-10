"""Label Expansion Opportunity — entry point.

Runs Module 1 (``trial_analyser``) and Module 2 (``web_analyser``)
for the drug configured in ``medical_potential.config``, merges their
results, and pushes the merged rows to BigQuery.

Run with:
    python -m medical_potential.label_expansion_opportunity.label_expansion_opportunity
"""

from __future__ import annotations

import logging

from medical_potential.config import DRUG_NAME
from medical_potential.label_expansion_opportunity import (
    push_to_bq,
    trial_analyser,
    web_analyser,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def label_expansion() -> list[dict]:
    """Runs the full Label Expansion Opportunity pipeline for ``DRUG_NAME``.

    Returns the merged, scored rows that were pushed to BigQuery.
    """
    logger.info("[LABEL_EXPANSION] Starting Label Expansion Opportunity pipeline for '%s'", DRUG_NAME)

    logger.info("[LABEL_EXPANSION] Running Module 1: trial_analyser")
    trial_rows = trial_analyser.analyse(DRUG_NAME)

    logger.info("[LABEL_EXPANSION] Running Module 2: web_analyser")
    web_rows = web_analyser.analyse(DRUG_NAME)

    if not trial_rows and not web_rows:
        logger.warning("[LABEL_EXPANSION] Both modules returned no results for '%s' - nothing to push", DRUG_NAME)
        return []

    logger.info("[LABEL_EXPANSION] Merging module results")
    merged_rows = push_to_bq.merge_results(trial_rows, web_rows)

    logger.info("[LABEL_EXPANSION] Pushing %d merged row(s) to BigQuery", len(merged_rows))
    push_to_bq.push_to_bigquery(merged_rows)

    logger.info("[LABEL_EXPANSION] Done. '%s': %d indication(s) processed", DRUG_NAME, len(merged_rows))
    return merged_rows


if __name__ == "__main__":
    label_expansion()
