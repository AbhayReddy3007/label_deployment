"""Label Expansion Score Calculation sub-package.

Computes a composite Final Score for each drug's discovered label-expansion
indications:

- ``data_fetcher``     — fetches LE_TABLE rows and enriches trial-sourced
  rows with region/sample-size/dosage/association data (BQ, then Gemini
  fallback).
- ``trial_selector``   — computes trial_weight for every row and selects
  the best trial per therapy_area/OT-disease (TA-I) combination.
- ``score_calculator`` — derives prior, maturity, breadth, coherence, and
  the final composite score; pushes results to
  ``LE_SCORE_CALCULATION_TABLE``.
"""

from .score_calculator import run_score_calculation

__all__ = ["run_score_calculation"]
