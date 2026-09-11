"""Web data extraction module for Dimension 5 serious safety profile.

Handles data collection from clinical trials, regulatory sources, and post-marketing
evidence using grounded Gemini calls.
"""

from .helper_functions import (
    determine_approval_market_status,
    extract_incremental_trials,
    extract_serious_safety_data,
    identify_regulatory_impact,
    identify_post_marketing_safety,
)
from .sae_categorization import (
    summarize_sae_across_trials,
    classify_sae_expectedness_and_category,
    update_existing_sae_events_in_table,
    update_new_sae_events_to_summary_table,
)
from .trial_level_weight_calculator import calculate_trial_weights_and_agg_sae

__all__ = [
    "determine_approval_market_status",
    "extract_incremental_trials",
    "extract_serious_safety_data",
    "identify_regulatory_impact",
    "identify_post_marketing_safety",
    "summarize_sae_across_trials",
    "classify_sae_expectedness_and_category",
    "update_existing_sae_events_in_table",
    "update_new_sae_events_to_summary_table",
    "calculate_trial_weights_and_agg_sae",
]
