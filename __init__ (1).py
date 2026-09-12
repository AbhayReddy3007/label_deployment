"""Label Expansion Opportunity pillar.

Re-exports the single entry-point ``label_expansion`` for external callers.

Two independent modules score/collect candidate label-expansion
indications for the configured drug:

- ``trial_analyser``  — Module 1: mines registered clinical trials
  (read from BigQuery) for indications and their trial phase.
- ``web_analyser``    — Module 2: uses Gemini + Google Search to find
  label-expansion signals directly from public web sources.

After discovery, Open Targets mapping resolves MOAs and indications
to their OT equivalents.

``label_expansion_opportunity.py`` is the entry point: it runs all
steps and pushes results to BigQuery.
"""

from .label_expansion_opportunity import label_expansion

__all__ = ["label_expansion"]
