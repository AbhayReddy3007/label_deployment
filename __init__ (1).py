"""Label Expansion Opportunity pillar.

Two independent modules score/collect candidate label-expansion
indications for the configured drug:

- ``trial_analyser``  — Module 1: mines registered clinical trials
  (read from BigQuery) for indications and their trial phase.
- ``web_analyser``    — Module 2: uses Gemini + Google Search to find
  label-expansion signals directly from public web sources (news,
  regulatory filings, pipeline updates) that may not yet be reflected
  in a registered trial.

``label_expansion_opportunity.py`` is the entry point: it runs both
modules, merges their results, and pushes the merged rows to
BigQuery via ``bq_utils``.
"""
