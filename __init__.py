"""Indication extractor sub-package.

Holds the two research modules that find candidate label-expansion
indications for a drug, plus their shared Gemini utilities:

- ``indication_research`` — shared Gemini client, ``gemini_generate()``,
  ``extract_json()``, and the ``SECONDARY_INDICATION_CRITERIA`` prompt text
  used by both modules below.
- ``trial_analyser``  — Module 1: mines registered clinical trials
  (read from BigQuery) for indications and their trial phase.
- ``web_analyser``    — Module 2: uses Gemini + Google Search to find
  label-expansion signals directly from public web sources.
"""
