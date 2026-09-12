"""Open Targets mapping sub-package.

Resolves MOA strings and indication strings to their Open Targets
equivalents and persists the mappings to BigQuery.

- ``moa_mapping``         — MOA → OT target name resolution.
- ``indication_mapping``  — Indication → OT disease name resolution.
- ``ot_utils``            — shared OT GraphQL, Gemini, and BQ helpers.
"""

from .moa_mapping import run_moa_mapping
from .indication_mapping import run_indication_mapping

__all__ = [
    "run_moa_mapping",
    "run_indication_mapping",
]
