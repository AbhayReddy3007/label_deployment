"""Report and rationale generation sub-package.

Generates human-readable outputs from a drug's label-expansion score
results (the dict ``label_expansion()`` returns - see
``label_expansion_opportunity.py``):

- ``generate_rationale`` — a short (<=50 word) plain-text rationale
  explaining what's driving a drug's label-expansion opportunities.
- ``generate_report``    — a 2-page PDF report summarizing the drug's
  label-expansion opportunity landscape.

Neither module persists its output itself (no BigQuery/storage writes) -
both simply return the generated content (text, or PDF bytes) plus the
payload used to generate it, for the caller to store however it needs to.
"""

from .generate_rationale import generate_label_expansion_rationale
from .generate_report import generate_label_expansion_report_bytes

__all__ = [
    "generate_label_expansion_rationale",
    "generate_label_expansion_report_bytes",
]
