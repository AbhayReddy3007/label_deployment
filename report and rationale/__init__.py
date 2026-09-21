"""Report and rationale generation sub-package.

Generates human-readable outputs from a drug's label-expansion score
results (the dict ``label_expansion()`` returns - see
``label_expansion_opportunity.py``):

- ``generate_rationale`` — a short (<=50 word) plain-text rationale
  explaining what's driving a drug's label-expansion opportunities.
- ``generate_report``    — a 2-page PDF report summarizing the drug's
  label-expansion opportunity landscape, plus ``fetch_score_rows`` for
  standalone report generation.

Generation and storage are kept separate, same as ``ssp_rationale.py`` /
``ssp_report.py``: ``generate_label_expansion_rationale`` and
``generate_label_expansion_report_bytes`` only generate and return content.
Storage uses the shared, project-wide helpers in ``medical_potential.gcp_utils``
(``upload_dimension_report_pdf_to_gcs``, ``upload_dimension_payload_cache_to_gcs``,
``append_dimension_score_to_bigquery``) - called directly by Step 7 of
``label_expansion()``, not duplicated here.
"""

from .generate_rationale import generate_label_expansion_rationale
from .generate_report import fetch_score_rows, generate_label_expansion_report_bytes

__all__ = [
    "generate_label_expansion_rationale",
    "generate_label_expansion_report_bytes",
    "fetch_score_rows",
]
