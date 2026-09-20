"""Report and rationale generation sub-package.

Generates human-readable outputs from a drug's label-expansion score
results (the dict ``label_expansion()`` returns - see
``label_expansion_opportunity.py``):

- ``generate_rationale`` — a short (<=50 word) plain-text rationale
  explaining what's driving a drug's label-expansion opportunities.
- ``generate_report``    — a 2-page PDF report summarizing the drug's
  label-expansion opportunity landscape. Also holds ``upload_report_pdf``
  and ``upload_json_payload`` (GCS upload helpers), since the PDF upload
  is specific to this module and the JSON upload is shared from here to
  avoid duplicating it in ``generate_rationale.py`` too.

Generation and storage are kept separate, same as ``ssp_rationale.py`` /
``ssp_report.py``: ``generate_label_expansion_rationale`` and
``generate_label_expansion_report_bytes`` only generate and return content;
nothing is uploaded until the caller (Step 7 of ``label_expansion()``)
explicitly calls ``upload_report_pdf`` / ``upload_json_payload``.
"""

from .generate_rationale import generate_label_expansion_rationale
from .generate_report import (
    PILLAR_NAME,
    generate_label_expansion_report_bytes,
    upload_json_payload,
    upload_report_pdf,
)

__all__ = [
    "generate_label_expansion_rationale",
    "generate_label_expansion_report_bytes",
    "upload_report_pdf",
    "upload_json_payload",
    "PILLAR_NAME",
]
