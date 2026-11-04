"""kazeval — Kazakh Frontier Lab benchmarking package.

Target: no model claim exists until a runner here reproduces it. Today that holds for
`measured` KazQAD records only; the KazMMLU record is legacy-script and `reported` records
are other people's numbers (see evallab/README.md). Tracks: KazQAD retrieval/reranking
(kk-MTEB), the hard-negatives MRR protocol (planka re-measurement), KazMMLU (Kazakh subset),
Qorgau guardrails.
"""

from kazeval.results import ResultRecord, load_records, save_record, validate_record

__version__ = "0.1.0"

__all__ = [
    "ResultRecord",
    "__version__",
    "load_records",
    "save_record",
    "validate_record",
]
