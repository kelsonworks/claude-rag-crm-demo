"""Glue: question -> retrieve -> answer -> CRM write-back."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .answerer import AnswerResult, answer_mock, answer_with_claude
from .chunker import Chunk
from .crm import CrmRecord, write_hubspot, write_mock
from .retriever import TfidfRetriever


@dataclass
class QueryOutcome:
    question: str
    hits: list[tuple[Chunk, float]]
    result: AnswerResult
    crm_status: str


def run_query(
    question: str,
    *,
    retriever: TfidfRetriever,
    customer_email: str,
    mock: bool,
    crm_log_path: Path,
    top_k: int = 3,
) -> QueryOutcome:
    hits = retriever.search(question, top_k=top_k)

    if mock:
        result = answer_mock(question, hits)
    else:
        result = answer_with_claude(question, hits)

    record = CrmRecord.build(customer_email=customer_email, question=question, result=result)

    hubspot_token = os.environ.get("HUBSPOT_TOKEN", "").strip()
    if mock:
        crm_status = write_mock(record, crm_log_path)
    elif hubspot_token:
        crm_status = write_hubspot(record, hubspot_token)
    else:
        crm_status = (
            write_mock(record, crm_log_path)
            + "  (HUBSPOT_TOKEN not set — fell back to the local log)"
        )

    return QueryOutcome(question=question, hits=hits, result=result, crm_status=crm_status)
