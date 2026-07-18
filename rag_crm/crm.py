"""CRM write-back: every question/answer pair becomes a CRM record.

Real mode: HubSpot. Finds or creates the contact by email, then attaches a
note with the question, answer, intent, and citations. Uses only the Python
standard library (urllib) so the project has no HTTP dependency.

Mock mode: appends the same record to output/crm_log.json so the full
pipeline can be inspected offline.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HUBSPOT_BASE = "https://api.hubapi.com"
NOTE_TO_CONTACT_ASSOCIATION_TYPE_ID = 202  # HubSpot-defined: note -> contact


@dataclass
class CrmRecord:
    timestamp: str
    customer_email: str
    question: str
    answer: str
    intent: str
    escalate: bool
    escalation_reason: str
    citations: list[str] = field(default_factory=list)
    answer_mode: str = "mock"  # which answerer produced this: mock | real

    @classmethod
    def build(cls, *, customer_email: str, question: str, result) -> "CrmRecord":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            customer_email=customer_email,
            question=question,
            answer=result.answer,
            intent=result.intent,
            escalate=result.escalate,
            escalation_reason=result.escalation_reason,
            citations=result.citations,
            answer_mode=result.mode,
        )


# --------------------------------------------------------------------------
# Mock mode — local JSON fixture
# --------------------------------------------------------------------------

def write_mock(record: CrmRecord, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if log_path.exists():
        records = json.loads(log_path.read_text(encoding="utf-8"))
    records.append(asdict(record))
    log_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"mock CRM: appended record #{len(records)} to {log_path}"


# --------------------------------------------------------------------------
# Real mode — HubSpot
# --------------------------------------------------------------------------

def _hubspot_request(token: str, method: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url=f"{HUBSPOT_BASE}{path}",
        method=method,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HubSpot {method} {path} failed ({error.code}): {detail}") from error


def _find_or_create_contact(token: str, email: str) -> str:
    """Return the HubSpot contact id for this email, creating it if needed."""
    search = _hubspot_request(
        token,
        "POST",
        "/crm/v3/objects/contacts/search",
        {
            "filterGroups": [
                {"filters": [{"propertyName": "email", "operator": "EQ", "value": email}]}
            ],
            "limit": 1,
        },
    )
    results = search.get("results", [])
    if results:
        return results[0]["id"]

    created = _hubspot_request(
        token,
        "POST",
        "/crm/v3/objects/contacts",
        {"properties": {"email": email, "lifecyclestage": "lead"}},
    )
    return created["id"]


def write_hubspot(record: CrmRecord, token: str) -> str:
    contact_id = _find_or_create_contact(token, record.customer_email)

    body_lines = [
        "Assistant conversation logged by the RAG support assistant.",
        "",
        f"Question: {record.question}",
        f"Answer: {record.answer}",
        f"Intent: {record.intent}",
        f"Escalate: {'yes — ' + record.escalation_reason if record.escalate else 'no'}",
        f"Sources: {', '.join(record.citations) if record.citations else 'none'}",
    ]
    _hubspot_request(
        token,
        "POST",
        "/crm/v3/objects/notes",
        {
            "properties": {
                "hs_note_body": "\n".join(body_lines),
                "hs_timestamp": str(int(time.time() * 1000)),
            },
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [
                        {
                            "associationCategory": "HUBSPOT_DEFINED",
                            "associationTypeId": NOTE_TO_CONTACT_ASSOCIATION_TYPE_ID,
                        }
                    ],
                }
            ],
        },
    )
    return f"HubSpot: note attached to contact {contact_id} ({record.customer_email})"
