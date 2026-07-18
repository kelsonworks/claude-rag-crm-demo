"""Answer generation: real Claude API call, or an offline mock stub.

Real mode calls the Messages API (model: claude-sonnet-5) with a structured
JSON output schema, so answer / intent / escalation come back in one call.
Mock mode returns canned-but-realistic responses for the scripted demo
questions and falls back to a deterministic extractive answer (built from the
retrieved chunks) for anything else. Mock mode makes zero network calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .chunker import Chunk

MODEL = "claude-sonnet-5"

INTENTS = [
    "pricing_inquiry",
    "plan_membership",
    "scheduling_request",
    "service_issue",
    "policy_question",
    "other",
]

SYSTEM_PROMPT = """\
You are the customer-facing assistant for Bluebird Heating & Air, a residential
HVAC company. You answer customer questions using ONLY the knowledge-base
excerpts provided in each request.

Rules:
- Ground every factual claim in the excerpts. Cite inline using the exact
  bracketed labels provided, e.g. [02-pricing-sheet.md § Financing]. Place each
  citation immediately after the claim it supports.
- If the excerpts do not cover the question, say so plainly and direct the
  customer to the office at (555) 014-2260. Never invent prices, policies, or
  availability.
- You cannot book, change, or cancel appointments. For anything requiring a
  human (scheduling, complaints, refunds, active emergencies), set escalate to
  true and explain that the office will follow up.
- Keep answers under 150 words, plain and direct.
"""

# JSON schema for structured output — one call returns the answer, the
# classified intent for the CRM, and the escalation flag.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Customer-facing answer with inline [doc § Section] citations.",
        },
        "intent": {"type": "string", "enum": INTENTS},
        "escalate": {"type": "boolean"},
        "escalation_reason": {
            "type": "string",
            "description": "Why a human should follow up; empty string if escalate is false.",
        },
    },
    "required": ["answer", "intent", "escalate", "escalation_reason"],
    "additionalProperties": False,
}

_CITATION_RE = re.compile(r"\[([^\[\]]+?§[^\[\]]+?)\]")


@dataclass
class AnswerResult:
    answer: str
    intent: str
    escalate: bool
    escalation_reason: str = ""
    citations: list[str] = field(default_factory=list)
    mode: str = "mock"

    def __post_init__(self) -> None:
        if not self.citations:
            self.citations = [f"[{m}]" for m in _CITATION_RE.findall(self.answer)]


def _format_context(hits: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        blocks.append(f"[{i}] {chunk.citation}\n{chunk.text}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Real mode — Claude API
# --------------------------------------------------------------------------

def answer_with_claude(question: str, hits: list[tuple[Chunk, float]]) -> AnswerResult:
    """One Messages API call: grounded answer + intent + escalation as JSON.

    The system prompt is static, so it carries a cache_control breakpoint.
    Note: prefixes under ~1024 tokens are below the caching minimum and are
    silently not cached — the breakpoint is a no-op at this corpus size but
    becomes a real saving as the system prompt / shared context grows.
    """
    import anthropic  # imported lazily so mock mode needs no dependencies

    client = anthropic.Anthropic()
    user_text = (
        f"Customer question:\n{question}\n\n"
        f"Knowledge-base excerpts:\n{_format_context(hits)}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_text}],
        output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
    )
    text = next(block.text for block in response.content if block.type == "text")
    data = json.loads(text)
    return AnswerResult(
        answer=data["answer"],
        intent=data["intent"],
        escalate=data["escalate"],
        escalation_reason=data.get("escalation_reason", ""),
        mode="real",
    )


# --------------------------------------------------------------------------
# Mock mode — no network, no keys
# --------------------------------------------------------------------------

def _normalize(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


# Canned responses for the scripted demo questions. Written by hand against
# the actual docs/ corpus so citations resolve to real sections.
_CANNED: dict[str, dict] = {
    _normalize(
        "How much does a diagnostic visit cost, and is the fee waived if I go ahead with the repair?"
    ): {
        "answer": (
            "A standard diagnostic visit during office hours is $89, which covers "
            "travel, a full diagnosis of one system, and a written flat-rate quote "
            "[02-pricing-sheet.md § Diagnostic & Service Call Fees]. If you approve "
            "the quoted repair during the same visit, that $89 is credited toward "
            "the repair total, so you don't pay it twice "
            "[02-pricing-sheet.md § Diagnostic & Service Call Fees]. After-hours, "
            "Sunday, or holiday dispatch is $189 flat and includes the first 30 "
            "minutes on site [02-pricing-sheet.md § Diagnostic & Service Call Fees]."
        ),
        "intent": "pricing_inquiry",
        "escalate": False,
        "escalation_reason": "",
    },
    _normalize(
        "What does the Comfort Club maintenance plan include and what does it cost per month?"
    ): {
        "answer": (
            "There are two Comfort Club tiers. Standard is $16/month (or $180/year) "
            "and includes two seasonal tune-ups, 15% off repairs, priority "
            "scheduling, one waived $89 diagnostic per membership year, and "
            "after-hours calls billed at $89 instead of $189 "
            "[04-maintenance-plans.md § Standard Plan — $16/month]. Plus is "
            "$27/month (or $300/year) and adds 20% off repairs, filter delivery, "
            "$150/year of loyalty credit toward a future replacement, and an annual "
            "dryer vent cleaning [04-maintenance-plans.md § Plus Plan — $27/month]. "
            "Both plans are month-to-month and refundable pro rata if you cancel "
            "[04-maintenance-plans.md § Overview]."
        ),
        "intent": "plan_membership",
        "escalate": False,
        "escalation_reason": "",
    },
    _normalize(
        "My AC stopped cooling last night - can someone come out today, and what are the after-hours charges?"
    ): {
        "answer": (
            "No-cool calls received before 2:00 pm get same-day priority during "
            "peak season, so call the office as early as you can [05-faq.md § How "
            "fast can you get here in an emergency?]. If it can't wait, the "
            "emergency line dispatches the on-call technician, usually within 2–3 "
            "hours; after-hours dispatch is $189 flat, or the standard $89 if "
            "you're a Comfort Club member [05-faq.md § How fast can you get here "
            "in an emergency?]. I can't book the visit myself — the office at "
            "(555) 014-2260 will confirm a same-day slot."
        ),
        "intent": "service_issue",
        "escalate": True,
        "escalation_reason": "Customer has an active no-cool issue and is asking for same-day service; dispatch needs to schedule the visit.",
    },
    _normalize("What's your cancellation policy if I need to reschedule an appointment?"): {
        "answer": (
            "You can cancel or reschedule at no charge with at least 24 hours "
            "notice [03-service-policies.md § Cancellations & Rescheduling]. "
            "Inside 24 hours a $45 late-cancellation fee applies, but it's waived "
            "the first time for every customer [03-service-policies.md § "
            "Cancellations & Rescheduling]. And if we ever miss our arrival window "
            "without notifying you, you get a $50 credit on your invoice "
            "[03-service-policies.md § Cancellations & Rescheduling]."
        ),
        "intent": "policy_question",
        "escalate": False,
        "escalation_reason": "",
    },
}

_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("service_issue", ("stopped", "broken", "not working", "no heat", "no cool",
                       "leak", "noise", "smell", "won t", "wont", "emergency")),
    ("plan_membership", ("plan", "club", "membership", "maintenance", "tune")),
    ("policy_question", ("cancel", "reschedule", "policy", "warranty", "refund",
                         "deposit", "guarantee")),
    ("pricing_inquiry", ("price", "cost", "fee", "charge", "how much", "rate",
                         "financing")),
    ("scheduling_request", ("schedule", "appointment", "book", "come out",
                            "visit", "today", "tomorrow", "available")),
]

_ESCALATE_TERMS = ("emergency", "today", "asap", "right away", "complaint",
                   "refund", "urgent", "no cool", "no heat", "stopped")


def _classify_intent(question: str) -> tuple[str, bool]:
    normalized = f" {_normalize(question)} "
    intent = "other"
    for name, terms in _INTENT_RULES:
        if any(f" {t} " in normalized or t in normalized for t in terms):
            intent = name
            break
    escalate = intent == "service_issue" or any(t in normalized for t in _ESCALATE_TERMS)
    return intent, escalate


def _sentences(text: str) -> list[str]:
    # Treat bullet items and sentence breaks as sentence boundaries.
    cleaned = re.sub(r"^\s*[-|*]\s*", "", text, flags=re.MULTILINE)
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _extractive_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """Deterministic fallback: quote the most relevant sentences, with citations."""
    from .retriever import tokenize

    query_terms = set(tokenize(question))
    candidates: list[tuple[int, str, str]] = []
    for chunk, _score in hits[:2]:
        for sentence in _sentences(chunk.text):
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap >= 2:
                candidates.append((overlap, sentence, chunk.citation))
    candidates.sort(key=lambda c: c[0], reverse=True)

    if not candidates:
        return (
            "I don't have that in the current knowledge base. The office at "
            "(555) 014-2260 can help directly "
            "[01-company-overview.md § Hours & Contact]."
        )
    picked = candidates[:3]
    return " ".join(f"{sentence} {citation}" for _overlap, sentence, citation in picked)


def answer_mock(question: str, hits: list[tuple[Chunk, float]]) -> AnswerResult:
    """Offline stand-in for the Claude call. Deterministic; zero network."""
    canned = _CANNED.get(_normalize(question))
    if canned is not None:
        return AnswerResult(mode="mock", **canned)

    intent, escalate = _classify_intent(question)
    return AnswerResult(
        answer=_extractive_answer(question, hits),
        intent=intent,
        escalate=escalate,
        escalation_reason="Flagged by mock keyword rules for human follow-up." if escalate else "",
        mode="mock",
    )
