#!/usr/bin/env python3
"""RAG support assistant with CRM write-back — demo entry point.

Mock mode (no keys, no network, no installs):

    python3 demo.py --mock

Real mode (needs ANTHROPIC_API_KEY; HUBSPOT_TOKEN optional):

    pip install -r requirements.txt
    python3 demo.py

Ask your own question in either mode:

    python3 demo.py --mock -q "Do you install heat pumps, and what do they cost?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag_crm.chunker import load_chunks
from rag_crm.pipeline import run_query
from rag_crm.retriever import TfidfRetriever

DOCS_DIR = ROOT / "docs"
CRM_LOG_PATH = ROOT / "output" / "crm_log.json"

DEMO_QUESTIONS = [
    "How much does a diagnostic visit cost, and is the fee waived if I go ahead with the repair?",
    "What does the Comfort Club maintenance plan include and what does it cost per month?",
    "My AC stopped cooling last night - can someone come out today, and what are the after-hours charges?",
    "What's your cancellation policy if I need to reschedule an appointment?",
]

RULE = "=" * 78
THIN = "-" * 78


def load_dotenv(path: Path) -> None:
    """Tiny stdlib .env loader — real values never override the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def print_outcome(outcome, index: int, total: int) -> None:
    print(RULE)
    print(f"[{index}/{total}] QUESTION: {outcome.question}")
    print(THIN)
    print("Retrieved context:")
    for chunk, score in outcome.hits:
        print(f"  {score:.3f}  {chunk.citation}")
    print(THIN)
    print(f"ANSWER ({outcome.result.mode} mode):")
    print(f"  {outcome.result.answer}")
    print(THIN)
    print(f"Intent:    {outcome.result.intent}")
    escalation = "yes - " + outcome.result.escalation_reason if outcome.result.escalate else "no"
    print(f"Escalate:  {escalation}")
    print(f"Citations: {', '.join(outcome.result.citations) or 'none'}")
    print(f"CRM:       {outcome.crm_status}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG assistant over company docs, with CRM write-back.")
    parser.add_argument("--mock", action="store_true",
                        help="run fully offline: stubbed Claude responses, CRM writes to output/crm_log.json")
    parser.add_argument("-q", "--question",
                        help="ask a single question instead of the scripted demo set")
    parser.add_argument("--email", default="jordan.lee@example.com",
                        help="customer email attached to the CRM record (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of chunks to retrieve per question (default: %(default)s)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not args.mock and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY is not set. Either export it (see .env.example)")
        print("or run the offline demo:  python3 demo.py --mock")
        return 1

    chunks = load_chunks(DOCS_DIR)
    retriever = TfidfRetriever(chunks)
    mode = "mock (offline)" if args.mock else "real (Claude API + HubSpot)"
    print(f"Indexed {len(chunks)} sections from {len({c.doc for c in chunks})} documents. Mode: {mode}")
    print()

    questions = [args.question] if args.question else DEMO_QUESTIONS
    for i, question in enumerate(questions, start=1):
        outcome = run_query(
            question,
            retriever=retriever,
            customer_email=args.email,
            mock=args.mock,
            crm_log_path=CRM_LOG_PATH,
            top_k=args.top_k,
        )
        print_outcome(outcome, i, len(questions))

    if args.mock:
        print(RULE)
        print(f"Done. Inspect the CRM log:  cat {CRM_LOG_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
