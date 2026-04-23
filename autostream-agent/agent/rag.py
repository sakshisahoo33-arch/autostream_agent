"""
agent/rag.py
Simple RAG pipeline that retrieves relevant context from the local knowledge base.
No vector DB required — uses keyword similarity for assignment scope.
"""

import json
import os
from pathlib import Path


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def load_knowledge_base() -> dict:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def kb_to_text(kb: dict) -> str:
    """
    Flatten the entire knowledge base into a readable text block
    that can be injected into the system prompt.
    """
    lines = []
    lines.append(f"Company: {kb['company']} — {kb['tagline']}")
    lines.append("")

    lines.append("=== PRICING PLANS ===")
    for key, plan in kb["plans"].items():
        lines.append(f"\n{plan['name']} — ${plan['price_monthly']}/month")
        for feat in plan["features"]:
            lines.append(f"  • {feat}")

    lines.append("\n=== POLICIES ===")
    for policy, detail in kb["policies"].items():
        lines.append(f"  {policy.capitalize()}: {detail}")

    lines.append("\n=== FAQ ===")
    for item in kb["faq"]:
        lines.append(f"  Q: {item['question']}")
        lines.append(f"  A: {item['answer']}")

    return "\n".join(lines)


def retrieve_context(query: str) -> str:
    """
    Return the full knowledge base text (suitable for a short KB like this).
    For a larger KB you would embed + cosine-search here.
    """
    kb = load_knowledge_base()
    return kb_to_text(kb)
