"""
Agent tools. Use the existing RAG API via HTTP (no import from main app = OCP).
Extend by adding more @tool functions and registering them in chain.py.
"""
import os
from typing import Optional

import requests
from langchain_core.tools import tool

# Base URL of this FastAPI app (for self-call to /search). Set e.g. in docker: SELF_BASE_URL=http://web:8001
SELF_BASE_URL = os.getenv("SELF_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
SEARCH_TIMEOUT = int(os.getenv("AGENT_SEARCH_TIMEOUT", "60"))


@tool
def search_travel_knowledge(query: str, city: Optional[str] = None) -> str:
    """
    Search the travel knowledge base (Wikivoyage) for information.
    Use this when you need facts about destinations, transport, attractions, or travel tips.
    :param query: Natural language question or keywords (e.g. "How to get from Brisbane to Gold Coast by public transport")
    :param city: Optional city filter (e.g. brisbane, gold_coast) to restrict results to one place.
    :return: Relevant text snippets from the knowledge base with source titles.
    """
    payload = {"query": query, "topk": 6, "ef": 128}
    if city:
        payload["city"] = city
    r = requests.post(
        f"{SELF_BASE_URL}/search",
        json=payload,
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    hits = data.get("hits", [])
    if not hits:
        return "No relevant results found in the knowledge base."
    parts = []
    for i, h in enumerate(hits, 1):
        title = h.get("title") or h.get("source_title") or "Unknown"
        snippet = (h.get("snippet") or "").strip()
        parts.append(f"[{i}] {title}\n{snippet}")
    return "\n\n".join(parts)
