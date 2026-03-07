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


# ---------- Translink SEQ GTFS (local data/SEQ_GTFS) ----------

def _get_gtfs():
    from .gtfs_client import search_stops as _search_stops, get_departures as _get_departures
    return _search_stops, _get_departures


@tool
def translink_search_stops(query: str) -> str:
    """
    Search Translink SEQ bus/train/ferry stops by name (e.g. 'Queen Street', 'Central', 'Brisbane').
    Use this to find stop_id and coordinates before asking for departures. Data is from local GTFS (South East Queensland).
    :param query: Stop or place name (substring match).
    :return: List of stop_id, stop_name, stop_lat, stop_lon. Use stop_id with translink_get_departures.
    """
    try:
        search_stops_fn, _ = _get_gtfs()
        rows = search_stops_fn(query, limit=15)
    except Exception as e:
        return f"Translink lookup failed: {e!s}"
    if not rows:
        return "No stops found for that query. Try a different name or check SEQ_GTFS data is present."
    lines = ["stop_id | stop_name | lat | lon"]
    for r in rows:
        lines.append(f"{r.get('stop_id', '')} | {r.get('stop_name', '')} | {r.get('stop_lat', '')} | {r.get('stop_lon', '')}")
    return "\n".join(lines)


@tool
def translink_get_departures(stop_id: str, after_time: str = "06:00") -> str:
    """
    Get next departures from a Translink SEQ stop after the given time. Use translink_search_stops first to get stop_id.
    :param stop_id: Stop ID from translink_search_stops (e.g. 100, 1000).
    :param after_time: Time as HH:MM or HH:MM:SS (e.g. 08:00 for 8am). Default 06:00.
    :return: List of route_short_name, trip_headsign, departure_time.
    """
    try:
        _, get_departures_fn = _get_gtfs()
        rows = get_departures_fn(stop_id, after_time, limit=20)
    except Exception as e:
        return f"Translink departures failed: {e!s}"
    if not rows:
        return f"No departures found for stop_id={stop_id} after {after_time}. Check stop_id or try later time."
    lines = ["route | headsign | departure_time"]
    for r in rows:
        lines.append(f"{r.get('route_short_name', '')} | {r.get('trip_headsign', '')} | {r.get('departure_time', '')}")
    return "\n".join(lines)
