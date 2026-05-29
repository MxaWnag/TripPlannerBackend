"""
Agent tools. Use the existing RAG API via HTTP (no import from main app = OCP).
Extend by adding more @tool functions and registering them in chain.py.
"""
import os
from typing import Optional

import requests
from langchain_core.tools import tool

from . import wikivoyage_client as wv

# Base URL of this FastAPI app (for self-call to /search).
# SELF_BASE_URL or AGENT_BASE_URL (e.g. Docker: http://host.docker.internal:8001).
def _self_base_url() -> str:
    return (
        os.getenv("SELF_BASE_URL")
        or os.getenv("AGENT_BASE_URL")
        or "http://127.0.0.1:8001"
    ).rstrip("/")


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
        f"{_self_base_url()}/search",
        json=payload,
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    hits = data.get("hits", [])
    note = (data.get("note") or "").strip()
    if not hits:
        if note:
            return f"No relevant results in the knowledge base. ({note})"
        return "No relevant results found in the knowledge base."
    parts = []
    for i, h in enumerate(hits, 1):
        title = h.get("title") or h.get("source_title") or "Unknown"
        snippet = (h.get("snippet") or "").strip()
        parts.append(f"[{i}] {title}\n{snippet}")
    return "\n\n".join(parts)


# ---------- Wikivoyage live (MediaWiki API) ----------

@tool
def wikivoyage_search(query: str, limit: int = 5) -> str:
    """
    Search English Wikivoyage for travel guide page titles (live API).
    Use when search_travel_knowledge returns nothing or the destination is not in the local index.
    :param query: City or topic (e.g. 'Sydney', 'Tokyo food').
    :param limit: Max number of page titles to return (1-10).
    :return: Matching page titles with Wikivoyage URLs. Follow with wikivoyage_get_page(title=...).
    """
    lim = max(1, min(int(limit), 10))
    try:
        hits = wv.search_titles(query, limit=lim)
    except Exception as e:
        return f"Wikivoyage search failed: {e!s}"
    if not hits:
        return f"No Wikivoyage pages found for '{query}'."
    lines = [f"Wikivoyage search: '{query}' ({len(hits)} hit(s), CC BY-SA)"]
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h['title']}\n{h['url']}")
    lines.append("\nNext: call wikivoyage_get_page with an exact title from above.")
    return "\n".join(lines)


@tool
def wikivoyage_get_page(title: str, sections: Optional[str] = None) -> str:
    """
    Fetch travel guide text from a Wikivoyage page by exact title (live MediaWiki API).
    Use after wikivoyage_search or when you know the page name (e.g. 'Sydney', 'Brisbane').
    :param title: Exact Wikivoyage page title.
    :param sections: Optional comma-separated section names (e.g. 'Get in,Get around,See').
                     Default: Understand, Get in, Get around, See, Do, Eat, Sleep, Stay safe.
    :return: Plain-text sections with source URL. Content is CC BY-SA — cite Wikivoyage in your answer.
    """
    t = (title or "").strip()
    if "/" in t:
        t = t.split("/")[0].strip()
        if not (sections and sections.strip()):
            sections = "See,Do,Get around,Get in"
    sec_list = None
    if sections and sections.strip():
        sec_list = [s.strip() for s in sections.split(",") if s.strip()]
    try:
        return wv.fetch_page_text(t, sections=sec_list)
    except Exception as e:
        return f"Wikivoyage get_page failed: {e!s}"


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
