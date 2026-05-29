"""
Agent execution: LangChain ChatOllama / ChatAnthropic + tool-calling loop.
Uses existing RAG /search via tools (no change to RAG code).
Per-step timing is recorded and written to logs/agent_perf.log (JSONL).
"""
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from . import wikivoyage_client as wv
from .tools import (
    search_travel_knowledge,
    translink_get_departures,
    translink_search_stops,
    wikivoyage_get_page,
    wikivoyage_search,
)

# Per-step performance logger: writes one JSON object per line to logs/agent_perf.log
AGENT_PERF_LOG_DIR = os.getenv("AGENT_PERF_LOG_DIR", "logs")
AGENT_PERF_LOG_FILE = os.getenv("AGENT_PERF_LOG_FILE", "agent_perf.log")


def _get_perf_logger() -> logging.Logger:
    """Return a logger that appends JSONL to logs/agent_perf.log."""
    logger = logging.getLogger("agent.perf")
    if logger.handlers:
        return logger
    os.makedirs(AGENT_PERF_LOG_DIR, exist_ok=True)
    path = os.path.join(AGENT_PERF_LOG_DIR, AGENT_PERF_LOG_FILE)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _log_perf(record: dict) -> None:
    """Write a single JSON line to the agent perf log."""
    record.setdefault("ts", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    _get_perf_logger().info(json.dumps(record, ensure_ascii=False))

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
AGENT_MODEL = os.getenv("AGENT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
WIKIVOYAGE_AUTO_FALLBACK = os.getenv("WIKIVOYAGE_AUTO_FALLBACK", "true").lower() in (
    "1",
    "true",
    "yes",
)
WIKIVOYAGE_MAX_CHARS = int(os.getenv("WIKIVOYAGE_MAX_CHARS", "8000"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
# Models that cannot use Ollama bind_tools (e.g. deepseek-r1) → deterministic tool pipeline + LLM synthesis
OLLAMA_PIPELINE_PREFIXES = tuple(
    p.strip().lower()
    for p in os.getenv("OLLAMA_PIPELINE_PREFIXES", "deepseek-r1").split(",")
    if p.strip()
)

# Claude agent (optional: requires langchain-anthropic)
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
# All tools exposed to the agent (extend here for new tools)
TOOLS = [
    search_travel_knowledge,
    wikivoyage_search,
    wikivoyage_get_page,
    translink_search_stops,
    translink_get_departures,
]
# For experiment baselines: RAG-only (no Translink), Translink-only (no RAG)
TOOLS_RAG_ONLY = [search_travel_knowledge, wikivoyage_search, wikivoyage_get_page]
TOOLS_TRANSLINK_ONLY = [translink_search_stops, translink_get_departures]


def get_tools_for_baseline(baseline: Optional[str] = None) -> List[Any]:
    """Return tool list for experiment baseline. None or 'full' = all tools; 'no_rag' = Translink only; 'no_tools' = RAG only."""
    if baseline == "no_rag":
        return TOOLS_TRANSLINK_ONLY
    if baseline == "no_tools":
        return TOOLS_RAG_ONLY
    return TOOLS


def _looks_like_transit_query(query: str) -> bool:
    q = (query or "").lower()
    transit_keywords = [
        "public transport", "translink", "station", "depart", "departure",
        "bus", "train", "ferry", "stop", "route", "platform", "g:link",
        "arriv", "return", "from ", " to ",
    ]
    return any(k in q for k in transit_keywords)


def _transit_only_query(query: str) -> bool:
    """SEQ stop/departure questions without a multi-day destination plan."""
    return _looks_like_transit_query(query) and not _looks_like_destination_query(query)


def _extract_from_to(query: str) -> tuple[Optional[str], Optional[str]]:
    m = re.search(
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+by\s+|\s+using\s+|\s+tomorrow|\s+this\s+|\s+next\s+|$)",
        query or "",
        re.I,
    )
    if not m:
        return None, None
    return m.group(1).strip(" ?.,"), m.group(2).strip(" ?.,")


def _rag_result_is_empty(result: str) -> bool:
    r = (result or "").lower()
    return "no relevant results" in r


def _rag_result_insufficient(result: str, query: str, city: Optional[str] = None) -> bool:
    """True when RAG missed or returned unrelated cities (e.g. Brisbane hits for a Sydney trip)."""
    if _rag_result_is_empty(result):
        return True
    hint = _guess_wikivoyage_title(query, city)
    if not hint or len(hint) < 3:
        return False
    r = (result or "").lower()
    hint_l = hint.lower()
    if hint_l in r:
        return False
    for token in hint_l.split():
        if len(token) >= 4 and token in r:
            return False
    return True


def _wikivoyage_tools_enabled(use_tools: List[Any]) -> bool:
    names = {getattr(t, "name", "") for t in use_tools}
    return "wikivoyage_search" in names and "wikivoyage_get_page" in names


def _guess_wikivoyage_title(query: str, city: Optional[str] = None) -> str:
    if city:
        return " ".join(part.capitalize() for part in str(city).replace("-", "_").split("_"))
    q = (query or "").strip()
    patterns = (
        r"(?:plan|itinerary|trip|visit|travel)(?:\s+\w+){0,6}?\s+(?:\d+\s+days?\s+)?(?:in|to|for)\s+([A-Za-z][A-Za-z\s\-']{1,48})",
        r"(?:\d+\s+days?\s+in|\bweek\s+in|\btrip\s+to)\s+([A-Za-z][A-Za-z\s\-']{1,48})",
        r"\b(?:in|to|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    )
    for pat in patterns:
        m = re.search(pat, q, re.I)
        if m:
            title = m.group(1).strip().rstrip("?.!,;:")
            if len(title) >= 3:
                return title.title()
    return q[:80].strip() or "travel"


def _looks_like_destination_query(query: str) -> bool:
    q = (query or "").lower()
    keys = (
        "plan",
        "itinerary",
        "trip",
        "visit",
        "days in",
        "week in",
        "travel to",
        "holiday in",
        "vacation in",
        "things to do",
    )
    return any(k in q for k in keys)


def _looks_like_general_planning_query(query: str) -> bool:
    q = (query or "").lower()
    planning_keywords = [
        "itinerary", "one-day", "one day", "attractions", "tips", "visit",
        "plan", "things to do", "morning", "afternoon",
    ]
    return any(k in q for k in planning_keywords)


def resolve_strategy_options(
    strategy: Optional[str],
    query: str,
    baseline: Optional[str],
    max_iterations: int,
) -> dict:
    """
    Return lightweight execution options for strategy modes.
    Defaults keep current behavior unchanged.
    """
    s = (strategy or "s0").lower()
    out = {
        "strategy": s,
        "baseline": baseline,
        "max_iterations": max_iterations,
        "early_stop": False,
        "two_stage": False,
        "stop_reason_hint": None,
    }
    if s == "s0":
        return out
    if s == "s1":
        # Intent-based tool gating (deterministic heuristic).
        is_transit = _looks_like_transit_query(query)
        is_general = _looks_like_general_planning_query(query)
        if is_transit and is_general:
            out["baseline"] = baseline  # mixed: keep baseline / full default
        elif is_transit:
            out["baseline"] = "no_rag"
        else:
            out["baseline"] = "no_tools"
        return out
    if s == "s2":
        out["early_stop"] = True
        return out
    if s == "s3":
        out["two_stage"] = True
        # Keep executor bounded to limit iterative drift.
        out["max_iterations"] = min(max_iterations, 4)
        return out
    return out

SYSTEM_PROMPT = """You are a travel planning assistant. You must plan explicitly step-by-step for each request.

**Planning (required):**
1. First, state your plan in clear steps (e.g. 1) Transport 2) Attractions 3) Tips).
2. Then call tools as needed: search_travel_knowledge first for general travel info; if few/no hits or unknown city, use wikivoyage_search then wikivoyage_get_page. For SEQ public transport use translink_search_stops then translink_get_departures.
3. Synthesize a reply with clear sections (e.g. ## Transport / ## Attractions) and cite sources [1][2].

**Tools:**
- search_travel_knowledge(query, city?): Local Wikivoyage RAG index (fast; may only cover ingested cities).
- wikivoyage_search(query, limit?): Live Wikivoyage — find page titles/URLs (use when RAG is empty or city not in index).
- wikivoyage_get_page(title, sections?): Live Wikivoyage — fetch guide sections by exact title from search results.
- translink_search_stops(query): Find Translink SEQ stops by name; returns stop_id, name, lat, lon. Use for Brisbane/Gold Coast/SEQ bus, train, ferry.
- translink_get_departures(stop_id, after_time): Next departures from a stop. Get stop_id from translink_search_stops first; after_time e.g. 08:00.

**Rules:**
- For "how to get from A to B by public transport" in SEQ: search stops for A and B, then use departures or describe routes from the data.
- Prefer search_travel_knowledge, then wikivoyage_search + wikivoyage_get_page for destinations not in RAG; use Translink tools for concrete stop names and departure times.
- Multi-day trip answers MUST use markdown headings `## Day 1`, `## Day 2`, … (not only `###` subsections).
- For wikivoyage_get_page use the city page title only (e.g. `Sydney`), never subpages like `Sydney/See`; pass section names via the `sections` parameter.
- When using live Wikivoyage text, cite the page URL (CC BY-SA).
- Answer in English. When citing, refer to tool results (e.g. [1] Brisbane#0 ...).
- **Tool use is only via the model's tool-calling mechanism.** Do not paste fake tool calls as JSON in your message (e.g. no `{"name":"translink_search_stops",...}` blocks in prose). If you need data, invoke the real tools and wait for results; never invent stop_id or placeholder tool JSON as a substitute.
- Translink: only use stop_id values that appear in translink_search_stops output; never invent IDs like 600034.
- After tools return, write the final answer from actual tool output only (no hypothetical "assume we found stop 1000" unless that stop_id came from translink_search_stops).
"""

PIPELINE_SYNTHESIS_PROMPT = """You are a travel planning assistant. Tool results are provided below (Wikivoyage guide, optional local index, optional Translink).

Write the final answer in English for the user.
- Multi-day trip requests: use ## Day 1, ## Day 2, … with concrete sights/activities taken ONLY from the guide text.
- Also use ## Transport and ## Tips when the guide or transit data supports them.
- Cite the Wikivoyage Source URL from the tool output (CC BY-SA).
- Do NOT invent fares, dollar amounts, Opal/smartcard names, routes, stop IDs, or URLs absent from the tool outputs.
- For transport, describe modes mentioned in the guide only; say "check local operators" instead of guessing prices.
- End with a line: **Source:** [City](Wikivoyage URL from tool output) (CC BY-SA).
- Output only the final answer (no JSON tool calls, no chain-of-thought)."""


def _ollama_model_uses_pipeline(model: Optional[str] = None) -> bool:
    name = (model or AGENT_MODEL or "").lower()
    return any(name.startswith(p) or p in name for p in OLLAMA_PIPELINE_PREFIXES)


def _build_llm_plain(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ChatOllama:
    return ChatOllama(
        model=model or AGENT_MODEL,
        base_url=base_url or OLLAMA_BASE,
        temperature=0.2,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=OLLAMA_NUM_PREDICT,
    )


def _build_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    tools: Optional[List[Any]] = None,
) -> ChatOllama:
    use_tools = tools if tools is not None else TOOLS
    llm = _build_llm_plain(model=model, base_url=base_url)
    if _ollama_model_uses_pipeline(model):
        return llm
    return llm.bind_tools(use_tools)


def _build_llm_claude(model: Optional[str] = None, tools: Optional[List[Any]] = None):
    """Build Claude LLM with tools. Requires langchain-anthropic and ANTHROPIC_API_KEY."""
    from langchain_anthropic import ChatAnthropic
    use_tools = tools if tools is not None else TOOLS
    return ChatAnthropic(
        model=model or ANTHROPIC_MODEL,
        api_key=ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.2,
        max_tokens=1024,
    ).bind_tools(use_tools)


def _emit(on_event: Optional[Callable[[dict[str, Any]], None]], payload: dict[str, Any]) -> None:
    if not on_event:
        return
    try:
        on_event(payload)
    except Exception:
        pass


def _extract_chunk_text(chunk: Any) -> str:
    """Text delta from a single stream chunk (str or content blocks)."""
    c = getattr(chunk, "content", None)
    if not c:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(c)


def _plain_text_from_message_content(content: Any) -> str:
    """Normalize AIMessage.content (str or block list) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()


def _aimessage_from_chunk(merged: AIMessageChunk) -> AIMessage:
    """Turn aggregated AIMessageChunk into AIMessage for tool routing (OCP: same downstream loop)."""
    raw_tc = getattr(merged, "tool_calls", None) or []
    return AIMessage(
        content=merged.content,
        tool_calls=list(raw_tc),
        id=getattr(merged, "id", None),
        response_metadata=dict(getattr(merged, "response_metadata", None) or {}),
    )


def _extract_text_tool_calls(text: str, tool_names: set[str]) -> List[dict]:
    """
    Ollama / smaller models often emit fake tool JSON in message text instead of
    structured AIMessage.tool_calls. Recover those so the agent loop can execute tools.
    """
    if not text or not tool_names:
        return []
    out: List[dict] = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        i = end
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool")
        if name not in tool_names:
            continue
        args = obj.get("parameters") or obj.get("args") or obj.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        out.append({"name": name, "args": args, "id": f"text-fallback-{len(out)}"})
    return out


def _record_tool_step(
    *,
    steps: List[dict[str, Any]],
    log_ctx: dict,
    on_event: Optional[Callable[[dict[str, Any]], None]],
    round_index: int,
    name: str,
    args: dict,
    result: str,
    duration_sec: float,
    auto_fallback: bool = False,
) -> None:
    _log_perf({
        **log_ctx,
        "event": "tool_call",
        "round": round_index,
        "tool": name,
        "duration_sec": round(duration_sec, 4),
        "auto_fallback": auto_fallback,
    })
    step = {
        "role": "tool",
        "name": name,
        "args": args,
        "result_preview": str(result)[:200],
        "duration_sec": round(duration_sec, 4),
        "round": round_index,
    }
    if auto_fallback:
        step["auto_fallback"] = True
    if name == "wikivoyage_get_page":
        step["result_full"] = str(result)
    steps.append(step)
    _emit(on_event, {"type": "step", "step": step})


def _run_auto_wikivoyage_fallback(
    *,
    use_tools: List[Any],
    hint: str,
    user_query: str,
    steps: List[dict[str, Any]],
    messages: List[BaseMessage],
    log_ctx: dict,
    on_event: Optional[Callable[[dict[str, Any]], None]],
    round_index: int,
) -> bool:
    """
    Deterministic live Wikivoyage fetch when local RAG is empty or the LLM skips tools.
    Returns True if guide text was injected into the conversation.
    """
    if not _wikivoyage_tools_enabled(use_tools):
        return False

    search_fn = next((t for t in use_tools if getattr(t, "name", "") == "wikivoyage_search"), None)
    get_fn = next((t for t in use_tools if getattr(t, "name", "") == "wikivoyage_get_page"), None)
    if not search_fn or not get_fn:
        return False

    _emit(
        on_event,
        {
            "type": "status",
            "phase": "tool",
            "round": round_index,
            "message": f"Auto Wikivoyage fallback: {hint}",
            "auto_fallback": True,
        },
    )

    t0 = time.perf_counter()
    try:
        search_out = search_fn.invoke({"query": hint, "limit": 3})
    except Exception as e:
        search_out = f"Wikivoyage search failed: {e!s}"
    _record_tool_step(
        steps=steps,
        log_ctx=log_ctx,
        on_event=on_event,
        round_index=round_index,
        name="wikivoyage_search",
        args={"query": hint, "limit": 3},
        result=search_out,
        duration_sec=time.perf_counter() - t0,
        auto_fallback=True,
    )

    page_title = hint
    for line in str(search_out).splitlines():
        line = line.strip()
        if line.startswith("[1]"):
            page_title = line[3:].strip().split("\n")[0].strip()
            break

    t1 = time.perf_counter()
    page_sections = ",".join(wv.sections_for_query(user_query))
    try:
        page_out = get_fn.invoke({"title": page_title, "sections": page_sections})
    except Exception as e:
        page_out = f"Wikivoyage get_page failed: {e!s}"
    _record_tool_step(
        steps=steps,
        log_ctx=log_ctx,
        on_event=on_event,
        round_index=round_index,
        name="wikivoyage_get_page",
        args={"title": page_title, "sections": page_sections},
        result=page_out,
        duration_sec=time.perf_counter() - t1,
        auto_fallback=True,
    )

    if page_out.startswith("Wikivoyage get_page failed") or page_out.startswith("Failed to"):
        return False

    messages.append(
        SystemMessage(
            content=(
                "Local RAG had no useful hits. Live Wikivoyage guide text was fetched automatically below. "
                "Write the final answer using ONLY that guide text. "
                "If the user asked for a multi-day plan, structure the reply as ## Day 1, ## Day 2, etc. "
                "Cite the Source URL from the guide. Do not invent transport fares, routes, or URLs."
            )
        )
    )
    messages.append(
        HumanMessage(
            content=(
                f"--- Wikivoyage (auto-fetched) ---\n{page_out}\n\n"
                f"--- User request ---\n{user_query}\n\n"
                "Provide the final travel answer now."
            )
        )
    )
    _log_perf({**log_ctx, "event": "wikivoyage_auto_fallback", "hint": hint, "page_title": page_title})
    return True


def _parse_trip_days(query: str) -> int:
    m = re.search(r"(\d+)\s*days?", query or "", re.I)
    if m:
        return max(1, min(int(m.group(1)), 14))
    return 3


def _extract_section_lines(guide_text: str, section_names: List[str]) -> List[str]:
    """Collect bullet lines and substantive sentences under ## Section headers."""
    targets = {n.lower() for n in section_names}
    lines = guide_text.splitlines()
    in_target = False
    items: List[str] = []
    for line in lines:
        if line.startswith("## "):
            hdr = line[3:].strip()
            hdr_plain = re.sub(r"<[^>]+>", "", hdr).strip()
            in_target = hdr_plain.lower() in targets or any(
                t in hdr_plain.lower() for t in targets
            )
            continue
        if not in_target:
            continue
        s = line.strip()
        if not s or re.match(r"^\[\s*edit\s*\]$", s, re.I):
            continue
        if re.match(r"^[\*\-•]\s+\S", s):
            items.append(re.sub(r"^[\*\-•]\s+", "", s).strip())
        elif re.match(r"^\d+\.\s+\S", s):
            items.append(re.sub(r"^\d+\.\s+", "", s).strip())
        elif len(s) > 35 and not s.startswith("|") and "°" not in s[:20]:
            items.append(s)
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        key = it.lower()[:96]
        if key in seen or len(it) < 10:
            continue
        seen.add(key)
        out.append(it)
    return out[:36]


def _extract_list_items(guide_text: str) -> List[str]:
    items = _extract_section_lines(guide_text, ["See", "Do", "Get around", "Get in"])
    if items:
        return items
    for line in guide_text.splitlines():
        s = line.strip()
        if re.match(r"^[\*\-•]\s+\S", s):
            items.append(re.sub(r"^[\*\-•]\s+", "", s).strip())
    return items[:24]


def _format_itinerary_draft(query: str, guide_text: str) -> str:
    days = _parse_trip_days(query)
    dest = _guess_wikivoyage_title(query)
    source_line = ""
    for line in guide_text.splitlines():
        if line.startswith("Source:"):
            source_line = line.strip()
            break
    bullets = _extract_list_items(guide_text)
    lines = [
        f"# {days}-day trip: {dest}",
        "",
        source_line or f"Source: {wv.page_url(dest)} (Wikivoyage, CC BY-SA)",
        "",
        "_Draft itinerary from Wikivoyage highlights (verify times and fares locally)._",
        "",
    ]
    transport = _extract_section_lines(guide_text, ["Get in", "Get around"])[:4]
    if transport:
        lines.append("## Transport")
        for t in transport:
            lines.append(f"- {t[:220]}")
        lines.append("")

    if bullets:
        per = max(1, (len(bullets) + days - 1) // days)
        for d in range(days):
            chunk = bullets[d * per : (d + 1) * per]
            lines.append(f"## Day {d + 1}")
            for b in chunk[:6]:
                short = b[:200] + ("…" if len(b) > 200 else "")
                lines.append(f"- {short}")
            lines.append("")
    else:
        for d in range(days):
            lines.append(f"## Day {d + 1}")
            lines.append("- Explore major **See** / **Do** sights from the Wikivoyage guide (reference below).")
            lines.append("")

    lines.append("## Tips")
    lines.append("- Check the Wikivoyage page for hours, closures, and booking needs before you go.")
    lines.append("")
    return "\n".join(lines).strip()


def _synthesize_answer_from_wikivoyage(steps: List[dict[str, Any]], query: str) -> Optional[str]:
    """If the LLM returns empty text, build an answer from fetched guide content."""
    for s in reversed(steps):
        if s.get("name") != "wikivoyage_get_page":
            continue
        text = (s.get("result_full") or s.get("result_preview") or "").strip()
        if not text or text.startswith("Wikivoyage get_page failed"):
            continue
        if _looks_like_destination_query(query):
            return _format_itinerary_draft(query, text)
        cap = min(WIKIVOYAGE_MAX_CHARS, 6500)
        return (
            f"# Travel guide (live Wikivoyage)\n\n"
            f"**Your request:** {query.strip()}\n\n"
            f"{text[:cap]}"
        )
    return None


def _strip_reasoning_tags(text: str) -> str:
    """Remove DeepSeek-R1 style reasoning blocks from model output."""
    if not text:
        return ""
    out = text
    # XML-style blocks emitted by some R1 templates
    out = re.sub(
        r"<\s*think\b[^>]*>[\s\S]*?<\s*/\s*think\s*>",
        "",
        out,
        flags=re.IGNORECASE,
    )
    if "" in out.lower() or "" in out.lower():
        out = re.sub(r"`[\s\S]*?`", "", out)
    return out.strip()


def _wikivoyage_fetched(steps: List[dict[str, Any]]) -> bool:
    return any(s.get("role") == "tool" and s.get("name") == "wikivoyage_get_page" for s in steps)


def _wikivoyage_guide_useful(steps: List[dict[str, Any]]) -> bool:
    for s in steps:
        if s.get("name") != "wikivoyage_get_page":
            continue
        text = (s.get("result_full") or s.get("result_preview") or "").strip()
        if not text:
            continue
        if text.startswith(("No matching sections", "Failed to", "Wikivoyage get_page failed")):
            continue
        return True
    return False


def _tool_corpus(steps: List[dict[str, Any]]) -> str:
    parts: List[str] = []
    for s in steps:
        if s.get("role") != "tool":
            continue
        parts.append(s.get("result_full") or s.get("result_preview") or "")
    return "\n".join(parts)


def _valid_stop_ids(corpus: str) -> set[str]:
    ids: set[str] = set()
    for line in (corpus or "").splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0].isdigit():
            ids.add(parts[0])
    return ids


def _scrub_unsupported_transport_claims(answer: str, corpus: str) -> str:
    """Drop fare/card specifics and fake stop IDs not present in tool output."""
    if not answer:
        return answer
    corpus_l = (corpus or "").lower()
    valid_stops = _valid_stop_ids(corpus)
    out_lines: List[str] = []
    for line in answer.splitlines():
        drop = False
        if re.search(r"\$\s*\d", line) and not re.search(r"\$\s*\d", corpus):
            drop = True
        if re.search(r"\bopal\b", line, re.I) and "opal" not in corpus_l:
            drop = True
        if valid_stops:
            for sid in re.findall(r"\b(\d{5,7})\b", line):
                if sid not in valid_stops:
                    line = re.sub(rf"\b{sid}\b", "_(stop id not in tool data)_", line)
        if not drop:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def _has_day_structure(text: str) -> bool:
    return bool(re.search(r"#{2,4}\s*Day\s*\d", text or "", re.I))


def _normalize_day_headings(text: str) -> str:
    """Product format uses ## Day N (normalize #### Day N from some models)."""
    return re.sub(
        r"^#{3,4}\s*Day\s*(\d+)\s*:?\s*",
        r"## Day \1: ",
        text or "",
        flags=re.I | re.M,
    )


def _last_assistant_text(steps: List[dict[str, Any]]) -> str:
    for s in reversed(steps):
        if s.get("role") == "assistant" and (s.get("content") or "").strip():
            return str(s["content"]).strip()
    return ""


def _wikivoyage_source_url(steps: List[dict[str, Any]], query: str) -> Optional[str]:
    for s in reversed(steps):
        if s.get("name") != "wikivoyage_get_page":
            continue
        text = s.get("result_full") or s.get("result_preview") or ""
        for line in text.splitlines():
            if line.startswith("Source:"):
                m = re.search(r"https?://\S+", line)
                if m:
                    return m.group(0).rstrip(").,")
        title = ""
        for line in text.splitlines():
            if line.startswith("Page:"):
                title = line[5:].strip()
                break
        if title:
            return wv.page_url(title)
    if _looks_like_destination_query(query):
        return wv.page_url(_guess_wikivoyage_title(query))
    return None


def _ensure_translink_source_footer(answer: str, steps: List[dict[str, Any]], query: str) -> str:
    if not _transit_only_query(query):
        return answer
    if not any(
        s.get("role") == "tool" and str(s.get("name", "")).startswith("translink")
        for s in steps
    ):
        return answer
    if "gtfs" in answer.lower() or "translink" in answer.lower():
        return answer
    return f"{answer.rstrip()}\n\n**Source:** Translink SEQ GTFS (local schedule data)."


def _ensure_wikivoyage_source_footer(answer: str, steps: List[dict[str, Any]], query: str) -> str:
    if not _wikivoyage_guide_useful(steps):
        return answer
    url = _wikivoyage_source_url(steps, query)
    if not url:
        return answer
    if url in answer or "wikivoyage.org/wiki/" in answer.lower():
        return answer
    dest = _guess_wikivoyage_title(query)
    return f"{answer.rstrip()}\n\n---\n\n**Source:** [{dest}]({url}) (Wikivoyage, CC BY-SA)"


def _finalize_answer(answer: str, steps: List[dict[str, Any]], query: str) -> str:
    cleaned = _strip_reasoning_tags((answer or "").strip())
    if len(cleaned) < 80:
        cleaned = _strip_reasoning_tags(_last_assistant_text(steps))
    corpus = _tool_corpus(steps)
    cleaned = _scrub_unsupported_transport_claims(cleaned, corpus)
    if _looks_like_destination_query(query):
        cleaned = _normalize_day_headings(cleaned)
    draft = _synthesize_answer_from_wikivoyage(steps, query) if _wikivoyage_guide_useful(steps) else None
    if draft and _looks_like_destination_query(query):
        if len(cleaned) < 120:
            cleaned = draft
        elif not _has_day_structure(cleaned) and len(cleaned) < 500:
            cleaned = draft
    if len(cleaned) < 120 and draft:
        cleaned = draft
    if not cleaned:
        cleaned = "Unable to produce an answer; try again or use the Claude chat endpoint."
    cleaned = _ensure_wikivoyage_source_footer(cleaned, steps, query)
    return _ensure_translink_source_footer(cleaned, steps, query)


def _bootstrap_classic_tools(
    query: str,
    use_tools: List[Any],
    messages: List[BaseMessage],
    steps: List[dict[str, Any]],
    log_ctx: dict,
    on_event: Optional[Callable[[dict[str, Any]], None]],
) -> None:
    """Pre-fetch key tools before the first LLM turn (classic loop only)."""
    tool_names = {getattr(t, "name", "") for t in use_tools}
    blocks: List[str] = []
    round_i = 0

    if _looks_like_destination_query(query) and not _transit_only_query(query):
        if "search_travel_knowledge" in tool_names:
            rag = _pipeline_invoke_tool(
                use_tools,
                "search_travel_knowledge",
                {"query": query},
                steps=steps,
                log_ctx=log_ctx,
                on_event=on_event,
                round_index=round_i,
            )
            blocks.append(f"### search_travel_knowledge\n{rag}")
            if _rag_result_insufficient(rag, query) and _wikivoyage_tools_enabled(use_tools):
                hint = _guess_wikivoyage_title(query)
                if "wikivoyage_search" in tool_names:
                    wv_s = _pipeline_invoke_tool(
                        use_tools,
                        "wikivoyage_search",
                        {"query": hint, "limit": 3},
                        steps=steps,
                        log_ctx=log_ctx,
                        on_event=on_event,
                        round_index=round_i,
                        auto_fallback=True,
                    )
                    blocks.append(f"### wikivoyage_search\n{wv_s}")
                    page_title = hint
                    for line in wv_s.splitlines():
                        if line.strip().startswith("[1]"):
                            page_title = line.strip()[3:].strip().split("\n")[0].strip()
                            break
                if "wikivoyage_get_page" in tool_names:
                    secs = ",".join(wv.sections_for_query(query))
                    wv_p = _pipeline_invoke_tool(
                        use_tools,
                        "wikivoyage_get_page",
                        {"title": page_title, "sections": secs},
                        steps=steps,
                        log_ctx=log_ctx,
                        on_event=on_event,
                        round_index=round_i,
                        auto_fallback=True,
                    )
                    blocks.append(f"### wikivoyage_get_page\n{wv_p}")

    if _looks_like_transit_query(query) and "translink_search_stops" in tool_names:
        from_hint, to_hint = _extract_from_to(query)
        search_jobs: List[tuple[str, str]] = []
        if from_hint:
            search_jobs.append(("origin", from_hint))
        if to_hint and to_hint.lower() != (from_hint or "").lower():
            search_jobs.append(("destination", to_hint))
        if not search_jobs:
            search_jobs = [("origin", "Roma Street")]
        for role, hint in search_jobs[:3]:
            tr = _pipeline_invoke_tool(
                use_tools,
                "translink_search_stops",
                {"query": hint},
                steps=steps,
                log_ctx=log_ctx,
                on_event=on_event,
                round_index=round_i,
            )
            blocks.append(f"### translink_search_stops ({role}: {hint})\n{tr}")
            if role == "origin" and "translink_get_departures" in tool_names and "stop_id |" in tr:
                hint_l = hint.lower()
                origin_stop_id: Optional[str] = None
                for line in tr.splitlines()[1:8]:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) < 2 or not parts[0].isdigit():
                        continue
                    if hint_l in parts[1].lower():
                        origin_stop_id = parts[0]
                        break
                if not origin_stop_id:
                    parts = [p.strip() for p in tr.splitlines()[1].split("|")]
                    if parts and parts[0].isdigit():
                        origin_stop_id = parts[0]
                if origin_stop_id:
                    dep = _pipeline_invoke_tool(
                        use_tools,
                        "translink_get_departures",
                        {"stop_id": origin_stop_id, "after_time": "08:00"},
                        steps=steps,
                        log_ctx=log_ctx,
                        on_event=on_event,
                        round_index=round_i,
                    )
                    blocks.append(
                        f"### translink_get_departures (origin {origin_stop_id})\n{dep}"
                    )

    if blocks:
        messages.append(
            SystemMessage(
                content=(
                    "Tool results were pre-fetched below. Use them for facts; you may call more tools "
                    "if needed, then give the final answer."
                )
            )
        )
        messages.append(
            HumanMessage(
                content="Pre-fetched tool results:\n\n" + "\n\n".join(blocks)
            )
        )


def _pipeline_invoke_tool(
    use_tools: List[Any],
    name: str,
    args: dict,
    *,
    steps: List[dict[str, Any]],
    log_ctx: dict,
    on_event: Optional[Callable[[dict[str, Any]], None]],
    round_index: int,
    auto_fallback: bool = False,
) -> str:
    _emit(
        on_event,
        {
            "type": "status",
            "phase": "tool",
            "round": round_index,
            "message": f"Tool: {name}",
            "tool": name,
        },
    )
    tool_fn = next((t for t in use_tools if getattr(t, "name", "") == name), None)
    t0 = time.perf_counter()
    if not tool_fn:
        result = f"Unknown tool: {name}"
    else:
        try:
            result = tool_fn.invoke(args)
        except Exception as e:
            result = f"Tool error: {e!s}"
    _record_tool_step(
        steps=steps,
        log_ctx=log_ctx,
        on_event=on_event,
        round_index=round_index,
        name=name,
        args=args,
        result=str(result),
        duration_sec=time.perf_counter() - t0,
        auto_fallback=auto_fallback,
    )
    return str(result)


def _run_ollama_pipeline(
    query: str,
    model: Optional[str],
    system_prompt: Optional[str],
    previous_messages: Optional[List[Tuple[str, str]]],
    use_tools: List[Any],
    log_ctx: dict,
    on_event: Optional[Callable[[dict[str, Any]], None]],
    stream_tokens: bool,
) -> tuple[str, List[dict[str, Any]], float]:
    """
    Deterministic tools + single LLM synthesis (for models without Ollama tool-calling).
    """
    tool_names = {getattr(t, "name", "") for t in use_tools}
    steps: List[dict[str, Any]] = []
    t0 = time.perf_counter()
    round_i = 1
    tool_blocks: List[str] = []

    _emit(on_event, {"type": "run_start", "max_iterations": 1, "tools": list(tool_names), "pipeline": True})

    if "search_travel_knowledge" in tool_names:
        rag = _pipeline_invoke_tool(
            use_tools,
            "search_travel_knowledge",
            {"query": query},
            steps=steps,
            log_ctx=log_ctx,
            on_event=on_event,
            round_index=round_i,
        )
        tool_blocks.append(f"### search_travel_knowledge\n{rag}")
        if _rag_result_insufficient(rag, query) and not _transit_only_query(query):
            hint = _guess_wikivoyage_title(query)
            if "wikivoyage_search" in tool_names:
                wv_s = _pipeline_invoke_tool(
                    use_tools,
                    "wikivoyage_search",
                    {"query": hint, "limit": 3},
                    steps=steps,
                    log_ctx=log_ctx,
                    on_event=on_event,
                    round_index=round_i,
                    auto_fallback=True,
                )
                tool_blocks.append(f"### wikivoyage_search\n{wv_s}")
                page_title = hint
                for line in wv_s.splitlines():
                    if line.strip().startswith("[1]"):
                        page_title = line.strip()[3:].strip().split("\n")[0].strip()
                        break
            if "wikivoyage_get_page" in tool_names:
                secs = ",".join(wv.sections_for_query(query))
                wv_p = _pipeline_invoke_tool(
                    use_tools,
                    "wikivoyage_get_page",
                    {"title": page_title, "sections": secs},
                    steps=steps,
                    log_ctx=log_ctx,
                    on_event=on_event,
                    round_index=round_i,
                    auto_fallback=True,
                )
                tool_blocks.append(f"### wikivoyage_get_page\n{wv_p}")
    elif _looks_like_destination_query(query) and _wikivoyage_tools_enabled(use_tools):
        hint = _guess_wikivoyage_title(query)
        if "wikivoyage_search" in tool_names:
            wv_s = _pipeline_invoke_tool(
                use_tools,
                "wikivoyage_search",
                {"query": hint, "limit": 3},
                steps=steps,
                log_ctx=log_ctx,
                on_event=on_event,
                round_index=round_i,
                auto_fallback=True,
            )
            tool_blocks.append(f"### wikivoyage_search\n{wv_s}")
        if "wikivoyage_get_page" in tool_names:
            wv_p = _pipeline_invoke_tool(
                use_tools,
                "wikivoyage_get_page",
                {"title": hint, "sections": ",".join(wv.sections_for_query(query))},
                steps=steps,
                log_ctx=log_ctx,
                on_event=on_event,
                round_index=round_i,
                auto_fallback=True,
            )
            tool_blocks.append(f"### wikivoyage_get_page\n{wv_p}")

    if _looks_like_transit_query(query) and "translink_search_stops" in tool_names:
        from_hint, to_hint = _extract_from_to(query)
        search_jobs: List[tuple[str, str]] = []
        if from_hint:
            search_jobs.append(("origin", from_hint))
        if to_hint and to_hint.lower() != (from_hint or "").lower():
            search_jobs.append(("destination", to_hint))
        if not search_jobs:
            search_jobs = [("origin", "Roma Street")]

        origin_stop_id: Optional[str] = None
        for role, hint in search_jobs[:3]:
            tr = _pipeline_invoke_tool(
                use_tools,
                "translink_search_stops",
                {"query": hint},
                steps=steps,
                log_ctx=log_ctx,
                on_event=on_event,
                round_index=round_i,
            )
            tool_blocks.append(f"### translink_search_stops ({role}: {hint})\n{tr}")
            if role != "origin" or "stop_id |" not in tr or "translink_get_departures" not in tool_names:
                continue
            hint_l = hint.lower()
            for line in tr.splitlines()[1:8]:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                if hint_l in parts[1].lower() or "bus" in (query or "").lower():
                    origin_stop_id = parts[0]
                    break
            if not origin_stop_id:
                parts = [p.strip() for p in tr.splitlines()[1].split("|")]
                if parts and parts[0].isdigit():
                    origin_stop_id = parts[0]
            if origin_stop_id:
                dep = _pipeline_invoke_tool(
                    use_tools,
                    "translink_get_departures",
                    {"stop_id": origin_stop_id, "after_time": "08:00"},
                    steps=steps,
                    log_ctx=log_ctx,
                    on_event=on_event,
                    round_index=round_i,
                )
                tool_blocks.append(
                    f"### translink_get_departures (origin stop {origin_stop_id}, after 08:00)\n{dep}"
                )

    _emit(on_event, {"type": "status", "phase": "llm", "round": 2, "message": "Synthesis (pipeline)…"})
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt or PIPELINE_SYNTHESIS_PROMPT)]
    if previous_messages:
        for role, content in previous_messages:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))
    synth_tail = "Write the complete travel answer now."
    if _transit_only_query(query):
        synth_tail = (
            "This is a SEQ public-transport question. Use translink_search_stops and "
            "translink_get_departures only. Depart from the origin stop in the user question; "
            "prefer routes whose headsign matches the destination. Do not cite Wikivoyage. "
            + synth_tail
        )
    messages.append(
        HumanMessage(
            content=(
                "Tool results (use ONLY these facts):\n\n"
                + "\n\n".join(tool_blocks)
                + f"\n\n{synth_tail}"
            )
        )
    )
    llm = _build_llm_plain(model=model)
    response, dt_llm = _llm_invoke_round(
        llm,
        messages,
        on_event=on_event,
        stream_tokens=stream_tokens,
        round_index=2,
    )
    steps.append({
        "role": "assistant",
        "content": getattr(response, "content", "") or "",
        "duration_sec": round(dt_llm, 4),
        "round": 2,
        "pipeline_synthesis": True,
    })
    _emit(on_event, {"type": "step", "step": steps[-1]})

    answer = _finalize_answer((response.content or "").strip(), steps, query)
    elapsed = time.perf_counter() - t0
    _log_perf({
        **log_ctx,
        "event": "run_complete",
        "total_sec": round(elapsed, 4),
        "rounds": 2,
        "tool_calls_count": sum(1 for s in steps if s.get("role") == "tool"),
        "stop_reason": "ollama_pipeline",
    })
    return answer, steps, elapsed


def _ensure_tool_calls(response: AIMessage, tool_names: set[str]) -> AIMessage:
    """Use native tool_calls when present; else parse JSON tool blobs from text content."""
    native = getattr(response, "tool_calls", None) or []
    if native:
        return response
    text = _plain_text_from_message_content(getattr(response, "content", ""))
    recovered = _extract_text_tool_calls(text, tool_names)
    if not recovered:
        return response
    meta = dict(getattr(response, "response_metadata", None) or {})
    meta["text_tool_call_fallback"] = True
    return AIMessage(
        content=response.content,
        tool_calls=recovered,
        id=getattr(response, "id", None),
        response_metadata=meta,
    )


def _llm_invoke_round(
    llm,
    messages: List[BaseMessage],
    *,
    on_event: Optional[Callable[[dict[str, Any]], None]],
    stream_tokens: bool,
    round_index: int,
) -> tuple[AIMessage, float]:
    """
    One LLM call: blocking invoke by default; when stream_tokens and on_event are set,
    use llm.stream() and emit text_delta events (true token/stream chunk streaming).
    Falls back to invoke if stream raises.
    """
    t_llm = time.perf_counter()
    if not stream_tokens or not on_event:
        response: AIMessage = llm.invoke(messages)
        return response, time.perf_counter() - t_llm

    merged: Optional[AIMessageChunk] = None
    try:
        for chunk in llm.stream(messages):
            if merged is None:
                merged = chunk  # type: ignore[assignment]
            else:
                merged = merged + chunk
            delta = _extract_chunk_text(chunk)
            if delta:
                _emit(on_event, {"type": "text_delta", "content": delta, "round": round_index})
    except Exception:
        response = llm.invoke(messages)
        return response, time.perf_counter() - t_llm

    if merged is None:
        return AIMessage(content=""), time.perf_counter() - t_llm
    try:
        response = _aimessage_from_chunk(merged)
    except Exception:
        response = llm.invoke(messages)
    return response, time.perf_counter() - t_llm


def _run_agent_loop(
    llm,
    query: str,
    system_prompt: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    previous_messages: Optional[List[Tuple[str, str]]] = None,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
    llm_backend: str = "ollama",
    tools: Optional[List[Any]] = None,
    perf_extra: Optional[dict] = None,
    early_stop: bool = False,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    stream_tokens: bool = False,
) -> tuple[str, List[dict[str, Any]], float]:
    """Shared tool-calling loop. If previous_messages is set, prepend (role, content) history for multi-turn.
    Logs each LLM invoke and tool call with duration to logs/agent_perf.log (JSONL).
    perf_extra: merged into every perf log line (e.g. experiment_baseline, client_run_id).
    """
    use_tools = tools if tools is not None else TOOLS
    tool_names = {getattr(t, "name", "") for t in use_tools}
    rid = run_id or str(uuid.uuid4())[:8]
    extras = {k: v for k, v in (perf_extra or {}).items() if v is not None}
    log_ctx = {"run_id": rid, "llm_backend": llm_backend, **extras}
    if session_id:
        log_ctx["session_id"] = session_id

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt or SYSTEM_PROMPT)]
    if previous_messages:
        for role, content in previous_messages:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))
    steps: List[dict[str, Any]] = []
    t0 = time.perf_counter()

    _log_perf({
        **log_ctx,
        "event": "run_start",
        "query_preview": (query or "")[:240],
        "tools_bound": [getattr(t, "name", "") for t in use_tools],
        "max_llm_rounds": max_iterations,
    })
    _emit(on_event, {"type": "run_start", "max_iterations": max_iterations, "tools": [getattr(t, "name", "") for t in use_tools]})

    _bootstrap_classic_tools(query, use_tools, messages, steps, log_ctx, on_event)

    transit_query = _looks_like_transit_query(query)
    stop_reason = "natural_completion"
    rag_empty = False
    wikivoyage_done = False
    for iteration in range(max_iterations):
        _emit(
            on_event,
            {"type": "status", "phase": "llm", "round": iteration + 1, "message": f"LLM invoke (round {iteration + 1})…"},
        )
        response, dt_llm = _llm_invoke_round(
            llm,
            messages,
            on_event=on_event,
            stream_tokens=stream_tokens,
            round_index=iteration + 1,
        )
        response = _ensure_tool_calls(response, tool_names)
        _log_perf({
            **log_ctx,
            "event": "llm_invoke",
            "round": iteration + 1,
            "duration_sec": round(dt_llm, 4),
            "text_tool_call_fallback": bool(
                (getattr(response, "response_metadata", None) or {}).get("text_tool_call_fallback")
            ),
        })
        steps.append({
            "role": "assistant",
            "content": getattr(response, "content", "") or "",
            "duration_sec": round(dt_llm, 4),
            "round": iteration + 1,
            "text_tool_call_fallback": bool(
                (getattr(response, "response_metadata", None) or {}).get("text_tool_call_fallback")
            ),
        })
        _emit(on_event, {"type": "step", "step": steps[-1]})

        if not getattr(response, "tool_calls", None):
            need_wv = (
                WIKIVOYAGE_AUTO_FALLBACK
                and _wikivoyage_tools_enabled(use_tools)
                and not wikivoyage_done
                and not _transit_only_query(query)
                and (rag_empty or _looks_like_destination_query(query))
            )
            if need_wv:
                hint = _guess_wikivoyage_title(query)
                if _run_auto_wikivoyage_fallback(
                    use_tools=use_tools,
                    hint=hint,
                    user_query=query,
                    steps=steps,
                    messages=messages,
                    log_ctx=log_ctx,
                    on_event=on_event,
                    round_index=iteration + 1,
                ):
                    wikivoyage_done = True
                    if steps and steps[-1].get("role") == "assistant":
                        steps[-1]["superseded"] = True
                    stop_reason = "wikivoyage_auto_fallback_finalize"
                    continue

            answer = _finalize_answer((response.content or "").strip(), steps, query)
            elapsed = time.perf_counter() - t0
            _log_perf({
                **log_ctx,
                "event": "run_complete",
                "total_sec": round(elapsed, 4),
                "rounds": iteration + 1,
                "tool_calls_count": sum(1 for s in steps if s.get("role") == "tool"),
                "stop_reason": stop_reason,
            })
            return answer, steps, elapsed

        messages.append(response)
        for tc in response.tool_calls:
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None) or ""
            args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else None) or {}
            tool_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None) or ""
            _emit(
                on_event,
                {
                    "type": "status",
                    "phase": "tool",
                    "round": iteration + 1,
                    "message": f"Tool: {name}",
                    "tool": name,
                    "args_preview": str(args)[:200],
                },
            )
            tool_fn = next((t for t in use_tools if t.name == name), None)
            t_tool = time.perf_counter()
            if not tool_fn:
                result = f"Unknown tool: {name}"
            else:
                try:
                    result = tool_fn.invoke(args)
                except Exception as e:
                    result = f"Tool error: {e!s}"
            dt_tool = time.perf_counter() - t_tool
            _record_tool_step(
                steps=steps,
                log_ctx=log_ctx,
                on_event=on_event,
                round_index=iteration + 1,
                name=name,
                args=args if isinstance(args, dict) else {},
                result=str(result),
                duration_sec=dt_tool,
            )
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id or name))

            if name in ("wikivoyage_search", "wikivoyage_get_page"):
                wikivoyage_done = True
            city_arg = args.get("city") if isinstance(args, dict) else None
            if name == "search_travel_knowledge" and _rag_result_insufficient(
                str(result), query, city_arg
            ):
                rag_empty = True
                if (
                    WIKIVOYAGE_AUTO_FALLBACK
                    and not wikivoyage_done
                    and _wikivoyage_tools_enabled(use_tools)
                ):
                    hint = _guess_wikivoyage_title(query, city_arg)
                    if _run_auto_wikivoyage_fallback(
                        use_tools=use_tools,
                        hint=hint,
                        user_query=query,
                        steps=steps,
                        messages=messages,
                        log_ctx=log_ctx,
                        on_event=on_event,
                        round_index=iteration + 1,
                    ):
                        wikivoyage_done = True
                        messages.append(
                            SystemMessage(
                                content=(
                                    "Wikivoyage live guide was auto-fetched after empty local RAG. "
                                    "Use the tool results above for facts; you may call more tools or answer now."
                                )
                            )
                        )

        if early_stop:
            used_tools = [s.get("name") for s in steps if s.get("role") == "tool"]
            has_rag = "search_travel_knowledge" in used_tools
            has_wv = "wikivoyage_get_page" in used_tools
            has_departures = "translink_get_departures" in used_tools
            has_stop_search = "translink_search_stops" in used_tools
            enough_evidence = (has_stop_search and has_departures) if transit_query else (
                has_rag or has_stop_search or has_wv
            )
            if enough_evidence:
                messages.append(SystemMessage(
                    content="You have sufficient evidence. Provide the final answer now without further tool calls."
                ))
                _emit(on_event, {"type": "status", "phase": "llm", "message": "Early-stop finalize (LLM)…"})
                final_response, dt_llm = _llm_invoke_round(
                    llm,
                    messages,
                    on_event=on_event,
                    stream_tokens=stream_tokens,
                    round_index=iteration + 2,
                )
                _log_perf({
                    **log_ctx,
                    "event": "llm_invoke",
                    "round": iteration + 2,
                    "duration_sec": round(dt_llm, 4),
                })
                steps.append({
                    "role": "assistant",
                    "content": getattr(final_response, "content", "") or "",
                    "duration_sec": round(dt_llm, 4),
                    "stage": "early_stop_finalize",
                    "round": iteration + 2,
                })
                _emit(on_event, {"type": "step", "step": steps[-1]})
                answer = _finalize_answer(
                    (final_response.content or "").strip(),
                    steps,
                    query,
                ) or "Reached early-stop without textual answer."
                elapsed = time.perf_counter() - t0
                _log_perf({
                    **log_ctx,
                    "event": "run_complete",
                    "total_sec": round(elapsed, 4),
                    "rounds": iteration + 2,
                    "tool_calls_count": sum(1 for s in steps if s.get("role") == "tool"),
                    "stop_reason": "early_stop_evidence",
                })
                return answer, steps, elapsed

    last_content = (messages[-1].content if messages and hasattr(messages[-1], "content") else "") or ""
    elapsed = time.perf_counter() - t0
    _log_perf({
        **log_ctx,
        "event": "run_complete",
        "total_sec": round(elapsed, 4),
        "rounds": max_iterations,
        "tool_calls_count": sum(1 for s in steps if s.get("role") == "tool"),
        "stop_reason": "max_iterations",
    })
    return _finalize_answer(last_content, steps, query) or "Reached max iterations.", steps, elapsed


def run_agent(
    query: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    previous_messages: Optional[List[Tuple[str, str]]] = None,
    session_id: Optional[str] = None,
    baseline: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    strategy: Optional[str] = None,
    perf_extra: Optional[dict] = None,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    stream_tokens: bool = False,
) -> tuple[str, List[dict[str, Any]], float]:
    """
    Run the agent with Ollama. Pass previous_messages for multi-turn (session) chat.
    baseline: 'no_rag' (Translink only), 'no_tools' (RAG only), or None (full).
    tools: override tool list; ignored if baseline is set.
    :return: (final_answer_text, list of step dicts for debugging, total_elapsed_seconds)
    """
    opts = resolve_strategy_options(strategy, query, baseline, max_iterations)
    resolved_baseline = opts["baseline"]
    resolved_iterations = int(opts["max_iterations"])
    use_tools = get_tools_for_baseline(resolved_baseline) if resolved_baseline else (tools if tools is not None else TOOLS)
    local_perf = dict(perf_extra or {})
    local_perf.setdefault("experiment_strategy", opts["strategy"])

    if opts["two_stage"]:
        return _run_two_stage(
            backend="ollama",
            query=query,
            model=model,
            system_prompt=system_prompt,
            max_iterations=resolved_iterations,
            previous_messages=previous_messages,
            session_id=session_id,
            tools_list=use_tools,
            perf_extra=local_perf,
            on_event=on_event,
            stream_tokens=stream_tokens,
        )

    model_name = model or AGENT_MODEL
    if _ollama_model_uses_pipeline(model_name):
        rid = str(uuid.uuid4())[:8]
        log_ctx = {"run_id": rid, "llm_backend": "ollama", "pipeline": True, **local_perf}
        if session_id:
            log_ctx["session_id"] = session_id
        return _run_ollama_pipeline(
            query,
            model_name,
            system_prompt,
            previous_messages,
            use_tools,
            log_ctx,
            on_event,
            stream_tokens,
        )

    llm = _build_llm(model=model_name, tools=use_tools)
    return _run_agent_loop(
        llm, query, system_prompt, resolved_iterations, previous_messages,
        session_id=session_id, llm_backend="ollama", tools=use_tools,
        perf_extra=local_perf, early_stop=bool(opts["early_stop"]),
        on_event=on_event,
        stream_tokens=stream_tokens,
    )


def run_agent_claude(
    query: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    previous_messages: Optional[List[Tuple[str, str]]] = None,
    session_id: Optional[str] = None,
    baseline: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    strategy: Optional[str] = None,
    perf_extra: Optional[dict] = None,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    stream_tokens: bool = False,
) -> tuple[str, List[dict[str, Any]], float]:
    """
    Run the agent with Claude (Anthropic). Pass previous_messages for multi-turn (session) chat.
    baseline: 'no_rag' (Translink only), 'no_tools' (RAG only), or None (full).
    tools: override tool list; ignored if baseline is set.
    :return: (final_answer_text, list of step dicts for debugging, total_elapsed_seconds)
    """
    if not (ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")):
        raise ValueError("ANTHROPIC_API_KEY is not set")
    opts = resolve_strategy_options(strategy, query, baseline, max_iterations)
    resolved_baseline = opts["baseline"]
    resolved_iterations = int(opts["max_iterations"])
    use_tools = get_tools_for_baseline(resolved_baseline) if resolved_baseline else (tools if tools is not None else TOOLS)
    local_perf = dict(perf_extra or {})
    local_perf.setdefault("experiment_strategy", opts["strategy"])

    if opts["two_stage"]:
        return _run_two_stage(
            backend="claude",
            query=query,
            model=model,
            system_prompt=system_prompt,
            max_iterations=resolved_iterations,
            previous_messages=previous_messages,
            session_id=session_id,
            tools_list=use_tools,
            perf_extra=local_perf,
            on_event=on_event,
            stream_tokens=stream_tokens,
        )

    llm = _build_llm_claude(model=model, tools=use_tools)
    return _run_agent_loop(
        llm, query, system_prompt, resolved_iterations, previous_messages,
        session_id=session_id, llm_backend="claude", tools=use_tools,
        perf_extra=local_perf, early_stop=bool(opts["early_stop"]),
        on_event=on_event,
        stream_tokens=stream_tokens,
    )


def _run_two_stage(
    backend: str,
    query: str,
    model: Optional[str],
    system_prompt: Optional[str],
    max_iterations: int,
    previous_messages: Optional[List[Tuple[str, str]]],
    session_id: Optional[str],
    tools_list: List[Any],
    perf_extra: Optional[dict],
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    stream_tokens: bool = False,
) -> tuple[str, List[dict[str, Any]], float]:
    """
    Two-stage strategy (S3):
      Stage A: planner-only LLM call (no tools)
      Stage B: regular tool loop with bounded rounds
    """
    planner_system = (
        "You are planning only. Produce a concise numbered plan for solving the user's request. "
        "Do not call any tools."
    )
    planner_messages: List[BaseMessage] = [SystemMessage(content=planner_system)]
    if previous_messages:
        for role, content in previous_messages:
            planner_messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    planner_messages.append(HumanMessage(content=query))

    llm_planner = _build_llm(model=model, tools=[]) if backend == "ollama" else _build_llm_claude(model=model, tools=[])
    _emit(on_event, {"type": "status", "phase": "planner", "message": "Two-stage: planner (no tools)…"})
    planner_resp, dt_plan = _llm_invoke_round(
        llm_planner,
        planner_messages,
        on_event=on_event,
        stream_tokens=stream_tokens,
        round_index=0,
    )
    _log_perf({
        "run_id": str(uuid.uuid4())[:8],
        "llm_backend": backend,
        **{k: v for k, v in (perf_extra or {}).items() if v is not None},
        "event": "planner_stage",
        "duration_sec": round(dt_plan, 4),
        "stage": "s3_plan",
    })

    plan_text = _plain_text_from_message_content(getattr(planner_resp, "content", None))
    plan_step = {
        "role": "assistant",
        "content": plan_text,
        "duration_sec": round(dt_plan, 4),
        "stage": "s3_plan",
        "round": 0,
    }
    _emit(on_event, {"type": "step", "step": plan_step})
    stage2_prompt = (system_prompt or SYSTEM_PROMPT) + "\n\nPlanning draft:\n" + plan_text
    llm_exec = _build_llm(model=model, tools=tools_list) if backend == "ollama" else _build_llm_claude(model=model, tools=tools_list)
    answer, steps, elapsed = _run_agent_loop(
        llm_exec,
        query,
        stage2_prompt,
        max_iterations=max_iterations,
        previous_messages=previous_messages,
        session_id=session_id,
        llm_backend=backend,
        tools=tools_list,
        perf_extra={**(perf_extra or {}), "stage": "s3_execute"},
        early_stop=False,
        on_event=on_event,
        stream_tokens=stream_tokens,
    )
    steps = [plan_step] + steps
    return answer, steps, elapsed + dt_plan
