"""
Agent execution: LangChain ChatOllama / ChatAnthropic + tool-calling loop.
Uses existing RAG /search via tools (no change to RAG code).
"""
import os
import time
from typing import Any, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from .tools import search_travel_knowledge

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
AGENT_MODEL = os.getenv("AGENT_LLM_MODEL", "llama3.1")
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))

# Claude agent (optional: requires langchain-anthropic)
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "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")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
# All tools exposed to the agent (extend here for new tools)
TOOLS = [search_travel_knowledge]

SYSTEM_PROMPT = """You are a travel planning assistant. You must plan explicitly step-by-step for each request.

**Planning (required):**
1. First, state your plan in clear steps, e.g.: "计划：1) 查询交通方式 2) 查询主要景点 3) 查询实用贴士与费用"
2. Then call search_travel_knowledge once per step (or as needed) with the appropriate query and optional city.
3. Finally, synthesize a reply with clear sections, e.g. "## 交通 / ## 景点 / ## 贴士", using the tool results and citing sources like [1][2].

**Rules:**
- Use the tool to look up concrete facts (transport, attractions, tips, costs). Prefer the knowledge base; if it has no relevant info, say so.
- Answer in both Chinese and English when the user asks in Chinese; otherwise be clear and concise.
- When citing, refer to snippets returned by the tool (e.g. "[1] Brisbane#0 ...")."""


def _build_llm(model: Optional[str] = None, base_url: Optional[str] = None) -> ChatOllama:
    return ChatOllama(
        model=model or AGENT_MODEL,
        base_url=base_url or OLLAMA_BASE,
        temperature=0.2,
        num_ctx=4096,
    ).bind_tools(TOOLS)


def _build_llm_claude(model: Optional[str] = None):
    """Build Claude LLM with tools. Requires langchain-anthropic and ANTHROPIC_API_KEY."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=model or ANTHROPIC_MODEL,
        api_key=ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.2,
        max_tokens=1024,
    ).bind_tools(TOOLS)


def _run_agent_loop(
    llm,
    query: str,
    system_prompt: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    previous_messages: Optional[List[Tuple[str, str]]] = None,
) -> tuple[str, List[dict[str, Any]], float]:
    """Shared tool-calling loop. If previous_messages is set, prepend (role, content) history for multi-turn."""
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

    for _ in range(max_iterations):
        response: AIMessage = llm.invoke(messages)
        steps.append({"role": "assistant", "content": getattr(response, "content", "") or ""})

        if not getattr(response, "tool_calls", None):
            answer = (response.content or "").strip()
            elapsed = time.perf_counter() - t0
            return answer, steps, elapsed

        messages.append(response)
        for tc in response.tool_calls:
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None) or ""
            args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else None) or {}
            tool_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None) or ""
            tool_fn = next((t for t in TOOLS if t.name == name), None)
            if not tool_fn:
                result = f"Unknown tool: {name}"
            else:
                try:
                    result = tool_fn.invoke(args)
                except Exception as e:
                    result = f"Tool error: {e!s}"
            steps.append({"role": "tool", "name": name, "args": args, "result_preview": str(result)[:200]})
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id or name))

    last_content = (messages[-1].content if messages and hasattr(messages[-1], "content") else "") or ""
    elapsed = time.perf_counter() - t0
    return (last_content or "Reached max iterations.").strip(), steps, elapsed


def run_agent(
    query: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    previous_messages: Optional[List[Tuple[str, str]]] = None,
) -> tuple[str, List[dict[str, Any]], float]:
    """
    Run the agent with Ollama. Pass previous_messages for multi-turn (session) chat.
    :return: (final_answer_text, list of step dicts for debugging, total_elapsed_seconds)
    """
    llm = _build_llm(model=model)
    return _run_agent_loop(llm, query, system_prompt, max_iterations, previous_messages)


def run_agent_claude(
    query: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    previous_messages: Optional[List[Tuple[str, str]]] = None,
) -> tuple[str, List[dict[str, Any]], float]:
    """
    Run the agent with Claude (Anthropic). Pass previous_messages for multi-turn (session) chat.
    :return: (final_answer_text, list of step dicts for debugging, total_elapsed_seconds)
    """
    if not (ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")):
        raise ValueError("ANTHROPIC_API_KEY is not set")
    llm = _build_llm_claude(model=model)
    return _run_agent_loop(llm, query, system_prompt, max_iterations, previous_messages)
