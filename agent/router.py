"""
Agent HTTP API. Mounted under /agent on the main FastAPI app (OCP: no change to RAG routes).
"""
import json
import queue
import threading
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chain import run_agent, run_agent_claude
from .session import append_turn, get_history, get_or_create_session

router = APIRouter(prefix="", tags=["agent"])


def _perf_extra_for_experiment(
    baseline_raw: Optional[str],
    max_llm_rounds: int,
    strategy: Optional[str] = None,
    client_run_id: Optional[str] = None,
) -> dict:
    """Fields merged into agent_perf.log for ablation / experiment correlation."""
    eb = baseline_raw if baseline_raw else "full"
    if baseline_raw == "single_turn":
        profile = "full_tools_single_llm_round"
    elif baseline_raw == "no_rag":
        profile = "translink_only"
    elif baseline_raw == "no_tools":
        profile = "rag_only"
    else:
        profile = "full"
    d: dict = {
        "experiment_baseline": eb,
        "experiment_strategy": strategy or "s0",
        "tools_profile": profile,
        "max_llm_rounds": max_llm_rounds,
    }
    if client_run_id:
        d["client_run_id"] = client_run_id
    return d


class AgentPlanIn(BaseModel):
    query: str = Field(..., description="User question or travel planning request")
    model: Optional[str] = Field(None, description="Ollama/Claude model name (default from env)")
    system_prompt: Optional[str] = Field(None, description="Override system prompt")
    max_iterations: int = Field(8, ge=1, le=20, description="Max agent tool-call rounds")
    baseline: Optional[str] = Field(
        None,
        description="Experiment baseline: no_rag (Translink only), no_tools (RAG only), single_turn (max 1 round), or null for full",
    )
    strategy: Optional[str] = Field(
        None,
        description="Optional strategy mode: s0 (default), s1 (tool-gating), s2 (early-stop), s3 (two-stage planner)",
    )
    client_run_id: Optional[str] = Field(
        None,
        description="Optional id from experiment harness; copied into agent_perf.log for joining runs",
    )


class AgentPlanOut(BaseModel):
    answer: str
    steps: List[Any] = Field(default_factory=list, description="Agent steps (assistant + tool calls)")
    took_seconds: float


class ChatIn(BaseModel):
    message: str = Field(..., description="Current user message in this turn")
    session_id: Optional[str] = Field(None, description="Existing session id for multi-turn; omit to start new")
    model: Optional[str] = Field(None, description="Ollama/Claude model name (default from env)")
    max_iterations: int = Field(8, ge=1, le=20, description="Max agent tool-call rounds per turn")
    baseline: Optional[str] = Field(
        None,
        description="Experiment baseline: no_rag, no_tools, single_turn, or null for full",
    )
    strategy: Optional[str] = Field(
        None,
        description="Optional strategy mode: s0 (default), s1 (tool-gating), s2 (early-stop), s3 (two-stage planner)",
    )
    client_run_id: Optional[str] = Field(
        None,
        description="Optional id from experiment harness; copied into agent_perf.log",
    )


class ChatOut(BaseModel):
    session_id: str = Field(..., description="Session id; use in next request for continuous dialogue")
    answer: str
    steps: List[Any] = Field(default_factory=list, description="Agent steps for this turn")
    took_seconds: float


def _resolve_baseline_and_iters(baseline: Optional[str], max_iterations: int) -> tuple[Optional[str], int]:
    """For single_turn baseline, use full tools but force 1 round; else pass baseline and user max_iterations."""
    if baseline == "single_turn":
        return None, 1  # full tools, 1 round
    return baseline, max_iterations


@router.post("/plan", response_model=AgentPlanOut)
def agent_plan(body: AgentPlanIn) -> AgentPlanOut:
    """
    Run the travel planning agent: LLM can call tools (e.g. search_travel_knowledge) in a loop.
    Uses the same RAG /search under the hood via tools, without modifying the original RAG code.
    Optional baseline: no_rag, no_tools, single_turn for experiments.
    """
    try:
        base, iters = _resolve_baseline_and_iters(body.baseline, body.max_iterations)
        perf_extra = _perf_extra_for_experiment(body.baseline, iters, body.strategy, body.client_run_id)
        answer, steps, took = run_agent(
            query=body.query,
            model=body.model,
            system_prompt=body.system_prompt,
            max_iterations=iters,
            session_id=None,
            baseline=base,
            strategy=body.strategy,
            perf_extra=perf_extra,
        )
        return AgentPlanOut(answer=answer, steps=steps, took_seconds=took)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e!s}") from e


@router.post("/plan_claude", response_model=AgentPlanOut)
def agent_plan_claude(body: AgentPlanIn) -> AgentPlanOut:
    """
    Same as /agent/plan but uses Claude (Anthropic) as the LLM. Requires ANTHROPIC_API_KEY.
    Optional baseline: no_rag, no_tools, single_turn for experiments.
    """
    try:
        base, iters = _resolve_baseline_and_iters(body.baseline, body.max_iterations)
        perf_extra = _perf_extra_for_experiment(body.baseline, iters, body.strategy, body.client_run_id)
        answer, steps, took = run_agent_claude(
            query=body.query,
            model=body.model,
            system_prompt=body.system_prompt,
            max_iterations=iters,
            session_id=None,
            baseline=base,
            strategy=body.strategy,
            perf_extra=perf_extra,
        )
        return AgentPlanOut(answer=answer, steps=steps, took_seconds=took)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude agent run failed: {e!s}") from e


def _history_to_previous_messages(history: List[dict]) -> List[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in history if m.get("role") and m.get("content")]


@router.post("/chat", response_model=ChatOut)
def agent_chat(body: ChatIn) -> ChatOut:
    """
    Multi-turn chat with session: pass session_id to continue the same conversation.
    Optional baseline: no_rag, no_tools, single_turn for experiments.
    """
    session_id = get_or_create_session(body.session_id)
    history = get_history(session_id)
    previous_messages = _history_to_previous_messages(history)
    base, iters = _resolve_baseline_and_iters(body.baseline, body.max_iterations)
    perf_extra = _perf_extra_for_experiment(body.baseline, iters, body.strategy, body.client_run_id)
    try:
        answer, steps, took = run_agent(
            query=body.message,
            model=body.model,
            max_iterations=iters,
            previous_messages=previous_messages if previous_messages else None,
            session_id=session_id,
            baseline=base,
            strategy=body.strategy,
            perf_extra=perf_extra,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent chat failed: {e!s}") from e
    append_turn(session_id, body.message, answer)
    return ChatOut(session_id=session_id, answer=answer, steps=steps, took_seconds=took)


_STREAM_SENTINEL = object()


def _ndjson_stream_chat(
    *,
    session_id: str,
    body: ChatIn,
    history: List[dict],
    runner,
) -> StreamingResponse:
    """runner(on_event) must call run_agent or run_agent_claude with on_event and return (answer, steps, took)."""
    previous_messages = _history_to_previous_messages(history)
    base, iters = _resolve_baseline_and_iters(body.baseline, body.max_iterations)
    perf_extra = _perf_extra_for_experiment(body.baseline, iters, body.strategy, body.client_run_id)
    q: queue.Queue[Any] = queue.Queue()

    def worker() -> None:
        try:
            def on_event(ev: dict[str, Any]) -> None:
                q.put(ev)

            answer, steps, took = runner(
                on_event=on_event,
                previous_messages=previous_messages if previous_messages else None,
                iters=iters,
                base=base,
                perf_extra=perf_extra,
            )
            append_turn(session_id, body.message, answer)
            q.put(
                {
                    "type": "complete",
                    "session_id": session_id,
                    "answer": answer,
                    "steps": steps,
                    "took_seconds": took,
                }
            )
        except Exception as e:
            q.put({"type": "error", "detail": str(e)})
        finally:
            q.put(_STREAM_SENTINEL)

    def generate():
        yield json.dumps({"type": "meta", "session_id": session_id}, ensure_ascii=False) + "\n"
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is _STREAM_SENTINEL:
                break
            yield json.dumps(item, ensure_ascii=False, default=str) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/chat/stream")
def agent_chat_stream(body: ChatIn) -> StreamingResponse:
    """Same as /chat but streams NDJSON events (status, step, complete) for live CLI progress."""
    session_id = get_or_create_session(body.session_id)
    history = get_history(session_id)

    def runner(*, on_event, previous_messages, iters, base, perf_extra):
        return run_agent(
            query=body.message,
            model=body.model,
            max_iterations=iters,
            previous_messages=previous_messages,
            session_id=session_id,
            baseline=base,
            strategy=body.strategy,
            perf_extra=perf_extra,
            on_event=on_event,
            stream_tokens=True,
        )

    return _ndjson_stream_chat(session_id=session_id, body=body, history=history, runner=runner)


@router.post("/chat_claude/stream")
def agent_chat_claude_stream(body: ChatIn) -> StreamingResponse:
    session_id = get_or_create_session(body.session_id)
    history = get_history(session_id)

    def runner(*, on_event, previous_messages, iters, base, perf_extra):
        return run_agent_claude(
            query=body.message,
            model=body.model,
            max_iterations=iters,
            previous_messages=previous_messages,
            session_id=session_id,
            baseline=base,
            strategy=body.strategy,
            perf_extra=perf_extra,
            on_event=on_event,
            stream_tokens=True,
        )

    return _ndjson_stream_chat(session_id=session_id, body=body, history=history, runner=runner)


@router.post("/chat_claude", response_model=ChatOut)
def agent_chat_claude(body: ChatIn) -> ChatOut:
    """
    Multi-turn chat with session. Uses Claude. Optional baseline: no_rag, no_tools, single_turn.
    """
    session_id = get_or_create_session(body.session_id)
    history = get_history(session_id)
    previous_messages = _history_to_previous_messages(history)
    base, iters = _resolve_baseline_and_iters(body.baseline, body.max_iterations)
    perf_extra = _perf_extra_for_experiment(body.baseline, iters, body.strategy, body.client_run_id)
    try:
        answer, steps, took = run_agent_claude(
            query=body.message,
            model=body.model,
            max_iterations=iters,
            previous_messages=previous_messages if previous_messages else None,
            session_id=session_id,
            baseline=base,
            strategy=body.strategy,
            perf_extra=perf_extra,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude agent chat failed: {e!s}") from e
    append_turn(session_id, body.message, answer)
    return ChatOut(session_id=session_id, answer=answer, steps=steps, took_seconds=took)
