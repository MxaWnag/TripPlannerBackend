"""
Agent HTTP API. Mounted under /agent on the main FastAPI app (OCP: no change to RAG routes).
"""
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .chain import run_agent, run_agent_claude
from .session import append_turn, get_history, get_or_create_session

router = APIRouter(prefix="", tags=["agent"])


class AgentPlanIn(BaseModel):
    query: str = Field(..., description="User question or travel planning request")
    model: Optional[str] = Field(None, description="Ollama/Claude model name (default from env)")
    system_prompt: Optional[str] = Field(None, description="Override system prompt")
    max_iterations: int = Field(8, ge=1, le=20, description="Max agent tool-call rounds")


class AgentPlanOut(BaseModel):
    answer: str
    steps: List[Any] = Field(default_factory=list, description="Agent steps (assistant + tool calls)")
    took_seconds: float


class ChatIn(BaseModel):
    message: str = Field(..., description="Current user message in this turn")
    session_id: Optional[str] = Field(None, description="Existing session id for multi-turn; omit to start new")
    model: Optional[str] = Field(None, description="Ollama/Claude model name (default from env)")
    max_iterations: int = Field(8, ge=1, le=20, description="Max agent tool-call rounds per turn")


class ChatOut(BaseModel):
    session_id: str = Field(..., description="Session id; use in next request for continuous dialogue")
    answer: str
    steps: List[Any] = Field(default_factory=list, description="Agent steps for this turn")
    took_seconds: float


@router.post("/plan", response_model=AgentPlanOut)
def agent_plan(body: AgentPlanIn) -> AgentPlanOut:
    """
    Run the travel planning agent: LLM can call tools (e.g. search_travel_knowledge) in a loop.
    Uses the same RAG /search under the hood via tools, without modifying the original RAG code.
    """
    try:
        answer, steps, took = run_agent(
            query=body.query,
            model=body.model,
            system_prompt=body.system_prompt,
            max_iterations=body.max_iterations,
        )
        return AgentPlanOut(answer=answer, steps=steps, took_seconds=took)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e!s}") from e


@router.post("/plan_claude", response_model=AgentPlanOut)
def agent_plan_claude(body: AgentPlanIn) -> AgentPlanOut:
    """
    Same as /agent/plan but uses Claude (Anthropic) as the LLM. Requires ANTHROPIC_API_KEY.
    Tools (e.g. search_travel_knowledge) unchanged; only the reasoning model is Claude.
    """
    try:
        answer, steps, took = run_agent_claude(
            query=body.query,
            model=body.model,
            system_prompt=body.system_prompt,
            max_iterations=body.max_iterations,
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
    LLM uses explicit step-by-step planning (交通 / 景点 / 贴士). Uses Ollama.
    """
    session_id = get_or_create_session(body.session_id)
    history = get_history(session_id)
    previous_messages = _history_to_previous_messages(history)
    try:
        answer, steps, took = run_agent(
            query=body.message,
            model=body.model,
            max_iterations=body.max_iterations,
            previous_messages=previous_messages if previous_messages else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent chat failed: {e!s}") from e
    append_turn(session_id, body.message, answer)
    return ChatOut(session_id=session_id, answer=answer, steps=steps, took_seconds=took)


@router.post("/chat_claude", response_model=ChatOut)
def agent_chat_claude(body: ChatIn) -> ChatOut:
    """
    Multi-turn chat with session: pass session_id to continue the same conversation.
    LLM uses explicit step-by-step planning. Uses Claude. Requires ANTHROPIC_API_KEY.
    """
    session_id = get_or_create_session(body.session_id)
    history = get_history(session_id)
    previous_messages = _history_to_previous_messages(history)
    try:
        answer, steps, took = run_agent_claude(
            query=body.message,
            model=body.model,
            max_iterations=body.max_iterations,
            previous_messages=previous_messages if previous_messages else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude agent chat failed: {e!s}") from e
    append_turn(session_id, body.message, answer)
    return ChatOut(session_id=session_id, answer=answer, steps=steps, took_seconds=took)
