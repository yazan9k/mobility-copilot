"""FastAPI surface for the copilot.

`/chat` returns the reply *and* the tool-call trajectory. The trajectory is
part of the contract because trajectory-level evaluation depends on it, and
because a reviewer looking at the API should be able to see what the agent
actually did rather than only what it said.

Run:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import core, llm
from config import AGENT_MODEL, AGENT_VERSION

app = FastAPI(
    title="Global Mobility Copilot",
    description=(
        "Internal relocation assistant for the fictional Meridian Systems. "
        "All policy, visa, and employee data is synthetic."
    ),
    version="1.0.0",
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="The employee's question.")
    history: list[Message] = Field(
        default_factory=list, description="Prior turns, oldest first."
    )
    version: str | None = Field(
        default=None, description="Agent version override, e.g. 'v1'. Used by evals."
    )


class ToolCallOut(BaseModel):
    turn: int
    name: str
    arguments: dict[str, Any]
    result_preview: str


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallOut]
    tool_names: list[str]
    retrieved_docs: list[str]
    turns: int
    hit_turn_limit: bool
    latency_ms: int
    usage: dict[str, int]
    version: str


@app.get("/health")
def health() -> dict[str, Any]:
    ok, detail = llm.health_check()
    return {
        "ok": ok,
        "detail": detail,
        "agent_model": AGENT_MODEL,
        "agent_version": AGENT_VERSION,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = core.run(
        message=request.message,
        history=[m.model_dump() for m in request.history],
        version=request.version,
    )
    return ChatResponse(
        reply=result.reply,
        tool_calls=[
            ToolCallOut(
                turn=t.turn,
                name=t.name,
                arguments=t.arguments,
                result_preview=t.result[:400],
            )
            for t in result.tool_calls
        ],
        tool_names=result.tool_names(),
        retrieved_docs=result.retrieved_docs,
        turns=result.turns,
        hit_turn_limit=result.hit_turn_limit,
        latency_ms=result.latency_ms,
        usage={
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        version=result.version,
    )
