"""Agentic RAG API - FastAPI server for multi-agent RAG system."""

import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentic_rag.chain import create_agentic_chain
import json

warnings.filterwarnings("ignore")
load_dotenv()

chain = create_agentic_chain()

AGENT_NAMES = [
    "pdf_agent",
    "web_agent",
    "calculator_agent",
    "wikipedia_agent",
    "code_agent",
    "sql_agent",
]

app = FastAPI(
    title="Agentic RAG API",
    description="Multi-agent RAG with LangGraph + LangSmith",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    iterations: int
    agents_used: list[str]
    agent_results: dict[str, str]


class HistoryMessage(BaseModel):
    type: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    status: str
    langgraph: bool
    langsmith: bool
    agents: list[str]


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        langgraph=True,
        langsmith=True,
        agents=AGENT_NAMES,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return root()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = chain.invoke(request.question, request.session_id)
        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            iterations=result["iterations"],
            agents_used=result["agents_used"],
            agent_results=result.get("agent_results", {}),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream progress events via SSE.

    Note: This emits events after `chain.invoke()` completes to keep the
    endpoint stable across LangGraph versions.
    """

    def event_generator():
        try:
            yield "event: start\ndata: {}\n\n".format(
                json.dumps({"session_id": request.session_id})
            )

            result = chain.invoke(request.question, request.session_id)
            agent_results: dict[str, str] = result.get("agent_results", {}) or {}

            for agent_name, agent_output in agent_results.items():
                yield "event: agent_done\ndata: {}\n\n".format(
                    json.dumps(
                        {
                            "agent_name": agent_name,
                            "output": agent_output,
                        }
                    )
                )

            yield "event: final\ndata: {}\n\n".format(
                json.dumps(
                    {
                        "answer": result.get("answer", ""),
                        "session_id": result.get("session_id", request.session_id),
                        "iterations": result.get("iterations", 0),
                        "agents_used": result.get("agents_used", []),
                    }
                )
            )
        except Exception as error:
            yield "event: error\ndata: {}\n\n".format(
                json.dumps({"error": str(error)})
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str) -> HistoryResponse:
    messages = chain.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[
            HistoryMessage(type=message.type, content=message.content)
            for message in messages
        ],
    )


@app.delete("/history/{session_id}")
def clear_history(session_id: str) -> dict:
    chain.clear_history(session_id)
    return {"message": f"History cleared for session {session_id}"}


@app.get("/sessions")
def list_sessions() -> dict:
    sessions = chain.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}


if __name__ == "__main__":
    import uvicorn

    print("Starting Agentic RAG API on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
