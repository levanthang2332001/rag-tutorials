# Plan 7: API & CLI

## Overview
Create FastAPI server and CLI chatbot to expose the multi-agent system.

## File: `agentic_api.py`

```python
"""Agentic RAG API - FastAPI server for multi-agent RAG system."""

import warnings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agentic_rag.chain import create_agentic_chain

# Suppress warnings
warnings.filterwarnings("ignore")
load_dotenv()

# Initialize chain
chain = create_agentic_chain()

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Agentic RAG API",
    description="Multi-agent RAG with LangGraph + LangSmith",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    session_id: str
    iterations: int
    agents_used: list[str]


class HistoryMessage(BaseModel):
    """Single message in history."""
    type: str
    content: str


class HistoryResponse(BaseModel):
    """Response model for history endpoint."""
    session_id: str
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    langgraph: bool
    langsmith: bool
    agents: list[str]


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        langgraph=True,
        langsmith=True,
        agents=["pdf_agent", "web_agent", "calculator_agent",
                "wikipedia_agent", "code_agent", "sql_agent"],
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return root()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint - invoke multi-agent RAG."""
    try:
        result = chain.invoke(request.question, request.session_id)
        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            iterations=result["iterations"],
            agents_used=result["agents_used"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    """Get conversation history for a session."""
    messages = chain.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[
            HistoryMessage(type=msg.type, content=msg.content)
            for msg in messages
        ],
    )


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Clear conversation history for a session."""
    chain.clear_history(session_id)
    return {"message": f"History cleared for session {session_id}"}


@app.get("/sessions")
def list_sessions():
    """List all active sessions."""
    sessions = chain.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting Agentic RAG API on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

## File: `agentic_chatbot.py`

```python
"""Agentic RAG CLI Chatbot - Interactive command-line interface."""

import sys
from agentic_rag.chain import create_agentic_chain


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 60)
    print("  Agentic RAG Chatbot (LangGraph + LangSmith)")
    print("=" * 60)
    print("\nAvailable Agents:")
    print("  - PDF Agent: Search internal documents")
    print("  - Web Agent: Search the internet")
    print("  - Calculator Agent: Math calculations")
    print("  - Wikipedia Agent: Factual knowledge")
    print("  - Code Agent: Python code execution")
    print("  - SQL Agent: Database queries")
    print("\nCommands:")
    print("  quit      - Exit the chatbot")
    print("  history   - Show conversation history")
    print("  clear     - Clear conversation history")
    print("  sessions  - List active sessions")
    print("=" * 60 + "\n")


def print_response(result: dict):
    """Pretty print agent response."""
    print(f"\n[Used {len(result['agents_used'])} agents in {result['iterations']} iteration(s)]")
    print(f"Agents: {', '.join(result['agents_used'])}")
    print("-" * 40)
    print(f"Agent: {result['answer']}")


def main():
    """Main CLI loop."""
    chain = create_agentic_chain()
    session_id = "cli_session"

    print_welcome()

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle commands
            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break

            elif user_input.lower() == "history":
                messages = chain.get_history(session_id)
                print("\n--- Conversation History ---")
                for msg in messages:
                    prefix = "You" if msg.type == "human" else "Agent"
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    print(f"  {prefix}: {content}")
                if not messages:
                    print("  (empty)")
                continue

            elif user_input.lower() == "clear":
                chain.clear_history(session_id)
                print("History cleared.")
                continue

            elif user_input.lower() == "sessions":
                sessions = chain.list_sessions()
                print(f"\nActive sessions: {sessions}")
                continue

            # Regular query
            result = chain.invoke(user_input, session_id)
            print_response(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
```

## API Usage Examples

### curl

```bash
# Chat
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "session_id": "test"}'

# Get history
curl http://localhost:8001/history/test

# Clear history
curl -X DELETE http://localhost:8001/history/test

# Health check
curl http://localhost:8001/health
```

### Python client

```python
import requests

# Chat
response = requests.post(
    "http://localhost:8001/chat",
    json={"question": "What is RAG?", "session_id": "test"}
)
print(response.json())

# Get history
response = requests.get("http://localhost:8001/history/test")
print(response.json())
```

## Running the Services

```bash
# Terminal 1: Run API
python agentic_api.py

# Terminal 2: Run CLI
python agentic_chatbot.py

# Or run API in background
uvicorn agentic_api:app --host 0.0.0.0 --port 8001
```

## Next Steps

This completes the implementation! See [index.plan.md](./index.plan.md) for the master plan overview.
