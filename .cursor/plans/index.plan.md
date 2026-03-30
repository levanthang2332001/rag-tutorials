---
name: Build Agentic RAG System
overview: "Build Multi-Agent RAG with LangGraph orchestration and LangSmith observability. Agents: PDF, Web, Calculator, Wikipedia, Code, SQL coordinated by Orchestrator with graph-based workflow."
todos:
  - id: setup
    content: "Setup: Create agentic_rag/ directory, add langchain-tavily, wikipedia to requirements.txt"
    status: pending
  - id: tools
    content: "Implement base tools: pdf_retriever, web_search, calculator, wikipedia, code_executor, sql_db"
    status: pending
  - id: langgraph
    content: "Build LangGraph workflow: StateGraph with Orchestrator + 6 specialist agents + Synthesizer"
    status: pending
  - id: langsmith
    content: Setup LangSmith tracing, add @trace decorators, configure eval datasets
    status: pending
  - id: memory
    content: Implement conversation memory using ChatMessageHistory
    status: pending
  - id: api-agentic
    content: Create agentic_api.py with /agentic/chat endpoint
    status: pending
  - id: cli-agentic
    content: Create agentic_chatbot.py CLI
    status: pending
  - id: test
    content: Test end-to-end with sample queries
    status: pending
isProject: false
---

# Build Agentic RAG System - Master Plan

## Overview

Build a multi-agent RAG system using:
- **LangGraph** for workflow orchestration (with 3 critical fixes)
- **LangSmith** for observability and evaluation
- **6 Specialist Agents** for different tasks
- **1 Orchestrator** for query routing

## Architecture

```mermaid
flowchart TD
    A[User Query] --> O[Orchestrator<br/>GPT-4o-mini<br/>Query Routing]

    O -->|"Send() parallel"| RA[run_agent]
    O -->|"Send() parallel"| WA[run_agent]
    O -->|"Send() parallel"| CA[run_agent]

    RA & WA & CA --> SY[Synthesizer<br/>GPT-4o<br/>Generate Answer]

    SY -->|"quality ok"| F[Final Response]
    SY -->|"missing + iter < max"| O
    SY -->|"iter >= max| F

    LS[LangSmith<br/>Tracing] -.->|traces| O & RA & WA & CA & SY

    style O fill:#ff6b6b
    style RA fill:#4ecdc4
    style WA fill:#45b7d1
    style CA fill:#96ceb4
    style SY fill:#f8b500
```

## 3 Critical Fixes in LangGraph

| Fix | Before | After |
|-----|--------|-------|
| **Fix 1** | Sequential agent execution (slow) | `Send()` API - parallel fan-out |
| **Fix 2** | Lambda in loop (late binding bug) | Dict comprehension + Send direct |
| **Fix 3** | Always `return END` | Real quality check + loop control |

## Plans Breakdown

| #   | Plan          | File                                                               | Purpose                            |
| --- | ------------- | ------------------------------------------------------------------ | ---------------------------------- |
| 1   | Setup         | [1-setup-dependencies.plan.md](./1-setup-dependencies.plan.md)     | Directory structure, requirements  |
| 2   | Tools         | [2-tools.plan.md](./2-tools.plan.md)                               | 6 base tools with @tool decorator  |
| 3   | Agents        | [3-specialist-agents.plan.md](./3-specialist-agents.plan.md)       | 6 specialized ReAct agents         |
| 4   | Graph v2     | [4-langgraph-workflow.plan.md](./4-langgraph-workflow.plan.md)     | LangGraph + Send() parallel + fixes |
| 5   | Observability | [5-langsmith-observability.plan.md](./5-langsmith-observability-plan.md) | LangSmith tracing & evaluation     |
| 6   | Memory        | [6-memory-chain.plan.md](./6-memory-chain.plan.md)                 | Conversation memory, chain wrapper |
| 7   | API/CLI       | [7-api-cli.plan.md](./7-api-cli.plan.md)                           | FastAPI server, CLI chatbot        |

## File Structure

```
d:/Tool-Airdrop/RAG/
├── config.py                    # Existing - shared setup
├── api.py                       # Existing - FastAPI (unchanged)
│
├── agentic_rag/                 # NEW: main package
│   ├── __init__.py              # LangSmith init
│   ├── state.py                 # AgentState, AgentInvokeState
│   ├── tools.py                 # 6 @tool decorators
│   ├── graph.py                 # LangGraph workflow (3 fixes)
│   ├── memory.py                # ChatMessageHistory wrapper
│   ├── chain.py                  # Graph invocation wrapper
│   ├── eval.py                   # LangSmith evaluation
│   │
│   └── agents/                  # Specialist agents
│       ├── __init__.py
│       ├── pdf_agent.py
│       ├── web_agent.py
│       ├── calculator_agent.py
│       ├── wikipedia_agent.py
│       ├── code_agent.py
│       └── sql_agent.py
│
├── agentic_api.py               # NEW: FastAPI server (port 8001)
├── agentic_chatbot.py           # NEW: CLI chatbot
│
└── requirements.txt             # MODIFY: add langgraph, langsmith
```

## Implementation Order

```
1. Setup dependencies
       ↓
2. Implement 6 tools
       ↓
3. Create 6 specialist agents
       ↓
4. Build LangGraph workflow (with 3 fixes)
       ↓
5. Setup LangSmith tracing
       ↓
6. Add memory & chain wrapper
       ↓
7. Create API & CLI
```

## Key Technologies

| Technology | Purpose |
|------------|---------|
| LangGraph  | Multi-agent workflow orchestration |
| LangSmith  | Observability, tracing, evaluation |
| LangChain  | `@tool` decorator, ReAct agents    |
| Tavily     | Web search API                     |
| Wikipedia  | Knowledge base                     |
| FAISS      | Vector storage (existing)          |

## Quick Start

```bash
# 1. Install dependencies
pip install langgraph langsmith langchain-tavily wikipedia

# 2. Set environment
export LANGSMITH_API_KEY=ls_xxxxx
export OPENAI_API_KEY=sk-xxxxx

# 3. Run API
python agentic_api.py

# 4. Or run CLI
python agentic_chatbot.py
```

## Example Queries

```python
# Test different agent routing
"What is RAG?"                    # → wikipedia_agent
"Calculate sqrt(144) * 2"        # → calculator_agent
"What's in our company policy?"    # → pdf_agent
"Latest AI news today?"           # → web_agent
"Generate a fibonacci sequence"   # → code_agent
"Show total sales from DB"        # → sql_agent
```

## Backward Compatibility

- Existing `api.py`, `chatbot.py`, `chatbot_memory.py` remain unchanged
- New `agentic_api.py` runs on port 8001
- Users opt-in by using new endpoints

## Estimated Files

| Type | Count |
|------|-------|
| New files | 17 |
| Modified | 1 (`requirements.txt`) |
