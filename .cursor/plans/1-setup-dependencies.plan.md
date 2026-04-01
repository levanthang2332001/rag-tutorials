---
name: ""
overview: ""
todos: []
isProject: false
---

# Plan 1: Setup & Dependencies

## Overview

Setup project structure và add dependencies cho Agentic RAG system.

## Directory Structure

```
d:/Tool-Airdrop/RAG/
├── agentic_rag/              # NEW: main package
│   └── __init__.py
├── agentic_rag/agents/       # NEW: specialist agents
│   └── __init__.py
├── agentic_chatbot.py        # NEW: CLI chatbot
├── agentic_api.py           # NEW: FastAPI
└── requirements.txt          # MODIFY: add dependencies
```

## Steps

### 1. Create Directory Structure

```bash
mkdir -p agentic_rag/agents
touch agentic_rag/__init__.py
touch agentic_rag/agents/__init__.py
```

### 2. Modify requirements.txt

**File**: `requirements.txt`

**ADD** these lines:

```
langchain-tavily
wikipedia
langgraph
langsmith
```

### 3. LangSmith Setup

**File**: `agentic_rag/__init__.py`

```python
"""Agentic RAG package - LangSmith initialization."""

import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "agentic-rag")

# Verify setup
if os.environ.get("LANGSMITH_API_KEY"):
    print(f"LangSmith tracing enabled for project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    print("Warning: LANGSMITH_API_KEY not set. Tracing disabled.")
```

### 4. Environment Variables

**File**: `.env`

```bash
# LangSmith (get from https://smith.langchain.com/)
LANGSMITH_API_KEY=ls_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag
```

## Dependencies Explained

| Package            | Purpose                            |
| ------------------ | ---------------------------------- |
| `langgraph`        | Multi-agent workflow orchestration |
| `langsmith`        | Observability, tracing, evaluation |
| `langchain-tavily` | Web search (Tavily API)            |
| `wikipedia`        | Wikipedia search                   |

## Next Steps

After setup, proceed to: [2-tools.plan.md](./2-tools.plan.md)
