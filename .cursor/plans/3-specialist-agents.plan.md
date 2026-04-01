---
name: ""
overview: ""
todos: []
isProject: false
---

# Plan 3: Specialist Agents

## Overview

Create 6 specialized agents, each wrapping a single tool with LangChain ReAct pattern and LangSmith tracing.

## Architecture

Each agent follows:

```
Query → ReAct Agent (GPT-4o-mini) → Tool → Result
```

All agents use same pattern - only the tool differs.

## File: `agentic_rag/agents/pdf_agent.py`

```python
"""PDF Agent - Specialized agent for PDF document search."""

import traceback
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langsmith import trace

from agentic_rag.tools import pdf_retriever


def create_pdf_agent() -> AgentExecutor:
    """Create PDF agent with ReAct prompt."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [pdf_retriever], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[pdf_retriever], verbose=False)


@trace(name="pdf_agent", tags=["agent", "pdf"])
def run_pdf_agent(query: str) -> str:
    """Run PDF agent to search internal documents."""
    try:
        agent = create_pdf_agent()
        result = agent.invoke({"input": query})
        return result.get("output", "No result found.")
    except Exception as e:
        traceback.print_exc()
        return f"Error in PDF agent: {str(e)}"
```

## File: `agentic_rag/agents/web_agent.py`

```python
"""Web Agent - Specialized agent for web search."""

import traceback
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langsmith import trace

from agentic_rag.tools import web_search


def create_web_agent() -> AgentExecutor:
    """Create web search agent with ReAct prompt."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [web_search], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[web_search], verbose=False)


@trace(name="web_agent", tags=["agent", "web"])
def run_web_agent(query: str) -> str:
    """Run web agent to search the internet."""
    try:
        agent = create_web_agent()
        result = agent.invoke({"input": query})
        return result.get("output", "No results found.")
    except Exception as e:
        traceback.print_exc()
        return f"Error in web agent: {str(e)}"
```

## File: `agentic_rag/agents/calculator_agent.py`

```python
"""Calculator Agent - Specialized agent for math operations."""

import traceback
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langsmith import trace

from agentic_rag.tools import calculator


def create_calculator_agent() -> AgentExecutor:
    """Create calculator agent with ReAct prompt."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [calculator], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[calculator], verbose=False)


@trace(name="calculator_agent", tags=["agent", "calculator"])
def run_calculator_agent(query: str) -> str:
    """Run calculator agent for math queries."""
    try:
        agent = create_calculator_agent()
        result = agent.invoke({"input": query})
        return result.get("output", "Could not calculate.")
    except Exception as e:
        traceback.print_exc()
        return f"Error in calculator agent: {str(e)}"
```

## File: `agentic_rag/agents/wikipedia_agent.py`

```python
"""Wikipedia Agent - Specialized agent for Wikipedia search."""

import traceback
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langsmith import trace

from agentic_rag.tools import wikipedia


def create_wikipedia_agent() -> AgentExecutor:
    """Create Wikipedia agent with ReAct prompt."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [wikipedia], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[wikipedia], verbose=False)


@trace(name="wikipedia_agent", tags=["agent", "wikipedia"])
def run_wikipedia_agent(query: str) -> str:
    """Run Wikipedia agent for factual queries."""
    try:
        agent = create_wikipedia_agent()
        result = agent.invoke({"input": query})
        return result.get("output", "No Wikipedia entry found.")
    except Exception as e:
        traceback.print_exc()
        return f"Error in Wikipedia agent: {str(e)}"
```

## File: `agentic_rag/agents/code_agent.py`

```python
"""Code Agent - Specialized agent for code execution."""

import traceback
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langsmith import trace

from agentic_rag.tools import code_executor


def create_code_agent() -> AgentExecutor:
    """Create code execution agent with ReAct prompt."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [code_executor], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[code_executor], verbose=False)


@trace(name="code_agent", tags=["agent", "code"])
def run_code_agent(query: str) -> str:
    """Run code agent for code execution requests."""
    try:
        agent = create_code_agent()
        result = agent.invoke({"input": query})
        return result.get("output", "Code executed with no output.")
    except Exception as e:
        traceback.print_exc()
        return f"Error in code agent: {str(e)}"
```

## File: `agentic_rag/agents/sql_agent.py`

```python
"""SQL Agent - Specialized agent for database queries."""

import traceback
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langsmith import trace

from agentic_rag.tools import sql_database


def create_sql_agent() -> AgentExecutor:
    """Create SQL agent with ReAct prompt."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [sql_database], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[sql_database], verbose=False)


@trace(name="sql_agent", tags=["agent", "sql"])
def run_sql_agent(query: str) -> str:
    """Run SQL agent for database queries."""
    try:
        agent = create_sql_agent()
        result = agent.invoke({"input": query})
        return result.get("output", "Query returned no results.")
    except Exception as e:
        traceback.print_exc()
        return f"Error in SQL agent: {str(e)}"
```

## File: `agentic_rag/agents/__init__.py`

```python
"""Agents package - exports all specialist agents."""

from .pdf_agent import run_pdf_agent
from .web_agent import run_web_agent
from .calculator_agent import run_calculator_agent
from .wikipedia_agent import run_wikipedia_agent
from .code_agent import run_code_agent
from .sql_agent import run_sql_agent

__all__ = [
    "run_pdf_agent",
    "run_web_agent",
    "run_calculator_agent",
    "run_wikipedia_agent",
    "run_code_agent",
    "run_sql_agent",
]
```

## Agent Registry

| Agent                  | Tool            | Model       | Purpose            |
| ---------------------- | --------------- | ----------- | ------------------ |
| `run_pdf_agent`        | `pdf_retriever` | gpt-4o-mini | Internal documents |
| `run_web_agent`        | `web_search`    | gpt-4o-mini | Internet search    |
| `run_calculator_agent` | `calculator`    | gpt-4o-mini | Math               |
| `run_wikipedia_agent`  | `wikipedia`     | gpt-4o-mini | Facts              |
| `run_code_agent`       | `code_executor` | gpt-4o-mini | Code execution     |
| `run_sql_agent`        | `sql_database`  | gpt-4o-mini | Database           |

## Next Steps

After specialist agents, proceed to: [4-langgraph-workflow.plan.md](./4-langgraph-workflow.plan.md)
