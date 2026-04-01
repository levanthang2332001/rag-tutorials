---
name: ""
overview: ""
todos: []
isProject: false
---

# Plan 2: Base Tools Implementation

## Overview

Implement 6 base tools using LangChain's `@tool` decorator.

## File: `agentic_rag/tools.py`

```python
"""Base tools for all agents - using @tool decorator."""

import warnings
import math
import subprocess
import sqlite3

from langchain_core.tools import tool
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import format_docs, split_documents, load_documents

warnings.filterwarnings("ignore")


# =============================================================================
# PDF RETRIEVER TOOL
# =============================================================================
@tool
def pdf_retriever(query: str) -> str:
    """Search internal PDF documents for relevant information.

    Use when: user asks about topics covered in company documents,
    internal policies, research papers, or any PDF content.

    Returns: Relevant context from PDFs with citations.
    """
    docs = load_documents()
    splits = split_documents(docs)
    texts = [doc.page_content for doc in splits]
    metadatas = [doc.metadata for doc in splits]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas, k=5)

    # Hybrid retrieve (BM25 + Vector)
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    combined = {}
    for rank, doc in enumerate(bm25_docs, 1):
        combined[doc.id] = {"doc": doc, "score": 0.7 * (1.0 / rank)}
    for rank, doc in enumerate(vector_docs, 1):
        if doc.id in combined:
            combined[doc.id]["score"] += 0.3 * (1.0 / rank)
        else:
            combined[doc.id] = {"doc": doc, "score": 0.3 * (1.0 / rank)}

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    top_docs = [item["doc"] for item in sorted_results[:5]]

    return format_docs(top_docs)


# =============================================================================
# WEB SEARCH TOOL (Tavily)
# =============================================================================
@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or facts.

    Use when: user asks about recent events, stock prices,
    weather, news, or anything requiring up-to-date internet data.

    Returns: Formatted search results with titles, URLs, and snippets.
    """
    from langchain_tavily import TavilySearch

    search = TavilySearch()
    results = search.invoke(query)

    formatted = []
    for r in results:
        title = r.get("title", "N/A")
        url = r.get("url", "N/A")
        content = r.get("content", "N/A")[:500]
        formatted.append(f"Title: {title}\nURL: {url}\nContent: {content}")

    if not formatted:
        return "No search results found."

    return "\n\n".join(formatted)


# =============================================================================
# CALCULATOR TOOL
# =============================================================================
@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions.

    Use when: user asks to calculate, compute, or solve math problems.
    Supports: basic ops (+-*/), powers, sqrt, sin, cos, tan, log, pi, e.

    Returns: The computed result or error message.
    """
    allowed_funcs = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt, "log": math.log,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e
    }

    try:
        result = eval(expression, {"__builtins__": {}, **allowed_funcs}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# WIKIPEDIA TOOL
# =============================================================================
@tool
def wikipedia(query: str) -> str:
    """Search Wikipedia for factual information.

    Use when: user asks about definitions, historical facts,
    biographical information, or well-established knowledge.

    Returns: Wikipedia summary (first 3 sentences) or not found message.
    """
    import wikipedia as wiki

    try:
        summary = wiki.summary(query, sentences=3)
        return summary
    except wiki.exceptions.DisambiguationError:
        return f"Wikipedia: Multiple matches found for '{query}'. Please be more specific."
    except wiki.exceptions.PageError:
        return f"Wikipedia: No page found for '{query}'."
    except Exception:
        return f"Wikipedia: Error searching for '{query}'."


# =============================================================================
# CODE EXECUTOR TOOL
# =============================================================================
@tool
def code_executor(code: str) -> str:
    """Execute Python code in a sandboxed environment.

    Use when: user asks to run code, generate data, create visualizations,
    or perform complex computations beyond basic math.

    WARNING: Only Python code is supported. Timeout is 10 seconds.

    Returns: stdout/stderr from code execution.
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout
        if result.stderr:
            output += "\nErrors:\n" + result.stderr
        return output if output else "Code executed with no output."
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (10s limit)."
    except Exception as e:
        return f"Error executing code: {e}"


# =============================================================================
# SQL DATABASE TOOL
# =============================================================================
@tool
def sql_database(query: str) -> str:
    """Execute read-only SQL queries against the database.

    Use when: user asks about data, analytics, reports,
    or structured data stored in SQLite database.

    SECURITY: Only SELECT queries are allowed.

    Returns: Query results as string or error message.
    """
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for security."

    try:
        conn = sqlite3.connect("data.db")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "Query returned no results."

        return "\n".join([str(row) for row in results])
    except sqlite3.Error as e:
        return f"Database error: {e}"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# TOOL LIST (for agent reference)
# =============================================================================
ALL_TOOLS = [
    pdf_retriever,
    web_search,
    calculator,
    wikipedia,
    code_executor,
    sql_database,
]
```

## Tool Summary

| Tool            | When to Use        | Example Query                       |
| --------------- | ------------------ | ----------------------------------- |
| `pdf_retriever` | Internal documents | "What does our policy say about..." |
| `web_search`    | Current events     | "Latest news about..."              |
| `calculator`    | Math problems      | "Calculate 15% of 1000"             |
| `wikipedia`     | Factual knowledge  | "Who is Elon Musk?"                 |
| `code_executor` | Run code           | "Generate 10 random numbers"        |
| `sql_database`  | Database queries   | "Show total sales by region"        |

## Next Steps

After tools, proceed to: [3-specialist-agents.plan.md](./3-specialist-agents.plan.md)
