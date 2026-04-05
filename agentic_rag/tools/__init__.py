"""Base tools for all agents - using @tool decorator."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from agentic_rag.config import format_docs

from .cache import _BoundedTtlCache, _WIKIPEDIA_CACHE  # noqa: F401
from .calculator_eval import evaluate_calculator_expression
from .code_guard import looks_like_natural_language_request
from .pdf_retrieval import (
    _PDF_RETRIEVERS_CACHE,  # noqa: F401
    _get_pdf_retrievers,
)

warnings.filterwarnings("ignore")


@tool
def pdf_retriever(query: str) -> str:
    """Search internal PDF documents for relevant information.

    Use when: user asks about topics covered in company documents,
    internal policies, research papers, or any PDF content.

    Returns: Relevant context from PDFs with citations.
    """
    retrievers = _get_pdf_retrievers()
    vector_retriever = retrievers["vector_retriever"]
    bm25_retriever = retrievers["bm25_retriever"]

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


@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or facts.

    Use when: user asks about recent events, stock prices,
    weather, news, or anything requiring up-to-date internet data.

    Returns: Formatted search results with `Source URL:` markers.
    """
    search = TavilySearch(search_depth="fast", max_results=3)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(search.invoke, query)
            results = future.result(timeout=15)
    except TimeoutError:
        return "Source: web\nSource URL: N/A\nSource Title: Tavily timeout\nExcerpt: Search timed out."
    except Exception as e:
        return f"Source: web\nSource URL: N/A\nSource Title: Tavily error\nExcerpt: {str(e)}"

    formatted = []
    for r in results:
        title = r.get("title", "N/A")
        url = r.get("url", "N/A")
        content = r.get("content", "N/A")[:500]
        formatted.append(
            "\n".join(
                [
                    "Source: web",
                    f"Source URL: {url}",
                    f"Source Title: {title}",
                    f"Excerpt: {content}",
                ]
            )
        )

    if not formatted:
        return "No search results found."

    joined = "\n\n".join(formatted)
    return joined[:4000]


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions.

    Use when: user asks to calculate, compute, or solve math problems.
    Supports: basic ops (+-*/), powers, sqrt, sin, cos, tan, log, pi, e.

    Returns: The computed result or error message.
    """
    if os.getenv("DEBUG_TOOLS") == "1":
        print(
            "[DEBUG_TOOLS] calculator input head:",
            (expression or "")[:120].replace("\n", " "),
        )
    return evaluate_calculator_expression(expression)


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual information.

    Use when: user asks about definitions, historical facts,
    biographical information, or well-established knowledge.

    Returns: Wikipedia summary with `Source URL:` markers.
    """
    if os.getenv("DEBUG_TOOLS") == "1":
        print(
            "[DEBUG_TOOLS] wikipedia_search input head:",
            (query or "")[:200].replace("\n", " "),
        )
    import wikipedia as wiki

    normalized = (query or "").strip()
    cache_key = normalized.lower()
    cached = _WIKIPEDIA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        page = wiki.page(query, auto_suggest=True)
        summary = wiki.summary(page.title, sentences=3)
        result = "\n".join(
            [
                "Source: wikipedia",
                f"Source URL: {page.url}",
                f"Source Title: {page.title}",
                f"Excerpt: {summary}",
            ]
        )
        _WIKIPEDIA_CACHE.set(cache_key, result)
        return result
    except wiki.exceptions.DisambiguationError:
        result = f"Wikipedia: Multiple matches found for '{query}'. Please be more specific."
        _WIKIPEDIA_CACHE.set(cache_key, result)
        return result
    except wiki.exceptions.PageError:
        result = f"Wikipedia: No page found for '{query}'."
        _WIKIPEDIA_CACHE.set(cache_key, result)
        return result
    except Exception:
        result = f"Wikipedia: Error searching for '{query}'."
        _WIKIPEDIA_CACHE.set(cache_key, result)
        return result


@tool
def code_executor(code: str) -> str:
    """Execute Python code in a sandboxed environment.

    Use when: user asks to run code, generate data, create visualizations,
    or perform complex computations beyond basic math.

    WARNING: Only Python code is supported. Timeout is 10 seconds.

    Returns: stdout/stderr from code execution.
    """
    if looks_like_natural_language_request(code):
        return (
            "Input appears to be a coding request, not executable Python code. "
            "Please provide Python code to run, or ask for code generation/explanation."
        )

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
        if result.stderr:
            output += "\nErrors:\n" + result.stderr
        return output if output else "Code executed with no output."
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (10s limit)."
    except Exception as e:
        return f"Error executing code: {e}"


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


ALL_TOOLS = [
    pdf_retriever,
    web_search,
    calculator,
    wikipedia_search,
    code_executor,
    sql_database,
]
