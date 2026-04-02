"""Base tools for all agents - using @tool decorator."""

import warnings
import math
import subprocess
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os

from langchain_core.tools import tool
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from agentic_rag.config import format_docs, split_documents, load_documents
from core.config import get_config

from langchain_tavily import TavilySearch

warnings.filterwarnings("ignore")

_PDF_RETRIEVERS_CACHE: dict[str, object] | None = None
_PDF_RETRIEVERS_LOCK = threading.Lock()
_WIKIPEDIA_CACHE: dict[str, str] = {}
_WIKIPEDIA_LOCK = threading.Lock()


def _get_pdf_retrievers() -> dict[str, object]:
    """Build and cache PDF retrievers once (FAISS + BM25)."""
    global _PDF_RETRIEVERS_CACHE
    if _PDF_RETRIEVERS_CACHE is not None:
        return _PDF_RETRIEVERS_CACHE

    with _PDF_RETRIEVERS_LOCK:
        if _PDF_RETRIEVERS_CACHE is not None:
            return _PDF_RETRIEVERS_CACHE

        docs = load_documents()
        splits = split_documents(docs)
        texts = [doc.page_content for doc in splits]
        metadatas = [doc.metadata for doc in splits]

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=get_config().openai_api_key,
        )
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_texts(
            texts=texts,
            metadatas=metadatas,
            k=5,
        )

        _PDF_RETRIEVERS_CACHE = {
            "vector_retriever": vector_retriever,
            "bm25_retriever": bm25_retriever,
        }
        return _PDF_RETRIEVERS_CACHE


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


@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or facts.

    Use when: user asks about recent events, stock prices,
    weather, news, or anything requiring up-to-date internet data.

    Returns: Formatted search results with `Source URL:` markers.
    """

    # Tavily can be slow/fragile depending on the query; enforce tight timeout.
    # We run it in a thread so the tool does not block indefinitely.
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

    return "\n\n".join(formatted)

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
    with _WIKIPEDIA_LOCK:
        cached = _WIKIPEDIA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        # Keep tool simple: it may be executed concurrently by LangGraph.
        # Avoid nested ThreadPoolExecutors which can be unstable.
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
        with _WIKIPEDIA_LOCK:
            _WIKIPEDIA_CACHE[cache_key] = result
        return result
    except wiki.exceptions.DisambiguationError:
        result = f"Wikipedia: Multiple matches found for '{query}'. Please be more specific."
        with _WIKIPEDIA_LOCK:
            _WIKIPEDIA_CACHE[cache_key] = result
        return result
    except wiki.exceptions.PageError:
        result = f"Wikipedia: No page found for '{query}'."
        with _WIKIPEDIA_LOCK:
            _WIKIPEDIA_CACHE[cache_key] = result
        return result
    except Exception:
        result = f"Wikipedia: Error searching for '{query}'."
        with _WIKIPEDIA_LOCK:
            _WIKIPEDIA_CACHE[cache_key] = result
        return result


@tool
def code_executor(code: str) -> str:
    """Execute Python code in a sandboxed environment.

    Use when: user asks to run code, generate data, create visualizations,
    or perform complex computations beyond basic math.

    WARNING: Only Python code is supported. Timeout is 10 seconds.

    Returns: stdout/stderr from code execution.
    """
    # Guard: if input looks like a natural-language request, do not execute.
    if _looks_like_natural_language_request(code):
        return (
            "Input appears to be a coding request, not executable Python code. "
            "Please provide Python code to run, or ask for code generation/explanation."
        )

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


def _looks_like_natural_language_request(text: str) -> bool:
    """Heuristic check to avoid executing plain natural-language prompts as code."""
    stripped = text.strip()
    if not stripped:
        return True
    # Simple signals that this is likely prose, not Python source.
    prose_markers = [
        "write",
        "explain",
        "how to",
        "please",
        "can you",
        "help me",
        "show me",
        "?",
    ]
    python_markers = [
        "def ",
        "import ",
        "print(",
        "for ",
        "while ",
        "if ",
        "class ",
        "=",
        ":",
    ]
    has_prose = any(marker in stripped.lower() for marker in prose_markers)
    has_python_signal = any(marker in stripped for marker in python_markers)
    return has_prose and not has_python_signal


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