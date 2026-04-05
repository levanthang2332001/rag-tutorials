"""Deterministic query classification helpers for orchestration."""


def extract_current_request(query: str) -> str:
    """Extract latest user request from contextual query payload."""
    marker = ">>> CURRENT USER REQUEST (needs answer) <<<"
    if marker not in query:
        return query
    extracted = query.split(marker, maxsplit=1)[1].strip()
    return extracted or query


def looks_like_code_query(query: str) -> bool:
    """Heuristic detection for programming/code related requests."""
    query_lower = query.lower()
    strong_signals = [
        "```",
        "traceback",
        "stack trace",
        "exception",
        "error:",
        "typeerror",
        "valueerror",
        "referenceerror",
        "nullpointer",
        "segmentation fault",
        "syntaxerror",
        "module not found",
        "npm ",
        "pip ",
        "pnpm ",
        "yarn ",
        "compile",
        "build failed",
        "refactor",
        "rewrite code",
        "write code",
        "debug",
        "fix this",
        "implement",
        "code review",
    ]
    structural_signals = ["def ", "class ", "import ", "console.log", "SELECT ", "CREATE TABLE "]
    return any(s in query_lower for s in strong_signals) or any(
        s.lower() in query_lower for s in structural_signals
    )


def looks_like_language_definition_query(query: str) -> bool:
    """Detect definition/overview questions about programming languages."""
    q = query.lower().strip()
    language_markers = ["python", "javascript", "typescript", "java", "c++", "c#", "golang", "rust"]
    definition_intents = ["what is", "define", "overview", "explain", "introduction to"]
    code_action_intents = ["write code", "refactor", "debug", "fix", "implement", "optimize"]
    is_language = any(lang in q for lang in language_markers)
    is_definition = any(intent in q for intent in definition_intents)
    is_action = any(intent in q for intent in code_action_intents) or "```" in q or "traceback" in q
    return is_language and is_definition and not is_action


def is_strict_code_query(query: str) -> bool:
    """Detect requests that should be handled only by code_agent."""
    query_lower = query.lower()
    strict_markers = [
        "refactor",
        "rewrite code",
        "write code",
        "implement",
        "debug",
        "fix this code",
        "optimize this code",
        "code review",
        "explain this code",
        "def ",
        "class ",
        "stack trace",
    ]
    non_code_markers = [
        "news",
        "wikipedia",
        "latest",
        "paper",
        "pdf",
        "database",
        "sql",
    ]
    is_code = any(marker in query_lower for marker in strict_markers)
    has_non_code = any(marker in query_lower for marker in non_code_markers)
    return is_code and not has_non_code
