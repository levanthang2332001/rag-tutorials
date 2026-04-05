"""Heuristics for when code_executor should refuse to run input as Python."""


def looks_like_natural_language_request(text: str) -> bool:
    """Avoid executing plain natural-language prompts as code."""
    stripped = text.strip()
    if not stripped:
        return True
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
