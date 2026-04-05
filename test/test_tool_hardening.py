from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_calculator_safe_eval_allows_basic_math() -> None:
    from agentic_rag.tools import calculator

    assert calculator.invoke({"expression": "2+2"}).strip() in {"4", "4.0"}
    assert calculator.invoke({"expression": "sqrt(144)"}).strip() in {"12", "12.0"}
    assert calculator.invoke({"expression": "pi"}).strip() != ""


def test_calculator_safe_eval_blocks_attributes() -> None:
    from agentic_rag.tools import calculator

    out = calculator.invoke({"expression": "__import__('os').system('echo hi')"})
    assert out.startswith("Error:")


def test_code_executor_uses_current_interpreter() -> None:
    from agentic_rag.tools import code_executor

    # Prints current interpreter path; should run under sys.executable.
    out = code_executor.invoke({"code": "import sys; print(sys.executable)"})
    assert sys.executable in out


def test_wikipedia_cache_is_bounded_type() -> None:
    import agentic_rag.tools as tools

    # Internal cache object should be bounded cache implementation (has get/set).
    assert hasattr(tools, "_WIKIPEDIA_CACHE")
    assert hasattr(tools._WIKIPEDIA_CACHE, "get")
    assert hasattr(tools._WIKIPEDIA_CACHE, "set")


if __name__ == "__main__":
    test_calculator_safe_eval_allows_basic_math()
    test_calculator_safe_eval_blocks_attributes()
    test_code_executor_uses_current_interpreter()
    test_wikipedia_cache_is_bounded_type()
    print("[SUCCESS] test_tool_hardening passed")

