"""
Lightweight tests that do not require OpenAI calls.

Purpose:
- Ensure our prompts enforce an `Answer:` + `Sources:` structure.
- Ensure `pdf_retriever` uses the lazy cache without rebuilding when cached.
"""

from __future__ import annotations

import os
import sys


sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


def test_prompts_have_sources() -> None:
    from core.system_prompt import SYSTEM_PROMPTS
    from agentic_rag.graph import SYNTHESIS_PROMPT_TEMPLATE

    for agent_type in ["pdf", "web", "wikipedia"]:
        prompt = SYSTEM_PROMPTS[agent_type]
        assert "Sources:" in prompt, f"Missing Sources: in system prompt {agent_type}"

    assert "Sources:" in SYNTHESIS_PROMPT_TEMPLATE


def test_pdf_cache_reuse_without_build() -> None:
    import agentic_rag.tools as tools

    sentinel = {"vector_retriever": object(), "bm25_retriever": object()}
    old_cache = tools._PDF_RETRIEVERS_CACHE
    try:
        tools._PDF_RETRIEVERS_CACHE = sentinel
        rebuilt = tools._get_pdf_retrievers()
        assert rebuilt is sentinel, "Cache should be reused (identity check)"
    finally:
        tools._PDF_RETRIEVERS_CACHE = old_cache


if __name__ == "__main__":
    test_prompts_have_sources()
    test_pdf_cache_reuse_without_build()
    print("[SUCCESS] test_sources_format_and_cache passed")

