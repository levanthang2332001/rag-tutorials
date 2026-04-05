"""LangGraph nodes: agent execution, synthesis, verification, routing."""

from __future__ import annotations

import json
import os
import time
from typing import Literal

from langgraph.graph import END
from langgraph.types import Send

from core.openai.openai import get_llm

from ..state import AgentInvokeState, AgentState

from .constants import (
    CONFIDENCE_THRESHOLD,
    MAX_ITERATIONS,
    MIN_ANSWER_LENGTH,
    SYNTHESIS_PROMPT_TEMPLATE,
    VERIFIER_PROMPT_TEMPLATE,
)
from .registry import AGENT_REGISTRY


def route_to_agents(state: AgentState) -> list[Send]:
    """Fan-out to selected agents in parallel via Send()."""
    sub_questions = state.get("sub_questions", []) or []
    query = state["query"]
    selected = state.get("selected_agents", [])

    if sub_questions:
        return [
            Send(
                "run_agent",
                AgentInvokeState(query=sq["text"], agent_name=sq["agent_name"]),
            )
            for sq in sub_questions
        ]

    if not selected:
        selected = ["wikipedia_agent"]

    return [
        Send("run_agent", AgentInvokeState(query=query, agent_name=agent_name))
        for agent_name in selected
    ]


def run_agent_node(state: AgentInvokeState) -> dict:
    """Run the selected agent and return the result."""
    agent_name = state["agent_name"]
    query = state["query"]
    started_at = time.time()

    agent_func = AGENT_REGISTRY.get(agent_name)
    if not agent_func:
        return {"agent_results": {agent_name: f"Agent '{agent_name}' not found."}}

    try:
        result = agent_func(query)
    except Exception as e:
        result = f"{agent_name} error: {str(e)}"

    if os.getenv("DEBUG_GRAPH") == "1":
        print(
            "[DEBUG_GRAPH] agent_done",
            {
                "agent_name": agent_name,
                "ms": int((time.time() - started_at) * 1000),
            },
        )
    return {"agent_results": {agent_name: str(result)}}


def _detect_missing_info(answer: str) -> bool:
    """Check if answer is missing information."""
    if len(answer) < MIN_ANSWER_LENGTH:
        return True

    insufficient_signals = [
        "not found",
        "no information",
        "need more information",
        "cannot answer",
        "insufficient data",
        "no relevant",
        "could not find",
    ]
    answer_lower = answer.lower()
    return any(signal in answer_lower for signal in insufficient_signals)


def synthesizer_node(state: AgentState) -> dict:
    """Synthesize results from all agents."""
    query = state["query"]
    results = state.get("agent_results", {})
    iterations = state.get("iterations", 0) + 1

    if not results:
        return {
            "answer": "Could not collect information from agents.",
            "iterations": iterations,
            "missing_info": True,
        }

    if list(results.keys()) == ["code_agent"]:
        code_answer = str(results["code_agent"]).strip()
        return {
            "answer": code_answer,
            "iterations": iterations,
            "missing_info": _detect_missing_info(code_answer),
        }

    sub_questions = state.get("sub_questions", []) or []
    task_type_by_agent = {
        sq.get("agent_name", ""): sq.get("task_type", sq.get("agent_name", ""))
        for sq in sub_questions
    }

    results_text = "\n\n".join(
        f"[{task_type_by_agent.get(agent, agent).upper()} | {agent.upper()}]\n{result}"
        for agent, result in results.items()
    )

    llm = get_llm()

    try:
        response = llm.invoke(
            SYNTHESIS_PROMPT_TEMPLATE.format(
                query=query,
                iteration=iterations,
                max_iter=MAX_ITERATIONS,
                results_text=results_text,
            )
        )
        answer = response.content.strip()
    except Exception as e:
        answer = f"Could not synthesize answer reliably due to an internal error: {str(e)}\n\n"
        answer += "\n\n".join(
            f"[{k}]\n{str(v)[:1500]}" for k, v in (results or {}).items()
        )

    missing_info = _detect_missing_info(answer)

    return {
        "answer": answer,
        "iterations": iterations,
        "missing_info": missing_info,
    }


def verifier_node(state: AgentState) -> dict:
    """Verify synthesized answer quality and decide whether to loop."""
    started_at = time.time()
    query = state.get("query", "")
    answer = state.get("answer", "") or ""
    agent_results: dict[str, str] = state.get("agent_results", {}) or {}

    if not agent_results:
        return {
            "confidence": 0.0,
            "missing_info": True,
            "conflicts": False,
            "needed_agents": [],
        }

    results_text = "\n\n".join(
        f"[{k}]\n{str(v)[:2000]}"
        for k, v in agent_results.items()
    )

    has_sources = "Sources:" in answer
    missing_info = _detect_missing_info(answer)
    if has_sources and not missing_info:
        result = {
            "confidence": 0.9,
            "missing_info": False,
            "conflicts": False,
            "needed_agents": [],
        }
        if os.getenv("DEBUG_GRAPH") == "1":
            print(
                "[DEBUG_GRAPH] verifier(heuristic_ok)",
                {"ms": int((time.time() - started_at) * 1000)},
            )
        return result

    query_lower = query.lower()
    needed_agents: list[str] = []
    is_math = any(k in query_lower for k in ["square root", "sqrt", "power", "calculate"])
    is_fact = any(
        k in query_lower
        for k in ["who", "founded", "capital", "company", "google", "microsoft"]
    )
    if is_math:
        needed_agents.append("calculator_agent")
    if is_fact:
        needed_agents.append("wikipedia_agent")

    llm = get_llm()
    try:
        response = llm.invoke(
            VERIFIER_PROMPT_TEMPLATE.format(
                query=query,
                answer=answer,
                results_text=results_text,
            )
        )
    except Exception:
        result = {
            "confidence": 0.4,
            "missing_info": missing_info,
            "conflicts": False,
            "needed_agents": needed_agents,
        }
        if os.getenv("DEBUG_GRAPH") == "1":
            print(
                "[DEBUG_GRAPH] verifier(fallback)",
                {"ms": int((time.time() - started_at) * 1000)},
            )
        return result

    raw = response.content.strip()
    if raw.startswith("```json"):
        raw = raw.split("```json", 1)[1]
    if raw.endswith("```"):
        raw = raw.split("```", 1)[0]

    try:
        parsed = json.loads(raw)
    except Exception:
        result = {
            "confidence": state.get("confidence", 0.0),
            "missing_info": state.get("missing_info", False),
            "conflicts": state.get("conflicts", False),
            "needed_agents": state.get("needed_agents", []),
        }
        if os.getenv("DEBUG_GRAPH") == "1":
            print(
                "[DEBUG_GRAPH] verifier(parse_fail)",
                {"ms": int((time.time() - started_at) * 1000)},
            )
        return result

    result = {
        "confidence": float(parsed.get("confidence", state.get("confidence", 0.0)) or 0.0),
        "missing_info": bool(parsed.get("missing_info", state.get("missing_info", False))),
        "conflicts": bool(parsed.get("conflicts", state.get("conflicts", False))),
        "needed_agents": parsed.get("needed_agents", []) or [],
    }
    if os.getenv("DEBUG_GRAPH") == "1":
        print(
            "[DEBUG_GRAPH] verifier",
            {"ms": int((time.time() - started_at) * 1000), "confidence": result["confidence"]},
        )
    return result


def should_continue(state: AgentState) -> Literal["orchestrator", "__end__"]:
    """Decide whether to loop or end based on verifier outputs."""
    iterations = state.get("iterations", 0)
    missing_info = state.get("missing_info", False)
    conflicts = state.get("conflicts", False)
    confidence = state.get("confidence", 0.0)

    if iterations >= MAX_ITERATIONS:
        return END

    if conflicts:
        return "orchestrator"

    if missing_info:
        return "orchestrator"

    if confidence < CONFIDENCE_THRESHOLD:
        return "orchestrator"

    return END
