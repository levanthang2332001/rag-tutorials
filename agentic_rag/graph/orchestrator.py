"""Orchestrator node: LLM routing with heuristic fallback."""

from __future__ import annotations

import json
import os
import time

from core.openai.openai import get_llm

from ..state import AgentState

from .constants import (
    AGENT_DESCRIPTIONS,
    AGENT_INSTRUCTIONS,
    AGENT_TASK_TYPES,
    ROUTER_FACT_KEYWORDS,
    ROUTER_MATH_KEYWORDS,
    ROUTING_PROMPT_TEMPLATE,
    WEB_REQUIRED_KEYWORDS,
)
from .registry import AGENT_REGISTRY
from .routing_heuristics import (
    extract_current_request,
    is_strict_code_query,
    looks_like_code_query,
    looks_like_language_definition_query,
)


def orchestrator_node(state: AgentState) -> dict:
    """Route query to appropriate agents - stores selected agents in state."""
    started_at = time.time()
    query = state["query"]
    current_request = extract_current_request(query)
    iterations = state.get("iterations", 0)
    previous_results = state.get("agent_results", {})

    previous_attempt = "No previous attempt" if not previous_results else (
        f"Used: {list(previous_results.keys())} - missing information"
    )

    agent_list = "\n".join(
        f"- {name}: {desc}"
        for name, desc in AGENT_DESCRIPTIONS.items()
    )

    llm = get_llm()

    routing_prompt = ROUTING_PROMPT_TEMPLATE.format(
        query=current_request,
        agent_list=agent_list,
        previous_attempt=previous_attempt,
    )

    def _build_sub_questions(selected_agents: list[str]) -> list[dict]:
        sub_questions: list[dict] = []
        for agent_name in selected_agents:
            task_type = AGENT_TASK_TYPES.get(agent_name, agent_name)
            instruction_tail = AGENT_INSTRUCTIONS.get(agent_name, "")
            instruction = (
                f"Original query: {current_request}\n{instruction_tail}".strip()
                if instruction_tail
                else current_request
            )
            sub_questions.append(
                {
                    "id": f"{task_type}:{agent_name}",
                    "task_type": task_type,
                    "agent_name": agent_name,
                    "text": instruction,
                }
            )
        return sub_questions

    try:
        response = llm.invoke(routing_prompt)
    except Exception:
        query_lower = current_request.lower()
        selected: list[str] = []

        if is_strict_code_query(current_request) or looks_like_code_query(current_request):
            selected = ["code_agent"]
        else:
            is_math = any(k in query_lower for k in ROUTER_MATH_KEYWORDS)
            is_factual = any(k in query_lower for k in ROUTER_FACT_KEYWORDS)

            if is_math:
                selected.append("calculator_agent")
            if is_factual:
                selected.append("wikipedia_agent")
            if not selected:
                selected = ["wikipedia_agent"]

        selected = selected[:3]
        sub_questions = _build_sub_questions(selected)
        result = {
            "selected_agents": selected,
            "sub_questions": sub_questions,
            "iterations": iterations,
            "query": current_request,
            "needed_agents": [],
        }
        if os.getenv("DEBUG_GRAPH") == "1":
            print(
                "[DEBUG_GRAPH] orchestrator(fallback)",
                {
                    "selected_agents": result["selected_agents"],
                    "sub_questions": len(result["sub_questions"]),
                    "iterations": iterations,
                    "ms": int((time.time() - started_at) * 1000),
                },
            )
        return result

    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1]
        if content.endswith("```"):
            content = content.split("```")[0]
        parsed = json.loads(content.strip())
        selected = [a for a in parsed.get("agents", []) if a in AGENT_REGISTRY]
    except (json.JSONDecodeError, KeyError, IndexError):
        selected = ["wikipedia_agent"]

    if not selected:
        selected = ["wikipedia_agent"]

    if is_strict_code_query(current_request) or looks_like_code_query(current_request):
        selected = ["code_agent"]
    elif looks_like_language_definition_query(current_request):
        selected = ["wikipedia_agent"]

    query_lower = current_request.lower()
    if not any(k in query_lower for k in WEB_REQUIRED_KEYWORDS):
        selected = [a for a in selected if a != "web_agent"]
        if not selected:
            selected = ["wikipedia_agent"]

    needed_agents = state.get("needed_agents", []) or []
    for agent_name in needed_agents:
        if agent_name in AGENT_REGISTRY and agent_name not in selected:
            selected.append(agent_name)

    selected = selected[:3]

    sub_questions = _build_sub_questions(selected)

    result = {
        "selected_agents": selected,
        "sub_questions": sub_questions,
        "iterations": iterations,
        "query": current_request,
        "needed_agents": [],
    }
    if os.getenv("DEBUG_GRAPH") == "1":
        print(
            "[DEBUG_GRAPH] orchestrator",
            {
                "selected_agents": result["selected_agents"],
                "sub_questions": len(result["sub_questions"]),
                "iterations": iterations,
                "ms": int((time.time() - started_at) * 1000),
            },
        )
    return result
