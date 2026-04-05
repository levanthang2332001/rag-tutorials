"""Map LangGraph ``stream_mode='updates'`` chunks to normalized SSE-style events."""

from __future__ import annotations

from typing import Any, Iterator

from ..state import AgentState


def _events_for_node(node_name: str, update: dict) -> Iterator[dict]:
    if node_name == "orchestrator":
        yield {
            "type": "orchestrator",
            "selected_agents": update.get("selected_agents", []),
            "sub_questions": update.get("sub_questions", []),
        }
        return

    if node_name == "run_agent":
        agent_results = update.get("agent_results", {}) or {}
        if isinstance(agent_results, dict):
            for agent_name, agent_output in agent_results.items():
                yield {
                    "type": "agent_done",
                    "agent_name": agent_name,
                    "output": str(agent_output),
                }
        return

    if node_name == "synthesizer":
        yield {
            "type": "synth_done",
            "answer": update.get("answer", ""),
            "iterations": update.get("iterations", 0),
            "missing_info": update.get("missing_info", False),
        }
        return

    if node_name == "verifier":
        yield {
            "type": "verify_done",
            "confidence": update.get("confidence", 0.0),
            "missing_info": update.get("missing_info", False),
            "conflicts": update.get("conflicts", False),
            "needed_agents": update.get("needed_agents", []),
        }


def iter_normalized_graph_events(
    graph: Any,
    initial_state: AgentState,
    final_state: dict[str, Any],
) -> Iterator[dict]:
    """Stream graph updates; merge state into ``final_state``; yield normalized events."""
    final_state.clear()
    final_state.update(initial_state)

    for chunk in graph.stream(initial_state, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue

        for node_name, update in chunk.items():
            if isinstance(update, dict):
                final_state.update(update)

            if isinstance(update, dict):
                yield from _events_for_node(node_name, update)


def build_final_event(session_id: str, final_state: dict[str, Any]) -> dict:
    """Terminal event with answer, iterations, and per-agent results."""
    answer = final_state.get("answer", "No answer generated")
    agent_results = final_state.get("agent_results", {}) or {}
    return {
        "type": "final",
        "answer": answer,
        "session_id": session_id,
        "iterations": final_state.get("iterations", 0),
        "agents_used": list(agent_results.keys()) if isinstance(agent_results, dict) else [],
        "agent_results": agent_results if isinstance(agent_results, dict) else {},
    }
