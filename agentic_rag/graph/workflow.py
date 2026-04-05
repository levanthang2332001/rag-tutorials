"""Compile the multi-agent LangGraph workflow."""

from langgraph.graph import END, StateGraph

from ..state import AgentState
from .nodes import (
    route_to_agents,
    run_agent_node,
    should_continue,
    synthesizer_node,
    verifier_node,
)
from .orchestrator import orchestrator_node


def create_agent_graph():
    """Build LangGraph with Send() parallel execution."""
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("run_agent", run_agent_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("verifier", verifier_node)

    builder.set_entry_point("orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_to_agents,
    )

    builder.add_edge("run_agent", "synthesizer")
    builder.add_edge("synthesizer", "verifier")

    builder.add_conditional_edges(
        "verifier",
        should_continue,
        {"orchestrator": "orchestrator", END: END},
    )

    return builder.compile()
