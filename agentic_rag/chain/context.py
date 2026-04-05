"""Conversation context: contextual query + initial graph state."""

from ..state import AgentState


def build_contextual_query(query: str, messages: list, n: int = 6) -> str:
    """Attach short recent context so agents can resolve follow-up references."""
    if not messages:
        return query

    context_lines = []
    for message in messages[-n:]:
        role = "User" if message.type == "human" else "Assistant"
        context_lines.append(f"{role}: {message.content}")

    context_block = "\n".join(context_lines)
    return (
        "=== CONVERSATION HISTORY (for context only) ===\n"
        f"{context_block}\n\n"
        f">>> CURRENT USER REQUEST (needs answer) <<<\n{query}"
    )


def build_initial_state(contextual_query: str) -> AgentState:
    """Default AgentState for invoke/stream entry."""
    return {
        "messages": [],
        "query": contextual_query,
        "selected_agents": [],
        "agent_results": {},
        "sub_questions": [],
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "confidence": 0.0,
        "conflicts": False,
        "needed_agents": [],
    }
