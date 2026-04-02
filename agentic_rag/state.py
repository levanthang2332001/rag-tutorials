"""State definition for LangGraph multi-agent workflow."""

from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


def reduce_agent_results(left: dict, right: dict) -> dict:
    """Merge agent results from parallel updates."""
    if left is None:
        return right or {}
    if right is None:
        return left
    # Merge - keep both keys
    result = dict(left)
    result.update(right)
    return result


class AgentState(TypedDict):
    """Shared state across all nodes."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    selected_agents: list[str]
    agent_results: Annotated[dict[str, str], reduce_agent_results]
    answer: Optional[str]
    iterations: int
    missing_info: bool
    confidence: float
    conflicts: bool
    needed_agents: list[str]
    sub_questions: list[SubQuestion]


class AgentInvokeState(TypedDict):
    """State for individual agent invocation via Send()."""
    query: str
    agent_name: str


class SubQuestion(TypedDict):
    """A decomposed sub-task to run on a specific specialist agent."""
    id: str
    task_type: str
    agent_name: str
    text: str