"""LangGraph multi-agent workflow with parallel agent execution."""

import json
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI

from .state import AgentState, AgentInvokeState
from .agents import (
    run_pdf_agent,
    run_web_agent,
    run_calculator_agent,
    run_wikipedia_agent,
    run_code_agent,
    run_sql_agent,
)

from lib.openai.openai import get_llm

AGENT_REGISTRY: dict[str, callable] = {
    "pdf_agent": run_pdf_agent,
    "web_agent": run_web_agent,
    "calculator_agent": run_calculator_agent,
    "wikipedia_agent": run_wikipedia_agent,
    "code_agent": run_code_agent,
    "sql_agent": run_sql_agent,
}

AGENT_DESCRIPTIONS = {
    "pdf_agent": "Find information in PDF documents",
    "web_agent": "Find information on the web",
    "calculator_agent": "Perform calculations",
    "wikipedia_agent": "Find information on Wikipedia",
    "code_agent": "Execute code to get information",
    "sql_agent": "Query a database for information",
}

MAX_ITERATIONS = 2
MIN_ANSWER_LENGTH = 80

ROUTING_PROMPT_TEMPLATE = """You are a router for a multi-agent RAG system.
Analyze the query and select the appropriate agents.

Query: {query}

Available agents:
{agent_list}

Previous attempt (if any): {previous_attempt}

Rules:
- Only select agents that are truly necessary, do not over-select
- If it's a math question: only calculator_agent
- If asking about general knowledge: only wikipedia_agent
- If needing multiple sources: select max 3 agents
- If previous attempt lacked information: select additional agents

Return only valid JSON (no explanation):
{{"agents": ["agent1", "agent2"], "reasoning": "brief reason"}}"""


def orchestrator_node(state: AgentState) -> dict:
    """Route query to appropriate agents - stores selected agents in state."""
    query = state["query"]
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
        query=query,
        agent_list=agent_list,
        previous_attempt=previous_attempt,
    )

    response = llm.invoke(routing_prompt)

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

    return {
        "selected_agents": selected,
        "iterations": iterations,
    }


def route_to_agents(state: AgentState) -> list[Send]:
    """Fan-out to selected agents in parallel via Send()."""
    query = state["query"]
    selected = state.get("selected_agents", [])

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

    agent_func = AGENT_REGISTRY.get(agent_name)
    if not agent_func:
        return {"agent_results": {agent_name: f"Agent '{agent_name}' not found."}}

    try:
        result = agent_func(query)
    except Exception as e:
        result = f"{agent_name} error: {str(e)}"

    return {"agent_results": {agent_name: str(result)}}


SYNTHESIS_PROMPT_TEMPLATE = """You are a synthesizer for a multi-agent RAG system.
Synthesize information from agents and answer the query.

Original query: {query}
Attempt: {iteration}/{max_iter}

Results from agents:
{results_text}

Guidelines:
- Synthesize into a coherent, complete answer
- If agents have conflicting information, note the differences
- If important information is missing, note what's missing
- Answer in the same language as the query
- Minimum length: 2-3 complete sentences"""


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

    results_text = "\n\n".join(
        f"[{agent.upper()}]\n{result}"
        for agent, result in results.items()
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    response = llm.invoke(
        SYNTHESIS_PROMPT_TEMPLATE.format(
            query=query,
            iteration=iterations,
            max_iter=MAX_ITERATIONS,
            results_text=results_text,
        )
    )

    answer = response.content.strip()
    missing_info = _detect_missing_info(answer)

    return {
        "answer": answer,
        "iterations": iterations,
        "missing_info": missing_info,
    }


def _detect_missing_info(answer: str) -> bool:
    """Check if answer is missing information."""
    if len(answer) < MIN_ANSWER_LENGTH:
        return True

    insufficient_signals = [
        "not found", "no information", "need more information",
        "cannot answer", "insufficient data",
        "no relevant", "could not find",
    ]
    answer_lower = answer.lower()
    return any(signal in answer_lower for signal in insufficient_signals)


def should_continue(state: AgentState) -> Literal["orchestrator", "__end__"]:
    """Decide whether to loop or end based on answer quality."""
    iterations = state.get("iterations", 0)
    missing_info = state.get("missing_info", False)
    answer = state.get("answer", "")

    if iterations >= MAX_ITERATIONS:
        return END

    if not missing_info and len(answer) >= MIN_ANSWER_LENGTH:
        return END

    return "orchestrator"


def create_agent_graph():
    """Build LangGraph with Send() parallel execution."""
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("run_agent", run_agent_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("orchestrator")

    # After orchestrator, fan-out to agents in parallel
    builder.add_conditional_edges(
        "orchestrator",
        route_to_agents,
    )

    # After run_agent, go to synthesizer
    builder.add_edge("run_agent", "synthesizer")

    # After synthesizer, either loop back or end
    builder.add_conditional_edges(
        "synthesizer",
        should_continue,
        {"orchestrator": "orchestrator", END: END},
    )

    return builder.compile()