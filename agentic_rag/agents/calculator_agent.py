"""Calculator Agent - Specialized agent for math operations."""

import traceback

from agentic_rag.tools import calculator
from agentic_rag.agents.create_agent import create_agent_runner, run_agent
from core.core.system_prompt import get_system_prompt


def create_calculator_agent():
    """Create calculator agent."""
    return create_agent_runner([calculator], get_system_prompt("calculator"))


def run_calculator_agent(query: str) -> str:
    """Run calculator agent for math queries."""
    try:
        agent = create_calculator_agent()
        result = run_agent(agent, query)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                return messages[-1].content
        return str(result)
    except Exception as e:
        traceback.print_exc()
        return f"Error in calculator agent: {str(e)}"