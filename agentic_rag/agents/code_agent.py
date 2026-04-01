"""Code Agent - Specialized agent for code execution."""

import traceback

from agentic_rag.tools import code_executor
from agentic_rag.agents.create_agent import create_agent_runner, run_agent
from core.system_prompt import get_system_prompt


def create_code_agent():
    """Create code execution agent."""
    return create_agent_runner([code_executor], get_system_prompt("code"))


def run_code_agent(query: str) -> str:
    """Run code agent for code execution requests."""
    try:
        agent = create_code_agent()
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
        return f"Error in code agent: {str(e)}"