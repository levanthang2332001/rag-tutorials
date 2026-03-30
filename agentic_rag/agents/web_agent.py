"""Web Agent - Specialized agent for web search."""

import traceback

from agentic_rag.tools import web_search
from agentic_rag.agents.create_agent import create_agent_runner, run_agent
from lib.core.system_prompt import get_system_prompt


def create_web_agent():
    """Create web search agent."""
    return create_agent_runner([web_search], get_system_prompt("web"))


def run_web_agent(query: str) -> str:
    """Run web agent to search the internet."""
    try:
        agent = create_web_agent()
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
        return f"Error in web agent: {str(e)}"