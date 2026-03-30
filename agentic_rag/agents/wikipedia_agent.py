"""Wikipedia Agent - Specialized agent for Wikipedia search."""

import traceback

from agentic_rag.tools import wikipedia_search
from agentic_rag.agents.create_agent import create_agent_runner, run_agent
from core.core.system_prompt import get_system_prompt


def create_wikipedia_agent():
    """Create Wikipedia agent."""
    return create_agent_runner([wikipedia_search], get_system_prompt("wikipedia"))


def run_wikipedia_agent(query: str) -> str:
    """Run Wikipedia agent for factual queries."""
    try:
        agent = create_wikipedia_agent()
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
        return f"Error in Wikipedia agent: {str(e)}"