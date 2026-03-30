"""SQL Agent - Specialized agent for database queries."""

import traceback

from agentic_rag.tools import sql_database
from agentic_rag.agents.create_agent import create_agent_runner, run_agent
from lib.core.system_prompt import get_system_prompt


def create_sql_agent():
    """Create SQL agent."""
    return create_agent_runner([sql_database], get_system_prompt("sql"))


def run_sql_agent(query: str) -> str:
    """Run SQL agent for database queries."""
    try:
        agent = create_sql_agent()
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
        return f"Error in SQL agent: {str(e)}"