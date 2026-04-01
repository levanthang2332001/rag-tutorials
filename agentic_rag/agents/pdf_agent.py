"""PDF Agent - Specialized agent for PDF document search."""

import traceback

from agentic_rag.tools import pdf_retriever
from agentic_rag.agents.create_agent import create_agent_runner, run_agent
from core.system_prompt import get_system_prompt


def create_pdf_agent():
    """Create PDF agent."""
    return create_agent_runner([pdf_retriever], get_system_prompt("pdf"))


def run_pdf_agent(query: str) -> str:
    """Run PDF agent to search internal documents."""
    try:
        agent = create_pdf_agent()
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
        return f"Error in PDF agent: {str(e)}"