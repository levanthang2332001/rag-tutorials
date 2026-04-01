"""Helper module for creating LangChain agents."""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def create_agent_runner(tools: list, system_prompt: str):
    """Create a LangChain agent with specified tools and system prompt.

    Args:
        tools: List of LangChain tools to make available to the agent.
        system_prompt: The system prompt that defines the agent's behavior.

    Returns:
        A compiled LangChain agent.
    """
    return create_agent(
        ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools,
        system_prompt=system_prompt,
    )


def run_agent(agent, query: str):
    """Run an agent with the given query.

    Args:
        agent: A compiled LangChain agent.
        query: The user query to process.

    Returns:
        The agent's response.
    """
    return agent.invoke({"messages": [("user", query)]})