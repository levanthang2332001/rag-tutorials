"""OpenAI API wrapper."""

from langchain_openai import ChatOpenAI
from core.config import get_config

config = get_config()

llm = ChatOpenAI(
    model=config.openai_model,
    temperature=0,
    # Hard timeout so tool/tests won't hang indefinitely.
    # LangChain/OpenAI calls are the most common source of long stalls.
    timeout=25,
    max_retries=1,
)

def get_llm():
    """Get the OpenAI LLM."""
    return llm