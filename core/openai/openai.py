"""OpenAI API wrapper."""

from langchain_openai import ChatOpenAI
from core.config import get_config

config = get_config()

llm = ChatOpenAI(
    model=config.openai_model,
    temperature=0,
)

def get_llm():
    """Get the OpenAI LLM."""
    return llm