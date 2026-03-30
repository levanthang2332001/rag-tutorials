from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langsmith import Client

# Absolute path to .env (project root)
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / ".env")


class Config(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    env_ignore_empty=True,
    extra="ignore",
    populate_by_name=True,
  )

  openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
  openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
  langsmith_api_key: str = Field(default="", validation_alias="LANGSMITH_API_KEY")
  langsmith_project: str = Field(default="", validation_alias="LANGSMITH_PROJECT")
  langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
  langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", validation_alias="LANGSMITH_ENDPOINT")

  @property
  def langsmith_client(self) -> Client:
    return Client(
      api_url=self.langsmith_endpoint,
      api_key=self.langsmith_api_key,
    )
  

@lru_cache(maxsize=1)
def get_config() -> Config:
  return Config()
