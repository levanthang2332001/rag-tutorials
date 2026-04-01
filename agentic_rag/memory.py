"""Conversation memory for multi-agent system."""

from langchain_community.chat_message_histories import ChatMessageHistory

class ConversationMemory:
  """Simple session-based conversation memory.
  Uses LangChain's ChatMessageHistory for persistence within sessions.
  """

  def __init__(self):
    self.sessions: dict[str, ChatMessageHistory] = {}

  def _get_or_create(self, session_id: str) -> ChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in self.sessions:
      self.sessions[session_id] = ChatMessageHistory()
    return self.sessions[session_id]

  def add_user_message(self, session_id: str, content: str):
    """Add a user message to history."""
    history = self._get_or_create(session_id)
    history.add_user_message(content)

  def add_ai_message(self, session_id: str, content: str):
    """Add an AI message to history."""
    history = self._get_or_create(session_id)
    history.add_ai_message(content)

  def get_messages(self, session_id: str) -> list:
    """Get all messages for a session."""
    return self._get_or_create(session_id).messages

  def get_last_n_messages(self, session_id: str, n: int = 10) -> list:
    """Get last N messages for a session."""
    messages = self.get_messages(session_id)
    return messages[-n:] if len(messages) > n else messages

  def clear(self, session_id: str):
    """Clear history for a session."""
    if session_id in self.sessions:
      del self.sessions[session_id]

  def list_sessions(self) -> list[str]:
    """List all active session IDs."""
    return list(self.sessions.keys())

    