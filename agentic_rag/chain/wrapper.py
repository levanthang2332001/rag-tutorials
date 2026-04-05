"""Chain wrapper for LangGraph multi-agent system."""

from __future__ import annotations

from typing import Any

from ..graph import create_agent_graph
from ..memory import ConversationMemory
from ..state import AgentState
from .context import build_contextual_query, build_initial_state
from .stream_events import build_final_event, iter_normalized_graph_events


class AgenticChain:
  """Main interface for the multi-agent RAG system.

  Wraps LangGraph workflow with memory and simple interface.
  """

  def __init__(self):
      self.graph = create_agent_graph()
      self.memory = ConversationMemory()

  def invoke(
      self,
      query: str,
      session_id: str = "default",
      return_raw_state: bool = False,
  ) -> dict:
      """Invoke the multi-agent graph.

      Args:
          query: User question
          session_id: Session ID for memory
          return_raw_state: If True, return full state

      Returns:
          dict with answer, session_id, iterations, agents_used, agent_results
      """
      session_id = session_id.strip() or "default"
      recent_messages = self.memory.get_last_n_messages(session_id, n=6)
      contextual_query = build_contextual_query(query, recent_messages, n=6)
      self.memory.add_user_message(session_id, query)

      initial_state: AgentState = build_initial_state(contextual_query)

      try:
          result = self.graph.invoke(initial_state)
      except Exception as e:
          print(f"Graph execution error: {e}")
          result = {
              "answer": f"Error executing query: {str(e)}",
              "iterations": 0,
              "agent_results": {},
          }

      answer = result.get("answer", "No answer generated")
      self.memory.add_ai_message(session_id, answer)

      response = {
          "answer": answer,
          "session_id": session_id,
          "iterations": result.get("iterations", 0),
          "agents_used": list(result.get("agent_results", {}).keys()),
          "agent_results": result.get("agent_results", {}),
      }

      if return_raw_state:
          response["raw_state"] = result

      return response

  def stream(
      self,
      query: str,
      session_id: str = "default",
  ):
      """Stream graph execution as normalized events.

      Yields dict events shaped like:
      - {"type": "start", "session_id": ..., "query": ...}
      - {"type": "orchestrator", "selected_agents": [...], "sub_questions": [...]}
      - {"type": "agent_done", "agent_name": ..., "output": ...}
      - {"type": "synth_done", "answer": ..., "iterations": ..., "missing_info": ...}
      - {"type": "verify_done", "confidence": ..., "missing_info": ..., "conflicts": ...}
      - {"type": "final", "answer": ..., "session_id": ..., "iterations": ..., "agents_used": [...], "agent_results": {...}}
      """
      session_id = session_id.strip() or "default"
      recent_messages = self.memory.get_last_n_messages(session_id, n=6)
      contextual_query = build_contextual_query(query, recent_messages, n=6)
      self.memory.add_user_message(session_id, query)

      initial_state: AgentState = build_initial_state(contextual_query)

      yield {"type": "start", "session_id": session_id, "query": query}

      final_state: dict[str, Any] = {}
      for event in iter_normalized_graph_events(self.graph, initial_state, final_state):
          yield event

      answer = final_state.get("answer", "No answer generated")
      self.memory.add_ai_message(session_id, answer)

      yield build_final_event(session_id, final_state)

  def chat(self, query: str, session_id: str = "default") -> str:
      """Simple chat interface - returns just the answer string."""
      result = self.invoke(query, session_id)
      return result["answer"]

  def get_history(self, session_id: str = "default") -> list:
      """Get conversation history for a session."""
      return self.memory.get_messages(session_id)

  def clear_history(self, session_id: str):
      """Clear history for a session."""
      self.memory.clear(session_id)

  def list_sessions(self) -> list[str]:
      """List all active sessions."""
      return self.memory.list_sessions()

  def get_last_interaction(self, session_id: str = "default") -> dict | None:
      """Get the last query-response pair."""
      messages = self.get_history(session_id)
      if len(messages) >= 2:
          last_user = messages[-2].content if messages[-2].type == "human" else ""
          last_ai = messages[-1].content if messages[-1].type == "ai" else ""
          return {"query": last_user, "answer": last_ai}
      return None


def create_agentic_chain() -> AgenticChain:
    """Factory function to create an AgenticChain instance."""
    return AgenticChain()
