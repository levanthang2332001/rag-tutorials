# Plan 6: Memory & Chain

## Overview
Implement conversation memory and wrap LangGraph with a simple chain interface.

## File: `agentic_rag/memory.py`

```python
"""Conversation memory for multi-agent system."""

from typing import Optional
from langchain_community.chat_message_histories import ChatMessageHistory


class ConversationMemory:
    """Simple session-based conversation memory.

    Uses LangChain's ChatMessageHistory for persistence within sessions.
    """

    def __init__(self):
        self.sessions: dict[str, ChatMessageHistory] = {}

    def get_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    def add_user_message(self, session_id: str, content: str):
        """Add a user message to history."""
        history = self.get_history(session_id)
        history.add_user_message(content)

    def add_ai_message(self, session_id: str, content: str):
        """Add an AI message to history."""
        history = self.get_history(session_id)
        history.add_ai_message(content)

    def get_messages(self, session_id: str) -> list:
        """Get all messages for a session."""
        return self.get_history(session_id).messages

    def get_last_n_messages(self, session_id: str, n: int = 10) -> list:
        """Get last N messages for a session."""
        messages = self.get_messages(session_id)
        return messages[-n:] if len(messages) > n else messages

    def clear(self, session_id: str):
        """Clear history for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def clear_all(self):
        """Clear all session histories."""
        self.sessions.clear()

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
```

## File: `agentic_rag/chain.py`

```python
"""Chain wrapper for LangGraph multi-agent system."""

from typing import Optional
from langchain_core.tracers import LangChainTracer
from langsmith import trace, Client

from .graph import create_agent_graph
from .memory import ConversationMemory
from .state import AgentState


class AgenticChain:
    """Main interface for the multi-agent RAG system.

    Wraps LangGraph workflow with memory and tracing.
    """

    def __init__(self):
        # Initialize LangGraph
        self.graph = create_agent_graph()

        # Initialize memory
        self.memory = ConversationMemory()

        # Initialize tracing
        self.tracer = LangChainTracer()
        self.client = Client()

    @trace(
        name="agentic_rag_full",
        tags=["full_pipeline", "multi-agent"],
        metadata={"graph": "agentic-rag"},
    )
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
            dict with answer, session_id, iterations, agents_used
        """
        # Add user message to memory
        self.memory.add_user_message(session_id, query)

        # Initial state for graph
        initial_state: AgentState = {
            "messages": [],
            "query": query,
            "selected_agents": [],
            "agent_results": {},
            "answer": None,
            "iterations": 0,
            "missing_info": True,
            "tracer": self.tracer,
        }

        # Run graph
        try:
            result = self.graph.invoke(initial_state)
        except Exception as e:
            print(f"Graph execution error: {e}")
            result = {
                "answer": f"Error executing query: {str(e)}",
                "iterations": 0,
                "agent_results": {},
            }

        # Extract answer
        answer = result.get("answer", "No answer generated")

        # Add response to memory
        self.memory.add_ai_message(session_id, answer)

        # Build response
        response = {
            "answer": answer,
            "session_id": session_id,
            "iterations": result.get("iterations", 0),
            "agents_used": list(result.get("agent_results", {}).keys()),
        }

        if return_raw_state:
            response["raw_state"] = result

        return response

    def chat(
        self,
        query: str,
        session_id: str = "default",
    ) -> str:
        """Simple chat interface - returns just the answer String."""
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

    def get_last_interaction(
        self,
        session_id: str = "default",
    ) -> dict | None:
        """Get the last query-response pair."""
        messages = self.get_history(session_id)
        if len(messages) >= 2:
            last_user = messages[-2].content if messages[-2].type == "human" else ""
            last_ai = messages[-1].content if messages[-1].type == "ai" else ""
            return {"query": last_user, "answer": last_ai}
        return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_agentic_chain() -> AgenticChain:
    """Factory function to create an AgenticChain instance."""
    return AgenticChain()
```

## Usage Examples

### Basic Usage

```python
from agentic_rag.chain import create_agentic_chain

chain = create_agentic_chain()

# Simple chat
answer = chain.chat("What is RAG?")
print(answer)

# With metadata
result = chain.invoke("Calculate 2^10")
print(f"Answer: {result['answer']}")
print(f"Agents used: {result['agents_used']}")
print(f"Iterations: {result['iterations']}")
```

### With Session Memory

```python
# Start a conversation
chain.chat("What is our company's AI policy?", session_id="user123")
chain.chat("Does this apply to contractors?", session_id="user123")

# Get conversation history
history = chain.get_history(session_id="user123")
for msg in history:
    print(f"{msg.type}: {msg.content}")

# Clear when done
chain.clear_history("user123")
```

## Next Steps

After memory and chain, proceed to: [7-api-cli.plan.md](./7-api-cli.plan.md)
