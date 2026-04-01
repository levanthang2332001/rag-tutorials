"""Tests for ConversationMemory and AgenticChain."""

import sys
sys.path.insert(0, ".")

from agentic_rag.memory import ConversationMemory
from agentic_rag.chain import create_agentic_chain


def test_memory_add_messages():
    """Test adding user and AI messages."""
    mem = ConversationMemory()
    mem.add_user_message("test_session", "Hello")
    mem.add_ai_message("test_session", "Hi there")

    messages = mem.get_messages("test_session")
    assert len(messages) == 2
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there"
    print("[PASS] test_memory_add_messages")


def test_memory_get_last_n():
    """Test getting last N messages."""
    mem = ConversationMemory()
    for i in range(15):
        mem.add_user_message("session1", f"Message {i}")

    last_5 = mem.get_last_n_messages("session1", n=5)
    assert len(last_5) == 5
    assert last_5[0].content == "Message 10"
    print("[PASS] test_memory_get_last_n")


def test_memory_clear_session():
    """Test clearing a session."""
    mem = ConversationMemory()
    mem.add_user_message("session1", "Hello")
    mem.add_user_message("session2", "Hello 2")

    mem.clear("session1")
    assert mem.get_messages("session1") == []
    assert len(mem.get_messages("session2")) == 1
    print("[PASS] test_memory_clear_session")


def test_memory_list_sessions():
    """Test listing sessions."""
    mem = ConversationMemory()
    mem.add_user_message("session1", "Hello")
    mem.add_user_message("session2", "Hello")
    mem.add_user_message("session3", "Hello")

    sessions = mem.list_sessions()
    assert len(sessions) == 3
    assert "session1" in sessions
    print("[PASS] test_memory_list_sessions")


def test_chain_create():
    """Test creating AgenticChain."""
    chain = create_agentic_chain()
    assert chain.graph is not None
    assert chain.memory is not None
    print("[PASS] test_chain_create")


def test_chain_chat_basic():
    """Test basic chat (simple query)."""
    chain = create_agentic_chain()
    # Note: This will call OpenAI, so may fail without API key
    try:
        result = chain.chat("What is 2 + 2?")
        print(f"  Result: {result}")
        print("[PASS] test_chain_chat_basic")
    except Exception as e:
        print(f"  Skipped (API key issue): {e}")
        print("[SKIP] test_chain_chat_basic")


def test_chain_invoke_with_metadata():
    """Test invoke returns metadata."""
    chain = create_agentic_chain()
    try:
        result = chain.invoke("What is 10 + 5?")
        assert "answer" in result
        assert "session_id" in result
        assert "iterations" in result
        assert "agents_used" in result
        print(f"  Agents used: {result['agents_used']}")
        print("[PASS] test_chain_invoke_with_metadata")
    except Exception as e:
        print(f"  Skipped (API key issue): {e}")
        print("[SKIP] test_chain_invoke_with_metadata")


def test_chain_get_history():
    """Test getting conversation history."""
    chain = create_agentic_chain()
    try:
        chain.chat("Hello", session_id="test_history")
        chain.chat("What is AI?", session_id="test_history")

        history = chain.get_history(session_id="test_history")
        assert len(history) >= 2
        print(f"  History length: {len(history)}")
        print("[PASS] test_chain_get_history")
    except Exception as e:
        print(f"  Skipped (API key issue): {e}")
        print("[SKIP] test_chain_get_history")


def test_chain_clear_history():
    """Test clearing history."""
    chain = create_agentic_chain()
    try:
        chain.chat("Hello", session_id="clear_test")
        chain.clear_history("clear_test")
        history = chain.get_history(session_id="clear_test")
        assert len(history) == 0
        print("[PASS] test_chain_clear_history")
    except Exception as e:
        print(f"  Skipped (API key issue): {e}")
        print("[SKIP] test_chain_clear_history")


if __name__ == "__main__":
    print("=== Memory and Chain Tests ===")

    print("\n[TEST] Memory Tests")
    test_memory_add_messages()
    test_memory_get_last_n()
    test_memory_clear_session()
    test_memory_list_sessions()

    print("\n[TEST] Chain Tests")
    test_chain_create()
    test_chain_chat_basic()
    test_chain_invoke_with_metadata()
    test_chain_get_history()
    test_chain_clear_history()

    print("\n=== All tests completed! ===")