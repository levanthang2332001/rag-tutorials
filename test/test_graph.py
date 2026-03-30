"""Test LangGraph workflow."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_rag.graph import create_agent_graph


def test_calculator():
    """Test calculator agent routing."""
    print("Starting calculator test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "What is 15 * 23?",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: What is 15 * 23?")
    print(f"Answer: {result.get('answer', 'No answer')}")
    print(f"Iterations: {result.get('iterations', 0)}")
    print()
    assert result.get("answer"), "Should have an answer"


def test_wikipedia():
    """Test wikipedia agent routing."""
    print("Starting wikipedia test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "What is Python programming language?",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: What is Python?")
    print(f"Answer: {result.get('answer', 'No answer')}")
    print(f"Iterations: {result.get('iterations', 0)}")
    print()
    assert result.get("answer"), "Should have an answer"


def test_multi_agent():
    """Test multiple agents coordination."""
    print("Starting multi-agent test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "Compare Python vs JavaScript",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: Compare Python vs JavaScript")
    print(f"Answer: {result.get('answer', 'No answer')}")
    print(f"Iterations: {result.get('iterations', 0)}")
    print()
    assert result.get("answer"), "Should have an answer"


def test_addition():
    """Test simple addition."""
    print("Starting addition test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "What is 100 + 250?",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: What is 100 + 250?")
    print(f"Answer: {result.get('answer', 'No answer')}")
    assert "350" in result.get("answer", ""), "Should contain 350"
    print("[PASS] Addition test passed\n")


def test_power():
    """Test power calculation."""
    print("Starting power test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "Calculate 2 to the power of 10",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: Calculate 2 to the power of 10")
    print(f"Answer: {result.get('answer', 'No answer')}")
    assert "1024" in result.get("answer", ""), "Should contain 1024"
    print("[PASS] Power test passed\n")


def test_wikipedia_AI():
    """Test Wikipedia agent for AI topic."""
    print("Starting Wikipedia AI test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "Who is the founder of Microsoft?",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: Who is the founder of Microsoft?")
    print(f"Answer: {result.get('answer', 'No answer')}")
    assert result.get("answer"), "Should have an answer"
    print("[PASS] Wikipedia AI test passed\n")


def test_wikipedia_country():
    """Test Wikipedia agent for country info."""
    print("Starting Wikipedia country test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "What is the capital of France?",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: What is the capital of France?")
    print(f"Answer: {result.get('answer', 'No answer')}")
    assert "Paris" in result.get("answer", ""), "Should mention Paris"
    print("[PASS] Wikipedia country test passed\n")


def test_complex_query():
    """Test complex multi-agent query."""
    print("Starting complex query test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "What is the square root of 144 and who founded Google?",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: What is sqrt(144) and who founded Google?")
    print(f"Answer: {result.get('answer', 'No answer')}")
    assert result.get("answer"), "Should have an answer"
    print("[PASS] Complex query test passed\n")


def test_subtraction():
    """Test subtraction."""
    print("Starting subtraction test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "What is 500 minus 123?",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: What is 500 minus 123?")
    print(f"Answer: {result.get('answer', 'No answer')}")
    assert "377" in result.get("answer", ""), "Should contain 377"
    print("[PASS] Subtraction test passed\n")


def test_division():
    """Test division."""
    print("Starting division test...")
    graph = create_agent_graph()
    result = graph.invoke({
        "query": "What is 100 divided by 4?",
        "selected_agents": [],
        "agent_results": {},
        "answer": "",
        "iterations": 0,
        "missing_info": False,
        "messages": []
    })
    print("Query: What is 100 divided by 4?")
    print(f"Answer: {result.get('answer', 'No answer')}")
    assert "25" in result.get("answer", ""), "Should contain 25"
    print("[PASS] Division test passed\n")


if __name__ == "__main__":
    print("=== Running Calculator Tests ===")
    test_calculator()
    test_addition()
    test_power()
    test_subtraction()
    test_division()

    print("=== Running Wikipedia Tests ===")
    test_wikipedia()
    test_wikipedia_AI()
    test_wikipedia_country()

    print("=== Running Multi-Agent Tests ===")
    test_multi_agent()
    test_complex_query()

    print("\n[SUCCESS] All tests passed!")