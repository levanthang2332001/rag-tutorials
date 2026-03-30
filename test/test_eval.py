"""Test eval module - LangSmith evaluators."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_rag.eval import (
    faithfulness_evaluator,
    answer_relevancy_evaluator,
    tool_selection_evaluator,
)


class MockRun:
    """Mock LangSmith run object."""
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


class MockExample:
    """Mock LangSmith example object."""
    pass


def test_faithfulness_high():
    """Test faithfulness evaluator with faithful answer."""
    run = MockRun(
        inputs={"query": "What is the capital of France?"},
        outputs={"answer": "The capital of France is Paris."}
    )
    example = MockExample()
    result = faithfulness_evaluator(run, example)
    print(f"[TEST] faithfulness_evaluator high score: {result}")
    assert result["key"] == "faithfulness"
    assert 0.6 <= result["score"] <= 1.0
    print("[PASS] Faithfulness high score test passed\n")


def test_faithfulness_low():
    """Test faithfulness evaluator with contradictory answer."""
    run = MockRun(
        inputs={"query": "What is the capital of France?"},
        outputs={"answer": "The capital of France is London."}
    )
    example = MockExample()
    result = faithfulness_evaluator(run, example)
    print(f"[TEST] faithfulness_evaluator low score: {result}")
    assert result["key"] == "faithfulness"
    assert 0.0 <= result["score"] <= 0.6
    print("[PASS] Faithfulness low score test passed\n")


def test_answer_relevancy_relevant():
    """Test answer relevancy with relevant answer."""
    run = MockRun(
        inputs={"query": "What is Python?"},
        outputs={"answer": "Python is a programming language."}
    )
    example = MockExample()
    result = answer_relevancy_evaluator(run, example)
    print(f"[TEST] answer_relevancy_evaluator relevant: {result}")
    assert result["key"] == "answer_relevancy"
    assert 0.6 <= result["score"] <= 1.0
    print("[PASS] Answer relevancy relevant test passed\n")


def test_answer_relevancy_irrelevant():
    """Test answer relevancy with irrelevant answer."""
    run = MockRun(
        inputs={"query": "What is Python?"},
        outputs={"answer": "The weather is sunny today."}
    )
    example = MockExample()
    result = answer_relevancy_evaluator(run, example)
    print(f"[TEST] answer_relevancy_evaluator irrelevant: {result}")
    assert result["key"] == "answer_relevancy"
    assert 0.0 <= result["score"] <= 0.4
    print("[PASS] Answer relevancy irrelevant test passed\n")


def test_tool_selection_correct():
    """Test tool selection with correct tools."""
    run = MockRun(
        inputs={"query": "What is RAG?", "expected_tools": ["wikipedia_agent"]},
        outputs={"agent_results": {"wikipedia_agent": {}}}
    )
    example = MockExample()
    result = tool_selection_evaluator(run, example)
    print(f"[TEST] tool_selection_evaluator correct: {result}")
    assert result["key"] == "tool_selection"
    assert result["score"] == 1.0
    print("[PASS] Tool selection correct test passed\n")


def test_tool_selection_partial():
    """Test tool selection with partial match."""
    run = MockRun(
        inputs={"query": "Complex query", "expected_tools": ["wikipedia_agent", "calculator_agent"]},
        outputs={"agent_results": {"wikipedia_agent": {}}}
    )
    example = MockExample()
    result = tool_selection_evaluator(run, example)
    print(f"[TEST] tool_selection_evaluator partial: {result}")
    assert result["key"] == "tool_selection"
    assert result["score"] == 0.5
    print("[PASS] Tool selection partial test passed\n")


def test_tool_selection_incorrect():
    """Test tool selection with incorrect tools."""
    run = MockRun(
        inputs={"query": "Math query", "expected_tools": ["calculator_agent"]},
        outputs={"agent_results": {"wikipedia_agent": {}}}
    )
    example = MockExample()
    result = tool_selection_evaluator(run, example)
    print(f"[TEST] tool_selection_evaluator incorrect: {result}")
    assert result["key"] == "tool_selection"
    assert result["score"] == 0.0
    print("[PASS] Tool selection incorrect test passed\n")


def test_tool_selection_empty_expected():
    """Test tool selection with no expected tools."""
    run = MockRun(
        inputs={"query": "Any query", "expected_tools": []},
        outputs={"agent_results": {}}
    )
    example = MockExample()
    result = tool_selection_evaluator(run, example)
    print(f"[TEST] tool_selection_evaluator empty expected: {result}")
    assert result["key"] == "tool_selection"
    assert result["score"] == 1.0
    print("[PASS] Tool selection empty expected test passed\n")


if __name__ == "__main__":
    print("=== Running Eval Module Tests ===\n")

    print("[TEST] Faithfulness Evaluator Tests")
    test_faithfulness_high()
    test_faithfulness_low()

    print("[TEST] Answer Relevancy Evaluator Tests")
    test_answer_relevancy_relevant()
    test_answer_relevancy_irrelevant()

    print("[TEST] Tool Selection Evaluator Tests")
    test_tool_selection_correct()
    test_tool_selection_partial()
    test_tool_selection_incorrect()
    test_tool_selection_empty_expected()

    print("[SUCCESS] All eval tests passed!")
