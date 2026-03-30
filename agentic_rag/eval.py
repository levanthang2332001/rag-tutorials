import os
import json
from typing import Any
from langsmith import Client, evaluate
from core.openai.openai import get_llm
from agentic_rag.graph import create_agent_graph


client = Client()
llm = get_llm()

def faithfulness_evaluator(run: Any, example: Any) -> dict:
  """Evaluate if answer is faithful to retrieved context."""
  prediction = run.outputs.get("answer", "")
  inputs = run.inputs

  eval_prompt = f"""Evaluate if the answer is faithful to the retrieved context.

  Question: {inputs.get('query', '')}

  Answer: {prediction}

  Is the answer based on the retrieved information? Score 1-5:
  1 = Answer contradicts information
  3 = Answer partially correct
  5 = Answer fully faithful

  Respond with JSON: {{"score": X, "reasoning": "..."}}"""

  response = llm.invoke(eval_prompt)

  try: 
    parsed = json.loads(response.content)
    return {
      "key": "faithfulness",
      "score": parsed.get("score", 3) / 5,
      "reasoning": parsed.get("reasoning", ""),
    }
  except:
    return {"key": "faithfulness", "score": 0.5}


def answer_relevancy_evaluator(run: Any, example: Any) -> dict:
  """Evaluate if answer is relevant to the question."""
  prediction = run.outputs.get("answer", "")
  inputs = run.inputs

  eval_prompt = f"""Evaluate if the answer is relevant to the question.

  Question: {inputs.get('query', '')}

  Answer: {prediction}

  Score 1-5:
  1 = Completely off-topic
  3 = Partially relevant
  5 = Fully relevant

  Respond with JSON: {{"score": X, "reasoning": "..."}}"""

  response = llm.invoke(eval_prompt)

  try:
    parsed = json.loads(response.content)
    return {
      "key": "answer_relevancy",
      "score": parsed.get("score", 3) / 5,
      "reasoning": parsed.get("reasoning", ""),
    }
  except:
    return {"key": "answer_relevancy", "score": 0.5}

def tool_selection_evaluator(run: Any, example: Any) -> dict:
  """Evaluate if correct tools were selected."""
  prediction = run.outputs or {}
  inputs = run.inputs
  expected_tools = inputs.get("expected_tools", [])

  actual_tools = list(prediction.get("agent_results", {}).keys())

  if not expected_tools:
    return {"key": "tool_selection", "score": 1.0}

  correct = len(set(expected_tools) & set(actual_tools))
  total = len(expected_tools)
  return {
    "key": "tool_selection",
    "score": correct / total if total > 0 else 0,
    "reasoning": f"Expected: {expected_tools}, Actual: {actual_tools}",
  }

# @evaluate(
#     dataset_name="agentic-rag-eval",
#     evaluators=[
#         faithfulness_evaluator,
#         answer_relevancy_evaluator,
#         tool_selection_evaluator,
#     ],
#     client=client,
#     metadata={"version": "1.0"},
# )
def run_evaluation(inputs: dict) -> dict:
  """Run evaluation on a single input."""
  from agentic_rag.graph import create_agent_graph
  graph = create_agent_graph()
  result = graph.invoke({
      "query": inputs["query"],
      "selected_agents": [],
      "agent_results": {},
      "answer": "",
      "iterations": 0,
      "missing_info": False,
      "messages": []
  })
  return {"answer": result.get("answer", "")}

def create_eval_dataset():
    """Create evaluation dataset with test cases."""

    examples = [
        {
            "query": "What is RAG?",
            "expected_tools": ["wikipedia_agent"],
        },
        {
            "query": "Calculate 2 to the power of 10",
            "expected_tools": ["calculator_agent"],
        },
        {
            "query": "What does our company policy say about AI usage?",
            "expected_tools": ["pdf_agent"],
        },
        {
            "query": "What is the latest news about AI?",
            "expected_tools": ["web_agent"],
        },
        {
            "query": "Generate 5 random numbers and show their sum",
            "expected_tools": ["code_agent"],
        },
    ]

    dataset = client.create_dataset(
        dataset_name="agentic-rag-eval",
        description="Evaluation dataset for Agentic RAG multi-agent system",
    )

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"query": ex["query"], "expected_tools": ex["expected_tools"]} for ex in examples],
    )

    print(f"Created dataset with {len(examples)} examples")
    return dataset


def list_datasets():
    """List all datasets in LangSmith."""
    datasets = client.list_datasets()
    for ds in datasets:
        print(f"- {ds.name}: {ds.description}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-dataset", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.create_dataset:
        create_eval_dataset()
    elif args.run_eval:
        run_evaluation()
    elif args.list:
        list_datasets()
    else:
        print("Usage: python eval.py [--create-dataset|--run-eval|--list]")