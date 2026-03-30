---
name: ""
overview: ""
todos: []
isProject: false
---

# Plan 5: LangSmith Observability

## Overview

Setup LangSmith for tracing, debugging, and evaluation of the multi-agent system.

## LangSmith Setup

### File: `agentic_rag/__init__.py`

```python
"""Agentic RAG package - LangSmith initialization."""

import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "agentic-rag")

# Verify setup
if os.environ.get("LANGSMITH_API_KEY"):
    print(f"LangSmith tracing enabled for project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    print("Warning: LANGSMITH_API_KEY not set. Tracing disabled.")
```

### Environment Variables

**File**: `.env`

```bash
# LangSmith (get from https://smith.langchain.com/)
LANGSMITH_API_KEY=ls_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag
```

## Tracing Decorators

All agents already have `@trace` decorators applied:

```python
from langsmith import trace

@trace(name="pdf_agent", tags=["agent", "pdf"])
def run_pdf_agent(query: str) -> str:
    # ... traced automatically
```

## LangSmith Evaluation

### File: `agentic_rag/eval.py`

```python
"""LangSmith evaluation setup for Agentic RAG."""

import os
from typing import Any
from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI

client = Client()

# =============================================================================
# EVALUATORS
# =============================================================================

def faithfulness_evaluator(run: Any, example: Any) -> dict:
    """Evaluate if answer is faithful to retrieved context.

    Uses LLM-as-judge to check if answer matches retrieved information.
    """
    prediction = run.outputs.get("answer", "")
    inputs = run.inputs

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
        import json
        parsed = json.loads(response.content)
        return {
            "key": "faithfulness",
            "score": parsed.get("score", 3) / 5,  # Normalize to 0-1
            "reasoning": parsed.get("reasoning", ""),
        }
    except:
        return {"key": "faithfulness", "score": 0.5}


def answer_relevancy_evaluator(run: Any, example: Any) -> dict:
    """Evaluate if answer is relevant to the question."""
    prediction = run.outputs.get("answer", "")
    inputs = run.inputs

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
        import json
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

    # Check which agents were actually used
    actual_tools = list(prediction.get("agent_results", {}).keys())

    # Calculate precision
    if not expected_tools:
        return {"key": "tool_selection", "score": 1.0}

    correct = len(set(expected_tools) & set(actual_tools))
    total = len(expected_tools)

    return {
        "key": "tool_selection",
        "score": correct / total if total > 0 else 0,
        "reasoning": f"Expected: {expected_tools}, Actual: {actual_tools}",
    }


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

@evaluate(
    dataset_name="agentic-rag-eval",
    evaluators=[
        faithfulness_evaluator,
        answer_relevancy_evaluator,
        tool_selection_evaluator,
    ],
    client=client,
    metadata={"version": "1.0"},
)
def run_evaluation(inputs: dict) -> dict:
    """Run evaluation on a single input."""
    from agentic_rag.chain import AgenticChain

    chain = AgenticChain()
    result = chain.invoke(inputs["query"])
    return {"answer": result["answer"]}


# =============================================================================
# DATASET MANAGEMENT
# =============================================================================

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
        name="agentic-rag-eval",
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


# =============================================================================
# USAGE
# =============================================================================

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
```

## LangSmith Dashboard Features

| Feature         | Description                                 |
| --------------- | ------------------------------------------- |
| **Traces**      | See every agent call with inputs/outputs    |
| **Spans**       | Nested view of operations within each agent |
| **Datasets**    | Q&A pairs for regression testing            |
| **Evaluations** | Automated scoring with custom metrics       |
| **Feedback**    | Manual human feedback on responses          |

## Next Steps

After LangSmith setup, proceed to: [6-memory-chain.plan.md](./6-memory-chain.plan.md)
