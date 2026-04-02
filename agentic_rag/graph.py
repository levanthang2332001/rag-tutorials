"""LangGraph multi-agent workflow with parallel agent execution."""

import json
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI

from .state import AgentState, AgentInvokeState
from .agents import (
    run_pdf_agent,
    run_web_agent,
    run_calculator_agent,
    run_wikipedia_agent,
    run_code_agent,
    run_sql_agent,
)

from core.openai.openai import get_llm

AGENT_REGISTRY: dict[str, callable] = {
    "pdf_agent": run_pdf_agent,
    "web_agent": run_web_agent,
    "calculator_agent": run_calculator_agent,
    "wikipedia_agent": run_wikipedia_agent,
    "code_agent": run_code_agent,
    "sql_agent": run_sql_agent,
}

AGENT_DESCRIPTIONS = {
    "pdf_agent": "Find information in PDF documents",
    "web_agent": "Find information on the web",
    "calculator_agent": "Perform calculations",
    "wikipedia_agent": "Find information on Wikipedia",
    "code_agent": "Execute code to get information",
    "sql_agent": "Query a database for information",
}

AGENT_TASK_TYPES: dict[str, str] = {
    "pdf_agent": "pdf",
    "web_agent": "web",
    "calculator_agent": "calculation",
    "wikipedia_agent": "wikipedia",
    "code_agent": "coding",
    "sql_agent": "sql",
}

WEB_REQUIRED_KEYWORDS = [
    "latest",
    "news",
    "current",
    "breaking",
    "today",
    "weather",
    "stock",
    "price",
    "real-time",
]

ROUTER_MATH_KEYWORDS = ["square root", "sqrt", "power", "calculate", "divided by"]
ROUTER_FACT_KEYWORDS = ["who", "founded", "capital", "company", "google", "microsoft"]

AGENT_INSTRUCTIONS: dict[str, str] = {
    "pdf_agent": (
        "Task: Use pdf_retriever to extract the most relevant passages from our internal PDFs "
        "that support answering the original query. Return an Answer and Sources."
    ),
    "web_agent": (
        "Task: Use web_search to find up-to-date sources relevant to the original query. "
        "Return an Answer and Sources."
    ),
    "wikipedia_agent": (
        "Task: Use wikipedia_search for a factual background relevant to the original query. "
        "Return an Answer and Sources."
    ),
    "calculator_agent": (
        "Task: Use calculator to compute the required values. Show the expression and result."
    ),
    "code_agent": (
        "Task: If runnable code execution is needed, use code_executor; otherwise provide code snippets "
        "and explanations. Focus on correctness."
    ),
    "sql_agent": (
        "Task: If needed, use sql_database (read-only SELECT) to fetch relevant data. "
        "Then summarize the results."
    ),
}

MAX_ITERATIONS = 2
MIN_ANSWER_LENGTH = 80
CONFIDENCE_THRESHOLD = 0.75

ROUTING_PROMPT_TEMPLATE = """You are a router for a multi-agent RAG system.
Analyze the query and select the appropriate agents.

Query: {query}

Available agents:
{agent_list}

Previous attempt (if any): {previous_attempt}

Rules:
- Only select agents that are truly necessary, do not over-select
- If it's a math question: only calculator_agent
- If asking about general knowledge: only wikipedia_agent
- If asking about content from papers/documents/PDFs OR mentioning "paper", "document", "pdf", "the paper says": use pdf_agent
- If asking about data in database/tables/SQL: use sql_agent
- If asking about code/programming: use code_agent
- If needing current news/web info: use web_agent
- If needing multiple sources: select max 3 agents
- If previous attempt lacked information: select additional agents

Return only valid JSON (no explanation):
{{"agents": ["agent1", "agent2"], "reasoning": "brief reason"}}"""


def orchestrator_node(state: AgentState) -> dict:
    """Route query to appropriate agents - stores selected agents in state."""
    query = state["query"]
    current_request = _extract_current_request(query)
    iterations = state.get("iterations", 0)
    previous_results = state.get("agent_results", {})

    previous_attempt = "No previous attempt" if not previous_results else (
        f"Used: {list(previous_results.keys())} - missing information"
    )

    agent_list = "\n".join(
        f"- {name}: {desc}"
        for name, desc in AGENT_DESCRIPTIONS.items()
    )

    llm = get_llm()

    routing_prompt = ROUTING_PROMPT_TEMPLATE.format(
        query=current_request,
        agent_list=agent_list,
        previous_attempt=previous_attempt,
    )

    def _build_sub_questions(selected_agents: list[str]) -> list[dict]:
        sub_questions: list[dict] = []
        for agent_name in selected_agents:
            task_type = AGENT_TASK_TYPES.get(agent_name, agent_name)
            instruction_tail = AGENT_INSTRUCTIONS.get(agent_name, "")
            instruction = (
                f"Original query: {current_request}\n{instruction_tail}".strip()
                if instruction_tail
                else current_request
            )
            sub_questions.append(
                {
                    "id": f"{task_type}:{agent_name}",
                    "task_type": task_type,
                    "agent_name": agent_name,
                    "text": instruction,
                }
            )
        return sub_questions

    try:
        response = llm.invoke(routing_prompt)
    except Exception:
        # Router LLM timeout/failure fallback: use lightweight heuristics.
        query_lower = current_request.lower()
        selected: list[str] = []

        if _is_strict_code_query(current_request) or _looks_like_code_query(current_request):
            selected = ["code_agent"]
        else:
            is_math = any(k in query_lower for k in ["square root", "sqrt", "power", "calculate", "divided by"])
            is_factual = any(k in query_lower for k in ["who", "founded", "capital", "company", "google", "microsoft"])

            if is_math:
                selected.append("calculator_agent")
            if is_factual:
                selected.append("wikipedia_agent")
            if not selected:
                selected = ["wikipedia_agent"]

        selected = selected[:3]
        sub_questions = _build_sub_questions(selected)
        return {
            "selected_agents": selected,
            "sub_questions": sub_questions,
            "iterations": iterations,
            "query": current_request,
            "needed_agents": [],
        }

    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1]
        if content.endswith("```"):
            content = content.split("```")[0]
        parsed = json.loads(content.strip())
        selected = [a for a in parsed.get("agents", []) if a in AGENT_REGISTRY]
    except (json.JSONDecodeError, KeyError, IndexError):
        selected = ["wikipedia_agent"]

    if not selected:
        selected = ["wikipedia_agent"]

    # Deterministic routing for coding intents - replace selection with code_agent only.
    if _is_strict_code_query(current_request) or _looks_like_code_query(current_request):
        selected = ["code_agent"]

    # Avoid slow/fragile web browsing unless the query clearly needs
    # time-sensitive/current information (reduces hangs).
    query_lower = current_request.lower()
    if not any(k in query_lower for k in WEB_REQUIRED_KEYWORDS):
        selected = [a for a in selected if a != "web_agent"]
        if not selected:
            selected = ["wikipedia_agent"]

    needed_agents = state.get("needed_agents", []) or []
    for agent_name in needed_agents:
        if agent_name in AGENT_REGISTRY and agent_name not in selected:
            selected.append(agent_name)

    selected = selected[:3]

    sub_questions = _build_sub_questions(selected)

    return {
        "selected_agents": selected,
        "sub_questions": sub_questions,
        "iterations": iterations,
        "query": current_request,
        "needed_agents": [],
    }


def _extract_current_request(query: str) -> str:
    """Extract latest user request from contextual query payload."""
    marker = ">>> CURRENT USER REQUEST (needs answer) <<<"
    if marker not in query:
        return query
    extracted = query.split(marker, maxsplit=1)[1].strip()
    return extracted or query


def _looks_like_code_query(query: str) -> bool:
    """Heuristic detection for programming/code related requests."""
    query_lower = query.lower()
    code_keywords = [
        "code",
        "python",
        "javascript",
        "typescript",
        "java",
        "c++",
        "c#",
        "function",
        "class",
        "bug",
        "debug",
        "fix",
        "implement",
        "algorithm",
        "sql query",
        "script",
        "refactor",
        "stack trace",
        "exception",
        "error in code",
    ]
    return any(keyword in query_lower for keyword in code_keywords)


def _is_strict_code_query(query: str) -> bool:
    """Detect requests that should be handled only by code_agent."""
    query_lower = query.lower()
    strict_markers = [
        "refactor",
        "rewrite code",
        "write code",
        "implement",
        "debug",
        "fix this code",
        "optimize this code",
        "code review",
        "explain this code",
        "def ",
        "class ",
        "stack trace",
    ]
    non_code_markers = [
        "news",
        "wikipedia",
        "latest",
        "paper",
        "pdf",
        "database",
        "sql",
    ]
    is_code = any(marker in query_lower for marker in strict_markers)
    has_non_code = any(marker in query_lower for marker in non_code_markers)
    return is_code and not has_non_code


def route_to_agents(state: AgentState) -> list[Send]:
    """Fan-out to selected agents in parallel via Send()."""
    sub_questions = state.get("sub_questions", []) or []
    query = state["query"]
    selected = state.get("selected_agents", [])

    if sub_questions:
        return [
            Send(
                "run_agent",
                AgentInvokeState(query=sq["text"], agent_name=sq["agent_name"]),
            )
            for sq in sub_questions
        ]

    if not selected:
        selected = ["wikipedia_agent"]

    return [
        Send("run_agent", AgentInvokeState(query=query, agent_name=agent_name))
        for agent_name in selected
    ]


def run_agent_node(state: AgentInvokeState) -> dict:
    """Run the selected agent and return the result."""
    agent_name = state["agent_name"]
    query = state["query"]

    agent_func = AGENT_REGISTRY.get(agent_name)
    if not agent_func:
        return {"agent_results": {agent_name: f"Agent '{agent_name}' not found."}}

    try:
        result = agent_func(query)
    except Exception as e:
        result = f"{agent_name} error: {str(e)}"

    return {"agent_results": {agent_name: str(result)}}


SYNTHESIS_PROMPT_TEMPLATE = """You are a synthesizer for a multi-agent RAG system.
Synthesize information from agents and answer the query.

Original query: {query}
Attempt: {iteration}/{max_iter}

Results from agents:
{results_text}

Guidelines:
- Synthesize into a coherent, complete answer
- If agents have conflicting information, note the differences
- If important information is missing, note what's missing
- Answer in the same language as the query
- If the query is about coding/refactoring/debugging, include concrete code blocks
- Minimum length: 2-3 complete sentences

Output format (MUST FOLLOW):
Answer:
<final answer text>

Sources:
- <deduplicated source item #1>
- <deduplicated source item #2>

How to build Sources:
- Extract any "Sources:" blocks that appear inside agent outputs and merge them.
- If no agent provides a "Sources:" block, infer sources by looking for:
  - "Source URL:" / "URL:" lines from web outputs
  - "[DOC_SOURCE:" markers from PDF context
  - "Wikipedia" / "URL:" lines from Wikipedia outputs
"""


def synthesizer_node(state: AgentState) -> dict:
    """Synthesize results from all agents."""
    query = state["query"]
    results = state.get("agent_results", {})
    iterations = state.get("iterations", 0) + 1

    if not results:
        return {
            "answer": "Could not collect information from agents.",
            "iterations": iterations,
            "missing_info": True,
        }

    # For strict coding flows, return code_agent output directly to avoid dilution.
    if list(results.keys()) == ["code_agent"]:
        code_answer = str(results["code_agent"]).strip()
        return {
            "answer": code_answer,
            "iterations": iterations,
            "missing_info": _detect_missing_info(code_answer),
        }

    sub_questions = state.get("sub_questions", []) or []
    task_type_by_agent = {
        sq.get("agent_name", ""): sq.get("task_type", sq.get("agent_name", "") )
        for sq in sub_questions
    }

    results_text = "\n\n".join(
        f"[{task_type_by_agent.get(agent, agent).upper()} | {agent.upper()}]\n{result}"
        for agent, result in results.items()
    )

    llm = get_llm()

    try:
        response = llm.invoke(
            SYNTHESIS_PROMPT_TEMPLATE.format(
                query=query,
                iteration=iterations,
                max_iter=MAX_ITERATIONS,
                results_text=results_text,
            )
        )
        answer = response.content.strip()
    except Exception as e:
        # Fallback: if synthesis LLM times out, return a minimal merged view.
        answer = f"Could not synthesize answer reliably due to an internal error: {str(e)}\n\n"
        answer += "\n\n".join(
            f"[{k}]\n{str(v)[:1500]}" for k, v in (results or {}).items()
        )

    missing_info = _detect_missing_info(answer)

    return {
        "answer": answer,
        "iterations": iterations,
        "missing_info": missing_info,
    }


VERIFIER_PROMPT_TEMPLATE = """You are a claim-check verifier for a multi-agent RAG system.
You will be given:
1) the original query
2) the synthesized answer
3) raw outputs from specialist agents

Original query:
{query}

Synthesized answer:
{answer}

Raw outputs from specialist agents:
{results_text}

Task:
- Decide whether the synthesized answer is faithful to the specialist agent outputs.
- If the answer is missing key information, set missing_info=true.
- If there are contradictions between agent outputs, set conflicts=true.

Return ONLY valid JSON (no markdown):
{{
  "confidence": <number between 0 and 1>,
  "missing_info": <true|false>,
  "conflicts": <true|false>,
  "needed_agents": [<agent_name strings>] 
}}
Notes:
- needed_agents should list additional agents to run if missing_info or conflicts is true.
- If you don't need more agents, return [].
"""


def verifier_node(state: AgentState) -> dict:
    """Verify synthesized answer quality and decide whether to loop."""
    query = state.get("query", "")
    answer = state.get("answer", "") or ""
    agent_results: dict[str, str] = state.get("agent_results", {}) or {}

    if not agent_results:
        return {
            "confidence": 0.0,
            "missing_info": True,
            "conflicts": False,
            "needed_agents": [],
        }

    results_text = "\n\n".join(
        f"[{k}]\n{str(v)[:2000]}"
        for k, v in agent_results.items()
    )

    # Fast heuristic claim-check:
    # If the synthesizer already produced a structured answer and doesn't look
    # missing, we skip the extra verifier LLM call to keep latency low.
    has_sources = "Sources:" in answer
    missing_info = _detect_missing_info(answer)
    if has_sources and not missing_info:
        return {
            "confidence": 0.9,
            "missing_info": False,
            "conflicts": False,
            "needed_agents": [],
        }

    # If we likely miss info, suggest extra agents deterministically.
    query_lower = query.lower()
    needed_agents: list[str] = []
    is_math = any(k in query_lower for k in ["square root", "sqrt", "power", "calculate"])
    is_fact = any(
        k in query_lower
        for k in ["who", "founded", "capital", "company", "google", "microsoft"]
    )
    if is_math:
        needed_agents.append("calculator_agent")
    if is_fact:
        needed_agents.append("wikipedia_agent")

    # Only call LLM verifier when heuristic isn't conclusive.
    llm = get_llm()
    try:
        response = llm.invoke(
            VERIFIER_PROMPT_TEMPLATE.format(
                query=query,
                answer=answer,
                results_text=results_text,
            )
        )
    except Exception:
        # Timeout/LLM failure fallback: rely on heuristics.
        return {
            "confidence": 0.4,
            "missing_info": missing_info,
            "conflicts": False,
            "needed_agents": needed_agents,
        }

    raw = response.content.strip()
    if raw.startswith("```json"):
        raw = raw.split("```json", 1)[1]
    if raw.endswith("```"):
        raw = raw.split("```", 1)[0]

    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "confidence": state.get("confidence", 0.0),
            "missing_info": state.get("missing_info", False),
            "conflicts": state.get("conflicts", False),
            "needed_agents": state.get("needed_agents", []),
        }

    return {
        "confidence": float(parsed.get("confidence", state.get("confidence", 0.0)) or 0.0),
        "missing_info": bool(parsed.get("missing_info", state.get("missing_info", False))),
        "conflicts": bool(parsed.get("conflicts", state.get("conflicts", False))),
        "needed_agents": parsed.get("needed_agents", []) or [],
    }


def _detect_missing_info(answer: str) -> bool:
    """Check if answer is missing information."""
    if len(answer) < MIN_ANSWER_LENGTH:
        return True

    insufficient_signals = [
        "not found", "no information", "need more information",
        "cannot answer", "insufficient data",
        "no relevant", "could not find",
    ]
    answer_lower = answer.lower()
    return any(signal in answer_lower for signal in insufficient_signals)


def should_continue(state: AgentState) -> Literal["orchestrator", "__end__"]:
    """Decide whether to loop or end based on verifier outputs."""
    iterations = state.get("iterations", 0)
    missing_info = state.get("missing_info", False)
    conflicts = state.get("conflicts", False)
    confidence = state.get("confidence", 0.0)

    if iterations >= MAX_ITERATIONS:
        return END

    if conflicts:
        return "orchestrator"

    if missing_info:
        return "orchestrator"

    if confidence < CONFIDENCE_THRESHOLD:
        return "orchestrator"

    return END


def create_agent_graph():
    """Build LangGraph with Send() parallel execution."""
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("run_agent", run_agent_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("verifier", verifier_node)

    builder.set_entry_point("orchestrator")

    # After orchestrator, fan-out to agents in parallel
    builder.add_conditional_edges(
        "orchestrator",
        route_to_agents,
    )

    # After run_agent, go to synthesizer
    builder.add_edge("run_agent", "synthesizer")

    # After synthesizer, verify claim-check quality
    builder.add_edge("synthesizer", "verifier")

    # After verifier, either loop back or end
    builder.add_conditional_edges(
        "verifier",
        should_continue,
        {"orchestrator": "orchestrator", END: END},
    )

    return builder.compile()