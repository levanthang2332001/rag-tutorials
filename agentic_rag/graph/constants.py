"""Prompts and static routing configuration for the agent graph."""

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
