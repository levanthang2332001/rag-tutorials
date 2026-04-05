"""Maps agent names to runner callables."""

from ..agents import (
    run_pdf_agent,
    run_web_agent,
    run_calculator_agent,
    run_wikipedia_agent,
    run_code_agent,
    run_sql_agent,
)

AGENT_REGISTRY: dict[str, callable] = {
    "pdf_agent": run_pdf_agent,
    "web_agent": run_web_agent,
    "calculator_agent": run_calculator_agent,
    "wikipedia_agent": run_wikipedia_agent,
    "code_agent": run_code_agent,
    "sql_agent": run_sql_agent,
}
