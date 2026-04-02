---
name: agent_results_citations
overview: Update the LangGraph multi-agent chatbot to return per-agent results, standardize citations/Sources, speed up PDF retrieval with caching, and add verifier + query decomposition + optional streaming.
todos:
  - id: t1-agent_results-api-ui
    content: Trả về `agent_results` trong `agentic_rag/chain.py` + mở rộng `api/agentic_api.py` + render `st.expander` theo từng agent trong `agentic_rag/streamlit_app.py`.
    status: completed
  - id: t2-citations-sources-format
    content: Chuẩn hóa format `Answer:` + `Sources:` trong `core/system_prompt.py` (pdf/web/wikipedia) và cập nhật `agentic_rag/graph.py` synthesizer prompt để luôn tạo/merge section `Sources:`.
    status: completed
  - id: t3-tool-output-normalize
    content: "Sửa `agentic_rag/tools.py` để tool outputs có marker thống nhất: `Source URL:` cho web/wikipedia; pdf context đã có marker nguồn; đảm bảo agent dễ trích vào `Sources:`."
    status: completed
  - id: t4-pdf-cache-lazy
    content: Thêm cache lazy cho `pdf_retriever` (FAISS retriever/index) trong `agentic_rag/tools.py`, tránh rebuild mỗi lần và cân nhắc lock đơn giản.
    status: completed
  - id: t5-verifier-node
    content: Thêm `verifier_node` và update `should_continue()` trong `agentic_rag/graph.py` để dùng confidence/missing/conflicts thay cho heuristic độ dài/keyword.
    status: completed
  - id: t6-query-decomposition
    content: Nâng orchestrator từ `selected_agents` -> `sub_questions`; sửa `agentic_rag/state.py`, `orchestrator_node`, `route_to_agents`, và synthesizer để tổng hợp theo task_type.
    status: completed
  - id: t7-streaming-endpoint-optional
    content: Thêm endpoint `POST /chat/stream` SSE trong `api/agentic_api.py` và (nếu muốn) cập nhật `agentic_rag/streamlit_app.py` để hiển thị tiến trình theo event.
    status: completed
  - id: t8-tests-update
    content: Chạy lại tests hiện có; bổ sung test cho format `Sources` và cache reuse của `pdf_retriever`.
    status: completed
isProject: false
---

## Goals

- UI/API hiển thị rõ “đa luồng thật” bằng cách trả về và render `agent_results` theo từng agent.
- Chuẩn hóa cách trình bày trích dẫn/`Sources` để giải quyết inconsistency giữa `system_prompt` và tool output.
- Tối ưu `pdf_retriever` bằng cache lazy để giảm latency.
- Thêm verifier/claim-check để tăng độ tin cậy (confidence + loop khi thiếu/mâu thuẫn).
- Nâng orchestrator từ “chọn agent set” sang “tách sub-questions” và fan-out theo sub-task.
- (Tùy chọn trong lượt này) thêm endpoint streaming `/chat/stream` để UI thấy progress theo thời gian thực.

## Data flow (high level)

```mermaid
flowchart LR
  U[User question + session_id] --> Orchestrator[orchestrator_node]
  Orchestrator --> FanOut[route_to_subtasks (Send())]
  FanOut --> Agents[Specialist agents run in parallel]
  Agents --> Synth[Synthesizer (combine + format)]
  Synth --> Verify[Verifier/claim-check]
  Verify -->|confidence OK| End[END]
  Verify -->|missing/conflict| Orchestrator
```

## Implementation plan

### 1) Return `agent_results` end-to-end (điểm nhấn ngay)

- Update response model:
  - `api/agentic_api.py`: thêm trường `agent_results: dict[str, str]` vào `ChatResponse`.
- Update chain:
  - `agentic_rag/chain.py`: trong `invoke()`, đưa `result.get("agent_results", {})` vào response.
- Update UI:
  - `agentic_rag/streamlit_app.py`: dùng `result["agent_results"]` để render mỗi agent trong `st.expander(f"{agent_name}")`.

### 2) Chuẩn hóa citations/`Sources` giữa tool và prompt

- Chuẩn hóa format output của specialist agent (không phụ thuộc tool raw):
  - `core/system_prompt.py`:
    - Với `pdf/web/wikipedia`, thêm “Final answer format” bắt buộc gồm 2 phần: `Answer:` và `Sources:`.
    - Mỗi bullet trong `Sources:` phải có ít nhất `type` (PDF/Web/Wikipedia) + `identifier` (file/page hoặc URL/article).
- Chuẩn hóa tool output để agent dễ trích:
  - `agentic_rag/tools.py`:
    - `wikipedia_search`: trả về kèm `Source URL` (vd `/wiki/<title>` hoặc đầy đủ `https://en.wikipedia.org/wiki/...`).
    - `web_search`: giữ URL nhưng thêm marker thống nhất như `Source URL:`.
    - `pdf_retriever`: đảm bảo chuỗi context có marker nguồn (ví dụ file/page) để agent copy vào `Sources:`.
- Chuẩn hóa synthesizer:
  - `agentic_rag/graph.py`:
    - Sửa `SYNTHESIS_PROMPT_TEMPLATE` để ép synthesizer giữ nguyên/merge phần `Sources:` từ từng agent (hoặc ít nhất xuất thêm section `Sources:` tổng hợp).

### 3) Cache PDF retrieval (tăng tốc)

- `agentic_rag/tools.py`:
  - Thêm module-level lazy cache cho indexes retriever (splits/texts/vectorstore) để `pdf_retriever()` không rebuild FAISS mỗi lần.
  - Cần guard thread-safety đơn giản (vd lock) nếu graph chạy nhiều nhánh.

### 4) Verifier/claim-check node sau synthesizer

- `agentic_rag/graph.py`:
  - Add node `verifier_node` sau `synthesizer_node`.
  - Verifier LLM nhận:
    - `query`
    - `answer` từ synthesizer
    - `agent_results` (để đối chiếu)
  - Verifier trả thêm:
    - `confidence: float (0-1)`
    - `missing_info: bool`
    - `conflicts: bool` (nếu có)
    - `needed_agents: list[str]` hoặc `needed_subtasks`
  - Update `should_continue()` để loop theo verifier thay vì heuristic độ dài keyword.

### 5) Query decomposition (sub-questions) thay vì chỉ chọn agent set

- `agentic_rag/state.py`:
  - Thêm field vào state, ví dụ `sub_questions: list[SubQuestion]` (TypedDict hoặc dict schema).
  - `SubQuestion` tối thiểu có: `text`, `target_agent`, `task_type`.
- `agentic_rag/graph.py`:
  - Modify `orchestrator_node`:
    - Tạo `sub_questions` (vd: định nghĩa/bối cảnh/ví dụ/số liệu/nguồn PDF/Web…)
    - Không chỉ return `selected_agents`.
  - Modify `route_to_agents`:
    - Fan-out `Send("run_agent", AgentInvokeState(query=<sub_question_text>, agent_name=<target_agent>))`.
- `agentic_rag/graph.py` synthesizer_node:
  - Tổng hợp theo task_type để output mạch lạc hơn.

### 6) Streaming tiến trình (optional trong lượt này)

- `api/agentic_api.py`:
  - Thêm endpoint `POST /chat/stream` trả `text/event-stream`.
  - Dùng graph stream/astream để emit event khi:
    - orchestrator chọn subtask
    - từng agent hoàn thành (agent_name + partial result)
    - verifier decide loop/END
- Front-end:
  - `agentic_rag/streamlit_app.py`:
    - Nếu bạn muốn, implement client-side consume SSE và update progress UI.

## Tests / verification

- Chạy lại các test hiện có:
  - `test/test_graph.py`, `test/test_memory_chain.py`, `test/test_eval.py`.
- Thêm test nhỏ:
  - Unit test cho format `Sources` (regex presence of `Sources:`) khi tool output thay đổi.
  - Unit test cho cache: gọi `pdf_retriever` 2 lần và assert thời gian/hoặc assert cache được reused (có thể bằng cách kiểm tra biến cache không bị reset).

## Risk & mitigations

- Decomposition + verifier + caching có thể làm thay đổi hành vi routing:
  - Bật theo từng bước bằng feature flag (ví dụ env var) để dễ rollback.
- Streaming có thể phức tạp trên Streamlit:
  - Giữ “non-streaming endpoint” hoạt động như hiện tại; streaming thêm sau khi confirm UI ổn.
