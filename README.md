# RAG Learning Project

Learn RAG (Retrieval-Augmented Generation) from basic to advanced level.

## Features

- **Phase 1**: Single & Multi-turn RAG Chatbot
- **Phase 2**: Advanced Retrieval (BM25 + Vector + Hybrid + Query Expansion)
- **Phase 3**: RAG Evaluation (Faithfulness, Precision, Recall, Relevancy)
- **Phase 4**: Production (Streamlit UI, FastAPI)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run single-turn chatbot
py chatbot.py

# Run multi-turn chatbot with memory
py chatbot_memory.py

# Run advanced retrieval demo
py hybrid_search.py

# Run evaluation
py evaluation.py

# Run Streamlit UI
streamlit run streamlit_app.py

# Run FastAPI
uvicorn api:app --reload
```

## Project Structure

```
RAG/
├── config.py              # Shared setup (load, split, embed, retrieve)
├── chatbot.py             # Single-turn RAG
├── chatbot_memory.py      # Multi-turn with chat history
├── hybrid_search.py      # Advanced retrieval (BM25 + Vector)
├── evaluation.py         # Evaluation metrics
├── streamlit_app.py      # Web UI
├── api.py               # FastAPI endpoints
└── papers/             # PDF documents
```

## Tech Stack

- **Framework**: LangChain
- **LLM**: OpenAI (GPT-4o-mini)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: FAISS
- **Keyword Search**: BM25
- **UI**: Streamlit
- **API**: FastAPI

## Phases

### Phase 1: Multi-turn Chat History
- ChatMessageHistory
- Question Condensation
- Chat History Management

### Phase 2: Advanced Retrieval
- BM25 Keyword Search
- Vector Semantic Search
- Hybrid Search (BM25 + Vector)
- Query Expansion
- Reranking

### Phase 3: RAG Evaluation
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

### Phase 4: Production
- Streamlit Web UI
- FastAPI REST API
- Docker deployment ready

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- OpenAI API Key

## Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## License

MIT License
