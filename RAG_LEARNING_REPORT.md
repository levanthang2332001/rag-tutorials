# RAG Learning Report

## Overview

This document summarizes all concepts learned in the RAG (Retrieval-Augmented Generation) learning journey, from basic to advanced level.

---

## Phase 1: Multi-turn Chat History

### Concepts Learned

#### 1.1 ChatMessageHistory

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Create history store
chat_history = ChatMessageHistory()

# Add messages
chat_history.add_user_message("What is AI?")
chat_history.add_ai_message("AI is Artificial Intelligence...")

# View messages
print(chat_history.messages)
# [HumanMessage(content='What is AI?'), AIMessage(content='AI is Artificial Intelligence...')]
```

#### 1.2 MessagesPlaceholder

Used to inject chat history into prompts:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context: {context}\n\nQuestion: {question}"),
])
```

#### 1.3 Question Condensation

When users ask follow-up questions like "How does it work?", the LLM condenses the question using chat history:

```
User: "What is AI?"
Bot: "AI is..."

User: "How does it work?"
→ Condensed: "How does AI work?"
→ This condensed question is used for retrieval
```

#### 1.4 Manual History-Aware RAG Chain

```python
def chat_with_history(question, chat_history_messages, debug=False):
    # 1. Condense question using chat history
    condensed_q = condense_question({
        "chat_history": chat_history_messages,
        "question": question
    })

    # 2. Retrieve with condensed question
    docs = retriever.invoke(condensed_q)

    # 3. Generate answer
    answer = rag_chain.invoke({"question": condensed_q})

    return answer
```

---

## Phase 2: Advanced Retrieval

### 2.1 BM25 Retriever (Keyword-based)

BM25 (Best Matching 25) is a classic algorithm for keyword-based search:

```python
from langchain_community.retrievers.bm25 import BM25Retriever

bm25_retriever = BM25Retriever.from_texts(
    texts=texts,
    metadatas=metadatas,
    k=5  # Return top 5 results
)
```

**Characteristics:**
- Exact keyword matching
- Fast and reliable
- Does NOT understand synonyms ("car" ≠ "automobile")

### 2.2 Vector Retriever (Semantic)

Vector search uses embeddings to find semantically similar documents:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    metadatas=metadatas
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

**Characteristics:**
- Understands semantic meaning
- Handles synonyms well
- "car" ≈ "automobile"

### 2.3 Hybrid Retriever (BM25 + Vector)

Combines both approaches for best results:

```python
def hybrid_retrieve(query, k=5, weights=[0.7, 0.3]):
    """
    Combine BM25 + Vector retrieval

    Args:
        query: Question
        k: Number of results
        weights: [bm25_weight, vector_weight]
    """
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    combined = {}
    for rank, doc in enumerate(bm25_docs, 1):
        score = weights[0] * (1.0 / rank)
        # ... combine scores

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results[:k]]
```

**Why Hybrid?**
- BM25: Exact keyword match
- Vector: Semantic understanding
- Hybrid: Best of both worlds

### 2.4 Reranking

After retrieval, reranking improves result quality:

```python
def simple_rerank(query, docs, top_k=5):
    """
    Rerank documents using keyword overlap

    Production: Use Cohere Rerank or cross-encoder model
    """
    query_words = set(query.lower().split())

    def keyword_score(doc):
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words)
        return overlap / len(query_words)

    scored_docs = [(doc, keyword_score(doc)) for doc in docs]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]
```

### 2.5 Query Expansion

Expands one question into multiple versions for better retrieval:

```python
expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a user question, generate 3 different ways to phrase it. Return only questions, one per line."),
    ("human", "Original question: {question}"),
])

def expand_query(question):
    response = llm.invoke(expansion_prompt.format(question=question))
    queries = response.content.strip().split("\n")
    return [q.strip() for q in queries if q.strip()]
```

**Example:**
```
Original: "What is AI in blockchain?"
Expanded:
1. "What role does AI play in blockchain?"
2. "How does AI enhance blockchain systems?"
3. "Relationship between AI and blockchain?"
```

---

## Phase 3: RAG Evaluation

### 3.1 Faithfulness

Measures if the answer is consistent with the context:

```
Context: "AI was invented in 1956"
Answer: "AI was invented in 1956" → Faithfulness = HIGH (10/10)
Answer: "AI was invented in 2020" → Faithfulness = LOW (2/10)
```

### 3.2 Answer Relevancy

Measures if the answer addresses the question:

```
Question: "What is AI?"
Answer: "AI stands for Artificial Intelligence..." → Relevancy = HIGH (10/10)
Answer: "Python is a programming language" → Relevancy = LOW (0/10)
```

### 3.3 Context Precision

Measures if retrieved documents contain relevant information:

```
Question: "Who invented AI?"
Retrieved: [AI history doc, ML doc] → Precision = HIGH (10/10)
Retrieved: [cooking doc, car doc] → Precision = LOW (0/10)
```

### 3.4 Context Recall

Measures if context covers all needed information:

```
Question: "Tell me about AI (history, types, applications)"
Context: [History only] → Recall = LOW
Context: [History, Types, Applications] → Recall = HIGH
```

### 3.5 Evaluation Metrics Summary

| Metric | Measures | Low Score Means |
|--------|----------|-----------------|
| Faithfulness | Answer matches context | Hallucination |
| Answer Relevancy | Answer addresses question | Irrelevant answer |
| Context Precision | Docs are relevant | Poor retrieval |
| Context Recall | Context covers full answer | Missing information |

---

## Phase 4: Production

### 4.1 Streamlit UI

Web interface for end users:

```python
import streamlit as st

st.title("RAG Chatbot")
st.caption("Advanced RAG with Hybrid Search")

if prompt := st.chat_input("Ask a question..."):
    response = rag_chain.invoke({"question": prompt})
    st.markdown(response)
```

**Run:**
```bash
streamlit run streamlit_app.py
```

### 4.2 FastAPI

REST API for programmatic access:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = True

@app.post("/chat")
def chat(request: QuestionRequest):
    answer = rag_chain.invoke({"question": request.question})
    return {"answer": answer}
```

**Run:**
```bash
uvicorn api:app --reload
```

### 4.3 When to Use What

| Use Case | Technology |
|----------|------------|
| Quick prototype/demo | Streamlit |
| End-user application | Streamlit or React/Vue |
| Mobile app backend | FastAPI |
| Integration with other services | FastAPI |
| Microservices | FastAPI |

---

## File Structure

```
RAG/
├── config.py              # Shared setup (load, split, embed, retrieve)
├── chatbot.py             # Single-turn RAG
├── chatbot_memory.py      # Multi-turn with chat history
├── hybrid_search.py      # Advanced retrieval (BM25 + Vector + Query Expansion)
├── evaluation.py         # Evaluation metrics
├── streamlit_app.py      # Web UI
├── api.py               # FastAPI endpoints
└── papers/              # PDF documents for RAG
```

---

## Key Configuration Parameters

### Chunking
```python
CHUNK_SIZE = 600        # Characters per chunk
CHUNK_OVERLAP = 200     # Overlap between chunks
```

### Retrieval
```python
RETRIEVAL_K = 5         # Number of documents to retrieve
RETRIEVAL_FETCH_K = 20  # Fetch more, then rerank
```

### Hybrid Search
```python
weights = [0.7, 0.3]   # 70% BM25, 30% Vector
```

---

## Best Practices

### 1. Chunk Size Selection
- Smaller chunks (300-600): More precise, better for specific questions
- Larger chunks (1000-2000): More context, better for complex questions

### 2. Retrieval Strategy
- Use BM25 for exact keyword queries
- Use Vector for semantic/conceptual queries
- Use Hybrid for best overall results

### 3. Query Expansion
- Generate 2-5 alternative phrasings
- Helps retrieve diverse relevant documents
- Increases recall

### 4. Evaluation
- Always evaluate on your specific use case
- Metrics guide improvement efforts
- RAGAS provides standardized evaluation framework

### 5. Production
- Cache vectorstore to avoid reloading
- Use async for better performance
- Implement rate limiting for API
- Monitor costs (each query = multiple API calls)

---

## Glossary

| Term | Definition |
|------|------------|
| RAG | Retrieval-Augmented Generation - combines retrieval and generation |
| Chunk | Small piece of document for retrieval |
| Embedding | Numerical representation of text |
| BM25 | Best Matching 25 - keyword search algorithm |
| Vector Store | Database for storing and searching embeddings |
| MMR | Maximum Marginal Relevance - diverse retrieval |
| Faithfulness | Answer consistency with context |
| Recall | Ability to retrieve all relevant documents |
| Precision | Ability to retrieve only relevant documents |

---

## Further Learning

### Advanced Topics
1. **Agentic RAG** - RAG with tool use and reasoning
2. **Graph RAG** - RAG with knowledge graphs
3. **Multi-modal RAG** - RAG with images and tables
4. **Fine-tuning Embeddings** - Custom embeddings for specific domains

### Resources
- LangChain Documentation: https://python.langchain.com/
- RAGAS: https://docs.ragas.io/
- DeepLearning.AI RAG Course: https://www.deeplearning.ai/courses/

---

## Summary

From this learning journey, you now understand:

- [x] Single-turn vs Multi-turn RAG
- [x] Chat history management
- [x] BM25 keyword search
- [x] Vector semantic search
- [x] Hybrid search combination
- [x] Query expansion
- [x] Reranking
- [x] Evaluation metrics (Faithfulness, Precision, Recall, Relevancy)
- [x] Streamlit UI creation
- [x] FastAPI development

You are now ready to build production-level RAG applications!
