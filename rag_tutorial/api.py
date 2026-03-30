"""Phase 4: Production - FastAPI.

API endpoint cho RAG chatbot

Install: pip install fastapi uvicorn
Run: uvicorn api:app --reload
"""

import warnings
from operator import itemgetter

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from config import format_docs, split_documents, load_documents

warnings.filterwarnings("ignore")
load_dotenv()

# =============================================================================
# SETUP
# =============================================================================
print("Loading documents...")
docs = load_documents()
splits = split_documents(docs)
texts = [doc.page_content for doc in splits]
metadatas = [doc.metadata for doc in splits]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas, k=5)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def hybrid_retrieve(query, k=5, weights=None):
    """Combine BM25 and vector retrieval with weighted scoring."""
    if weights is None:
        weights = [0.7, 0.3]

    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    combined = {}
    for rank, doc in enumerate(bm25_docs, 1):
        if doc.id not in combined:
            combined[doc.id] = {"doc": doc, "score": weights[0] * (1.0 / rank)}

    for rank, doc in enumerate(vector_docs, 1):
        if doc.id not in combined:
            combined[doc.id] = {"doc": doc, "score": weights[1] * (1.0 / rank)}
        else:
            combined[doc.id]["score"] += weights[1] * (1.0 / rank)

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results[:k]]


def expand_query(question):
    """Expand a question into 3 different phrasings."""
    expansion_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a user question, generate 3 different ways to phrase it. "
         "Return only questions, one per line."),
        ("human", "Original question: {question}"),
    ])
    response = llm.invoke(expansion_prompt.format(question=question))
    queries = response.content.strip().split("\n")
    return [q.strip() for q in queries if q.strip()]


def advanced_retrieve(question, k=5):
    """Query expansion + hybrid retrieval + deduplication."""
    expanded_queries = expand_query(question)
    all_docs = []
    seen_ids = set()

    for q in expanded_queries:
        retrieved_docs = hybrid_retrieve(q, k=k)
        for doc in retrieved_docs:
            if doc.id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc.id)

    return all_docs[:k]


TEMPLATE = """You are a helpful assistant. Answer based on the context.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)

rag_chain = (
    {
        "context": (
            itemgetter("question")
            | RunnableLambda(advanced_retrieve)
            | format_docs
        ),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="RAG Chatbot API",
    description="Advanced RAG chatbot with Hybrid Search",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str
    include_sources: bool = True


class Source(BaseModel):
    """Source document model."""
    content: str
    metadata: dict


class AnswerResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: list[Source] | None = None

# In-memory chat histories (per session)
chat_histories = {}

# Routes
@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "RAG Chatbot API", "docs": "/docs"}

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    """Chat endpoint - nhận câu hỏi, trả lời."""
    try:
        answer = rag_chain.invoke({"question": request.question})

        sources = None
        if request.include_sources:
            retrieved_docs = advanced_retrieve(request.question, k=3)
            sources = [
                Source(content=doc.page_content[:500], metadata=doc.metadata)
                for doc in retrieved_docs
            ]

        return AnswerResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/chat/history")
def chat_with_history(request: QuestionRequest, session_id: str = "default"):
    """Chat endpoint với chat history."""
    try:
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()

        history = chat_histories[session_id]
        history.add_user_message(request.question)

        answer = rag_chain.invoke({
            "question": request.question,
            "chat_history": history.messages
        })

        history.add_ai_message(answer)

        retrieved_docs = advanced_retrieve(request.question, k=3)
        sources = [
            Source(content=doc.page_content[:500], metadata=doc.metadata)
            for doc in retrieved_docs
        ]

        return AnswerResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.delete("/chat/history/{session_id}")
def clear_history(session_id: str):
    """Xóa chat history của một session"""
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"message": f"History cleared for session {session_id}"}

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
