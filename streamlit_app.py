"""
Phase 4: Production - Streamlit UI

Web interface for RAG chatbot

Install: pip install streamlit
Run: streamlit run streamlit_app.py
"""

import warnings
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import format_docs, split_documents, load_documents

warnings.filterwarnings("ignore")
load_dotenv()

# =============================================================================
# SETUP - Run once when app starts
# =============================================================================
@st.cache_resource
def setup_rag():
    """Cache vectorstore to avoid reloading on every query"""
    print("Setting up RAG...")

    docs = load_documents()
    splits = split_documents(docs)
    texts = [doc.page_content for doc in splits]
    metadatas = [doc.metadata for doc in splits]

    # Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

    # BM25
    bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas, k=5)

    # Vector retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return bm25_retriever, vector_retriever

# Setup
bm25_retriever, vector_retriever = setup_rag()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def hybrid_retrieve(query, k=5, weights=None):
    """Combine BM25 + Vector."""
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
    """Expand question into 3 phrasings."""
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
    """Query expansion + hybrid retrieval + rerank."""
    expanded_queries = expand_query(question)
    all_docs = []
    seen_ids = set()

    for q_expanded in expanded_queries:
        retrieved = hybrid_retrieve(q_expanded, k=k)
        for doc in retrieved:
            if doc.id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc.id)

    query_words = set(question.lower().split())

    def keyword_score(doc):
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words)
        return overlap / len(query_words) if query_words else 0

    scored_docs = [(doc, keyword_score(doc)) for doc in all_docs]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:k]]


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
# STREAMLIT UI
# =============================================================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.caption("Advanced RAG with Hybrid Search + Query Expansion")

# Sidebar - Settings
with st.sidebar:
    st.header("Settings")
    k_value = st.slider("Number of docs", 3, 10, 5)
    bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.7, 0.1)
    st.divider()
    st.caption("Hybrid: BM25 + Vector search")
    st.caption("Query Expansion: 3 queries")

# Chat History (in-memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt_input := st.chat_input("Ask a question about AI and Blockchain..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"question": prompt_input})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
