"""
Phase 2: Advanced Retrieval

Learn:
1. Hybrid Search - BM25 + Vector combination
2. Reranking - Reorder results
3. Query Expansion - Expand questions

Install: pip install rank_bm25
"""

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from config import format_docs, split_documents, load_documents
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

load_dotenv()

# =============================================================================
# SETUP - Load documents
# =============================================================================
print("Loading documents...")
docs = load_documents()
splits = split_documents(docs)
texts = [doc.page_content for doc in splits]
metadatas = [doc.metadata for doc in splits]
print(f"Loaded {len(splits)} splits")

# =============================================================================
# 1. BM25 RETRIEVER - Keyword-based search
# =============================================================================
print("\n" + "="*60)
print("1. BM25 Retriever (Keyword-based)")
print("="*60)

bm25_retriever = BM25Retriever.from_texts(
    texts=texts,
    metadatas=metadatas,
    k=5
)

# =============================================================================
# 2. VECTOR RETRIEVER - Semantic search
# =============================================================================
print("\n" + "="*60)
print("2. Vector Retriever (Semantic)")
print("="*60)

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    metadatas=metadatas
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =============================================================================
# 3. HYBRID RETRIEVER - Combine BM25 + Vector
# =============================================================================
print("\n" + "="*60)
print("3. Hybrid Retriever (BM25 + Vector)")
print("="*60)

def hybrid_retrieve(query, k=5, weights=[0.5, 0.5]):
    """
    Combine BM25 + Vector retrieval

    Args:
        query: Question
        k: Number of results to return
        weights: [bm25_weight, vector_weight]

    Returns:
        List of documents with combined scores
    """
    # Retrieve from both methods
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    # Combine scores
    combined = {}
    seen_ids = set()

    # BM25 docs with scores (using rank position)
    for rank, doc in enumerate(bm25_docs, 1):
        score = weights[0] * (1.0 / rank)
        if doc.id not in combined:
            combined[doc.id] = {"doc": doc, "score": score}
            seen_ids.add(doc.id)
        else:
            combined[doc.id]["score"] += score

    # Vector docs with scores
    for rank, doc in enumerate(vector_docs, 1):
        score = weights[1] * (1.0 / rank)
        if doc.id not in combined:
            combined[doc.id] = {"doc": doc, "score": score}
            seen_ids.add(doc.id)
        else:
            combined[doc.id]["score"] += score

    # Sort by combined score
    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results[:k]]

# =============================================================================
# 4. RERANKING - Reorder results
# =============================================================================
print("\n" + "="*60)
print("4. Reranking (Post-retrieval)")
print("="*60)

def simple_rerank(query, docs, top_k=5):
    """
    Simple reranking using keyword overlap

    Production: Use Cohere Rerank or cross-encoder
    """
    query_words = set(query.lower().split())

    def keyword_score(doc):
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words)
        return overlap / len(query_words) if query_words else 0

    scored_docs = [(doc, keyword_score(doc)) for doc in docs]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

# =============================================================================
# 5. QUERY EXPANSION - Expand questions
# =============================================================================
print("\n" + "="*60)
print("5. Query Expansion")
print("="*60)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a user question, generate 3 different ways to phrase it. Return only questions, one per line."),
    ("human", "Original question: {question}"),
])

def expand_query(question):
    """Expand 1 question into 3 versions"""
    response = llm.invoke(expansion_prompt.format(question=question))
    queries = response.content.strip().split("\n")
    return [q.strip() for q in queries if q.strip()]

# =============================================================================
# 6. FULL ADVANCED RAG CHAIN
# =============================================================================
print("\n" + "="*60)
print("6. Full Advanced RAG Chain")
print("="*60)

template = """You are a helpful assistant. Answer based on the context.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)


def advanced_retrieve(question, k=5):
    """Query expansion + hybrid retrieval"""
    expanded_queries = expand_query(question)
    all_docs = []
    seen_ids = set()

    for q in expanded_queries:
        docs = hybrid_retrieve(q, k=k)
        for doc in docs:
            if doc.id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc.id)

    return simple_rerank(question, all_docs, top_k=k)


advanced_rag_chain = (
    {
        "context": itemgetter("question") | RunnableLambda(advanced_retrieve) | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What is AI?",
        "How does AI work in blockchain?",
    ]

    print("\n" + "="*60)
    print("TESTING")
    print("="*60)

    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")

        # BM25
        bm25_results = bm25_retriever.invoke(query)
        print(f"BM25: {len(bm25_results)} docs")

        # Vector
        vector_results = vector_retriever.invoke(query)
        print(f"Vector: {len(vector_results)} docs")

        # Hybrid
        hybrid_results = hybrid_retrieve(query, k=5)
        print(f"Hybrid: {len(hybrid_results)} docs")

        # Query Expansion
        expanded = expand_query(query)
        print(f"Expanded to {len(expanded)} queries:")
        for q in expanded:
            print(f"  - {q}")

    # Full chain test
    print("\n--- Full Chain Test ---")
    test_q = "How does AI help in trading?"
    result = advanced_rag_chain.invoke({"question": test_q})
    print(f"\nQuestion: {test_q}")
    print(f"Answer: {result[:300]}...")

    print("\n" + "="*60)
    print("Phase 2 Demo Complete!")
    print("="*60)
