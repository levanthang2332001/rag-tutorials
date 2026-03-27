"""
Phase 3: RAG Evaluation

Learn evaluation metrics:
1. Faithfulness - Is answer faithful to context?
2. Answer Relevancy - Is answer relevant to question?
3. Context Precision - Is retrieved context precise?
4. Context Recall - Does context cover all needed info?

Install: pip install ragas
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
# SETUP
# =============================================================================
print("Loading documents...")
docs = load_documents()
splits = split_documents(docs)
texts = [doc.page_content for doc in splits]
metadatas = [doc.metadata for doc in splits]
print(f"Loaded {len(splits)} splits")

# Create retrievers
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas, k=5)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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
        score = weights[0] * (1.0 / rank)
        if doc.id not in combined:
            combined[doc.id] = {"doc": doc, "score": score}
        else:
            combined[doc.id]["score"] += score

    for rank, doc in enumerate(vector_docs, 1):
        score = weights[1] * (1.0 / rank)
        if doc.id not in combined:
            combined[doc.id] = {"doc": doc, "score": score}
        else:
            combined[doc.id]["score"] += score

    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results[:k]]


def expand_query(question):
    """Expand question into 3 different phrasings."""
    expansion_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a user question, generate 3 different ways to phrase it. "
         "Return only questions, one per line."),
        ("human", "Original question: {question}"),
    ])
    response = llm.invoke(expansion_prompt.format(question=question))
    queries = response.content.strip().split("\n")
    return [q.strip() for q in queries if q.strip()]


def advanced_retrieve(question, k=10):
    """Query expansion + hybrid retrieval + keyword rerank."""
    expanded_queries = expand_query(question)
    all_docs = []
    seen_ids = set()

    for q in expanded_queries:
        retrieved_docs = hybrid_retrieve(q, k=k)
        for doc in retrieved_docs:
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


TEMPLATE = (
    "You are a helpful assistant. Answer based ONLY on the context.\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer based ONLY on the context above. "
    "If the answer is not in the context, say 'I don't know'."
)

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
# EVALUATION METRICS - MANUAL IMPLEMENTATION
# =============================================================================
print("\n" + "="*60)
print("PHASE 3: RAG EVALUATION")
print("="*60)

# =============================================================================
# 1. FAITHFULNESS - Is answer faithful to context?
# =============================================================================
print("\n" + "="*60)
print("1. FAITHFULNESS")
print("="*60)
print("""
Faithfulness = Is the answer consistent with the context?

Example:
Context: "AI was invented in 1956"
Answer: "AI was invented in 1956" → Faithfulness = HIGH
Answer: "AI was invented in 2020" → Faithfulness = LOW
""")

faithfulness_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluator. Given a context and answer, rate faithfulness from 1-10.
Consider: Are all claims in the answer supported by the context?
Return only a number from 1-10."""),
    ("human", "Context: {context}\n\nAnswer: {answer}"),
])

def evaluate_faithfulness(context, answer):
    """Evaluate faithfulness."""
    response = llm.invoke(faithfulness_prompt.format(context=context, answer=answer))
    try:
        return int(response.content.strip())
    except (ValueError, AttributeError):
        return 0

# =============================================================================
# 2. ANSWER RELEVANCY - Is answer relevant to question?
# =============================================================================
print("\n" + "="*60)
print("2. ANSWER RELEVANCY")
print("="*60)
print("""
Answer Relevancy = Does the answer address the question?

Example:
Question: "What is AI?"
Answer: "AI stands for Artificial Intelligence..." → Relevancy = HIGH
Answer: "Python is a programming language" → Relevancy = LOW
""")

relevancy_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a question and answer, rate how well the answer addresses the question from 1-10.
Return only a number from 1-10."""),
    ("human", "Question: {question}\n\nAnswer: {answer}"),
])

def evaluate_answer_relevancy(question, answer):
    """Evaluate answer relevancy."""
    response = llm.invoke(relevancy_prompt.format(question=question, answer=answer))
    try:
        return int(response.content.strip())
    except (ValueError, AttributeError):
        return 0

# =============================================================================
# 3. CONTEXT PRECISION - Is retrieved context precise?
# =============================================================================
print("\n" + "="*60)
print("3. CONTEXT PRECISION")
print("="*60)
print("""
Context Precision = Do the retrieved docs contain info to answer the question?

Example:
Question: "Who invented AI?"
Retrieved Docs: [doc about AI history, doc about ML] → Precision = HIGH
Retrieved Docs: [doc about cooking, doc about cars] → Precision = LOW
""")

precision_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a question and retrieved context, rate context precision from 1-10.
How much of the retrieved context is relevant to answering the question?
Return only a number from 1-10."""),
    ("human", "Question: {question}\n\nContext: {context}"),
])

def evaluate_context_precision(question, ctx):
    """Evaluate context precision."""
    response = llm.invoke(precision_prompt.format(question=question, context=ctx))
    try:
        return int(response.content.strip())
    except (ValueError, AttributeError):
        return 0

# =============================================================================
# 4. CONTEXT RECALL - Does context cover all needed info?
# =============================================================================
print("\n" + "="*60)
print("4. CONTEXT RECALL")
print("="*60)
print("""
Context Recall = Does context cover ALL info needed to answer?

Example:
Question: "Tell me about AI history, types, and applications"
Context: Only has AI history → Recall = LOW
Context: Has history, types, AND applications → Recall = HIGH
""")

recall_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a question and context, rate context recall from 1-10.\n"
     "Does the context contain all information needed to fully answer?\n"
     "Return only a number from 1-10."),
    ("human", "Question: {question}\n\nContext: {context}"),
])

def evaluate_context_recall(question, ctx):
    """Evaluate context recall."""
    response = llm.invoke(recall_prompt.format(question=question, context=ctx))
    try:
        return int(response.content.strip())
    except (ValueError, AttributeError):
        return 0

# =============================================================================
# DEMO EVALUATION
# =============================================================================
print("\n" + "="*60)
print("RUNNING EVALUATION DEMO")
print("="*60)

test_cases = [
    {
        "question": "What are the challenges of AI in blockchain?",
        "expected": "Scalability, latency, computational overhead, regulatory uncertainty"
    },
    {
        "question": "How does AI help in trading?",
        "expected": "AI analyzes data, generates trading signals, detects fraud"
    },
    {
        "question": "What is AI-enhanced cryptocurrency trading?",
        "expected": "AI trading bots, fraud detection, Sharpe ratios"
    },
]

print("\n" + "-"*60)
for i, test in enumerate(test_cases, 1):
    print(f"\nTest Case {i}: {test['question']}")

    rag_answer = rag_chain.invoke({"question": test["question"]})
    ctx_docs = advanced_retrieve(test["question"], k=5)
    ctx = format_docs(ctx_docs)

    f_score = evaluate_faithfulness(ctx, rag_answer)
    r_score = evaluate_answer_relevancy(test["question"], rag_answer)
    p_score = evaluate_context_precision(test["question"], ctx)
    c_score = evaluate_context_recall(test["question"], ctx)

    print(f"  Faithfulness:      {f_score}/10")
    print(f"  Answer Relevancy: {r_score}/10")
    print(f"  Context Precision: {p_score}/10")
    print(f"  Context Recall:    {c_score}/10")
    print(f"  Avg Score:         {(f_score + r_score + p_score + c_score) / 4:.1f}/10")

    print(f"\n  Answer: {rag_answer[:150]}...")

print("\n" + "="*60)
print("Phase 3 Demo Complete!")
print("="*60)

# =============================================================================
# RAGAS - PRODUCTION EVALUATION
# =============================================================================
"""
================================================================================
RAGAS - Production Evaluation Framework
================================================================================

RAGAS (RAG Assessment) is the standard framework for RAG evaluation:

Install: pip install ragas datasets

Metrics:
- ragasFaithfulness: Faithfulness
- ragasAnswerRelevancy: Answer relevancy
- ragasContextPrecision: Context precision
- ragasContextRecall: Context recall

Usage:

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

data = {
    "question": ["What is AI?", "How does AI work?"],
    "answer": ["AI is...", "AI works by..."],
    "contexts": [["AI is artificial intelligence..."], ["AI uses algorithms..."]],
    "ground_truths": [["AI is artificial intelligence, sim..."], ["AI works by..."]]
}

dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(result)
"""