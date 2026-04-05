"""Lazy-cached PDF hybrid retrievers (FAISS + BM25)."""

from __future__ import annotations

import threading

from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from agentic_rag.config import load_documents, split_documents
from core.config import get_config

_PDF_RETRIEVERS_CACHE: dict[str, object] | None = None
_PDF_RETRIEVERS_LOCK = threading.Lock()


def _get_pdf_retrievers() -> dict[str, object]:
    """Build and cache PDF retrievers once (FAISS + BM25)."""
    global _PDF_RETRIEVERS_CACHE
    if _PDF_RETRIEVERS_CACHE is not None:
        return _PDF_RETRIEVERS_CACHE

    with _PDF_RETRIEVERS_LOCK:
        if _PDF_RETRIEVERS_CACHE is not None:
            return _PDF_RETRIEVERS_CACHE

        docs = load_documents()
        splits = split_documents(docs)
        texts = [doc.page_content for doc in splits]
        metadatas = [doc.metadata for doc in splits]

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=get_config().openai_api_key,
        )
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_texts(
            texts=texts,
            metadatas=metadatas,
            k=5,
        )

        _PDF_RETRIEVERS_CACHE = {
            "vector_retriever": vector_retriever,
            "bm25_retriever": bm25_retriever,
        }
        return _PDF_RETRIEVERS_CACHE
