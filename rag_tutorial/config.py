"""RAG Configuration - Shared setup for all RAG files."""

import warnings

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

MARKDOWN_SEPARATOR = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___\n",
    "\n\n",
    "\n",
    " ",
    "",
]

CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 5
RETRIEVAL_FETCH_K = 20


def load_documents():
    """Load PDF documents from ./papers directory"""
    loader = DirectoryLoader(
        path="./papers",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def split_documents(docs):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATOR,
    )
    return text_splitter.split_documents(docs)


def create_vectorstore(splits):
    """Create FAISS vector store from document splits"""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )


def create_retriever(vectorstore):
    """Create retriever from vector store"""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_FETCH_K},
    )


def setup_rag():
    """Full RAG setup: load, split, embed, create vector store and retriever"""
    docs = load_documents()
    splits = split_documents(docs)
    vectorstore = create_vectorstore(splits)
    retriever = create_retriever(vectorstore)
    return {
        "docs": docs,
        "splits": splits,
        "vectorstore": vectorstore,
        "retriever": retriever,
    }


def get_retriever():
    """Get retriever (convenience function)"""
    return setup_rag()["retriever"]


def format_docs(docs):
    """Format documents into a single string for context"""
    return "\n\n".join([d.page_content for d in docs])
