"""Configuration for Agentic RAG package."""

import warnings

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

MARKDOWN_SEPARATOR = [
    r"\n#{1,6} ",
    r"```\n",
    r"\n\*\*\*+\n",
    r"\n---+",
    r"\n___",
    r"\n\n",
    r"\n",
    r" ",
    "",
]

CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 5


def load_documents():
    """Load PDF documents from ./papers directory."""
    loader = DirectoryLoader(
        path="./papers",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def split_documents(docs):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATOR,
    )
    return text_splitter.split_documents(docs)


def format_docs(docs):
    """Format documents into a single string for context."""
    return "\n\n".join([d.page_content for d in docs])