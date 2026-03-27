"""Single-turn RAG Chatbot."""

from dotenv import load_dotenv
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import get_retriever, format_docs

load_dotenv()

retriever = get_retriever()

TEMPLATE = (
    "You are a strict, helpful assistant. Answer based on the following "
    "context and rules:\n"
    "Rules:\n"
    "- Always answer in the same language as the question\n"
    "- If you don't know the answer, say 'I don't know'\n"
    "- If the question is not related to the context, say 'I don't know'\n"
    "- If the question is not clear, ask for more information\n"
    "- Do not use outside knowledge, guessing, or web information\n"
    "- If application, cite sources as (source:page) using metadata\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
)

prompt = ChatPromptTemplate.from_template(TEMPLATE)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

rag_chain = (
    {"context": itemgetter("question") | retriever | format_docs,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

user_question = input("Enter a question: ")
answer = rag_chain.invoke({"question": user_question})

print(answer)
