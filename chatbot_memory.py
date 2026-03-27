"""
Multi-turn RAG Chatbot with Memory

Flow: Condense → Retrieve → Answer
"""

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
from dotenv import load_dotenv

from config import get_retriever, format_docs

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = get_retriever()

# Prompt for LLM to expand short question + history to full question
condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question that can be understood without the chat history. Do NOT answer the question, just reformulate it if needed."),
    ("human", "Chat History: {chat_history}\n\nCurrent Question: {question}"),
])

def condense_question(input_dict):
    """Condense question using chat history"""
    chat_history = input_dict["chat_history"]
    question = input_dict["question"]

    if not chat_history:
        return question

    # Format chat history as string
    history_str = "\n".join([
        f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in chat_history
    ])

    # Invoke LLM to condense
    condensed = llm.invoke(
        condense_prompt.format(chat_history=history_str, question=question)
    )
    return condensed.content

# Prompt to answer
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer based on the context."),
    ("human", "Context: {context}\n\nQuestion: {question}"),
])

# RAG chain
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# Full flow: Condense → Retrieve → Answer
def chat_with_history(question, chat_history_messages, debug=False):
    # 1. Condense question
    condensed_q = condense_question({
        "chat_history": chat_history_messages,
        "question": question
    })

    if debug:
        print(f"\n[DEBUG] Original Question: {question}")
        print(f"[DEBUG] Condensed Question: {condensed_q}")
        print(f"[DEBUG] Chat History: {len(chat_history_messages)} messages")

    # 2. Retrieve
    docs = retriever.invoke(condensed_q)

    if debug:
        print(f"[DEBUG] Retrieved: {len(docs)} docs")
        for i, doc in enumerate(docs[:2], 1):
            print(f"  Doc {i}: {doc.page_content[:80]}...")

    # 3. Answer
    answer = rag_chain.invoke({"question": condensed_q})

    if debug:
        print(f"[DEBUG] Answer: {answer[:100]}...")

    return answer

# Chat History Store
chat_history = []


print("=" * 60)
print("Multi-turn RAG Chatbot (Manual History-Aware)")
print("Flow: Condense → Retrieve → Answer")
print("Type 'exit' to quit, 'debug' to toggle debug mode")
print("=" * 60)

debug_mode = False

while True:
    question = input("\nYou: ")
    if question.lower() == "exit":
        break
    if question.lower() == "debug":
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        continue

    # Invoke
    answer = chat_with_history(question, chat_history, debug=debug_mode)

    print(f"\nBot: {answer}")

    # Save to history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
