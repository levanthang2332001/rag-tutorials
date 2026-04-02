"""Streamlit UI for Agentic RAG chain."""

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agentic_rag.chain import create_agentic_chain


st.set_page_config(page_title="Agentic RAG Chat", page_icon="🤖", layout="wide")
st.title("Agentic RAG Chat")
st.caption("LangGraph + memory + multi-agent routing")

if "chain" not in st.session_state:
    with st.spinner("Initializing agentic chain..."):
        try:
            st.session_state.chain = create_agentic_chain()
        except Exception as error:
            st.error(f"Failed to initialize chain: {error}")
            st.stop()
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session"

with st.sidebar:
    st.subheader("Session")
    st.session_state.session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
    )
    if st.button("Clear history"):
        st.session_state.chain.clear_history(st.session_state.session_id)
        st.success("History cleared")
    if st.button("Show sessions"):
        sessions = st.session_state.chain.list_sessions()
        st.write(sessions if sessions else ["(none)"])


history = st.session_state.chain.get_history(st.session_state.session_id)
for message in history:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)


prompt = st.chat_input("Ask something...")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke(
                query=prompt,
                session_id=st.session_state.session_id,
            )
        st.write(result["answer"])
        with st.expander("Details"):
            st.write(
                {
                    "iterations": result["iterations"],
                    "agents_used": result["agents_used"],
                    "session_id": result["session_id"],
                }
            )
        agent_results = result.get("agent_results", {})
        if agent_results:
            st.subheader("Per-agent results")
            for agent_name, agent_output in agent_results.items():
                with st.expander(f"{agent_name}"):
                    st.write(agent_output)
