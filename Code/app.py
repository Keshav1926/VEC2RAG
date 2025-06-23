import os
import streamlit as st
from rag_pipeline import build_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Initialize QA chain
qa_chain = build_chain()

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize current display state
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""

# App title
st.title("📚 RAG-powered Chatbot")

# Input
query = st.text_input("Ask a question:", key="query_input")

send_button = st.button("Send")
clear_button = st.button("Clear", key="clear_btn")

if send_button and query:
    # Invoke RAG chain explicitly
    result = qa_chain({"query": query})
    answer = result.get("result", "No answer returned.")
    # Save to history
    st.session_state.history.append((query, answer))
    st.session_state.current_answer = answer

if clear_button:
    st.session_state.current_answer = ""
    if 'query_input' in st.session_state:
        del st.session_state['query_input']
    st.rerun()

# Display current answer if exists
if st.session_state.current_answer:
    st.markdown(f"**Answer:** {st.session_state.current_answer}")

# Bottom-expander for full conversation history
with st.expander("Show conversation history"):
    for q, a in st.session_state.history:
        st.write(f"Q: {q}")
        st.write(f"A: {a}")

# Sidebar info
st.sidebar.title("Settings")
st.sidebar.write(f"Collection: amlgo-docs")
st.sidebar.write(f"Embedding model: sentence-transformers/all-MiniLM-L6-v2")
st.sidebar.write(f"Current LLM model: llama-3.3-70b-instruct")
st.sidebar.write(f"Number of chunks: 45")