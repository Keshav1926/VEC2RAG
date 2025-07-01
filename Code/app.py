import streamlit as st
import time
from rag_pipeline import build_streaming_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize current display state
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""

# Initialize processing state
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# App title
st.title("📚 RAG-powered Chatbot")

# Input
query = st.text_input("Ask a question:", key="query_input", disabled=st.session_state.is_processing)

send_button = st.button("Send", disabled=st.session_state.is_processing)
clear_button = st.button("Clear", key="clear_btn", disabled=st.session_state.is_processing)

if send_button and query:
    st.session_state.is_processing = True
    
    # Initialize streaming chain
    qa_chain = build_streaming_chain()
    
    # Create placeholder for streaming output
    answer_placeholder = st.empty()
    full_answer = ""
    
    try:
        # Stream the response
        for chunk in qa_chain.stream({"query": query}):
            full_answer += chunk
            answer_placeholder.markdown(f"**Answer:** {full_answer}▋")
            time.sleep(0.02)  # Small delay to make streaming visible
        
        answer_placeholder.markdown(f"**Answer:** {full_answer}")
        
        # Save to history
        st.session_state.history.append((query, full_answer))
        st.session_state.current_answer = full_answer
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
    
    finally:
        st.session_state.is_processing = False
        st.rerun()

if clear_button:
    st.session_state.current_answer = ""
    st.session_state.history = []
    if 'query_input' in st.session_state:
        del st.session_state['query_input']
    st.rerun()

# Display current answer if exists (for page refreshes)
if st.session_state.current_answer and not st.session_state.is_processing:
    st.markdown(f"**Last Answer:** {st.session_state.current_answer}")

# Bottom-expander for full conversation history
if st.session_state.history:
    with st.expander("Show conversation history", expanded=False):
        for i, (q, a) in enumerate(reversed(st.session_state.history)):
            st.write(f"**Q{len(st.session_state.history)-i}:** {q}")
            st.write(f"**A{len(st.session_state.history)-i}:** {a}")
            if i < len(st.session_state.history) - 1:
                st.divider()

# Sidebar info
st.sidebar.title("Settings")
st.sidebar.write(f"**Collection:** amlgo-docs")
st.sidebar.write(f"**Embedding model:** sentence-transformers/all-MiniLM-L6-v2")
st.sidebar.write(f"**LLM model:** meta/llama-3.1-405b-instruct")
st.sidebar.write(f"**Number of chunks:** 45")
st.sidebar.write(f"**Streaming:** ✅ Enabled")

if st.session_state.is_processing:
    st.sidebar.write("**Status:** 🔄 Processing...")
else:
    st.sidebar.write("**Status:** ✅ Ready")

st.sidebar.markdown("---")
st.sidebar.write(f"**Total conversations:** {len(st.session_state.history)}")
