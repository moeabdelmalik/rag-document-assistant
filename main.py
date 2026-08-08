import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import load_document, chunk_document, embed_and_store, generate
from styles import CUSTOM_CSS

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="centered",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- TITLE ---
st.markdown("""
    <div class="title-block">
        <h1>📄 RAG <span class="accent">Assistant</span></h1>
        <p>Upload a document. Ask anything about it.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# --- SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# --- SECTION 1: Upload Document ---
st.markdown("### 01 — Upload Document")
uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # Only reprocess if a new file is uploaded
    if st.session_state.doc_name != uploaded_file.name:
        with st.spinner("Reading and processing your document..."):
            suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            documents = load_document(tmp_path)
            chunks = chunk_document(documents)
            vector_store = embed_and_store(chunks)

            st.session_state.vector_store = vector_store
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chat_history = []

            os.unlink(tmp_path)

        st.markdown(f"""
            <div class="status-box">
                ✅ <strong>{uploaded_file.name}</strong> processed successfully —
                {len(chunks)} chunks created and stored in ChromaDB.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="status-box">
                ✅ <strong>{uploaded_file.name}</strong> is ready. Ask your questions below.
            </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# --- SECTION 2: Ask a Question ---
st.markdown("### 02 — Ask a Question")

question = st.text_input(
    "Your question",
    placeholder="e.g. What is this document about?",
    label_visibility="collapsed",
    disabled=st.session_state.vector_store is None,
)

ask_button = st.button(
    "Ask",
    disabled=st.session_state.vector_store is None or not question.strip(),
)

if ask_button and question.strip():
    with st.spinner("Searching for answer..."):
        answer = generate(st.session_state.vector_store, question)

    st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">Answer</div>
            {answer}
        </div>
    """, unsafe_allow_html=True)

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
    })

if st.session_state.vector_store is None:
    st.caption("⬆️ Upload a document first to enable questions.")

# --- SECTION 3: Chat History ---
if st.session_state.chat_history:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 03 — Chat History")

    for item in reversed(st.session_state.chat_history):
        st.markdown(f"""
            <div class="chat-item">
                <div class="chat-question">Q: {item['question']}</div>
                <div class="chat-answer">{item['answer']}</div>
            </div>
        """, unsafe_allow_html=True)