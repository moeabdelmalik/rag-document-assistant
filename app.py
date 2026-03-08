import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load the API key from the .env file
load_dotenv()

# --- FUNCTION 1: LOAD ---
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT files are supported.")
    documents = loader.load()
    return documents

# --- FUNCTION 2: CHUNK ---
def chunk_document(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

# --- FUNCTION 3: EMBED AND STORE ---
def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

# --- FUNCTION 4: RETRIEVE ---
def retrieve(vector_store, question):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    relevant_chunks = retriever.invoke(question)
    return relevant_chunks

# --- FUNCTION 5: GENERATE ---
def generate(vector_store, question):
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )
    prompt_template = """
    Use the following context to answer the question.
    If you don't know the answer, just say "I don't know."
    Don't make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    answer = chain.invoke({"query": question})
    return answer["result"]


# --- STREAMLIT UI ---

# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        .main {
            background-color: #0f0f0f;
        }

        .stApp {
            background-color: #0f0f0f;
            color: #f0f0f0;
        }

        h1, h2, h3 {
            font-family: 'Syne', sans-serif !important;
        }

        .title-block {
            text-align: center;
            padding: 2rem 0 1rem 0;
        }

        .title-block h1 {
            font-size: 2.8rem;
            font-weight: 800;
            color: #f0f0f0;
            letter-spacing: -1px;
            margin-bottom: 0.3rem;
        }

        .title-block p {
            color: #888;
            font-size: 1rem;
            font-weight: 300;
        }

        .accent {
            color: #00e5a0;
        }

        .status-box {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-left: 3px solid #00e5a0;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: #ccc;
        }

        .answer-box {
            background: #151515;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-top: 1.2rem;
            color: #f0f0f0;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        .answer-label {
            font-family: 'Syne', sans-serif;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #00e5a0;
            margin-bottom: 0.6rem;
        }

        .chat-history {
            margin-top: 2rem;
        }

        .chat-item {
            background: #1a1a1a;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            border: 1px solid #222;
        }

        .chat-question {
            font-size: 0.8rem;
            color: #888;
            margin-bottom: 0.3rem;
            font-weight: 500;
        }

        .chat-answer {
            color: #f0f0f0;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .divider {
            border: none;
            border-top: 1px solid #222;
            margin: 2rem 0;
        }

        /* Streamlit widget overrides */
        .stFileUploader > div {
            background: #1a1a1a !important;
            border: 1px dashed #333 !important;
            border-radius: 10px !important;
        }

        .stTextInput > div > div > input {
            background: #1a1a1a !important;
            border: 1px solid #333 !important;
            color: #f0f0f0 !important;
            border-radius: 8px !important;
        }

        .stButton > button {
            background: #00e5a0 !important;
            color: #0f0f0f !important;
            font-family: 'Syne', sans-serif !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 2rem !important;
            width: 100% !important;
            transition: opacity 0.2s !important;
        }

        .stButton > button:hover {
            opacity: 0.85 !important;
        }

        .stSpinner > div {
            border-top-color: #00e5a0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="title-block">
        <h1>📄 RAG <span class="accent">Assistant</span></h1>
        <p>Upload a document. Ask anything about it.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Initialize session state
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
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Only reprocess if a new file is uploaded
    if st.session_state.doc_name != uploaded_file.name:
        with st.spinner("Reading and processing your document..."):
            # Save uploaded file to a temp location
            suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Run the RAG pipeline
            documents = load_document(tmp_path)
            chunks = chunk_document(documents)
            vector_store = embed_and_store(chunks)

            # Save to session state
            st.session_state.vector_store = vector_store
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chat_history = []

            # Clean up temp file
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
    disabled=st.session_state.vector_store is None
)

ask_button = st.button(
    "Ask",
    disabled=st.session_state.vector_store is None or not question.strip()
)

if ask_button and question.strip():
    with st.spinner("Searching for answer..."):
        answer = generate(st.session_state.vector_store, question)

    # Show current answer
    st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">Answer</div>
            {answer}
        </div>
    """, unsafe_allow_html=True)

    # Save to chat history
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer
    })

# Placeholder if no document uploaded
if st.session_state.vector_store is None:
    st.caption("⬆️ Upload a document first to enable questions.")

# --- SECTION 3: Chat History ---
if st.session_state.chat_history:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 03 — Chat History")

    for i, item in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"""
            <div class="chat-item">
                <div class="chat-question">Q: {item['question']}</div>
                <div class="chat-answer">{item['answer']}</div>
            </div>
        """, unsafe_allow_html=True)