"""
Core RAG pipeline: load -> chunk -> embed/store -> retrieve -> generate.
No Streamlit imports here on purpose, so this file can be reused or tested
independently of the UI.
"""

import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

RAG_PROMPT = ChatPromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer, just say "I don't know."
Don't make up answers.

Context:
{context}

Question:
{question}
""")


def load_document(file_path: str):
    """Load a PDF or TXT file into LangChain Document objects."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT files are supported.")
    return loader.load()


def chunk_document(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def embed_and_store(chunks, persist_directory: str = "./chroma_db"):
    """Embed chunks locally and store them in a Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def retrieve(vector_store, question: str, k: int = 3):
    """Fetch the top-k most relevant chunks for a question."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def generate(vector_store, question: str, model_name: str = "llama-3.3-70b-versatile", k: int = 3) -> str:
    """Run the full retrieve -> prompt -> LLM chain and return the answer text."""
    llm = ChatGroq(
        model_name=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)