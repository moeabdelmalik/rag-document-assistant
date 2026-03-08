#  RAG Document Assistant

A simple AI-powered document assistant built with Python. Upload a PDF or TXT file and ask questions about it — the assistant will answer based on the content of your document.

Built as a practice project to learn AI engineering concepts like Retrieval-Augmented Generation (RAG).

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Instead of asking an LLM to guess answers, we give it relevant content from your document to base its answers on.

The pipeline works like this:

```
Load document → Chunk text → Embed chunks → Store in ChromaDB
                                    ↓
             User asks a question → Retrieve relevant chunks → Groq generates answer
```

---

##  Features

- Upload PDF or TXT documents
- Automatically chunks and embeds your document
- Stores vectors locally using ChromaDB
- Retrieves the most relevant chunks for each question
- Generates accurate answers using Groq's LLaMA model
- Clean dark UI built with Streamlit
- Chat history saved during the session

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| LangChain | RAG pipeline framework |
| ChromaDB | Vector database |
| HuggingFace `all-MiniLM-L6-v2` | Embedding model |
| Groq `llama-3.3-70b-versatile` | LLM for answer generation |
| Streamlit | UI framework |

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/moeabdelmalik/rag-document-assistant.git
cd rag-document-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

##  How to Use

1. Open the app in your browser
2. Upload a PDF or TXT file
3. Wait for the document to be processed
4. Type a question about your document
5. Get an answer powered by Groq! ✅

---

##  Project Structure

```
rag-document-assistant/
├── app.py              # Main application — full RAG pipeline + Streamlit UI
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── .env                # API key (never pushed to GitHub)
├── .gitignore          # Ignores .env and chroma_db
└── chroma_db/          # Local vector database (auto-created on first run)
```

---

##  Dependencies

```
langchain
langchain-community
langchain-groq
langchain-core
langchain-text-splitters
langchain-huggingface
chromadb
python-dotenv
pypdf
sentence-transformers
streamlit
```

---

##  Author

Built by **Moe Abdelmalik** as a practice project for learning AI engineering.

---

##  License

This project is open source and available under the [MIT License](LICENSE).