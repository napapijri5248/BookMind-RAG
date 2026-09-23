# 📚 BookMind-RAG

> An end-to-end, modular Retrieval-Augmented Generation (RAG) assistant and experimentation suite built with LangChain, ChromaDB, HuggingFace embeddings, and Ollama.

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-red.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 🌟 Overview

**BookMind-RAG** transforms how you interact with books and complex documents. It lets you upload any PDF, extracts and indexes its contents into a vector database, and allows you to ask questions grounded strictly in the document's content with zero hallucinations.

In addition to the interactive web application, this repository includes a full **RAG experimentation lab** exploring different document loaders, chunking strategies, vector databases, and advanced retrieval techniques (Similarity Search, Maximal Marginal Relevance, Multi-Query expansion, and arXiv academic retrieval).

---

## 🚀 Key Features

- **Interactive Web Interface (`app.py`)**:
  - Full-stack Streamlit application for uploading PDF books.
  - Automated chunking, embedding, and indexing into ChromaDB.
  - Conversational chat interface with memory and grounded response generation.
  - Sidebar analytics (page count, chunk count, model configuration).

- **Persistent CLI Pipeline (`createDB.py` & `main.py`)**:
  - Decoupled ingestion script that embeds and persists documents into a local `chroma-db/` folder.
  - Interactive terminal Q&A loop querying the persisted database with MMR retrieval.

- **Advanced Retrieval Strategies (`retrievers/`)**:
  - **MMR (Maximal Marginal Relevance)**: Balances query relevance with diversity to eliminate redundant context.
  - **Multi-Query Retrieval**: Uses an LLM to generate multiple perspectives of a user query to overcome distance-metric limitations.
  - **ArXiv Retriever**: Connects to the arXiv academic repository to retrieve research papers with metadata.

- **Document Loaders & Parsers (`document_loaders/`)**:
  - Tested with PDF documents (`PyPDFLoader`), web pages (`WebBaseLoader`), and raw text files (`TextLoader`).
  - Tuned `RecursiveCharacterTextSplitter` with configurable chunk size and overlap.

---

## 📐 Architecture

```
User Query / PDF Document
         │
         ▼
┌──────────────────┐
│  Document Loader │ (PyPDFLoader / WebBaseLoader / TextLoader)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Text Splitter   │ (RecursiveCharacterTextSplitter: 1000 chunk size, 200 overlap)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Embedding Model  │ (sentence-transformers/all-MiniLM-L6-v2)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Vector Store    │ (ChromaDB)
└────────┬─────────┘
         ▼
┌──────────────────┐
│    Retriever     │ (MMR: k=4, fetch_k=10, lambda_mult=0.5)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Augmented Prompt │ (ChatPromptTemplate + Grounded Constraints)
└────────┬─────────┘
         ▼
┌──────────────────┐
│       LLM        │ (ChatOllama / gpt-oss:120b-cloud)
└────────┬─────────┘
         ▼
   Final Answer
```

---

## 📁 Repository Structure

```text
BookMind-RAG/
├── app.py                     # Streamlit web assistant application
├── createDB.py                # Ingestion script: embeds PDF into chroma-db/
├── main.py                    # CLI terminal chat querying chroma-db/
├── main2.py                   # Standalone LLM summarization experiment
├── document_loaders/          # Document loading experiments
│   ├── page.py                # WebBaseLoader URL scraping
│   ├── pdf.py                 # PyPDFLoader testing
│   ├── test.py                # TextLoader and recursive splitting tests
│   ├── notes.txt              # Conceptual RAG notes
│   └── ts_notes.txt           # Sample text file
├── retrievers/                # Retrieval algorithm implementations
│   ├── mmr.py                 # Similarity Search vs. MMR comparison
│   ├── multiquery.py          # Multi-query expansion retriever
│   └── arixv.py               # arXiv paper retrieval
├── vector_store/              # Vector storage fundamentals
│   └── DB.py                  # ChromaDB document indexing & similarity search
├── .env.example               # Template for environment variables
├── .gitignore                 # Excludes secrets, databases, and binaries
├── pyproject.toml             # UV / Python packaging configuration
├── requirements.txt           # Full dependencies list
└── README.md                  # Project documentation
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python `>= 3.10` (Python 3.13 recommended)
- [Ollama](https://ollama.com/) installed and running (or access to an Ollama endpoint)

### 2. Clone the Repository
```bash
git clone https://github.com/napapijri5248/BookMind-RAG.git
cd BookMind-RAG
```

### 3. Install Dependencies

Using `pip`:
```bash
pip install -r requirements.txt
```

Or using `uv`:
```bash
uv sync
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your respective API keys if using external providers (Groq, HuggingFace, Ollama Cloud).

---

## 💻 Running the Application

### Option A: Streamlit Web UI (Recommended)
Launch the interactive web assistant:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, upload any PDF, and start asking questions!

### Option B: Command-Line Interface (CLI)
1. Ingest a document and create the database:
   ```bash
   python createDB.py
   ```
2. Start the interactive chat loop:
   ```bash
   python main.py
   ```
   *(Type your question, or enter `0` to exit.)*

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/napapijri5248/BookMind-RAG/issues).

---

## 📜 Author

Developed with ❤️ by [napapijri5248](https://github.com/napapijri5248).
