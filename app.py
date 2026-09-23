
import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SESSION STATE
# ==========================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================
# LOAD LLM
# ==========================================

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="gpt-oss:120b-cloud",
        base_url="https://ollama.com"
    )


embedding_model = load_embedding_model()
model = load_llm()


# ==========================================
# PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the question using ONLY the provided context.

            If the answer is not present in the context,
            say exactly:

            I could not find the answer in the document.

            Do not make up information.

            Context:
            {context}
            """
        ),
        (
            "human",
            """
            Question:
            {question}
            """
        )
    ]
)


# ==========================================
# PROCESS PDF
# ==========================================

def process_pdf(uploaded_file):

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    pdf_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    # Save uploaded PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # Create embeddings and vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="book_collection"
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    return vectorstore, retriever, len(docs), len(chunks)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">📚 Book RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a book and ask questions using AI</div>',
    unsafe_allow_html=True
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("📖 Upload Your Book")

    uploaded_file = st.file_uploader(
        "Choose a PDF book",
        type=["pdf"]
    )

    process_button = st.button(
        "🔄 Process Book",
        use_container_width=True
    )

    st.divider()

    st.subheader("⚙️ Configuration")

    st.write("**Embedding Model:**")
    st.code("all-MiniLM-L6-v2")

    st.write("**LLM:**")
    st.code("gpt-oss:120b-cloud")

    st.write("**Retriever:**")
    st.code("MMR")

    st.write("**Chunk Size:** 1000")
    st.write("**Chunk Overlap:** 200")

    st.divider()

    if st.session_state.book_name:

        st.success(
            f"Current Book:\n\n{st.session_state.book_name}"
        )

    if st.button(
        "🗑️ Clear Current Book",
        use_container_width=True
    ):

        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.book_name = None
        st.session_state.messages = []

        st.rerun()


# ==========================================
# PROCESS UPLOADED BOOK
# ==========================================

if process_button:

    if uploaded_file is None:

        st.warning("Please upload a PDF book first.")

    else:

        if uploaded_file.size == 0:

            st.error("The uploaded file is empty.")

        else:

            with st.spinner(
                "Processing book... This may take some time."
            ):

                try:

                    (
                        vectorstore,
                        retriever,
                        page_count,
                        chunk_count
                    ) = process_pdf(uploaded_file)

                    st.session_state.vectorstore = vectorstore

                    st.session_state.retriever = retriever

                    st.session_state.book_name = uploaded_file.name

                    st.session_state.messages = []

                    st.success(
                        f"Successfully processed: {uploaded_file.name}"
                    )

                    st.info(
                        f"Pages: {page_count} | "
                        f"Chunks: {chunk_count}"
                    )

                except Exception as e:

                    st.error(
                        f"Error while processing PDF: {str(e)}"
                    )


# ==========================================
# CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# CHAT INPUT
# ==========================================

query = st.chat_input(
    "Ask a question about your book..."
)


if query:

    if st.session_state.retriever is None:

        st.warning(
            "Please upload and process a book before asking questions."
        )

    else:

        # Display user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):

            st.markdown(query)

        # Retrieve documents
        with st.chat_message("assistant"):

            with st.spinner("Searching the book..."):

                try:

                    docs = st.session_state.retriever.invoke(query)

                    context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in docs
                        ]
                    )

                    # Generate prompt
                    final_prompt = prompt.invoke(
                        {
                            "context": context,
                            "question": query
                        }
                    )

                    # Invoke LLM
                    response = model.invoke(final_prompt)

                    answer = response.content

                    st.markdown(answer)

                    # Save assistant response
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Error while generating answer: {str(e)}"
                    )