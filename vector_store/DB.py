from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs=[
    Document(page_content="Python is widely used in AI",metadata={'source':'AI_book'}),
    Document(page_content="Pandas is used for data analysis in PYTHON",metadata={'source':'DataScience_book'}),
    Document(page_content="Neural networks are used in DL",metadata={'source':'DL_book'})

]

# from langchain_ollama import OllamaEmbeddings

# embeddings = OllamaEmbeddings()
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory='chroma-db'
)

# RETRIEVERS
result=vectorstore.similarity_search('what is used for data analysis?',k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)

retriever=vectorstore.as_retriever()
docs=retriever.invoke('Explain deep learning')

for d in docs:
    print(d.page_content)