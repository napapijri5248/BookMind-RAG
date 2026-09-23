import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama

docs=[
    Document(page_content="This is My laptop"),
    Document(page_content="My laptop is HP victus"),
    Document(page_content="my laptop is good"),
    Document(page_content="all laptops are good"),
    Document(page_content="i am a great guy"),
]

embeddings=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=Chroma.from_documents(docs,embeddings)

model = ChatOllama(
    model="gpt-oss:120b-cloud",
    base_url="https://ollama.com"
)

mqr_ret=vectorstore.as_retriever()

mqr=MultiQueryRetriever.from_llm(
    retriever=mqr_ret,
    llm=model
)

query="how is my laptop?"

docs=mqr_ret.invoke(query)
print('Retrieved docs \n')

for doc in docs:
    print(doc.page_content)




















# sim_ret=vectorstore.as_retriever(
#     search_type='similarity',
#     search_kwargs={'k':3}
# )

# print('SS search results : \n')

# sim_docs=sim_ret.invoke('How is my laptop?')

# for doc in sim_docs:
#     print(doc.page_content)


# mmr_ret=vectorstore.as_retriever(
#     search_type='mmr',
#     search_kwargs={'k':3}
# )

# print('MMR search results : \n')

# mmr_docs=mmr_ret.invoke('How is my laptop?')

# for doc in mmr_docs:
#     print(doc.page_content)