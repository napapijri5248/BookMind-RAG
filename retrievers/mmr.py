import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

docs=[
    Document(page_content="This is My laptop"),
    Document(page_content="My laptop is HP victus"),
    Document(page_content="my laptop is good"),
    Document(page_content="all laptops are good"),
    Document(page_content="i am a great guy"),
]

embeddings=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=Chroma.from_documents(docs,embeddings)

sim_ret=vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':3}
)

print('SS search results : \n')

sim_docs=sim_ret.invoke('How is my laptop?')

for doc in sim_docs:
    print(doc.page_content)


mmr_ret=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3}
)

print('MMR search results : \n')

mmr_docs=mmr_ret.invoke('How is my laptop?')

for doc in mmr_docs:
    print(doc.page_content)