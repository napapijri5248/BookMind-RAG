import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader(r"document_loaders\\Technology Development Program- FTE- 2027.pdf")

docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10,
)

chunks=splitter.split_documents(docs)
print(len(chunks))
for i in chunks:
    print(i.page_content)
    print('---------------')
# print(docs[2].page_content)