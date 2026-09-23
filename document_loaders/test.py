import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader

# from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# splitter=CharacterTextSplitter(
#     separator="",
#     chunk_size=1000,
#     chunk_overlap=0,
# )
splitter=RecursiveCharacterTextSplitter(
    # separator="",
    chunk_size=10,
    chunk_overlap=1,
)

data = TextLoader(r"document_loaders\\ts_notes.txt")

docs=data.load()

chunks=splitter.split_documents(docs)

print(len(chunks))
for i in chunks:
    print(i.page_content)
    print()
# print(chunks)
# print(docs[0].page_content)