import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# data = PyPDFLoader(r"document_loaders\\DL.pdf")
# docs=data.load()

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks=splitter.split_documents(docs)

model = ChatOllama(
    model="gpt-oss:120b-cloud",
    base_url="https://ollama.com",
    temperature=0.9,
    num_predict=500
)

# # final_prompt=template.format_messages(data=docs[0].page_content)
# final_prompt=template.format_messages(data=docs)

# response = model.invoke(final_prompt)

# print(response.content)


template=ChatPromptTemplate.from_messages(
    [('system','You are a AI that summarizes the text'),
     ('human','{data}')]
)
