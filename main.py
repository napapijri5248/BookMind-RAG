from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# create embedding model for user query
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# retrieve dataset from chroma-db
vectorstore=Chroma(
    persist_directory='chroma-db',
    embedding_function=embedding_model
)

# retriever
retriever=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k':4,
        'fetch_k':10,
        'lambda_mult':0.5
        # near 0-> more diverse results
        # near 1->less diverse results
    }
)

# making llm
model = ChatOllama(
    model="gpt-oss:120b-cloud",
    base_url="https://ollama.com"
)

# create database already ready

# prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ('system',
         """
        You are  a helpful AI assistant.
        Use only the provided context to answer the questions.
        If the answer is not present in the context,
        say "I could not find the answer in the document"
            """),
            (
                'human',
                '''
                Context:{context}
                Question:{question}
'''
            )
    ]
)

print("Rag system created.")
print('press 0 to exit')

while True:
    query=input("You : ")
    if query=="0":
        break

    docs=retriever.invoke(query)

    context='\n\n'.join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        'context':context,
        'question':query
    })

    response=model.invoke(final_prompt)

    print(f'\n AI : {response.content}')