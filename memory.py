from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

PERSIST_DIR = "./memory_db"

embeddings = OllamaEmbeddings(model="llama3.2")

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

def store_memory(text: str):
    vectorstore.add_texts([text])
    vectorstore.persist()

def recall_memories(query: str, k: int = 3):
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]