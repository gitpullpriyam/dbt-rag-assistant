import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.ingestion.embedder import CHROMA_PATH, EMBEDDING_MODEL


def load_vector_store() -> Chroma:
    load_dotenv()
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )


def retrieve(question: str, k: int = 3, where: dict = None) -> list[Document]:
    """Embed question and return top-k most relevant chunks from Chroma."""
    vector_store = load_vector_store()
    return vector_store.similarity_search(question, k=k, filter=where)

#test retrieve function checking if it returns the expected number of results.
# if __name__ == "__main__":
#     results = retrieve("What is the purpose of a dbt model?", k=3)
#     for doc in results:
#         print(doc.page_content)
