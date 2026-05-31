import os
import tiktoken
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.ingestion.loader import load_dbt_files

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
_COST_PER_MILLION_TOKENS = 0.02


def build_vector_store(dbt_path: str) -> Chroma:
    """Embed all dbt chunks and persist them to Chroma. Returns the vector store."""
    load_dotenv()

    docs = load_dbt_files(dbt_path)

    total_tokens = _count_tokens(docs)
    estimated_cost = (total_tokens / 1_000_000) * _COST_PER_MILLION_TOKENS
    print(f"Chunks:          {len(docs)}")
    print(f"Total tokens:    {total_tokens:,}")
    print(f"Estimated cost:  ${estimated_cost:.6f}")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"Done. {len(docs)} chunks stored in {CHROMA_PATH}/")
    return vector_store


def _count_tokens(docs) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(doc.page_content)) for doc in docs)


if __name__ == "__main__":
    build_vector_store("data/jaffle_shop")
