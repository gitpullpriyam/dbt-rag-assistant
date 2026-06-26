import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src.retrieval.retriever import retrieve

GENERATION_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = """You are an assistant that answers questions about a dbt project.
Answer ONLY using the context provided below. Cite the source file for each piece of information you use.
If the context does not contain enough information to answer, say "I don't know based on the provided context. Can you elaborate or provide more details?"
Do not use any knowledge outside of the provided context."""


def _build_context(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("file_path", "unknown")
        sections.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(sections)


def answer(question: str, k: int = 5) -> dict:
    """Retrieve relevant chunks and generate a grounded, cited answer."""
    load_dotenv()

    docs = retrieve(question, k=k)
    context = _build_context(docs)

    prompt = f"{context}\n\nQuestion: {question}"

    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [doc.metadata["file_path"] for doc in docs],
    }

#test answer function checking if it returns an answer and sources as per instruction and logic.
if __name__ == "__main__":
    result = answer("How is customer_lifetime_value calculated?")
    print(result["answer"])
    print("\nSources:", result["sources"])
