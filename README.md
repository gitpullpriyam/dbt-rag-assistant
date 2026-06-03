# dbt RAG Assistant

A Retrieval-Augmented Generation (RAG) system for natural-language Q&A over a dbt project.
Ask questions like *"what does the customers model do?"* or *"where does the orders table come from?"*
and get grounded answers with citations to the source SQL and YAML files.

---

## How It Works

```
data/jaffle_shop/     →  raw dbt files (SQL models, YAML docs)
src/ingestion/        →  load files, chunk, attach metadata
src/retrieval/        →  embed question, find top-k relevant chunks
src/generation/       →  build prompt, call LLM, return cited answer
src/app/              →  Streamlit UI tying it all together
```

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Primary language |
| Orchestration | LangChain | Clean abstractions for chaining retrieval and generation |
| Vector store | Chroma (local) | Zero infra, persists to disk |
| Embeddings | OpenAI `text-embedding-3-small` | Low cost, strong semantic quality |
| Generation | OpenAI `gpt-4o-mini` | Fast, capable, cost-efficient |
| UI | Streamlit | Rapid prototyping, zero frontend code |

---

## Setup

```bash
git clone <repo-url>
cd dbt-rag-assistant

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OpenAI API key
```

**Build the vector store** (one-time, re-run when the corpus changes):

```bash
python -m src.ingestion.embedder
```

**Run the app:**

```bash
python -m streamlit run src/app/app.py
```

---

## Project Scope

**Phase 1 — Core Pipeline**
- Ingestion: walk dbt project, chunk SQL models and YAML schema files, tag metadata
- Embeddings: encode chunks with OpenAI `text-embedding-3-small`, persist to Chroma
- Retrieval: semantic search over vector store, configurable top-k, metadata filtering
- Generation: grounded prompt construction, cited answers via `gpt-4o-mini`
- Interface: Streamlit Q&A app with expandable source citations

**Phase 2 — Evaluation**
- RAG triad scoring (context relevance, groundedness, answer relevance) via Ragas
- Hand-built eval set across query types, before/after benchmark comparisons
