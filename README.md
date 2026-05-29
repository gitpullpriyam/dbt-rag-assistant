# dbt RAG Assistant

A Retrieval-Augmented Generation (RAG) system for natural-language Q&A over a dbt project.
Ask questions like *"what does the customers model do?"* or *"where does the orders table come from?"*
and get grounded answers with citations to the source SQL and YAML files.

Built as a portfolio + learning project alongside DeepLearning.AI's
*Building and Evaluating Advanced RAG Applications*.

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
| Orchestration | LangChain | Portfolio-backed |
| Vector store | Chroma (local) | Zero infra, persists to disk |
| Embeddings | OpenAI `text-embedding-3-small` | Cheap, good enough |
| Generation | OpenAI `gpt-4o-mini` | Cheap, capable |
| UI | Streamlit | Fastest path to a demo |

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

---

## Roadmap

- [x] Module 1 — Project setup & security foundations
- [ ] Module 2 — Understanding RAG & the dbt corpus
- [ ] Module 3 — Ingestion: loading & chunking
- [ ] Module 4 — Embeddings & vector storage
- [ ] Module 5 — Retrieval
- [ ] Module 6 — Generation & prompt engineering
- [ ] Module 7 — Streamlit app
- [ ] Module 8 — Evaluation (Phase 2)
