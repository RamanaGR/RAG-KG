# RAG-KG — GraphRAG with Neo4j

A **Graph Retrieval-Augmented Generation (GraphRAG)** boilerplate using **Python**, **Neo4j**, and **LangChain**. It combines **vector search** (embeddings on chunks) with **graph traversal** (e.g. `Chunk`-[:MENTIONS]->`Entity`) to improve retrieval and answer quality.

---

## Tech stack

| Component        | Technology                                      |
|-----------------|--------------------------------------------------|
| Database         | Neo4j (local or Aura/Kocla)                      |
| Orchestration   | LangChain                                        |
| Embeddings / LLM | OpenAI or **Ollama** (configurable via `.env`)   |
| Config          | python-dotenv                                    |
| Documents       | PDF and plain text (.txt)                        |

---

## Directory structure

```
RAG-KG/
├── .env              # Your secrets (copy from .env.example)
├── .env.example      # Template for NEO4J_*, OPENAI_*, OLLAMA_*, etc.
├── main.py           # CLI: ask questions, optional --ingest
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── database.py   # Neo4j connection, schema (constraints + vector index)
    ├── ingest.py     # Load PDF/Text → chunk → embed → upsert to Neo4j
    ├── retrieval.py  # Hybrid search (vector + Cypher graph) + RAG chain
    ├── ingest_healthcare.py   # Ingest healthcare CSV → Neo4j with 768-dim vector index
    ├── healthcare_rag_app.py  # Healthcare RAG: hybrid Cypher + Vector, Ollama only
    ├── healthcare/
    │   ├── __init__.py
    │   ├── schema.py  # Healthcare constraints + 768-dim vector index for provider bios
    │   └── healthcare.csv  # Provider, Patient, Specialization, Location, Bio
    └── rag_with_kg/  # Sample scripts (vector_db, health_care_kg, health_care_lc_retrieval)
```

---

## Setup

### 1. Clone and virtual environment

```bash
cd RAG-KG
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env with your values.
```

| Variable | Description |
|----------|-------------|
| `NEO4J_URI` | Neo4j connection (e.g. `bolt://localhost:7687` or `neo4j+s://...` for Aura) |
| `NEO4J_USER` / `NEO4J_USERNAME` | Username (Aura often uses `NEO4J_USERNAME`) |
| `NEO4J_PASSWORD` | Password |
| `NEO4J_DATABASE` | Optional; required for some Aura setups |
| `LLM_PROVIDER` | `openai` or `ollama` |
| **OpenAI** | `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_LLM_MODEL` |
| **Ollama** | `OLLAMA_EMBEDDING_MODEL` (e.g. `nomic-embed-text`), `OLLAMA_LLM_MODEL` (e.g. `llama3.2`) |
| `VECTOR_DIMENSION` | Must match embedding model (e.g. `768` for nomic-embed-text, `1536` for OpenAI) |

### 3. Neo4j

- **Local:** Run Neo4j and ensure the vector index can be created (Neo4j 5.11+).
- **Aura/Kocla:** Use `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` as provided.

### 4. Ollama (optional)

If using `LLM_PROVIDER=ollama`:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## Usage

### Ingest documents

Ingest a PDF or text file into Neo4j (chunks + embeddings + optional entity graph):

```bash
python -m src.ingest path/to/file.pdf
# or
python -m src.ingest path/to/file.txt
```

Or via the CLI:

```bash
python main.py --ingest path/to/file.pdf
```

### Ask questions (CLI)

```bash
# Interactive mode
python main.py

# Single question
python main.py "Your question here?"

# Ingest then ask
python main.py --ingest doc.pdf "What is this document about?"
```

---

## How it works

1. **Ingest (`src/ingest.py`):** Loads PDF/Text → chunks with overlap → embeds (OpenAI or Ollama) → upserts **Chunk** nodes and optional **Entity** nodes with **(Chunk)-[:MENTIONS]->(Entity)**.
2. **Retrieval (`src/retrieval.py`):** Embeds the question → **vector search** on `Chunk.embedding` → **graph expansion** via Cypher: chunks that share **Entity** nodes with the top vector hits are added for context.
3. **Answer:** Retrieved chunks are passed as context to the LLM (OpenAI or Ollama); the model returns an answer.

---

## Healthcare RAG (branch `dev-healthcare`)

Production-ready **Healthcare RAG** on branch `dev-healthcare`: hybrid retrieval over a provider graph using **Ollama only** (nomic-embed-text, llama3.2) and a **768-dimension** vector index.

### Schema

- **Nodes:** `HealthcareProvider` (name, bio, bio_embedding), `Patient`, `Specialization`, `Location`
- **Relationships:** `(HealthcareProvider)-[:SPECIALIZES_IN]->(Specialization)`, `-[:LOCATED_AT]->(Location)`, `-[:TREATS]->(Patient)`

### Setup

- Use Neo4j credentials (e.g. from `note-neo4j-creds.md`). Set `NEO4J_URI`, `NEO4J_USER`/`NEO4J_USERNAME`, `NEO4J_PASSWORD`, and optionally `NEO4J_DATABASE`.
- Set `OLLAMA_BASE_URL` if Ollama is not at `http://localhost:11434`.
- Run `ollama serve` and pull: `ollama pull nomic-embed-text` and `ollama pull llama3.2`.

### Ingest

```bash
# Ingest src/healthcare/healthcare.csv (Provider, Patient, Specialization, Location, Bio)
python -m src.ingest_healthcare
# Or: python -m src.ingest_healthcare path/to/healthcare.csv
```

### Run Healthcare RAG

```bash
python -m src.healthcare_rag_app "Find a cardiologist"
# Interactive: python -m src.healthcare_rag_app
# Ingest then ask: python -m src.healthcare_rag_app --ingest src/healthcare/healthcare.csv "Who specializes in neurology?"
```

### Hybrid retriever

- **Cypher:** Traverses `(HealthcareProvider)-[:SPECIALIZES_IN]->(Specialization)` and `-[:LOCATED_AT]->(Location)`; keyword match on specialization, location, provider name.
- **Vector:** Similarity search on `HealthcareProvider.bio_embedding` (768-dim, Nomic).

---

## License

Use and modify as needed for your project.
