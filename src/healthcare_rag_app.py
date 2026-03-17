#!/usr/bin/env python3
"""
Healthcare RAG app: hybrid retrieval (Cypher + Vector) over HealthcareProvider graph.
Uses Ollama only: OllamaEmbeddings (nomic-embed-text, 768-dim), ChatOllama (llama3.2).
"""

import os
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from database import get_driver, get_database

load_dotenv()

# Ollama only (production healthcare config)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "5"))
TOP_K_CYPHER = int(os.getenv("TOP_K_CYPHER", "5"))

# -----------------------------------------------------------------------------
# Cypher: traverse (HealthcareProvider)-[:SPECIALIZES_IN]->(Specialization) etc.
# -----------------------------------------------------------------------------

CYPHER_PROVIDERS_BY_SPECIALIZATION = """
MATCH (p:HealthcareProvider)-[:SPECIALIZES_IN]->(s:Specialization)
WHERE toLower(s.name) CONTAINS toLower($query)
   OR toLower(p.name) CONTAINS toLower($query)
WITH p, s.name AS specialization
ORDER BY p.name
LIMIT $topK
RETURN p.id AS id, p.name AS name, p.bio AS bio, specialization
"""

CYPHER_PROVIDERS_BY_LOCATION = """
MATCH (p:HealthcareProvider)-[:LOCATED_AT]->(l:Location)
WHERE toLower(l.name) CONTAINS toLower($query)
WITH p
LIMIT $topK
RETURN p.id AS id, p.name AS name, p.bio AS bio
"""

CYPHER_ALL_PROVIDER_CONTEXT = """
MATCH (p:HealthcareProvider)
OPTIONAL MATCH (p)-[:SPECIALIZES_IN]->(s:Specialization)
OPTIONAL MATCH (p)-[:LOCATED_AT]->(l:Location)
WITH p, collect(DISTINCT s.name) AS specs, collect(DISTINCT l.name) AS locs
WHERE p.id IN $providerIds
RETURN p.id AS id, p.name AS name, p.bio AS bio, specs, locs
"""


def get_embedding_model():
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_llm():
    return ChatOllama(
        model=OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def vector_search_providers(session, query_embedding: list[float], top_k: int = TOP_K_VECTOR) -> list[dict]:
    """Vector similarity search on HealthcareProvider.bio_embedding (768-dim, cosine)."""
    def _run(tx):
        result = tx.run(
            """
            CALL db.index.vector.queryNodes('provider_bio_embedding_index', $k, $embedding)
            YIELD node, score
            RETURN node.id AS id, node.name AS name, node.bio AS bio, score
            ORDER BY score DESC
            """,
            k=top_k,
            embedding=query_embedding,
        )
        return [
            {"id": r["id"], "name": r["name"], "bio": r["bio"], "score": r["score"], "source": "vector"}
            for r in result
        ]
    return session.execute_read(_run)


def cypher_search_providers(session, query: str, top_k: int = TOP_K_CYPHER) -> list[dict]:
    """Cypher search: (HealthcareProvider)-[:SPECIALIZES_IN]->(Specialization) and location."""
    def _run(tx):
        seen = set()
        out = []
        # By specialization / provider name
        result = tx.run(
            CYPHER_PROVIDERS_BY_SPECIALIZATION,
            {"query": query, "topK": top_k},
        )
        for r in result:
            pid = r["id"]
            if pid not in seen:
                seen.add(pid)
                out.append({
                    "id": pid,
                    "name": r["name"],
                    "bio": r["bio"],
                    "score": 1.0,
                    "source": "cypher",
                    "specialization": r.get("specialization"),
                })
        # By location
        result2 = tx.run(
            CYPHER_PROVIDERS_BY_LOCATION,
            {"query": query, "topK": top_k},
        )
        for r in result2:
            pid = r["id"]
            if pid not in seen:
                seen.add(pid)
                out.append({
                    "id": pid,
                    "name": r["name"],
                    "bio": r["bio"],
                    "score": 1.0,
                    "source": "cypher",
                })
        return out[:top_k]

    return session.execute_read(_run)


def hybrid_retriever(
    session,
    query: str,
    embed_model,
    top_k_vector: int = TOP_K_VECTOR,
    top_k_cypher: int = TOP_K_CYPHER,
) -> list[Document]:
    """
    Hybrid retrieval: Vector search on provider Bio + Cypher over SPECIALIZES_IN / LOCATED_AT.
    Returns deduplicated LangChain Documents for RAG context.
    """
    seen = set()
    docs: list[Document] = []

    # 1) Vector search on bio_embedding
    query_emb = embed_model.embed_query(query)
    vector_hits = vector_search_providers(session, query_emb, top_k=top_k_vector)
    for h in vector_hits:
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        text = f"Provider: {h['name']}\nBio: {h['bio']}"
        docs.append(
            Document(
                page_content=text,
                metadata={"id": h["id"], "name": h["name"], "score": float(h["score"]), "source": "vector"},
            )
        )

    # 2) Cypher search (specialization, location, keyword in name)
    cypher_hits = cypher_search_providers(session, query, top_k=top_k_cypher)
    for h in cypher_hits:
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        text = f"Provider: {h['name']}\nBio: {h['bio']}"
        docs.append(
            Document(
                page_content=text,
                metadata={"id": h["id"], "name": h["name"], "source": "cypher"},
            )
        )

    return docs


def query_healthcare_rag(
    question: str,
    driver=None,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> str:
    """
    Run hybrid retrieval (Cypher + Vector), build context from provider bios, return LLM response.
    Uses Ollama only (nomic-embed-text, llama3.2).
    """
    password = password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD required for healthcare RAG")

    close_driver = driver is None
    driver = driver or get_driver(uri=uri, user=user, password=password)
    db = get_database()
    session = driver.session(database=db) if db else driver.session()

    try:
        embed = get_embedding_model()
        docs = hybrid_retriever(session, question, embed)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        if not context.strip():
            return "No relevant healthcare providers found in the knowledge base."

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a healthcare assistant. Answer the question using only the following provider context. If the context does not contain enough information, say so.\n\nContext:\n{context}"),
            ("human", "{question}"),
        ])
        chain = prompt | get_llm() | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
    finally:
        session.close()
        if close_driver:
            driver.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Healthcare RAG: ask questions over provider graph (Ollama + Neo4j)")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--ingest", metavar="CSV", help="Run ingest_healthcare on CSV first")
    args = parser.parse_args()

    if args.ingest:
        from ingest_healthcare import ingest_healthcare
        path = Path(args.ingest)
        if not path.exists():
            path = Path(__file__).resolve().parent / "healthcare" / "healthcare.csv"
        n = ingest_healthcare(str(path))
        print(f"Ingested {n} rows from {path}")
        if not args.question:
            return

    question = args.question or input("Question> ").strip()
    if not question:
        return
    print(query_healthcare_rag(question))


if __name__ == "__main__":
    main()
