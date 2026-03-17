#!/usr/bin/env python3
"""
Ingest healthcare CSV into Neo4j: Provider, Patient, Specialization, Location, Bio.
Creates HealthcareProvider nodes with bio_embedding (768-dim, Nomic), and relationships
SPECIALIZES_IN, LOCATED_AT, TREATS. Uses Ollama (nomic-embed-text) only.
"""

import csv
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings

from database import get_driver, get_database
from healthcare.schema import init_healthcare_schema, HEALTHCARE_VECTOR_DIMENSION

load_dotenv()

OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _provider_id(name: str) -> str:
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:24]


def _patient_id(name: str) -> str:
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:24]


def load_healthcare_csv(path: str) -> list[dict]:
    """Load healthcare CSV with columns Provider, Patient, Specialization, Location, Bio."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): (v.strip() if v else "").strip() for k, v in row.items()}
            if row.get("Provider") and row.get("Bio"):
                rows.append(row)
    return rows


def upsert_healthcare(
    driver,
    rows: list[dict],
    embeddings: list[list[float]],
) -> int:
    """
    Upsert HealthcareProvider, Specialization, Location, Patient and relationships.
    Each row gets one Provider (merged by id), one Bio embedding; Specialization, Location, Patient merged by name/id.
    """
    if len(rows) != len(embeddings):
        raise ValueError("rows and embeddings length must match")

    # Dedupe providers by id and keep one embedding per provider (first occurrence)
    provider_embeddings: dict[str, list[float]] = {}
    for row, emb in zip(rows, embeddings):
        pid = _provider_id(row["Provider"])
        if pid not in provider_embeddings:
            provider_embeddings[pid] = emb

    def _tx(tx):
        n = 0
        for row, emb in zip(rows, embeddings):
            provider_name = row["Provider"]
            patient_name = row["Patient"]
            spec_name = row["Specialization"]
            loc_name = row["Location"]
            bio = row["Bio"]
            pid = _provider_id(provider_name)
            patient_id = _patient_id(patient_name)

            # Use first embedding we stored for this provider
            use_emb = provider_embeddings[pid]

            tx.run(
                """
                MERGE (p:HealthcareProvider {id: $pid})
                SET p.name = $name, p.bio = $bio, p.bio_embedding = $embedding
                """,
                pid=pid,
                name=provider_name,
                bio=bio,
                embedding=use_emb,
            )
            tx.run(
                """
                MERGE (s:Specialization {name: $name})
                WITH s
                MATCH (p:HealthcareProvider {id: $pid})
                MERGE (p)-[:SPECIALIZES_IN]->(s)
                """,
                name=spec_name,
                pid=pid,
            )
            tx.run(
                """
                MERGE (l:Location {name: $name})
                WITH l
                MATCH (p:HealthcareProvider {id: $pid})
                MERGE (p)-[:LOCATED_AT]->(l)
                """,
                name=loc_name,
                pid=pid,
            )
            tx.run(
                """
                MERGE (pt:Patient {id: $patient_id})
                SET pt.name = $name
                WITH pt
                MATCH (p:HealthcareProvider {id: $pid})
                MERGE (p)-[:TREATS]->(pt)
                """,
                patient_id=patient_id,
                name=patient_name,
                pid=pid,
            )
            n += 1
        return n

    # Neo4j 5/6: execute_write is on the session, not the driver
    db = get_database()
    session = driver.session(database=db) if db else driver.session()
    try:
        return session.execute_write(_tx)
    finally:
        session.close()


def ingest_healthcare(
    csv_path: str,
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    vector_dim: int = HEALTHCARE_VECTOR_DIMENSION,
) -> int:
    """
    Load healthcare CSV, embed Bio with Ollama (nomic-embed-text), create 768-dim vector index,
    and upsert graph. Returns number of rows processed.
    """
    from database import NEO4J_URI, NEO4J_USER

    uri = uri or os.getenv("NEO4J_URI", NEO4J_URI)
    user = user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD required for ingest")

    driver = get_driver(uri=uri, user=user, password=password)
    db = get_database()
    session = driver.session(database=db) if db else driver.session()
    try:
        init_healthcare_schema(session, vector_dim=vector_dim)
    finally:
        session.close()

    rows = load_healthcare_csv(csv_path)
    if not rows:
        driver.close()
        return 0

    embed = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
    )
    bios = [r["Bio"] for r in rows]
    embeddings = embed.embed_documents(bios)
    count = upsert_healthcare(driver, rows, embeddings)
    driver.close()
    return count


if __name__ == "__main__":
    import sys
    default_path = Path(__file__).resolve().parent / "healthcare" / "healthcare.csv"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default_path)
    n = ingest_healthcare(path)
    print(f"Ingested {n} healthcare rows from {path}")
