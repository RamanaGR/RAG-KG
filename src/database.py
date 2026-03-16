"""
Neo4j connection utility and schema initialization for GraphRAG.
Supports Vector Index (embeddings on Chunk) and Graph structure (Chunk-[:MENTIONS]->Entity).
"""

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# Aura/Kocla often use NEO4J_USERNAME; fallback to NEO4J_USER then "neo4j"
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
# Optional: database name (e.g. for Neo4j Aura)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# Vector index dimension (must match embedding model output, e.g. text-embedding-3-small = 1536)
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1536"))


def get_driver(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> Driver:
    """Create and return a Neo4j driver instance. Supports Aura/Kocla (NEO4J_USERNAME, NEO4J_DATABASE)."""
    uri = uri or NEO4J_URI
    user = user or NEO4J_USER
    password = password or NEO4J_PASSWORD
    if not password:
        raise ValueError("NEO4J_PASSWORD must be set in environment or passed explicitly")
    return GraphDatabase.driver(uri, auth=(user, password))


def get_database() -> str | None:
    """Return the database name from env (for Aura/Kocla). None means default 'neo4j'."""
    return os.getenv("NEO4J_DATABASE") or None


@contextmanager
def neo4j_session(driver: Driver | None = None, database: str | None = None) -> Generator:
    """Context manager for a Neo4j session. Uses NEO4J_DATABASE when set (Aura/Kocla)."""
    _driver = driver or get_driver()
    db = database if database is not None else NEO4J_DATABASE
    session = _driver.session(database=db) if db else _driver.session()
    try:
        yield session
    finally:
        session.close()
        if driver is None:
            _driver.close()


def init_schema(session=None) -> None:
    """
    Initialize Neo4j schema: constraints and vector index for Chunk nodes.
    Chunk: id, text, source, embedding (vector).
    Entity: id, name.
    Relationship: (Chunk)-[:MENTIONS]->(Entity).
    """
    def _run(tx):
        # Uniqueness constraints
        tx.run("""
            CREATE CONSTRAINT chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
        """)
        tx.run("""
            CREATE CONSTRAINT entity_id IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.id IS UNIQUE
        """)
        # Vector index for similarity search on Chunk.embedding
        tx.run("""
            CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
            }}
        """, dim=VECTOR_DIMENSION)

    if session is not None:
        session.execute_write(_run)
        return
    with neo4j_session() as s:
        s.execute_write(_run)


def verify_connection(driver: Driver | None = None) -> bool:
    """Verify Neo4j connectivity. Returns True if successful."""
    _driver = driver or get_driver()
    try:
        _driver.verify_connectivity()
        return True
    except Exception:
        return False
    finally:
        if driver is None:
            _driver.close()
