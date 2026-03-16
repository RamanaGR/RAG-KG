"""
Ingest PDF or text files: load, chunk, embed, and upsert into Neo4j.
Creates Chunk nodes with embeddings and optional (Chunk)-[:MENTIONS]->(Entity) graph.
"""

import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from src.database import get_driver, init_schema, get_database, NEO4J_URI, NEO4J_USER

load_dotenv()

# Default chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embedding provider: "openai" or "ollama" (follows LLM_PROVIDER when not set)
EMBEDDING_PROVIDER = (os.getenv("EMBEDDING_PROVIDER") or os.getenv("LLM_PROVIDER", "openai")).lower()
EMBEDDING_MODEL = (
    os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    if EMBEDDING_PROVIDER == "ollama"
    else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)


def load_documents(path: str) -> list[Document]:
    """Load documents from a file. Supports .pdf and .txt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    elif suffix in (".txt", ".text", ""):
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")
    for d in docs:
        d.metadata.setdefault("source", str(path))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _chunk_id(text: str, source: str, index: int) -> str:
    """Stable id for a chunk for upsert semantics."""
    content = f"{source}|{index}|{text[:500]}"
    return hashlib.sha256(content.encode()).hexdigest()


def _extract_entities_simple(text: str) -> list[str]:
    """
    Placeholder entity extraction: capitalize words (e.g. "Paris", "CEO").
    Replace with NER (spaCy, etc.) or LLM for production.
    """
    entities = []
    for word in text.replace(".", " ").split():
        if len(word) > 2 and word[0].isupper():
            entities.append(word.strip(",:;"))
    return list(dict.fromkeys(entities))[:10]  # dedupe, limit


def upsert_chunks(
    driver,
    chunks: list[Document],
    embeddings: list[list[float]],
    create_mentions: bool = True,
) -> int:
    """
    Upsert Chunk nodes with embeddings. Optionally create Entity nodes and
    (Chunk)-[:MENTIONS]->(Entity) relationships.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length must match")

    def _tx(tx):
        n = 0
        for i, (doc, emb) in enumerate(zip(chunks, embeddings)):
            source = doc.metadata.get("source", "unknown")
            cid = _chunk_id(doc.page_content, source, i)
            tx.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.text = $text, c.source = $source, c.embedding = $embedding
                """,
                id=cid,
                text=doc.page_content,
                source=source,
                embedding=emb,
            )
            n += 1
            if create_mentions:
                for entity_name in _extract_entities_simple(doc.page_content):
                    if not entity_name:
                        continue
                    eid = f"entity_{hashlib.sha256(entity_name.lower().encode()).hexdigest()[:16]}"
                    tx.run(
                        """
                        MERGE (e:Entity {id: $eid})
                        SET e.name = $name
                        WITH e
                        MATCH (c:Chunk {id: $cid})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        eid=eid,
                        name=entity_name,
                        cid=cid,
                    )
        return n

    return driver.execute_write(_tx)


def ingest_file(
    file_path: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    create_mentions: bool = True,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> int:
    """
    Full pipeline: load file -> chunk -> embed -> upsert to Neo4j.
    Returns the number of chunks upserted.
    """
    uri = uri or NEO4J_URI
    user = user or NEO4J_USER
    password = password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD required for ingest")

    driver = get_driver(uri=uri, user=user, password=password)
    db = get_database()
    session = driver.session(database=db) if db else driver.session()
    try:
        init_schema(session)
    finally:
        session.close()
    docs = load_documents(file_path)
    chunks = chunk_documents(docs)
    if not chunks:
        driver.close()
        return 0

    if EMBEDDING_PROVIDER == "ollama":
        embed = OllamaEmbeddings(model=EMBEDDING_MODEL)
    else:
        embed = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    texts = [c.page_content for c in chunks]
    embeddings = embed.embed_documents(texts)
    count = upsert_chunks(driver, chunks, embeddings, create_mentions=create_mentions)
    driver.close()
    return count


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <path_to_pdf_or_txt>")
        sys.exit(1)
    path = sys.argv[1]
    n = ingest_file(path)
    print(f"Ingested {n} chunks from {path}")
