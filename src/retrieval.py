"""
Hybrid retrieval: Vector search on Chunk embeddings + Cypher graph traversal.
Uses (Chunk)-[:MENTIONS]->(Entity) to fetch related chunks for richer context.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.database import get_driver, get_database

load_dotenv()

# LLM provider: "openai" or "ollama"
LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "openai")).lower()
LLM_MODEL = (
    os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
    if LLM_PROVIDER == "ollama"
    else os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
)

# Embedding provider (defaults to LLM_PROVIDER)
EMBEDDING_PROVIDER = (os.getenv("EMBEDDING_PROVIDER") or os.getenv("LLM_PROVIDER", "openai")).lower()
EMBEDDING_MODEL = (
    os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    if EMBEDDING_PROVIDER == "ollama"
    else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "5"))
TOP_K_GRAPH = int(os.getenv("TOP_K_GRAPH", "5"))


# -----------------------------------------------------------------------------
# Sample Cypher: traverse (Chunk)-[:MENTIONS]->(Entity) for graph-backed context
# -----------------------------------------------------------------------------
# 1) From chunk IDs (e.g. from vector search), get entities they mention and
#    then get *other* chunks that mention the same entities (sibling chunks).
#    This gives the LLM context that is related by shared entities, not just
#    by vector similarity.
# -----------------------------------------------------------------------------

CYPHER_CHUNKS_BY_ENTITY_NEIGHBORS = """
// Given $chunkIds from vector search, traverse MENTIONS to get related chunks.
// (Chunk)-[:MENTIONS]->(Entity)<-[:MENTIONS]-(Chunk) -> sibling chunks
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Chunk)
WHERE c.id IN $chunkIds AND other.id <> c.id
WITH other, count(DISTINCT e) AS sharedEntities
ORDER BY sharedEntities DESC
LIMIT $topK
RETURN other.id AS id, other.text AS text, other.source AS source
"""

CYPHER_CHUNKS_AND_ENTITIES = """
// For a set of chunk IDs, return those chunks and the entities they mention.
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE c.id IN $chunkIds
RETURN c.id AS chunkId, c.text AS text, c.source AS source,
       collect(DISTINCT e.name) AS entities
ORDER BY chunkId
"""


def vector_search(session, query_embedding: list[float], top_k: int = TOP_K_VECTOR) -> list[dict]:
    """Neo4j vector index search on Chunk.embedding (cosine)."""
    def _run(tx):
        result = tx.run(
            """
            CALL db.index.vector.queryNodes('chunk_embedding_index', $k, $embedding)
            YIELD node, score
            RETURN node.id AS id, node.text AS text, node.source AS source, score
            ORDER BY score DESC
            """,
            k=top_k,
            embedding=query_embedding,
        )
        return [{"id": r["id"], "text": r["text"], "source": r["source"], "score": r["score"]} for r in result]
    return session.execute_read(_run)


def graph_related_chunks(session, chunk_ids: list[str], top_k: int = TOP_K_GRAPH) -> list[dict]:
    """
    Use (Chunk)-[:MENTIONS]->(Entity)<-[:MENTIONS]-(Chunk) to retrieve
    related chunks that share entities with the given chunks. Provides
    better context than vector search alone by following the graph.
    """
    if not chunk_ids:
        return []
    def _run(tx):
        result = tx.run(
            CYPHER_CHUNKS_BY_ENTITY_NEIGHBORS,
            chunkIds=chunk_ids,
            topK=top_k,
        )
        return [{"id": r["id"], "text": r["text"], "source": r["source"]} for r in result]
    return session.execute_read(_run)


def hybrid_retrieve(
    session,
    query: str,
    embed_model: Embeddings,
    top_k_vector: int = TOP_K_VECTOR,
    top_k_graph: int = TOP_K_GRAPH,
    use_graph: bool = True,
) -> list[Document]:
    """
    Hybrid retrieval: vector search + optional graph expansion via MENTIONS.
    Returns a list of LangChain Documents with metadata (source, score, from_graph).
    """
    query_emb = embed_model.embed_query(query)
    vector_hits = vector_search(session, query_emb, top_k=top_k_vector)
    seen = set()
    docs: list[Document] = []
    for h in vector_hits:
        key = h["id"]
        if key in seen:
            continue
        seen.add(key)
        docs.append(
            Document(
                page_content=h["text"],
                metadata={"source": h["source"], "score": float(h["score"]), "chunk_id": key, "from_graph": False},
            )
        )
    if use_graph and vector_hits:
        chunk_ids = [h["id"] for h in vector_hits]
        graph_hits = graph_related_chunks(session, chunk_ids, top_k=top_k_graph)
        for h in graph_hits:
            if h["id"] in seen:
                continue
            seen.add(h["id"])
            docs.append(
                Document(
                    page_content=h["text"],
                    metadata={"source": h["source"], "chunk_id": h["id"], "from_graph": True},
                )
            )
    return docs


def get_llm():
    """Return LLM from env: ChatOllama or ChatOpenAI based on LLM_PROVIDER."""
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=LLM_MODEL, temperature=0)
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


def get_embedding_model() -> Embeddings:
    """Return embedding model from env: OllamaEmbeddings or OpenAIEmbeddings based on EMBEDDING_PROVIDER."""
    if EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(model=EMBEDDING_MODEL)
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_chain(llm=None):
    """Build a simple RAG chain: prompt + LLM + string parser."""
    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the following context. If the context does not contain enough information, say so.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def query_rag(
    question: str,
    driver=None,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> str:
    """
    Run hybrid retrieval, build context string, and return LLM response.
    Uses Ollama or OpenAI per LLM_PROVIDER / EMBEDDING_PROVIDER; supports NEO4J_DATABASE (Aura/Kocla).
    """
    password = password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD required for retrieval")
    close_driver = driver is None
    driver = driver or get_driver(uri=uri, user=user, password=password)
    db = get_database()
    session = driver.session(database=db) if db else driver.session()
    try:
        embed = get_embedding_model()
        docs = hybrid_retrieve(session, question, embed, use_graph=True)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        if not context.strip():
            return "No relevant context found in the knowledge base."
        chain = build_chain()
        return chain.invoke({"context": context, "question": question})
    finally:
        session.close()
        if close_driver:
            driver.close()
