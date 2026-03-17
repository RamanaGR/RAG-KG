"""
Vector search over HealthcareProvider using Ollama embeddings (Python-based).
Embeds the question in Python with OllamaEmbeddings (nomic-embed-text), then
queries Neo4j vector index. Compatible with provider_bio_embedding_index (768-dim).
"""

from dotenv import load_dotenv
import os

from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

load_dotenv()

# Neo4j (Aura / local)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# Ollama (embeddings + LLM)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Vector index (768-dim for nomic-embed-text, created by ingest_healthcare)
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "provider_bio_embedding_index")
TOP_K = int(os.getenv("TOP_K", "5"))

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

chat = ChatOllama(
    model=OLLAMA_LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


def get_embedding_model():
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def vector_search(question: str, top_k: int = TOP_K):
    """Embed question with Ollama in Python, then query Neo4j vector index."""
    embed_model = get_embedding_model()
    question_embedding = embed_model.embed_query(question)

    result = kg.query(
        """
        CALL db.index.vector.queryNodes($indexName, $topK, $embedding)
        YIELD node, score
        RETURN node.name AS name, node.bio AS bio, score
        ORDER BY score DESC
        """,
        params={
            "indexName": VECTOR_INDEX_NAME,
            "topK": top_k,
            "embedding": question_embedding,
        },
    )
    return result


if __name__ == "__main__":
    question = "give me a list of healthcare providers in the area of dermatology"

    result = vector_search(question, top_k=5)

    for record in result:
        print(f"Name: {record['name']}")
        print(f"Bio: {record['bio']}")
        print(f"Score: {record['score']}")
        print("---")
