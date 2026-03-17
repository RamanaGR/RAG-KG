"""
Healthcare Neo4j schema: constraints and 768-dimension vector index for HealthcareProvider.bio_embedding.
Nodes: HealthcareProvider, Patient, Specialization, Location.
Relationships: (HealthcareProvider)-[:SPECIALIZES_IN]->(Specialization),
               (HealthcareProvider)-[:LOCATED_AT]->(Location),
               (HealthcareProvider)-[:TREATS]->(Patient).
"""

# Nomic embeddings = 768 dimensions (required for vector index)
HEALTHCARE_VECTOR_DIMENSION = 768


def init_healthcare_schema(session, vector_dim: int = HEALTHCARE_VECTOR_DIMENSION) -> None:
    """
    Create constraints and vector index for healthcare graph.
    Vector index on HealthcareProvider.bio_embedding for similarity search on provider bios.
    """
    def _run(tx):
        # Uniqueness constraints
        tx.run("""
            CREATE CONSTRAINT provider_id IF NOT EXISTS
            FOR (p:HealthcareProvider) REQUIRE p.id IS UNIQUE
        """)
        tx.run("""
            CREATE CONSTRAINT specialization_name IF NOT EXISTS
            FOR (s:Specialization) REQUIRE s.name IS UNIQUE
        """)
        tx.run("""
            CREATE CONSTRAINT location_name IF NOT EXISTS
            FOR (l:Location) REQUIRE l.name IS UNIQUE
        """)
        tx.run("""
            CREATE CONSTRAINT patient_id IF NOT EXISTS
            FOR (p:Patient) REQUIRE p.id IS UNIQUE
        """)
        # Vector index on HealthcareProvider.bio_embedding (768 for nomic-embed-text)
        tx.run("""
            CREATE VECTOR INDEX provider_bio_embedding_index IF NOT EXISTS
            FOR (p:HealthcareProvider) ON (p.bio_embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
            }}
        """, dim=vector_dim)

    session.execute_write(_run)
