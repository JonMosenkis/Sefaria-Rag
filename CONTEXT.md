Goal: Get hired for AI roles soon by shipping a production-flavored RAG retrieval system over Sefaria.

Week 1 scope: English-only ingestion + embeddings + vector search returning top refs/snippets.

Storage: SQLite for chunks/metadata + Qdrant for vectors (running locally).

No LLM answering yet (retrieval only).

API: /search?q=... returns list of hits: ref, snippet, score, metadata.

