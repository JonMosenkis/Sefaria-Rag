import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from app.db import fetch_chunks_by_ids, get_conn, init_db


def embed_query(client: OpenAI, model: str, text: str) -> List[float]:
    response = client.embeddings.create(model=model, input=text, encoding_format="float")
    if not response.data:
        raise ValueError("Embedding response was empty; check input text and model.")
    return response.data[0].embedding


def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    load_dotenv()
    db_path = os.getenv("DB_PATH", "./chunks.db")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "sefaria")
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    oa_client = OpenAI()
    q_client = QdrantClient(url=qdrant_url)

    query_vec = embed_query(oa_client, model, query)
    results = q_client.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=limit,
        with_payload=True,
    )

    ids = [int(hit.id) for hit in results]
    conn = get_conn(db_path)
    init_db(conn)
    rows = fetch_chunks_by_ids(conn, ids)
    by_id = {int(row["id"]): row for row in rows}

    hits: List[Dict[str, Any]] = []
    for hit in results:
        row = by_id.get(int(hit.id))
        if not row:
            continue
        text = row["text"]
        snippet = text[:300]
        hits.append(
            {
                "ref": row["ref"],
                "snippet": snippet,
                "score": float(hit.score),
                "metadata": {
                    "chunk_id": int(row["id"]),
                },
            }
        )

    return hits
