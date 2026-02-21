import argparse
import os
import time
from typing import List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.clients import get_openai_client
from app.db import (
    count_chunks,
    count_embedded,
    get_conn,
    init_db,
    list_unembedded,
    mark_embedded,
)


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def embed_batch(client, model: str, texts: List[str]) -> List[List[float]]:
    start = time.time()
    response = client.embeddings.create(model=model, input=texts, encoding_format="float")
    elapsed = time.time() - start
    print(f"Embedded batch of {len(texts)} in {elapsed:.2f}s")
    return [item.embedding for item in response.data]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Embed SQLite chunks into Qdrant.")
    parser.add_argument("--db-path", default=os.getenv("DB_PATH", "./chunks.db"))
    parser.add_argument(
        "--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    parser.add_argument(
        "--collection", default=os.getenv("QDRANT_COLLECTION", "sefaria")
    )
    parser.add_argument(
        "--model", default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(os.getenv("EMBED_BATCH_SIZE", "64"))
    )
    args = parser.parse_args()

    conn = get_conn(args.db_path)
    init_db(conn)

    oa_client = get_openai_client()
    q_client = QdrantClient(url=args.qdrant_url)

    total = 0
    batch_num = 0
    total_chunks = count_chunks(conn)
    embedded_count = count_embedded(conn)
    print(f"Total chunks: {total_chunks} (already embedded: {embedded_count})")
    while True:
        rows = list_unembedded(conn, args.batch_size)
        if not rows:
            break

        batch_num += 1
        remaining = total_chunks - embedded_count - total
        print(
            f"Batch {batch_num}: embedding {len(rows)} chunks "
            f"(remaining: {max(remaining, 0)})"
        )
        texts = [row["text"] for row in rows]
        embeddings = embed_batch(oa_client, args.model, texts)
        ensure_collection(q_client, args.collection, len(embeddings[0]))

        points = []
        for row, vector in zip(rows, embeddings):
            chunk_id = int(row["id"])
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "ref": row["ref"],
                    },
                )
            )

        q_client.upsert(collection_name=args.collection, points=points)
        mark_embedded(conn, [row["id"] for row in rows])
        total += len(rows)
        print(f"Batch {batch_num} complete (total embedded: {total})")

    print(f"Embedding complete (total embedded: {total})")


if __name__ == "__main__":
    main()
