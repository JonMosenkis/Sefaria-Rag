import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from app.clients import get_openai_client
from app.db import fetch_chunks_by_ids, get_conn, init_db


def embed_query(client, model: str, text: str) -> List[float]:
    response = client.embeddings.create(model=model, input=text, encoding_format="float")
    if not response.data:
        raise ValueError("Embedding response was empty; check input text and model.")
    return response.data[0].embedding


def query_qdrant(q_client: QdrantClient, collection: str, query_vec: List[float], limit: int):
    if hasattr(q_client, "query_points"):
        params = inspect.signature(q_client.query_points).parameters
        if "query_vector" in params:
            results = q_client.query_points(
                collection_name=collection,
                query_vector=query_vec,
                limit=limit,
                with_payload=True,
            )
        else:
            results = q_client.query_points(
                collection_name=collection,
                query=query_vec,
                limit=limit,
                with_payload=True,
            )
        if hasattr(results, "points"):
            return results.points
        if hasattr(results, "result"):
            return results.result
        return results

    return q_client.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=limit,
        with_payload=True,
    )


def build_hits(qdrant_hits, rows_by_id: Dict[int, Any]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for hit in qdrant_hits:
        row = rows_by_id.get(int(hit.id))
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


@dataclass(frozen=True)
class SearchContext:
    db_path: str
    qdrant_url: str
    collection: str
    model: str

    @staticmethod
    def from_env() -> "SearchContext":
        load_dotenv()
        return SearchContext(
            db_path=os.getenv("DB_PATH", "./chunks.db"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection=os.getenv("QDRANT_COLLECTION", "sefaria"),
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        )


class SearchService:
    def __init__(
        self,
        ctx: SearchContext,
        oa_client,
        q_client: QdrantClient,
        conn=None,
    ) -> None:
        self.ctx = ctx
        self.oa_client = oa_client
        self.q_client = q_client
        self.conn = conn or get_conn(ctx.db_path)
        init_db(self.conn)

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_vec = embed_query(self.oa_client, self.ctx.model, query)
        qdrant_hits = query_qdrant(
            self.q_client,
            self.ctx.collection,
            query_vec,
            limit,
        )

        print(f"Qdrant hits: {len(qdrant_hits)}")

        ids = [int(hit.id) for hit in qdrant_hits]
        rows = fetch_chunks_by_ids(self.conn, ids)
        print(f"SQLite rows fetched: {len(rows)} (ids requested: {len(ids)})")
        by_id = {int(row["id"]): row for row in rows}

        hits = build_hits(qdrant_hits, by_id)
        print(f"Search hits built: {len(hits)}")
        if hits:
            print(f"First hit ref: {hits[0].get('ref')}")
        return hits


def build_search_service(ctx: Optional[SearchContext] = None) -> SearchService:
    context = ctx or SearchContext.from_env()
    oa_client = get_openai_client()
    q_client = QdrantClient(url=context.qdrant_url)
    return SearchService(context, oa_client, q_client)


def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    return build_search_service().search(query, limit)
