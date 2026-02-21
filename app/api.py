import os

from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv

from app.models import SearchResponse, SearchHit
from app.search import search

load_dotenv()

app = FastAPI(title="Sefaria RAG", version="0.1.0")


@app.get("/search", response_model=SearchResponse)
def search_endpoint(q: str = Query(..., min_length=2), limit: int = 5):
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 50")
    hits_raw = search(q, limit=limit)
    hits = [SearchHit(**hit) for hit in hits_raw]
    return SearchResponse(query=q, hits=hits)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("app.api:app", host=host, port=port, reload=True)
