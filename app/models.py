from typing import Any, Dict, List

from pydantic import BaseModel


class SearchHit(BaseModel):
    ref: str
    snippet: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]
