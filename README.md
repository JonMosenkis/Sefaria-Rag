# Sefaria RAG (Week 1)

English-only ingestion + embeddings + vector search over a Sefaria subset.

## Setup

1. Create a `.env` with your keys and local config:

```bash
OPENAI_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=sefaria
DB_PATH=./chunks.db
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Ingest

Put your Sefaria subset in `data/` (`.json` or `.jsonl`). Then run:

```bash
python -m app.ingest --data-dir ./data
```

## Embed into Qdrant

```bash
python -m app.embed
```

## Search API

```bash
python -m app.api
```

Example request:

```bash
curl "http://localhost:8000/search?q=creation"
```
