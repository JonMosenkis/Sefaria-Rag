import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

from app.db import get_conn, init_db, insert_chunks


HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def is_english(text: str) -> bool:
    if not text:
        return False
    return HEBREW_RE.search(text) is None


def flatten_text(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(flatten_text(item))
        return out
    if isinstance(value, dict):
        if "en" in value:
            return flatten_text(value.get("en"))
        if "text" in value:
            return flatten_text(value.get("text"))
    return []


def extract_ref(obj: Dict[str, Any], fallback: str) -> str:
    for key in ("ref", "reference", "title", "displayRef", "primaryTitle"):
        if key in obj and obj[key]:
            return str(obj[key])
    return fallback


def extract_english_text(obj: Dict[str, Any]) -> List[str]:
    lang = obj.get("lang") or obj.get("language")
    if lang and str(lang).lower() not in ("en", "english"):
        return []

    if "en" in obj:
        return flatten_text(obj.get("en"))

    if "text" in obj:
        return flatten_text(obj.get("text"))

    return []


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(cleaned):
        chunk = cleaned[i : i + chunk_size]
        chunks.append(chunk)
        i += step
    return chunks


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if isinstance(item, dict):
                        yield item
                return
            yield data
        return


def build_chunk_rows(
    obj: Dict[str, Any],
    source: str,
    chunk_size: int,
    overlap: int,
    fallback_ref: str,
) -> List[Dict[str, Any]]:
    ref = extract_ref(obj, fallback_ref)
    texts = extract_english_text(obj)
    rows: List[Dict[str, Any]] = []

    for text in texts:
        if not text or not is_english(text):
            continue
        for chunk in chunk_text(text, chunk_size, overlap):
            rows.append(
                {
                    "ref": ref,
                    "text": chunk,
                    "metadata": {
                        "source": source,
                        "ref": ref,
                    },
                }
            )
    return rows


def ingest(data_dir: Path, chunk_size: int, overlap: int, db_path: Optional[str]) -> int:
    conn = get_conn(db_path)
    init_db(conn)

    total = 0
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".json", ".jsonl"):
            continue
        fallback_ref = path.stem
        for obj in iter_records(path):
            rows = build_chunk_rows(
                obj,
                source=str(path.relative_to(data_dir)),
                chunk_size=chunk_size,
                overlap=overlap,
                fallback_ref=fallback_ref,
            )
            total += insert_chunks(conn, rows)

    return total


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest Sefaria subset into SQLite.")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data"))
    parser.add_argument("--db-path", default=os.getenv("DB_PATH", "./chunks.db"))
    parser.add_argument(
        "--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "800"))
    )
    parser.add_argument(
        "--overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "100"))
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    total = ingest(data_dir, args.chunk_size, args.overlap, args.db_path)
    print(f"Inserted {total} chunks into SQLite")


if __name__ == "__main__":
    main()
