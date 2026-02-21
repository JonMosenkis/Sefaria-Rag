import json
import os
import sqlite3
from typing import Iterable, List, Optional


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or os.getenv("DB_PATH", "./chunks.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ref TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedded INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedded)"
    )
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    data = [
        (r["ref"], r["text"], json.dumps(r.get("metadata", {}))) for r in rows
    ]
    if not data:
        return 0
    conn.executemany(
        "INSERT INTO chunks (ref, text, metadata_json) VALUES (?, ?, ?)", data
    )
    conn.commit()
    return len(data)


def list_unembedded(conn: sqlite3.Connection, limit: int) -> List[sqlite3.Row]:
    cur = conn.execute(
        "SELECT id, ref, text, metadata_json FROM chunks WHERE embedded = 0 LIMIT ?",
        (limit,),
    )
    return cur.fetchall()


def count_chunks(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) AS count FROM chunks")
    row = cur.fetchone()
    return int(row["count"]) if row else 0


def count_embedded(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) AS count FROM chunks WHERE embedded = 1")
    row = cur.fetchone()
    return int(row["count"]) if row else 0


def mark_embedded(conn: sqlite3.Connection, ids: Iterable[int]) -> None:
    ids = list(ids)
    if not ids:
        return
    conn.executemany("UPDATE chunks SET embedded = 1 WHERE id = ?", [(i,) for i in ids])
    conn.commit()


def fetch_chunks_by_ids(conn: sqlite3.Connection, ids: Iterable[int]) -> List[sqlite3.Row]:
    ids = list(ids)
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    cur = conn.execute(
        f"SELECT id, ref, text, metadata_json FROM chunks WHERE id IN ({placeholders})",
        ids,
    )
    return cur.fetchall()
