from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Optional


class SQLiteCache:
    """Cache SQLite simples para respostas de LLM (economia + reprodutibilidade)."""

    def __init__(self, path: str):
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS llm_cache (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            con.commit()

    @staticmethod
    def make_key(payload: dict) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        with sqlite3.connect(self.path) as con:
            row = con.execute("SELECT v FROM llm_cache WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT OR REPLACE INTO llm_cache (k, v) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            con.commit()
