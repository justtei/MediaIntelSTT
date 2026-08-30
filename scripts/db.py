"""Async Postgres persistence for recognized speech — the corpus future
fine-tuning draws from. Plain asyncpg, no ORM: the schema (db/schema.sql)
is two tables and the query patterns are simple insert/select, so an ORM
would add indirection without buying anything.

Postgres (via `docker compose up -d`) was chosen over SQLite/MongoDB because
this data is explicitly headed toward ML fine-tuning corpus export: it
benefits from real indexes over channel/language/time for filtering training
splits, foreign-key integrity between channels and their segments, and being
comfortable at the "huge" row counts a 24/7 multi-channel transcription
service accumulates — all while staying free and trivial to run locally via
Docker.
"""
import os
import uuid
from typing import Optional

import asyncpg

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://stt:stt@localhost:5432/stt")

_pool: Optional[asyncpg.Pool] = None


async def connect(dsn: str = DATABASE_URL) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
    return _pool


async def disconnect():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def is_connected() -> bool:
    return _pool is not None


def _pool_or_raise() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("db.connect() has not been called")
    return _pool


async def create_channel(name: str, source_type: str, url: Optional[str], language: str) -> str:
    row = await _pool_or_raise().fetchrow(
        """INSERT INTO channels (name, source_type, url, language, status)
           VALUES ($1, $2, $3, $4, 'starting')
           RETURNING id""",
        name, source_type, url, language,
    )
    return str(row["id"])


async def set_channel_status(channel_id: str, status: str) -> None:
    if status in ("stopped", "error"):
        await _pool_or_raise().execute(
            "UPDATE channels SET status = $2, stopped_at = now() WHERE id = $1",
            uuid.UUID(channel_id), status,
        )
    else:
        await _pool_or_raise().execute(
            "UPDATE channels SET status = $2 WHERE id = $1",
            uuid.UUID(channel_id), status,
        )


async def list_channels() -> list[dict]:
    rows = await _pool_or_raise().fetch(
        """SELECT id, name, source_type, url, language, status, created_at, stopped_at
           FROM channels ORDER BY created_at DESC"""
    )
    return [dict(r) for r in rows]


async def insert_segment(
    channel_id: str, *, text: str, language: str,
    t0_s: Optional[float], t1_s: Optional[float],
    infer_s: Optional[float], rtf: Optional[float],
    reason: Optional[str], source: Optional[str],
    backend: Optional[str], device: Optional[str],
) -> None:
    await _pool_or_raise().execute(
        """INSERT INTO segments
           (channel_id, text, language, t0_s, t1_s, infer_s, rtf, reason, source, backend, device)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
        uuid.UUID(channel_id), text, language, t0_s, t1_s, infer_s, rtf, reason, source, backend, device,
    )


async def list_segments(channel_id: str, limit: int = 50) -> list[dict]:
    rows = await _pool_or_raise().fetch(
        """SELECT id, text, language, t0_s, t1_s, infer_s, rtf, reason, source, created_at
           FROM segments WHERE channel_id = $1
           ORDER BY created_at DESC LIMIT $2""",
        uuid.UUID(channel_id), limit,
    )
    return [dict(r) for r in rows]


async def segment_count(channel_id: Optional[str] = None) -> int:
    if channel_id is None:
        row = await _pool_or_raise().fetchrow("SELECT count(*) AS n FROM segments")
    else:
        row = await _pool_or_raise().fetchrow(
            "SELECT count(*) AS n FROM segments WHERE channel_id = $1", uuid.UUID(channel_id))
    return row["n"]
