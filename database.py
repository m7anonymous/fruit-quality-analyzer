"""
database.py
-----------
SQLite-backed persistence layer for prediction history.
Stores every inference result so users can review past analyses
through the Streamlit app or REST API.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

# ── DB path (auto-created on first run) ──────────────────────────────────────
DB_DIR  = Path(__file__).parent / "db"
DB_PATH = DB_DIR / "predictions.db"


def get_connection() -> sqlite3.Connection:
    """Open (and auto-create) the SQLite database."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row     # rows behave like dicts
    return conn


def init_db():
    """
    Create tables if they do not already exist.
    Call once at application start-up.
    """
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            filename        TEXT,
            predicted_class TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            shelf_life_days TEXT,
            shelf_status    TEXT,
            top3_json       TEXT,           -- JSON array of {class, confidence}
            model_version   TEXT DEFAULT 'v1'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ts ON predictions (timestamp)
    """)
    conn.commit()
    conn.close()


def save_prediction(
    filename:        str,
    predicted_class: str,
    confidence:      float,
    shelf_life_days: str,
    shelf_status:    str,
    top3:            list,
    model_version:   str = "v1",
) -> int:
    """
    Persist one prediction record.

    Args:
        filename        : Original image filename (or 'upload').
        predicted_class : Winning class label.
        confidence      : Float 0–1.
        shelf_life_days : Human-readable shelf-life string.
        shelf_status    : e.g., "Excellent", "Fair", "Expired".
        top3            : List of dicts [{class, confidence}, …].
        model_version   : Tag for the model that made the prediction.

    Returns:
        The auto-generated row id.
    """
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO predictions
            (timestamp, filename, predicted_class, confidence,
             shelf_life_days, shelf_status, top3_json, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            filename,
            predicted_class,
            round(float(confidence), 4),
            shelf_life_days,
            shelf_status,
            json.dumps(top3),
            model_version,
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def fetch_history(limit: int = 50) -> list:
    """
    Retrieve the most recent `limit` prediction records, newest first.

    Returns:
        List of dicts with all column values.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    records = []
    for row in rows:
        rec = dict(row)
        rec["top3"] = json.loads(rec.pop("top3_json", "[]") or "[]")
        records.append(rec)
    return records


def fetch_stats() -> dict:
    """
    Return aggregate statistics about all stored predictions.

    Returns:
        Dict with keys: total, by_class, avg_confidence
    """
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    rows = conn.execute(
        "SELECT predicted_class, COUNT(*) AS cnt FROM predictions GROUP BY predicted_class"
    ).fetchall()
    by_class = {r["predicted_class"]: r["cnt"] for r in rows}

    avg_conf = conn.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0] or 0.0

    conn.close()
    return {"total": total, "by_class": by_class, "avg_confidence": round(avg_conf, 4)}


def clear_history():
    """Delete all stored predictions (for testing / reset)."""
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


# ── Auto-init on import ───────────────────────────────────────────────────────
init_db()
