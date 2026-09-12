"""Database connection helpers."""

import sqlite3
import os
from config import DB_PATH

_connection = None


def get_db() -> sqlite3.Connection:
    """Get or create a database connection (singleton per process)."""
    global _connection
    if _connection is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("PRAGMA foreign_keys=ON")
    return _connection


def close_db():
    """Close the database connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    """Execute a single SQL statement."""
    return get_db().execute(sql, params)


def executemany(sql: str, params_list: list) -> sqlite3.Cursor:
    """Execute a SQL statement against multiple parameter sets."""
    return get_db().executemany(sql, params_list)


def fetchone(sql: str, params: tuple = ()) -> dict | None:
    """Execute and fetch one row as a dict."""
    row = execute(sql, params).fetchone()
    return dict(row) if row else None


def fetchall(sql: str, params: tuple = ()) -> list[dict]:
    """Execute and fetch all rows as dicts."""
    return [dict(r) for r in execute(sql, params).fetchall()]


def commit():
    """Commit the current transaction."""
    get_db().commit()
