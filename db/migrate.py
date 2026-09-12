"""Database migration — idempotent schema setup."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db, commit


def migrate():
    """Run schema.sql to create all tables. Idempotent via IF NOT EXISTS."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    db = get_db()
    db.executescript(schema_sql)
    commit()
    print("[OK] Database migrated successfully.")

    # Print table summary
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    for t in tables:
        count = db.execute(f"SELECT COUNT(*) as c FROM [{t['name']}]").fetchone()["c"]
        print(f"  {t['name']}: {count} rows")


if __name__ == "__main__":
    migrate()
