#!/usr/bin/env python3
# ==========================================================
# scripts/init_db.py — Initialize the omerGPT DuckDB schema
# ==========================================================
import duckdb
from pathlib import Path
import sys

def init_db(
    db_path: str = "data/market_data.duckdb",
    schema_path: str = "storage/schema.sql",
):
    """Initialize DuckDB database and load schema SQL."""
    data_dir = Path(db_path).parent
    schema_file = Path(schema_path)

    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)

    if not schema_file.exists():
        sys.exit(f"❌ Schema file not found: {schema_file}")

    print(f"📀 Connecting to database at: {db_path}")
    con = duckdb.connect(str(db_path))

    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        # Execute all SQL commands from schema.sql
        con.execute(schema_sql)
        con.commit()

        # Quick sanity check: list all tables
        tables = con.execute("SHOW TABLES").fetchall()
        print(f"✅ Schema initialized successfully. {len(tables)} tables created:")
        for t in tables:
            print("   •", t[0])

    except Exception as e:
        sys.exit(f"❌ Failed to initialize schema: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    init_db()
