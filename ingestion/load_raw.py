"""
Load landing CSV files into the raw schema of the DuckDB warehouse.

Design rules for the raw layer:
  1. Keep data exactly as received. Every column is loaded as text (VARCHAR),
     so dirty values like '20260415' or ' ca123... ' are never rejected or
     silently changed. Cleaning happens later, in the dbt staging models.
  2. Add audit columns so every row can be traced back to its source file:
       _source_file   name of the file the row came from
       _file_hash     fingerprint (MD5) of that file's contents
       _loaded_at     when the row was loaded
  3. Loads are idempotent: running the script twice gives the same result.
     A load log records each file's fingerprint. Unchanged files are skipped,
     and a changed file has its old rows replaced, never duplicated.

Run from the project root:
    python ingestion/load_raw.py
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import duckdb

# Which files feed which raw table. The * lets later daily files such as
# custodian_payments_20261101.csv land in the same table automatically.
SOURCES = {
    "securities": "securities*.csv",
    "dividend_events": "dividend_events*.csv",
    "positions": "positions*.csv",
    "custodian_payments": "custodian_payments*.csv",
}


def file_md5(path: Path) -> str:
    """Fingerprint a file. Any change to its contents changes the hash."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_data_rows(path: Path) -> int:
    """Count rows in the file, excluding the header, to verify the load."""
    with open(path, "rb") as f:
        return max(sum(1 for line in f if line.strip()) - 1, 0)


def setup(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw._load_log (
            table_name   VARCHAR,
            source_file  VARCHAR,
            file_hash    VARCHAR,
            row_count    INTEGER,
            loaded_at    TIMESTAMP
        )
    """)


def load_file(con, table: str, path: Path) -> str:
    """Load one file into raw.<table>. Returns what happened."""
    file_hash = file_md5(path)

    # Has this exact file already been loaded?
    already = con.execute(
        "SELECT 1 FROM raw._load_log WHERE source_file = ? AND file_hash = ?",
        [path.name, file_hash],
    ).fetchone()
    if already:
        return "skipped (unchanged)"

    loaded_at = datetime.now(timezone.utc).replace(tzinfo=None)
    csv = path.resolve().as_posix().replace("'", "''")
    select_sql = f"""
        SELECT *,
               '{path.name}'      AS _source_file,
               '{file_hash}'      AS _file_hash,
               TIMESTAMP '{loaded_at}' AS _loaded_at
        FROM read_csv('{csv}', header = true, all_varchar = true)
    """

    # A transaction makes the load all or nothing: if anything fails,
    # no half loaded data is left behind.
    con.execute("BEGIN TRANSACTION")
    try:
        exists = con.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'raw' AND table_name = ?", [table]
        ).fetchone()

        if exists:
            # Remove any earlier version of this file, then insert the new one
            con.execute(f"DELETE FROM raw.{table} WHERE _source_file = ?", [path.name])
            con.execute(f"INSERT INTO raw.{table} BY NAME {select_sql}")
            con.execute("DELETE FROM raw._load_log WHERE source_file = ?", [path.name])
        else:
            con.execute(f"CREATE TABLE raw.{table} AS {select_sql}")

        loaded = con.execute(
            f"SELECT COUNT(*) FROM raw.{table} WHERE _source_file = ?", [path.name]
        ).fetchone()[0]

        # Control total: rows in the table must equal rows in the file
        expected = count_data_rows(path)
        if loaded != expected:
            raise ValueError(f"{path.name}: file has {expected} rows but {loaded} were loaded")

        con.execute(
            "INSERT INTO raw._load_log VALUES (?, ?, ?, ?, ?)",
            [table, path.name, file_hash, loaded, loaded_at],
        )
        con.execute("COMMIT")
        return f"loaded {loaded:,} rows"
    except Exception:
        con.execute("ROLLBACK")
        raise


def main():
    parser = argparse.ArgumentParser(description="Load landing CSVs into DuckDB raw schema")
    parser.add_argument("--landing", default="data/landing", help="folder with source CSVs")
    parser.add_argument("--db", default="data/warehouse.duckdb", help="DuckDB database file")
    args = parser.parse_args()

    landing = Path(args.landing)
    con = duckdb.connect(args.db)
    setup(con)

    for table, pattern in SOURCES.items():
        files = sorted(landing.glob(pattern))
        if not files:
            print(f"raw.{table:<20} no files found matching {pattern}")
            continue
        for path in files:
            result = load_file(con, table, path)
            print(f"raw.{table:<20} {path.name:<32} {result}")

    con.close()


if __name__ == "__main__":
    main()
