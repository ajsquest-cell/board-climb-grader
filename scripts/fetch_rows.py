import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"
TABLE_NAME = "climbs"

# Fetch the first X rows from the "climbs" table
def fetch_first_rows(limit):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
    
    SELECT_SQL = f"SELECT * FROM {TABLE_NAME} LIMIT ?"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL, (limit,))
        return cursor.fetchall()

# Fetch a specific row by climb_uuid
def fetch_row_by_climb_uuid(climb_uuid):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    SELECT_BY_UUID_SQL = f"SELECT * FROM {TABLE_NAME} WHERE climb_uuid = ?"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_BY_UUID_SQL, (climb_uuid,))
        return cursor.fetchone()

# Testing
if __name__ == "__main__":
    if len(sys.argv) == 1:
        count = 10
    elif len(sys.argv) == 2:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print("Valid integer required.")
            sys.exit(1)
    else:
        print(f"Usage: python {sys.argv[0]} [number_of_rows]")
        sys.exit(1)

    try:
        rows = fetch_first_rows(count)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    if not rows:
        print("No rows found.")
        sys.exit(0)

    for index, row in enumerate(rows, start=1):
        print(f"Row {index}:")
        for key in row.keys():
            print(f"  {key}: {row[key]}")
        print()

