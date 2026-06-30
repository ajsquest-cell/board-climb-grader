import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"
TABLE_NAME = "climbs"

SELECT_SQL = f"SELECT * FROM {TABLE_NAME} LIMIT ?"


def fetch_first_rows(limit):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL, (limit,))
        return cursor.fetchall()


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

