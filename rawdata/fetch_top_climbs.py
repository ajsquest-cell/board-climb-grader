import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"
TABLE_NAME = "climb_cache_fields"

# Fetch the top X climbs based on number of ascensents
def fetch_top_climbs(limit):

    SELECT_SQL = f"SELECT * FROM {TABLE_NAME} ORDER BY ascensionist_count DESC LIMIT ?"

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL, (limit,))
        return cursor.fetchall()


# Fetch climb stats by climb_uuid
def get_climb_stats(climb_uuid):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    SELECT_STATS_SQL = f"""SELECT ascensionist_count, display_difficulty, quality_average 
        FROM {TABLE_NAME} WHERE climb_uuid = ?"""

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(SELECT_STATS_SQL, (climb_uuid,))
        result = cursor.fetchone()
        
        if result:
            return {
                'ascensionist_count': result[0],
                'display_difficulty': result[1],
                'quality_average': result[2]
            }
        else:
            return None


# Testing
if __name__ == "__main__":
    if len(sys.argv) == 1:
        count = 100
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
        rows = fetch_top_climbs(count)
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
    
    # Example: Get stats for a specific climb
    if rows:
        first_climb_uuid = rows[0]['climb_uuid']
        print(f"\n=== Example: Fetching stats for climb {first_climb_uuid} ===")
        stats = get_climb_stats(first_climb_uuid)
        if stats:
            print(f"Ascensionist Count: {stats['ascensionist_count']}")
            print(f"Display Difficulty: {stats['display_difficulty']}")
            print(f"Quality Average: {stats['quality_average']}")
        else:
            print("Climb not found.")


