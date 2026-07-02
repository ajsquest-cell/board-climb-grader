import sqlite3
import sys
from pathlib import Path

from fetch_climb_stats import fetch_top_climbs

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"
TABLE_NAME = "beta_links"

# Fetch climb stats by climb_uuid
def fetch_angle(climb_uuid):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    SELECT_STATS_SQL = f"""SELECT angle FROM {TABLE_NAME} WHERE climb_uuid = ?"""

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(SELECT_STATS_SQL, (climb_uuid,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            return None
        
# Testing
if __name__ == "__main__":
    rows = fetch_top_climbs(1)
    if rows:
        first_climb_uuid = rows[0]['climb_uuid']
        angle = fetch_angle(first_climb_uuid)
        print(f"Angle for climb {first_climb_uuid}: {angle}")