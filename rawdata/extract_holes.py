import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"

SELECT_SQL = "SELECT id, x, y FROM holes"

# Outputs a dictionary mapping hole_id to (x, y) coords
def extract_holes():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(SELECT_SQL)
        rows = cursor.fetchall()

    # dictionary: Hole_id -> (x, y)
    placements = {hole_id: (x, y) for hole_id, x, y in rows}
    
    return placements


if __name__ == "__main__":
    placements = extract_holes()
    # Testing
    for hole_id, (x, y) in placements.items():
        print(f"id={hole_id}, x={x}, y={y}")
