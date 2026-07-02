import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"

# Table names:
BETA_LINKS_TABLE = "beta_links"
CLIMB_CACHE_FIELDS_TABLE = "climb_cache_fields"
CLIMBS_TABLE = "climbs"
DIFFICULTY_GRADES_TABLE = "difficulty_grades"

# Fetch climb stats by climb_uuid
def fetch_angle(climb_uuid):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
    
    SELECT_STATS_SQL = f"""SELECT angle FROM {BETA_LINKS_TABLE} WHERE climb_uuid = ?"""

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(SELECT_STATS_SQL, (climb_uuid,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            return None

# Fetch the first X rows from the "climbs" table
def fetch_first_rows(limit):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
    
    SELECT_SQL = f"SELECT * FROM {CLIMBS_TABLE} LIMIT ?"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL, (limit,))
        return cursor.fetchall()


# Fetch a specific row by climb_uuid
def fetch_row_by_climb_uuid(climb_uuid):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    SELECT_BY_UUID_SQL = f"SELECT * FROM {CLIMBS_TABLE} WHERE climb_uuid = ?"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_BY_UUID_SQL, (climb_uuid,))
        return cursor.fetchone()

# Fetch the top X climbs based on number of ascensents
def fetch_top_climbs(limit):

    SELECT_SQL = f"SELECT * FROM {CLIMB_CACHE_FIELDS_TABLE} ORDER BY ascensionist_count DESC LIMIT ?"

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL, (limit,))
        return cursor.fetchall()


# Fetch climb stats by climb_uuid
def fetch_climb_stats(climb_uuid):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    SELECT_STATS_SQL = f"""SELECT ascensionist_count, display_difficulty, quality_average 
        FROM {CLIMB_CACHE_FIELDS_TABLE} WHERE climb_uuid = ?"""

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
        
# Outputs a dictionary mapping hole_id to (x, y) coords
def extract_holes():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    SELECT_SQL = "SELECT id, x, y FROM holes"

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(SELECT_SQL)
        rows = cursor.fetchall()

    # dictionary: Hole_id -> (x, y)
    placements = {hole_id: (x, y) for hole_id, x, y in rows}
    
    return placements

def fetch_difficulty_grades():

    SELECT_SQL = f"SELECT * FROM {DIFFICULTY_GRADES_TABLE}"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL)
        records = cursor.fetchall()

        # Return the fetched records
        return records

    except sqlite3.Error as e:
        print(f"An error occurred while fetching data: {e}")
    finally:
        if conn:
            conn.close()

    return None