import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"

# Table names:
BETA_LINKS_TABLE = "beta_links" # Angle
CLIMB_CACHE_FIELDS_TABLE = "climb_cache_fields" # Difficulty, quality, ascensionist count
CLIMBS_TABLE = "climbs" # Frames and other climb data
DIFFICULTY_GRADES_TABLE = "difficulty_grades" # Grading Scale
CLIMB_STATS_TABLE = "climb_stats" # Climb stats

# In-memory cache for table sizes
TABLE_SIZE_CACHE = {}


# Get the size of a table in the database, cache to prevent repeated calculation
def get_table_size(table_name):
    if table_name in TABLE_SIZE_CACHE:
        return TABLE_SIZE_CACHE[table_name]

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        size = cursor.fetchone()[0]

    TABLE_SIZE_CACHE[table_name] = size
    return size


# Fetch X rows from the "climbs" table starting at offset
def fetch_rows_from(limit, offset=0):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    # Check if offset is within bounds of the table size
    total_count = get_table_size(CLIMBS_TABLE)
    if offset >= total_count or offset < 0:
        return []

    SELECT_SQL = f"SELECT * FROM {CLIMBS_TABLE} LIMIT ? OFFSET ?"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL, (limit, offset))
        return cursor.fetchall()


# Fetch a sample of rows from several random locations in the table.
def fetch_random_rows(limit):
    if limit <= 0:
        return []

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check random offsets stay inside the table bounds
        total_count = get_table_size(CLIMBS_TABLE)
        if total_count == 0:
            return []

        # Return whole table if at table limit
        if limit >= total_count:
            cursor.execute(f"SELECT * FROM {CLIMBS_TABLE}")
            rows = cursor.fetchall()
        else:
            # Use random offsets to pull chunks from different parts of the table
            chunk_size = max(1, limit // 4)
            num_chunks = max(1, min(4, limit))
            offsets = [random.randrange(total_count - chunk_size + 1) for _ in range(num_chunks)]

            rows = []
            for offset in offsets:
                cursor.execute(
                    f"SELECT * FROM {CLIMBS_TABLE} LIMIT ? OFFSET ?",
                    (chunk_size, offset),
                )
                rows.extend(cursor.fetchall())

    return rows[:limit]


# Fetch a specific row by climb_uuid
def fetch_row_by_id(climb_uuid):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    SELECT_BY_UUID_SQL = f"SELECT * FROM {CLIMBS_TABLE} WHERE climb_uuid = ?"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_BY_UUID_SQL, (climb_uuid,))
        return cursor.fetchone()


# Fetch the most popular X climbs based on number of total completions
def fetch_popular_climbs(limit):
    if limit <= 0:
        return []

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    total_count = get_table_size(CLIMB_CACHE_FIELDS_TABLE)
    if total_count == 0:
        return []

    SELECT_SQL = f"SELECT * FROM {CLIMB_CACHE_FIELDS_TABLE} ORDER BY ascensionist_count DESC LIMIT ?"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL, (min(limit, total_count),))
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
            # data in a dictionary
            return {
                'ascensionist_count': result[0],
                'display_difficulty': result[1],
                'quality_average': result[2]
            }
        else:
            return None
    

# Fetch climb angle by climb_uuid
def fetch_angle(climb_uuid):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
    
    SELECT_STATS_SQL = f"""SELECT angle FROM {CLIMB_STATS_TABLE} WHERE climb_uuid = ?"""
    ##Switched to Climb Stats table to get angle, looking at the data its the only one with no nulls

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(SELECT_STATS_SQL, (climb_uuid,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
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


# get the difficulty grades scale from the database
def extract_difficulty_grades():

    SELECT_SQL = f"SELECT * FROM {DIFFICULTY_GRADES_TABLE}"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(SELECT_SQL)
        records = cursor.fetchall()

        # Return the grade scale
        return records

    except sqlite3.Error as e:
        print(f"An error occurred while fetching data: {e}")
    finally:
        if conn:
            conn.close()
    return None
