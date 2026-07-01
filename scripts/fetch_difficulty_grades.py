import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"
TABLE_NAME = "difficulty_grades"

def fetch_difficulty_grades():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {TABLE_NAME}")
        records = cursor.fetchall()

        # Return the fetched records
        return records

    except sqlite3.Error as e:
        print(f"An error occurred while fetching data: {e}")
    finally:
        if conn:
            conn.close()

    return None

if __name__ == "__main__":
    grades = fetch_difficulty_grades()

    for record in grades:
        print(record)