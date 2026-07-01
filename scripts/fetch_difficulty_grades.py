import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"
TABLE_NAME = "difficulty_grades"

