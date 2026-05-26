import sqlite3, os

db_path = r"D:\GeoEastRC\GeoEastData\DATA\sqlitedb\ndpsqlite\ndp_catalog.db"
print(f"Opening: {db_path}")
print(f"Exists: {os.path.exists(db_path)}")
print(f"Size: {os.path.getsize(db_path)}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"Tables: {tables}")
conn.close()

db_path2 = r"D:\GeoEastRC\GeoEastData\DATA\sqlitedb\ndpsqlite\ndpsys.db"
print(f"\nOpening: {db_path2}")
print(f"Exists: {os.path.exists(db_path2)}")
print(f"Size: {os.path.getsize(db_path2)}")

conn2 = sqlite3.connect(db_path2)
cursor2 = conn2.cursor()
tables2 = cursor2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"Tables: {tables2}")
conn2.close()
