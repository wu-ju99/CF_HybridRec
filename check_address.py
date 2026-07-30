import sqlite3
DB_PATH = r"C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_original_schema_enhanced.sqlite"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA table_info(address)")
cols = cur.fetchall()
print("address 表列:")
for c in cols:
    print(f"  {c[1]} ({c[2]})")
cur.execute("SELECT * FROM address LIMIT 20")
print("\n前20条数据:")
for r in cur.fetchall():
    print(f"  {r}")
conn.close()
