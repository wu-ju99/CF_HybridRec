import sqlite3

DB_PATH = r"C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_original_schema_enhanced.sqlite"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 获取所有表名
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print(f"总表数: {len(tables)}")

# 探索关键表
key_tables = [
    'user', 'usertaste', 'taste', 'cuisine', 'foodtype', 'nature',
    'ingredient', 'ingredient2taste', 'ingredient2nature',
    'recipe', 'recipeingredient',
    'cookstep', 'cookmethod',
    'recipecomposite', 'composition',
    'userfondnessrecipe', 'useravoidrecipe', 'useracutalrecipe',
    'userbrowse', 'userbrowsedetail',
    'address', 'occupation'
]

for t in key_tables:
    if t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = cur.fetchall()
        print(f"\n--- {t} ---")
        for c in cols:
            print(f"  {c[1]} ({c[2]})")
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cur.fetchone()[0]
        print(f"  行数: {cnt}")
    else:
        print(f"\n--- {t} --- 表不存在")

# 查看 address 表内容
if 'address' in tables:
    cur.execute("SELECT id, name, parent, level FROM address LIMIT 30")
    print("\n--- address 示例 ---")
    for r in cur.fetchall():
        print(f"  id={r[0]}, name={r[1]}, parent={r[2]}, level={r[3]}")

# 查看 occupation 表内容
if 'occupation' in tables:
    cur.execute("SELECT id, name FROM occupation")
    print("\n--- occupation 示例 ---")
    for r in cur.fetchall():
        print(f"  id={r[0]}, name={r[1]}")
else:
    # occupation 可能是 user 表的直接字段
    cur.execute("SELECT DISTINCT occupation FROM user WHERE occupation IS NOT NULL LIMIT 30")
    print("\n--- user.occupation 值 ---")
    for r in cur.fetchall():
        print(f"  {r[0]}")

# 查看 recipe 的 cuisine 分布
cur.execute("SELECT cuisine, COUNT(*) FROM recipe WHERE cuisine > 0 GROUP BY cuisine ORDER BY COUNT(*) DESC LIMIT 20")
print("\n--- recipe.cuisine 分布 ---")
for r in cur.fetchall():
    print(f"  cuisine={r[0]}, count={r[1]}")

# 查看现有 user 的 birthday 分布
cur.execute("SELECT birthday FROM user WHERE birthday IS NOT NULL AND birthday != '' LIMIT 10")
print("\n--- user.birthday 样例 ---")
for r in cur.fetchall():
    print(f"  {r[0]}")

conn.close()
