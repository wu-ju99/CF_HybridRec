"""
生成 1000 个新用户 + 多样化交互数据 → 另存为新数据库
"""
import sqlite3
import random
import copy
from datetime import datetime, timedelta

SRC_DB = r"C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_original_schema_enhanced.sqlite"
DST_DB = r"C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_v2_enhanced.sqlite"

random.seed(42)

# ─── 第一步：复制数据库 ───
print("复制数据库...")
import shutil
shutil.copy2(SRC_DB, DST_DB)
conn = sqlite3.connect(DST_DB)
cur = conn.cursor()

# ─── 第二步：读取参考数据 ───
print("读取参考数据...")

# 口味
cur.execute("SELECT id, name FROM taste WHERE id > 0")
tastes = cur.fetchall()  # [(1,'甘'), (2,'辛'), ...]
taste_map = {t[1]: t[0] for t in tastes}

# 菜系
cur.execute("SELECT id, name FROM cuisine WHERE id > 0")
cuisines = cur.fetchall()

# 所有菜谱 ID
cur.execute("SELECT id, name, cuisine FROM recipe WHERE id > 0")
all_recipes = cur.fetchall()
recipe_ids = [r[0] for r in all_recipes]
recipe_cuisine_map = {r[0]: r[2] for r in all_recipes}  # recipe_id -> cuisine_id

# 食材 → 口味映射
cur.execute("SELECT food, taste FROM ingredient2taste")
ing2taste = cur.fetchall()  # [(ing_id, taste_id), ...]
# 食材 → 菜谱
cur.execute("SELECT recipe, ingredient FROM recipeingredient")
recipe_ingredients = cur.fetchall()
# 构建 食材→口味 倒排
ing_tastes = {}
for ing_id, taste_id in ing2taste:
    ing_tastes.setdefault(ing_id, set()).add(taste_id)
# 构建 菜谱→口味 聚合（用于生成偏好）
recipe_taste_profile = {}
for r_id, ing_id in recipe_ingredients:
    if r_id in recipe_cuisine_map:
        for t_id in ing_tastes.get(ing_id, set()):
            recipe_taste_profile.setdefault(r_id, set()).add(t_id)

# address 表（744k+行，只取前 5000 个采样即可）
cur.execute("SELECT id, name, parent, province, city, district FROM address LIMIT 5000")
addresses = cur.fetchall()
# 简单区分：有 province 无 city 的视为省级，有 city 的视为市级
# province/city 可能为 None 或空字符串
def is_not_empty(v):
    return v is not None and str(v).strip() != ''
def is_empty(v):
    return v is None or str(v).strip() == ''
province_ids = [a[0] for a in addresses if is_not_empty(a[3]) and is_empty(a[4])]
city_ids = [a[0] for a in addresses if is_not_empty(a[4])]
all_addr_ids = [a[0] for a in addresses]
# 兜底：如果没分出来，就用全部
if not province_ids: province_ids = all_addr_ids[:200]
if not city_ids: city_ids = all_addr_ids[200:400]

# occupation 表
cur.execute("SELECT id, name FROM occupation")
occupations = cur.fetchall()
if occupations:
    occupation_ids = [o[0] for o in occupations]
else:
    # 如果 occupation 是独立表不存在，用常见职业 ID
    occupation_ids = list(range(1, 21))

# ingredient 表
cur.execute("SELECT id, foodtype FROM ingredient WHERE id > 0")
ingredients = cur.fetchall()
ing_foodtype = {i[0]: i[1] for i in ingredients}

# 查看现有最大用户 ID
cur.execute("SELECT MAX(id) FROM user")
max_uid = cur.fetchone()[0] or 1000000
next_uid = max(max_uid + 1, 1001001)
print(f"新用户起始 ID: {next_uid}")

# ─── 第三步：构建用户画像原型 ───
# 定义 8 种用户原型，每种有独特的口味偏好和交互模式

user_prototypes = [
    {
        'name': '嗜辣青年',
        'gender': '男',
        'age_range': (18, 35),
        'preferred_tastes': {'辛': (4,5), '酸': (3,4), '咸': (3,5)},
        'avoided_tastes': {'甘': (3,5)},
        'preferred_cuisines': [1, 4, 5],  # 川菜、湘菜等
        'interaction_bias': 'meat_heavy',   # 偏荤
        'browse_rate': 0.6,
        'fond_rate': 0.3,      # 高喜好率
        'avoid_rate': 0.1,
        'cook_rate': 0.15,
        'occ_bias': ['程序员', '销售', '外卖骑手', '学生'],
    },
    {
        'name': '养生老年',
        'gender': '男',
        'age_range': (55, 80),
        'preferred_tastes': {'甘': (3,5), '淡': (3,5)},
        'avoided_tastes': {'辛': (3,5), '咸': (3,5), '苦': (2,4)},
        'preferred_cuisines': [2, 3, 6],  # 粤菜、鲁菜、淮扬
        'interaction_bias': 'light_veg',
        'browse_rate': 0.4,
        'fond_rate': 0.25,
        'avoid_rate': 0.2,     # 高规避率
        'cook_rate': 0.2,
        'occ_bias': ['退休', '教师', '公务员'],
    },
    {
        'name': '精致主妇',
        'gender': '女',
        'age_range': (25, 50),
        'preferred_tastes': {'甘': (3,5), '淡': (2,4), '酸': (2,4)},
        'avoided_tastes': {'苦': (3,5), '辛': (1,3)},
        'preferred_cuisines': [2, 3, 7, 8],  # 粤菜、鲁菜、苏菜、浙菜
        'interaction_bias': 'balanced',
        'browse_rate': 0.5,
        'fond_rate': 0.3,
        'avoid_rate': 0.1,
        'cook_rate': 0.25,     # 高下厨率
        'occ_bias': ['教师', '护士', '会计', '行政'],
    },
    {
        'name': '健身青年',
        'gender': '男',
        'age_range': (20, 40),
        'preferred_tastes': {'淡': (3,5), '甘': (2,4)},
        'avoided_tastes': {'咸': (3,5), '辛': (2,4)},
        'preferred_cuisines': [2, 6, 9],  # 粤菜、轻食
        'interaction_bias': 'high_protein',
        'browse_rate': 0.5,
        'fond_rate': 0.25,
        'avoid_rate': 0.15,
        'cook_rate': 0.2,
        'occ_bias': ['健身教练', '白领', '学生'],
    },
    {
        'name': '甜食女生',
        'gender': '女',
        'age_range': (15, 30),
        'preferred_tastes': {'甘': (4,5), '淡': (2,4)},
        'avoided_tastes': {'苦': (3,5), '辛': (2,5), '咸': (2,4)},
        'preferred_cuisines': [2, 3, 10, 11],  # 粤菜、甜品
        'interaction_bias': 'sweet_dessert',
        'browse_rate': 0.6,
        'fond_rate': 0.3,
        'avoid_rate': 0.08,
        'cook_rate': 0.12,
        'occ_bias': ['学生', '设计师', '前台'],
    },
    {
        'name': '嗜辣女生',
        'gender': '女',
        'age_range': (18, 40),
        'preferred_tastes': {'辛': (4,5), '酸': (3,5), '咸': (3,4)},
        'avoided_tastes': {'甘': (2,4)},
        'preferred_cuisines': [1, 4, 5],  # 川湘
        'interaction_bias': 'meat_heavy',
        'browse_rate': 0.55,
        'fond_rate': 0.3,
        'avoid_rate': 0.1,
        'cook_rate': 0.18,
        'occ_bias': ['销售', '市场', '主播', '学生'],
    },
    {
        'name': '传统中老',
        'gender': '女',
        'age_range': (45, 75),
        'preferred_tastes': {'咸': (3,5), '甘': (2,4)},
        'avoided_tastes': {'苦': (3,5), '辛': (2,4)},
        'preferred_cuisines': [3, 6, 7, 8],  # 鲁菜、家常
        'interaction_bias': 'balanced',
        'browse_rate': 0.4,
        'fond_rate': 0.2,
        'avoid_rate': 0.2,
        'cook_rate': 0.25,
        'occ_bias': ['退休', '农民', '工人'],
    },
    {
        'name': '美食探索者',
        'gender': '男',
        'age_range': (25, 55),
        'preferred_tastes': {'辛': (2,5), '酸': (2,5), '甘': (2,5), '咸': (2,5)},
        'avoided_tastes': {},
        'preferred_cuisines': [],  # 什么都吃
        'interaction_bias': 'diverse',
        'browse_rate': 0.65,
        'fond_rate': 0.3,
        'avoid_rate': 0.05,
        'cook_rate': 0.2,
        'occ_bias': ['厨师', '美食博主', '记者', '导游'],
    },
]

# 扩展更多原型细微变化（给每个原型加些变体）
expanded_prototypes = []
for proto in user_prototypes:
    expanded_prototypes.append(proto)
    # 对主要原型做一些偏移变体
    for offset in range(1, 4):
        variant = copy.deepcopy(proto)
        variant['name'] = f"{proto['name']}_v{offset}"
        # 年龄稍偏移
        a_min, a_max = proto['age_range']
        variant['age_range'] = (a_min + offset * 3, a_max + offset * 3)
        # 口味偏好稍有变化
        for t, (lo, hi) in variant.get('preferred_tastes', {}).items():
            variant['preferred_tastes'][t] = (max(1, lo-1), min(5, hi+1))
        expanded_prototypes.append(variant)

# ─── 第四步：生成 1000 个用户 ───
print(f"生成 {1000} 个新用户（{len(expanded_prototypes)} 种原型）...")

new_users = []
new_usertaste = []
new_fondness = []
new_avoid = []
new_actual = []
new_browse = []
new_browse_detail = []

for i in range(1000):
    uid = next_uid + i
    proto = expanded_prototypes[i % len(expanded_prototypes)]

    # 性别
    gender = proto['gender']

    # 生日 → 年龄
    age = random.randint(*proto['age_range'])
    birthday = (datetime(2026, 1, 1) - timedelta(days=int(age * 365.25))).strftime('%Y-%m-%d')

    # 职业
    occ_id = random.choice(occupation_ids) if occupation_ids else random.randint(1, 20)

    # 出生地 / 工作地（从 address 表随机选）
    birthplace = random.choice(all_addr_ids) if all_addr_ids else None
    workplace = random.choice(all_addr_ids) if all_addr_ids else None

    # 用户名
    name = f"新用户{i+1:04d}"

    new_users.append((uid, name, gender, birthday, occ_id, birthplace, workplace))

# 批量插入用户（模拟原表结构）
# 先看看 user 表有哪些列
cur.execute("PRAGMA table_info(user)")
user_cols = [c[1] for c in cur.fetchall()]
print(f"user 表列: {user_cols}")

# 构建 INSERT
placeholders = ','.join(['?'] * len(user_cols))
user_insert_cols = ','.join(user_cols)
insert_user_sql = f"INSERT INTO user ({user_insert_cols}) VALUES ({placeholders})"

for uid, name, gender, birthday, occ_id, bp, wp in new_users:
    row = {}
    for col in user_cols:
        if col == 'id': row[col] = uid
        elif col == 'name': row[col] = name
        elif col == 'gender': row[col] = gender
        elif col == 'birthday': row[col] = birthday
        elif col == 'occupation': row[col] = occ_id
        elif col == 'birthplace': row[col] = bp
        elif col == 'workplace': row[col] = wp
        elif col == 'firstname': row[col] = name
        elif col == 'lastname': row[col] = ''
        elif col == 'username': row[col] = f"user{uid}"
        elif col == 'password': row[col] = '123456'
        elif col == 'enabled': row[col] = 1
        else: row[col] = None
    vals = [row.get(col) for col in user_cols]
    cur.execute(insert_user_sql, vals)
conn.commit()
print(f"  已插入 {len(new_users)} 个用户")

# ─── 第五步：生成 usertaste（口味偏好）───
print("生成用户口味偏好...")
# 先看 usertaste 表结构
cur.execute("PRAGMA table_info(usertaste)")
ut_cols = [c[1] for c in cur.fetchall()]
print(f"usertaste 列: {ut_cols}")

# 查当前最大 usertaste id
cur.execute("SELECT MAX(id) FROM usertaste")
max_ut_id = cur.fetchone()[0] or 0
next_ut_id = max_ut_id + 1

ut_insert_cols = ','.join(ut_cols)
ut_placeholders = ','.join(['?'] * len(ut_cols))

for idx, (uid, name, gender, birthday, occ_id, bp, wp) in enumerate(new_users):
    proto = expanded_prototypes[idx % len(expanded_prototypes)]
    pref_tastes = proto.get('preferred_tastes', {})
    avoid_tastes = proto.get('avoided_tastes', {})

    # 给用户分配口味
    user_assigned = set()
    # 偏好口味：挑 3-6 种，level 按范围给
    for t_name, (lo, hi) in pref_tastes.items():
        if t_name in taste_map and t_name not in user_assigned:
            level = random.randint(lo, hi)
            vals = {}
            for col in ut_cols:
                if col == 'id': vals[col] = next_ut_id; next_ut_id += 1
                elif col == 'user': vals[col] = uid
                elif col == 'taste': vals[col] = taste_map[t_name]
                elif col == 'level': vals[col] = level
                elif col == 'name': vals[col] = t_name
                else: vals[col] = None
            new_usertaste.append([vals.get(col) for col in ut_cols])
            user_assigned.add(t_name)

    # 规避口味（用低 level 1-2 表示不喜欢）
    for t_name, (lo, hi) in avoid_tastes.items():
        if t_name in taste_map and t_name not in user_assigned:
            level = random.randint(1, max(2, lo))
            vals = {}
            for col in ut_cols:
                if col == 'id': vals[col] = next_ut_id; next_ut_id += 1
                elif col == 'user': vals[col] = uid
                elif col == 'taste': vals[col] = taste_map[t_name]
                elif col == 'level': vals[col] = level  # 低分表示不喜欢
                elif col == 'name': vals[col] = t_name
                else: vals[col] = None
            new_usertaste.append([vals.get(col) for col in ut_cols])
            user_assigned.add(t_name)

# 批量插入 usertaste（每批 500 条）
batch_size = 500
for i in range(0, len(new_usertaste), batch_size):
    batch = new_usertaste[i:i+batch_size]
    for vals in batch:
        cur.execute(f"INSERT INTO usertaste ({ut_insert_cols}) VALUES ({','.join(['?']*len(vals))})", vals)
conn.commit()
print(f"  已插入 {len(new_usertaste)} 条口味偏好")

# ─── 第六步：生成交互数据 ───
print("生成交互数据...")

# 按原型分析菜谱的"口味匹配度"
# 对每个菜谱，计算它属于哪些口味的集合
recipe_taste_scores = {}
for rid in recipe_ids:
    tastes = recipe_taste_profile.get(rid, set())
    # 计算各口味在菜谱中的占比
    recipe_taste_scores[rid] = tastes

# 对于原型，计算每个菜谱的匹配分
def score_recipe_for_proto(rid, proto):
    tastes_in_recipe = recipe_taste_scores.get(rid, set())
    score = 0
    # 偏好口味加分
    for t_name, (lo, hi) in proto.get('preferred_tastes', {}).items():
        t_id = taste_map.get(t_name)
        if t_id and t_id in tastes_in_recipe:
            score += lo  # 最低期望分作为权重
    # 规避口味减分
    for t_name, (lo, hi) in proto.get('avoided_tastes', {}).items():
        t_id = taste_map.get(t_name)
        if t_id and t_id in tastes_in_recipe:
            score -= hi
    # 偏好的菜系加分
    pref_cuis = proto.get('preferred_cuisines', [])
    cu = recipe_cuisine_map.get(rid)
    if cu and pref_cuis and cu in pref_cuis:
        score += 2
    return score

# 查交互表当前最大 ID
for table, id_col in [('userfondnessrecipe', 'id'), ('useravoidrecipe', 'id'),
                      ('useracutalrecipe', 'id'), ('userbrowse', 'id'),
                      ('userbrowsedetail', 'id')]:
    cur.execute(f"SELECT MAX({id_col}) FROM {table}")
    max_id = cur.fetchone()[0] or 0
    # print(f"  {table} max {id_col}: {max_id}")

cur.execute("SELECT MAX(id) FROM userfondnessrecipe")
max_fond_id = cur.fetchone()[0] or 0
cur.execute("SELECT MAX(id) FROM useravoidrecipe")
max_avoid_id = cur.fetchone()[0] or 0
cur.execute("SELECT MAX(id) FROM useracutalrecipe")
max_actual_id = cur.fetchone()[0] or 0
cur.execute("SELECT MAX(id) FROM userbrowse")
max_browse_id = cur.fetchone()[0] or 0
cur.execute("SELECT MAX(id) FROM userbrowsedetail")
max_bd_id = cur.fetchone()[0] or 0

# 查看各交互表的列
cur.execute("PRAGMA table_info(userfondnessrecipe)")
fond_cols = [c[1] for c in cur.fetchall()]
cur.execute("PRAGMA table_info(useravoidrecipe)")
avoid_cols = [c[1] for c in cur.fetchall()]
cur.execute("PRAGMA table_info(useracutalrecipe)")
actual_cols = [c[1] for c in cur.fetchall()]
# 查看 userbrowse 和 userbrowsedetail
cur.execute("PRAGMA table_info(userbrowse)")
browse_cols = [c[1] for c in cur.fetchall()]
cur.execute("PRAGMA table_info(userbrowsedetail)")
bd_cols = [c[1] for c in cur.fetchall()]

# 对每个用户
fond_insert_data = []
avoid_insert_data = []
actual_insert_data = []
browse_insert_data = []
bd_insert_data = []

for idx, (uid, name, gender, birthday, occ_id, bp, wp) in enumerate(new_users):
    proto = expanded_prototypes[idx % len(expanded_prototypes)]

    # 为所有菜谱计算匹配分
    scored = [(score_recipe_for_proto(rid, proto), rid) for rid in recipe_ids]
    scored.sort(reverse=True)
    high_match = [r for s, r in scored if s > 0]
    medium_match = [r for s, r in scored if s <= 0]
    low_match = [r for s, r in scored if s < -1]

    # 实际做过（cook_rate 概率选高分菜谱）
    n_cook = max(0, int(len(high_match) * proto.get('cook_rate', 0.15) * random.uniform(0.8, 1.2)))
    cooked = random.sample(high_match, min(n_cook, len(high_match))) if high_match and n_cook > 0 else []
    for rid in cooked:
        max_actual_id += 1
        vals = {}
        for col in actual_cols:
            if col == 'id': vals[col] = max_actual_id
            elif col in ('user', 'user_id'): vals[col] = uid
            elif col in ('recipe', 'recipe_id'): vals[col] = rid
            elif col in ('name',): vals[col] = f"cook_{uid}_{rid}"
            elif col == 'description': vals[col] = ''
            elif col == 'quantity': vals[col] = 1.0
            elif col == 'time': vals[col] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else: vals[col] = None
        actual_insert_data.append([vals.get(col) for col in actual_cols])

    # 喜好标记（从高分菜谱选 + 部分中分菜谱）
    n_fond = max(0, int(len(high_match) * proto.get('fond_rate', 0.25) * random.uniform(0.8, 1.2)))
    fond_pool = [r for r in high_match if r not in cooked]
    # 也加一些中分菜谱作为轻度喜好
    medium_fond = random.sample(medium_match, min(int(n_fond * 0.3), len(medium_match))) if medium_match else []
    fond_selected = random.sample(fond_pool, min(n_fond, len(fond_pool))) if fond_pool else []
    fond_selected = list(set(fond_selected + medium_fond))
    for rid in fond_selected:
        # 强度与匹配分相关
        s = score_recipe_for_proto(rid, proto)
        intensity = min(5, max(1, int(s / 2 + 2)))
        max_fond_id += 1
        vals = {}
        for col in fond_cols:
            if col == 'id': vals[col] = max_fond_id
            elif col in ('user', 'user_id'): vals[col] = uid
            elif col in ('recipe', 'recipe_id'): vals[col] = rid
            elif col in ('intensity', 'rating'): vals[col] = intensity
            else: vals[col] = None
        fond_insert_data.append([vals.get(col) for col in fond_cols])

    # 规避（从低分菜谱选）
    n_avoid = max(0, int(len(recipe_ids) * proto.get('avoid_rate', 0.1) * random.uniform(0.8, 1.2)))
    avoid_selected = random.sample(low_match, min(n_avoid, len(low_match))) if low_match and n_avoid > 0 else []
    for rid in avoid_selected:
        max_avoid_id += 1
        vals = {}
        for col in avoid_cols:
            if col == 'id': vals[col] = max_avoid_id
            elif col in ('user', 'user_id'): vals[col] = uid
            elif col in ('recipe', 'recipe_id'): vals[col] = rid
            elif col in ('intensity', 'rating'): vals[col] = random.randint(1, 3)
            else: vals[col] = None
        avoid_insert_data.append([vals.get(col) for col in avoid_cols])

    # 浏览（各类菜谱混合，但高分菜谱浏览更多）
    n_browse = max(0, int(random.gauss(45, 10)))
    browse_pool = []
    # 60% 来自高分菜谱
    n_high = int(n_browse * 0.6)
    browse_pool += random.sample(high_match, min(n_high, len(high_match)))
    # 30% 来自中分菜谱
    n_mid = int(n_browse * 0.3)
    browse_pool += random.sample(medium_match, min(n_mid, len(medium_match)))
    # 10% 来自低分菜谱（偶尔看看不喜欢的）
    n_low = max(1, int(n_browse * 0.1))
    if low_match:
        browse_pool += random.sample(low_match, min(n_low, len(low_match)))
    browse_pool = list(set(browse_pool))
    random.shuffle(browse_pool)
    browse_pool = browse_pool[:n_browse]

    if browse_pool:
        max_browse_id += 1
        bvals = {}
        for col in browse_cols:
            if col == 'id': bvals[col] = max_browse_id
            elif col in ('user', 'user_id'): bvals[col] = uid
            elif col == 'name': bvals[col] = f"browse_{uid}"
            else: bvals[col] = None
        browse_insert_data.append([bvals.get(col) for col in browse_cols])
        browse_main_id = max_browse_id

        for rid in browse_pool:
            max_bd_id += 1
            bdvals = {}
            for col in bd_cols:
                if col == 'id': bdvals[col] = max_bd_id
                elif col in ('userbrowse', 'browse_id'): bdvals[col] = browse_main_id
                elif col == 'entityid': bdvals[col] = rid
                else: bdvals[col] = None
            bd_insert_data.append([bdvals.get(col) for col in bd_cols])

# 批量插入
for tbl_name, data, cols in [
    ('userfondnessrecipe', fond_insert_data, fond_cols),
    ('useravoidrecipe', avoid_insert_data, avoid_cols),
    ('useracutalrecipe', actual_insert_data, actual_cols),
]:
    if data:
        cols_str = ','.join(cols)
        ph = ','.join(['?'] * len(cols))
        for i in range(0, len(data), 200):
            batch = data[i:i+200]
            for vals in batch:
                cur.execute(f"INSERT INTO {tbl_name} ({cols_str}) VALUES ({ph})", vals)
        conn.commit()
        print(f"  {tbl_name}: {len(data)} 条")

# 处理 userbrowse（1:N 关系，先插 browse 再插 detail）
if browse_insert_data:
    cols_str = ','.join(browse_cols)
    ph = ','.join(['?'] * len(browse_cols))
    for vals in browse_insert_data:
        cur.execute(f"INSERT INTO userbrowse ({cols_str}) VALUES ({ph})", vals)
    conn.commit()
    print(f"  userbrowse: {len(browse_insert_data)} 条")

if bd_insert_data:
    cols_str = ','.join(bd_cols)
    ph = ','.join(['?'] * len(bd_cols))
    for i in range(0, len(bd_insert_data), 200):
        batch = bd_insert_data[i:i+200]
        for vals in batch:
            cur.execute(f"INSERT INTO userbrowsedetail ({cols_str}) VALUES ({ph})", vals)
    conn.commit()
    print(f"  userbrowsedetail: {len(bd_insert_data)} 条")

# ─── 统计验证 ───
print("\n" + "=" * 60)
print("  新数据库生成完毕！统计信息：")
print("=" * 60)

# 新用户统计
cur.execute("SELECT COUNT(*) FROM user WHERE id >= ?", (next_uid,))
print(f"  新增用户: {cur.fetchone()[0]}")

# 交互统计
cur.execute("SELECT COUNT(*) FROM userfondnessrecipe WHERE user >= ?", (next_uid,))
print(f"  新增喜好: {cur.fetchone()[0]} 条")
cur.execute("SELECT COUNT(*) FROM useravoidrecipe WHERE user >= ?", (next_uid,))
print(f"  新增规避: {cur.fetchone()[0]} 条")
cur.execute("SELECT COUNT(*) FROM useracutalrecipe WHERE user >= ?", (next_uid,))
print(f"  新增实际做过: {cur.fetchone()[0]} 条")
cur.execute("""
    SELECT COUNT(*) FROM userbrowse ub
    JOIN userbrowsedetail ubd ON ubd.userbrowse = ub.id
    WHERE ub.user >= ?
""", (next_uid,))
print(f"  新增浏览: {cur.fetchone()[0]} 条")

# 口味多样性检查
cur.execute("""
    SELECT ut.taste, t.name, COUNT(DISTINCT ut.user) as cnt, AVG(ut.level) as avg_level
    FROM usertaste ut
    JOIN taste t ON t.id = ut.taste
    WHERE ut.user >= ?
    GROUP BY ut.taste
    ORDER BY cnt DESC
""", (next_uid,))
print(f"\n  新增用户口味分布:")
for r in cur.fetchall():
    print(f"    {r[1]}: {r[2]} 用户, 平均 level={r[3]:.1f}")

# 用户交互量分布
cur.execute("""
    SELECT ub.user, COUNT(*) as cnt
    FROM userbrowse ub
    JOIN userbrowsedetail ubd ON ubd.userbrowse = ub.id
    WHERE ub.user >= ?
    GROUP BY ub.user
""", (next_uid,))
browse_counts = [r[1] for r in cur.fetchall()]
if browse_counts:
    print(f"\n  浏览分布: min={min(browse_counts)}, max={max(browse_counts)}, avg={sum(browse_counts)/len(browse_counts):.0f}")

conn.close()
print(f"\n新数据库已保存到: {DST_DB}")
print("现在修改 cf_demo_v2.py 里的 DB_PATH 指向新数据库即可重新训练")
