"""
基于协同过滤的混合菜谱推荐算法 Demo v2 — GPU 加速版
===================================================
与 cf_demo_v2.py 算法逻辑完全一致，仅将 NumPy 替换为 CuPy。
自动检测 GPU，无 GPU 时回退到 NumPy。

依赖：pip install cupy-cuda12x numpy pandas scikit-learn --break-system-packages
（根据 CUDA 版本调整：cupy-cuda11x / cupy-cuda12x）

用法：python cf_demo_v2_gpu.py
"""

import sqlite3
import pandas as pd
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─── GPU 自动检测 ────────────────────────────────────────
try:
    import cupy as cp
    _HAS_CUDA = True
    # 确认实际可用
    _ = cp.zeros(1)
    print(f"[GPU] CuPy {cp.__version__} 已加载，设备: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except Exception as e:
    import numpy as cp
    _HAS_CUDA = False
    print(f"[GPU] CuPy 不可用 ({e})，回退到 NumPy CPU 模式")

xp = cp  # 统一使用 xp，后续所有数组操作都用 xp.xxx

# ─── 数据库路径 ──────────────────────────────────────────
DB_PATH = r"C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_v2_enhanced.sqlite"


# ============================================================
# GPU 工具函数（替代 sklearn 的 CPU 实现）
# ============================================================

def cosine_similarity_gpu(X):
    """GPU 版余弦相似度矩阵，与 sklearn.metrics.pairwise.cosine_similarity 等价"""
    norm = xp.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1  # 避免除零
    X_normed = X / norm
    return xp.dot(X_normed, X_normed.T)


def truncated_svd_gpu(X, n_components, random_state=42):
    """GPU 版 TruncatedSVD，返回降维后的矩阵 (n_rows, n_components)"""
    rs = xp.random.RandomState(random_state)
    # 随机初始化
    n, d = X.shape
    k = min(n_components, d, n)
    # 使用随机 SVD: 先随机投影再 QR + SVD
    # 比直接做全 SVD 快很多
    Q, _ = xp.linalg.qr(xp.random.randn(d, k))
    for _ in range(3):  # 幂迭代提升精度
        Q, _ = xp.linalg.qr(X.T @ (X @ Q))
    B = X @ Q
    U, s, Vt = xp.linalg.svd(B, full_matrices=False)
    return U[:, :k] * s[:k]


# ============================================================
# 第一部分：从 SQLite 加载真实数据（同原版，CPU 上执行）
# ============================================================

def load_data_from_db():
    """从增强版 SQLite 数据库读取所有算法需要的表"""
    conn = sqlite3.connect(DB_PATH)
    print("=" * 70)
    print("  从 SQLite 数据库加载数据")
    print("=" * 70)

    # ── 1. user 表 ──
    df_user = pd.read_sql_query("""
        SELECT id, name, gender, birthday, occupation, birthplace, workplace
        FROM user
        WHERE id > 0
    """, conn)
    print(f"\n  [user]           {len(df_user):>6} 行  "
          f"使用: id, gender, birthday, occupation, birthplace, workplace")

    # ── 2. usertaste 表 ──
    df_usertaste = pd.read_sql_query("""
        SELECT ut.user, ut.taste, t.name AS taste_name, ut.level
        FROM usertaste ut
        JOIN taste t ON t.id = ut.taste
        WHERE ut.user > 0
    """, conn)
    print(f"  [usertaste]      {len(df_usertaste):>6} 行  "
          f"使用: user, taste→taste.name, level")

    # ── 3. taste 表 (字典) ──
    df_taste = pd.read_sql_query("SELECT id, name FROM taste WHERE id > 0", conn)
    taste_list = df_taste['name'].tolist()
    print(f"  [taste]          {len(df_taste):>6} 行  口味: {', '.join(taste_list)}")

    # ── 4. cuisine 表 (字典) ──
    df_cuisine = pd.read_sql_query("SELECT id, name FROM cuisine WHERE id > 0", conn)
    print(f"  [cuisine]        {len(df_cuisine):>6} 行  菜系数: {len(df_cuisine)}")

    # ── 5. foodtype 表 (字典) ──
    df_foodtype = pd.read_sql_query("SELECT id, name FROM foodtype WHERE id > 0", conn)
    print(f"  [foodtype]       {len(df_foodtype):>6} 行  类型数: {len(df_foodtype)}")

    # ── 6. nature 表 (字典) ──
    df_nature = pd.read_sql_query("SELECT id, name FROM nature WHERE id > 0", conn)
    print(f"  [nature]         {len(df_nature):>6} 行  性质: {', '.join(df_nature['name'].tolist())}")

    # ── 7. ingredient 表 ──
    df_ingredient = pd.read_sql_query("""
        SELECT id, name, foodtype FROM ingredient WHERE id > 0
    """, conn)
    print(f"  [ingredient]     {len(df_ingredient):>6} 行  "
          f"使用: id, name, foodtype")

    # ── 8. ingredient2taste 表 ──
    df_ing2taste = pd.read_sql_query("""
        SELECT food AS ingredient_id, taste AS taste_id FROM ingredient2taste
    """, conn)
    print(f"  [ingredient2taste] {len(df_ing2taste):>5} 行  "
          f"使用: food→ingredient, taste")

    # ── 9. ingredient2nature 表 ──
    df_ing2nature = pd.read_sql_query("""
        SELECT food AS ingredient_id, nature AS nature_id FROM ingredient2nature
    """, conn)
    print(f"  [ingredient2nature] {len(df_ing2nature):>5} 行  "
          f"使用: food→ingredient, nature")

    # ── 10. recipe 表 ──
    df_recipe = pd.read_sql_query("""
        SELECT id, name, cuisine, gi, timeconsumming, cost FROM recipe WHERE id > 0
    """, conn)
    print(f"  [recipe]         {len(df_recipe):>6} 行  "
          f"使用: id, name, cuisine, gi, timeconsumming, cost")

    # ── 11. recipeingredient 表 ──
    df_recipe_ing = pd.read_sql_query("""
        SELECT recipe AS recipe_id, ingredient AS ingredient_id
        FROM recipeingredient
        WHERE recipe > 0 AND ingredient > 0
    """, conn)
    print(f"  [recipeingredient] {len(df_recipe_ing):>5} 行  "
          f"使用: recipe, ingredient")

    # ── 12. cookstep 表 ──
    df_cookstep = pd.read_sql_query("""
        SELECT recipe AS recipe_id, cookmethod AS cookmethod_id
        FROM cookstep WHERE recipe > 0
    """, conn)
    print(f"  [cookstep]       {len(df_cookstep):>6} 行  "
          f"使用: recipe, cookmethod")

    # ── 13. cookmethod 表 (字典) ──
    df_cookmethod = pd.read_sql_query("SELECT id, name FROM cookmethod WHERE id > 0", conn)
    print(f"  [cookmethod]     {len(df_cookmethod):>6} 行  "
          f"方式: {', '.join(df_cookmethod['name'].tolist()[:8])}")

    # ── 14. recipecomposite + composition → 菜谱营养 ──
    df_recipe_nut = pd.read_sql_query("""
        SELECT rc.recipe AS recipe_id, c.name AS nutrient_name, rc.quantity
        FROM recipecomposite rc
        JOIN composition c ON c.id = rc.composition
        WHERE rc.recipe > 0
    """, conn)
    nut_pivot = df_recipe_nut.pivot_table(
        index='recipe_id', columns='nutrient_name', values='quantity', aggfunc='first'
    ).reset_index()
    common_nutrients = ['能量', '蛋白质', '脂肪', '碳水化合物', '膳食纤维']
    for col in common_nutrients:
        if col not in nut_pivot.columns:
            nut_pivot[col] = 0.0
    nut_pivot = nut_pivot[['recipe_id'] + common_nutrients].fillna(0)
    print(f"  [recipecomposite+composition+content] → {len(nut_pivot)} 菜谱营养 "
          f"维度: {common_nutrients}")

    # ── 15. 用户-菜谱交互数据 ──
    df_fond = pd.read_sql_query("""
        SELECT user AS user_id, recipe AS recipe_id, intensity, 'fond' AS source
        FROM userfondnessrecipe WHERE user > 0 AND recipe > 0
    """, conn)
    print(f"  [userfondnessrecipe] {len(df_fond):>5} 行")

    df_avoid = pd.read_sql_query("""
        SELECT user AS user_id, recipe AS recipe_id, intensity, 'avoid' AS source
        FROM useravoidrecipe WHERE user > 0 AND recipe > 0
    """, conn)
    print(f"  [useravoidrecipe]   {len(df_avoid):>5} 行")

    df_actual = pd.read_sql_query("""
        SELECT user AS user_id, recipe AS recipe_id, 5 AS intensity, 'actual' AS source
        FROM useracutalrecipe WHERE user > 0 AND recipe > 0
    """, conn)
    print(f"  [useracutalrecipe]  {len(df_actual):>5} 行")

    df_browse = pd.read_sql_query("""
        SELECT ub.user AS user_id, ubd.entityid AS recipe_id, 2 AS intensity, 'browse' AS source
        FROM userbrowse ub
        JOIN userbrowsedetail ubd ON ubd.userbrowse = ub.id
        WHERE ub.user > 0 AND ubd.entityid > 0
    """, conn)
    print(f"  [userbrowse+browsedetail] {len(df_browse):>5} 行")

    conn.close()

    # ── 合并交互构建伪评分 ──
    interactions = []
    for _, row in df_actual.iterrows():
        interactions.append({'user_id': int(row['user_id']), 'recipe_id': int(row['recipe_id']), 'rating': 5.0})
    for _, row in df_fond.iterrows():
        r = min(5.0, 3.0 + float(row['intensity'] or 2) * 0.4)
        interactions.append({'user_id': int(row['user_id']), 'recipe_id': int(row['recipe_id']), 'rating': r})
    for _, row in df_avoid.iterrows():
        interactions.append({'user_id': int(row['user_id']), 'recipe_id': int(row['recipe_id']), 'rating': 1.0})
    for _, row in df_browse.iterrows():
        interactions.append({'user_id': int(row['user_id']), 'recipe_id': int(row['recipe_id']), 'rating': 2.5})

    df_interactions = pd.DataFrame(interactions).drop_duplicates(subset=['user_id', 'recipe_id'])
    print(f"\n  → 合并交互矩阵: {len(df_interactions)} 条 (去重后)")

    return {
        'user': df_user,
        'usertaste': df_usertaste,
        'taste': df_taste,
        'taste_list': taste_list,
        'cuisine': df_cuisine,
        'foodtype': df_foodtype,
        'nature': df_nature,
        'ingredient': df_ingredient,
        'ing2taste': df_ing2taste,
        'ing2nature': df_ing2nature,
        'recipe': df_recipe,
        'recipe_ing': df_recipe_ing,
        'cookstep': df_cookstep,
        'cookmethod': df_cookmethod,
        'recipe_nut': nut_pivot,
        'interactions': df_interactions,
    }


# ============================================================
# 第二部分：特征工程（完全从数据库表头提取）
# ============================================================

def build_features(data):
    """从数据库真实字段构建用户向量 U 和菜谱向量 V，返回 GPU 数组"""
    print("\n" + "=" * 70)
    print("  特征工程：构建 U (用户) 和 V (菜谱) 向量")
    print("=" * 70)

    df_user = data['user']
    df_taste_ut = data['usertaste']
    taste_list = data['taste_list']
    n_tastes = len(taste_list)
    df_cuisine = data['cuisine']
    n_cuisines = len(df_cuisine)
    df_foodtype = data['foodtype']
    n_foodtypes = len(df_foodtype)
    df_nature = data['nature']
    n_natures = len(df_nature)
    df_ing = data['ingredient']
    df_ing2t = data['ing2taste']
    df_ing2n = data['ing2nature']
    df_recipe = data['recipe']
    df_recipe_ing = data['recipe_ing']
    df_cookstep = data['cookstep']
    df_cookmethod = data['cookmethod']
    n_cookmethods = len(df_cookmethod)
    nut_pivot = data['recipe_nut']

    from datetime import datetime
    now = datetime.now()
    def get_age_group(bday_str):
        if pd.isna(bday_str) or str(bday_str).strip() == '':
            return '未知'
        try:
            bday = pd.to_datetime(bday_str)
            age = (now - bday).days / 365.25
        except:
            return '未知'
        if age < 1: return '婴儿'
        elif age < 3: return '幼儿'
        elif age < 12: return '儿童'
        elif age < 18: return '少年'
        elif age < 45: return '青年'
        else: return '老年'

    df_user['age_group'] = df_user['birthday'].apply(get_age_group)
    age_order = ['婴儿', '幼儿', '儿童', '少年', '青年', '老年', '未知']

    occ_list = sorted(df_user['occupation'].dropna().unique())
    if len(occ_list) > 15:
        occ_counts = df_user['occupation'].value_counts()
        occ_list = occ_counts.head(12).index.tolist()

    all_birth = sorted(df_user['birthplace'].dropna().unique())
    all_work = sorted(df_user['workplace'].dropna().unique())
    all_addrs = sorted(set(list(all_birth) + list(all_work)))
    n_addr = len(all_addrs) if len(all_addrs) <= 50 else 50

    print("\n  ┌─ 用户特征向量 U 的构成 ─────────────────────")
    print(f"  │ 特征组               维度    来源表.字段")
    print(f"  ├─────────────────────────────────────────────")

    user_ids = df_user['id'].tolist()
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    gender_dim = 3
    print(f"  │ 1.性别(OneHot)      {gender_dim:>4}     user.gender")
    age_dim = len(age_order)
    print(f"  │ 2.年龄组(OneHot)    {age_dim:>4}     user.birthday→推算")
    occ_dim = len(occ_list)
    print(f"  │ 3.职业(OneHot)      {occ_dim:>4}     user.occupation")
    taste_dim = n_tastes
    print(f"  │ 4.口味偏好(归一化)   {taste_dim:>4}     usertaste.taste+level")
    bplace_dim = n_addr
    print(f"  │ 5.出生地(OneHot)    {bplace_dim:>4}     user.birthplace→address")
    wplace_dim = n_addr
    print(f"  │ 6.工作地(OneHot)    {wplace_dim:>4}     user.workplace→address")
    d_u = gender_dim + age_dim + occ_dim + taste_dim + bplace_dim + wplace_dim
    print(f"  ├─────────────────────────────────────────────")
    print(f"  │ 总计 d_u = {d_u}")
    print(f"  └─────────────────────────────────────────────")

    # 用 xp 构建 U（直接构建 GPU 数组或 CPU 数组取决于 xp）
    U_cpu = xp.zeros((n_users, d_u), dtype=xp.float32)

    user_taste_map = {}
    for _, row in df_taste_ut.iterrows():
        uid = row['user']
        if uid not in user_id_to_idx:
            continue
        tname = row['taste_name']
        if tname not in taste_list:
            continue
        level_val = row['level']
        level = float(level_val) if pd.notna(level_val) else 0.0
        if uid not in user_taste_map:
            user_taste_map[uid] = xp.zeros(n_tastes, dtype=xp.float32)
        t_idx = taste_list.index(tname)
        user_taste_map[uid][t_idx] = level / 5.0

    for uid in user_ids:
        idx = user_id_to_idx[uid]
        row = df_user[df_user['id'] == uid].iloc[0]
        pos = 0
        g = str(row['gender']).strip()
        U_cpu[idx, pos + (0 if g == '男' else 1 if g == '女' else 2)] = 1
        pos += gender_dim
        ag = row['age_group']
        U_cpu[idx, pos + (age_order.index(ag) if ag in age_order else len(age_order)-1)] = 1
        pos += age_dim
        occ = row['occupation']
        if pd.notna(occ) and occ in occ_list:
            U_cpu[idx, pos + occ_list.index(occ)] = 1
        pos += occ_dim
        if uid in user_taste_map:
            U_cpu[idx, pos:pos+taste_dim] = user_taste_map[uid]
        pos += taste_dim
        bp = row['birthplace']
        if pd.notna(bp) and bp in all_addrs:
            bp_idx = all_addrs.index(bp)
            if bp_idx < n_addr:
                U_cpu[idx, pos + bp_idx] = 1
        pos += bplace_dim
        wp = row['workplace']
        if pd.notna(wp) and wp in all_addrs:
            wp_idx = all_addrs.index(wp)
            if wp_idx < n_addr:
                U_cpu[idx, pos + wp_idx] = 1

    # 如果 U_cpu 是 numpy 数组且 GPU 可用，转移到 GPU
    if _HAS_CUDA and isinstance(U_cpu, xp.ndarray) and not hasattr(U_cpu, 'device'):
        U = cp.asarray(U_cpu)
    else:
        U = U_cpu

    print(f"  → U shape: {U.shape}")

    # ── 构建菜谱特征向量 V ──
    print(f"\n  ┌─ 菜谱特征向量 V 的构成 ─────────────────────")
    print(f"  │ 特征组               维度    来源表.字段")
    print(f"  ├─────────────────────────────────────────────")

    recipe_ids = df_recipe['id'].tolist()
    recipe_id_to_idx = {rid: i for i, rid in enumerate(recipe_ids)}
    n_recipes = len(recipe_ids)

    cuisine_dim = n_cuisines
    print(f"  │ 1.菜系(OneHot)      {cuisine_dim:>4}     recipe.cuisine→cuisine")
    print(f"  │ 2.GI/时长/成本       3     recipe.gi, timeconsumming, cost")
    taste_agg_dim = n_tastes
    print(f"  │ 3.口味特征(聚合)     {taste_agg_dim:>4}     recipeingredient→ingredient→ingredient2taste→taste")
    nature_dim = n_natures
    print(f"  │ 4.食性特征(聚合)     {nature_dim:>4}     recipeingredient→ingredient→ingredient2nature→nature")
    nut_dim = 5
    print(f"  │ 5.营养成分(归一化)    {nut_dim:>4}     recipecomposite→composition→content")
    ft_dim = n_foodtypes
    print(f"  │ 6.食材类型(TF-IDF)   {ft_dim:>4}     recipeingredient→ingredient.foodtype→foodtype")
    cm_dim = n_cookmethods
    print(f"  │ 7.烹饪方式(MultiHot) {cm_dim:>4}     cookstep.cookmethod→cookmethod")
    d_v = cuisine_dim + 3 + taste_agg_dim + nature_dim + nut_dim + ft_dim + cm_dim
    print(f"  ├─────────────────────────────────────────────")
    print(f"  │ 总计 d_v = {d_v}")
    print(f"  └─────────────────────────────────────────────")

    V_cpu = xp.zeros((n_recipes, d_v), dtype=xp.float32)

    ing_foodtype = dict(zip(df_ing['id'], df_ing['foodtype']))
    ing_to_tastes = defaultdict(set)
    for _, row in df_ing2t.iterrows():
        ing_to_tastes[int(row['ingredient_id'])].add(int(row['taste_id']))
    ing_to_natures = defaultdict(set)
    for _, row in df_ing2n.iterrows():
        ing_to_natures[int(row['ingredient_id'])].add(int(row['nature_id']))

    ft_doc_count = defaultdict(int)
    for _, row in df_recipe_ing.iterrows():
        rid = row['recipe_id']
        ing = row['ingredient_id']
        ft = ing_foodtype.get(ing)
        if ft and ft > 0:
            ft_doc_count[rid] = ft_doc_count.get(rid, set())
            if isinstance(ft_doc_count[rid], set):
                ft_doc_count[rid].add(ft)
    ft_global = defaultdict(int)
    for rid, fts in ft_doc_count.items():
        for ft in fts:
            ft_global[ft] += 1

    recipe_cm = defaultdict(set)
    for _, row in df_cookstep.iterrows():
        cm = row['cookmethod_id']
        if pd.notna(cm) and cm > 0:
            recipe_cm[int(row['recipe_id'])].add(int(cm))

    for rid in recipe_ids:
        if rid not in recipe_id_to_idx:
            continue
        idx = recipe_id_to_idx[rid]
        r_row = df_recipe[df_recipe['id'] == rid]
        if len(r_row) == 0:
            continue
        r_row = r_row.iloc[0]
        pos = 0
        cid = r_row['cuisine']
        if pd.notna(cid) and 0 < cid <= n_cuisines:
            V_cpu[idx, pos + int(cid) - 1] = 1
        pos += cuisine_dim
        gi_val = r_row['gi']
        gi = float(gi_val) if pd.notna(gi_val) else 50.0
        V_cpu[idx, pos] = min(max(gi / 100.0, 0), 1)
        t_val = r_row['timeconsumming']
        t = float(t_val) if pd.notna(t_val) else 30.0
        V_cpu[idx, pos+1] = xp.log1p(t) / xp.log1p(120)
        cost_val = r_row['cost']
        cost = float(cost_val) if pd.notna(cost_val) else 20.0
        V_cpu[idx, pos+2] = min(max(cost / 80.0, 0), 1)
        pos += 3
        ings = df_recipe_ing[df_recipe_ing['recipe_id'] == rid]['ingredient_id'].values
        taste_vec = xp.zeros(n_tastes, dtype=xp.float32)
        taste_count = 0
        for ing in ings:
            for t_id in ing_to_tastes.get(int(ing), set()):
                if 0 < t_id <= n_tastes:
                    taste_vec[int(t_id)-1] += 1
                    taste_count += 1
        if taste_count > 0:
            taste_vec /= taste_count
        else:
            taste_vec = xp.ones(n_tastes, dtype=xp.float32) * 0.3
        V_cpu[idx, pos:pos+n_tastes] = taste_vec
        pos += taste_agg_dim
        nature_vec = xp.zeros(n_natures, dtype=xp.float32)
        for ing in ings:
            for n_id in ing_to_natures.get(int(ing), set()):
                if 0 < n_id <= n_natures:
                    nature_vec[int(n_id)-1] += 1
        if xp.sum(nature_vec) > 0:
            nature_vec /= xp.sum(nature_vec)
        V_cpu[idx, pos:pos+n_natures] = nature_vec
        pos += nature_dim
        nut_row = nut_pivot[nut_pivot['recipe_id'] == rid]
        common = ['能量', '蛋白质', '脂肪', '碳水化合物', '膳食纤维']
        max_vals = {'能量': 800, '蛋白质': 50, '脂肪': 60, '碳水化合物': 100, '膳食纤维': 15}
        if len(nut_row) > 0:
            for ci, cn in enumerate(common):
                raw_val = nut_row.iloc[0].get(cn, 0)
                val = float(raw_val) if pd.notna(raw_val) else 0.0
                V_cpu[idx, pos+ci] = min(max(val / max_vals.get(cn, 100), 0), 1)
        pos += nut_dim
        ft_vec = xp.zeros(n_foodtypes, dtype=xp.float32)
        ft_counts = defaultdict(int)
        total_ings = 0
        for ing in ings:
            ft = ing_foodtype.get(int(ing))
            if ft and ft > 0 and ft <= n_foodtypes:
                ft_counts[int(ft)-1] += 1
                total_ings += 1
        if total_ings > 0:
            for ft_idx in range(n_foodtypes):
                tf = ft_counts[ft_idx] / total_ings
                idf = xp.log(n_recipes / (1 + ft_global.get(ft_idx+1, 1)))
                ft_vec[ft_idx] = tf * idf
        else:
            ft_vec = xp.ones(n_foodtypes, dtype=xp.float32) / n_foodtypes
        V_cpu[idx, pos:pos+n_foodtypes] = ft_vec
        pos += ft_dim
        for cm in recipe_cm.get(rid, set()):
            if 0 < cm <= n_cookmethods:
                V_cpu[idx, pos + int(cm) - 1] = 1

    # 转移到 GPU
    if _HAS_CUDA and isinstance(V_cpu, xp.ndarray) and not hasattr(V_cpu, 'device'):
        V = cp.asarray(V_cpu)
    else:
        V = V_cpu

    print(f"  → V shape: {V.shape}")
    U = xp.nan_to_num(U, nan=0.0)
    V = xp.nan_to_num(V, nan=0.0)
    return U, V, user_id_to_idx, recipe_id_to_idx


# ============================================================
# 第三部分：构建评分矩阵（GPU 版本）
# ============================================================

def build_rating_matrix(data, user_id_to_idx, recipe_id_to_idx):
    """从真实交互数据构建评分矩阵，返回 GPU 数组"""
    interactions = data['interactions']
    n_users = len(user_id_to_idx)
    n_recipes = len(recipe_id_to_idx)

    R = xp.full((n_users, n_recipes), xp.nan, dtype=xp.float32)

    for _, row in interactions.iterrows():
        u = int(row['user_id'])
        r = int(row['recipe_id'])
        if u in user_id_to_idx and r in recipe_id_to_idx:
            R[user_id_to_idx[u], recipe_id_to_idx[r]] = float(row['rating'])

    rated = int(xp.sum(~xp.isnan(R)))
    density = rated / (n_users * n_recipes)
    print(f"\n[评分矩阵] R: {R.shape}, 评分数={rated}, 稠密度={density:.6f}")
    return R


# ============================================================
# 第四部分~第十部分：CF 模型（GPU 版本）
# ============================================================

class UserBasedCF:
    def __init__(self, U, K=50, alpha=0.6, tau_min=5):
        self.U, self.K, self.alpha, self.tau_min = U, K, alpha, tau_min
    def fit(self, R):
        self.R, self.n_users, self.n_recipes = R.copy(), R.shape[0], R.shape[1]
        self.r_mean = xp.nanmean(R, axis=1)
        Rc = xp.nan_to_num(R - self.r_mean[:, xp.newaxis], 0)
        self.sim_matrix = cosine_similarity_gpu(Rc)
        self.profile_sim = cosine_similarity_gpu(self.U)
        n = self.n_users
        self.sim_fused = xp.zeros((n, n), dtype=xp.float32)
        for u in range(n):
            for v in range(u+1, n):
                cu = float(xp.sum(~xp.isnan(R[u]) & ~xp.isnan(R[v])))
                ae = self.alpha * min(1.0, cu/self.tau_min)
                sv = ae * float(self.sim_matrix[u,v]) + (1-ae) * float(self.profile_sim[u,v])
                self.sim_fused[u,v] = self.sim_fused[v,u] = sv
            self.sim_fused[u,u] = 1.0
        print(f"  [User-CF] 相似度矩阵: {self.sim_fused.shape}")
    def predict(self, u, i):
        if not xp.isnan(self.R[u,i]): return float(self.R[u,i])
        rated = xp.where(~xp.isnan(self.R[:,i]))[0]
        if len(rated)==0: return float(self.r_mean[u])
        sims = self.sim_fused[u, rated]
        top = xp.argsort(sims)[-self.K:]
        tu, ts = rated[top], sims[top]
        pm = ts > 0
        if not xp.any(pm): return float(self.r_mean[u])
        tu, ts = tu[pm], ts[pm]
        num = float(xp.sum(ts * (self.R[tu, i] - self.r_mean[tu])))
        den = float(xp.sum(xp.abs(ts)))
        return float(self.r_mean[u]) + num/den if den else float(self.r_mean[u])
    def predict_all(self):
        R_pred = self.R.copy()
        for u in range(self.n_users):
            for i in range(self.n_recipes):
                if xp.isnan(R_pred[u,i]):
                    R_pred[u,i] = self.predict(u,i)
        return R_pred


class ItemBasedCF:
    def __init__(self, V, K=30, beta=0.6, tau_min=5):
        self.V, self.K, self.beta, self.tau_min = V, K, beta, tau_min
    def fit(self, R):
        self.R, self.n_users, self.n_recipes = R.copy(), R.shape[0], R.shape[1]
        rm = xp.nanmean(R, axis=1)
        Rc = xp.nan_to_num(R - rm[:, xp.newaxis], 0)
        self.co_sim = cosine_similarity_gpu(Rc.T)
        self.content_sim = cosine_similarity_gpu(self.V)
        n = self.n_recipes
        self.item_uc = xp.sum(~xp.isnan(R), axis=0)
        self.sim_fused = xp.zeros((n, n), dtype=xp.float32)
        for i in range(n):
            for j in range(i+1, n):
                cu = float(xp.sum(~xp.isnan(R[:,i]) & ~xp.isnan(R[:,j])))
                be = self.beta * min(1.0, cu/self.tau_min)
                sv = be * float(self.co_sim[i,j]) + (1-be) * float(self.content_sim[i,j])
                self.sim_fused[i,j] = self.sim_fused[j,i] = sv
            self.sim_fused[i,i] = 1.0
        print(f"  [Item-CF] 相似度矩阵: {self.sim_fused.shape}")
    def predict(self, u, i):
        if not xp.isnan(self.R[u,i]): return float(self.R[u,i])
        rated = xp.where(~xp.isnan(self.R[u]))[0]
        if len(rated)==0: return 3.0
        sims = self.sim_fused[i, rated]
        top = xp.argsort(sims)[-self.K:]
        ti, ts = rated[top], sims[top]
        pm = ts > 0
        if not xp.any(pm): return float(xp.nanmean(self.R[u,rated]))
        ti, ts = ti[pm], ts[pm]
        return float(xp.sum(ts * self.R[u,ti]) / xp.sum(xp.abs(ts)))
    def predict_all(self):
        R_pred = self.R.copy()
        for u in range(self.n_users):
            for i in range(self.n_recipes):
                if xp.isnan(R_pred[u,i]): R_pred[u,i] = self.predict(u,i)
        return R_pred


class SVDRecommender:
    def __init__(self, k=50, lr=0.01, reg=0.05, epochs=40, V=None):
        self.k, self.lr, self.reg, self.epochs, self.V = k, lr, reg, epochs, V
    def fit(self, R):
        self.R, self.n_users, self.n_recipes = R.copy(), R.shape[0], R.shape[1]
        self.mu = float(xp.nanmean(R))
        rng = xp.random.RandomState(42)
        self.b_u = xp.zeros(self.n_users, dtype=xp.float32)
        self.b_i = xp.zeros(self.n_recipes, dtype=xp.float32)

        if self.V is not None and self.V.shape[0] == self.n_recipes:
            min_dim = min(self.k, self.V.shape[1])
            qi_init = truncated_svd_gpu(self.V, min_dim)
            if qi_init.shape[1] < self.k:
                pad = xp.zeros((self.n_recipes, self.k - qi_init.shape[1]), dtype=xp.float32)
                qi_init = xp.concatenate([qi_init, pad], axis=1)
            self.q_i = (qi_init * 0.1).astype(xp.float32)
        else:
            self.q_i = rng.normal(0, 0.1, (self.n_recipes, self.k)).astype(xp.float32)
        self.p_u = rng.normal(0, 0.1, (self.n_users, self.k)).astype(xp.float32) * 0.1

        pairs = [(u, i, float(R[u,i]))
                 for u in range(self.n_users)
                 for i in range(self.n_recipes)
                 if not xp.isnan(R[u,i])]
        import random as _random
        lr = self.lr
        for ep in range(self.epochs):
            tl = 0.0
            _random.shuffle(pairs)
            for u, i, r_ui in pairs:
                pred = self.mu + self.b_u[u] + self.b_i[i] + xp.dot(self.p_u[u], self.q_i[i])
                err = r_ui - float(pred)
                tl += err * err
                self.b_u[u] += lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += lr * (err - self.reg * self.b_i[i])
                pu_old = self.p_u[u].copy()
                self.p_u[u] += lr * (err * self.q_i[i] - self.reg * self.p_u[u])
                self.q_i[i] += lr * (err * pu_old - self.reg * self.q_i[i])
            lr *= 0.95
            if (ep+1) % 10 == 0 or ep == 0:
                print(f"  [SVD] Epoch {ep+1:3d}/{self.epochs} RMSE(train)={xp.sqrt(tl/len(pairs)):.4f}")
    def predict(self, u, i):
        return float(self.mu + self.b_u[u] + self.b_i[i] + xp.dot(self.p_u[u], self.q_i[i]))
    def predict_all(self):
        R_pred = self.mu + self.b_u[:, None] + self.b_i[None, :] + xp.dot(self.p_u, self.q_i.T)
        R_pred = xp.clip(R_pred, 1, 5)
        for u in range(self.n_users):
            for i in range(self.n_recipes):
                if not xp.isnan(self.R[u,i]):
                    R_pred[u,i] = self.R[u,i]
        return R_pred


class TwoTowerRecommender:
    def __init__(self, reg=0.1, latent_dim=8):
        self.reg, self.latent_dim = reg, latent_dim
    def fit(self, R, U, V):
        self.R, self.U, self.V = R.copy(), U, V
        self.n_users, self.n_recipes = R.shape
        self.mu = float(xp.nanmean(R))
        du, dv = U.shape[1], V.shape[1]
        print(f"  [TwoTower] d_u={du}, d_v={dv}, latent_dim={self.latent_dim}")
        self.w_u = xp.random.randn(du).astype(xp.float32) * 0.01
        self.w_v = xp.random.randn(dv).astype(xp.float32) * 0.01
        k = min(self.latent_dim, du, dv)
        self.A = xp.random.randn(du, k).astype(xp.float32) * 0.01
        self.B = xp.random.randn(dv, k).astype(xp.float32) * 0.01
        lr = 0.001
        pairs = [(u, i, float(R[u,i]))
                 for u in range(self.n_users)
                 for i in range(self.n_recipes)
                 if not xp.isnan(R[u,i])]
        import random as _random
        for ep in range(40):
            tl = 0.0
            _random.shuffle(pairs)
            w_u, w_v = self.w_u, self.w_v
            A, B = self.A, self.B
            reg = self.reg
            for u, i, r_ui in pairs:
                up = U[u] @ A      # 每次用最新的 A 算
                vp = V[i] @ B      # 每次用最新的 B 算
                pred = self.mu + xp.dot(w_u, U[u]) + xp.dot(w_v, V[i]) + xp.dot(up, vp)
                err = r_ui - float(pred)
                tl += err * err
                w_u += lr * (err * U[u] - reg * w_u)
                w_v += lr * (err * V[i] - reg * w_v)
                A += lr * (err * xp.outer(U[u], vp) - reg * A)
                B += lr * (err * xp.outer(V[i], up) - reg * B)
            self.A, self.B, self.w_u, self.w_v = A, B, w_u, w_v
            lr *= 0.95
            n = len(pairs)
            if (ep+1) % 10 == 0 or ep == 0:
                print(f"  [TwoTower] Epoch {ep+1:3d}/40 RMSE(train)={xp.sqrt(tl/max(n,1)):.4f}")
    def predict(self, u, i, U, V):
        linear = self.mu + xp.dot(self.w_u, U[u]) + xp.dot(self.w_v, V[i])
        inter = xp.dot(U[u] @ self.A, V[i] @ self.B)
        return float(linear + inter)
    def predict_all(self, U, V):
        linear = self.mu + xp.dot(U, self.w_u).reshape(-1,1) + xp.dot(V, self.w_v).reshape(1,-1)
        inter = (U @ self.A) @ (V @ self.B).T
        R_pred = xp.clip(linear + inter, 1, 5)
        for u in range(self.n_users):
            for i in range(self.n_recipes):
                if not xp.isnan(self.R[u,i]):
                    R_pred[u,i] = self.R[u,i]
        return R_pred


class HybridFusion:
    def __init__(self, gamma_u=0.1, gamma_i=0.1):
        self.gamma_u, self.gamma_i = gamma_u, gamma_i
    def fit(self, R_train, R_ubcf, R_ibcf, R_svd, R_tt, U, V):
        self.R_train = R_train
        nu = xp.sum(~xp.isnan(R_train), axis=1)
        ni = xp.sum(~xp.isnan(R_train), axis=0)
        self.c_u = 1 - xp.exp(-self.gamma_u * nu)
        self.c_i = 1 - xp.exp(-self.gamma_i * ni)
        best_rmse, best_w = float('inf'), [0.25]*4
        print("  [融合] 网格搜索权重...")
        for w1 in xp.arange(0.1, 0.6, 0.1):
            w1 = float(w1)
            for w2 in xp.arange(0.1, 0.6, 0.1):
                w2 = float(w2)
                for w3 in xp.arange(0.1, 0.6, 0.1):
                    w3 = float(w3)
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < 0.05: continue
                    w = [w1, w2, w3, w4]
                    se = n = 0
                    for u in range(R_train.shape[0]):
                        cu = float(self.c_u[u])
                        for i in range(R_train.shape[1]):
                            if not xp.isnan(R_train[u,i]):
                                ci = float(self.c_i[i])
                                phi = [cu*(1-ci)+0.5, (1-cu)*ci+0.5, cu*ci+0.3, (1-cu)*(1-ci)+0.3]
                                wd = [w[k]*phi[k] for k in range(4)]
                                s = sum(wd)
                                wd = [v/s for v in wd]
                                pred = (wd[0]*float(R_ubcf[u,i]) + wd[1]*float(R_ibcf[u,i]) +
                                        wd[2]*float(R_svd[u,i]) + wd[3]*float(R_tt[u,i]))
                                se += (float(R_train[u,i]) - pred) ** 2
                                n += 1
                    rmse = xp.sqrt(se / n) if n else float('inf')
                    if rmse < best_rmse:
                        best_rmse, best_w = rmse, w
        self.base_weights = xp.array(best_w, dtype=xp.float32)
        print(f"  → 基础权重: UBCF={best_w[0]:.2f} IBCF={best_w[1]:.2f} SVD={best_w[2]:.2f} TwoTower={best_w[3]:.2f}")
    def predict_all(self, R_ubcf, R_ibcf, R_svd, R_tt):
        R_pred = xp.zeros_like(R_ubcf)
        for u in range(R_pred.shape[0]):
            cu = float(self.c_u[u])
            for i in range(R_pred.shape[1]):
                ci = float(self.c_i[i])
                phi = [cu*(1-ci)+0.5, (1-cu)*ci+0.5, cu*ci+0.3, (1-cu)*(1-ci)+0.3]
                wd = [float(self.base_weights[k]) * phi[k] for k in range(4)]
                s = sum(wd)
                wd = [v/s for v in wd]
                R_pred[u,i] = (wd[0]*R_ubcf[u,i] + wd[1]*R_ibcf[u,i] +
                               wd[2]*R_svd[u,i] + wd[3]*R_tt[u,i])
        return R_pred


# ============================================================
# 评价（GPU 版本）
# ============================================================

def evaluate(R_true, R_pred, R_train, K=20):
    test_mask = ~xp.isnan(R_true) & xp.isnan(R_train)
    tv = R_true[test_mask]
    pv = R_pred[test_mask]
    if len(tv) == 0:
        print("[评价] 测试集为空")
        return {}
    rmse = float(xp.sqrt(xp.mean((tv - pv) ** 2)))
    mae = float(xp.mean(xp.abs(tv - pv)))
    prec, rec, ndcg = [], [], []
    for u in range(R_true.shape[0]):
        ti = xp.where(~xp.isnan(R_true[u]) & xp.isnan(R_train[u]))[0]
        if len(ti) < 2: continue
        tr = xp.where(~xp.isnan(R_train[u]))[0]
        un = xp.setdiff1d(xp.arange(R_true.shape[1]), tr)
        if len(un) == 0: continue
        top = un[xp.argsort(R_pred[u, un])[-K:][::-1]]
        liked = ti[R_true[u, ti] >= 3.5]
        hits = len(set(top.tolist()) & set(liked.tolist()))
        prec.append(hits / K)
        if len(liked) > 0:
            rec.append(hits / len(liked))
        dcg = idcg = 0
        for kk, item in enumerate(top):
            rel = float(R_true[u, item]) if not xp.isnan(R_true[u, item]) else 0
            dcg += (2**rel - 1) / xp.log2(kk + 2)
        for kk, rel in enumerate(sorted([float(R_true[u, i]) for i in ti], reverse=True)[:K]):
            idcg += (2**rel - 1) / xp.log2(kk + 2)
        if idcg > 0:
            ndcg.append(float(dcg / idcg))
    rec_set = set()
    for u in range(R_true.shape[0]):
        tr = xp.where(~xp.isnan(R_train[u]))[0]
        un = xp.setdiff1d(xp.arange(R_true.shape[1]), tr)
        if len(un) > 0:
            for item in un[xp.argsort(R_pred[u, un])[-K:][::-1]]:
                rec_set.add(int(item))
    cov = len(rec_set) / R_true.shape[1]
    results = {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        f'Precision@{K}': round(float(xp.mean(xp.array(prec))), 4) if prec else 0,
        f'Recall@{K}': round(float(xp.mean(xp.array(rec))), 4) if rec else 0,
        f'NDCG@{K}': round(float(xp.mean(xp.array(ndcg))), 4) if ndcg else 0,
        'Coverage': round(cov, 4),
        'TestSamples': int(len(tv)),
    }
    print("\n" + "="*50)
    print("  评 价 结 果")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:15s}: {v}")
    print("="*50)
    return results


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    gpu_tag = f"[GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}]" if _HAS_CUDA else "[CPU Mode]"
    print(f"  CF混合推荐系统 v2 — GPU 加速版 {gpu_tag}")
    print("=" * 70)

    data = load_data_from_db()
    U, V, uid2idx, rid2idx = build_features(data)

    R = build_rating_matrix(data, uid2idx, rid2idx)
    R_train = R.copy()
    R_test = xp.full_like(R, xp.nan)
    rng = xp.random.RandomState(42)
    for u in range(R.shape[0]):
        rated = xp.where(~xp.isnan(R[u]))[0]
        if len(rated) >= 3:
            t_items = rng.choice(rated, size=max(1, int(len(rated)*0.2)), replace=False)
            for i in t_items:
                R_test[u, i] = R[u, i]
                R_train[u, i] = xp.nan
    print(f"  训练集: {int(xp.sum(~xp.isnan(R_train)))}, 测试集: {int(xp.sum(~xp.isnan(R_test)))}")

    print("\n>>> User-Based CF")
    ubcf = UserBasedCF(U, K=50, alpha=0.6, tau_min=5)
    ubcf.fit(R_train)
    R_ubcf = ubcf.predict_all()

    print("\n>>> Item-Based CF")
    ibcf = ItemBasedCF(V, K=30, beta=0.6, tau_min=5)
    ibcf.fit(R_train)
    R_ibcf = ibcf.predict_all()

    print("\n>>> SVD 矩阵分解")
    svd = SVDRecommender(k=30, lr=0.01, reg=0.05, epochs=40, V=V)
    svd.fit(R_train)
    R_svd = svd.predict_all()

    print("\n>>> Two-Tower 双塔模型")
    tt = TwoTowerRecommender(reg=0.1, latent_dim=32)
    tt.fit(R_train, U, V)
    R_tt = tt.predict_all(U, V)

    print("\n>>> 动态混合融合")
    hf = HybridFusion(gamma_u=0.1, gamma_i=0.1)
    hf.fit(R_train, R_ubcf, R_ibcf, R_svd, R_tt, U, V)
    R_final = hf.predict_all(R_ubcf, R_ibcf, R_svd, R_tt)
    R_final = xp.clip(R_final, 1, 5)

    print("\n>>> 离线评估")
    evaluate(R_test, R_final, R_train, K=20)

    # ─── 示例推荐（打印时需要将 GPU 数组转回 CPU） ───
    print("\n>>> Top-10 推荐示例")
    df_user_out = data['user']
    df_recipe_out = data['recipe']
    df_cuisine_out = data['cuisine']

    import random
    seed = 42
    rng_py = random.Random(seed)
    id_pool = df_user_out[df_user_out['id'] > 1001000]['id'].tolist()
    demo_users = rng_py.sample(id_pool, min(10, len(id_pool)))

    # 转回 CPU 以便用 pandas/numpy 打印
    if _HAS_CUDA:
        R_final_cpu = cp.asnumpy(R_final)
        R_train_cpu = cp.asnumpy(R_train)
    else:
        R_final_cpu = R_final
        R_train_cpu = R_train

    import numpy as np

    for du in demo_users:
        if du not in uid2idx: continue
        u_idx = uid2idx[du]
        tr = np.where(~np.isnan(R_train_cpu[u_idx]))[0]
        un = np.setdiff1d(np.arange(R_final_cpu.shape[1]), tr)
        if len(un) == 0: continue
        top10 = un[np.argsort(R_final_cpu[u_idx, un])[-10:][::-1]]
        urow = df_user_out[df_user_out['id'] == du].iloc[0]
        print(f"\n  👤 用户 {du} ({urow['gender']}, {urow['age_group']})")
        print(f"     历史交互: {len(tr)} 个菜谱")
        rated_high = tr[R_train_cpu[u_idx, tr] >= 4.0]
        rid_list = list(rid2idx.keys())
        if len(rated_high) > 0:
            names = [str(df_recipe_out[df_recipe_out['id'] == rid_list[ri]].iloc[0]['name'])
                     for ri in rated_high[:3]]
            print(f"     高分菜谱: {', '.join(names)}")
        print(f"     Top-10 推荐:")
        for rank, ri in enumerate(top10):
            rid = rid_list[ri]
            rr = df_recipe_out[df_recipe_out['id'] == rid]
            if len(rr) == 0: continue
            cid = rr.iloc[0]['cuisine']
            cname = str(df_cuisine_out[df_cuisine_out['id'] == cid].iloc[0]['name']) \
                if pd.notna(cid) and cid > 0 else '未知'
            print(f"       {rank+1:2d}. {str(rr.iloc[0]['name'])[:30]:30s} "
                  f"预测={R_final_cpu[u_idx, ri]:.1f}  [{cname}]")

    print("\n" + "=" * 70)
    print("  运行完毕！")
    print("=" * 70)


if __name__ == '__main__':
    main()
