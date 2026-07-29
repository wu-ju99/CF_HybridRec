"""
基于协同过滤的混合菜谱推荐算法 Demo v2
==========================================
数据来源：数据库v1(1) 增强版 SQLite（原始160表 + 合成用户/行为数据）
完全基于真实数据库表头和字段。

用法：python cf_demo_v2.py
依赖：pip install numpy pandas scikit-learn --break-system-packages

输出：
  1. 数据加载统计（每张表的行数、使用的列）
  2. 特征工程详情（每个特征的来源表和字段）
  3. 模型训练进度
  4. 评估指标（RMSE, MAE, Precision@20, Recall@20, NDCG@20, Coverage）
  5. 用户推荐示例
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DB_PATH = r"C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_original_schema_enhanced.sqlite"

# ============================================================
# 第一部分：从 SQLite 加载真实数据
# ============================================================

def load_data_from_db():
    """从增强版 SQLite 数据库读取所有算法需要的表"""
    conn = sqlite3.connect(DB_PATH)
    print("=" * 70)
    print("  从 SQLite 数据库加载数据")
    print("=" * 70)

    # ── 1. user 表 ──
    # 字段: id, name, username, password, firstname, lastname, birthday,
    #       career, email, phone, cellphone, qq, wechat, enabled,
    #       description, gender, authrole, birthplace, workplace, occupation
    # 使用: id, gender, birthday(→年龄组), occupation, birthplace, workplace
    df_user = pd.read_sql_query("""
        SELECT id, name, gender, birthday, occupation, birthplace, workplace
        FROM user
        WHERE id > 0
    """, conn)
    print(f"\n  [user]           {len(df_user):>6} 行  "
          f"使用: id, gender, birthday, occupation, birthplace, workplace")

    # ── 2. usertaste 表 ──
    # 字段: id, name, description, user, taste, level
    # 使用: user, taste(→taste.name), level(→level表或直接取数值)
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
    # 字段: id, name, foodtype, ingredientype, edible, water, ...
    # 使用: id, name, foodtype
    df_ingredient = pd.read_sql_query("""
        SELECT id, name, foodtype
        FROM ingredient
        WHERE id > 0
    """, conn)
    print(f"  [ingredient]     {len(df_ingredient):>6} 行  "
          f"使用: id, name, foodtype")

    # ── 8. ingredient2taste 表 ──
    df_ing2taste = pd.read_sql_query("""
        SELECT food AS ingredient_id, taste AS taste_id
        FROM ingredient2taste
    """, conn)
    print(f"  [ingredient2taste] {len(df_ing2taste):>5} 行  "
          f"使用: food→ingredient, taste")

    # ── 9. ingredient2nature 表 ──
    df_ing2nature = pd.read_sql_query("""
        SELECT food AS ingredient_id, nature AS nature_id
        FROM ingredient2nature
    """, conn)
    print(f"  [ingredient2nature] {len(df_ing2nature):>5} 行  "
          f"使用: food→ingredient, nature")

    # ── 10. recipe 表 ──
    # 字段: id, name, description, gi, timeconsumming, cost, cuisine, source, ...
    # 使用: id, name, cuisine, gi, timeconsumming, cost
    df_recipe = pd.read_sql_query("""
        SELECT id, name, cuisine, gi, timeconsumming, cost
        FROM recipe
        WHERE id > 0
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
    # 字段: id, name, description, ordinal, recipe, cookmethod, cooktool, intensity, duration
    # 使用: recipe, cookmethod
    df_cookstep = pd.read_sql_query("""
        SELECT recipe AS recipe_id, cookmethod AS cookmethod_id
        FROM cookstep
        WHERE recipe > 0
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
    # Pivot to get common nutrients
    nut_pivot = df_recipe_nut.pivot_table(
        index='recipe_id', columns='nutrient_name', values='quantity', aggfunc='first'
    ).reset_index()
    # Keep common nutrients
    common_nutrients = ['能量', '蛋白质', '脂肪', '碳水化合物', '膳食纤维']
    for col in common_nutrients:
        if col not in nut_pivot.columns:
            nut_pivot[col] = np.nan
    nut_pivot = nut_pivot[['recipe_id'] + common_nutrients].fillna(0)
    print(f"  [recipecomposite+composition+content] → {len(nut_pivot)} 菜谱营养 "
          f"维度: {common_nutrients}")

    # ── 15. 用户-菜谱交互数据 ──
    # 从多个表合并构建评分矩阵
    # 来源1: userfondnessrecipe (喜好=间接正向信号, intensity作为评分参考)
    df_fond = pd.read_sql_query("""
        SELECT user AS user_id, recipe AS recipe_id, intensity, 'fond' AS source
        FROM userfondnessrecipe WHERE user > 0 AND recipe > 0
    """, conn)
    print(f"  [userfondnessrecipe] {len(df_fond):>5} 行")

    # 来源2: useravoidrecipe (规避=负向信号)
    df_avoid = pd.read_sql_query("""
        SELECT user AS user_id, recipe AS recipe_id, intensity, 'avoid' AS source
        FROM useravoidrecipe WHERE user > 0 AND recipe > 0
    """, conn)
    print(f"  [useravoidrecipe]   {len(df_avoid):>5} 行")

    # 来源3: useracutalrecipe (实际做过=强正向信号)
    df_actual = pd.read_sql_query("""
        SELECT user AS user_id, recipe AS recipe_id, 5 AS intensity, 'actual' AS source
        FROM useracutalrecipe WHERE user > 0 AND recipe > 0
    """, conn)
    print(f"  [useracutalrecipe]  {len(df_actual):>5} 行")

    # 来源4: userbrowse (浏览=弱正向信号)
    df_browse = pd.read_sql_query("""
        SELECT ub.user AS user_id, ubd.entityid AS recipe_id, 2 AS intensity, 'browse' AS source
        FROM userbrowse ub
        JOIN userbrowsedetail ubd ON ubd.userbrowse = ub.id
        WHERE ub.user > 0 AND ubd.entityid > 0
    """, conn)
    print(f"  [userbrowse+browsedetail] {len(df_browse):>5} 行")

    conn.close()

    # ── 合并交互构建伪评分 ──
    # 做菜 = 5分, 喜好高intensity = 4-5, 浏览 = 2-3, 规避 = 1
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
    """从数据库真实字段构建用户向量 U 和菜谱向量 V"""
    print("\n" + "=" * 70)
    print("  特征工程：构建 U (用户) 和 V (菜谱) 向量")
    print("=" * 70)

    # ── 解析数据引用 ──
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

    # ── 年龄组推导 ──
    # 从 birthday 字段推算
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

    # ── 职业分类 ──
    occ_list = sorted(df_user['occupation'].dropna().unique())
    # 合并小类别
    if len(occ_list) > 15:
        # 只保留 top-12，其余归为"其他"
        occ_counts = df_user['occupation'].value_counts()
        occ_list = occ_counts.head(12).index.tolist()

    # ── 省份分类 ──
    # birthplace 和 workplace 指向 address 表的外键
    # 直接从 user 表读取 birthplace/workplace 外键值
    all_birth = sorted(df_user['birthplace'].dropna().unique())
    all_work = sorted(df_user['workplace'].dropna().unique())
    all_addrs = sorted(set(list(all_birth) + list(all_work)))
    n_addr = len(all_addrs) if len(all_addrs) <= 50 else 50  # 限制维度

    # ── 构建用户特征向量 U ──
    print("\n  ┌─ 用户特征向量 U 的构成 ─────────────────────")
    print(f"  │ 特征组               维度    来源表.字段")
    print(f"  ├─────────────────────────────────────────────")

    user_ids = df_user['id'].tolist()
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    # 1. gender (3维: 男/女/未知)
    gender_dim = 3
    print(f"  │ 1.性别(OneHot)      {gender_dim:>4}     user.gender")

    # 2. age_group (7维: 婴儿/幼儿/儿童/少年/青年/老年/未知)
    age_dim = len(age_order)
    print(f"  │ 2.年龄组(OneHot)    {age_dim:>4}     user.birthday→推算")

    # 3. occupation (N维, 限制≤15)
    occ_dim = len(occ_list)
    print(f"  │ 3.职业(OneHot)      {occ_dim:>4}     user.occupation")

    # 4. taste (N_taste维, 每个取usertaste.level归一化)
    taste_dim = n_tastes
    print(f"  │ 4.口味偏好(归一化)   {taste_dim:>4}     usertaste.taste+level")

    # 5. birthplace (N_addr维 OneHot)
    bplace_dim = n_addr
    print(f"  │ 5.出生地(OneHot)    {bplace_dim:>4}     user.birthplace→address")

    # 6. workplace (N_addr维 OneHot)
    wplace_dim = n_addr
    print(f"  │ 6.工作地(OneHot)    {wplace_dim:>4}     user.workplace→address")
    d_u = gender_dim + age_dim + occ_dim + taste_dim + bplace_dim + wplace_dim
    print(f"  ├─────────────────────────────────────────────")
    print(f"  │ 总计 d_u = {d_u}")
    print(f"  └─────────────────────────────────────────────")

    U = np.zeros((n_users, d_u), dtype=np.float32)

    # 预计算每个用户的 taste 向量
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
            user_taste_map[uid] = np.zeros(n_tastes)
        t_idx = taste_list.index(tname)
        user_taste_map[uid][t_idx] = level / 5.0

    for uid in user_ids:
        idx = user_id_to_idx[uid]
        row = df_user[df_user['id'] == uid].iloc[0]
        pos = 0

        # 性别
        g = str(row['gender']).strip()
        U[idx, pos + (0 if g == '男' else 1 if g == '女' else 2)] = 1
        pos += gender_dim

        # 年龄组
        ag = row['age_group']
        U[idx, pos + (age_order.index(ag) if ag in age_order else len(age_order)-1)] = 1
        pos += age_dim

        # 职业
        occ = row['occupation']
        if pd.notna(occ) and occ in occ_list:
            U[idx, pos + occ_list.index(occ)] = 1
        pos += occ_dim

        # 口味
        if uid in user_taste_map:
            U[idx, pos:pos+taste_dim] = user_taste_map[uid]
        pos += taste_dim

        # 出生地
        bp = row['birthplace']
        if pd.notna(bp) and bp in all_addrs:
            bp_idx = all_addrs.index(bp)
            if bp_idx < n_addr:
                U[idx, pos + bp_idx] = 1
        pos += bplace_dim

        # 工作地
        wp = row['workplace']
        if pd.notna(wp) and wp in all_addrs:
            wp_idx = all_addrs.index(wp)
            if wp_idx < n_addr:
                U[idx, pos + wp_idx] = 1

    print(f"  → U shape: {U.shape}")

    # ── 构建菜谱特征向量 V ──
    print(f"\n  ┌─ 菜谱特征向量 V 的构成 ─────────────────────")
    print(f"  │ 特征组               维度    来源表.字段")
    print(f"  ├─────────────────────────────────────────────")

    recipe_ids = df_recipe['id'].tolist()
    recipe_id_to_idx = {rid: i for i, rid in enumerate(recipe_ids)}
    n_recipes = len(recipe_ids)

    # 1. cuisine (N_cuisine维 OneHot)
    cuisine_dim = n_cuisines
    print(f"  │ 1.菜系(OneHot)      {cuisine_dim:>4}     recipe.cuisine→cuisine")

    # 2. gi, time, cost (3维归一化数值)
    print(f"  │ 2.GI/时长/成本       3     recipe.gi, timeconsumming, cost")

    # 3. taste (N_taste维, 由食材聚合)
    taste_agg_dim = n_tastes
    print(f"  │ 3.口味特征(聚合)     {taste_agg_dim:>4}     recipeingredient→ingredient→ingredient2taste→taste")

    # 4. nature (N_nature维, 食材性质聚合)
    nature_dim = n_natures
    print(f"  │ 4.食性特征(聚合)     {nature_dim:>4}     recipeingredient→ingredient→ingredient2nature→nature")

    # 5. nutrition (5维归一化)
    nut_dim = 5
    print(f"  │ 5.营养成分(归一化)    {nut_dim:>4}     recipecomposite→composition→content")

    # 6. foodtype (N_foodtype维 TF-IDF)
    ft_dim = n_foodtypes
    print(f"  │ 6.食材类型(TF-IDF)   {ft_dim:>4}     recipeingredient→ingredient.foodtype→foodtype")

    # 7. cookmethod (N_cookmethod维 MultiHot)
    cm_dim = n_cookmethods
    print(f"  │ 7.烹饪方式(MultiHot) {cm_dim:>4}     cookstep.cookmethod→cookmethod")
    d_v = cuisine_dim + 3 + taste_agg_dim + nature_dim + nut_dim + ft_dim + cm_dim
    print(f"  ├─────────────────────────────────────────────")
    print(f"  │ 总计 d_v = {d_v}")
    print(f"  └─────────────────────────────────────────────")

    V = np.zeros((n_recipes, d_v), dtype=np.float32)

    # 预计算食材级映射
    ing_foodtype = dict(zip(df_ing['id'], df_ing['foodtype']))
    # ingredient→tastes (多对多)
    ing_to_tastes = defaultdict(set)
    for _, row in df_ing2t.iterrows():
        ing_to_tastes[int(row['ingredient_id'])].add(int(row['taste_id']))
    # ingredient→natures
    ing_to_natures = defaultdict(set)
    for _, row in df_ing2n.iterrows():
        ing_to_natures[int(row['ingredient_id'])].add(int(row['nature_id']))

    # 食材类型文档频率 (TF-IDF)
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

    # 食谱→cookmethods
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

        # 1. 菜系 OneHot
        cid = r_row['cuisine']
        if pd.notna(cid) and 0 < cid <= n_cuisines:
            V[idx, pos + int(cid) - 1] = 1
        pos += cuisine_dim

        # 2. GI/时长/成本 归一化
        gi_val = r_row['gi']
        gi = float(gi_val) if pd.notna(gi_val) else 50.0
        V[idx, pos] = np.clip(gi / 100.0, 0, 1)

        t_val = r_row['timeconsumming']
        t = float(t_val) if pd.notna(t_val) else 30.0
        V[idx, pos+1] = np.log1p(t) / np.log1p(120)

        cost_val = r_row['cost']
        cost = float(cost_val) if pd.notna(cost_val) else 20.0
        V[idx, pos+2] = np.clip(cost / 80.0, 0, 1)
        pos += 3

        # 3. 口味聚合
        ings = df_recipe_ing[df_recipe_ing['recipe_id'] == rid]['ingredient_id'].values
        taste_vec = np.zeros(n_tastes)
        taste_count = 0
        for ing in ings:
            for t_id in ing_to_tastes.get(int(ing), set()):
                if 0 < t_id <= n_tastes:
                    taste_vec[int(t_id)-1] += 1
                    taste_count += 1
        if taste_count > 0:
            taste_vec /= taste_count
        else:
            taste_vec = np.ones(n_tastes) * 0.3  # 无数据默认中性
        V[idx, pos:pos+n_tastes] = taste_vec
        pos += taste_agg_dim

        # 4. 食性聚合
        nature_vec = np.zeros(n_natures)
        for ing in ings:
            for n_id in ing_to_natures.get(int(ing), set()):
                if 0 < n_id <= n_natures:
                    nature_vec[int(n_id)-1] += 1
        if nature_vec.sum() > 0:
            nature_vec /= nature_vec.sum()
        V[idx, pos:pos+n_natures] = nature_vec
        pos += nature_dim

        # 5. 营养成分归一化
        nut_row = nut_pivot[nut_pivot['recipe_id'] == rid]
        common = ['能量', '蛋白质', '脂肪', '碳水化合物', '膳食纤维']
        max_vals = {'能量': 800, '蛋白质': 50, '脂肪': 60, '碳水化合物': 100, '膳食纤维': 15}
        if len(nut_row) > 0:
            for ci, cn in enumerate(common):
                raw_val = nut_row.iloc[0].get(cn, 0)
                val = float(raw_val) if pd.notna(raw_val) else 0.0
                V[idx, pos+ci] = np.clip(val / max_vals.get(cn, 100), 0, 1)
        pos += nut_dim

        # 6. 食材类型 TF-IDF
        ft_vec = np.zeros(n_foodtypes)
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
                idf = np.log(n_recipes / (1 + ft_global.get(ft_idx+1, 1)))
                ft_vec[ft_idx] = tf * idf
        else:
            ft_vec = np.ones(n_foodtypes) / n_foodtypes
        V[idx, pos:pos+n_foodtypes] = ft_vec
        pos += ft_dim

        # 7. 烹饪方式 MultiHot
        for cm in recipe_cm.get(rid, set()):
            if 0 < cm <= n_cookmethods:
                V[idx, pos + int(cm) - 1] = 1

    print(f"  → V shape: {V.shape}")

    # 安全兜底：所有 NaN 替换为 0
    U = np.nan_to_num(U, nan=0.0)
    V = np.nan_to_num(V, nan=0.0)

    nan_u = np.sum(np.isnan(U))
    nan_v = np.sum(np.isnan(V))
    if nan_u > 0 or nan_v > 0:
        print(f"  ⚠ 仍有NaN: U={nan_u}, V={nan_v}")

    return U, V, user_id_to_idx, recipe_id_to_idx


# ============================================================
# 第三部分：构建评分矩阵
# ============================================================

def build_rating_matrix(data, user_id_to_idx, recipe_id_to_idx):
    """从真实交互数据构建评分矩阵"""
    interactions = data['interactions']
    n_users = len(user_id_to_idx)
    n_recipes = len(recipe_id_to_idx)

    R = np.full((n_users, n_recipes), np.nan, dtype=np.float32)

    for _, row in interactions.iterrows():
        u = int(row['user_id'])
        r = int(row['recipe_id'])
        if u in user_id_to_idx and r in recipe_id_to_idx:
            R[user_id_to_idx[u], recipe_id_to_idx[r]] = float(row['rating'])

    rated = np.sum(~np.isnan(R))
    print(f"\n[评分矩阵] R: {R.shape}, 评分数={int(rated)}, 稀疏度={rated/(n_users*n_recipes):.6f}")
    return R


# ============================================================
# 第四部分~第十部分：CF模型（复用 v1 的类定义）
# ============================================================

# --- User-CF (不变) ---
class UserBasedCF:
    def __init__(self, U, K=50, alpha=0.6, tau_min=5):
        self.U, self.K, self.alpha, self.tau_min = U, K, alpha, tau_min
    def fit(self, R):
        self.R, self.n_users, self.n_recipes = R.copy(), R.shape[0], R.shape[1]
        self.r_mean = np.nanmean(R, axis=1)
        Rc = np.nan_to_num(R - self.r_mean[:, np.newaxis], 0)
        self.sim_matrix = cosine_similarity(Rc)
        self.profile_sim = cosine_similarity(self.U)
        n = self.n_users
        self.sim_fused = np.zeros((n, n))
        for u in range(n):
            for v in range(u+1, n):
                cu = np.sum(~np.isnan(R[u]) & ~np.isnan(R[v]))
                ae = self.alpha * min(1.0, cu/self.tau_min)
                sv = ae*self.sim_matrix[u,v] + (1-ae)*self.profile_sim[u,v]
                self.sim_fused[u,v] = self.sim_fused[v,u] = sv
            self.sim_fused[u,u] = 1.0
        print(f"  [User-CF] 相似度矩阵: {self.sim_fused.shape}")
    def predict(self, u, i):
        if not np.isnan(self.R[u,i]): return self.R[u,i]
        rated = np.where(~np.isnan(self.R[:,i]))[0]
        if len(rated)==0: return self.r_mean[u]
        sims = self.sim_fused[u, rated]
        top = np.argsort(sims)[-self.K:]
        tu, ts = rated[top], sims[top]
        pm = ts > 0
        if not pm.any(): return self.r_mean[u]
        tu, ts = tu[pm], ts[pm]
        num = np.sum(ts*(self.R[tu,i]-self.r_mean[tu]))
        den = np.sum(np.abs(ts))
        return self.r_mean[u] + num/den if den else self.r_mean[u]
    def predict_all(self):
        R_pred = self.R.copy()
        mc = int(np.sum(np.isnan(R_pred)))
        done = 0
        for u in range(self.n_users):
            for i in range(self.n_recipes):
                if np.isnan(R_pred[u,i]):
                    R_pred[u,i] = self.predict(u,i)
                    done += 1
        return R_pred

# --- Item-CF (不变) ---
class ItemBasedCF:
    def __init__(self, V, K=30, beta=0.6, tau_min=5):
        self.V, self.K, self.beta, self.tau_min = V, K, beta, tau_min
    def fit(self, R):
        self.R, self.n_users, self.n_recipes = R.copy(), R.shape[0], R.shape[1]
        rm = np.nanmean(R, axis=1)
        Rc = np.nan_to_num(R - rm[:, np.newaxis], 0)
        self.co_sim = cosine_similarity(Rc.T)
        self.content_sim = cosine_similarity(self.V)
        n = self.n_recipes
        self.item_uc = np.sum(~np.isnan(R), axis=0)
        self.sim_fused = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                cu = np.sum(~np.isnan(R[:,i]) & ~np.isnan(R[:,j]))
                be = self.beta * min(1.0, cu/self.tau_min)
                sv = be*self.co_sim[i,j] + (1-be)*self.content_sim[i,j]
                self.sim_fused[i,j] = self.sim_fused[j,i] = sv
            self.sim_fused[i,i] = 1.0
        print(f"  [Item-CF] 相似度矩阵: {self.sim_fused.shape}")
    def predict(self, u, i):
        if not np.isnan(self.R[u,i]): return self.R[u,i]
        rated = np.where(~np.isnan(self.R[u]))[0]
        if len(rated)==0: return 3.0
        sims = self.sim_fused[i, rated]
        top = np.argsort(sims)[-self.K:]
        ti, ts = rated[top], sims[top]
        pm = ts > 0
        if not pm.any(): return np.nanmean(self.R[u,rated])
        ti, ts = ti[pm], ts[pm]
        return np.sum(ts*self.R[u,ti])/np.sum(np.abs(ts))
    def predict_all(self):
        R_pred = self.R.copy()
        for u in range(self.n_users):
            for i in range(self.n_recipes):
                if np.isnan(R_pred[u,i]): R_pred[u,i] = self.predict(u,i)
        return R_pred

# --- SVD (不变) ---
class SVDRecommender:
    def __init__(self, k=50, lr=0.01, reg=0.05, epochs=40, V=None):
        self.k, self.lr, self.reg, self.epochs, self.V = k, lr, reg, epochs, V
    def fit(self, R):
        self.R, self.n_users, self.n_recipes = R.copy(), R.shape[0], R.shape[1]
        self.mu = np.nanmean(R)
        rng = np.random.default_rng(42)
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_recipes)
        if self.V is not None and self.V.shape[0]==self.n_recipes:
            svd = TruncatedSVD(n_components=min(self.k, self.V.shape[1]), random_state=42)
            qi = svd.fit_transform(self.V)
            if qi.shape[1] < self.k:
                qi = np.hstack([qi, np.zeros((self.n_recipes, self.k-qi.shape[1]))])
            self.q_i = (qi * 0.1).astype(np.float32)
        else:
            self.q_i = rng.normal(0, 0.1, (self.n_recipes, self.k)).astype(np.float32)
        self.p_u = rng.normal(0, 0.1, (self.n_users, self.k)).astype(np.float32)*0.1
        pairs = [(u,i,R[u,i]) for u in range(self.n_users) for i in range(self.n_recipes) if not np.isnan(R[u,i])]
        rng.shuffle(pairs)
        lr = self.lr
        for ep in range(self.epochs):
            tl = 0; rng.shuffle(pairs)
            for u,i,r_ui in pairs:
                pred = self.mu+self.b_u[u]+self.b_i[i]+np.dot(self.p_u[u],self.q_i[i])
                err = r_ui-pred; tl+=err**2
                self.b_u[u]+=lr*(err-self.reg*self.b_u[u])
                self.b_i[i]+=lr*(err-self.reg*self.b_i[i])
                pu_old = self.p_u[u].copy()
                self.p_u[u]+=lr*(err*self.q_i[i]-self.reg*self.p_u[u])
                self.q_i[i]+=lr*(err*pu_old-self.reg*self.q_i[i])
            lr *= 0.95
            if (ep+1)%10==0 or ep==0:
                print(f"  [SVD] Epoch {ep+1:3d}/{self.epochs} RMSE(train)={np.sqrt(tl/len(pairs)):.4f}")
    def predict(self, u, i):
        return self.mu+self.b_u[u]+self.b_i[i]+np.dot(self.p_u[u],self.q_i[i])
    def predict_all(self):
        R_pred = self.mu+self.b_u[:,None]+self.b_i[None,:]+np.dot(self.p_u,self.q_i.T)
        R_pred = np.clip(R_pred,1,5)
        for u in range(self.n_users):
            for i in range(self.n_recipes):
                if not np.isnan(self.R[u,i]): R_pred[u,i]=self.R[u,i]
        return R_pred

# --- Content-Based (带交互矩阵M) ---
class ContentBasedRecommender:
    """
    带交互矩阵的内容推荐。
    M = A @ B^T, A∈R^{du×k}, B∈R^{dv×k}
    R̂_ui = μ + w_u^T U_u + w_v^T V_i + (A^T U_u)^T (B^T V_i)
    """
    def __init__(self, reg=0.1, latent_dim=8):
        self.reg, self.latent_dim = reg, latent_dim
    def fit(self, R, U, V):
        self.R, self.U, self.V = R.copy(), U, V
        self.n_users, self.n_recipes = R.shape
        self.mu = np.nanmean(R)
        du, dv = U.shape[1], V.shape[1]
        print(f"  [Content] d_u={du}, d_v={dv}, latent_dim={self.latent_dim}")
        self.w_u = np.random.randn(du)*0.01
        self.w_v = np.random.randn(dv)*0.01
        k = min(self.latent_dim, du, dv)
        # A: (du, k) 映射用户特征, B: (dv, k) 映射菜谱特征（全局共享）
        self.A = np.random.randn(du, k)*0.01
        self.B = np.random.randn(dv, k)*0.01
        lr = 0.005
        for ep in range(40):
            tl=n=0
            for u in range(self.n_users):
                up = U[u] @ self.A      # (k,)
                for i in range(self.n_recipes):
                    if not np.isnan(R[u,i]):
                        vp = V[i] @ self.B  # (k,)
                        pred = self.mu + np.dot(self.w_u,U[u]) + np.dot(self.w_v,V[i]) + np.dot(up,vp)
                        err = R[u,i]-pred; tl+=err**2; n+=1
                        # 线性项
                        self.w_u += lr*(err*U[u] - self.reg*self.w_u)
                        self.w_v += lr*(err*V[i] - self.reg*self.w_v)
                        # 交互矩阵梯度: ∂L/∂A = -err * U_u ⊗ vp, ∂L/∂B = -err * V_i ⊗ up
                        self.A += lr*(err * np.outer(U[u], vp) - self.reg*self.A)
                        self.B += lr*(err * np.outer(V[i], up) - self.reg*self.B)
            lr*=0.95
            if (ep+1)%10==0 or ep==0:
                print(f"  [Content] Epoch {ep+1:3d}/40 RMSE(train)={np.sqrt(tl/max(n,1)):.4f}")
    def predict(self, u, i, U, V):
        linear = self.mu + np.dot(self.w_u,U[u]) + np.dot(self.w_v,V[i])
        inter  = np.dot(U[u] @ self.A, V[i] @ self.B)
        return linear + inter
    def predict_all(self, U, V):
        linear = self.mu + np.dot(U,self.w_u).reshape(-1,1) + np.dot(V,self.w_v).reshape(1,-1)
        inter  = (U @ self.A) @ (V @ self.B).T
        R_pred = np.clip(linear+inter, 1, 5)
        for u in range(self.n_users):
            for i in range(self.n_recipes):
                if not np.isnan(self.R[u,i]): R_pred[u,i]=self.R[u,i]
        return R_pred

# --- 混合融合 (不变) ---
class HybridFusion:
    def __init__(self, gamma_u=0.1, gamma_i=0.1):
        self.gamma_u, self.gamma_i = gamma_u, gamma_i
    def fit(self, R_train, R_ubcf, R_ibcf, R_svd, R_content, U, V):
        self.R_train = R_train
        nu = np.sum(~np.isnan(R_train), axis=1)
        ni = np.sum(~np.isnan(R_train), axis=0)
        self.c_u = 1-np.exp(-self.gamma_u*nu)
        self.c_i = 1-np.exp(-self.gamma_i*ni)
        best_rmse, best_w = float('inf'), [0.25]*4
        print("  [融合] 网格搜索权重...")
        for w1 in np.arange(0.1, 0.6, 0.1):
            for w2 in np.arange(0.1, 0.6, 0.1):
                for w3 in np.arange(0.1, 0.6, 0.1):
                    w4 = 1.0-w1-w2-w3
                    if w4<0.05: continue
                    w=[w1,w2,w3,w4]
                    se=n=0
                    for u in range(R_train.shape[0]):
                        cu=self.c_u[u]
                        for i in range(R_train.shape[1]):
                            if not np.isnan(R_train[u,i]):
                                ci=self.c_i[i]
                                phi=np.array([cu*(1-ci)+0.5,(1-cu)*ci+0.5,cu*ci+0.3,(1-cu)*(1-ci)+0.3])
                                wd=w*phi; wd/=wd.sum()
                                pred=wd[0]*R_ubcf[u,i]+wd[1]*R_ibcf[u,i]+wd[2]*R_svd[u,i]+wd[3]*R_content[u,i]
                                se+=(R_train[u,i]-pred)**2; n+=1
                    rmse=np.sqrt(se/n) if n else float('inf')
                    if rmse<best_rmse: best_rmse,best_w=rmse,w
        self.base_weights=np.array(best_w)
        print(f"  → 基础权重: UBCF={best_w[0]:.2f} IBCF={best_w[1]:.2f} SVD={best_w[2]:.2f} Content={best_w[3]:.2f}")
    def predict_all(self, R_ubcf,R_ibcf,R_svd,R_content):
        R_pred=np.zeros_like(R_ubcf)
        for u in range(R_pred.shape[0]):
            cu=self.c_u[u]
            for i in range(R_pred.shape[1]):
                ci=self.c_i[i]
                phi=np.array([cu*(1-ci)+0.5,(1-cu)*ci+0.5,cu*ci+0.3,(1-cu)*(1-ci)+0.3])
                wd=self.base_weights*phi; wd/=wd.sum()
                R_pred[u,i]=wd[0]*R_ubcf[u,i]+wd[1]*R_ibcf[u,i]+wd[2]*R_svd[u,i]+wd[3]*R_content[u,i]
        return R_pred

# --- 评价 (不变) ---
def evaluate(R_true, R_pred, R_train, K=20):
    test_mask = ~np.isnan(R_true) & np.isnan(R_train)
    tv = R_true[test_mask]; pv = R_pred[test_mask]
    if len(tv)==0:
        print("[评价] 测试集为空"); return {}
    rmse = np.sqrt(np.mean((tv-pv)**2))
    mae = np.mean(np.abs(tv-pv))
    prec, rec, ndcg = [],[],[]
    for u in range(R_true.shape[0]):
        ti = np.where(~np.isnan(R_true[u]) & np.isnan(R_train[u]))[0]
        if len(ti)<2: continue
        tr = np.where(~np.isnan(R_train[u]))[0]
        un = np.setdiff1d(np.arange(R_true.shape[1]), tr)
        if len(un)==0: continue
        top = un[np.argsort(R_pred[u,un])[-K:][::-1]]
        liked = ti[R_true[u,ti]>=3.5]
        hits = len(set(top)&set(liked))
        prec.append(hits/K)
        if len(liked)>0: rec.append(hits/len(liked))
        dcg=idcg=0
        for kk,item in enumerate(top):
            rel=R_true[u,item] if not np.isnan(R_true[u,item]) else 0
            dcg+=(2**rel-1)/np.log2(kk+2)
        for kk,rel in enumerate(sorted([R_true[u,i] for i in ti], reverse=True)[:K]):
            idcg+=(2**rel-1)/np.log2(kk+2)
        if idcg>0: ndcg.append(dcg/idcg)
    rec_set = set()
    for u in range(R_true.shape[0]):
        tr=np.where(~np.isnan(R_train[u]))[0]
        un=np.setdiff1d(np.arange(R_true.shape[1]),tr)
        if len(un)>0:
            rec_set.update(un[np.argsort(R_pred[u,un])[-K:][::-1]])
    cov = len(rec_set)/R_true.shape[1]
    results={'RMSE':round(rmse,4),'MAE':round(mae,4),f'Precision@{K}':round(np.mean(prec),4) if prec else 0,
             f'Recall@{K}':round(np.mean(rec),4) if rec else 0,f'NDCG@{K}':round(np.mean(ndcg),4) if ndcg else 0,
             'Coverage':round(cov,4),'TestSamples':len(tv)}
    print("\n"+"="*50); print("  评 价 结 果"); print("="*50)
    for k,v in results.items(): print(f"  {k:15s}: {v}")
    print("="*50)
    return results


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("  CF混合推荐系统 v2 — 基于数据库v1(1)增强SQLite")
    print("=" * 70)

    # Step 1: 加载数据
    data = load_data_from_db()

    # Step 2: 特征工程
    U, V, uid2idx, rid2idx = build_features(data)

    # Step 3: 评分矩阵 + 训练/测试划分
    R = build_rating_matrix(data, uid2idx, rid2idx)
    R_train = R.copy()
    R_test = np.full_like(R, np.nan)
    rng = np.random.default_rng(42)
    for u in range(R.shape[0]):
        rated = np.where(~np.isnan(R[u]))[0]
        if len(rated)>=3:
            t_items = rng.choice(rated, size=max(1,int(len(rated)*0.2)), replace=False)
            for i in t_items: R_test[u,i]=R[u,i]; R_train[u,i]=np.nan
    print(f"  训练集: {int(np.sum(~np.isnan(R_train)))}, 测试集: {int(np.sum(~np.isnan(R_test)))}")

    # Step 4: User-CF
    print("\n>>> User-Based CF")
    ubcf = UserBasedCF(U, K=50, alpha=0.6, tau_min=5); ubcf.fit(R_train)
    R_ubcf = ubcf.predict_all()

    # Step 5: Item-CF
    print("\n>>> Item-Based CF")
    ibcf = ItemBasedCF(V, K=30, beta=0.6, tau_min=5); ibcf.fit(R_train)
    R_ibcf = ibcf.predict_all()

    # Step 6: SVD
    print("\n>>> SVD 矩阵分解")
    svd = SVDRecommender(k=30, lr=0.01, reg=0.05, epochs=40, V=V); svd.fit(R_train)
    R_svd = svd.predict_all()

    # Step 7: Content
    print("\n>>> Content-Based (含交互矩阵 M)")
    cb = ContentBasedRecommender(reg=0.1, latent_dim=8); cb.fit(R_train, U, V)
    R_content = cb.predict_all(U, V)

    # Step 8: 融合
    print("\n>>> 动态混合融合")
    hf = HybridFusion(gamma_u=0.1, gamma_i=0.1); hf.fit(R_train,R_ubcf,R_ibcf,R_svd,R_content,U,V)
    R_final = hf.predict_all(R_ubcf,R_ibcf,R_svd,R_content)
    R_final = np.clip(R_final, 1, 5)

    # Step 9: 评估
    print("\n>>> 离线评估")
    evaluate(R_test, R_final, R_train, K=20)

    # Step 10: 示例推荐
    print("\n>>> Top-10 推荐示例")
    df_user_out = data['user']
    df_recipe_out = data['recipe']
    df_cuisine_out = data['cuisine']
    demo_users = list(df_user_out[df_user_out['id'] > 1000000]['id'].head(3))
    for du in demo_users:
        if du not in uid2idx: continue
        u_idx = uid2idx[du]
        tr = np.where(~np.isnan(R_train[u_idx]))[0]
        un = np.setdiff1d(np.arange(R_final.shape[1]), tr)
        if len(un)==0: continue
        top10 = un[np.argsort(R_final[u_idx,un])[-10:][::-1]]
        urow = df_user_out[df_user_out['id']==du].iloc[0]
        print(f"\n  👤 用户 {du} ({urow['gender']}, {urow['age_group']})")
        print(f"     历史交互: {len(tr)} 个菜谱")
        rated_high = tr[R_train[u_idx,tr]>=4.0]
        rid_list = list(rid2idx.keys())
        if len(rated_high)>0:
            names = [str(df_recipe_out[df_recipe_out['id']==rid_list[ri]].iloc[0]['name']) for ri in rated_high[:3]]
            print(f"     高分菜谱: {', '.join(names)}")
        print(f"     Top-10 推荐:")
        for rank, ri in enumerate(top10):
            rid = rid_list[ri]
            rr = df_recipe_out[df_recipe_out['id']==rid]
            if len(rr)==0: continue
            cid = rr.iloc[0]['cuisine']
            cname = str(df_cuisine_out[df_cuisine_out['id']==cid].iloc[0]['name']) if pd.notna(cid) and cid>0 else '未知'
            print(f"       {rank+1:2d}. {str(rr.iloc[0]['name'])[:30]:30s} "
                  f"预测={R_final[u_idx,ri]:.1f}  [{cname}]")

    print("\n" + "=" * 70)
    print("  运行完毕！")
    print("=" * 70)

if __name__ == '__main__':
    main()
