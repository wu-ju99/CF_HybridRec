# CF-HybridRec

CF-HybridRec 是一个基于协同过滤的混合菜谱推荐算法项目。它基于 dietrecommendation 增强版 SQLite 数据库（187 张表），实现了"User-CF + Item-CF + SVD 矩阵分解 + Two-Tower 双塔模型"四模块动态融合的混合推荐流程，完全使用数据库原生表头特征，不引入外部特征。

当前项目已经支持：

- 用户五维画像构建（人口统计 + 口味偏好 + 地域 + 职业 + 年龄）
- 菜谱七维特征工程（菜系 / GI·时长·成本数值 / 食材口味聚合 / 食性聚合 / 营养成分 / 食材类型 TF-IDF / 烹饪方式 MultiHot）
- User-Based CF（评分相似度 + 画像相似度自适应融合）
- Item-Based CF（共现相似度 + 内容相似度自适应融合）
- SVD 矩阵分解（SGD 优化，菜谱因子用内容特征初始化）
- Two-Tower 双塔模型（$R̂_{ui}=μ+w_u^T U_u+w_v^T V_i+(A^T U_u)^T(B^T V_i)$，用户塔 + 物品塔）
- 动态置信度自适应融合权重
- 离线评估体系（RMSE / MAE / Precision@K / Recall@K / NDCG@K / Coverage）
- 新用户数据一键生成（1000 个多样化用户 + 差异化交互行为）

完整算法设计见：[cf_algorithm_design.md](cf_algorithm_design.md)。

## 1. 数据边界

### 1.1 原始数据库

默认读取（旧数据库）：

```text
C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_original_schema_enhanced.sqlite
```

含 500 合成用户，交互稠密度约 0.72%，`user` 表有效特征仅 21 维（性别+年龄组+口味，职业/地域数据为空）。

### 1.2 增强数据库（通过脚本生成）

运行 `generate_new_users.py` 生成：

```text
C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_v2_enhanced.sqlite
```

| 指标 | 原始库 | 增强库 |
|------|--------|--------|
| 用户数 | 506 | **1506** |
| 用户特征维度 d_u | 21 | **128**（职业+地域已填充） |
| 交互稠密度 | 0.72% | **19.9%** |
| 总交互数 | 1.5 万 | **122 万** |
| 喜好标记 | 0.55 万 | **62 万** |
| 实际做菜 | 777 | **31.8 万** |

生成逻辑：8 种用户原型（嗜辣青年/养生老年/精致主妇/健身青年/甜食女生/嗜辣女生/传统中老/美食探索者）× 口味偏移变体，每种原型有独立的食材偏好、菜系偏好、交互行为模式。职业/出生地/工作地从数据库原生 `occupation`/`address` 表填充。

需要注意：

- 原始合成用户 ID 范围：1000001–1000500
- 新增用户 ID 范围：1001001–1002000
- Demo 使用行为→评分映射做菜=5 / 喜好=3+intensity×0.4 / 浏览=2.5 / 规避=1
- `ingredient2nature` 表无数据，菜谱食性特征在当前版本中不能发挥效果

## 2. 算法总体思路

完整算法设计、数学公式、超参数选择见 [cf_algorithm_design.md](cf_algorithm_design.md)。

推荐流程如下：

```text
SQLite 数据库 (187 tables)
  → 读取 user / usertaste / recipe / recipeingredient / ingredient /
         cookstep / recipecomposite / useracutalrecipe / userfondnessrecipe /
         useravoidrecipe / userbrowse 等 16 张表
  → 构建用户特征向量 U (d_u 维) 和菜谱特征向量 V (d_v 维)
  → 构建评分矩阵 R (m × n)
  → 训练/测试时间切分 (8:2)
  → 四模块并行计算:
      ├─ User-CF:     KNN 邻居加权预测
      ├─ Item-CF:     KNN 邻居加权预测
      ├─ SVD:         SGD 矩阵分解
      └─ Two-Tower:   双塔模型（用户塔⊗物品塔交互 + 线性偏置）
  → 动态置信度融合:
      w_k(c_u, c_i) — 交互密度高时提 SVD 权重，稀疏时提 Two-Tower 权重
  → 返回 Top-K 推荐列表
```

融合公式：

```
R̂_final = w₁·R̂_UBCF + w₂·R̂_IBCF + w₃·R̂_SVD + w₄·R̂_TwoTower

其中 wₖ 通过网格搜索 + 置信度自适应确定：
wₖ'(c_u, c_i) = wₖ_base · φₖ(c_u, c_i)
φ₁ = c_u·(1-c_i) + 0.5    (UserCF 在用户密集、菜谱稀疏时权重高)
φ₂ = (1-c_u)·c_i + 0.5    (ItemCF 在用户稀疏、菜谱密集时权重高)
φ₃ = c_u·c_i + 0.3         (SVD 在两者都密集时权重高)
φ₄ = (1-c_u)·(1-c_i) + 0.3 (Two-Tower 在两者都稀疏时兜底)
```

## 3. 主要使用的数据表与特征映射

### 3.1 用户画像特征 U（6 组，d_u = 3+7+N_occ+11+N_addr×2）

| 特征组 | 维度 | 来源表.字段 | 编码方式 |
|--------|------|-----------|----------|
| 性别 | 3 | `user.gender` | OneHot {男, 女, 未知} |
| 年龄组 | 7 | `user.birthday` → 推算 | OneHot {婴儿, 幼儿, 儿童, 少年, 青年, 老年, 未知} |
| 职业 | 7 | `user.occupation` → `occupation.id` | OneHot |
| 口味偏好 | 11 | `usertaste.taste` → `taste.name`, `usertaste.level` | 归一化 [0, 1] |
| 出生地 | ≤50 | `user.birthplace` → `address.id` | OneHot |
| 工作地 | ≤50 | `user.workplace` → `address.id` | OneHot |

### 3.2 菜谱特征向量 V（7 组，d_v = 14+3+11+10+5+1076+26）

| 特征组 | 维度 | 来源表.字段链路 | 编码方式 |
|--------|------|---------------|----------|
| 菜系 | 14 | `recipe.cuisine` → `cuisine.id` | OneHot |
| 数值特征 | 3 | `recipe.gi`, `recipe.timeconsumming`, `recipe.cost` | 归一化 |
| 口味聚合 | 11 | `recipeingredient` → `ingredient` → `ingredient2taste` → `taste` | 食材口味分布均值 |
| 食性聚合 | 10 | `recipeingredient` → `ingredient` → `ingredient2nature` → `nature` | 食材食性分布均值 |
| 营养成分 | 5 | `recipecomposite` → `composition` → `content` | 归一化 |
| 食材类型 | 1076 | `recipeingredient` → `ingredient.foodtype` → `foodtype` | TF-IDF |
| 烹饪方式 | 26 | `cookstep.cookmethod` → `cookmethod` | MultiHot |

### 3.3 交互信号表（构建评分矩阵 R）

| 来源表 | 信号强度 | 评分映射 |
|--------|---------|---------|
| `useracutalrecipe` | 实际做过（强正） | rating = 5.0 |
| `userfondnessrecipe` | 标记喜欢（正） | rating = 3.0 + intensity × 0.4 |
| `userbrowse` + `userbrowsedetail` | 浏览过（弱正） | rating = 2.5 |
| `useravoidrecipe` | 标记规避（负） | rating = 1.0 |

### 3.4 字典表

| 表名 | 用途 |
|------|------|
| `taste` | 口味字典（甘/辛/酸/苦/咸/涩/淡/微辛/微酸/微苦/微咸 共 11 种） |
| `cuisine` | 菜系字典（鲁/川/粤/苏/闽/浙/湘/徽 等 14 种） |
| `foodtype` | 食材类型字典（蔬菜/水果/肉类/水产 等 1076 种） |
| `nature` | 食性字典（寒/凉/平/温/热/微寒/微凉/微温/大热/有毒 共 10 种） |
| `cookmethod` | 烹饪方式字典（炒/煮/蒸/炖/炸/烤/煎/拌 等 26 种） |

## 4. 安装

```bash
# 依赖（纯 CPU，无需 GPU）
pip install numpy pandas scikit-learn --break-system-packages

# Python 标准库（无需安装）
# sqlite3, collections, datetime, warnings
```

验证环境：

```bash
python -c "import numpy; import pandas; import sklearn; print('numpy:', numpy.__version__); print('pandas:', pandas.__version__); print('sklearn:', sklearn.__version__)"
```

## 5. 快速开始

### 5.1 使用增强版 SQLite 数据库运行

```bash
python cf_demo_v2.py
```

输出包括：

- 数据加载统计（每张表的行数和使用的字段）
- 特征工程详情（U 和 V 的维度和来源）
- 四模块训练进度
- 离线评估指标（RMSE / MAE / Precision@20 / Recall@20 / NDCG@20 / Coverage）
- Top-10 推荐示例

修改 `cf_demo_v2.py` 第 28 行的 `DB_PATH` 可切换数据库：

```python
DB_PATH = r"C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_v2_enhanced.sqlite"
```

### 5.2 生成新用户数据

```bash
python generate_new_users.py
```

生成 1000 个多样化用户（8 种原型 × 口味变体），交互行为包含明显差异化模式（偏辣/偏淡/偏甜/偏荤/偏素等），并填充职业、出生地、工作地字段。

### 5.3 查看用户交互详情

```bash
python check_user_ids.py
```

查看指定用户的实际交互记录（做过/喜好/规避/浏览明细），用于验证数据生成质量或排查推荐问题。

## 6. 实验评估

### 6.1 单次评估

```bash
python cf_demo_v2.py
```

输出指标：

```text
RMSE            : 预测准确度（均方根误差）
MAE             : 预测准确度（平均绝对误差）
Precision@20    : Top-20 推荐中"喜欢"菜谱的比例
Recall@20       : 用户所有"喜欢"菜谱中被推荐的比例
NDCG@20         : 归一化折损累计增益（考虑排序位置）
Coverage        : 推荐覆盖的不同菜谱比例
```

### 6.2 原始数据库 vs 增强数据库评估对比

| 指标 | 原始库（500 用户，0.72%） | 增强库（1500 用户，19.9%） |
|------|--------------------------|---------------------------|
| SVD RMSE | 1.05 | **0.43** |
| 用户间推荐区分度 | 差（Top-10 趋同） | 待测试 |
| Coverage | 1.2% | 待测试 |

### 6.3 消融实验

编辑 `cf_demo_v2.py` 的 main() 函数，注释掉对应模块：

```python
# 仅 User-CF
R_final = R_ubcf

# 仅 SVD
R_final = R_svd

# 仅 Two-Tower
R_final = R_tt

# 混合（默认）
R_final = hf.predict_all(...)
```

### 6.4 调参建议

| 参数 | 默认值 | 搜索范围 | 说明 |
|------|--------|---------|------|
| K (UBCF 邻居) | 50 | {20, 30, 50, 80, 100} | 越大推荐越趋向热门 |
| K (IBCF 邻居) | 30 | {10, 20, 30, 50} | 同上 |
| k (SVD 因子) | 30 | {20, 50, 100} | 越大拟合越强但越慢 |
| SVD epochs | 40 | {20, 40, 60} | 观察 RMSE 收敛 |
| Two-Tower latent_dim | 32 | {8, 16, 32, 64} | 越大交互矩阵越丰富 |
| γ_u, γ_i | 0.1 | {0.05, 0.1, 0.2} | 置信度衰减速率 |

## 7. 项目结构

```text
D:\科研实习\代码\
  cf_demo_v2.py               # 主程序：四模块混合推荐
  generate_new_users.py        # 新用户数据生成脚本
  check_user_ids.py            # 用户交互详情查看工具
  explore_schema.py            # 数据库结构探索工具
  cf_algorithm_design.md       # 完整算法设计文档
  README.md                    # 本文件
```

## 8. 已知局限与后续方向

### 8.1 已知局限

- **ingredient2nature 表无数据**：菜谱食性特征 (nature_dim=10) 始终为均匀分布，当前不发挥作用
- **无真实 rating**：`comment` 表没有评分字段，使用行为→评分映射，精度有限
- **Two-Tower 收敛较慢**：在 122 万训练集上需要较多 epoch 才能稳定，已通过预计算 UA/VB 加速
- **合成数据仍非真实行为**：即使增强了多样性，合成数据的分布规律仍不能完全模拟真实用户
- **HybridFusion 网格搜索较慢**：三层 for 循环遍历 125 种权重组合，在大数据集上耗时

### 8.2 后续改进方向

- 引入 BPR (Bayesian Personalized Ranking) 替代伪评分矩阵，直接用隐式反馈训练
- 增加 MMR (Maximal Marginal Relevance) 多样性重排，缓解 popularity bias
- Two-Tower 升级为双塔 DNN 或加入 LLM 语义 embedding
- 引入时间衰减因子，近期交互权重更高
- 规则层过滤接入 `hcirecommendrecipe`、`seasonrecommendingrecipe`、`sportrecommendrecipe` 等表
- 使用 PyTorch 重写 SVD 和 Two-Tower 训练部分，支持 GPU 加速
