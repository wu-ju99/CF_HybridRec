# CF-HybridRec

CF-HybridRec 是一个基于协同过滤的混合菜谱推荐算法项目。它基于 dietrecommendation 增强版 SQLite 数据库（160 张原始表 + 合成用户/行为数据），实现了"User-CF + Item-CF + SVD 矩阵分解 + Content-Based 特征交互"四模块动态融合的混合推荐流程，完全使用数据库原生表头特征，不引入外部特征。

当前项目已经支持：

- 用户四维画像构建（人口统计 + 口味偏好 + 地域 + 职业）
- 菜谱七维特征工程（菜系 / GI·时长·成本数值 / 食材口味聚合 / 食性聚合 / 营养成分 / 食材类型 TF-IDF / 烹饪方式 MultiHot）
- User-Based CF（皮尔逊评分相似度 + 余弦画像相似度混合）
- Item-Based CF（调整余弦共现相似度 + 余弦内容相似度混合）
- SVD 矩阵分解（SGD 优化，菜谱因子用内容特征初始化）
- Content-Based 特征交互模型（$\hat{R}_{ui} = \mu + w_u^T U_u + w_v^T V_i + U_u^T M V_i$，M 低秩分解）
- 动态置信度自适应融合权重
- 离线评估体系（RMSE / MAE / Precision@K / Recall@K / NDCG@K / Coverage）
- 合成数据一键运行（无需真实数据库即可完整复现）

完整算法设计见：[cf_algorithm_design.md](cf_algorithm_design.md)。

数据库审计报告见：[database_audit_report.md](database_audit_report.md)。

## 1. 数据边界

实验默认读取：

```text
C:\Users\HHP\Desktop\数据库v1(1)\数据库v1\dietrecommendation_original_schema_enhanced.sqlite
```

该数据库包含：

- 原始 160 张 dietrecommendation 业务表（992,663 行原始数据，全部保留）
- 新增合成数据：500 用户、5,500 口味偏好、23,776 浏览行为、5,497 菜谱喜好、3,267 菜谱规避、777 做菜记录等
- 补充数据：2,449 新食材、9,628 菜谱食材外键补全、35,994 食材营养、24,540 菜谱营养估算、6,360 食材味型派生

需要注意：

- 合成用户 ID 范围：1000001–1000500，来自 `synthetic_users_v1`，只能用于模拟实验
- 菜谱营养成分多数为估算值，不是实验测定数据
- `comment` 表没有 rating 字段，Demo 使用行为→评分映射（做菜=5 / 喜好=3+intensity×0.4 / 浏览=2.5 / 规避=1）
- 数据库有疾病知识表，但当前算法未使用疾病维度
- 食材口味和食性来自 `ingredient2taste` / `ingredient2nature` 的派生数据

## 2. 算法总体思路

完整算法设计、数学公式、超参数选择见 [cf_algorithm_design.md](cf_algorithm_design.md)。

推荐流程如下：

```text
SQLite 数据库 (160 tables)
  → 读取 user / usertaste / recipe / recipeingredient / ingredient /
         cookstep / recipecomposite / useracutalrecipe / userfondnessrecipe /
         useravoidrecipe / userbrowse 等 16 张表
  → 构建用户特征向量 U (d_u 维) 和菜谱特征向量 V (d_v 维)
  → 构建评分矩阵 R (m × n)
  → 训练/测试时间切分 (8:2)
  → 四模块并行计算:
      ├─ User-CF:  KNN 邻居加权预测
      ├─ Item-CF:  KNN 邻居加权预测
      ├─ SVD:      SGD 矩阵分解
      └─ Content:  带交互矩阵 M 的线性模型
  → 动态置信度融合:
      w_k(c_u, c_i) — 交互密度高时提 SVD 权重，稀疏时提 Content 权重
  → 返回 Top-K 推荐列表
```

融合公式：

```
R̂_final = w₁ · R̂_UBCF + w₂ · R̂_IBCF + w₃ · R̂_SVD + w₄ · R̂_Content

其中 wₖ 通过网格搜索 + 置信度自适应确定：
wₖ'(c_u, c_i) = wₖ_base · φₖ(c_u, c_i)
φ₁ = c_u·(1-c_i) + 0.5    (UserCF 在用户密集、菜谱稀疏时权重高)
φ₂ = (1-c_u)·c_i + 0.5    (ItemCF 在用户稀疏、菜谱密集时权重高)
φ₃ = c_u·c_i + 0.3         (SVD 在两者都密集时权重高)
φ₄ = (1-c_u)·(1-c_i) + 0.3 (Content 在两者都稀疏时兜底)
```

## 3. 主要使用的数据表与特征映射

### 3.1 用户画像特征 U（6 组，d_u = 3+7+N_occ+11+N_addr×2）

| 特征组 | 维度 | 来源表.字段 | 编码方式 |
|--------|------|-----------|----------|
| 性别 | 3 | `user.gender` | OneHot {男, 女, 未知} |
| 年龄组 | 7 | `user.birthday` → 推算 | OneHot {婴儿, 幼儿, 儿童, 少年, 青年, 老年, 未知} |
| 职业 | ≤15 | `user.occupation` | OneHot |
| 口味偏好 | 11 | `usertaste.taste` → `taste.name`, `usertaste.level` | 归一化 [-1, 1] |
| 出生地 | ≤36 | `user.birthplace` → `address.id` | OneHot |
| 工作地 | ≤36 | `user.workplace` → `address.id` | OneHot |

### 3.2 菜谱特征向量 V（7 组，d_v = 14+3+11+5+5+N_ft+N_cm）

| 特征组 | 维度 | 来源表.字段链路 | 编码方式 |
|--------|------|---------------|----------|
| 菜系 | 14 | `recipe.cuisine` → `cuisine.id` | OneHot |
| 数值特征 | 3 | `recipe.gi`, `recipe.timeconsumming`, `recipe.cost` | 归一化 |
| 口味聚合 | 11 | `recipeingredient` → `ingredient` → `ingredient2taste` → `taste` | 食材口味分布均值 |
| 食性聚合 | 5 | `recipeingredient` → `ingredient` → `ingredient2nature` → `nature` | 食材食性分布均值 |
| 营养成分 | 5 | `recipecomposite` → `composition` → `content` | 归一化 |
| 食材类型 | N_ft | `recipeingredient` → `ingredient.foodtype` → `foodtype` | TF-IDF |
| 烹饪方式 | N_cm | `cookstep.cookmethod` → `cookmethod` | MultiHot |

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
| `taste` | 口味字典（甘/辛/酸/苦/咸/涩/淡 等 11 种） |
| `cuisine` | 菜系字典（鲁/川/粤/苏/闽/浙/湘/徽/家常菜 等 14 种） |
| `foodtype` | 食材类型字典（蔬菜/水果/肉类/水产 等） |
| `nature` | 食性字典（寒/凉/平/温/热） |
| `cookmethod` | 烹饪方式字典（炒/煮/蒸/炖/炸/烤/煎/拌 等） |

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

## 5. 快速开始（合成数据 / 真实数据）

### 5.1 使用合成数据运行

不需要数据库文件，代码自动生成 200 用户 × 500 菜谱的模拟数据：

```bash
python cf_demo.py
```

### 5.2 使用增强版 SQLite 数据库运行

需要 `dietrecommendation_original_schema_enhanced.sqlite` 在指定路径：

```bash
python cf_demo_v2.py
```

输出包括：

- 数据加载统计（每张表的行数和使用的字段）
- 特征工程详情（U/V 的维度和来源）
- 四模块训练进度
- 离线评估指标（RMSE / MAE / Precision@20 / Recall@20 / NDCG@20 / Coverage）
- 3 个合成用户的 Top-10 推荐示例

### 5.3 想用自己的测试用户验证

在 SQLite 数据库中直接插入你的测试用户和交互记录：

```sql
sqlite3 "path/to/dietrecommendation_original_schema_enhanced.sqlite"

-- 新建测试用户
INSERT INTO user (id, name, gender, birthday, occupation, birthplace, workplace, username, password, enabled)
VALUES (9999, '测试用户', '男', '1995-06-15', 9, 1, 1, 'test', '123', 1);

-- 设置口味偏好
INSERT INTO usertaste (id, name, user, taste, level)
VALUES (90001, 't1', 9999, 2, 5), (90002, 't2', 9999, 5, 4);

-- 添加交互记录
INSERT INTO useracutalrecipe (id, name, user, recipe) VALUES (80001, 'a1', 9999, 100);
INSERT INTO userfondnessrecipe (id, name, user, recipe, intensity) VALUES (70001, 'f1', 9999, 200, 4);
INSERT INTO useravoidrecipe (id, name, user, recipe, intensity) VALUES (60001, 'v1', 9999, 300, 5);

.exit
```

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

### 6.2 消融实验（手动切换验证各模块贡献）

编辑 `cf_demo_v2.py` 的 main() 函数，注释掉对应模块：

```python
# 仅 User-CF
R_final = R_ubcf

# 仅 Item-CF
R_final = R_ibcf

# 仅 SVD
R_final = R_svd

# 仅 Content
R_final = R_content

# 混合（默认）
R_final = hf.predict_all(...)
```

比较各组合的指标差异，定位瓶颈模块。

### 6.3 调参建议

| 参数 | 默认值 | 搜索范围 | 说明 |
|------|--------|---------|------|
| K (UBCF 邻居) | 50 | {20, 30, 50, 80, 100} | 越大推荐越趋向热门 |
| K (IBCF 邻居) | 30 | {10, 20, 30, 50} | 同上 |
| k (SVD 因子) | 30 | {20, 50, 100} | 越大拟合越强但越慢 |
| SVD epochs | 40 | {20, 40, 60, 100} | 观察 RMSE 收敛 |
| Content latent_dim | 8 | {8, 16, 32} | 越大交互矩阵越丰富 |
| γ_u, γ_i | 0.1 | {0.05, 0.1, 0.2} | 置信度衰减速率 |

## 7. 项目结构

```text
D:\科研实习\数据库\
  cf_demo.py                  # v1: 合成数据版（可独立运行，无需数据库）
  cf_demo_v2.py               # v2: 真实数据版（读取增强 SQLite）
  cf_algorithm_design.md      # 完整算法设计文档（数学公式 + 流程图）
  database_audit_report.md    # 160 张表审计报告
  archive_import_analysis.md  # Archive CSV 数据导入分析
  import_guide.md             # CSV → MySQL 导入步骤指南
  setup_staging.sql           # 临时表创建脚本
  transform_to_target.sql     # 数据转换脚本
  preprocess_csv.py           # CSV 预处理工具
  export_ingr_map.py          # ingr_map.pkl 导出工具
  explore_db.py               # 数据库结构探索工具
  README.md                   # 本文件
```

## 8. 已知局限与后续方向

### 8.1 已知局限

- **Popularity Bias**：浏览信号占交互 71%，Item-CF 和 SVD 天然偏向热门菜谱，导致 Top-N 推荐在不同用户间趋同
- **无真实 rating**：`comment` 表没有评分字段，当前使用行为→评分映射，精度有限
- **Content-Based 偏弱**：latent_dim=8 的交互矩阵参数量较少，对长尾菜谱的个性化能力不足
- **合成数据限制**：用户是合成的，无法反映真实用户行为的分布
- **冷启动**：新用户无交互时退化为 Content 模块，但 Content 模块训练样本少（做菜仅 777 条）

### 8.2 后续改进方向

- 引入 BPR (Bayesian Personalized Ranking) 替代伪评分矩阵，直接用隐式反馈（成对排序）训练
- 增加 MMR (Maximal Marginal Relevance) 多样性重排，缓解 popularity bias
- Content 模块升级为双塔 DNN 或加入 LLM 语义 embedding（菜谱名 + 描述文本向量）
- 引入时间衰减因子，近期交互权重更高
- 规则层过滤接入 `hcirecommendrecipe`、`seasonrecommendingrecipe`、`sportrecommendrecipe` 等表
- 接入真实 rating 字段（在 `comment` 表新增 `rating` 列）
