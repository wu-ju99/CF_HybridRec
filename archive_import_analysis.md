# Archive 数据导入分析报告

## 一、Archive 文件夹数据概览

| 文件名 | 行数 | 大小估计 | 数据类型 | 用途 |
|--------|------|----------|----------|------|
| `RAW_recipes.csv` | ~231,638 | ~80MB | 原始数据 | 食谱核心数据 |
| `RAW_interactions.csv` | ~1,327,044 | ~110MB | 原始数据 | 用户-食谱交互记录 |
| `PP_users.csv` | ~25,078 | ~120MB | 处理后的向量 | 用户特征向量 |
| `PP_recipes.csv` | ~178,266 | ~180MB | 处理后的向量 | 食谱特征向量 |
| `interactions_train.csv` | ~698,903 | ~55MB | 训练集拆分 | 与RAW_interactions重复 |
| `interactions_test.csv` | ~12,457 | ~1MB | 测试集拆分 | 与RAW_interactions重复 |
| `interactions_validation.csv` | ~7,025 | ~0.5MB | 验证集拆分 | 与RAW_interactions重复 |
| `ingr_map.pkl` | - | ~1MB | Python pickle | 食材ID→名称映射字典 |

数据来源：food.com 的食谱和用户评价数据（英文）

---

## 二、各文件结构详情

### 1. RAW_recipes.csv — 核心食谱表
```
name, id, minutes, contributor_id, submitted, tags, nutrition, n_steps, steps, description, ingredients, n_ingredients
```
| 字段 | 示例 | 类型 |
|------|------|------|
| name | "arriba baked winter squash mexican style" | 英文食谱名 |
| id | 137739 | 唯一ID |
| minutes | 55 | 烹饪时长(分钟) |
| contributor_id | 47892 | 贡献者ID |
| submitted | 2005-09-16 | 提交日期 |
| tags | "['60-minutes-or-less', 'mexican', 'vegetarian']" | JSON列表，含时长/菜系/饮食类型/场合等标签 |
| nutrition | "[51.5, 0.0, 13.0, 0.0, 2.0, 0.0, 4.0]" | [卡路里, 脂肪, 糖, 钠, 蛋白质, 饱和脂肪, 碳水] |
| n_steps | 11 | 步骤数 |
| steps | "['step 1', 'step 2', ...]" | JSON列表，英文烹饪步骤 |
| description | "autumn is my favorite..." | 英文描述 |
| ingredients | "['winter squash', 'honey', 'butter']" | JSON列表，英文食材名 |
| n_ingredients | 7 | 食材数 |

### 2. RAW_interactions.csv — 用户交互记录
```
user_id, recipe_id, date, rating, review
```
| 字段 | 示例 | 类型 |
|------|------|------|
| user_id | 38094 | 用户ID |
| recipe_id | 40893 | 对应的食谱ID |
| date | 2003-02-17 | 交互日期 |
| rating | 4 | 评分(1-5) |
| review | "Great with a salad..." | 英文文字评价 |

### 3. PP_users.csv — 处理后的用户特征（不推荐直接导入）
```
u, techniques, items, n_items, ratings, n_ratings
```
- techniques: 58维烹饪技法向量（数值编码，需外部映射表才能还原）
- items: 用户评价过的食谱ID列表
- ratings: 对应评分列表
- **这些是机器学习预处理后的数值数据，没有映射表无法还原为有意义的信息**

### 4. PP_recipes.csv — 处理后的食谱特征（部分有用）
```
id, i, name_tokens, ingredient_tokens, steps_tokens, techniques, calorie_level, ingredient_ids
```
- ingredient_ids: 食材的数字ID列表，可通过 `ingr_map.pkl` 还原为食材名
- calorie_level: 卡路里等级(0/1/2)
- 其余字段均为tokenized向量，无法直接还原

### 5. ingr_map.pkl — 食材映射字典
- Python pickle格式
- 结构: `{食材名: 数字ID}` 或 `{数字ID: 食材名}`
- 是解码 PP_recipes 中 ingredient_ids 的关键

---

## 三、与现有数据库的匹配分析

### ✅ 适合导入的数据（按优先级排列）

#### 第一优先级：RAW_recipes.csv → 多表分发

| Archive字段 | 目标表 | 目标字段 | 需要处理 |
|-------------|--------|----------|----------|
| name | `recipe` | name | **英译中** |
| description | `recipe` | description | **英译中** |
| minutes | `recipe` | 新增 extratrdetail | 记录烹饪时长 |
| submitted | `recipe` | 新增字段 | 提交日期 |
| tags | `cuisine` | name | 解析菜系标签(如mexican→墨西哥菜)，**英译中** |
| tags | `foodtype` | name | 解析食物类型标签 |
| tags | `classification` | name | 解析饮食分类(如vegetarian→素食) |
| ingredients[] | `ingredient` | name | 提取去重，**英译中**，与现有ingredient做匹配 |
| ingredients[] | `recipeingredient` | recipe+ingredient | 建立食谱-食材关联 |
| steps[] | `cookstep` | description | 每一步作为一条记录，**英译中** |
| nutrition[] | `content` | value | 解析7个营养值，映射到composition |
| nutrition[] | `recipecomposite` | 整体营养 | 汇总食谱营养成分 |

#### 第二优先级：RAW_interactions.csv → 用户行为数据

| Archive字段 | 目标表 | 目标字段 | 需要处理 |
|-------------|--------|----------|----------|
| user_id | `user` | 新增用户 | 以archive的user_id创建新用户记录 |
| recipe_id | `useracutalrecipe` | user+recipe | 关联用户和其评价过的食谱 |
| rating | `comment` | 扩展属性 | 1-5评分 |
| review | `comment` | description | 英文评论文本，**可选翻译** |
| date | `useracutalrecipe` | 新增字段 | 记录评价日期 |

#### 第三优先级（辅助参考）

| 文件 | 用途 |
|------|------|
| `ingr_map.pkl` | 解码 PP_recipes 中 ingredient_ids → 食材英文名 → 再匹配你的 ingredient 表 |
| `PP_recipes.csv` | ingredient_ids + calorie_level 可用于补充食材关联和热量信息 |

---

### ❌ 不适合导入的数据

| 文件 | 原因 |
|------|------|
| `PP_users.csv` | techniques/items字段为数值向量编码，缺乏还原映射表，导入数据库后不可读、不可用 |
| `PP_recipes.csv` (name_tokens/ingredient_tokens/steps_tokens/techniques) | 同为tokenized数值向量，无法还原为文本。仅有 ingredient_ids 和 calorie_level 可提取 |
| `interactions_train.csv` | 与 RAW_interactions 完全重复（仅为训练集拆分），无需重复导入 |
| `interactions_test.csv` | 同上，测试集拆分 |
| `interactions_validation.csv` | 同上，验证集拆分 |

> **结论**：只需导入 `RAW_recipes.csv` 和 `RAW_interactions.csv` 两张表。`ingr_map.pkl` 作为辅助映射工具。其余文件可忽略。

---

## 四、统一语言与表头方案

### 4.1 食材名称英译中方案

archive 中有约 10,000+ 种不重复的英文食材名。处理策略：

```
优先级 A — 直接匹配
在现有 ingredient 表中按英文名搜索（如已有 englishalias 记录）→ 匹配上则直接关联

优先级 B — 翻译后匹配
将英文食材名翻译为中文 → 在现有 ingredient 表中按中文名搜索 → 匹配上则关联

优先级 C — 新增食材
翻译为中文后作为新记录插入 ingredient 表，同时：
  - 在 englishalias 表中记录英文原文
  - 设置 foodtype（从 tags 推断）
  - 设置默认的 nature、taste 等属性为 NULL（待后续人工补全）
```

### 4.2 食谱名称英译中方案

231,637 条食谱名称需要翻译。建议：
- 批量机器翻译 + 人工抽检
- 保留英文原名在 description 字段中作为参考

### 4.3 Tags 解析映射方案

RAW_recipes 的 tags 包含多种类型，需分类解析：

| Tag 类型 | Tag 示例 | 目标表 | 中文名称 |
|----------|----------|--------|----------|
| 时长 | 60-minutes-or-less | 新增 classification 分类 | 60分钟以内 |
| 菜系 | mexican, italian, chinese | `cuisine` | 墨西哥菜、意大利菜、中国菜 |
| 饮食类型 | vegetarian, vegan | `classification` | 素食、纯素食 |
| 餐型 | breakfast, main-dish, dessert | `mealtype` | 早餐、主菜、甜点 |
| 烹饪方式 | easy, oven, stove-top | `cookmethod` | 简单、烤箱、炉灶 |
| 场合 | holiday-event, christmas | 新增 classification | 节日、圣诞 |
| 季节 | fall, winter, summer | `season` | 秋、冬、夏 |
| 食材大类 | vegetables, pork, meat | `foodtype` | 蔬菜、猪肉、肉类 |

### 4.4 Nutrition 解析方案

RAW_recipes nutrition 格式为 `"[卡路里, 总脂肪(%DV), 糖(%DV), 钠(%DV), 蛋白质(%DV), 饱和脂肪(%DV), 碳水(%DV)]"`。

映射到你的 `composition` 表：
```
索引0 → 卡路里 → composition: 能量
索引1 → 总脂肪 → composition: 脂肪
索引2 → 糖     → composition: 糖
索引3 → 钠     → composition: 钠
索引4 → 蛋白质 → composition: 蛋白质
索引5 → 饱和脂肪 → composition: 饱和脂肪
索引6 → 碳水   → composition: 碳水化合物
```
数值存入 `content` 表和 `recipecomposite` 表。

### 4.5 Steps 解析方案

RAW_recipes steps 格式为 `"['step1 text', 'step2 text', ...]"`：
- 每条 step 对应 `cookstep` 表一条记录
- ordinal 从 1 开始递增
- description 字段存入步骤英文文本（或翻译为中文）
- recipe 字段关联对应的 recipe ID

### 4.6 表头/字段映射总结

```
RAW_recipes.name            → recipe.name (需英译中)
RAW_recipes.description     → recipe.description (需英译中)
RAW_recipes.ingredients[]   → ingredient.name (需英译中)
RAW_recipes.steps[]         → cookstep.description (需英译中)
RAW_recipes.tags            → cuisine/foodtype/classification/mealtype/season/cookmethod
RAW_recipes.nutrition       → content + recipecomposite
RAW_interactions.user_id    → user (新建)
RAW_interactions.recipe_id  → useracutalrecipe.recipe
RAW_interactions.rating     → 存入 comment 或 extrattrdetail
RAW_interactions.review     → comment.description
```

---

## 五、建议的导入流程

```
第一步：导入 ingr_map.pkl → 解析食材名称映射表

第二步：从 RAW_recipes 提取所有不重复食材 → 与现有 ingredient 表匹配 → 新建不存在的食材

第三步：从 RAW_recipes 提取所有不重复 tags → 分类导入 cuisine/foodtype/classification/mealtype 等字典表

第四步：逐批导入 recipe（含 name 中文翻译、description、关联 cuisine 等）

第五步：导入 cookstep（关联 recipe）

第六步：导入 recipeingredient（关联 recipe + ingredient）

第七步：导入 nutrition → content + recipecomposite

第八步：从 RAW_interactions 导入新 user

第九步：导入 useracutalrecipe（关联 user + recipe + rating + date）

第十步：导入 comment（关联 user + recipe + review）
```

---

## 六、总结

| 类别 | 文件 | 行数 | 建议 |
|------|------|------|------|
| ✅ 核心导入 | RAW_recipes.csv | 231,638 | 拆分为 recipe、ingredient、cookstep、recipeingredient、content 等表 |
| ✅ 核心导入 | RAW_interactions.csv | 1,327,044 | 导入 user、useracutalrecipe、comment 表 |
| 🔧 辅助映射 | ingr_map.pkl | - | 用于解码食材ID |
| ⚠️ 部分可用 | PP_recipes.csv | 178,266 | 仅 ingredient_ids 和 calorie_level 有用 |
| ❌ 不适合 | PP_users.csv | 25,078 | 数值向量，无映射表 |
| ❌ 不适合 | interactions_train/test/validation.csv | 718,384 | 与RAW_interactions重复 |

**最大挑战**：23万条食谱名 + 步骤 + 1万多种食材名需要从英文翻译为中文，建议使用翻译API批量处理 + 人工抽检验收。
