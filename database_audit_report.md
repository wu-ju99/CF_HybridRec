# 数据库审计报告

对 `dietrecommendation` 数据库 160 张表的全面检查。

---

## 一、完全空表（没有任何数据，连 -2 占位符都没有）

这些表已建好结构，但未插入任何记录：

| 表名 | 用途 | 严重程度 |
|------|------|----------|
| `avoid2avoid` | 禁忌与禁忌的关联 | ⚠️ 中 |
| `avoid2ingredient` | 禁忌→食材映射 | ⚠️ 中 |
| `avoid2recipe` | 禁忌→食谱映射 | ⚠️ 中 |
| `classification2classification` | 分类层级关系 | ⚠️ 中 |
| `cycle2cycle` | 周期层级关系 | ⚠️ 中 |
| `ingredientincompatibility` | 食材相克关系 | 🔴 高 |
| `ingredientproduceaddress` | 食材产地信息 | ⚠️ 中 |

> 共 **7 张表**完全没有数据。如果系统业务功能依赖这些关联关系，当前将无法生效。

---

## 二、仅有占位符的表（只有 id=-2 的 NullEntity 记录）

这些表虽然有"数据"，但实际只有一个系统占位符，没有真实业务数据：

### 用户操作记录类（系统运行后会增长）
| 表名 | 说明 |
|------|------|
| `useractualmeal` | 用户实际用餐 |
| `useractualsport` | 用户实际运动 |
| `useracutalingredient` | 用户实际摄入食材 |
| `useracutalmeal` | 用户实际用餐（与上重复？） |
| `useracutalrecipe` | 用户实际使用的食谱 |
| `useravoidingredient` / `useravoidmeal` / `useravoidrecipe` | 用户个人规避偏好 |
| `userfondnessingredient` / `userfondnessmeal` / `userfondnessrecipe` | 用户个人喜好 |
| `userbrowse` / `userbrowsedetail` | 用户浏览记录 |
| `usereciperecommend` / `usereciperecommendetail` | 针对用户的食谱推荐 |
| `userecommendingredient` / `userecommendingredientdetail` | 针对用户的食材推荐 |
| `userecommendmeal` / `userecommendmealdetail` | 针对用户的餐推荐 |
| `userplansport` | 用户运动计划 |
| `usertaste` | 用户口味偏好 |

### 多因素推荐类（需运营配置）
| 表名 | 说明 |
|------|------|
| `multiavoidingredient` / `multiavoidmeal` / `multiavoidrecipe` | 多条件联合规避 |
| `multifondnessingredient` / `multifondnessmeal` / `multifondnessrecipe` | 多条件联合喜好 |
| `multirecommendingredient` / `multirecommendmeal` / `multirecommendrecipe` | 多条件联合推荐 |

### 分类偏好类
| 表名 | 说明 |
|------|------|
| `classifcationavoidingredient` / `classifcationavoidmeal` / `classifcationavoidrecipe` | 分类人群规避规则 |
| `classifcationfondnessingredient` / `classifcationfondnessmeal` / `classifcationfondnessrecipe` | 分类人群喜好规则 |

### 疾病相关
| 表名 | 说明 |
|------|------|
| `diseaseavoidingredient` / `diseaseavoidmeal` / `diseaseavoidrecipe` | 疾病规避规则 |
| `diseaserecommendingredient` / `diseaserecommendmeal` / `diseaserecommendrecipe` | 疾病推荐规则 |

### 季节/症状/运动/分类推荐
| 表名 | 说明 |
|------|------|
| `seasonrecommendingredient` / `seasonrecommendingmeal` / `seasonrecommendingrecipe` | 季节推荐 |
| `symptomrecommendingredient` / `symptomrecommendmeal` / `symptomrecommendrecipe` | 症状推荐 |
| `sportrecommendingredient` / `sportrecommendmeal` / `sportrecommendrecipe` | 运动推荐 |
| `classificationrecommendingredient` / `classificationrecommendingredientdetail` / `classificationrecommendmeal` / `classificationrecommendmealdetail` / `classificationrecommendrecipedetail` | 分类推荐详情 |

### 其他
| 表名 | 说明 |
|------|------|
| `chinesealias` | 食材中文别名 |
| `englishalias` | 食材英文别名 |
| `latinalias` | 食材拉丁别名 |
| `cookstepdetail` | 烹饪步骤用料明细 |
| `ingredientrecommend` / `ingredientrecommendation` | 食材推荐主表 |
| `mealrecommend` / `mealrecommendetail` | 餐推荐 |
| `hcirecommendcomposition` / `hcirecommendingredient` / `hcirecommendmeal` / `hcirecommendrecipe` | 健康指标推荐 |
| `picture` / `video` | 媒体资源 |
| `mealeffect` | 餐功效 |
| `ingredientproduceaddress` / `ingredientsampleaddress` | 食材产地/采样地 |
| `suggest` | 建议表 |

> 粗略估计，160 张表中约有 **100+ 张表只有 -2 占位符**，即超过 60% 的表处于"结构就绪但无数据"状态。

---

## 三、表间关系问题

### 3.1 外键引用了几乎为空的表

多处 FK 指向只含 6 条测试数据的 `user` 表：

- `classificationrecommendingredient.provider` → `user.id`
- `classificationrecommendmeal.provider` → `user.id`
- `classificationrecommendrecipe.provider` → `user.id`（此表有 580+ 条数据，每条 provider=2）
- `comment.user` → `user.id`
- `picture.user` → `user.id`
- `video.user` → `user.id`
- 所有 `user*` 表的 FK 均指向 `user.id`

> 这些 FK 目前靠 id=-2 的 NullEntity 维持引用完整性，但真实用户数据严重不足。

### 3.2 cuisine 表数据分类混乱

`cuisine` 表包含：
```
1-8: 八大菜系（鲁川粤苏闽浙湘徽）
9: 家常菜
10: 饮品
11: 京菜
12: 促进睡眠 ← ❌ 这不是菜系，是功效！
13: 鄂菜
14: 豫菜
```

`id=12` 的"促进睡眠"明显应该放在 `mealeffect` 表或作为 `classification` 条目，放在 `cuisine` 里会导致数据混乱。

### 3.3 effect 字段语义不一致

`classificationrecommendrecipe` 表的 `effect` 字段：
- 列名叫 `effect`（效果）
- 但 FK 指向 `cycle.id`（周期表）
- 索引名叫 `recommendtype`
- 同一个字段三个名字，语义矛盾

### 3.4 disease 表的自引用 FK 命名混淆

```sql
`maintype` bigint(20) → REFERENCES disease(id)  -- "主类型"指向自己？
`subtype` bigint(20)  → REFERENCES disease(id)  -- "子类型"指向自己？
```

如果 `maintype`/`subtype` 是疾病分类的层级关系，叫 `parent` 更清晰。与已有的 `parent` 字段功能重叠。

---

## 四、拼写与命名错误

| 位置 | 错误 | 正确 | 影响 |
|------|------|------|------|
| 表名 ×8 | `classifcation*` | `classification*` | 少了字母 `i`，影响所有相关表名和索引名 |
| `ingredientincompatibility` | `soruce` (列名) | `source` | 外键列名拼错，写代码时容易出错 |
| `analysismethod` | `accuraty` | `accuracy` | 列名拼错 |
| 表名 ×3 | `useracutal*` | `useractual*` | 多了 `useractualmeal` 和 `useracutalmeal` 两个功能几乎相同的表 |
| 索引名 | `classifcationavoidingredient_ibfk_2` 出现在 `classifcationfondnessingredient` | `classifcationfondnessingredient_ibfk_2` | 复制粘贴错误，索引名与实际表名不符 |

---

## 五、数据类型问题

### 5.1 不合理的 bigint(255)

`picture` 和 `video` 表的所有 FK 列使用了 `bigint(255)`：
```sql
`user` bigint(255) NULL DEFAULT NULL,
`ingredient` bigint(255) NULL DEFAULT NULL,
```
MySQL 中 `bigint` 固定为 8 字节，`(255)` 是显示宽度修饰符，无实际意义且容易误导。标准写法是 `bigint(20)`。

### 5.2 字符集不统一

部分表使用 `utf8`，部分使用 `utf8mb4`：
- `address` — utf8
- `analysismethod` — utf8mb4
- `user` — utf8
- 大部分新表 — utf8mb4

混用可能导致中文特殊字符（如 emoji 或生僻字）的部分表存储失败。

---

## 六、数据质量问题

### 6.1 负值营养成分

`ingredientcomposite` 表中存在负值：
```
黄嘌呤: '-2.0毫克'  (id=117527)
黄嘌呤: '-2.0毫克'  (id=117532)
```
营养素含量不应为负数，这些数据需要人工核实修正。

### 6.2 address 表 id=8 缺失

address 表从 id=7 直接跳到 id=9，id=8 空缺（可能是数据删除后未重新整理）。

### 6.3 AUTO_INCREMENT 与已用 ID 不匹配

部分表的 `AUTO_INCREMENT` 值小于已使用的最大 ID，这本身不会导致错误（MySQL 会自动调整），但说明导出时可能未正确设置。例如：
- `classification` 表数据 id 最大=14，AUTO_INCREMENT=15（正常）
- `classifcationavoidingredient` AUTO_INCREMENT=1，但有 id=-2 的记录（下一个自增会是 1，正常）

### 6.4 user 表密码明文存储

```sql
INSERT INTO `user` VALUES (1, 'Root User', 'admin', '123456', ...);
INSERT INTO `user` VALUES (2, 'Super User', 'su', '123456', ...);
```
所有用户的密码都是 `123456`，且为明文存储，存在安全隐患。

---

## 七、统计摘要

| 统计项 | 数值 |
|--------|------|
| 表总数 | 160 |
| 有实质数据的表 | ~50 张 |
| 仅有 -2 占位符的表 | ~100 张 |
| 完全无数据的表 | 7 张 |
| 拼写错误的表名 | 8 张 (`classifcation*`) |
| 拼写错误的列名 | 2 处 (`soruce`, `accuraty`) |
| 数据类型不规范的表 | 2 张 (`picture`, `video`) |
| 数据分类错误的记录 | 1 条 (cuisine.促进睡眠) |
| 含负值异常的记录 | 2 条 (ingredientcomposite) |

---

## 八、建议修复优先级

**P0 — 立刻修复：**
1. 修正 `ingredientincompatibility` 表的 `soruce` 列名为 `source`
2. 将 `cuisine` 表中"促进睡眠"移到 `mealeffect` 或新建的 `classification` 条目
3. 修正 `picture`/`video` 表中 `bigint(255)` 为 `bigint(20)`
4. 统一所有表字符集为 `utf8mb4`

**P1 — 尽快处理：**
5. 修正 8 张 `classifcation*` 表名为 `classification*`（涉及表结构、索引、外键、代码）
6. 修正 `analysismethod.accuraty` 为 `accuracy`
7. 决策是合并还是删除重复的 `useracutal*` 和 `useractual*` 两组表

**P2 — 后续优化：**
8. 修正 `ingredientcomposite` 中的负值数据
9. 为 `-2` 占位符超过 10 条的其他空表填充基础数据
10. 密码改用哈希存储

[查看完整分析报告](computer://D:\科研实习\数据库\database_audit_report.md)
