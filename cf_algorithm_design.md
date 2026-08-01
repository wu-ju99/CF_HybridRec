# 基于协同过滤的混合菜谱推荐算法设计

## 目录
1. [数据源与特征映射](#一数据源与特征映射)
2. [特征工程与向量化](#二特征工程与向量化)
3. [协同过滤模块](#三协同过滤模块)
4. [矩阵分解模块 (SVD++)](#四矩阵分解模块)
5. [内容增强模块](#五内容增强模块)
6. [混合融合策略](#六混合融合策略)
7. [冷启动处理](#七冷启动处理)
8. [评价体系](#八评价体系)
9. [权重选择方法](#九权重选择方法)
10. [实现流程总图](#十实现流程总图)

---

## 一、数据源与特征映射

完全基于现有 `dietrecommendation` 数据库表头，不使用任何外部特征。

### 1.1 用户-菜谱交互矩阵（协同过滤核心）

来源表及其角色：

```
comment (rating)  ────────────┐
useracutalrecipe (user,recipe)──→  用户-菜谱评分矩阵 R ∈ ℝ^(m×n)
userbrowse (浏览记录)  ───────┘      m = 用户数, n = 菜谱数
```

**评分构造方式：**
- `comment` 表有 rating 时，直接使用（1-5 分）
- `useracutalrecipe` 有记录但无 rating 时，视为隐式反馈（r=3，中性默认）
- `userbrowse` 记录作为弱隐式反馈（r=2，表示有关注但未实际制作）

### 1.2 用户特征向量 Uᵢ

全部来自数据库已有字段：

| 数据库来源 | 特征维度 | 编码方式 | 维度数 |
|-----------|---------|---------|--------|
| `user.gender` | 性别 | One-hot: {男, 女, 未知} | 3 |
| `user.birthday` → 计算 `classification` id | 年龄人群 | One-hot（对应classification.id: 1婴儿,2幼儿,3儿童,4少年,5青年,6老年） | 6 |
| `user.occupation` | 职业类型 | One-hot（对应occupation.id） | N_occ |
| `usertaste` → `taste.id` + `intensity.id` | 口味偏好 | 每种口味×强度归一化 | N_taste(=5) |
| `userbirthplace` → `address.id` | 出生地域 | 取省级行政区 One-hot | N_prov |
| `userworkplace` → `address.id` | 工作地域 | 取省级行政区 One-hot | N_prov |

**用户特征向量维度：** d_u = 3 + 6 + N_occ + 5 + 2×N_prov

### 1.3 菜谱特征向量 Vⱼ

| 数据库来源 | 特征 | 编码 | 维度 |
|-----------|------|------|------|
| `recipe.cuisine` | 菜系 | One-hot (对应cuisine.id) | N_cuisine |
| `recipe.gi` | 升糖指数 | 归一化数值 | 1 |
| `recipe.timeconsumming` | 烹饪时长 | log归一化 | 1 |
| `recipe.cost` | 成本 | 归一化数值 | 1 |
| `recipeingredient` → `ingredient` → `foodtype` | 食材类型分布 | TF-IDF加权分布 | N_foodtype |
| `recipeingredient` → `ingredient` → `ingredient2taste` → `taste` | 口味特征 | 聚合后归一化 | N_taste |
| `recipeingredient` → `ingredient` → `ingredient2nature` → `nature` | 食材性质 | One-hot分布 (寒热温凉等) | N_nature |
| `recipecomposite` → `composition` → `content` | 营养成分 | 7维归一化向量 | 7 |
| `cookstep` → `cookmethod` | 烹饪方式 | 多热编码分布 | N_cookmethod |
| `classifcationavoidrecipe` / `diseaseavoidrecipe` 等 | 避忌标签 | 多热编码 | N_avoid_tags |

**菜谱特征向量维度：** d_v = N_cuisine + 3 + N_foodtype + 5 + N_nature + 7 + N_cookmethod + N_avoid_tags

---

## 二、特征工程与向量化

### 2.1 数值特征归一化

所有连续值特征使用 **Min-Max 归一化**（对非正态分布更鲁棒）：

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

`recipe.gi` 例外：已有明确的范围 [0, 100+]，使用截断归一化 `x_clipped/100`。

`recipe.timeconsumming` 使用对数归一化（因为烹饪时长呈长尾分布）：

$$t_{norm} = \frac{\log(1 + t)}{\log(1 + t_{max})}$$

### 2.2 类别特征 One-Hot 编码

对 `cuisine`, `foodtype`, `nature`, `cookmethod`, `classification`, `occupation` 等类别特征：

$$\text{OneHot}(c) = [0,0,...,1,...,0] \in \{0,1\}^{|C|}$$

### 2.3 食材→口味特征聚合

菜谱 j 含多种食材，每种食材关联多种口味（来自 `ingredient2taste`），使用加权平均：

$$\vec{v}_{taste}^{(j)} = \frac{1}{|I_j|} \sum_{k \in I_j} \vec{t}_{ik}$$

其中 $I_j$ 是菜谱 j 的食材集合，$\vec{t}_{ik}$ 是食材 k 的口味向量（5维，每维取 intensity 归一化值）。

### 2.4 食材类型 TF-IDF 加权

将每个菜谱视为"文档"，食材类型视为"词"，计算 TF-IDF：

$$tfidf(t, j) = \frac{f_{t,j}}{|I_j|} \cdot \log\frac{N}{1 + n_t}$$

其中 $f_{t,j}$ 是菜谱 j 中属于类型 t 的食材数，N 是总菜谱数，$n_t$ 是包含该类型食材的菜谱数。

### 2.5 用户口味偏好向量

从 `usertaste` 表读取，若无数据则用默认值（全 0）：

$$\vec{u}_{taste}^{(i)} = [i_1, i_2, i_3, i_4, i_5]$$

其中 $i_k$ 是用户 i 对第 k 种口味的 intensity 归一化值（来自 `intensity.id`）。没有记录的口味填 0。

### 2.6 最终向量拼接

**用户特征向量**（总维度 d_u）：

$$\mathbf{u}_i = [\text{OneHot}(gender) \;\|\; \text{OneHot}(age\_group) \;\|\; \text{OneHot}(occupation) \;\|\; \vec{u}_{taste} \;\|\; \text{OneHot}(birth\_province) \;\|\; \text{OneHot}(work\_province)]$$

**菜谱特征向量**（总维度 d_v）：

$$\mathbf{v}_j = [\text{OneHot}(cuisine) \;\|\; gi_{norm} \;\|\; t_{norm} \;\|\; cost_{norm} \;\|\; \vec{v}_{taste}^{(j)} \;\|\; \vec{v}_{nature}^{(j)} \;\|\; \vec{v}_{nutrition}^{(j)} \;\|\; \vec{v}_{foodtype\_tfidf}^{(j)} \;\|\; \text{MultiHot}(cookmethod) \;\|\; \text{MultiHot}(avoid\_tags)]$$

---

## 三、协同过滤模块

### 3.1 用户-菜谱评分矩阵构造

定义评分矩阵 $\mathbf{R} \in \mathbb{R}^{m \times n}$：

$$
R_{ui} = 
\begin{cases}
\text{comment.rating} & \text{如果用户 u 评价了菜谱 i} \\
3 & \text{如果用户 u 制作了但未评价 (来自 useracutalrecipe)} \\
2 & \text{如果用户 u 仅浏览过 (来自 userbrowse)} \\
\emptyset & \text{其他情况（待预测）}
\end{cases}
$$

### 3.2 基于用户的协同过滤 (User-Based CF)

**Step 1：用户相似度计算**

采用 **混合相似度**：评分行为相似度 + 用户画像相似度。

评分行为相似度（皮尔逊相关系数，对评分尺度差异鲁棒）：

$$sim_{rating}(u, v) = \frac{\sum_{i \in I_{uv}} (R_{ui} - \bar{r}_u)(R_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (R_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i \in I_{uv}} (R_{vi} - \bar{r}_v)^2}}$$

其中 $I_{uv}$ 是用户 u 和 v 共同评价过的菜谱集合。

用户画像相似度（余弦相似度）：

$$sim_{profile}(u, v) = \frac{\mathbf{u}_u \cdot \mathbf{u}_v}{\|\mathbf{u}_u\| \cdot \|\mathbf{u}_v\|}$$

融合相似度（加权组合）：

$$sim(u, v) = \alpha \cdot sim_{rating}(u, v) + (1-\alpha) \cdot sim_{profile}(u, v)$$

其中 $\alpha \in [0, 1]$ 是权重，当共同评分项少于阈值时降低 $\alpha$：

$$\alpha_{eff} = \alpha \cdot \min\left(1, \frac{|I_{uv}|}{\tau_{min}}\right)$$

$\tau_{min}$ 建议取 5。

**Step 2：评分预测**

选取 Top-K 最相似用户 $N_k(u)$，预测：

$$\hat{R}_{ui}^{UBCF} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} sim(u,v) \cdot (R_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |sim(u,v)|}$$

### 3.3 基于物品的协同过滤 (Item-Based CF)

**Step 1：菜谱相似度计算**

评分共现相似度（调整余弦相似度，消除用户评分偏差）：

$$sim_{co}(i, j) = \frac{\sum_{u \in U_{ij}} (R_{ui} - \bar{r}_u)(R_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (R_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u \in U_{ij}} (R_{uj} - \bar{r}_u)^2}}$$

菜谱内容相似度：

$$sim_{content}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \cdot \|\mathbf{v}_j\|}$$

融合相似度：

$$sim_{item}(i, j) = \beta \cdot sim_{co}(i, j) + (1-\beta) \cdot sim_{content}(i, j)$$

同样使用置信度衰减：  $\beta_{eff} = \beta \cdot \min(1, |U_{ij}| / \tau_{min})$

**Step 2：评分预测**

$$\hat{R}_{ui}^{IBCF} = \frac{\sum_{j \in N_k(i)} sim_{item}(i,j) \cdot R_{uj}}{\sum_{j \in N_k(i)} |sim_{item}(i,j)|}$$

---

## 四、矩阵分解模块 (SVD++)

### 4.1 基本 SVD 模型

将评分矩阵分解为低秩近似：

$$\mathbf{R} \approx \mathbf{P} \mathbf{Q}^T$$

其中 $\mathbf{P} \in \mathbb{R}^{m \times k}$ 是用户潜在因子矩阵，  $\mathbf{Q} \in \mathbb{R}^{n \times k}$   是菜谱潜在因子矩阵， $k$  是因子维度。

单个评分的预测：

$$\hat{R}_{ui}^{SVD} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

其中：
- $\mu$ = 全局平均评分
- $b_u$ = 用户偏置（某些用户倾向打高分/低分）
- $b_i$ = 菜谱偏置（某些菜谱天然受欢迎/不受欢迎）
- $\mathbf{p}_u$ = 用户 u 的 k 维潜在因子向量
- $\mathbf{q}_i$ = 菜谱 i 的 k 维潜在因子向量

### 4.2 SVD++（加入隐式反馈）

用户制作/浏览过但未评分的菜谱也包含信息。SVD++ 将隐式反馈纳入用户表示：

$$\hat{R}_{ui}^{SVD++} = \mu + b_u + b_i + \mathbf{q}_i^T \left( \mathbf{p}_u + |N(u)|^{-0.5} \sum_{j \in N(u)} \mathbf{y}_j \right)$$

其中 $N(u)$ 是用户 u 有过隐式反馈的菜谱集合（来自 `useracutalrecipe` 或 `userbrowse`），$\mathbf{y}_j$ 是菜谱 j 的隐式因子向量。

### 4.3 融入内容特征的 SVD++

利用菜谱内容特征向量初始化 $\mathbf{q}_i$（而不是随机初始化），加速收敛并改善冷启动：

$$\mathbf{q}_i^{(0)} = \mathbf{W} \cdot \mathbf{v}_i$$

其中 $\mathbf{W} \in \mathbb{R}^{k \times d_v}$ 是可学习的映射矩阵， $\mathbf{v}_i$  是菜谱 i 的内容特征向量。

### 4.4 损失函数

$$\mathcal{L} = \sum_{(u,i) \in \mathcal{D}_{train}} (R_{ui} - \hat{R}_{ui})^2 + \lambda \left( \sum_u \|\mathbf{p}_u\|^2 + \sum_i \|\mathbf{q}_i\|^2 + \sum_u b_u^2 + \sum_i b_i^2 + \sum_{j \in N(\cdot)} \|\mathbf{y}_j\|^2 + \|\mathbf{W}\|_F^2 \right)$$

其中 $\lambda$ 是正则化系数， $\|\cdot\|_F$  是 Frobenius 范数。

### 4.5 参数更新（随机梯度下降 SGD）

对每条观测 $(u,i)$：

```
e_ui = R_ui - R̂_ui

b_u ← b_u + η * (e_ui - λ * b_u)
b_i ← b_i + η * (e_ui - λ * b_i)

p_u ← p_u + η * (e_ui * q_i            - λ * p_u)
q_i ← q_i + η * (e_ui * (p_u + Σy_j)   - λ * q_i)

对于每个 j ∈ N(u):
    y_j ← y_j + η * (e_ui * |N(u)|^{-0.5} * q_i - λ * y_j)
```

其中 $\eta$ 是学习率，使用指数衰减： $\eta_t = \eta_0 \cdot \gamma^{t/T}$ 。

---

## 五、内容增强模块

针对交互稀疏场景（新用户、新菜谱），直接从特征向量预测评分：

### 5.1 双塔模型 (Two-Tower / Siamese)

$$\hat{R}_{ui}^{content} = \sigma\left( f_u(\mathbf{u}_i) \cdot f_v(\mathbf{v}_j) \right)$$

其中 $f_u$ 和 $f_v$ 是小型全连接网络（各 2-3 层），将用户和菜谱特征映射到同一 k 维空间后做内积。 $\sigma$  将值映射到 [1, 5]。

简单版（线性，可解释性强）：

$$\hat{R}_{ui}^{content} = \mu + \mathbf{w}_u^T \mathbf{u}_i + \mathbf{w}_v^T \mathbf{v}_j + \mathbf{u}_i^T \mathbf{M} \mathbf{v}_j$$

其中 $\mathbf{M}$ 是交互矩阵，捕获特征之间的交叉效应。

### 5.2 基于菜谱相似度的内容推荐

当用户交互极少时，退化为纯内容推荐：

$$\hat{R}_{ui}^{cold\_content} = \frac{\sum_{j \in H(u)} sim_{content}(i, j) \cdot R_{uj}}{\sum_{j \in H(u)} sim_{content}(i, j)}$$

其中 $H(u)$ 是用户 u 已交互过的菜谱集合。

---

## 六、混合融合策略

### 6.1 分层加权融合

最终预测是四个模块的加权组合：

$$\hat{R}_{ui}^{final} = w_1 \hat{R}_{ui}^{UBCF} + w_2 \hat{R}_{ui}^{IBCF} + w_3 \hat{R}_{ui}^{SVD++} + w_4 \hat{R}_{ui}^{content}$$

约束： $\sum_{k=1}^4 w_k = 1, \quad w_k \geq 0$ 

### 6.2 动态权重（置信度自适应）

权重不应固定，而应根据用户/菜谱的交互密度动态调整：

**用户端置信度：**

$$c_u = 1 - e^{-\gamma_u \cdot n_u}$$

其中 $n_u$ 是用户 u 的总交互数，$\gamma_u$ 是衰减速率（建议 0.1）。

**菜谱端置信度：**

$$c_i = 1 - e^{-\gamma_i \cdot n_i}$$

其中 $n_i$ 是菜谱 i 被交互的总次数。

**动态权重调整：**
- 当 $c_u$ 高且 $c_i$ 高 → 提高 $w_3$（SVD++ 权重），因为隐式因子学得充分
- 当 $c_u$ 高但 $c_i$ 低 → 提高 $w_1$（User-CF），利用相似用户
- 当 $c_u$ 低但 $c_i$ 高 → 提高 $w_2$（Item-CF），利用相似菜谱
- 当两者都低 → 提高 $w_4$（Content），靠特征兜底

具体公式：

$$w_k' = w_k \cdot \phi_k(c_u, c_i)$$
$$w_k = \frac{w_k'}{\sum_{j} w_j'}$$

其中 $\phi_k$ 是每个模块的置信度函数：

$$\begin{aligned}
\phi_1(c_u, c_i) &= c_u \cdot (1 - c_i) + 0.5 \\
\phi_2(c_u, c_i) &= (1 - c_u) \cdot c_i + 0.5 \\
\phi_3(c_u, c_i) &= c_u \cdot c_i + 0.3 \\
\phi_4(c_u, c_i) &= (1 - c_u) \cdot (1 - c_i) + 0.3
\end{aligned}$$

### 6.3 融合流程图

```
                    ┌─────────────┐
                    │ 用户u, 菜谱i │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 计算 c_u │ │ 计算 c_i │ │ 特征向量 │
        │ (交互数) │ │ (被交互数)│ │ Uᵢ, Vⱼ   │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
    ┌────────┴────┬───────┴───────┬────┴────────┐
    ▼             ▼               ▼             ▼
┌──────┐    ┌──────┐        ┌──────┐      ┌──────┐
│UserCF│    │ItemCF│        │SVD++ │      │Content│
│  ŷ₁  │    │  ŷ₂  │        │  ŷ₃  │      │  ŷ₄   │
└──┬───┘    └──┬───┘        └──┬───┘      └──┬───┘
   │           │               │              │
   └───────────┴───────┬───────┴──────────────┘
                       ▼
              ┌─────────────────┐
              │ 动态权重 wₖ(cᵤ,cᵢ)│
              │ ŷ_final = Σ wₖŷₖ │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Top-N 推荐列表  │
              └─────────────────┘
```

---

## 七、冷启动处理

### 7.1 新用户冷启动

当用户无任何交互记录时：

1. **仅使用用户画像**：用 $\mathbf{u}_i$ 通过 content 模块预测
2. **人口统计匹配**：找到同 age_group + gender + occupation 的已有用户的平均偏好，作为初始推荐
3. **热门但多样**：推荐全局热门菜谱，但强制覆盖不同 cuisine 和 taste

$$\hat{R}_{ui}^{new\_user} = \bar{r}_i + \mathbf{w}_{demo}^T \cdot (U_i^{demo} \odot \bar{U}_{demo}^{pref})$$

### 7.2 新菜谱冷启动

当菜谱无任何交互记录时：

1. **仅使用菜谱特征**：用 $\mathbf{v}_j$ 通过 content 模块预测
2. **相似的已流行菜谱**：找最相似的 Top-5 已评级菜谱，取它们的平均评分
3. **利用菜谱元数据**：通过 cuisine/cost/time 相似的菜谱推断初始评分

---

## 八、评价体系

### 8.1 离线评估指标

**A. 预测准确度指标**

均方根误差（RMSE）：

$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{D}_{test}|} \sum_{(u,i) \in \mathcal{D}_{test}} (R_{ui} - \hat{R}_{ui})^2}$$

平均绝对误差（MAE）：

$$\text{MAE} = \frac{1}{|\mathcal{D}_{test}|} \sum_{(u,i) \in \mathcal{D}_{test}} |R_{ui} - \hat{R}_{ui}|$$

**B. 排序质量指标**

Precision@K（推荐 K 个中命中的比例）：

$$\text{Precision@K} = \frac{1}{m} \sum_{u=1}^{m} \frac{|\text{TopK}(u) \cap \text{Relevant}(u)|}{K}$$

Recall@K（用户喜欢的所有菜谱中被推荐出来的比例）：

$$\text{Recall@K} = \frac{1}{m} \sum_{u=1}^{m} \frac{|\text{TopK}(u) \cap \text{Relevant}(u)|}{|\text{Relevant}(u)|}$$

归一化折损累计增益（NDCG@K）：

$$\text{NDCG@K} = \frac{1}{m} \sum_{u=1}^{m} \frac{DCG_u@K}{IDCG_u@K}$$

$$DCG_u@K = \sum_{k=1}^{K} \frac{2^{rel_{uk}} - 1}{\log_2(k+1)}$$

其中 $rel_{uk}$ 是用户 u 对排在第 k 位的菜谱的真实评分。

**C. 推荐质量指标**

覆盖率（Coverage）：

$$\text{Coverage} = \frac{|\bigcup_{u} \text{TopK}(u)|}{n}$$

衡量推荐系统覆盖了多少不同的菜谱。

多样性（Diversity）：

$$\text{Diversity} = \frac{1}{m} \sum_{u=1}^{m} \frac{1}{K(K-1)} \sum_{i \neq j \in \text{TopK}(u)} (1 - sim_{content}(i, j))$$

新颖性（Novelty）：

$$\text{Novelty} = \frac{1}{mK} \sum_{u} \sum_{i \in \text{TopK}(u)} -\log_2\left(\frac{|U_i|}{m}\right)$$

其中 $|U_i|$ 是与菜谱 i 有过交互的用户数。越冷门的菜谱被推荐，新颖性越高。

### 8.2 评估的数学原理（如何指导模型改进）

**假设检验驱动优化：**

将每个指标分解为可归因的子项。以 RMSE 为例：

$$\text{RMSE}^2 = \underbrace{\frac{1}{|\mathcal{D}|}\sum (R_{ui} - \hat{R}_{ui}^{UBCF})^2}_{\text{UserCF误差}} + \underbrace{\frac{1}{|\mathcal{D}|}\sum (R_{ui} - \hat{R}_{ui}^{SVD})^2}_{\text{SVD误差}} + \ldots - \text{协方差项}$$

通过消融实验（Ablation Study），依次移除每个模块，观察各指标的退化量，精准定位瓶颈模块。

**超参数敏感度分析：**

对每个超参数 $\theta$（如 k, λ, α, β），计算偏导数近似：

$$\frac{\partial \text{RMSE}}{\partial \theta} \approx \frac{\text{RMSE}(\theta + \epsilon) - \text{RMSE}(\theta - \epsilon)}{2\epsilon}$$

高敏感度参数需要精细调优，低敏感度参数可取默认值。

### 8.3 离线实验设计

使用时间戳切分（更贴近真实场景）：

```
训练集: 所有 2025-12-01 之前的交互
验证集: 2025-12-01 ~ 2026-01-01 的交互（调超参数用）
测试集: 2026-01-01 之后的交互（最终评估用）
```

每个用户至少保留 1 条交互在测试集中，否则不纳入评估。

---

## 九、权重选择方法

### 9.1 融合权重 $\{w_1, w_2, w_3, w_4\}$

**方法一：网格搜索 + 交叉验证**

```
For w₁ in {0.1, 0.2, ..., 0.7}:
  For w₂ in {0.1, 0.2, ..., 0.7}:
    For w₃ in {0.1, 0.2, ..., 0.7}:
      w₄ = 1 - w₁ - w₂ - w₃
      if w₄ < 0: continue
      在验证集上计算 RMSE
      选择 RMSE 最小的组合
```

**方法二：贝叶斯优化（推荐，效率高）**

使用高斯过程（Gaussian Process）作为代理模型，Expected Improvement（EI）作为采集函数：

$$EI(\mathbf{w}) = \mathbb{E}[\max(0, f_{min} - f(\mathbf{w}))]$$

比网格搜索收敛更快，通常 50-100 轮即可找到近似最优。

### 9.2 SVD++ 超参数

| 参数 | 含义 | 搜索范围 | 建议初始值 |
|------|------|---------|-----------|
| k | 潜在因子维度 | {20, 50, 100, 150, 200} | 100 |
| λ | 正则化系数 | {0.001, 0.01, 0.05, 0.1, 0.5} | 0.05 |
| η₀ | 初始学习率 | {0.001, 0.005, 0.01, 0.02} | 0.01 |
| epochs | 迭代轮数 | {20, 50, 100, 200} | 50 |

### 9.3 相似度融合权重 α, β

α（User-CF 中评分相似度 vs 画像相似度）：
- 用户交互数 $n_u < 5$：α = 0.3（依赖画像）
- 用户交互数 $5 \leq n_u < 20$：α = 0.5
- 用户交互数 $n_u \geq 20$：α = 0.8（依赖评分行为）

β（Item-CF 中共现相似度 vs 内容相似度）同理。

### 9.4 置信度参数 γ_u, γ_i

选择使验证集 RMSE 最小的值，搜索范围 {0.05, 0.1, 0.2, 0.5}。初始建议 γ_u = γ_i = 0.1。

---

## 十、实现流程总图

```
┌──────────────────────────────────────────────────────────┐
│                    第一步：数据抽取                        │
│  ┌─────────┐  ┌───────────┐  ┌───────────┐              │
│  │ comment │  │useracutal │  │userbrowse │  ...         │
│  │ (评分)  │  │ recipe    │  │ (浏览)    │              │
│  └────┬────┘  └─────┬─────┘  └─────┬─────┘              │
│       └──────────────┼─────────────┘                     │
│                      ▼                                    │
│           构建 R 矩阵 (m×n 评分矩阵)                       │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   第二步：特征向量化                       │
│  user表→Uᵢ(d_u维)    recipe/ingredient/cookstep等→Vⱼ(d_v维)│
│  类别OneHot  数值归一化  TF-IDF  MultiHot                │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│              第三步：协同过滤 (四模块并行)                 │
│  ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐        │
│  │User-CF │ │Item-CF │ │  SVD++   │ │Content   │        │
│  │KNN(sim)│ │KNN(sim)│ │SGD迭代   │ │双塔/线性 │        │
│  │  ŷ₁    │ │  ŷ₂    │ │   ŷ₃     │ │   ŷ₄     │        │
│  └───┬────┘ └───┬────┘ └────┬─────┘ └────┬─────┘        │
└──────┼──────────┼──────────┼────────────┼──────────────┘
       └──────────┴─────┬────┴────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│                第四步：动态混合融合                        │
│         计算 c_u, c_i → 动态权重 wₖ(c_u, c_i)             │
│         ŷ_final = w₁ŷ₁ + w₂ŷ₂ + w₃ŷ₃ + w₄ŷ₄             │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│                  第五步：Top-N 推荐                        │
│     对每个用户，按 ŷ_final 降序排列所有未交互菜谱          │
│     截取 Top-K = 20 (可配置) 作为推荐列表                 │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│                第六步：离线评估                            │
│  RMSE, MAE, Precision@K, Recall@K, NDCG@K               │
│  Coverage, Diversity, Novelty                            │
│  消融实验 + 超参数敏感度分析                               │
└──────────────────────────────────────────────────────────┘
```

---

## 附录：关键超参数速查

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| K (UBCF邻居数) | 50 | 相似用户数 |
| K (IBCF邻居数) | 30 | 相似菜谱数 |
| k (SVD因子) | 100 | 潜在维度 |
| λ | 0.05 | L2正则化系数 |
| η₀ | 0.01 | 初始学习率 |
| α | 0.5→0.8 自适应 | UBCF评分vs画像权重 |
| β | 0.5→0.8 自适应 | IBCF共现vs内容权重 |
| τ_min | 5 | 置信度衰减阈值 |
| γ_u, γ_i | 0.1 | 置信度衰减速率 |
| w₁,w₂,w₃,w₄ | 验证集搜索 | 融合权重 |
| Top-K 推荐 | 20 | 返回菜谱数 |

[查看完整算法设计文档](computer://D:\科研实习\数据库\cf_algorithm_design.md)
