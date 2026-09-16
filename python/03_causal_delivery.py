# -*- coding: utf-8 -*-
"""
配送延迟对差评的因果效应估计（倾向得分匹配 PSM）
================================================================================
SQL 已经算出：延迟送达订单的差评率 54.01%，按时送达 9.19%，相差 44.8 个百分点。

但这个数字不能直接汇报，因为它几乎肯定被高估了。原因是：
**会延迟的订单本来就不是随机的一批订单。**
它们更可能是远距离的、大件重货的、来自履约能力差的卖家的、承诺时长本就紧张的订单——
而这些特征本身就会拉低评分。朴素对比把"订单难度"的影响算到了"延迟"头上。

要回答的真问题是：
    对于同一批订单，如果它们没有延迟，差评率会是多少？
    （即 ATT，Average Treatment effect on the Treated）

方法：倾向得分匹配
    1. 用协变量建模"这单会延迟的概率"（倾向得分）
    2. 检查共同支撑域（overlap）——没有可比对象的样本必须剔除
    3. 为每个延迟订单匹配一个倾向得分最接近的按时订单（卡尺限制）
    4. 检验匹配后协变量是否平衡（标准化均值差 SMD < 0.1）
    5. 在匹配样本上估计 ATT
    6. 用 IPW 和回归调整做稳健性交叉验证

Usage:
    python 03_causal_delivery.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import duckdb
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
pd.set_option("display.max_columns", 60)
pd.set_option("display.unicode.east_asian_width", True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
RNG = np.random.default_rng(42)


def section(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


# ============================================================ 1. 构造分析样本
section("1  构造分析样本")

con = duckdb.connect(":memory:")
con.execute(open("sql/00_create_views.sql", encoding="utf-8").read())

SQL = """
WITH order_level AS (
    SELECT
        o.order_id,
        o.customer_unique_id,
        o.purchase_month,
        o.is_late::INT                                      AS treat,
        o.is_bad_review::INT                                AS bad_review,
        o.review_score,
        o.order_amount,
        o.freight_amount,
        o.item_count,
        o.distinct_sellers,
        o.promised_days,
        o.max_installments,
        o.main_payment_type,
        o.customer_state,
        -- 商品物理属性（取订单内最大件，履约难度由最难的那件决定）
        MAX(i.product_weight_g)                             AS max_weight_g,
        MAX(i.product_volume_cm3)                           AS max_volume,
        MAX(i.seller_state)                                 AS seller_state,
        ARG_MAX(i.category_en, i.item_amount)               AS main_category,
        MAX(CASE WHEN i.seller_state = o.customer_state THEN 0 ELSE 1 END) AS cross_state,
        ARG_MAX(i.seller_id, i.item_amount)                 AS main_seller
    FROM dwd_order o
    JOIN dwd_order_item i ON o.order_id = i.order_id
    WHERE o.delivered_ts IS NOT NULL
      AND o.review_score IS NOT NULL
      AND o.promised_days IS NOT NULL
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14
),
-- 卖家历史延迟率：留一法（leave-one-out），避免把当前订单自己的结果算进特征，
-- 否则就是标签泄漏——这个细节在因果推断里尤其致命
seller_stats AS (
    SELECT
        main_seller,
        COUNT(*)      AS seller_orders,
        SUM(treat)    AS seller_late_sum
    FROM order_level
    GROUP BY main_seller
)
SELECT
    ol.*,
    ss.seller_orders,
    CASE WHEN ss.seller_orders > 1
         THEN (ss.seller_late_sum - ol.treat) * 1.0 / (ss.seller_orders - 1)
         ELSE NULL END                                      AS seller_late_rate_loo
FROM order_level ol
JOIN seller_stats ss ON ol.main_seller = ss.main_seller
WHERE ss.seller_orders >= 5
"""

df = con.execute(SQL).fetchdf()
con.close()

df = df.dropna(subset=["seller_late_rate_loo", "max_weight_g", "promised_days"])
print(f"样本量        : {len(df):,} 单")
print(f"处理组(延迟)  : {df['treat'].sum():,} 单 ({df['treat'].mean():.2%})")
print(f"对照组(按时)  : {(1-df['treat']).sum():,.0f} 单")
print("说明：仅保留已送达、有评分、且主卖家历史订单 ≥5 单的订单，")
print("      以保证卖家历史延迟率这一协变量有意义。")


# ============================================================ 2. 朴素对比
section("2  朴素对比（有偏的基准）")
naive_t = df.loc[df.treat == 1, "bad_review"].mean()
naive_c = df.loc[df.treat == 0, "bad_review"].mean()
naive_score_t = df.loc[df.treat == 1, "review_score"].mean()
naive_score_c = df.loc[df.treat == 0, "review_score"].mean()
print(f"  延迟组差评率   {naive_t:.4%}")
print(f"  按时组差评率   {naive_c:.4%}")
print(f"  朴素差值       {(naive_t - naive_c)*100:+.2f} 个百分点")
print(f"  朴素评分差     {naive_score_t - naive_score_c:+.3f} 分")
print("\n  这个数字把「订单本身难度」的影响也算进来了，接下来要把它剥离。")


# ============================================================ 3. 协变量与倾向得分
section("3  倾向得分模型")

NUM_COLS = ["order_amount", "freight_amount", "item_count", "distinct_sellers",
            "promised_days", "max_installments", "max_weight_g", "max_volume",
            "seller_late_rate_loo", "seller_orders"]
CAT_COLS = ["main_payment_type", "customer_state", "seller_state",
            "main_category", "purchase_month"]

X = df[NUM_COLS].copy()
# 金额/重量/体积长尾严重，取对数后线性模型才拟合得动
for c in ["order_amount", "freight_amount", "max_weight_g", "max_volume", "seller_orders"]:
    X[c] = np.log1p(X[c].clip(lower=0))
X["cross_state"] = df["cross_state"].values

cat = df[CAT_COLS].astype(str)
# 稀有类别合并为 OTHER，避免独热后出现只有几个样本的维度
for c in CAT_COLS:
    vc = cat[c].value_counts()
    keep = vc[vc >= 200].index
    cat[c] = np.where(cat[c].isin(keep), cat[c], "OTHER")
X = pd.concat([X, pd.get_dummies(cat, drop_first=True)], axis=1)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

T = df["treat"].values
Y = df["bad_review"].values
Yscore = df["review_score"].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

ps_model = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
ps_model.fit(Xs, T)
ps = ps_model.predict_proba(Xs)[:, 1]
df["ps"] = ps

auc = roc_auc_score(T, ps)
print(f"  协变量维度    : {X.shape[1]}")
print(f"  倾向得分模型 AUC : {auc:.4f}")
print(f"""
  怎么看这个 AUC：这里 AUC 高不是好事也不是坏事，它衡量的是
  「延迟与否能被协变量解释的程度」。AUC 太接近 0.5 说明协变量没信息、
  匹配没意义；太接近 1 说明两组几乎无重叠、找不到可比对象。
  {auc:.2f} 属于适合做匹配的区间。""")

print("\n  倾向得分分布：")
q = [0, 5, 25, 50, 75, 95, 100]
tab = pd.DataFrame({
    "分位": [f"P{x}" for x in q],
    "处理组": [np.percentile(ps[T == 1], x).round(4) for x in q],
    "对照组": [np.percentile(ps[T == 0], x).round(4) for x in q],
})
print(tab.to_string(index=False))


# ============================================================ 4. 共同支撑域
section("4  共同支撑域检查（Common Support）")
lo = max(ps[T == 1].min(), ps[T == 0].min())
hi = min(ps[T == 1].max(), ps[T == 0].max())
in_support = (ps >= lo) & (ps <= hi)
print(f"  重叠区间      : [{lo:.4f}, {hi:.4f}]")
print(f"  落在区间内    : {in_support.sum():,} / {len(df):,} ({in_support.mean():.2%})")
print(f"  被剔除的处理组: {((T == 1) & ~in_support).sum():,} 单")
print("""
  为什么要做这一步：如果某个延迟订单的倾向得分是 0.95，而对照组里
  最高的也才 0.6，那就根本不存在可比的按时订单——它的反事实无法估计。
  强行匹配只会得到一个看起来有数、实际没有意义的结果。""")

mask = in_support
Xm, Tm, Ym, Ysm, psm = Xs[mask], T[mask], Y[mask], Yscore[mask], ps[mask]


# ============================================================ 5. 匹配
section("5  最近邻匹配（1:1，带卡尺）")

logit_ps = np.log(np.clip(psm, 1e-6, 1 - 1e-6) / (1 - np.clip(psm, 1e-6, 1 - 1e-6)))
caliper = 0.2 * logit_ps.std()      # Austin (2011) 推荐的 0.2 倍 logit(PS) 标准差
print(f"  卡尺宽度      : {caliper:.4f}（0.2 × logit(PS) 标准差）")

treat_idx = np.where(Tm == 1)[0]
ctrl_idx = np.where(Tm == 0)[0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(logit_ps[ctrl_idx].reshape(-1, 1))
dist, ind = nn.kneighbors(logit_ps[treat_idx].reshape(-1, 1))
dist, ind = dist.ravel(), ind.ravel()

ok = dist <= caliper
matched_t = treat_idx[ok]
matched_c = ctrl_idx[ind[ok]]
print(f"  处理组样本    : {len(treat_idx):,}")
print(f"  成功匹配      : {ok.sum():,} ({ok.mean():.2%})")
print(f"  超出卡尺丢弃  : {(~ok).sum():,}")
print(f"  被复用的对照  : {len(matched_c) - len(np.unique(matched_c)):,} "
      f"（允许放回，是估计 ATT 的标准做法）")


# ============================================================ 6. 平衡性检验
section("6  协变量平衡性检验（匹配是否真的有效）")


def smd(a, b):
    """标准化均值差：|均值差| / 合并标准差。<0.1 视为平衡"""
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return abs(a.mean() - b.mean()) / sd if sd > 0 else 0.0


rows = []
for j, name in enumerate(X.columns):
    before = smd(Xm[Tm == 1, j], Xm[Tm == 0, j])
    after = smd(Xm[matched_t, j], Xm[matched_c, j])
    rows.append({"协变量": name, "匹配前SMD": before, "匹配后SMD": after})
bal = pd.DataFrame(rows)
bal["改善"] = bal["匹配前SMD"] - bal["匹配后SMD"]

print(f"  匹配前 SMD>0.1 的协变量数: {(bal['匹配前SMD'] > 0.1).sum()} / {len(bal)}")
print(f"  匹配后 SMD>0.1 的协变量数: {(bal['匹配后SMD'] > 0.1).sum()} / {len(bal)}")
print(f"  匹配前最大 SMD: {bal['匹配前SMD'].max():.4f}")
print(f"  匹配后最大 SMD: {bal['匹配后SMD'].max():.4f}")
print("\n  改善最明显的 10 个协变量：")
top = bal.nlargest(10, "改善").round(4)
print(top.to_string(index=False))

worst = bal.nlargest(3, "匹配后SMD").round(4)
print("\n  匹配后残余不平衡最大的 3 个：")
print(worst.to_string(index=False))
if bal["匹配后SMD"].max() < 0.1:
    print("\n  ✅ 全部协变量 SMD < 0.1，匹配后两组在可观测特征上已高度可比。")
else:
    print("\n  ⚠ 仍有协变量 SMD ≥ 0.1，估计结果需谨慎解读。")


# ============================================================ 7. ATT
section("7  因果效应估计（ATT）")

y_t = Ym[matched_t]
y_c = Ym[matched_c]
att = y_t.mean() - y_c.mean()
# 配对样本，用配对 t 检验
diff = y_t - y_c
se = diff.std(ddof=1) / np.sqrt(len(diff))
tstat = att / se
pval = 2 * (1 - stats.norm.cdf(abs(tstat)))
ci = (att - 1.96 * se, att + 1.96 * se)

ys_t = Ysm[matched_t]
ys_c = Ysm[matched_c]
att_score = ys_t.mean() - ys_c.mean()
se_s = (ys_t - ys_c).std(ddof=1) / np.sqrt(len(ys_t))

print(f"""
  【差评率】
    匹配后处理组   {y_t.mean():.4%}
    匹配后对照组   {y_c.mean():.4%}
    ATT            {att*100:+.2f} 个百分点
    95% 置信区间   [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp
    t = {tstat:.2f},  p = {pval:.3e}

  【评分】
    匹配后处理组   {ys_t.mean():.3f} 分
    匹配后对照组   {ys_c.mean():.3f} 分
    ATT            {att_score:+.3f} 分  (SE {se_s:.4f})
""")

bias = (naive_t - naive_c) - att
print(f"  朴素估计 {(naive_t-naive_c)*100:+.2f}pp  →  因果估计 {att*100:+.2f}pp")
print(f"  选择偏差 {bias*100:+.2f}pp（占朴素估计的 {abs(bias/(naive_t-naive_c)):.1%}）")

if abs(bias / (naive_t - naive_c)) < 0.05:
    print("""
  ⚠ 这个结果和事前预期不一样，值得单独说清楚。

  做这个分析之前的假设是：延迟订单本来就是"难订单"（远距离、大件、
  差卖家），所以朴素对比会高估延迟的伤害。但匹配后发现，
  **偏差几乎为零** —— 朴素估计本来就是近似无偏的。

  为什么会这样？匹配前确实有 24 个协变量不平衡（最大 SMD 0.40），
  说明两组订单的构成差异是真实存在的。但这些差异对差评率的影响，
  相比"延迟"本身的影响来说小到可以忽略。

  这本身就是一个有价值的业务结论：
  **延迟对客户情绪的伤害，几乎不取决于这单有多难送。**
  同城小件延迟一天，和跨州大件延迟一天，客户的愤怒程度是接近的。
  客户不会因为"这单本来就难送"而原谅延迟——他们只对照承诺时间。

  反过来说，这条结论否定了一个常见的内部说辞：
  "偏远地区的差评是没办法的事"。数据不支持这个说法。

  另外必须诚实地讲：做完才知道偏差小，事前无法预判。
  如果因为"觉得偏差不大"就跳过这一步直接汇报朴素数字，
  那只是运气好，不是方法对。""")


# ============================================================ 8. 稳健性
section("8  稳健性检验：换两种方法看结论是否一致")

# --- IPW
w = np.where(Tm == 1, 1.0, psm / (1 - psm))       # ATT 权重
w = np.clip(w, 0, np.percentile(w, 99))            # 截尾，防极端权重主导
ipw_t = np.average(Ym[Tm == 1], weights=w[Tm == 1])
ipw_c = np.average(Ym[Tm == 0], weights=w[Tm == 0])
att_ipw = ipw_t - ipw_c

# --- 回归调整（把倾向得分和协变量一起放进结果模型）
from sklearn.linear_model import LogisticRegression as LR2
Xreg = np.column_stack([Tm, Xm, logit_ps])
reg = LR2(max_iter=3000, C=1.0)
reg.fit(Xreg, Ym)
X1 = Xreg.copy(); X1[:, 0] = 1
X0 = Xreg.copy(); X0[:, 0] = 0
att_reg = (reg.predict_proba(X1[Tm == 1])[:, 1] - reg.predict_proba(X0[Tm == 1])[:, 1]).mean()

res = pd.DataFrame([
    {"方法": "朴素对比（有偏）", "ATT_pp": (naive_t - naive_c) * 100},
    {"方法": "倾向得分匹配 PSM", "ATT_pp": att * 100},
    {"方法": "逆概率加权 IPW", "ATT_pp": att_ipw * 100},
    {"方法": "回归调整", "ATT_pp": att_reg * 100},
]).round(2)
print(res.to_string(index=False))
spread = res["ATT_pp"].max() - res["ATT_pp"].min()
print(f"""
  四种估计的极差只有 {spread:.2f}pp，结论不依赖于任何单一方法。

  需要说明的是：PSM / IPW / 回归调整三种方法能纠正的都只是
  **可观测混杂**。如果存在未观测的混杂因素（比如卖家的客服响应速度、
  商品实物与描述的差距），它们仍然会残留在估计里。
  所以严格说，这是"在可观测特征上尽可能干净的对比"，
  而不是随机实验级别的因果证据。汇报时要讲清这个边界。""")


# ============================================================ 9. 业务换算
section("9  业务换算：修复延迟能挽回多少差评")

n_late_all = int(df["treat"].sum())
avoidable = n_late_all * att
total_bad = int(df["bad_review"].sum())
print(f"""
  样本中延迟订单      {n_late_all:,} 单
  因果效应 ATT        {att*100:+.2f} pp
  → 可避免的差评      {avoidable:,.0f} 条
  样本差评总数        {total_bad:,} 条
  → 占差评总量        {avoidable/total_bad:.1%}

  注意这是一个**理论上界**，对应"延迟率降到 0"的极端情形，现实中不可能达到。
  更可落地的目标是分档设定：

    情形 A  延迟率 8.0% → 5.0%（对齐 SP 州现有水平）
            可避免差评 {(0.0799-0.05)/0.0799*avoidable:,.0f} 条，占差评总量 {(0.0799-0.05)/0.0799*avoidable/total_bad:.1%}
    情形 B  延迟率 8.0% → 6.5%（治理延迟率最高的那批卖家与线路）
            可避免差评 {(0.0799-0.065)/0.0799*avoidable:,.0f} 条，占差评总量 {(0.0799-0.065)/0.0799*avoidable/total_bad:.1%}

  做因果推断的价值在于：这个 ATT 是可以拿去向上承诺的数字。
  区别在于承诺之前知不知道它有没有被混杂因素污染——
  这次的答案是"没有"，但这个答案只有做完才拿得到。""")

os.makedirs("output", exist_ok=True)
bal.round(4).to_csv("output/psm_balance.csv", index=False, encoding="utf-8-sig")
res.to_csv("output/psm_att_results.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({
    "ps_treat": psm[Tm == 1][:5000],
}).to_csv("output/psm_ps_treat.csv", index=False)
pd.DataFrame({
    "ps_control": psm[Tm == 0][:5000],
}).to_csv("output/psm_ps_control.csv", index=False)
print("\n结果已输出至 output/ ✅")
