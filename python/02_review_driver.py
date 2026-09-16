# -*- coding: utf-8 -*-
"""
差评驱动因素分析与下单时预警模型
================================================================================
前一步（因果推断）回答的是"延迟造成了多少差评"。
这一步回答两个可落地的问题：

    1. 除了延迟，还有什么在制造差评？各自权重多大？
    2. 能不能在**下单那一刻**就识别出高危订单，从而提前干预？

第 2 问的关键约束是**特征时点**：预警要在下单时可用，所以只能用
下单瞬间已知的信息（商品、卖家历史、地理、承诺时长、支付方式），
绝对不能用 delivery_days、is_late 这些"事后才知道"的字段。
把它们放进模型会得到一个 AUC 很漂亮但线上无法部署的废模型——
这是业务建模里最常见的穿越型数据泄漏。

因此本脚本训练两个模型做对比：
    模型 A  事后解释模型：含履约结果特征，用于归因"差评是怎么来的"
    模型 B  事前预警模型：只用下单时可得特征，用于线上拦截

Usage:
    python 02_review_driver.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.unicode.east_asian_width", True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)


def section(t):
    print("\n" + "=" * 94)
    print(t)
    print("=" * 94)


# ============================================================ 1. 构造样本
section("1  构造建模样本")
con = duckdb.connect(":memory:")
con.execute(open("sql/00_create_views.sql", encoding="utf-8").read())

df = con.execute("""
WITH base AS (
    SELECT
        o.order_id, o.purchase_ts, o.purchase_month,
        o.is_bad_review::INT                            AS y,
        o.review_score,
        -- 下单时可得
        o.order_amount, o.freight_amount, o.item_count, o.distinct_sellers,
        o.promised_days, o.max_installments, o.main_payment_type,
        o.used_voucher::INT                             AS used_voucher,
        o.customer_state, o.freight_ratio,
        MAX(i.product_weight_g)                         AS max_weight_g,
        MAX(i.product_volume_cm3)                       AS max_volume,
        AVG(i.product_photos_qty)                       AS avg_photos,
        ARG_MAX(i.category_en, i.item_amount)           AS main_category,
        ARG_MAX(i.seller_id, i.item_amount)             AS main_seller,
        MAX(i.seller_state)                             AS seller_state,
        MAX(CASE WHEN i.seller_state = o.customer_state THEN 0 ELSE 1 END) AS cross_state,
        -- 事后才知道（仅供模型A）
        o.delivery_days, o.is_late::INT                 AS is_late,
        o.days_early, o.seller_handling_days, o.carrier_transit_days
    FROM dwd_order o
    JOIN dwd_order_item i ON o.order_id = i.order_id
    WHERE o.delivered_ts IS NOT NULL AND o.review_score IS NOT NULL
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,23,24,25,26,27
)
SELECT * FROM base
""").fetchdf()
con.close()

# ---------------------------------------------------------------- 卖家历史特征
# ⚠ 这一段是本脚本最容易出错的地方，单独说明。
#
# 第一版实现用留一法（全期均值剔除当前订单）算卖家历史差评率，模型 B 的
# AUC 冲到了 0.93。这个数字是假的：对验证集里 2018 年 6 月的订单来说，
# 它的"卖家历史差评率"里混进了 7、8 月才发生的订单——模型看到了未来。
#
# 正确做法是**扩展窗口**：每个订单只能看到该卖家在它之前完成的订单。
# 这与线上真实可得的信息完全一致。
df = df.sort_values(["main_seller", "purchase_ts"]).reset_index(drop=True)
g = df.groupby("main_seller", sort=False)

df["seller_orders"] = g.cumcount()                       # 此前已有多少单
for src, dst in [("y", "seller_bad_rate_hist"),
                 ("is_late", "seller_late_rate_hist"),
                 ("seller_handling_days", "seller_handling_hist")]:
    # expanding().mean() 含当前行，shift(1) 把它挪掉，只保留严格更早的历史
    df[dst] = g[src].apply(lambda s: s.expanding().mean().shift(1)).values

before = len(df)
df = df[df["seller_orders"] >= 5]                        # 至少 5 单历史才有参考价值
df = df.dropna(subset=["seller_bad_rate_hist", "max_weight_g"])
df = df.sort_values("purchase_ts").reset_index(drop=True)

print(f"原始订单 : {before:,}")
print(f"建模样本 : {len(df):,}（要求主卖家此前已有 ≥5 单历史）")
print(f"差评率   : {df['y'].mean():.2%}")
print("""
注：卖家历史特征用扩展窗口计算，每单只能看到该卖家在它之前的订单。
    用全期均值（哪怕做了留一）会让模型看到未来，AUC 虚高但线上无效。""")

CAT = ["main_payment_type", "customer_state", "seller_state", "main_category"]
PRE_NUM = ["order_amount", "freight_amount", "item_count", "distinct_sellers",
           "promised_days", "max_installments", "used_voucher", "freight_ratio",
           "max_weight_g", "max_volume", "avg_photos", "cross_state",
           "seller_orders", "seller_late_rate_hist", "seller_handling_hist",
           "seller_bad_rate_hist"]
POST_NUM = ["delivery_days", "is_late", "days_early",
            "seller_handling_days", "carrier_transit_days"]


def build(cols_num):
    X = df[cols_num].copy()
    for c in CAT:
        X[c] = df[c].astype("category")
    return X.replace([np.inf, -np.inf], np.nan)


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def train(X, y, tag):
    # 按时间切分：用早期数据训练、后期验证，模拟真实上线场景。
    # 随机切分会让模型见到"未来"，高估线上表现。
    order = np.argsort(df["purchase_ts"].values)
    n_tr = int(len(order) * 0.75)
    tr, te = order[:n_tr], order[n_tr:]
    m = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                           num_leaves=31, min_child_samples=50, subsample=0.85,
                           subsample_freq=1, colsample_bytree=0.85,
                           reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[te], y[te])], eval_metric="auc",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    p = m.predict_proba(X.iloc[te])[:, 1]
    print(f"  {tag:<28} AUC {roc_auc_score(y[te], p):.4f} | "
          f"KS {ks(y[te], p):.4f} | PR-AUC {average_precision_score(y[te], p):.4f}")
    return m, p, te


# ============================================================ 2. 两个模型
section("2  两个模型：事后解释 vs 事前预警")
y = df["y"].values
print("  训练/验证按下单时间 75/25 切分（非随机切分，避免模型见到未来）\n")

XA = build(PRE_NUM + POST_NUM)
mA, pA, teA = train(XA, y, "模型A 含履约结果（解释用）")

XB = build(PRE_NUM)
mB, pB, teB = train(XB, y, "模型B 仅下单时特征（预警用）")

print(f"""
  两个 AUC 的差距（{roc_auc_score(y[teA], pA) - roc_auc_score(y[teB], pB):.4f}）就是
  「履约过程」贡献的信息量。模型 B 更弱是符合预期的——
  下单那一刻，这单会不会延迟本身就还没发生。

  关于模型 B 的 AUC {roc_auc_score(y[teB], pB):.3f}，要诚实地讲：这不是一个强模型。
  但它弱得有道理——差评的主要成因（延迟）在下单时根本还没发生，
  能提前看到的只有"结构性风险"：这条线路历史上就慢、这个卖家备货一向拖、
  这类大件本来就难送。天花板就在这里。

  真正该问的不是"AUC 能不能再高一点"，而是"这个区分度够不够支撑干预"。
  一个 AUC 0.64 但 TOP 10% 提升 2.4 倍的模型，只要干预成本足够低，
  依然是划算的；反过来，硬把 AUC 刷到 0.9 的唯一办法是把履约结果
  塞进特征——那样的模型上线时拿不到这些字段，等于没有。""")


# ============================================================ 3. 归因
section("3  差评归因：什么在制造差评（模型A 增益占比）")
imp = pd.DataFrame({
    "特征": XA.columns,
    "增益": mA.booster_.feature_importance(importance_type="gain")
}).sort_values("增益", ascending=False)
imp["占比_pct"] = (imp["增益"] / imp["增益"].sum() * 100).round(2)
imp["增益"] = imp["增益"].round(0)


def group(f):
    if f in POST_NUM:
        return "履约表现"
    if f.startswith("seller_"):
        return "卖家历史"
    if f in ("customer_state", "seller_state", "cross_state"):
        return "地理距离"
    if f in ("main_category", "max_weight_g", "max_volume", "avg_photos"):
        return "商品属性"
    return "订单与支付"


imp["归类"] = imp["特征"].map(group)
print(imp.head(12).to_string(index=False))
print("\n  按驱动因素分组的解释力：")
g = imp.groupby("归类")["占比_pct"].sum().sort_values(ascending=False).round(2)
print(g.to_string())


# ============================================================ 4. 预警价值
section("4  预警模型的业务价值（模型B）")
te = teB
base_rate = y[te].mean()
d = pd.DataFrame({"y": y[te], "p": pB})
d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10,
                      labels=[f"D{i}" for i in range(1, 11)])
lt = d.groupby("decile", observed=True).agg(
    订单数=("y", "size"), 差评数=("y", "sum"), 预测均值=("p", "mean"),
    实际差评率=("y", "mean")).reset_index()
lt["提升度"] = (lt["实际差评率"] / base_rate).round(2)
lt["累计捕获_pct"] = (lt["差评数"].cumsum() / d["y"].sum() * 100).round(2)
lt["实际差评率"] = (lt["实际差评率"] * 100).round(2)
lt["预测均值"] = lt["预测均值"].round(4)
print(f"  验证集 {len(te):,} 单，基线差评率 {base_rate:.2%}\n")
print(lt.to_string(index=False))

top10 = d.nlargest(int(len(d) * 0.1), "p")
top20 = d.nlargest(int(len(d) * 0.2), "p")
print(f"""
  怎么用这个模型：
    对分数最高的 10%（{len(top10):,} 单）做主动干预 —— 提前催发货、
    主动告知物流进度、必要时补偿运费券。
    这批订单的实际差评率 {top10['y'].mean():.1%}，是基线的
    {top10['y'].mean()/base_rate:.2f} 倍，覆盖了全部差评的
    {top10['y'].sum()/d['y'].sum():.1%}。

    若扩到 TOP 20%（{len(top20):,} 单），覆盖率升到
    {top20['y'].sum()/d['y'].sum():.1%}，但命中率降到 {top20['y'].mean():.1%}。
    具体取哪一档，取决于单次干预的成本和一条差评的代价，
    这是业务方的决策，不是模型能替他们做的。""")

os.makedirs("output", exist_ok=True)
imp.head(25).to_csv("output/review_feature_importance.csv", index=False, encoding="utf-8-sig")
lt.to_csv("output/review_lift_table.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([
    {"模型": "A_含履约结果", "AUC": round(roc_auc_score(y[teA], pA), 4),
     "KS": round(ks(y[teA], pA), 4), "PR_AUC": round(average_precision_score(y[teA], pA), 4)},
    {"模型": "B_仅下单时特征", "AUC": round(roc_auc_score(y[teB], pB), 4),
     "KS": round(ks(y[teB], pB), 4), "PR_AUC": round(average_precision_score(y[teB], pB), 4)},
]).to_csv("output/review_model_metrics.csv", index=False, encoding="utf-8-sig")
g.to_frame("占比_pct").to_csv("output/review_driver_groups.csv", encoding="utf-8-sig")
print("\n结果已输出至 output/ ✅")
