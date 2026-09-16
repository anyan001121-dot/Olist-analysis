# -*- coding: utf-8 -*-
"""
图表输出与结论导出
生成 README 用的 PNG，并把所有核心结论导出为 JSON 供 HTML 报告页使用，
保证报告里每个数字都来自真实计算结果而非手写。

Usage:
    python 04_visualize.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
FIG = os.path.join(BASE_DIR, "output", "figures")
os.makedirs(FIG, exist_ok=True)

for cand in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "red": "#d03b3b", "muted": "#898781", "good": "#0ca30c"}
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

con = duckdb.connect(":memory:")
con.execute(open("sql/00_create_views.sql", encoding="utf-8").read())
R = {}


def q(s):
    return con.execute(s).fetchdf()


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print(f"  ✓ {name}")


print("生成图表 ...")

# ------------------------------------------------ 概览
ov = q("""
SELECT COUNT(DISTINCT order_id) orders, COUNT(DISTINCT customer_unique_id) customers,
 SUM(order_amount)/1e6 gmv_m, AVG(order_amount) aov, AVG(review_score) score,
 AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END)*100 late_rate,
 AVG(CASE WHEN is_bad_review THEN 1.0 ELSE 0 END)*100 bad_rate,
 AVG(delivery_days) del_days FROM dwd_order
""").iloc[0]
R["overview"] = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else int(v))
                 for k, v in ov.items()}
R["sellers"] = int(q("SELECT COUNT(DISTINCT seller_id) n FROM dwd_order_item").iloc[0].n)
R["items"] = int(q("SELECT COUNT(*) n FROM dwd_order_item").iloc[0].n)

# ------------------------------------------------ 1 延迟 vs 评分
late = q("""
SELECT is_late, COUNT(*) n, AVG(review_score) score,
 AVG(CASE WHEN is_bad_review THEN 1.0 ELSE 0 END)*100 bad,
 AVG(CASE WHEN is_top_review THEN 1.0 ELSE 0 END)*100 top5,
 AVG(delivery_days) avg_days
FROM dwd_order WHERE delivered_ts IS NOT NULL AND review_score IS NOT NULL
GROUP BY 1 ORDER BY 1
""")
R["late_vs_review"] = late.round(3).to_dict("records")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
labels = ["按时/提前送达", "延迟送达"]
bad = late.sort_values("is_late")["bad"].values
n = late.sort_values("is_late")["n"].values
bars = ax.bar(labels, bad, color=[C["blue"], C["red"]], width=0.5)
ax.set_ylabel("差评率 (%)")
ax.set_title("延迟送达对差评率的影响", fontsize=13, fontweight="bold", pad=12)
for b, v, cnt in zip(bars, bad, n):
    ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.1f}%\n({cnt:,} 单)",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(bad) * 1.28)
ax.annotate(f"{bad[1]/bad[0]:.1f} 倍", xy=(1, bad[1]/2), fontsize=15,
            color="white", ha="center", fontweight="bold")
save(fig, "01_late_vs_review.png")

# ------------------------------------------------ 2 剂量反应
dose = q("""
SELECT CASE WHEN days_early>=15 THEN '提前15天+' WHEN days_early>=8 THEN '提前8-14天'
 WHEN days_early>=3 THEN '提前3-7天' WHEN days_early>=0 THEN '提前0-2天'
 WHEN days_early>=-3 THEN '延迟1-3天' WHEN days_early>=-7 THEN '延迟4-7天'
 WHEN days_early>=-15 THEN '延迟8-15天' ELSE '延迟15天+' END bucket,
 CASE WHEN days_early>=15 THEN 1 WHEN days_early>=8 THEN 2 WHEN days_early>=3 THEN 3
 WHEN days_early>=0 THEN 4 WHEN days_early>=-3 THEN 5 WHEN days_early>=-7 THEN 6
 WHEN days_early>=-15 THEN 7 ELSE 8 END ord,
 COUNT(*) n, AVG(review_score) score,
 AVG(CASE WHEN is_bad_review THEN 1.0 ELSE 0 END)*100 bad
FROM dwd_order WHERE delivered_ts IS NOT NULL AND review_score IS NOT NULL
GROUP BY 1,2 ORDER BY ord
""")
R["dose_response"] = dose.round(3).to_dict("records")

fig, ax = plt.subplots(figsize=(9.5, 4.4))
colors = [C["blue"]] * 4 + [C["red"]] * 4
bars = ax.bar(dose["bucket"], dose["bad"], color=colors, width=0.62)
ax.axvline(3.5, color="#334155", ls="--", lw=1.3)
ax.text(3.55, ax.get_ylim()[1]*0.9, "承诺时间分界", fontsize=9, color="#334155")
ax.set_ylabel("差评率 (%)")
ax.set_title("差评率随送达时效的剂量反应 —— 一旦违诺即断崖", fontsize=13,
             fontweight="bold", pad=12)
for b, v in zip(bars, dose["bad"]):
    ax.text(b.get_x()+b.get_width()/2, v+1.2, f"{v:.1f}", ha="center", fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=22, ha="right", fontsize=9)
ax.set_ylim(0, dose["bad"].max()*1.18)
save(fig, "02_dose_response.png")

# ------------------------------------------------ 3 月度趋势
mon = q("""
SELECT STRFTIME(purchase_month,'%Y-%m') mon, COUNT(*) orders,
 SUM(order_amount)/1e4 gmv_w, AVG(review_score) score,
 AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END)*100 late_rate,
 AVG(delivery_days) del_days, AVG(carrier_transit_days) transit,
 AVG(seller_handling_days) handling
FROM dwd_order GROUP BY 1 ORDER BY 1
""")
R["monthly"] = mon.round(3).to_dict("records")

fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1]})
ax = axes[0]
ax.bar(mon["mon"], mon["gmv_w"], color=C["blue"], width=0.6)
ax.set_ylabel("GMV（万雷亚尔）")
ax.set_title("月度 GMV 与履约质量", fontsize=13, fontweight="bold", pad=10)
ax = axes[1]
ax.plot(mon["mon"], mon["late_rate"], marker="o", lw=2, color=C["red"], label="延迟率 %")
ax2 = ax.twinx()
ax2.plot(mon["mon"], mon["score"], marker="s", lw=2, color=C["aqua"], label="平均评分")
ax2.grid(False)
ax.set_ylabel("延迟率 (%)", color=C["red"])
ax2.set_ylabel("平均评分", color=C["aqua"])
peak = mon.loc[mon["late_rate"].idxmax()]
ax.annotate(f"{peak['mon']}\n延迟率 {peak['late_rate']:.1f}%",
            xy=(peak["mon"], peak["late_rate"]), xytext=(-60, 10),
            textcoords="offset points", color=C["red"], fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["red"]))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
save(fig, "03_monthly_trend.png")

# ------------------------------------------------ 4 延迟责任拆解
seg = q("""
SELECT CASE WHEN is_late THEN '延迟送达' ELSE '按时/提前' END grp,
 AVG(DATE_DIFF('hour',purchase_ts,approved_ts))/24.0 approve,
 AVG(seller_handling_days) handling, AVG(carrier_transit_days) transit,
 AVG(delivery_days) total_days
FROM dwd_order WHERE delivered_ts IS NOT NULL AND carrier_ts IS NOT NULL
 AND approved_ts IS NOT NULL GROUP BY 1 ORDER BY 1
""")
R["delivery_legs"] = seg.round(3).to_dict("records")
attr = q("""
WITH b AS (SELECT MEDIAN(seller_handling_days) ms, MEDIAN(carrier_transit_days) mc
 FROM dwd_order WHERE NOT is_late AND delivered_ts IS NOT NULL AND carrier_ts IS NOT NULL),
l AS (SELECT o.seller_handling_days-b.ms se, o.carrier_transit_days-b.mc ce
 FROM dwd_order o CROSS JOIN b WHERE o.is_late AND o.delivered_ts IS NOT NULL
 AND o.carrier_ts IS NOT NULL)
SELECT AVG(se) seller_excess, AVG(ce) carrier_excess,
 AVG(se)*100.0/(AVG(se)+AVG(ce)) seller_pct, AVG(ce)*100.0/(AVG(se)+AVG(ce)) carrier_pct
FROM l""").iloc[0]
R["delay_attribution"] = {k: round(float(v), 2) for k, v in attr.items()}

fig, ax = plt.subplots(figsize=(9, 3.6))
s = seg.set_index("grp")
ypos = [0, 1]
legs = [("支付审核", "approve", C["muted"]), ("卖家备货发货", "handling", C["orange"]),
        ("承运商运输", "transit", C["blue"])]
left = np.zeros(2)
order_idx = ["按时/提前", "延迟送达"]
for name, col, color in legs:
    vals = s.loc[order_idx, col].values
    ax.barh(ypos, vals, left=left, color=color, height=0.5, label=name)
    for yp, v, l0 in zip(ypos, vals, left):
        if v > 1:
            ax.text(l0 + v/2, yp, f"{v:.1f}天", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    left += vals
ax.set_yticks(ypos); ax.set_yticklabels(order_idx)
ax.set_xlabel("天数")
ax.set_title(f"订单全链路时长拆解 —— 延迟订单的超时 {attr.carrier_pct:.0f}% 来自运输段",
             fontsize=12.5, fontweight="bold", pad=12)
ax.legend(frameon=False, ncol=3, loc="lower right")
save(fig, "04_delivery_legs.png")

# ------------------------------------------------ 5 PSM
if os.path.exists("output/psm_att_results.csv"):
    att = pd.read_csv("output/psm_att_results.csv")
    R["psm_att"] = att.to_dict("records")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = [C["muted"], C["blue"], C["aqua"], C["orange"]]
    bars = ax.barh(att["方法"], att["ATT_pp"], color=cols, height=0.55)
    ax.set_xlabel("差评率提升（百分点）")
    ax.set_title("四种估计方法的因果效应一致", fontsize=12.5, fontweight="bold", pad=12)
    for b, v in zip(bars, att["ATT_pp"]):
        ax.text(v + 0.5, b.get_y()+b.get_height()/2, f"+{v:.2f}pp", va="center",
                fontsize=10, fontweight="bold")
    ax.set_xlim(0, att["ATT_pp"].max()*1.18)
    save(fig, "05_psm_att.png")

if os.path.exists("output/psm_balance.csv"):
    bal = pd.read_csv("output/psm_balance.csv")
    R["psm_balance"] = {
        "n_cov": int(len(bal)),
        "before_gt01": int((bal["匹配前SMD"] > 0.1).sum()),
        "after_gt01": int((bal["匹配后SMD"] > 0.1).sum()),
        "max_before": round(float(bal["匹配前SMD"].max()), 4),
        "max_after": round(float(bal["匹配后SMD"].max()), 4),
        "top": bal.nlargest(8, "改善").round(4).to_dict("records"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(bal["匹配前SMD"], bal["匹配后SMD"], s=18, alpha=0.6, color=C["blue"])
    lim = bal["匹配前SMD"].max() * 1.08
    ax.plot([0, lim], [0, lim], ls="--", color=C["muted"], lw=1)
    ax.axhline(0.1, color=C["red"], ls=":", lw=1.4)
    ax.axvline(0.1, color=C["red"], ls=":", lw=1.4)
    ax.text(lim*0.62, 0.105, "平衡阈值 SMD=0.1", color=C["red"], fontsize=9)
    ax.set_xlabel("匹配前 SMD"); ax.set_ylabel("匹配后 SMD")
    ax.set_title("协变量平衡性：匹配前 vs 匹配后", fontsize=12.5,
                 fontweight="bold", pad=12)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    save(fig, "06_psm_balance.png")

# ------------------------------------------------ 6 复购口径
rep = q("""
WITH bo AS (SELECT order_customer_id k, COUNT(DISTINCT order_id) n FROM dwd_order GROUP BY 1),
bp AS (SELECT customer_unique_id k, COUNT(DISTINCT order_id) n FROM dwd_order GROUP BY 1),
seq AS (SELECT customer_unique_id, purchase_date,
 LAG(purchase_date) OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) pd FROM dwd_order),
rr AS (SELECT DISTINCT customer_unique_id FROM seq WHERE pd IS NOT NULL
 AND DATE_DIFF('day',pd,purchase_date)>=1)
SELECT
 (SELECT COUNT(*) FILTER (WHERE n>=2)*100.0/COUNT(*) FROM bo) wrong_key,
 (SELECT COUNT(*) FILTER (WHERE n>=2)*100.0/COUNT(*) FROM bp) surface,
 (SELECT COUNT(*) FROM rr)*100.0/(SELECT COUNT(*) FROM bp) real_rate
""").iloc[0]
R["repurchase"] = {k: round(float(v), 3) for k, v in rep.items()}

fig, ax = plt.subplots(figsize=(8, 3.8))
names = ["用 customer_id\n（订单级键，错）", "用 customer_unique_id\n（自然人）",
         "再剔除同日拆单\n（真实复购）"]
vals = [rep.wrong_key, rep.surface, rep.real_rate]
bars = ax.bar(names, vals, color=[C["red"], C["orange"], C["blue"]], width=0.5)
ax.set_ylabel("复购率 (%)")
ax.set_title("同一份数据，三种口径下的复购率", fontsize=13, fontweight="bold", pad=12)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.06, f"{v:.3f}%", ha="center",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, max(vals)*1.25)
plt.setp(ax.get_xticklabels(), fontsize=9)
save(fig, "07_repurchase_definition.png")

# ------------------------------------------------ 7 州级
st = q("""
SELECT customer_state st, COUNT(*) n, AVG(delivery_days) avg_days,
 AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END)*100 late_rate, AVG(review_score) score,
 SUM(order_amount)/1e4 gmv_w
FROM dwd_order WHERE delivered_ts IS NOT NULL GROUP BY 1 HAVING COUNT(*)>=300
ORDER BY avg_days DESC""")
R["states"] = st.round(3).to_dict("records")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(st["avg_days"], st["score"], s=np.sqrt(st["n"])*2.2,
                c=st["late_rate"], cmap="YlOrRd", edgecolor="white", linewidth=0.8)
for _, r in st.iterrows():
    if r["n"] > 1800 or r["avg_days"] > 20:
        ax.annotate(r["st"], (r["avg_days"], r["score"]), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
cb = plt.colorbar(sc, ax=ax); cb.set_label("延迟率 (%)")
ax.set_xlabel("平均送达天数"); ax.set_ylabel("平均评分")
ax.set_title("各州送达时效与评分（气泡大小=订单量）", fontsize=12.5,
             fontweight="bold", pad=12)
save(fig, "08_state_delivery.png")

# ------------------------------------------------ 8 预警模型
if os.path.exists("output/review_lift_table.csv"):
    lt = pd.read_csv("output/review_lift_table.csv")
    R["warn_lift"] = lt.to_dict("records")
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(lt["decile"], lt["提升度"], color=C["blue"], width=0.6)
    ax.axhline(1, color=C["red"], ls="--", lw=1.2)
    ax.text(9.2, 1.05, "基线 1.00×", color=C["red"], fontsize=9)
    ax.set_ylabel("提升度 (Lift)")
    ax.set_xlabel("按预测差评概率从高到低十等分")
    ax.set_title("下单时预警模型的十分位提升表", fontsize=12.5, fontweight="bold", pad=12)
    for b, v in zip(bars, lt["提升度"]):
        ax.text(b.get_x()+b.get_width()/2, v+0.04, f"{v:.2f}", ha="center", fontsize=9)
    save(fig, "09_warning_lift.png")
if os.path.exists("output/review_model_metrics.csv"):
    R["warn_metrics"] = pd.read_csv("output/review_model_metrics.csv").to_dict("records")
if os.path.exists("output/review_driver_groups.csv"):
    R["driver_groups"] = pd.read_csv("output/review_driver_groups.csv").to_dict("records")

# ------------------------------------------------ 9 其他结论
R["categories"] = q("""
SELECT category_en cat, SUM(item_amount)/1e4 gmv_w, COUNT(DISTINCT order_id) orders,
 AVG(review_score) score, AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END)*100 late_rate
FROM dwd_order_item GROUP BY 1 ORDER BY gmv_w DESC LIMIT 12
""").round(3).to_dict("records")

R["new_vs_old"] = q("""
WITH f AS (SELECT customer_unique_id, MIN(purchase_month) cm FROM dwd_order GROUP BY 1)
SELECT STRFTIME(o.purchase_month,'%Y-%m') mon,
 SUM(CASE WHEN o.purchase_month>f.cm THEN o.order_amount ELSE 0 END)*100.0
   /SUM(o.order_amount) old_pct
FROM dwd_order o JOIN f ON o.customer_unique_id=f.customer_unique_id
GROUP BY 1 ORDER BY 1""").round(3).to_dict("records")

R["seller_tiers"] = q("""
WITH s AS (SELECT i.seller_id, COUNT(DISTINCT i.order_id) orders,
 SUM(i.item_amount) gmv, AVG(o.review_score) score,
 AVG(CASE WHEN o.is_late THEN 1.0 ELSE 0 END) late
 FROM dwd_order_item i JOIN dwd_order o ON i.order_id=o.order_id GROUP BY 1)
SELECT CASE WHEN orders>=500 THEN '头部(500单+)' WHEN orders>=100 THEN '腰部(100-499)'
 WHEN orders>=20 THEN '长尾(20-99)' ELSE '零星(<20)' END tier,
 COUNT(*) sellers, SUM(gmv)/1e4 gmv_w, AVG(score) score, AVG(late)*100 late_rate
FROM s GROUP BY 1 ORDER BY MIN(orders) DESC""").round(3).to_dict("records")

R["promise"] = q("""
SELECT CASE WHEN promised_days<=10 THEN '承诺10天内' WHEN promised_days<=20 THEN '11-20天'
 WHEN promised_days<=30 THEN '21-30天' ELSE '30天以上' END bucket,
 CASE WHEN promised_days<=10 THEN 1 WHEN promised_days<=20 THEN 2
 WHEN promised_days<=30 THEN 3 ELSE 4 END ord,
 COUNT(*) n, AVG(delivery_days) actual, AVG(days_early) buffer,
 AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END)*100 late_rate, AVG(review_score) score
FROM dwd_order WHERE delivered_ts IS NOT NULL AND promised_days IS NOT NULL
GROUP BY 1,2 ORDER BY ord""").round(3).to_dict("records")

R["cross_state"] = q("""
WITH oi AS (SELECT order_id, MAX(CASE WHEN seller_state=customer_state THEN 0 ELSE 1 END) cs
 FROM dwd_order_item WHERE seller_state IS NOT NULL GROUP BY 1)
SELECT CASE WHEN cs=1 THEN '跨州配送' ELSE '同州配送' END t, COUNT(*) n,
 AVG(o.delivery_days) avg_days, AVG(CASE WHEN o.is_late THEN 1.0 ELSE 0 END)*100 late_rate,
 AVG(o.review_score) score
FROM dwd_order o JOIN oi ON o.order_id=oi.order_id
WHERE o.delivered_ts IS NOT NULL GROUP BY 1""").round(3).to_dict("records")


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


with open("output/report_data.json", "w", encoding="utf-8") as f:
    json.dump(clean(R), f, ensure_ascii=False, indent=2, allow_nan=False)
print(f"\n图表输出至 {FIG}")
print("结论数据导出 output/report_data.json ✅")
con.close()
