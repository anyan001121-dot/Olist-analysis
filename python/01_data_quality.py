# -*- coding: utf-8 -*-
"""
数据获取与质量探查
================================================================================
真实数据和模拟数据最大的区别是：**它会主动骗你**。
缺失、重复、一对多膨胀、语义不符的主键——这些都不会抛异常，
只会静默地让你的结论出错。

所以在写任何一行分析 SQL 之前，先做这一步，并把发现固化成
sql/00_create_views.sql 里的口径约定。

本脚本做两件事：
    1. 从 Olist 官方 GitHub 仓库下载 9 张原始表（已存在则跳过）
    2. 逐项探查数据质量，输出需要在口径层处理的问题清单

Usage:
    python 01_data_quality.py
"""

import os
import sys
import urllib.request
import warnings
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.unicode.east_asian_width", True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

BASE_URL = ("https://raw.githubusercontent.com/olist/work-at-olist-data/"
            "master/datasets")
FILES = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]


def section(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


# ============================================================ 0. 下载
section("0  获取数据")
os.makedirs(DATA_DIR, exist_ok=True)
for f in FILES:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"  已存在  {f:<46} {os.path.getsize(path)/1024/1024:>6.1f} MB")
        continue
    print(f"  下载中  {f} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(f"{BASE_URL}/{f}", path)
        print(f"{os.path.getsize(path)/1024/1024:.1f} MB ✓")
    except Exception as e:
        print(f"失败: {e}")
        sys.exit(1)

con = duckdb.connect(":memory:")
T = {
    "orders": "olist_orders_dataset", "items": "olist_order_items_dataset",
    "pay": "olist_order_payments_dataset", "rev": "olist_order_reviews_dataset",
    "cust": "olist_customers_dataset", "prod": "olist_products_dataset",
    "sell": "olist_sellers_dataset", "trans": "product_category_name_translation",
}
for k, v in T.items():
    con.execute(f"CREATE VIEW {k} AS SELECT * FROM "
                f"read_csv_auto('data/{v}.csv', header=true, sample_size=-1)")


def q(s):
    return con.execute(s).fetchdf()


# ============================================================ 1. 规模
section("1  数据规模与时间范围")
print(q("""
SELECT '订单' 表, COUNT(*) 行数 FROM orders
UNION ALL SELECT '订单明细', COUNT(*) FROM items
UNION ALL SELECT '支付记录', COUNT(*) FROM pay
UNION ALL SELECT '评论', COUNT(*) FROM rev
UNION ALL SELECT '客户', COUNT(*) FROM cust
UNION ALL SELECT '商品', COUNT(*) FROM prod
UNION ALL SELECT '卖家', COUNT(*) FROM sell
""").to_string(index=False))

print("\n时间范围：")
print(q("""SELECT MIN(order_purchase_timestamp) 最早, MAX(order_purchase_timestamp) 最晚
FROM orders""").to_string(index=False))

print("\n月度订单量（检查数据边缘是否稀疏）：")
m = q("""SELECT STRFTIME(order_purchase_timestamp,'%Y-%m') 月份, COUNT(*) 订单数
FROM orders GROUP BY 1 ORDER BY 1""")
print(pd.concat([m.head(5), m.tail(4)]).to_string(index=False))
print("""
  ⚠ 发现一：首尾月份极度稀疏（2016-09 仅 4 单、2016-12 仅 1 单、
     2018-09 仅 16 单、2018-10 仅 4 单），属于数据采集的边缘残留。
     不截断会让首尾月的环比同比完全失真。
     → 口径决定：分析窗口取 2017-01-01 ~ 2018-08-31""")


# ============================================================ 2. 主键语义
section("2  主键语义检查（本数据集最大的坑）")
print(q("""SELECT COUNT(*) 客户表行数, COUNT(DISTINCT customer_id) 唯一customer_id,
COUNT(DISTINCT customer_unique_id) 唯一customer_unique_id FROM cust""").to_string(index=False))
print("""
  ⚠ 发现二：customer_id 数量 == 表行数 == 订单数，说明它是**订单级**的
     一次性键，不是自然人。customer_unique_id 才是。

     用错口径的后果：
""")
print(q("""
WITH bo AS (SELECT c.customer_id k, COUNT(DISTINCT o.order_id) n
            FROM orders o JOIN cust c ON o.customer_id=c.customer_id GROUP BY 1),
     bp AS (SELECT c.customer_unique_id k, COUNT(DISTINCT o.order_id) n
            FROM orders o JOIN cust c ON o.customer_id=c.customer_id GROUP BY 1)
SELECT '用 customer_id' 口径, COUNT(*) 客户数,
       ROUND(COUNT(*) FILTER (WHERE n>=2)*100.0/COUNT(*),3) 复购率_pct FROM bo
UNION ALL
SELECT '用 customer_unique_id', COUNT(*),
       ROUND(COUNT(*) FILTER (WHERE n>=2)*100.0/COUNT(*),3) FROM bp
""").to_string(index=False))
print("     → 口径决定：所有客户级分析一律用 customer_unique_id")


# ============================================================ 3. 一对多
section("3  一对多关系（JOIN 膨胀风险）")
print(q("""
SELECT '每单明细数' 关系, ROUND(AVG(n),3) 均值, MAX(n) 最大 FROM (SELECT order_id,COUNT(*) n FROM items GROUP BY 1)
UNION ALL SELECT '每单支付记录数', ROUND(AVG(n),3), MAX(n) FROM (SELECT order_id,COUNT(*) n FROM pay GROUP BY 1)
UNION ALL SELECT '每单评论数', ROUND(AVG(n),3), MAX(n) FROM (SELECT order_id,COUNT(*) n FROM rev GROUP BY 1)
""").to_string(index=False))

print("\n直接 JOIN 会让 GMV 虚高多少：")
print(q("""
SELECT '① 先聚合再JOIN（正确）' 口径, ROUND(SUM(amt)/1e6,3) GMV_百万
FROM (SELECT order_id, SUM(price+freight_value) amt FROM items GROUP BY 1)
UNION ALL SELECT '② items × payments 直接JOIN',
 ROUND((SELECT SUM(i.price+i.freight_value) FROM items i JOIN pay p ON i.order_id=p.order_id)/1e6,3)
UNION ALL SELECT '③ 再JOIN上 reviews',
 ROUND((SELECT SUM(i.price+i.freight_value) FROM items i JOIN pay p ON i.order_id=p.order_id
        JOIN rev r ON i.order_id=r.order_id)/1e6,3)
""").to_string(index=False))
print("""
  ⚠ 发现三：②相对①虚高 4.57%。三张表都是一对多，直接 JOIN 后 SUM
     会把同一笔金额重复计算。
     → 口径决定：所有金额字段先在子查询里聚合到订单粒度，再向外 JOIN
                （见 sql/00_create_views.sql 的 dwd_order_amount）

  ⚠ 每单最多 3 条评论 → 评论表需按 order_id 去重，否则评分分布被重复计数""")


# ============================================================ 4. 缺失与完整性
section("4  缺失值与外键完整性")
print("订单时间戳缺失：")
print(q("""SELECT COUNT(*) 总订单,
 SUM(order_approved_at IS NULL) 未审核,
 SUM(order_delivered_carrier_date IS NULL) 未交承运商,
 SUM(order_delivered_customer_date IS NULL) 未送达客户 FROM orders""").to_string(index=False))

print("\n外键完整性：")
print(q("""SELECT
 (SELECT COUNT(*) FROM orders o LEFT JOIN items i ON o.order_id=i.order_id WHERE i.order_id IS NULL) 订单无明细,
 (SELECT COUNT(*) FROM orders o LEFT JOIN pay p ON o.order_id=p.order_id WHERE p.order_id IS NULL) 订单无支付,
 (SELECT COUNT(*) FROM items i LEFT JOIN prod p ON i.product_id=p.product_id WHERE p.product_id IS NULL) 明细无商品,
 (SELECT COUNT(*) FROM prod WHERE product_category_name IS NULL) 商品无品类
""").to_string(index=False))
print("""
  → 口径决定：
     · 775 个无明细订单在金额分析中排除（无法计算 GMV）
     · 610 个无品类商品归入「未分类」而非丢弃，否则各品类之和 ≠ 总量
     · 2,965 个未送达订单在时效分析中排除，但保留在 GMV 统计中""")


# ============================================================ 5. 订单状态
section("5  订单状态分布")
print(q("""SELECT order_status 状态, COUNT(*) 数量,
ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER (),2) 占比_pct
FROM orders GROUP BY 1 ORDER BY 2 DESC""").to_string(index=False))
print("""
  → 口径决定：canceled + unavailable 合计 1,234 单（1.24%）未实际成交，
     计入 GMV 会高估，予以排除""")


# ============================================================ 6. 异常值
section("6  数值异常检查")
print(q("""SELECT '商品单价' 字段, ROUND(MIN(price),2) 最小, ROUND(MEDIAN(price),2) 中位,
 ROUND(QUANTILE_CONT(price,0.99),2) P99, ROUND(MAX(price),2) 最大 FROM items
UNION ALL SELECT '运费', ROUND(MIN(freight_value),2), ROUND(MEDIAN(freight_value),2),
 ROUND(QUANTILE_CONT(freight_value,0.99),2), ROUND(MAX(freight_value),2) FROM items
UNION ALL SELECT '支付金额', ROUND(MIN(payment_value),2), ROUND(MEDIAN(payment_value),2),
 ROUND(QUANTILE_CONT(payment_value,0.99),2), ROUND(MAX(payment_value),2) FROM pay
UNION ALL SELECT '分期数', MIN(payment_installments), MEDIAN(payment_installments),
 QUANTILE_CONT(payment_installments,0.99), MAX(payment_installments) FROM pay
""").to_string(index=False))

print("\n时间逻辑异常（送达早于发货等）：")
print(q("""SELECT
 SUM(CASE WHEN order_delivered_customer_date < order_delivered_carrier_date THEN 1 ELSE 0 END) 送达早于交承运,
 SUM(CASE WHEN order_approved_at < order_purchase_timestamp THEN 1 ELSE 0 END) 审核早于下单,
 SUM(CASE WHEN order_delivered_carrier_date < order_purchase_timestamp THEN 1 ELSE 0 END) 发货早于下单
FROM orders""").to_string(index=False))

print("\n支付金额 0 元的订单：")
print(q("""SELECT COUNT(*) 笔数, COUNT(DISTINCT order_id) 订单数
FROM pay WHERE payment_value = 0""").to_string(index=False))


# ============================================================ 7. 小结
section("7  探查结论：需要在口径层处理的问题清单")
print("""
  ①  分析窗口截断到 2017-01 ~ 2018-08          （首尾月份样本过稀）
  ②  客户口径统一用 customer_unique_id          （customer_id 是订单级键）
  ③  金额先聚合到订单粒度再 JOIN                （一对多膨胀 4.57%）
  ④  评论按 order_id 去重取首条                 （每单最多 3 条评论）
  ⑤  排除 canceled / unavailable 订单           （未实际成交）
  ⑥  排除 775 个无明细订单                      （无法计算金额）
  ⑦  无品类商品归入「未分类」                    （保证分项之和 = 总量）
  ⑧  时效分析排除未送达订单，GMV 分析保留        （两类分析口径不同）

  以上全部固化在 sql/00_create_views.sql 中。
  另有一处只有在做复购分析时才会暴露的陷阱（同日拆单），
  见 sql/03_repurchase_ltv.sql 的前两个查询。
""")
con.close()
