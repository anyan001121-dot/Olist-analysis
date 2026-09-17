-- ============================================================================
-- 01_business_overview.sql   业务全景与增长拆解
-- 业务问题：平台长什么样？增长从哪来？钱主要在哪些品类和地区？
-- 技术点：窗口函数算环比/累计占比、GMV 乘法因子分解、帕累托集中度
-- ============================================================================

-- @query: 整体业务概览
SELECT
    COUNT(DISTINCT order_id)                                        AS 订单数,
    COUNT(DISTINCT customer_unique_id)                              AS 客户数,
    ROUND(SUM(order_amount) / 1e6, 2)                               AS GMV_百万雷亚尔,
    ROUND(AVG(order_amount), 2)                                     AS 客单价,
    ROUND(SUM(freight_amount) * 100.0 / SUM(order_amount), 2)       AS 运费占比_pct,
    ROUND(AVG(item_count), 3)                                       AS 平均件数,
    ROUND(AVG(review_score), 3)                                     AS 平均评分,
    ROUND(AVG(delivery_days), 2)                                    AS 平均送达天数,
    ROUND(AVG(is_late_int) * 100, 2)      AS 延迟送达率_pct,
    ROUND(AVG(is_bad_int) * 100, 2) AS 差评率_pct
FROM dwd_order;


-- @query: 月度 GMV 趋势与环比
SELECT
    STRFTIME(purchase_month, '%Y-%m')                               AS 月份,
    COUNT(DISTINCT order_id)                                        AS 订单数,
    COUNT(DISTINCT customer_unique_id)                              AS 下单客户数,
    ROUND(SUM(order_amount) / 1e4, 1)                               AS GMV_万,
    ROUND(AVG(order_amount), 1)                                     AS 客单价,
    ROUND((SUM(order_amount) / LAG(SUM(order_amount))
           OVER (ORDER BY purchase_month) - 1) * 100, 2)            AS GMV环比_pct,
    -- 3 个月移动平均，抹平促销月波动看真实趋势
    ROUND(AVG(SUM(order_amount)) OVER (ORDER BY purchase_month
          ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) / 1e4, 1)       AS GMV_3月均值_万,
    ROUND(AVG(review_score), 3)                                     AS 平均评分,
    ROUND(AVG(is_late_int) * 100, 2)      AS 延迟率_pct
FROM dwd_order
GROUP BY purchase_month
ORDER BY purchase_month;


-- @query: GMV 增长的因子分解（新客 vs 老客）
-- 口径：客户首单所在月之后的订单算老客订单
WITH first_order AS (
    SELECT customer_unique_id, MIN(purchase_month) AS cohort_month
    FROM dwd_order GROUP BY 1
)
SELECT
    STRFTIME(o.purchase_month, '%Y-%m')                             AS 月份,
    ROUND(SUM(CASE WHEN o.purchase_month = f.cohort_month
                   THEN o.order_amount ELSE 0 END) / 1e4, 1)        AS 新客GMV_万,
    ROUND(SUM(CASE WHEN o.purchase_month > f.cohort_month
                   THEN o.order_amount ELSE 0 END) / 1e4, 1)        AS 老客GMV_万,
    ROUND(SUM(CASE WHEN o.purchase_month > f.cohort_month
                   THEN o.order_amount ELSE 0 END) * 100.0
          / SUM(o.order_amount), 2)                                 AS 老客GMV占比_pct,
    COUNT(DISTINCT CASE WHEN o.purchase_month = f.cohort_month
                        THEN o.customer_unique_id END)              AS 新客数,
    COUNT(DISTINCT CASE WHEN o.purchase_month > f.cohort_month
                        THEN o.customer_unique_id END)              AS 老客数
FROM dwd_order o
JOIN first_order f ON o.customer_unique_id = f.customer_unique_id
GROUP BY o.purchase_month
ORDER BY o.purchase_month;


-- @query: 品类 GMV 贡献与帕累托集中度
WITH cat AS (
    SELECT
        category_en,
        SUM(item_amount)                                            AS gmv,
        COUNT(DISTINCT order_id)                                    AS orders,
        AVG(price)                                                  AS avg_price,
        AVG(review_score)                                           AS avg_score
    FROM dwd_order_item
    GROUP BY category_en
)
SELECT
    category_en                                                     AS 品类,
    ROUND(gmv / 1e4, 1)                                             AS GMV_万,
    ROUND(gmv * 100.0 / SUM(gmv) OVER (), 2)                        AS GMV占比_pct,
    ROUND(SUM(gmv) OVER (ORDER BY gmv DESC) * 100.0
          / SUM(gmv) OVER (), 2)                                    AS 累计占比_pct,
    orders                                                          AS 订单数,
    ROUND(avg_price, 1)                                             AS 均价,
    ROUND(avg_score, 3)                                             AS 平均评分,
    ROW_NUMBER() OVER (ORDER BY gmv DESC)                           AS 排名
FROM cat
ORDER BY gmv DESC
LIMIT 15;


-- @query: 品类集中度（多少品类撑起 80% GMV）
WITH cat AS (
    SELECT category_en, SUM(item_amount) AS gmv FROM dwd_order_item GROUP BY 1
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (ORDER BY gmv DESC)                       AS rk,
        SUM(gmv) OVER (ORDER BY gmv DESC) * 100.0 / SUM(gmv) OVER () AS cum_pct,
        COUNT(*) OVER ()                                            AS total_cat
    FROM cat
)
SELECT
    MIN(rk) FILTER (WHERE cum_pct >= 50)                            AS 撑起50pct需品类数,
    MIN(rk) FILTER (WHERE cum_pct >= 80)                            AS 撑起80pct需品类数,
    MIN(rk) FILTER (WHERE cum_pct >= 90)                            AS 撑起90pct需品类数,
    MAX(total_cat)                                                  AS 品类总数
FROM ranked;


-- @query: 地域分布（州级）
SELECT
    customer_state                                                  AS 州,
    COUNT(DISTINCT order_id)                                        AS 订单数,
    ROUND(SUM(order_amount) / 1e4, 1)                               AS GMV_万,
    ROUND(SUM(order_amount) * 100.0 / SUM(SUM(order_amount)) OVER (), 2) AS GMV占比_pct,
    ROUND(AVG(order_amount), 1)                                     AS 客单价,
    ROUND(AVG(freight_ratio) * 100, 2)                              AS 运费占比_pct,
    ROUND(AVG(delivery_days), 1)                                    AS 平均送达天数,
    ROUND(AVG(is_late_int) * 100, 2)      AS 延迟率_pct,
    ROUND(AVG(review_score), 3)                                     AS 平均评分
FROM dwd_order
GROUP BY customer_state
HAVING COUNT(*) >= 300
ORDER BY GMV_万 DESC
LIMIT 12;


-- @query: 支付方式分布与分期行为
SELECT
    main_payment_type                                               AS 支付方式,
    COUNT(*)                                                        AS 订单数,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)              AS 占比_pct,
    ROUND(AVG(order_amount), 1)                                     AS 客单价,
    ROUND(AVG(max_installments), 2)                                 AS 平均分期数,
    ROUND(AVG(review_score), 3)                                     AS 平均评分
FROM dwd_order
WHERE main_payment_type IS NOT NULL
GROUP BY main_payment_type
ORDER BY 订单数 DESC;


-- @query: 分期数与客单价的关系（信用卡订单）
SELECT
    CASE WHEN max_installments <= 1 THEN '1_不分期'
         WHEN max_installments <= 3 THEN '2_2-3期'
         WHEN max_installments <= 6 THEN '3_4-6期'
         WHEN max_installments <= 10 THEN '4_7-10期'
         ELSE '5_10期以上' END                                      AS 分期档,
    COUNT(*)                                                        AS 订单数,
    ROUND(AVG(order_amount), 1)                                     AS 客单价,
    ROUND(MEDIAN(order_amount), 1)                                  AS 客单价中位数,
    ROUND(AVG(review_score), 3)                                     AS 平均评分
FROM dwd_order
WHERE main_payment_type = 'credit_card' AND max_installments IS NOT NULL
GROUP BY 1
ORDER BY 1;
