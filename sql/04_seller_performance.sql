-- ============================================================================
-- 04_seller_performance.sql   卖家生态与质量分层
-- 业务问题：平台的卖家结构是怎样的？谁在拖累履约和口碑？该治理谁？
-- 技术点：多指标分层、集中度分析、贡献度归因、按业务阈值过滤噪声
-- ============================================================================

-- @query: 卖家生态总览与集中度
WITH s AS (
    SELECT
        seller_id,
        COUNT(DISTINCT order_id)                                    AS orders,
        SUM(item_amount)                                            AS gmv,
        COUNT(DISTINCT category_en)                                 AS categories,
        AVG(review_score)                                           AS avg_score
    FROM dwd_order_item
    GROUP BY seller_id
),
r AS (
    SELECT *,
        ROW_NUMBER() OVER (ORDER BY gmv DESC)                       AS rn,
        COUNT(*) OVER ()                                            AS total,
        SUM(gmv) OVER (ORDER BY gmv DESC
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum,
        SUM(gmv) OVER ()                                            AS total_gmv
    FROM s
)
SELECT
    MAX(total)                                                      AS 卖家总数,
    MIN(rn) FILTER (WHERE cum * 100.0 / total_gmv >= 50)            AS 贡献50pct的卖家数,
    MIN(rn) FILTER (WHERE cum * 100.0 / total_gmv >= 80)            AS 贡献80pct的卖家数,
    ROUND(MIN(rn) FILTER (WHERE cum * 100.0 / total_gmv >= 80)
          * 100.0 / MAX(total), 2)                                  AS 贡献80pct的卖家占比_pct,
    ROUND(AVG(orders), 2)                                           AS 卖家平均订单数,
    ROUND(MEDIAN(orders), 0)                                        AS 卖家订单数中位数
FROM r;


-- @query: 卖家规模分层与质量表现
WITH s AS (
    SELECT
        i.seller_id,
        COUNT(DISTINCT i.order_id)                                  AS orders,
        SUM(i.item_amount)                                          AS gmv,
        AVG(o.review_score)                                         AS avg_score,
        AVG(o.is_late_int)                AS late_rate,
        AVG(o.seller_handling_days)                                 AS handling_days,
        AVG(i.freight_value / NULLIF(i.price, 0))                   AS freight_ratio
    FROM dwd_order_item i
    JOIN dwd_order o ON i.order_id = o.order_id
    GROUP BY i.seller_id
)
SELECT
    CASE WHEN orders >= 500 THEN '1_头部(500单+)'
         WHEN orders >= 100 THEN '2_腰部(100-499单)'
         WHEN orders >= 20  THEN '3_长尾(20-99单)'
         ELSE '4_零星(<20单)' END                                   AS 卖家分层,
    COUNT(*)                                                        AS 卖家数,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)              AS 卖家占比_pct,
    SUM(orders)                                                     AS 订单数,
    ROUND(SUM(gmv) / 1e4, 1)                                        AS GMV_万,
    ROUND(SUM(gmv) * 100.0 / SUM(SUM(gmv)) OVER (), 2)              AS GMV占比_pct,
    ROUND(AVG(avg_score), 3)                                        AS 平均评分,
    ROUND(AVG(late_rate) * 100, 2)                                  AS 平均延迟率_pct,
    ROUND(AVG(handling_days), 2)                                    AS 平均备货天数
FROM s
GROUP BY 1
ORDER BY 1;


-- @query: 问题卖家定位（延迟率高且有一定体量）
-- 过滤条件：至少 50 单，保证统计意义；小卖家的高延迟率可能只是偶然
WITH s AS (
    SELECT
        i.seller_id,
        MAX(i.seller_state)                                         AS seller_state,
        COUNT(DISTINCT i.order_id)                                  AS orders,
        SUM(i.item_amount)                                          AS gmv,
        AVG(o.review_score)                                         AS avg_score,
        AVG(o.is_late_int)                AS late_rate,
        AVG(o.is_bad_int)          AS bad_rate,
        AVG(o.seller_handling_days)                                 AS handling_days,
        AVG(o.delivery_days)                                        AS delivery_days
    FROM dwd_order_item i
    JOIN dwd_order o ON i.order_id = o.order_id
    WHERE o.delivered_ts IS NOT NULL
    GROUP BY i.seller_id
    HAVING COUNT(DISTINCT i.order_id) >= 50
)
SELECT
    LEFT(seller_id, 8) || '…'                                       AS 卖家,
    seller_state                                                    AS 所在州,
    orders                                                          AS 订单数,
    ROUND(gmv / 1e4, 1)                                             AS GMV_万,
    ROUND(late_rate * 100, 2)                                       AS 延迟率_pct,
    ROUND(bad_rate * 100, 2)                                        AS 差评率_pct,
    ROUND(avg_score, 3)                                             AS 平均评分,
    ROUND(handling_days, 2)                                         AS 备货天数,
    ROUND(delivery_days, 1)                                         AS 送达天数
FROM s
ORDER BY late_rate DESC
LIMIT 15;


-- @query: 问题卖家的治理价值测算
-- 如果把延迟率最高的那批卖家治理到平台中位水平，能挽回多少差评
WITH s AS (
    SELECT
        i.seller_id,
        COUNT(DISTINCT i.order_id)                                  AS orders,
        AVG(o.is_late_int)                AS late_rate,
        AVG(o.is_bad_int)          AS bad_rate
    FROM dwd_order_item i
    JOIN dwd_order o ON i.order_id = o.order_id
    WHERE o.delivered_ts IS NOT NULL
    GROUP BY i.seller_id
    HAVING COUNT(DISTINCT i.order_id) >= 50
),
tagged AS (
    SELECT *,
        NTILE(10) OVER (ORDER BY late_rate DESC)                    AS decile
    FROM s
),
benchmark AS (SELECT MEDIAN(bad_rate) AS med_bad FROM s)
SELECT
    t.decile                                                        AS 延迟率十分位,
    COUNT(*)                                                        AS 卖家数,
    SUM(t.orders)                                                   AS 订单数,
    ROUND(AVG(t.late_rate) * 100, 2)                                AS 平均延迟率_pct,
    ROUND(AVG(t.bad_rate) * 100, 2)                                 AS 平均差评率_pct,
    ROUND(SUM(t.orders * t.bad_rate), 0)                            AS 差评数,
    -- 若该组差评率降至全体中位数，可减少的差评量
    ROUND(SUM(t.orders * GREATEST(t.bad_rate - b.med_bad, 0)), 0)   AS 可减少差评数
FROM tagged t CROSS JOIN benchmark b
GROUP BY t.decile
ORDER BY t.decile;


-- @query: 卖家所在州的履约表现（卖家地理集中度）
SELECT
    seller_state                                                    AS 卖家所在州,
    COUNT(DISTINCT seller_id)                                       AS 卖家数,
    COUNT(DISTINCT order_id)                                        AS 订单数,
    ROUND(SUM(item_amount) / 1e4, 1)                                AS GMV_万,
    ROUND(SUM(item_amount) * 100.0 / SUM(SUM(item_amount)) OVER (), 2) AS GMV占比_pct,
    ROUND(AVG(review_score), 3)                                     AS 平均评分,
    ROUND(AVG(is_late_int) * 100, 2)      AS 延迟率_pct,
    ROUND(AVG(delivery_days), 1)                                    AS 平均送达天数
FROM dwd_order_item
WHERE seller_state IS NOT NULL
GROUP BY seller_state
HAVING COUNT(DISTINCT order_id) >= 300
ORDER BY GMV_万 DESC
LIMIT 10;


-- @query: 品类的履约难度与口碑风险
-- 大件重货天然更难履约，这类品类的差评未必是卖家的错，需要区别对待
SELECT
    category_en                                                     AS 品类,
    COUNT(DISTINCT order_id)                                        AS 订单数,
    ROUND(AVG(product_weight_g) / 1000.0, 2)                        AS 平均重量_kg,
    ROUND(AVG(product_volume_cm3) / 1000.0, 1)                      AS 平均体积_升,
    ROUND(AVG(freight_value), 2)                                    AS 平均运费,
    ROUND(AVG(freight_value / NULLIF(price, 0)) * 100, 1)           AS 运费价格比_pct,
    ROUND(AVG(delivery_days), 1)                                    AS 平均送达天数,
    ROUND(AVG(is_late_int) * 100, 2)      AS 延迟率_pct,
    ROUND(AVG(review_score), 3)                                     AS 平均评分
FROM dwd_order_item
GROUP BY category_en
HAVING COUNT(DISTINCT order_id) >= 1000
ORDER BY 延迟率_pct DESC
LIMIT 15;
