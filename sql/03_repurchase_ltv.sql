-- ============================================================================
-- 03_repurchase_ltv.sql   复购、客户价值与口径陷阱
--
-- 业务问题：这个平台的复购到底有多差？差在哪？体验问题是否真的影响了复购？
-- 技术点：口径对比、队列留存、LAG 算复购间隔、帕累托、首单归因
--
-- ⚠️ 本文件第一个查询演示的是本数据集最大的坑，务必先看。
-- ============================================================================

-- @query: 【口径陷阱】复购率的两种算法，结论天差地别
-- Olist 的 customer_id 是订单级的一次性键（99,441 订单 = 99,441 个 id），
-- customer_unique_id 才是自然人（96,096 个）。
-- 用错口径会得出"复购率为 0"这种荒唐结论，而且不会报错——这是最危险的一类 bug。
WITH by_order_key AS (
    SELECT order_customer_id AS k, COUNT(DISTINCT order_id) AS n
    FROM dwd_order GROUP BY 1
),
by_person AS (
    SELECT customer_unique_id AS k, COUNT(DISTINCT order_id) AS n
    FROM dwd_order GROUP BY 1
)
SELECT '❌ 用 customer_id（订单级键）'                               AS 口径,
       COUNT(*)                                                     AS 客户数,
       ROUND(COUNT(*) FILTER (WHERE n >= 2) * 100.0 / COUNT(*), 3)  AS 复购率_pct,
       ROUND(AVG(n), 4)                                             AS 人均订单数
FROM by_order_key
UNION ALL
SELECT '✅ 用 customer_unique_id（自然人）',
       COUNT(*),
       ROUND(COUNT(*) FILTER (WHERE n >= 2) * 100.0 / COUNT(*), 3),
       ROUND(AVG(n), 4)
FROM by_person;


-- @query: 【口径陷阱之二】同日"复购"其实是购物车拆单
-- 上一个查询修好了自然人口径，但复购间隔的 P25 是 0 天——大量"复购"发生在同一天。
-- Olist 是多卖家平台，一个购物车里的商品来自不同卖家时会被拆成多个 order_id。
-- 下面验证：同日复购对中，有多少指向不同卖家（= 拆单而非真复购）。
WITH seq AS (
    SELECT
        customer_unique_id, order_id, purchase_ts, purchase_date,
        ROW_NUMBER() OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS nth,
        LAG(order_id)      OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS prev_order,
        LAG(purchase_ts)   OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS prev_ts,
        LAG(purchase_date) OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS prev_date
    FROM dwd_order
),
same_day AS (
    SELECT order_id, prev_order FROM seq
    WHERE nth >= 2 AND prev_date = purchase_date
),
order_seller AS (
    SELECT DISTINCT order_id, seller_id FROM dwd_order_item
)
SELECT
    (SELECT COUNT(*) FROM seq WHERE nth >= 2)                       AS 复购笔数,
    (SELECT COUNT(*) FROM seq WHERE nth >= 2
       AND DATE_DIFF('minute', prev_ts, purchase_ts) <= 30)         AS 间隔30分钟内,
    COUNT(*)                                                        AS 同日复购对,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM seq WHERE nth >= 2), 1) AS 同日占比_pct,
    SUM(CASE WHEN a.seller_id <> b.seller_id THEN 1 ELSE 0 END)     AS 指向不同卖家,
    ROUND(SUM(CASE WHEN a.seller_id <> b.seller_id THEN 1 ELSE 0 END)
          * 100.0 / COUNT(*), 1)                                    AS 拆单证据占比_pct
FROM same_day m
JOIN order_seller a ON m.order_id  = a.order_id
JOIN order_seller b ON m.prev_order = b.order_id;


-- @query: 修正后的真实复购率（剔除同日拆单）
WITH seq AS (
    SELECT customer_unique_id, purchase_date,
           LAG(purchase_date) OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS prev_date
    FROM dwd_order
),
real_repurchase AS (
    SELECT DISTINCT customer_unique_id FROM seq
    WHERE prev_date IS NOT NULL
      AND DATE_DIFF('day', prev_date, purchase_date) >= 1           -- 至少隔一天
)
SELECT
    (SELECT COUNT(*) FROM dws_customer)                             AS 客户总数,
    (SELECT COUNT(*) FROM dws_customer WHERE order_count >= 2)      AS 表面复购客户数,
    (SELECT ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dws_customer), 3)
     FROM dws_customer WHERE order_count >= 2)                      AS 表面复购率_pct,
    (SELECT COUNT(*) FROM real_repurchase)                          AS 真实复购客户数,
    ROUND((SELECT COUNT(*) FROM real_repurchase) * 100.0
          / (SELECT COUNT(*) FROM dws_customer), 3)                 AS 真实复购率_pct;


-- @query: 购买频次分布与价值集中度
WITH c AS (
    SELECT customer_unique_id, COUNT(DISTINCT order_id) AS n, SUM(order_amount) AS amt
    FROM dwd_order GROUP BY 1
)
SELECT
    CASE WHEN n = 1 THEN '1单' WHEN n = 2 THEN '2单'
         WHEN n BETWEEN 3 AND 4 THEN '3-4单' ELSE '5单及以上' END    AS 购买频次,
    COUNT(*)                                                        AS 客户数,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 3)              AS 人数占比_pct,
    ROUND(SUM(amt) / 1e4, 1)                                        AS GMV_万,
    ROUND(SUM(amt) * 100.0 / SUM(SUM(amt)) OVER (), 2)              AS GMV占比_pct,
    ROUND(AVG(amt), 1)                                              AS 人均消费,
    -- 价值密度 >1 表示该群体人效高于平均
    ROUND((SUM(amt) * 100.0 / SUM(SUM(amt)) OVER ())
        / (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()), 2)            AS 价值密度
FROM c
GROUP BY 1
ORDER BY MIN(n);


-- @query: 消费金额帕累托（TOP 用户贡献度）
WITH c AS (
    SELECT customer_unique_id, SUM(order_amount) AS amt FROM dwd_order GROUP BY 1
),
r AS (
    SELECT *,
        ROW_NUMBER() OVER (ORDER BY amt DESC)                       AS rn,
        COUNT(*) OVER ()                                            AS tot,
        SUM(amt) OVER (ORDER BY amt DESC
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum,
        SUM(amt) OVER ()                                            AS total_amt
    FROM c
)
SELECT bucket AS 用户分位,
       MAX(ROUND(rn * 100.0 / tot, 0))                              AS 累计人数占比_pct,
       MAX(ROUND(cum * 100.0 / total_amt, 2))                       AS 累计GMV占比_pct
FROM (
    SELECT *, CASE WHEN rn * 1.0 / tot <= 0.05 THEN 'TOP 5%'
                   WHEN rn * 1.0 / tot <= 0.10 THEN 'TOP 10%'
                   WHEN rn * 1.0 / tot <= 0.20 THEN 'TOP 20%'
                   WHEN rn * 1.0 / tot <= 0.50 THEN 'TOP 50%'
                   ELSE 'TOP 100%' END AS bucket
    FROM r
) t
GROUP BY bucket
ORDER BY 累计人数占比_pct;


-- @query: 复购间隔分布
WITH seq AS (
    SELECT
        customer_unique_id, order_id, purchase_date,
        ROW_NUMBER() OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS nth,
        LAG(purchase_date) OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS prev_date
    FROM dwd_order
)
SELECT
    CASE WHEN nth = 2 THEN '首单→二单'
         WHEN nth = 3 THEN '二单→三单'
         ELSE '四单及以后' END                                      AS 复购阶段,
    COUNT(*)                                                        AS 样本数,
    ROUND(AVG(DATE_DIFF('day', prev_date, purchase_date)), 1)       AS 平均间隔天数,
    MEDIAN(DATE_DIFF('day', prev_date, purchase_date))              AS 中位间隔天数,
    QUANTILE_CONT(DATE_DIFF('day', prev_date, purchase_date), 0.25) AS P25,
    QUANTILE_CONT(DATE_DIFF('day', prev_date, purchase_date), 0.75) AS P75,
    -- 30 天内复购的比例，衡量复购的紧密程度
    ROUND(SUM(CASE WHEN DATE_DIFF('day', prev_date, purchase_date) <= 30
                   THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1)        AS 短周期复购占比_pct
FROM seq
WHERE nth >= 2
GROUP BY 1
ORDER BY MIN(nth);


-- @query: 首单体验是否影响复购（关键因果问题的相关性证据）
-- 注意：这里只是相关性。真正的因果效应见 python/03_causal_delivery.py 的 PSM 分析
WITH first_order AS (
    SELECT
        customer_unique_id, order_id, is_late, review_score, is_bad_review,
        delivery_days, order_amount, purchase_date,
        ROW_NUMBER() OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS rn
    FROM dwd_order
),
c AS (
    SELECT customer_unique_id, COUNT(DISTINCT order_id) AS n FROM dwd_order GROUP BY 1
)
SELECT
    CASE WHEN f.is_late THEN '首单延迟送达' ELSE '首单按时送达' END  AS 首单体验,
    COUNT(*)                                                        AS 客户数,
    ROUND(COUNT(*) FILTER (WHERE c.n >= 2) * 100.0 / COUNT(*), 3)   AS 复购率_pct,
    ROUND(AVG(c.n), 4)                                              AS 人均订单数,
    ROUND(AVG(f.review_score), 3)                                   AS 首单评分,
    ROUND(AVG(f.order_amount), 1)                                   AS 首单金额
FROM first_order f
JOIN c ON f.customer_unique_id = c.customer_unique_id
WHERE f.rn = 1 AND f.is_late IS NOT NULL
GROUP BY 1;


-- @query: 首单评分与复购率
WITH first_order AS (
    SELECT customer_unique_id, review_score,
           ROW_NUMBER() OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS rn
    FROM dwd_order
),
c AS (SELECT customer_unique_id, COUNT(DISTINCT order_id) AS n FROM dwd_order GROUP BY 1)
SELECT
    f.review_score                                                  AS 首单评分,
    COUNT(*)                                                        AS 客户数,
    ROUND(COUNT(*) FILTER (WHERE c.n >= 2) * 100.0 / COUNT(*), 3)   AS 复购率_pct,
    -- 相对整体复购率的倍数
    ROUND((COUNT(*) FILTER (WHERE c.n >= 2) * 1.0 / COUNT(*))
        / (SUM(COUNT(*) FILTER (WHERE c.n >= 2)) OVER ()
           * 1.0 / SUM(COUNT(*)) OVER ()), 3)                       AS 相对倍数
FROM first_order f
JOIN c ON f.customer_unique_id = c.customer_unique_id
WHERE f.rn = 1 AND f.review_score IS NOT NULL
GROUP BY f.review_score
ORDER BY f.review_score;


-- @query: 首单品类对复购的影响
WITH first_order AS (
    SELECT customer_unique_id, order_id,
           ROW_NUMBER() OVER (PARTITION BY customer_unique_id ORDER BY purchase_ts) AS rn
    FROM dwd_order
),
first_cat AS (
    SELECT f.customer_unique_id, i.category_en,
           ROW_NUMBER() OVER (PARTITION BY f.customer_unique_id ORDER BY i.item_amount DESC) AS rk
    FROM first_order f
    JOIN dwd_order_item i ON f.order_id = i.order_id
    WHERE f.rn = 1
),
c AS (
    SELECT customer_unique_id, COUNT(DISTINCT order_id) AS n, SUM(order_amount) AS ltv
    FROM dwd_order GROUP BY 1
)
SELECT
    fc.category_en                                                  AS 首单品类,
    COUNT(*)                                                        AS 首单客户数,
    ROUND(COUNT(*) FILTER (WHERE c.n >= 2) * 100.0 / COUNT(*), 3)   AS 复购率_pct,
    ROUND(AVG(c.ltv), 1)                                            AS 平均LTV,
    ROUND(AVG(c.n), 4)                                              AS 人均订单数
FROM first_cat fc
JOIN c ON fc.customer_unique_id = c.customer_unique_id
WHERE fc.rk = 1
GROUP BY fc.category_en
HAVING COUNT(*) >= 800
ORDER BY 复购率_pct DESC
LIMIT 15;


-- @query: 月度队列留存（客户在首单后第 N 月是否回购）
WITH cohort AS (
    SELECT customer_unique_id, MIN(purchase_month) AS cohort_month
    FROM dwd_order GROUP BY 1
),
act AS (
    SELECT
        c.cohort_month, o.customer_unique_id,
        DATE_DIFF('month', c.cohort_month, o.purchase_month) AS month_n
    FROM dwd_order o JOIN cohort c ON o.customer_unique_id = c.customer_unique_id
),
sized AS (SELECT cohort_month, COUNT(*) AS size FROM cohort GROUP BY 1)
SELECT
    STRFTIME(a.cohort_month, '%Y-%m')                               AS 首单月,
    s.size                                                          AS 队列规模,
    ROUND(COUNT(DISTINCT CASE WHEN month_n = 1 THEN a.customer_unique_id END) * 100.0 / s.size, 3) AS M1,
    ROUND(COUNT(DISTINCT CASE WHEN month_n = 2 THEN a.customer_unique_id END) * 100.0 / s.size, 3) AS M2,
    ROUND(COUNT(DISTINCT CASE WHEN month_n = 3 THEN a.customer_unique_id END) * 100.0 / s.size, 3) AS M3,
    ROUND(COUNT(DISTINCT CASE WHEN month_n BETWEEN 4 AND 6 THEN a.customer_unique_id END) * 100.0 / s.size, 3) AS M4_6,
    ROUND(COUNT(DISTINCT CASE WHEN month_n >= 1 THEN a.customer_unique_id END) * 100.0 / s.size, 3) AS 任意月回购
FROM act a JOIN sized s ON a.cohort_month = s.cohort_month
WHERE a.cohort_month <= DATE '2018-02-01'
GROUP BY a.cohort_month, s.size
ORDER BY a.cohort_month;
