-- ============================================================================
-- 00_create_views.sql   ODS → DWD 视图层与口径定义
--
-- 真实数据不像模拟数据那样干净。在写任何分析之前，必须先把口径钉死，
-- 否则同一个"GMV"会有三种算法、三个答案。本文件固化以下约定：
--
-- 【口径 1】客户的唯一标识是 customer_unique_id，不是 customer_id
--     Olist 的 customer_id 是"一单一个"的订单级客户键，99,441 个订单对应
--     99,441 个 customer_id，但只有 96,096 个 customer_unique_id。
--     用 customer_id 算复购率会得到 0.00%（每个 id 只出现一次），
--     用 customer_unique_id 才是真实的 3.12%。这是本数据集最大的坑。
--
-- 【口径 2】GMV = 商品金额 + 运费，且必须"先聚合再 JOIN"
--     orders / order_items / order_payments 三张表都是一对多
--     （每单最多 21 条明细、29 条支付记录、3 条评论）。
--     直接 JOIN 后 SUM 会让 GMV 从 1,584.8 万虚高到 1,657.2 万（+4.57%）。
--     所有涉及金额的视图都先在子查询里按 order_id 聚合到一行，再向外 JOIN。
--
-- 【口径 3】有效订单排除 canceled 与 unavailable
--     这两类合计 1,234 单（1.24%），未实际成交，计入 GMV 会高估。
--
-- 【口径 4】分析窗口 2017-01-01 ~ 2018-08-31
--     2016-09 仅 4 单、2016-12 仅 1 单、2018-09 仅 16 单、2018-10 仅 4 单，
--     属于数据采集的边缘残留。不截断会让首尾月份的同比环比完全失真。
--
-- 【口径 5】每单只取第一条评论
--     有 544 个订单存在多条评论（最多 3 条）。不去重会让评分分布被重复计数。
-- ============================================================================

-- ---------------------------------------------------------------- ODS 原始表
CREATE OR REPLACE VIEW ods_orders AS
    SELECT * FROM read_csv_auto('data/olist_orders_dataset.csv', header=true, sample_size=-1);
CREATE OR REPLACE VIEW ods_order_items AS
    SELECT * FROM read_csv_auto('data/olist_order_items_dataset.csv', header=true, sample_size=-1);
CREATE OR REPLACE VIEW ods_payments AS
    SELECT * FROM read_csv_auto('data/olist_order_payments_dataset.csv', header=true, sample_size=-1);
CREATE OR REPLACE VIEW ods_reviews AS
    SELECT * FROM read_csv_auto('data/olist_order_reviews_dataset.csv', header=true, sample_size=-1);
CREATE OR REPLACE VIEW ods_customers AS
    SELECT * FROM read_csv_auto('data/olist_customers_dataset.csv', header=true, sample_size=-1);
CREATE OR REPLACE VIEW ods_products AS
    SELECT * FROM read_csv_auto('data/olist_products_dataset.csv', header=true, sample_size=-1);
CREATE OR REPLACE VIEW ods_sellers AS
    SELECT * FROM read_csv_auto('data/olist_sellers_dataset.csv', header=true, sample_size=-1);
CREATE OR REPLACE VIEW ods_category_translation AS
    SELECT * FROM read_csv_auto('data/product_category_name_translation.csv', header=true, sample_size=-1);


-- ---------------------------------------------------------------- 品类维表
-- 610 个商品的 product_category_name 为空，统一归入「未分类」而非丢弃，
-- 否则这部分订单会在品类汇总时凭空消失，导致各品类之和 ≠ 总量。
CREATE OR REPLACE VIEW dim_product AS
SELECT
    p.product_id,
    COALESCE(p.product_category_name, 'sem_categoria')          AS category_pt,
    COALESCE(t.product_category_name_english, '未分类')          AS category_en,
    p.product_weight_g,
    p.product_length_cm * p.product_height_cm * p.product_width_cm AS product_volume_cm3,
    p.product_photos_qty,
    p.product_description_lenght                                AS description_length
FROM ods_products p
LEFT JOIN ods_category_translation t
       ON p.product_category_name = t.product_category_name;


-- ---------------------------------------------------------------- 订单级金额（先聚合）
-- 一行一个 order_id，供外层安全 JOIN，杜绝一对多膨胀
CREATE OR REPLACE VIEW dwd_order_amount AS
SELECT
    order_id,
    SUM(price)                      AS product_amount,
    SUM(freight_value)              AS freight_amount,
    SUM(price + freight_value)      AS order_amount,
    COUNT(*)                        AS item_count,
    COUNT(DISTINCT product_id)      AS distinct_products,
    COUNT(DISTINCT seller_id)       AS distinct_sellers,
    MAX(shipping_limit_date)        AS shipping_limit_date
FROM ods_order_items
GROUP BY order_id;


-- ---------------------------------------------------------------- 订单级支付（先聚合）
CREATE OR REPLACE VIEW dwd_order_payment AS
SELECT
    order_id,
    SUM(payment_value)                                          AS payment_amount,
    MAX(payment_installments)                                   AS max_installments,
    COUNT(*)                                                    AS payment_records,
    -- 金额最大的那笔支付方式作为该订单的主支付方式
    ARG_MAX(payment_type, payment_value)                        AS main_payment_type,
    BOOL_OR(payment_type = 'voucher')                           AS used_voucher
FROM ods_payments
GROUP BY order_id;


-- ---------------------------------------------------------------- 订单级评论（去重）
CREATE OR REPLACE VIEW dwd_order_review AS
SELECT order_id, review_score, review_creation_date, review_answer_timestamp,
       has_comment
FROM (
    SELECT
        order_id,
        review_score,
        review_creation_date,
        review_answer_timestamp,
        review_comment_message IS NOT NULL                      AS has_comment,
        -- 同一订单多条评论时取最早的一条
        ROW_NUMBER() OVER (PARTITION BY order_id
                           ORDER BY review_creation_date, review_id) AS rn
    FROM ods_reviews
) t
WHERE rn = 1;


-- ---------------------------------------------------------------- DWD 订单宽表（核心）
CREATE OR REPLACE VIEW dwd_order AS
SELECT
    o.order_id,
    c.customer_unique_id,                                       -- 口径1：自然人
    o.customer_id                                               AS order_customer_id,
    o.order_status,
    o.order_purchase_timestamp                                  AS purchase_ts,
    CAST(o.order_purchase_timestamp AS DATE)                    AS purchase_date,
    DATE_TRUNC('month', o.order_purchase_timestamp)             AS purchase_month,
    o.order_approved_at                                         AS approved_ts,
    o.order_delivered_carrier_date                              AS carrier_ts,
    o.order_delivered_customer_date                             AS delivered_ts,
    o.order_estimated_delivery_date                             AS estimated_ts,

    -- 金额（口径2：来自已聚合的子视图）
    a.product_amount,
    a.freight_amount,
    a.order_amount,
    a.item_count,
    a.distinct_products,
    a.distinct_sellers,
    p.payment_amount,
    p.max_installments,
    p.main_payment_type,
    p.used_voucher,
    ROUND(a.freight_amount / NULLIF(a.order_amount, 0), 4)      AS freight_ratio,

    -- 时效指标
    DATE_DIFF('day', o.order_purchase_timestamp,
                     o.order_delivered_customer_date)           AS delivery_days,
    DATE_DIFF('day', o.order_purchase_timestamp,
                     o.order_estimated_delivery_date)           AS promised_days,
    -- 正数=比承诺早到，负数=延迟
    DATE_DIFF('day', o.order_delivered_customer_date,
                     o.order_estimated_delivery_date)           AS days_early,
    (o.order_delivered_customer_date > o.order_estimated_delivery_date) AS is_late,
    DATE_DIFF('day', o.order_approved_at,
                     o.order_delivered_carrier_date)            AS seller_handling_days,
    DATE_DIFF('day', o.order_delivered_carrier_date,
                     o.order_delivered_customer_date)           AS carrier_transit_days,

    -- 评论
    r.review_score,
    (r.review_score <= 2)                                       AS is_bad_review,
    (r.review_score = 5)                                        AS is_top_review,
    r.has_comment,

    -- 客户属性
    cu.customer_state,
    cu.customer_city,
    cu.customer_zip_code_prefix
FROM ods_orders o
JOIN ods_customers c   ON o.customer_id = c.customer_id
JOIN ods_customers cu  ON o.customer_id = cu.customer_id
LEFT JOIN dwd_order_amount  a ON o.order_id = a.order_id
LEFT JOIN dwd_order_payment p ON o.order_id = p.order_id
LEFT JOIN dwd_order_review  r ON o.order_id = r.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')         -- 口径3
  AND o.order_purchase_timestamp >= DATE '2017-01-01'           -- 口径4
  AND o.order_purchase_timestamp <  DATE '2018-09-01'
  AND a.order_id IS NOT NULL;                                   -- 排除 775 个无明细订单


-- ---------------------------------------------------------------- 订单-商品明细宽表
-- 需要下钻到商品/卖家维度时用这张；注意它是明细粒度，不能直接 SUM 订单级指标
CREATE OR REPLACE VIEW dwd_order_item AS
SELECT
    i.order_id,
    i.order_item_id,
    i.product_id,
    i.seller_id,
    i.price,
    i.freight_value,
    i.price + i.freight_value                                   AS item_amount,
    d.category_en,
    d.product_weight_g,
    d.product_volume_cm3,
    d.product_photos_qty,
    s.seller_state,
    s.seller_city,
    o.purchase_date,
    o.purchase_month,
    o.customer_unique_id,
    o.customer_state,
    o.is_late,
    o.review_score,
    o.delivery_days
FROM ods_order_items i
JOIN dwd_order o    ON i.order_id = o.order_id          -- 继承口径 3/4 的过滤
JOIN dim_product d  ON i.product_id = d.product_id
LEFT JOIN ods_sellers s ON i.seller_id = s.seller_id;


-- ---------------------------------------------------------------- 客户级汇总
CREATE OR REPLACE VIEW dws_customer AS
SELECT
    customer_unique_id,
    COUNT(DISTINCT order_id)                                    AS order_count,
    SUM(order_amount)                                           AS total_amount,
    AVG(order_amount)                                           AS avg_order_amount,
    MIN(purchase_date)                                          AS first_order_date,
    MAX(purchase_date)                                          AS last_order_date,
    DATE_DIFF('day', MIN(purchase_date), MAX(purchase_date))    AS lifespan_days,
    AVG(review_score)                                           AS avg_review_score,
    SUM(CASE WHEN is_late THEN 1 ELSE 0 END)                    AS late_order_count,
    AVG(delivery_days)                                          AS avg_delivery_days,
    MAX(customer_state)                                         AS customer_state
FROM dwd_order
GROUP BY customer_unique_id;
