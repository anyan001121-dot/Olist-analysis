-- ============================================================================
-- 02_delivery_quality.sql   履约时效分析（本项目核心）
--
-- 业务问题：平台差评率 13.72%，钱花在哪能最有效地把它降下来？
-- 分析路径：延迟送达 → 差评 → 是否影响复购 → 延迟本身由谁造成 → 该改什么
-- 技术点：时长链路拆解、分档对比、贡献度归因、窗口函数算集中度
--
-- 关键背景：这个平台的老客 GMV 占比只有 2~3%，几乎完全靠拉新驱动。
--           在这种结构下，体验问题不会立刻体现在复购上，而是体现在
--           评分和口碑上——所以评分是比复购更灵敏的早期预警指标。
-- ============================================================================

-- @query: 延迟送达对评分的影响（核心发现）
SELECT
    CASE WHEN is_late THEN '延迟送达' ELSE '按时/提前' END          AS 送达情况,
    COUNT(*)                                                        AS 订单数,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)              AS 占比_pct,
    ROUND(AVG(review_score), 3)                                     AS 平均评分,
    ROUND(AVG(CASE WHEN is_bad_review THEN 1.0 ELSE 0 END) * 100, 2) AS 差评率_pct,
    ROUND(AVG(CASE WHEN is_top_review THEN 1.0 ELSE 0 END) * 100, 2) AS 五星率_pct,
    ROUND(AVG(delivery_days), 1)                                    AS 平均送达天数,
    ROUND(AVG(order_amount), 1)                                     AS 客单价
FROM dwd_order
WHERE delivered_ts IS NOT NULL AND review_score IS NOT NULL
GROUP BY is_late;


-- @query: 延迟天数与评分的剂量反应关系
-- 不只看"延迟 vs 不延迟"，而是看延迟多久开始伤害评分——这决定了补救的容忍窗口
SELECT
    CASE WHEN days_early >= 15 THEN '1_提前15天以上'
         WHEN days_early >= 8  THEN '2_提前8-14天'
         WHEN days_early >= 3  THEN '3_提前3-7天'
         WHEN days_early >= 0  THEN '4_提前0-2天'
         WHEN days_early >= -3 THEN '5_延迟1-3天'
         WHEN days_early >= -7 THEN '6_延迟4-7天'
         WHEN days_early >= -15 THEN '7_延迟8-15天'
         ELSE '8_延迟15天以上' END                                  AS 时效区间,
    COUNT(*)                                                        AS 订单数,
    ROUND(AVG(review_score), 3)                                     AS 平均评分,
    ROUND(AVG(CASE WHEN is_bad_review THEN 1.0 ELSE 0 END) * 100, 2) AS 差评率_pct,
    -- 相对上一档的差评率变化，看在哪一档发生断崖
    ROUND(AVG(CASE WHEN is_bad_review THEN 1.0 ELSE 0 END) * 100
        - LAG(AVG(CASE WHEN is_bad_review THEN 1.0 ELSE 0 END) * 100)
          OVER (ORDER BY MIN(CASE WHEN days_early >= 15 THEN 1
                                  WHEN days_early >= 8 THEN 2
                                  WHEN days_early >= 3 THEN 3
                                  WHEN days_early >= 0 THEN 4
                                  WHEN days_early >= -3 THEN 5
                                  WHEN days_early >= -7 THEN 6
                                  WHEN days_early >= -15 THEN 7
                                  ELSE 8 END)), 2)                  AS 环比变化_pct_point
FROM dwd_order
WHERE delivered_ts IS NOT NULL AND review_score IS NOT NULL
GROUP BY 1
ORDER BY 1;


-- @query: 延迟时长的链路拆解（延迟到底是谁造成的）
-- 订单全链路：下单 → 审核 → 卖家交承运商 → 送达客户
-- 把总时长拆成三段，看延迟订单主要卡在哪一段
SELECT
    CASE WHEN is_late THEN '延迟送达' ELSE '按时/提前' END          AS 送达情况,
    COUNT(*)                                                        AS 订单数,
    ROUND(AVG(DATE_DIFF('hour', purchase_ts, approved_ts)) / 24.0, 2) AS 支付审核_天,
    ROUND(AVG(seller_handling_days), 2)                             AS 卖家备货发货_天,
    ROUND(AVG(carrier_transit_days), 2)                             AS 承运商运输_天,
    ROUND(AVG(delivery_days), 2)                                    AS 总时长_天,
    ROUND(AVG(promised_days), 2)                                    AS 承诺时长_天,
    -- 各段占总时长的比例
    ROUND(AVG(seller_handling_days) * 100.0 / AVG(delivery_days), 1) AS 卖家段占比_pct,
    ROUND(AVG(carrier_transit_days) * 100.0 / AVG(delivery_days), 1) AS 运输段占比_pct
FROM dwd_order
WHERE delivered_ts IS NOT NULL AND carrier_ts IS NOT NULL AND approved_ts IS NOT NULL
GROUP BY is_late;


-- @query: 延迟归因：超时是卖家慢还是运输慢
-- 对每个延迟订单，看它的卖家段和运输段分别比"按时订单的中位数"超出多少
WITH benchmark AS (
    SELECT
        MEDIAN(seller_handling_days)  AS med_seller,
        MEDIAN(carrier_transit_days)  AS med_carrier
    FROM dwd_order
    WHERE NOT is_late AND delivered_ts IS NOT NULL AND carrier_ts IS NOT NULL
),
late_orders AS (
    SELECT
        o.order_id,
        o.seller_handling_days - b.med_seller                       AS seller_excess,
        o.carrier_transit_days - b.med_carrier                      AS carrier_excess
    FROM dwd_order o CROSS JOIN benchmark b
    WHERE o.is_late AND o.delivered_ts IS NOT NULL AND o.carrier_ts IS NOT NULL
)
SELECT
    COUNT(*)                                                        AS 延迟订单数,
    ROUND(AVG(seller_excess), 2)                                    AS 卖家段超出_天,
    ROUND(AVG(carrier_excess), 2)                                   AS 运输段超出_天,
    ROUND(AVG(seller_excess) * 100.0
          / (AVG(seller_excess) + AVG(carrier_excess)), 1)          AS 卖家段责任占比_pct,
    ROUND(AVG(carrier_excess) * 100.0
          / (AVG(seller_excess) + AVG(carrier_excess)), 1)          AS 运输段责任占比_pct,
    ROUND(COUNT(*) FILTER (WHERE seller_excess > carrier_excess) * 100.0
          / COUNT(*), 1)                                            AS 主因是卖家的订单占比_pct
FROM late_orders;


-- @query: 月度延迟率与评分（定位 2018-03 履约危机）
SELECT
    STRFTIME(purchase_month, '%Y-%m')                               AS 月份,
    COUNT(*)                                                        AS 订单数,
    ROUND(AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END) * 100, 2)      AS 延迟率_pct,
    ROUND(AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END) * 100
        - LAG(AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END) * 100)
          OVER (ORDER BY purchase_month), 2)                        AS 延迟率环比_pct_point,
    ROUND(AVG(review_score), 3)                                     AS 平均评分,
    ROUND(AVG(CASE WHEN is_bad_review THEN 1.0 ELSE 0 END) * 100, 2) AS 差评率_pct,
    ROUND(AVG(delivery_days), 1)                                    AS 平均送达天数,
    ROUND(AVG(seller_handling_days), 2)                             AS 卖家备货_天,
    ROUND(AVG(carrier_transit_days), 2)                             AS 运输_天
FROM dwd_order
WHERE delivered_ts IS NOT NULL
GROUP BY purchase_month
ORDER BY purchase_month;


-- @query: 各州履约质量与距离效应
-- 卖家高度集中在 SP 州，跨州配送的距离成本直接体现在时效上
SELECT
    customer_state                                                  AS 客户所在州,
    COUNT(*)                                                        AS 订单数,
    ROUND(AVG(delivery_days), 1)                                    AS 平均送达天数,
    ROUND(AVG(promised_days), 1)                                    AS 平均承诺天数,
    ROUND(AVG(days_early), 1)                                       AS 平均提前天数,
    ROUND(AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END) * 100, 2)      AS 延迟率_pct,
    ROUND(AVG(review_score), 3)                                     AS 平均评分,
    ROUND(AVG(freight_ratio) * 100, 1)                              AS 运费占比_pct,
    ROUND(AVG(order_amount), 1)                                     AS 客单价
FROM dwd_order
WHERE delivered_ts IS NOT NULL
GROUP BY customer_state
HAVING COUNT(*) >= 300
ORDER BY 平均送达天数 DESC;


-- @query: 跨州配送 vs 同州配送
WITH oi AS (
    SELECT
        i.order_id,
        MAX(CASE WHEN i.seller_state = i.customer_state THEN 0 ELSE 1 END) AS is_cross_state
    FROM dwd_order_item i
    WHERE i.seller_state IS NOT NULL
    GROUP BY i.order_id
)
SELECT
    CASE WHEN oi.is_cross_state = 1 THEN '跨州配送' ELSE '同州配送' END AS 配送类型,
    COUNT(*)                                                        AS 订单数,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)              AS 占比_pct,
    ROUND(AVG(o.delivery_days), 1)                                  AS 平均送达天数,
    ROUND(AVG(CASE WHEN o.is_late THEN 1.0 ELSE 0 END) * 100, 2)    AS 延迟率_pct,
    ROUND(AVG(o.review_score), 3)                                   AS 平均评分,
    ROUND(AVG(o.freight_ratio) * 100, 1)                            AS 运费占比_pct
FROM dwd_order o
JOIN oi ON o.order_id = oi.order_id
WHERE o.delivered_ts IS NOT NULL
GROUP BY 1;


-- @query: 承诺时长的设定是否合理（预期管理）
-- 平台给客户的预计送达时间普遍留了很大缓冲，缓冲越大客户等待越久，
-- 但缓冲小了又容易违诺——这是一个需要权衡的产品决策
SELECT
    CASE WHEN promised_days <= 10 THEN '1_承诺10天内'
         WHEN promised_days <= 20 THEN '2_承诺11-20天'
         WHEN promised_days <= 30 THEN '3_承诺21-30天'
         ELSE '4_承诺30天以上' END                                  AS 承诺时长档,
    COUNT(*)                                                        AS 订单数,
    ROUND(AVG(promised_days), 1)                                    AS 平均承诺天数,
    ROUND(AVG(delivery_days), 1)                                    AS 平均实际天数,
    ROUND(AVG(days_early), 1)                                       AS 平均富余天数,
    ROUND(AVG(CASE WHEN is_late THEN 1.0 ELSE 0 END) * 100, 2)      AS 延迟率_pct,
    ROUND(AVG(review_score), 3)                                     AS 平均评分
FROM dwd_order
WHERE delivered_ts IS NOT NULL AND promised_days IS NOT NULL
GROUP BY 1
ORDER BY 1;


-- @query: 差评的可归因结构（多少差评能用延迟解释）
WITH base AS (
    SELECT
        is_bad_review,
        is_late,
        COUNT(*) AS n
    FROM dwd_order
    WHERE delivered_ts IS NOT NULL AND review_score IS NOT NULL
    GROUP BY 1, 2
)
SELECT
    SUM(n) FILTER (WHERE is_bad_review)                             AS 差评总数,
    SUM(n) FILTER (WHERE is_bad_review AND is_late)                 AS 延迟订单的差评数,
    ROUND(SUM(n) FILTER (WHERE is_bad_review AND is_late) * 100.0
          / SUM(n) FILTER (WHERE is_bad_review), 2)                 AS 延迟可解释的差评占比_pct,
    -- 反事实：若延迟订单的差评率降到按时订单水平，能消掉多少差评
    ROUND(SUM(n) FILTER (WHERE is_late)
          * (SUM(n) FILTER (WHERE is_bad_review AND is_late) * 1.0 / SUM(n) FILTER (WHERE is_late)
           - SUM(n) FILTER (WHERE is_bad_review AND NOT is_late) * 1.0 / SUM(n) FILTER (WHERE NOT is_late)),
          0)                                                        AS 可消除的差评数,
    ROUND(SUM(n) FILTER (WHERE is_late)
          * (SUM(n) FILTER (WHERE is_bad_review AND is_late) * 1.0 / SUM(n) FILTER (WHERE is_late)
           - SUM(n) FILTER (WHERE is_bad_review AND NOT is_late) * 1.0 / SUM(n) FILTER (WHERE NOT is_late))
          * 100.0 / SUM(n) FILTER (WHERE is_bad_review), 1)         AS 占差评总数_pct
FROM base;
