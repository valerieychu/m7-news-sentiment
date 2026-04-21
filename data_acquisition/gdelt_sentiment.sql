WITH raw_articles AS (
  SELECT
    PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS article_date,

    SourceCommonName,

    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone_overall,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS FLOAT64) AS tone_positive,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64) AS tone_negative,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(3)] AS FLOAT64) AS tone_polarity,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(4)] AS FLOAT64) AS activity_ref_density,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(6)] AS FLOAT64) AS word_count,

    CASE
      WHEN REGEXP_CONTAINS(UPPER(V2Organizations), r'\bAPPLE\b')       THEN 'AAPL'
      WHEN REGEXP_CONTAINS(UPPER(V2Organizations), r'\bMICROSOFT\b')   THEN 'MSFT'
      WHEN REGEXP_CONTAINS(UPPER(V2Organizations), r'GOOGLE|ALPHABET') THEN 'GOOGL'
      WHEN REGEXP_CONTAINS(UPPER(V2Organizations), r'\bAMAZON\b')      THEN 'AMZN'
      WHEN REGEXP_CONTAINS(UPPER(V2Organizations), r'\bNVIDIA\b')      THEN 'NVDA'
      WHEN REGEXP_CONTAINS(UPPER(V2Organizations), r'\bMETA\b')        THEN 'META'
      WHEN REGEXP_CONTAINS(UPPER(V2Organizations), r'\bTESLA\b')       THEN 'TSLA'
      ELSE NULL
    END AS ticker

  FROM `gdelt-bq.gdeltv2.gkg`
  WHERE
    DATE >= 20250101000000
    AND DATE <= 20260412235959
    AND SourceCollectionIdentifier = 1
    AND (
      REGEXP_CONTAINS(UPPER(V2Organizations), r'\bAPPLE\b')
      OR REGEXP_CONTAINS(UPPER(V2Organizations), r'\bMICROSOFT\b')
      OR REGEXP_CONTAINS(UPPER(V2Organizations), r'GOOGLE|ALPHABET')
      OR REGEXP_CONTAINS(UPPER(V2Organizations), r'\bAMAZON\b')
      OR REGEXP_CONTAINS(UPPER(V2Organizations), r'\bNVIDIA\b')
      OR REGEXP_CONTAINS(UPPER(V2Organizations), r'\bMETA\b')
      OR REGEXP_CONTAINS(UPPER(V2Organizations), r'\bTESLA\b')
    )
    AND V2Tone IS NOT NULL
    AND V2Tone != ''
    AND ARRAY_LENGTH(SPLIT(V2Tone, ',')) >= 7
),

filtered AS (
  SELECT * FROM raw_articles WHERE ticker IS NOT NULL
)

SELECT
  ticker,
  article_date,

  COUNT(*)                          AS article_count,
  COUNT(DISTINCT SourceCommonName)  AS unique_source_count,
  SUM(word_count)                   AS total_word_count,

  SUM(tone_overall  * word_count) / NULLIF(SUM(word_count), 0) AS tone_weighted,
  SUM(tone_positive * word_count) / NULLIF(SUM(word_count), 0) AS tone_positive_weighted,
  SUM(tone_negative * word_count) / NULLIF(SUM(word_count), 0) AS tone_negative_weighted,
  SUM(tone_polarity * word_count) / NULLIF(SUM(word_count), 0) AS tone_polarity_weighted,

  AVG(tone_overall)    AS tone_avg,
  STDDEV(tone_overall) AS tone_stddev,
  MIN(tone_overall)    AS tone_min,
  MAX(tone_overall)    AS tone_max,

  COUNTIF(tone_overall >  1.0)      AS positive_article_count,
  COUNTIF(tone_overall < -1.0)      AS negative_article_count,
  COUNTIF(ABS(tone_overall) <= 1.0) AS neutral_article_count,

  AVG(activity_ref_density)         AS avg_activity_density

FROM filtered
GROUP BY ticker, article_date
ORDER BY ticker, article_date