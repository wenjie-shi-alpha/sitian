SELECT CASE key
  WHEN 'val' THEN '暖季验证' WHEN 'winter_selection' THEN '冬季选择'
  WHEN 'test' THEN '时间终评' WHEN 'ood_test' THEN '整城终评' END AS split,
  json_extract(value, '$.truth_event_cases') AS event_cases,
  json_extract(value, '$.cases') AS total_cases,
  json_extract(value, '$.issue_dates') AS issue_dates
FROM json_each(:evidence_json, '$.profiles')
WHERE key IN ('val', 'winter_selection', 'test', 'ood_test')
ORDER BY CASE key WHEN 'val' THEN 1 WHEN 'winter_selection' THEN 2
  WHEN 'test' THEN 3 WHEN 'ood_test' THEN 4 END;
