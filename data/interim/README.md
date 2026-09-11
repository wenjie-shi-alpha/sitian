# Interim artifact status (2026-08-28)

Canonical after the AQI-standard, data-validity, process-threshold and reward audit:

- `data_quality_audit.json` and `valid_cases_{train,val,test}.json`
- `eval_{train,val,test}.json`
- `eval_tabular.json`
- `probe_group_v04.json`
- `rl_preflight.json`

The older `probe_val.json`, `audit_group_variance.json`, `probe_group_v03.json`,
`pool_baseline_eval.json`, and `eval_all.log` were produced with earlier scoring/data
semantics. They are retained only for provenance and must not be used for current
go/no-go decisions.
