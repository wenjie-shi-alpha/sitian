# Supporting notes

- Audience: technical. Delivery: portable HTML. The report summary, evidence sections, definitions, methods/robustness and next steps cover the required technical report roles. Decision-relevant questions (target GPU compatibility, native policy performance, winter holdout) are integrated into those sections.
- Visual contract: four-category single-series bar chart of hidden-truth event case counts, on the portable HTML surface, using the shared reader's default palette, labels as non-color distinction and a zero-baseline count axis. Adjacent prose interprets limited final event coverage. The subtitle explicitly states that the city test is nested inside the temporal test; bars are not additive. A compact exact-value audit table preserves case/city/date denominators. No fabricated completion percentages.
- The exact-value table is chronological by evaluation role and explicitly labels nested subsets and event definition. Narrative confidence intervals retain the case/cluster denominator.
- Source coverage: local code, saved artifacts, fresh CPU audits and official primary web references. No cloud benchmark or new policy rollout was run.
- Test receipt: python3 -m pytest tests -q -> 202 passed in 9.73s on 2026-09-07.
- Reproduction: run audit.py from the project with PYTHONPATH=src; rerun scripts/compare_probe_clustered.py for guidance, tabular, and five-day calendar blocks as recorded in the adjacent comparison artifacts; rerun build_report.py, then the plugin portable delivery command.
- No model, data, reward, split, training script or scientific-claim source document was edited. New artifacts and dated audit outputs only.
