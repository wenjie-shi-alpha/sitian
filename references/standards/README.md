# Standards provenance

These are frozen copies downloaded from the Ministry of Ecology and Environment on
2026-08-28 and 2026-09-01. `manifest.json` records the official URLs, SHA256 hashes,
effective/superseded dates, relevant printed page numbers and review-image hashes. The
screenshots preserve the exact pages used to implement and test both sides of the
2026-03-01 standards transition: HJ 633—2012/2026 AQI breakpoints and the
HJ 663—2013/2026 O3 daily-validity rule.

HJ 663—2026 permits real-time monitoring data to be published after each exact hour
with a lag of up to one hour. Therefore an observation labelled 08:00 is not assumed
available for an 08:00 forecast issue: the harness uses a conservative strict
pre-08:00 cutoff (latest exact-hour record 07:00), enforced in case construction and
by `CaseBundle.audit_time_gate()`.

After replacing any PDF, update the manifest and rerun the AQI/data-validity tests. Do
not silently reuse evaluation curves across a standards or reward-version change.
