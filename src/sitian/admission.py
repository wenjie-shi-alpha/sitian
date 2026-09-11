"""Candidate screening and publication provenance; neither authorizes training."""
from __future__ import annotations
from .data_contract import timestamp

REQUIRED_EVIDENCE = ('synoptic', 'composition', 'fires', 'spatial_obs')
ESTIMATED_BASES = {'configured_latency', 'provider_schedule', 'legacy_fixed_latency'}


def exclusion_reasons(quality: dict, missing_evidence: set[str]) -> list[str]:
    reasons = []
    if (not isinstance(quality.get('truth_failures'), list)
            or type(quality.get('cams_native_available')) is not bool
            or type(quality.get('original_builder_input_under_48')) is not bool):
        reasons.append('quality_audit_incomplete')
    if not quality.get('cams_native_available'):
        reasons.append('required_cams_missing')
    if quality.get('truth_failures'):
        reasons.append('raw_truth_invalid')
    if quality.get('original_builder_input_under_48'):
        reasons.append('input_under_existing_48_timestamp_gate')
    if set(REQUIRED_EVIDENCE) & missing_evidence:
        reasons.append('required_open_evidence_missing')
    return reasons


def publication_assessment(record: dict, cutoff, *, basis: str | None = None) -> dict:
    """A timestamp or a self-declared flag is not independent publication proof.

    Actual event verification belongs to a separate, source-bound evidence audit.
    This routine never promotes a schedule or download timestamp to such proof.
    """
    stamp = record.get('available_at')
    actual_basis = basis or record.get('availability_basis') or 'not_recorded'
    result = {'recorded_available_at': stamp, 'basis': actual_basis,
              'actual_publication_verified': False, 'before_issue': None}
    if stamp is None:
        return {**result, 'status': 'not_recorded'}
    try:
        parsed = timestamp(stamp)
    except (TypeError, ValueError):
        return {**result, 'status': 'invalid_timestamp', 'before_issue': False}
    if parsed >= cutoff:
        return {**result, 'status': 'at_or_after_issue', 'before_issue': False}
    return {**result, 'status': 'estimated' if actual_basis in ESTIMATED_BASES else 'declared_unverified',
            'before_issue': True}
