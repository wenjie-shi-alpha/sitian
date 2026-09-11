from datetime import datetime,timezone
from sitian.admission import exclusion_reasons,publication_assessment


def test_optional_regions_do_not_silently_exclude_a_valid_case():
    good={'cams_native_available':True,'truth_failures':[],'original_builder_input_under_48':False,'regional_available':False}
    assert exclusion_reasons(good,set())==[]
    assert exclusion_reasons(good,{'synoptic'})==['required_open_evidence_missing']


def test_bad_truth_and_missing_evidence_keep_both_reasons():
    q={'cams_native_available':True,'truth_failures':[{'PM10':17}],'original_builder_input_under_48':False}
    assert exclusion_reasons(q,{'fires'})==['raw_truth_invalid','required_open_evidence_missing']


def test_estimates_and_self_assertions_never_become_verified():
    cutoff=datetime(2025,1,2,tzinfo=timezone.utc)
    r={'available_at':'2025-01-01T22:00:00+00:00','available_at_verified':True}
    a=publication_assessment(r,cutoff,basis='provider_schedule')
    assert a['status']=='estimated' and a['before_issue'] is True
    assert a['actual_publication_verified'] is False
    assert publication_assessment(r,cutoff)['status']=='declared_unverified'
    assert publication_assessment({},cutoff)['status']=='not_recorded'


def test_time_gate_does_not_accept_cutoff_or_naive_timestamps():
    cutoff=datetime(2025,1,2,tzinfo=timezone.utc)
    assert publication_assessment({'available_at':'2025-01-02T00:00:00+00:00'},cutoff)['status']=='at_or_after_issue'
    assert publication_assessment({'available_at':'2025-01-01T22:00:00'},cutoff)['status']=='invalid_timestamp'


def test_missing_audit_fields_fail_closed():
    assert 'quality_audit_incomplete' in exclusion_reasons({'cams_native_available': True}, set())
