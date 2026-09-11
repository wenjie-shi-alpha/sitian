from scripts.rl_preflight import _standards_archive_audit


def test_frozen_standards_cover_both_sides_of_2026_transition():
    audit = _standards_archive_audit()
    assert audit["pass"] is True
    assert audit["exact_required_set"] is True
    assert audit["required_standards"] == [
        "HJ 633-2012", "HJ 633-2026", "HJ 663-2013", "HJ 663-2026",
    ]
    assert all(row["official_source"] for row in audit["documents"])
    assert all(row["transition_metadata"] for row in audit["documents"])
