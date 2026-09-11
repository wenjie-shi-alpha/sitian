from scripts.audit_reward_contract import run_audit


def test_reward_red_team_contract_passes():
    report = run_audit()
    assert report["passed"] is True
    assert len(report["checks"]) >= 10
    assert all(row["pass"] for row in report["checks"].values())
