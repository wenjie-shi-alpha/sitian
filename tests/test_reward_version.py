from sitian.scoring import REWARD_VERSION, RewardConfig, reward_spec, score_forecast


def test_reward_spec_is_stable_and_embedded_on_invalid_score():
    first = reward_spec(RewardConfig())
    second = reward_spec(RewardConfig())
    assert first == second
    assert first["version"] == REWARD_VERSION == "0.8.4"
    result = score_forecast({}, {}, issue_date="2026-01-01", horizon=1)
    assert result.details["reward_spec"] == first
