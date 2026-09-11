import json

import pytest

pytest.importorskip("pyarrow")

from scripts.prepare_verl_dataset import _row
from sitian.case import CaseBundle


def save_case(path):
    b = CaseBundle("dataset-contract", "2025-01-03", "北京", 1, ["北京"],
                   meta={"multi_pollutant": True},
                   truth={"daily": {"2025-01-04": {key: 1234567 for key in CaseBundle._TRUTH_KEYS}}})
    b.save(path)
    return b


def test_dataset_carries_same_resource_contract_without_truth_in_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("FH_HISTORICAL_CASE_INDEX", raising=False)
    save_case(tmp_path)
    row = _row(tmp_path, "train", 0, False)
    resources = row["extra_info"]["harness_resources"]
    assert "guidance_bias_sha256" in resources
    for tool in row["extra_info"]["tools_kwargs"].values():
        assert tool["create_kwargs"]["harness_resources"] == resources
    assert "1234567" not in json.dumps(row["prompt"])
    assert row["data_source"].endswith("v0.8.4")


@pytest.mark.parametrize("value", [None, -1, True, float("nan")])
def test_national_dataset_rejects_invalid_non_pm25_truth(tmp_path, value):
    b = save_case(tmp_path)
    b.truth["daily"]["2025-01-04"]["co_avg"] = value
    b.save(tmp_path)
    with pytest.raises(ValueError, match="six-pollutant"):
        _row(tmp_path, "train", 0, False)
