import importlib.util
from pathlib import Path

from sitian.synth import make_case


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_model.py"
SPEC = importlib.util.spec_from_file_location("probe_model", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


def test_rollout_seed_is_stable_and_paired():
    first = PROBE.rollout_seed(7, "北京_2026-06-28", 0)
    assert first == PROBE.rollout_seed(7, "北京_2026-06-28", 0)
    assert 0 <= first < 2**63
    assert first != PROBE.rollout_seed(7, "北京_2026-06-28", 1)
    assert first != PROBE.rollout_seed(8, "北京_2026-06-28", 0)
    assert first != PROBE.rollout_seed(7, "北京_2026-06-29", 0)


def test_truth_event_is_derived_from_complete_truth_not_sampling_stratum():
    event = make_case("accumulation", seed=3)
    clean = make_case("clean", seed=3)
    event.meta["stratum"] = "pm10_primary"
    clean.meta["stratum"] = "event"

    assert PROBE.truth_has_event(event) is True
    assert PROBE.truth_has_event(clean) is False
