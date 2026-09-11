"""Regression coverage for the Ray log interleaving that stopped training."""
import importlib.util
import math
from pathlib import Path


def load_parser():
    path = Path(__file__).resolve().parents[1] / 'scripts/summarize_batch_benchmark.py'
    spec = importlib.util.spec_from_file_location('training_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interleaved_ray_timestamp_is_not_a_metric():
    parser = load_parser()
    text = ('\x1b[36m(TaskRunnerV1 pid=1)\x1b[0m step:6 - actor/grad_norm:0.00284'
            ' - timing_s/step:1668.37 - off_policy/min:np.int64(0)'
            '(TaskRunnerV1 pid=1) INFO:2026-09-11 22:29:04,437:pending: 0')
    assert parser.parse_step_metrics(text) == {
        6: {'actor/grad_norm': .00284, 'timing_s/step': 1668.37, 'off_policy/min': 0}}
    assert parser.VALUE.findall('INFO:2026-09-11 ') == []


def test_numpy_scientific_and_nonfinite_values_remain_auditable():
    metrics = load_parser().parse_step_metrics(
        'step:7 - timing_s/step:np.float64(1.2e3) - actor/lr:3e-06'
        ' - actor/pg_loss:-.02 - actor/grad_norm:nan - bad:1.2.3')[7]
    assert metrics['timing_s/step'] == 1200
    assert metrics['actor/lr'] == .000003
    assert metrics['actor/pg_loss'] == -.02
    assert math.isnan(metrics['actor/grad_norm'])
    assert 'bad' not in metrics


def test_validation_and_partial_lines_do_not_count_as_training():
    parser = load_parser()
    assert parser.parse_step_metrics(
        'step:3 - val/reward:np.float64(.55)\n'
        'step:4 - actor/lr:3e-06 - timing_s/ste') == {}
