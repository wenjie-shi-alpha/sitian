"""Check actual adapter tensors and optimizer steps across a resumed update."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--expected-before-step', type=int, required=True)
    parser.add_argument('--expected-after-step', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    filename = 'model_world_size_1_rank_0.pt'
    before = torch.load(args.before / 'actor' / filename, map_location='cpu', weights_only=True)
    after = torch.load(args.after / 'actor' / filename, map_location='cpu', weights_only=True)
    same_keys = before.keys() == after.keys()
    changed = 0
    finite = True
    max_delta = 0.0
    shapes_match = same_keys
    for key in before.keys() & after.keys():
        left, right = before[key], after[key]
        shapes_match &= left.shape == right.shape
        finite &= bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
        if left.shape != right.shape:
            continue
        changed += int(not torch.equal(left, right))
        max_delta = max(max_delta, float((right.float() - left.float()).abs().max()))
    optimizer_steps = {}
    state_counts = {}
    for label, folder in [('before', args.before), ('after', args.after)]:
        opt = torch.load(folder / 'actor' / 'optim_world_size_1_rank_0.pt',
                         map_location='cpu', weights_only=True)
        state_counts[label] = len(opt['state'])
        optimizer_steps[label] = sorted({float(v['step']) for v in opt['state'].values()})
        del opt
    checks = {
        'same_nonempty_adapter_structure': bool(before) and same_keys and shapes_match,
        'all_saved_tensors_finite': finite,
        'actual_parameter_values_changed': changed > 0 and max_delta > 0,
        'optimizer_state_count_matches': state_counts['before'] == state_counts['after'] == len(before),
        'optimizer_step_before_matches': optimizer_steps['before'] == [float(args.expected_before_step)],
        'optimizer_step_after_matches': optimizer_steps['after'] == [float(args.expected_after_step)],
    }
    result = {
        'success': all(checks.values()), 'checks': checks,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'before': str(args.before.resolve()), 'after': str(args.after.resolve()),
        'adapter_tensors': len(before), 'changed_tensors': changed,
        'max_absolute_parameter_change': max_delta,
        'optimizer_steps': optimizer_steps, 'optimizer_state_counts': state_counts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
