# -*- coding: utf-8 -*-
"""Manual reverse differentiation and activation replay; no PyTorch/GPU run."""
import argparse
import ast
import json
import math
import pathlib
import random


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def forward(weights, x):
    values = [x]
    for w in weights:
        values.append(sigmoid(w * values[-1]))
    return values


def backward_segment(weights, start, values, g, grads):
    for offset in range(len(values) - 2, -1, -1):
        i = start + offset
        previous, current = values[offset:offset + 2]
        local = g * current * (1 - current)
        grads[i] = local * previous
        g = local * weights[i]
    return g


def checkpointed(weights, x, segment_size, target):
    checkpoints = {}
    h = x
    for i, w in enumerate(weights):
        if i % segment_size == 0:
            checkpoints[i] = h
        h = sigmoid(w * h)
    output = h
    g = output - target
    grads = [0.0] * len(weights)
    calls = len(weights)
    peak_slots = len(checkpoints) + 1
    for start in reversed(range(0, len(weights), segment_size)):
        end = min(start + segment_size, len(weights))
        values = forward(weights[start:end], checkpoints[start])
        calls += end - start
        # Count unique activation indices, excluding params/gradients/Python overhead.
        peak_slots = max(peak_slots, len(set(checkpoints) | set(range(start, end + 1))))
        g = backward_segment(weights, start, values, g, grads)
        del checkpoints[start]
        del values
    return output, grads, g, calls, peak_slots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('outputs/教学验证.json'))
    args = parser.parse_args()
    weights = [0.6 + 0.04 * i for i in range(16)]
    x, target = 0.3, 0.1
    values = forward(weights, x)
    baseline_grads = [0.0] * 16
    baseline_input_grad = backward_segment(weights, 0, values, values[-1] - target, baseline_grads)
    out, grads, input_grad, calls, slots = checkpointed(weights, x, 4, target)
    error = max(abs(a - b) for a, b in zip(grads, baseline_grads))
    assert error < 1e-14 and abs(input_grad - baseline_input_grad) < 1e-14
    assert out == values[-1] and calls == 32 and slots == 8
    eps = 1e-5
    numerical = []
    for i in range(16):
        plus, minus = weights[:], weights[:]
        plus[i] += eps
        minus[i] -= eps
        lp = 0.5 * (forward(plus, x)[-1] - target) ** 2
        lm = 0.5 * (forward(minus, x)[-1] - target) ** 2
        numerical.append((lp - lm) / (2 * eps))
    finite_error = max(abs(a - b) for a, b in zip(grads, numerical))
    assert finite_error < 1e-9
    rng = random.Random(1)
    before = rng.getstate()
    original_mask = 2 if rng.random() >= 0.5 else 0
    wrong_mask = 2 if rng.random() >= 0.5 else 0
    rng.setstate(before)
    restored_mask = 2 if rng.random() >= 0.5 else 0
    assert original_mask != wrong_mask and original_mask == restored_mask
    source_checks = 'not requested'
    if args.sources:
        for name in ['modeling-bert-v4.30.0.py', 'modeling-utils-v4.30.0.py', 'checkpoint-v2.0.1.py']:
            ast.parse((args.sources / name).read_text(encoding='utf-8'))
        bert = (args.sources / 'modeling-bert-v4.30.0.py').read_text(encoding='utf-8')
        assert 'if self.gradient_checkpointing and self.training:' in bert
        assert 'torch.utils.checkpoint.checkpoint(' in bert
        source_checks = 'syntax and BERT training/checkpoint branch passed; no torch execution'
    result = {'status': 'partial', 'torch_run': False, 'network': '16 scalar sigmoid layers, manual derivatives',
              'output': out, 'loss': 0.5 * (out - target) ** 2, 'weight_gradients': grads,
              'max_gradient_difference': error, 'max_finite_difference_error': finite_error,
              'input_gradient': input_grad, 'baseline_forward_calls': 16, 'replay_forward_calls_total': calls,
              'baseline_activation_slots': 17, 'replay_activation_slots_peak': slots,
              'slot_definition': 'symbolic unique live activation indices, not measured memory',
              'rng_masks': [original_mask, wrong_mask, restored_mask], 'source_checks': source_checks,
              'historical_comparison': {'memory_mib': [12917, 9305], 'memory_reduction_pct': 100*(12917-9305)/12917,
                  'runtime_s': [23.78,38.96], 'runtime_increase_pct': 100*(38.96/23.78-1),
                  'samples_per_second': [21.53,13.14], 'throughput_reduction_pct': 100*(1-13.14/21.53),
                  'checkpointing_only_ablation': False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
