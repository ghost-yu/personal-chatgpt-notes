# -*- coding: utf-8 -*-
"""Small SFT checks; no model training and no PyTorch dependency."""
import argparse
import ast
import json
import math
import pathlib
import random
import types
import warnings


def pack(rows, length, eos):
    stream = []
    for row in rows:
        stream.extend(row + [eos])
    chunks = [stream[i:i + length] for i in range(0, len(stream), length)]
    return [x for x in chunks if len(x) == length], len(stream) % length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('outputs/教学验证.json'))
    args = parser.parse_args()
    rows = [[1, 2], [3, 4, 5], [6]]
    chunks, dropped = pack(rows, 4, 9)
    assert chunks == [[1, 2, 9, 3], [4, 5, 9, 6]] and dropped == 1
    labels = [x[:] for x in chunks]
    # CLM predicts labels[t+1] with logits[t]; no second shift in collator.
    targets = [x[1:] for x in labels]
    assert targets == [[2, 9, 3], [5, 9, 6]]
    masked = [[-100 if x == 9 else x for x in row] for row in labels]
    assert masked == [[1, 2, -100, 3], [4, 5, -100, 6]]
    probabilities = [0.8, 0.5, 0.25]
    loss = -sum(math.log(p) for p in probabilities) / len(probabilities)
    assert abs(loss - 0.7675283643313485) < 1e-12
    rank, dim, layers = 16, 4096, 32
    adapter_parameters = layers * 2 * rank * (dim + dim)
    assert adapter_parameters == 8388608
    static = 'not requested'
    executed = False
    if args.sources:
        sft = (args.sources / 'sft-v0.4.7.py').read_text(encoding='utf-8')
        assert 'if not isinstance(model, PeftModel)' in sft
        assert 'if not packing:' in sft
        tree = ast.parse((args.sources / 'utils-v0.4.7.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ConstantLengthDataset')
        module = ast.Module(body=[cls], type_ignores=[])
        # Execute only this official class, with explicit list-based tensor stub.
        namespace = {'IterableDataset': object, 'torch': types.SimpleNamespace(LongTensor=list),
                     'random': random, 'warnings': warnings}
        exec(compile(ast.fix_missing_locations(module), '<official-class-fixture>', 'exec'), namespace)
        class TokenizerFixture:
            eos_token_id = 9
            def __call__(self, buffer, truncation=False):
                return {'input_ids': [json.loads(x) for x in buffer]}
        dataset = [{'text': json.dumps(row)} for row in rows]
        result = list(namespace['ConstantLengthDataset'](TokenizerFixture(), dataset,
                      dataset_text_field='text', seq_length=4, shuffle=False))
        assert [x['input_ids'] for x in result] == chunks
        assert [x['labels'] for x in result] == labels
        executed = True
        static = 'TRL v0.4.7 guard and official packing class fixture passed'
    report = {'status': 'partial', 'torch_model_run': False, 'tokenizer': 'integer fixture, not XgenTokenizer',
              'packed_input_ids': chunks, 'tail_dropped_tokens': dropped,
              'shifted_targets': targets, 'packing_preserves_eos_labels_in_v0_4_7': True,
              'nonpacking_lm_collator_eos_mask_example': masked,
              'cross_entropy_example': loss, 'lora_q_v_parameters': adapter_parameters,
              'vocab_rows_removed': 51200 - 50313, 'official_class_executed_with_list_tensor_stub': executed,
              'source_check': static}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
