"""Execute the six reading examples and compare counts, outputs and gradients."""
from pathlib import Path
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import platform
import re
import sys
import warnings

import torch
import transformers

ROOT = Path(__file__).resolve().parents[2]
MATERIAL = Path('G:/zzz/Personal ChatGPT 系列/29 Llama2 7B vs Llama3 8B/资料')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('lesson29_previous', ROOT / '复现/scripts/verify_delivery.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    original, _, core = previous.preservation()
    note = (ROOT / '解释.md').read_text('utf-8').split('<details>', 1)[0]
    blocks = re.findall(r'^```python\n(.*?)^```', note, re.M | re.S)
    assert len(blocks) == 6
    functions = {n.name: n for b in blocks for n in ast.parse(b).body if isinstance(n, ast.FunctionDef)}
    original_fn = next(n for n in ast.parse(''.join(original['cells'][11]['source'])).body if isinstance(n, ast.FunctionDef))
    assert ast.dump(functions['print_trainable_parameters']) == ast.dump(original_fn)
    scope = {'__name__': '__foundation_examples__'}
    executions = []
    for i, block in enumerate(blocks, 1):
        out = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out):
            warnings.simplefilter('always')
            exec(compile(block, f'解释.md:block{i}', 'exec'), scope)
        executions.append({'block': i, 'stdout': out.getvalue(), 'warnings': [str(w.message) for w in caught]})
    assert scope['hidden'].tolist() == [[[0., 1.], [1., 1.]]]
    assert scope['logits'].tolist() == [[[0., 1., 1., -1.], [1., 1., 2., -2.]]]
    assert scope['embedding'].weight is not scope['head'].weight
    expected = [
        {'embedding': 131072000, 'lm_head': 131072000, 'attention': 2147483648, 'mlp': 4328521728, 'norms': 266240},
        {'embedding': 525336576, 'lm_head': 525336576, 'attention': 1342177280, 'mlp': 5637144576, 'norms': 266240},
    ]
    for model, ledger, counts in zip(scope['full_models'].values(), scope['ledgers'], expected):
        assert all(p.device.type == 'meta' for p in model.parameters())
        assert previous.breakdown(model) == ledger == counts
        assert model.model.embed_tokens.weight is not model.lm_head.weight
    totals = [sum(x.values()) for x in expected]
    assert totals == [6738415616, 8030261248]
    assert totals[1] - totals[0] == 1291845632
    expected_out = torch.tensor([10 + 10 * math.exp(1)/(1+math.exp(1)), 10 + 10/(1+math.exp(1))])
    torch.testing.assert_close(scope['shared_outputs'].flatten(), expected_out)
    torch.testing.assert_close(scope['attention_weights'].sum(-1), torch.ones(1, 2, 1))
    assert math.isclose(scope['scalar_y'].item(), 6 / (1 + math.exp(-1)), abs_tol=1e-14)
    assert sum(p.numel() for p in scope['scalar_mlp'].parameters()) == 3
    tiny = {}
    for name, result in scope['tiny_results'].items():
        model, tokens = result['model'], result['tokens']
        assert result['parameters'] == (7280 if name == 'MHA' else 8784)
        assert result['logits_shape'] == [1, 4, 32 if name == 'MHA' else 64]
        assert result['cache_shape'] == [1, 4 if name == 'MHA' else 1, 4, 4]
        assert result['cache_difference'] < 1e-6
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sequence = tokens[:, :3].clone()
        # Recompute the whole prefix for each step, independently of generate's cache.
        with torch.inference_mode():
            for _ in range(3):
                next_id = model(sequence, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                sequence = torch.cat((sequence, next_id), dim=1)
        assert torch.equal(sequence, result['generated'])
        assert torch.equal(sequence[:, :3], tokens[:, :3]) and list(sequence.shape) == [1, 6]
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        tiny[name] = {k: v for k, v in result.items() if k not in ['model', 'tokens', 'generated']}
        tiny[name].update({'generated_ids': sequence.tolist(), 'manual_full_prefix_greedy_equal': True,
                           'parameters_unchanged': True, 'no_parameter_gradients': True})
    previous.REPORT['boundaries'] = {}
    previous.core_checks(core)
    crops = json.loads((ROOT / '复现/outputs/论文截图来源.json').read_text('utf-8'))
    assert len(crops) == 3
    for item in crops:
        assert sha(MATERIAL / item['paper']) == item['pdf_sha256']
        assert sha(ROOT / item['published_path']) == item['png_sha256']
        assert item['published_path'] in note
    report = {
        'status': 'passed', 'scope': 'Six published examples; full meta architecture counts; CPU tiny forward/cache/generation; complete core forward and gradients',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                        'transformers': transformers.__version__, 'threads': torch.get_num_threads()},
        'preservation': previous.REPORT['preservation'], 'original_parameter_function_ast_equal': True,
        'executions': executions, 'meta_parts': expected, 'meta_totals': totals, 'delta': totals[1] - totals[0],
        'bf16_weights_GiB': [p*2/1024**3 for p in totals], 'bf16_fixed_length_4096_KV_GiB': [2., .5],
        'tiny': tiny, 'complete_core': previous.REPORT['core'], 'boundaries': previous.REPORT['boundaries'],
        'paper_crops': crops, 'note_sha256': sha(ROOT / '解释.md'), 'script_sha256': sha(__file__),
        'not_run': ['Original 7B/8B checkpoint loading, inference and quality assessment', 'GPU memory or speed measurements', 'New optimizer training'],
    }
    (ROOT / '复现/outputs/基础讲解验证.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
    print(json.dumps({'status': report['status'], 'blocks': len(blocks), 'meta_totals': totals,
                      'delta': report['delta'], 'tiny': tiny, 'complete_core': report['complete_core']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
