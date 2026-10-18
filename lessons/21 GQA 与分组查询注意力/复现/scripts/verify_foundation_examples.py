"""Run current GQA teaching blocks and compare full Meta Attention with a per-head reference."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_HELPER = ROOT.parent / '20 KV cache 与增量推理/复现/scripts/verify_foundation_examples.py'
HELPER_NAME = '_lesson21_cpu_cache_loader'
if HELPER_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(HELPER_NAME, PREVIOUS_HELPER)
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[HELPER_NAME] = runtime
    spec.loader.exec_module(runtime)
else:
    runtime = sys.modules[HELPER_NAME]
cpu_llama_session = runtime.cpu_llama_session


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attention(module, groups, weights=None):
    args = module.ModelArgs(dim=8, n_heads=4, n_kv_heads=groups,
                            max_batch_size=1, max_seq_len=8)
    layer = module.Attention(args).double().eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0, .1)
    if weights is not None:
        layer.load_state_dict(weights)
    return layer


def rotate_components(x, positions):
    # The fixed implementation rotates through float32/complex64, then restores the input dtype.
    pair = x.float().reshape(*x.shape[:-1], -1, 2)
    angles = positions.float()[:, None] * torch.tensor([1.], dtype=torch.float32)[None, :]
    c, s = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    a, b = pair[..., 0], pair[..., 1]
    return torch.stack([a*c-b*s, a*s+b*c], -1).flatten(-2).to(x.dtype)


def per_head_reference(layer, x, wrong_order=False):
    b, length, _ = x.shape
    groups = layer.n_local_kv_heads
    q = (x @ layer.wq.weight.T).view(b, length, 4, 2)
    k = (x @ layer.wk.weight.T).view(b, length, groups, 2)
    v = (x @ layer.wv.weight.T).view(b, length, groups, 2)
    q = rotate_components(q, torch.arange(length))
    k = rotate_components(k, torch.arange(length))
    mask = torch.triu(torch.full((length, length), -float('inf')), diagonal=1)
    heads = []
    for head in range(4):
        group = head % groups if wrong_order else head // (4 // groups)
        score = q[:, :, head] @ k[:, :, group].transpose(-1, -2) / (2 ** .5)
        probability = (score + mask).float().softmax(-1).to(x.dtype)
        heads.append(probability @ v[:, :, group])
    joined = torch.stack(heads, dim=2).reshape(b, length, 8)
    return joined @ layer.wo.weight.T


def full_attention_checks(module):
    torch.manual_seed(210)
    x0 = torch.randn(1, 3, 8, dtype=torch.float64)
    weight = torch.arange(1, 25, dtype=torch.float64).reshape(1, 3, 8) / 24
    frequencies = module.precompute_freqs_cis(2, 8)
    mask = torch.triu(torch.full((3, 3), -float('inf')), diagonal=1)
    rows = []
    gqa_state = None
    for groups in (4, 2, 1):
        actual = attention(module, groups)
        reference = attention(module, groups, actual.state_dict())
        x, xr = x0.clone().requires_grad_(), x0.clone().requires_grad_()
        result = actual(x, 0, frequencies[:3], mask)
        expected = per_head_reference(reference, xr)
        forward_error = (result-expected).abs().max().item()
        torch.testing.assert_close(result, expected, atol=2e-7, rtol=2e-6)
        ga = torch.autograd.grad((result*weight).sum(), [x]+list(actual.parameters()))
        gr = torch.autograd.grad((expected*weight).sum(), [xr]+list(reference.parameters()))
        gradient_errors = {}
        for name, a, b in zip(['input']+list(dict(actual.named_parameters())), ga, gr):
            gradient_errors[name] = (a-b).abs().max().item()
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
        if groups == 2:
            gqa_state = {k: v.detach().clone() for k, v in actual.state_dict().items()}
            wrong = per_head_reference(reference, x0, wrong_order=True)
            wrong_error = (wrong-expected.detach()).abs().max().item()
            assert wrong_error > 1e-5
        rows.append({'kv_heads': groups, 'output_shape': list(result.shape),
                     'parameters': sum(p.numel() for p in actual.parameters()),
                     'forward_component_reference_error': forward_error,
                     'gradient_errors': gradient_errors})

    # Same full source, shared weights, correct absolute positions, and per-prefix causal masks.
    with torch.inference_mode():
        cache = attention(module, 2, gqa_state)
        dense = attention(module, 2, gqa_state)
        x6 = torch.randn(1, 6, 8, dtype=torch.float64)
        errors, read_shapes = [], []
        for end in (3, 4, 5, 6):
            start = 0 if end == 3 else end-1
            old_k = cache.cache_k[:, :start].clone()
            old_v = cache.cache_v[:, :start].clone()
            causal = torch.triu(torch.full((end, end), -float('inf')), diagonal=1)
            y = cache(x6[:, start:end], start, frequencies[start:end], mask if start == 0 else None)
            full = dense(x6[:, :end], 0, frequencies[:end], causal)[:, start:end]
            torch.testing.assert_close(y, full, atol=2e-7, rtol=2e-6)
            errors.append((y-full).abs().max().item())
            assert torch.equal(cache.cache_k[:, :start], old_k)
            assert torch.equal(cache.cache_v[:, :start], old_v)
            torch.testing.assert_close(cache.cache_k[:, :end], dense.cache_k[:, :end], atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(cache.cache_v[:, :end], dense.cache_v[:, :end], atol=2e-7, rtol=2e-6)
            read_shapes.append(list(cache.cache_k[:, :end].shape))

    # A specially tied MHA gives the same forward and summed KV gradients as GQA.
    gqa = attention(module, 2, gqa_state)
    mha = attention(module, 4)
    with torch.no_grad():
        mha.wq.weight.copy_(gqa.wq.weight)
        mha.wo.weight.copy_(gqa.wo.weight)
        for name in ('wk', 'wv'):
            getattr(mha, name).weight.copy_(getattr(gqa, name).weight.reshape(2, 2, 8)
                                            .repeat_interleave(2, 0).reshape(8, 8))
    xa, xb = x0.clone().requires_grad_(), x0.clone().requires_grad_()
    ya, yb = gqa(xa, 0, frequencies[:3], mask), mha(xb, 0, frequencies[:3], mask)
    torch.testing.assert_close(ya, yb, atol=2e-7, rtol=2e-6)
    ga = torch.autograd.grad((ya*weight).sum(), [xa]+list(gqa.parameters()))
    gb = torch.autograd.grad((yb*weight).sum(), [xb]+list(mha.parameters()))
    names = ['input']+list(dict(gqa.named_parameters()))
    tied_errors = {}
    for name, a, b in zip(names, ga, gb):
        if name in ('wk.weight', 'wv.weight'):
            b = b.reshape(2, 2, 2, 8).sum(1).reshape(4, 8)
        tied_errors[name] = (a-b).abs().max().item()
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)

    # General mean-pooling changes the computation; pooling the tied special case recovers GQA.
    original = attention(module, 4)
    pooled = attention(module, 2)
    recovered = attention(module, 2)
    with torch.no_grad():
        for target, source in [(pooled, original), (recovered, mha)]:
            target.wq.weight.copy_(source.wq.weight)
            target.wo.weight.copy_(source.wo.weight)
            for name in ('wk', 'wv'):
                getattr(target, name).weight.copy_(getattr(source, name).weight.reshape(2, 2, 2, 8)
                                                   .mean(1).reshape(4, 8))
        general_error = (original(x0, 0, frequencies[:3], mask)-pooled(x0, 0, frequencies[:3], mask)).abs().max().item()
        assert general_error > 1e-5
        assert torch.equal(recovered.wk.weight, gqa.wk.weight)
        assert torch.equal(recovered.wv.weight, gqa.wv.weight)
    return {'three_modes': rows, 'wrong_cyclic_group_output_error': wrong_error,
            'incremental_3_1_1_1_errors': errors, 'cache_read_shapes': read_shapes,
            'cache_preserves_old_prefix': True, 'persistent_cache_kv_heads': 2,
            'tied_mha_forward_error': (ya-yb).abs().max().item(),
            'tied_mha_summed_kv_gradient_errors': tied_errors,
            'general_mean_pool_output_change': general_error,
            'pooling_tied_heads_recovers_original_weights': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-source', default=runtime.DEFAULT_SOURCE)
    parser.add_argument('--fairscale', default=runtime.DEFAULT_FAIRSCALE)
    args = parser.parse_args()
    os.environ['LESSON20_META_SOURCE'] = args.meta_source
    os.environ['LESSON20_FAIRSCALE'] = args.fairscale
    os.chdir(ROOT.parents[1])
    torch.set_num_threads(1)
    note = ROOT / '解释.md'
    body = note.read_text(encoding='utf-8').split('<details>', 1)[0]
    blocks = re.findall(r'```python\n(.*?)```', body, re.S)
    assert len(blocks) == 5
    scope, outputs = {'__name__': '_lesson21_note_examples'}, []
    for index, block in enumerate(blocks, 1):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(block, f'lesson21-note-block-{index}', 'exec'), scope)
        outputs.append({'block': index, 'stdout': stream.getvalue()})
    source = Path(args.meta_source)
    original = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'repeat_kv')
    teaching = next(n for n in ast.parse(blocks[1]).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'repeat_kv')
    assert ast.dump(original, include_attributes=False) == ast.dump(teaching, include_attributes=False)
    assert [row[2] for row in scope['attention_rows']] == [256, 192, 160]
    assert [row[3] for row in scope['attention_rows']] == [512, 256, 128]
    assert torch.equal(scope['source'].grad.flatten(), torch.tensor([3., 7.]))
    assert torch.equal(scope['w_k_grouped'][:, 0, 0], torch.tensor([8., 40.]))
    with cpu_llama_session() as module:
        torch.testing.assert_close(scope['repeated'], module.repeat_kv(scope['kv'], 2), atol=0, rtol=0)
        assert module.repeat_kv(scope['kv'], 1) is scope['kv']
        checks = full_attention_checks(module)
    report = {'note_sha256': sha(note), 'script_sha256': sha(__file__),
              'cpu_loader_sha256': sha(PREVIOUS_HELPER), 'official_source_sha256': sha(source),
              'official_commit': '7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e',
              'cpu_allocation_edits': 2, 'attention_forward_or_repeat_body_edits': 0,
              'python': sys.version, 'torch': torch.__version__, 'blocks': outputs,
              'official_repeat_function_ast_equal': True,
              'teaching_rows': scope['attention_rows'], 'shared_kv_probabilities': scope['probabilities'].tolist(),
              'shared_kv_outputs': scope['outputs'].tolist(), 'repeat_gradient': scope['source'].grad.flatten().tolist(),
              'repeat_shape': list(scope['repeated'].shape),
              'expanded_view_shares_storage': scope['expanded'].untyped_storage().data_ptr() == scope['kv'].untyped_storage().data_ptr(),
              'reshape_shares_storage': scope['repeated'].untyped_storage().data_ptr() == scope['kv'].untyped_storage().data_ptr(),
              'pool_first_elements': scope['w_k_grouped'][:, 0, 0].tolist(), 'full_attention': checks,
              'scope': 'Five current blocks; complete fixed Meta Attention with two CPU cache allocation edits; random weights and synthetic hidden states. No pretrained Llama, T5 uptraining, task scores or GPU timing.'}
    target = ROOT / '复现/outputs/基础讲解验证.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'blocks': 5,
                      'parameters': [r[2] for r in scope['attention_rows']],
                      'cache_bytes': [r[3] for r in scope['attention_rows']],
                      'report': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
