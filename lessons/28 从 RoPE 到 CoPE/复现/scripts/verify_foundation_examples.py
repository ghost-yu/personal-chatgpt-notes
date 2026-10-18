"""Run all six published blocks, original complete classes and independent references."""
from pathlib import Path
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import sys
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PDF = Path(os.environ.get('LESSON28_PDF', 'G:/zzz/Personal ChatGPT 系列/28 从 RoPE 到 CoPE/原论文.pdf'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(s):
    return ast.dump(ast.parse(s))


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('lesson28_previous', ROOT / '复现/scripts/verify_delivery.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    cells, _, preservation = previous.audit()
    note = (ROOT / '解释.md').read_text('utf-8').split('<details>', 1)[0]
    blocks = re.findall(r'^```python\n(.*?)^```', note, re.M | re.S)
    assert len(blocks) == 6
    definitions = {n.name: n for b in blocks for n in ast.parse(b).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for i, name in [(11, 'precompute_freqs_cis'), (23, 'get_rope_matrix')]:
        original = next(n for n in ast.parse(previous.src(cells[i])).body if isinstance(n, ast.FunctionDef))
        assert ast.dump(definitions[name]) == ast.dump(original), name
    original_classes = ast.parse(previous.blocks(previous.src(cells[39]))[0]).body
    for node in original_classes:
        assert ast.dump(definitions[node.name]) == ast.dump(node), node.name
    scope = {'__name__': '__foundation_examples__'}
    executions = []
    for i, block in enumerate(blocks, 1):
        out = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out):
            warnings.simplefilter('always')
            exec(compile(block, f'解释.md:block{i}', 'exec'), scope)
        executions.append({'block': i, 'stdout': out.getvalue(), 'warnings': [str(x.message) for x in caught]})
    q, k, v, mask = (scope[n] for n in ['q', 'k', 'v', 'mask'])
    # Independent scalar NumPy ordinary attention, without torch softmax/matmul.
    expected = np.zeros((1, 3, 2))
    weights = np.zeros((1, 3, 3))
    for i in range(3):
        logits = [sum(float(q[0, i, d] * k[0, j, d]) for d in range(2)) / math.sqrt(2) for j in range(i + 1)]
        exp = [math.exp(x - max(logits)) for x in logits]
        for j, x in enumerate(exp):
            weights[0, i, j] = x / sum(exp)
            for d in range(2):
                expected[0, i, d] += x / sum(exp) * float(v[0, j, d])
    np.testing.assert_allclose(scope['plain_weights'].numpy(), weights, atol=1e-15)
    np.testing.assert_allclose(scope['plain_output'].numpy(), expected, atol=1e-15)
    assert torch.equal(scope['plain_output'][0, 0], v[0, 0])
    # Independent real half-layout rotation for the original float32 matrix.
    def rotate(x, position):
        out = np.empty(4, dtype=np.float64)
        for r, omega in enumerate([1., .01]):
            c, s = math.cos(position * omega), math.sin(position * omega)
            out[r] = c * float(x[r]) - s * float(x[r + 2])
            out[r + 2] = s * float(x[r]) + c * float(x[r + 2])
        return out
    rq, rk = scope['rq'], scope['rk']
    independent_dot = float(rotate(rq, 2) @ rotate(rk, 5))
    independent_relative = float(rq.double().numpy() @ rotate(rk, 3))
    assert abs(independent_dot - independent_relative) < 1e-12
    assert abs(scope['left_score'].item() - independent_dot) < 4e-6
    torch.testing.assert_close(scope['left_score'], scope['relative_score'])
    phases = scope['phases']
    angles = torch.tensor([[0., 0.], [1., .01], [2., .02]])
    torch.testing.assert_close(phases, torch.complex(angles.cos(), angles.sin()))
    assert phases.shape == (3, 2) and phases.dtype == torch.complex64
    original_orthogonality_error = (scope['rm'].T @ scope['rm'] - torch.eye(4)).abs().max().item()
    assert original_orthogonality_error < 2e-7
    # Complete ideal sentence diagram, suffix sum and inclusive all-ones convention.
    assert scope['sentence_positions'].tolist() == [2., 2., 2., 2., 1., 1., 1., 1., 0., 0., 0., 0.]
    assert scope['soft_positions'].tolist() == [1.5, 1.5, 1.25, .75]
    assert [sum(scope['soft_gates'][j:]).item() for j in range(4)] == scope['soft_positions'].tolist()
    ones = torch.ones(4)
    assert ones.flip(-1).cumsum(-1).flip(-1).tolist() == [4., 3., 2., 1.]
    # Direct vector interpolation and analytic gradients.
    assert scope['position_vector'].tolist() == [3., 1.]
    assert scope['position_bias'].item() == 5 and scope['position'].grad.item() == 12
    torch.testing.assert_close(scope['position_table'].grad,
        torch.tensor([[0., 0.], [.75, 1.5], [.25, .5], [0., 0.]], dtype=torch.float64))
    # Real CoPE reaches p=1.25 as sigmoid(0)+sigmoid(log(3)).
    CoPE, SelfAttn = scope['CoPE'], scope['SelfAttn']
    hand_module = CoPE(4, 2).double()
    with torch.no_grad():
        hand_module.pos_emb.copy_(scope['position_table'].detach().T[None])
    hand_logits = torch.tensor([[[0., math.log(3.)]]], dtype=torch.float64, requires_grad=True)
    hand_q = scope['one_query'].reshape(1, 1, 2)
    hand_bias = hand_module(hand_q, hand_logits)[0, 0, 0]
    torch.testing.assert_close(hand_bias, torch.tensor(5., dtype=torch.float64))
    hand_gradient = torch.autograd.grad(hand_bias, hand_logits)[0]
    torch.testing.assert_close(hand_gradient, torch.tensor([[[3., 2.25]]], dtype=torch.float64))
    # Five actual Adam steps are present in optimizer state, not inferred from loop length.
    model, optimizer = scope['model'], scope['optimizer']
    assert sum(p.numel() for p in model.parameters()) == 8
    assert list(dict(model.named_parameters())) == ['cope.pos_emb']
    assert len(optimizer.state) == 1 and all(s['step'].item() == 5 for s in optimizer.state.values())
    losses = scope['loss_history']
    assert len(losses) == 6 and losses[-1] < losses[0]
    assert model.cope.pos_emb.detach().abs().sum() > 0
    assert torch.equal(scope['initial'], scope['plain_output'])
    assert torch.equal(scope['final_output'][0, 0], v[0, 0])
    assert all(math.isfinite(x) for x in losses)
    # Two distinct evaluation graphs: optimized scalar interpolation versus full vectors.
    qq, kk, vv = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    output = model(qq, kk, vv, mask)
    dense = previous.dense_reference(qq, kk, vv, model.cope.pos_emb, mask)
    torch.testing.assert_close(output, dense, atol=1e-12, rtol=1e-12)
    coefficient = torch.arange(1, output.numel() + 1, dtype=torch.float64).reshape_as(output)
    arguments = (qq, kk, vv, model.cope.pos_emb)
    actual_grad = torch.autograd.grad((output * coefficient).sum(), arguments)
    dense_grad = torch.autograd.grad((dense * coefficient).sum(), arguments)
    grad_errors = []
    for a, b in zip(actual_grad, dense_grad):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
        grad_errors.append((a - b).abs().max().item())
    embedding = model.cope.pos_emb.detach().clone().requires_grad_()
    fn = lambda a, b, c, e: torch.func.functional_call(model, {'cope.pos_emb': e}, (a, b, c, mask))
    gradcheck = torch.autograd.gradcheck(fn, (qq, kk, vv, embedding), eps=1e-6, atol=2e-5, rtol=1e-4)
    assert gradcheck
    # Zero initialization keeps ordinary attention, while the table receives gradients.
    zero_module = CoPE(4, 2).double()
    unmasked = scope['content_logits'].detach().clone().requires_grad_()
    zero_bias = zero_module(q, unmasked)
    zero_gate_grad, zero_table_grad = torch.autograd.grad(zero_bias.sum(), (unmasked, zero_module.pos_emb))
    assert torch.equal(zero_gate_grad, torch.zeros_like(zero_gate_grad))
    assert zero_table_grad.abs().sum() > 0
    learned_gate_grad = torch.autograd.grad(model.cope(q, unmasked).sum(), unmasked)[0]
    assert learned_gate_grad.abs().max() > 0
    # Integer interpolation knots and clamp saturation are checked independently.
    knots = CoPE(4, 1).double()
    with torch.no_grad():
        knots.pos_emb.copy_(torch.tensor([[[0., 1., 3., 5.]]], dtype=torch.float64))
    logit0 = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
    knot_grad = torch.autograd.grad(knots(torch.ones(1, 1, 1, dtype=torch.float64), logit0)[0, 0, 0], logit0)[0]
    assert torch.equal(knot_grad, torch.zeros_like(knot_grad))
    capped = CoPE(2, 1).double()
    with torch.no_grad():
        capped.pos_emb.copy_(torch.tensor([[[0., 1.]]], dtype=torch.float64))
    large = torch.full((1, 1, 2), 10., dtype=torch.float64, requires_grad=True)
    cap_bias = capped(torch.ones(1, 1, 1, dtype=torch.float64), large)[0, 0, 0]
    cap_grad = torch.autograd.grad(cap_bias, large)[0]
    assert cap_bias.item() == 1 and torch.equal(cap_grad, torch.zeros_like(cap_grad))
    prefix_errors = []
    with torch.no_grad():
        full = model(q, k, v, mask)
        for end in range(1, 4):
            prefix = model(q[:, :end], k[:, :end], v[:, :end], mask[:end, :end])
            error = (full[:, :end] - prefix).abs().max().item()
            assert error < 1e-12
            prefix_errors.append(error)
        cq, ck, cv = [x.clone() for x in (q, k, v)]
        cq[:, 2] += 100
        ck[:, 2] -= 100
        cv[:, 2] += 100
        changed = model(cq, ck, cv, mask)
        assert torch.equal(changed[:, :2], full[:, :2])
        all_masked = model(q, k, v, torch.zeros_like(mask))
        assert torch.isnan(all_masked).all()
    figures = json.loads((ROOT / '复现/outputs/论文截图来源.json').read_text('utf-8'))
    for fig in figures:
        assert sha(PDF) == fig['pdf_sha256']
        assert sha(ROOT / fig['published_path']) == fig['png_sha256']
        assert '](' + fig['published_path'] + ')' in note
    result = {
        'status': 'passed', 'date': '2026-10-18',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'threads': 1},
        'note_sha256': sha(ROOT / '解释.md'), 'script_sha256': sha(__file__), 'pdf_sha256': sha(PDF),
        'preservation': preservation,
        'complete_original_functions_ast_equal': ['precompute_freqs_cis', 'get_rope_matrix'],
        'complete_original_classes_ast_equal': ['CoPE', 'SelfAttn'],
        'published_blocks': executions,
        'ordinary_attention': {'weights': scope['plain_weights'].tolist(), 'output': scope['plain_output'].tolist(), 'independent_scalar_numpy_equal': True},
        'RoPE': {'phase_shape': list(phases.shape), 'phase_dtype': str(phases.dtype), 'left_dot': scope['left_score'].item(),
                 'relative_dot': scope['relative_score'].item(), 'independent_double_dot': independent_dot,
                 'orthogonality_max_error': original_orthogonality_error},
        'counting': {'sentence_positions': scope['sentence_positions'].tolist(), 'soft_positions': scope['soft_positions'].tolist(), 'all_gates_one_counts_inclusive': True},
        'interpolation': {'vector': scope['position_vector'].tolist(), 'bias': 5., 'position_derivative': 12.,
                          'real_CoPE_bias': hand_bias.item(), 'real_CoPE_logit_gradient': hand_gradient.tolist()},
        'training': {'parameters': 8, 'Adam_updates': 5, 'optimizer_steps_verified_from_state': True,
                     'losses': losses, 'initial_equals_plain_exactly': True, 'first_row_output_unchanged': True,
                     'only_position_table_optimized': True, 'QKV_fixed': True},
        'independent_reference': {'forward_max_error': (output - dense).abs().max().item(),
                                  'Q_K_V_table_gradient_max_errors': grad_errors, 'double_gradcheck_all_four_inputs': gradcheck},
        'gradient_paths': {'zero_table_gate_gradient_zero': True, 'zero_table_table_gradient_nonzero': True,
                           'trained_table_gate_gradient_max': learned_gate_grad.abs().max().item(),
                           'integer_knot_gate_gradient': knot_grad.tolist(), 'saturated_gate_gradient': cap_grad.tolist()},
        'causality': {'full_vs_prefix_errors': prefix_errors, 'future_does_not_change_prefix': True, 'all_masked_rows_NaN': True},
        'paper_crops_verified': len(figures),
        'scope': 'Six CPU examples, complete original notebook functions and Appendix-B classes; no full paper tasks, language model, GPU or KV-cache benchmark.'
    }
    (ROOT / '复现/outputs/基础讲解验证.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps({'status': 'passed', 'blocks': len(blocks), 'paper_crops': len(figures), 'losses': losses,
                      'forward_error': result['independent_reference']['forward_max_error'], 'gradient_errors': grad_errors}, ensure_ascii=False))


if __name__ == '__main__':
    main()
