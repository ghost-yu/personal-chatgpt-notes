# -*- coding: utf-8 -*-
"""Run the five reading blocks against complete official RoPE functions."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

LESSON = Path(__file__).resolve().parents[2]
NOTE = LESSON / '解释.md'
MODEL = Path('G:/zzz/Personal ChatGPT 系列/19 RoPE apply_rotary_emb/资料/llama-model.py')
FAIRSCALE = Path('G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local')
sys.path.insert(0, str(FAIRSCALE))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def no_doc(node):
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        node.body.pop(0)
    return ast.dump(node)


def real_rotation_reference(x, angles, inverse=False):
    """Independent scalar two-component equations, evaluated in float64."""
    paired = x.detach().double().reshape(*x.shape[:-1], -1, 2)
    a, b = paired[..., 0], paired[..., 1]
    angle = angles.double()[None, :, None, :]
    if inverse:
        angle = -angle
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(3)


def main():
    assert sha(MODEL) == '56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    note = NOTE.read_text(encoding='utf-8')
    blocks = re.findall(r'```python\n(.*?)```', note.split('<details>', 1)[0], re.S)
    assert len(blocks) == 5
    scope = {'__name__': 'lesson19_foundation'}
    trace = []
    for i, block in enumerate(blocks, 1):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(block, str(NOTE) + ':block' + str(i), 'exec'), scope)
        trace.append({'block': i, 'stdout': capture.getvalue()})
    s = scope
    xq, xk = s['xq'], s['xk']
    assert list(xq.shape) == [1, 2, 2, 4] and list(xk.shape) == [1, 2, 1, 4]
    assert list(s['paired_q'].shape) == [1, 2, 2, 2, 2]
    assert list(s['complex_q'].shape) == [1, 2, 2, 2]
    assert list(s['broadcast_freqs'].shape) == [1, 2, 1, 2]
    assert s['broadcast_freqs'].data_ptr() == s['freqs_cis'].data_ptr()
    torch.testing.assert_close(s['manual_q'], s['q_rotated'], atol=0, rtol=0)
    wrong_axis_error = (s['wrong_q'] - s['q_rotated']).abs().max().item()
    assert s['wrong_q'].shape == s['q_rotated'].shape and wrong_axis_error > 1
    torch.testing.assert_close(s['wrong_q'][0, 1, 0], xq[0, 1, 0], atol=0, rtol=0)

    path = LESSON / '代码/tutorials/llama2_src_03_RoPE_apply_rotary_emb.ipynb'
    assert sha(path) == '59fb6124d4c14dfe40c1cb1053a43458f0b745183447195abab18e3609d94f1e'
    raw = json.loads(path.read_bytes())
    original_apply = ast.parse(''.join(raw['cells'][16]['source'])).body[0]
    body_functions = {n.name: n for n in ast.parse(blocks[1]).body if isinstance(n, ast.FunctionDef)}
    assert ast.dump(original_apply) == ast.dump(body_functions['apply_rotary_emb'])
    curve_original = ast.parse(''.join(raw['cells'][23]['source'])).body[0]
    curve_example = next(n for n in ast.parse(blocks[4]).body if isinstance(n, ast.FunctionDef))
    assert ast.dump(curve_original) == ast.dump(curve_example)

    spec = importlib.util.spec_from_file_location('lesson19_fixed_meta', MODEL)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    official_tree = ast.parse(MODEL.read_text(encoding='utf-8'))
    for name in ['apply_rotary_emb', 'reshape_for_broadcast']:
        node = next(n for n in official_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert no_doc(node) == ast.dump(body_functions[name])
    official_table = official.precompute_freqs_cis(4, 2)
    torch.testing.assert_close(official_table, s['freqs_cis'], atol=0, rtol=0)
    q_before, k_before, f_before = xq.detach().clone(), xk.detach().clone(), s['freqs_cis'].clone()
    oq = xq.detach().clone().requires_grad_()
    ok = xk.detach().clone().requires_grad_()
    outq, outk = official.apply_rotary_emb(oq, ok, official_table)
    torch.testing.assert_close(outq, s['q_rotated'], atol=0, rtol=0)
    torch.testing.assert_close(outk, s['k_rotated'], atol=0, rtol=0)
    official_score = torch.dot(outq[0, 0, 0], outk[0, 1, 0])
    official_score.backward()
    torch.testing.assert_close(oq.grad, xq.grad, atol=0, rtol=0)
    torch.testing.assert_close(ok.grad, xk.grad, atol=0, rtol=0)
    torch.testing.assert_close(s['score'], s['shifted_score'], atol=3e-6, rtol=1e-6)
    assert torch.equal(xq.detach(), q_before) and torch.equal(xk.detach(), k_before)
    assert torch.equal(s['freqs_cis'], f_before)
    q_ref = real_rotation_reference(xq, s['angles'])
    k_ref = real_rotation_reference(xk, s['angles'])
    torch.testing.assert_close(outq.detach().double(), q_ref, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(outk.detach().double(), k_ref, atol=2e-6, rtol=1e-6)

    # Probe the original definition separately before supplying its missing helper.
    original_scope = {'torch': torch, 'Tuple': Tuple}
    exec(compile(''.join(raw['cells'][16]['source']), 'original-cell16', 'exec'), original_scope)
    try:
        original_scope['apply_rotary_emb'](q_before, k_before, official_table)
    except NameError as e:
        missing_helper = str(e)
        assert 'reshape_for_broadcast' in missing_helper
    else:
        raise AssertionError('expected undefined original helper')
    original_scope['reshape_for_broadcast'] = official.reshape_for_broadcast
    restored_q, restored_k = original_scope['apply_rotary_emb'](q_before, k_before, official_table)
    torch.testing.assert_close(restored_q, outq, atol=0, rtol=0)
    torch.testing.assert_close(restored_k, outk, atol=0, rtol=0)

    # Cover every output component with a fixed linear loss and inverse rotation.
    gq = (torch.arange(xq.numel()).reshape_as(xq).float() + 1) / 10
    gk = (torch.arange(xk.numel()).reshape_as(xk).float() + 1) / 10
    iq, ik = q_before.clone().requires_grad_(), k_before.clone().requires_grad_()
    yq, yk = official.apply_rotary_emb(iq, ik, official_table)
    ((yq * gq).sum() + (yk * gk).sum()).backward()
    grad_q_ref = real_rotation_reference(gq, s['angles'], inverse=True)
    grad_k_ref = real_rotation_reference(gk, s['angles'], inverse=True)
    torch.testing.assert_close(iq.grad.double(), grad_q_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ik.grad.double(), grad_k_ref, atol=1e-6, rtol=1e-6)

    # Caller slices the absolute positions, including nonzero start_pos.
    table10 = official.precompute_freqs_cis(4, 10)
    a, b = official.apply_rotary_emb(q_before, k_before, table10[3:5])
    expected_angles = torch.outer(torch.arange(3, 5).float(), s['omega'])
    torch.testing.assert_close(a.double(), real_rotation_reference(q_before, expected_angles), atol=2e-6, rtol=1e-6)
    position_reset_error = (a - outq.detach()).abs().max().item()
    assert position_reset_error > 1

    dtype_cases = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        dq = q_before.detach().clone().to(dtype).requires_grad_()
        dk = k_before.detach().clone().to(dtype).requires_grad_()
        aq, ak = official.apply_rotary_emb(dq, dk, official_table)
        (aq.float().sum() + ak.float().sum()).backward()
        assert aq.dtype == ak.dtype == dtype and torch.isfinite(dq.grad).all() and torch.isfinite(dk.grad).all()
        dtype_cases.append({'input_output_dtype': str(dtype), 'finite_gradients': True})
    strided = torch.arange(32, dtype=torch.float32).reshape(1, 2, 2, 8)[..., ::2]
    try:
        official.apply_rotary_emb(strided, strided, official_table)
    except RuntimeError as e:
        stride_error = str(e)
    else:
        raise AssertionError('stride2 should fail complex view')
    official.apply_rotary_emb(strided.contiguous(), strided.contiguous(), official_table)

    course_curve, paper_curve = s['course_curve'], s['paper_range_curve']
    abs_sums = np.abs(s['partial_sums'])
    independent_course = abs_sums[:, :-1].sum(axis=1) / s['P']
    np.testing.assert_allclose(course_curve, independent_course, atol=2e-13, rtol=1e-13)
    np.testing.assert_allclose(paper_curve - course_curve, abs_sums[:, -1] / s['P'], atol=2e-13, rtol=1e-13)
    assert course_curve[0] == 31.5 and paper_curve[0] == 32.5
    increases = int(np.count_nonzero(np.diff(course_curve) > 0))
    assert increases == 113 and abs(course_curve[-1] - 6.412620) < 1e-6
    assert s['s'](0) / 32 == 63 and s['s'](0, d=64) / 32 == 15.5
    # Check the cited Abel identity and bound, including h_P=0.
    index = np.arange(s['P'])
    h = index / s['P'] + 1j * ((index % 5) - 2) / 3
    differences = np.r_[h[1:], 0] - h
    content_sum = s['phase_steps'] @ h
    abel_sum = -(s['partial_sums'] * differences[None, :]).sum(axis=1)
    bound = np.max(np.abs(differences)) * abs_sums.sum(axis=1)
    np.testing.assert_allclose(content_sum, abel_sum, atol=2e-12, rtol=1e-13)
    assert np.all(np.abs(content_sum) <= bound + 2e-12)
    r_scope = {'np': np}
    exec(''.join(raw['cells'][6]['source']), r_scope)
    R = r_scope['R']
    np.testing.assert_allclose(R(30) @ R(60), R(90), atol=1e-15, rtol=0)
    np.testing.assert_allclose(R(30).T @ R(60), R(30), atol=1e-15, rtol=0)

    # Read the original tokenizer using the notebook's own relative path.
    tokenizer = LESSON / '代码/tutorials/tokenizer.model'
    assert sha(tokenizer) == '9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'
    cwd = Path.cwd()
    tokenizer_scope = {}
    try:
        os.chdir(tokenizer.parent)
        exec(''.join(raw['cells'][18]['source']), tokenizer_scope)
        tree = ast.parse(''.join(raw['cells'][19]['source']))
        exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), 'original-cell19-prefix', 'exec'), tokenizer_scope)
        ids = eval(compile(ast.Expression(tree.body[-1].value), 'original-cell19-output', 'eval'), tokenizer_scope)
    finally:
        os.chdir(cwd)
    assert ids == [1, 306, 4658, 278, 6593, 310, 2834, 338]

    result = dict(date_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, device='cpu', blocks=trace,
        official_model_sha256=sha(MODEL), original_notebook_sha256=sha(path),
        note_sha256=sha(NOTE), verifier_sha256=sha(Path(__file__)),
        original_apply_ast_identical=True, official_apply_helper_ast_identical_ignoring_docstrings=True,
        original_phase_sum_ast_identical=True, official_frequency_exact=True,
        q_shape=list(xq.shape), k_shape=list(xk.shape), paired_shape=list(s['paired_q'].shape),
        complex_shape=list(s['complex_q'].shape), broadcast_shape=list(s['broadcast_freqs'].shape),
        broadcast_view_shares_storage=True, inputs_unchanged=True,
        q_output=outq.detach().tolist(), k_output=outk.detach().tolist(),
        forward_reference_max_error=(outq.detach().double() - q_ref).abs().max().item(),
        wrong_axis_max_error=wrong_axis_error, wrong_axis_same_shape=True,
        score=s['score'].item(), shifted_score=s['shifted_score'].item(),
        score_q_gradient=xq.grad.tolist(), score_k_gradient=xk.grad.tolist(), official_gradients_exact=True,
        full_q_inverse_gradient_max_error=(iq.grad.double() - grad_q_ref).abs().max().item(),
        full_k_inverse_gradient_max_error=(ik.grad.double() - grad_k_ref).abs().max().item(),
        nonzero_start_pos=3, nonzero_position_reset_max_error=position_reset_error,
        dtype_cases=dtype_cases, invalid_stride_error=stride_error, contiguous_fixed=True,
        original_missing_helper_error=missing_helper, original_with_official_helper_exact=True,
        tokenizer_ids=ids, degree_matrix_composition_checked=True,
        phase_curve=dict(course_first=float(course_curve[0]), course_last=float(course_curve[-1]),
            paper_range_first=float(paper_curve[0]), paper_range_last=float(paper_curve[-1]), increases=increases,
            independent_course_max_error=float(np.max(np.abs(course_curve - independent_course))),
            endpoint_identity_max_error=float(np.max(np.abs((paper_curve-course_curve)-abs_sums[:,-1]/s['P']))),
            abel_identity_max_error=float(np.max(np.abs(content_sum - abel_sum))),
            cited_bound_checked=True, bound_positions=256, bound_pairs=64),
        scope='正文五块、完整固定官方三个RoPE函数、原分词两格与数学；未重跑旧15格整notebook或模型任务')
    output = LESSON / '复现/outputs/基础讲解验证.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['q_shape', 'k_shape', 'broadcast_shape', 'score', 'shifted_score',
        'wrong_axis_max_error', 'forward_reference_max_error', 'full_q_inverse_gradient_max_error',
        'full_k_inverse_gradient_max_error', 'phase_curve', 'tokenizer_ids']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
