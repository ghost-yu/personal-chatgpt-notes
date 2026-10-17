# -*- coding: utf-8 -*-
"""Run the reading examples, official frequency function, and independent rotations."""
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
from datetime import datetime, timezone
from pathlib import Path

import torch

LESSON = Path(__file__).resolve().parents[2]
NOTE = LESSON / '解释.md'
MODEL = Path('G:/zzz/Personal ChatGPT 系列/18 RoPE 复数形式/资料/llama-model.py')
FAIRSCALE = Path('G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local')
sys.path.insert(0, str(FAIRSCALE))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rotation_matrix(position, omega):
    blocks = []
    for angle in position * omega:
        c, s = math.cos(angle.item()), math.sin(angle.item())
        blocks.append(torch.tensor([[c, -s], [s, c]], dtype=torch.float64))
    return torch.block_diag(*blocks)


def main():
    assert sha(MODEL) == '56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    main_text = NOTE.read_text(encoding='utf-8').split('<details>', 1)[0]
    blocks = re.findall(r'```python\n(.*?)```', main_text, re.S)
    assert len(blocks) == 4
    scope = {'__name__': 'lesson18_foundation'}
    trace = []
    for i, block in enumerate(blocks, 1):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(block, str(NOTE) + ':block' + str(i), 'exec'), scope)
        trace.append({'block': i, 'stdout': capture.getvalue()})
    s = scope
    assert abs(s['z_rotated'] - (-2 + 1j)) < 1e-14
    assert abs(abs(s['z_rotated']) - abs(s['z'])) < 1e-14
    assert list(s['freqs_cis'].shape) == [4, 2] and s['freqs_cis'].dtype == torch.complex64
    torch.testing.assert_close(s['freqs_cis'][0], torch.ones(2, dtype=torch.complex64), atol=0, rtol=0)
    torch.testing.assert_close(s['real_pe'], s['reconstructed_pe'], atol=1e-7, rtol=1e-6)
    reconstruction_error = (s['real_pe'] - s['reconstructed_pe']).abs().max().item()
    real_view, table = s['real_view'], s['freqs_cis']
    assert real_view.data_ptr() == table.data_ptr()
    assert real_view.numel() * real_view.element_size() == table.numel() * table.element_size() == 64
    flipped = torch.flip(real_view, dims=(2,))
    assert flipped.data_ptr() != table.data_ptr()
    assert sum(p.numel() for p in s['spe'].parameters()) == 0

    raw = json.loads((LESSON / '代码/tutorials/llama2_src_03_RoPE_01_complex_position_encoding.ipynb').read_bytes())
    original_function = ast.parse(''.join(raw['cells'][19]['source'])).body[0]
    example_function = next(n for n in ast.parse(blocks[1]).body if isinstance(n, ast.FunctionDef))
    assert ast.dump(original_function) == ast.dump(example_function)
    original_class = ast.parse(''.join(raw['cells'][14]['source'])).body[0]
    example_class = next(n for n in ast.parse(blocks[2]).body if isinstance(n, ast.ClassDef))
    assert ast.dump(original_class) == ast.dump(example_class)
    official = load_module(MODEL, 'lesson18_fixed_meta')
    official_table = official.precompute_freqs_cis(4, 4)
    torch.testing.assert_close(table, official_table, atol=0, rtol=0)
    official_fn = next(n for n in ast.parse(MODEL.read_text(encoding='utf-8')).body
                       if isinstance(n, ast.FunctionDef) and n.name == 'precompute_freqs_cis')
    # Only the notebook's two shape prints and official docstring differ.
    def mathematical_ast(node):
        node.body = [n for n in node.body if not (
            isinstance(n, ast.Expr) and (isinstance(n.value, ast.Constant) or (
                isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'print')))]
        return ast.dump(node)
    assert mathematical_ast(original_function) == mathematical_ast(official_fn)

    q, k, omega, m, n = (s[key] for key in ['q', 'k', 'omega', 'm', 'n'])
    rm, rn = rotation_matrix(m, omega), rotation_matrix(n, omega)
    torch.testing.assert_close(s['q_rotated'].detach(), rm @ q.detach(), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(s['k_rotated'].detach(), rn @ k.detach(), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(s['score'], s['shifted_score'], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(s['score'], s['relative_score'], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(q.grad, rm.T @ rn @ k.detach(), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(k.grad, rn.T @ rm @ q.detach(), atol=1e-14, rtol=1e-14)
    norm_error = abs(q.norm().item() - s['q_rotated'].norm().item())
    assert norm_error < 1e-14
    # The 90-degree hand example uses omega=pi/2 explicitly.
    hand_q, hand_k = torch.tensor([1., 2.]), torch.tensor([2., 1.])
    assert torch.dot(hand_q, torch.tensor([-1., 2.])).item() == 3
    assert torch.dot(torch.tensor([-2., 1.]), -hand_k).item() == 3

    default_args = official.ModelArgs()
    default_table = official.precompute_freqs_cis(default_args.dim // default_args.n_heads,
                                                 default_args.max_seq_len * 2)
    assert list(default_table.shape) == [4096, 64]
    assert default_table.numel() * default_table.element_size() == 2097152
    long_table = official.precompute_freqs_cis(4, 12)
    sliced = long_table[7:10]
    expected_slice = torch.polar(torch.ones(3, 2), torch.outer(torch.arange(7, 10).float(),
                                              torch.tensor([1., .01])))
    torch.testing.assert_close(sliced, expected_slice, atol=0, rtol=0)
    reset_error = (sliced - long_table[:3]).abs().max().item()
    assert reset_error > .6
    wrong_frequencies = 1. / (10000 ** torch.arange(0, 4, 2))
    assert abs(wrong_frequencies[1].item() - 1e-8) < 1e-14

    # Run the complete original second class in a separate namespace.
    bug_scope = {'torch': torch, 'nn': torch.nn, 'd_model': 6}
    bad_source = ''.join(raw['cells'][26]['source'])
    exec(compile(bad_source, 'original-cell26', 'exec'), bug_scope)
    bad = bug_scope['SinPositionEncoding'](4, 4)
    correct = s['SinPositionEncoding'](4, 4)()
    wrong_global_error = (bad() - correct).abs().max().item()
    assert wrong_global_error > .1
    old_bad = bad()
    bug_scope['d_model'] = 12
    changed_global_error = (bad() - old_bad).abs().max().item()
    assert changed_global_error > .1
    del bug_scope['d_model']
    try:
        bad()
    except NameError:
        missing_global = True
    else:
        raise AssertionError('expected original class to require the missing global')

    class FixForwardGlobal(ast.NodeTransformer):
        count = 0
        def visit_FunctionDef(self, node):
            return self.generic_visit(node) if node.name == 'forward' else node
        def visit_Name(self, node):
            if node.id == 'd_model' and isinstance(node.ctx, ast.Load):
                self.count += 1
                return ast.copy_location(ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                      attr='d_model', ctx=ast.Load()), node)
            return node
    fixer = FixForwardGlobal()
    fixed_tree = ast.fix_missing_locations(fixer.visit(ast.parse(bad_source)))
    assert fixer.count == 1
    fixed_scope = {'torch': torch, 'nn': torch.nn}
    exec(compile(fixed_tree, 'teaching-one-name-fix', 'exec'), fixed_scope)
    fixed_errors = {}
    for dim in (2, 4, 6, 8):
        a = fixed_scope['SinPositionEncoding'](4, dim)()
        b = s['SinPositionEncoding'](4, dim)()
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
        fixed_errors[dim] = (a - b).abs().max().item()
    try:
        s['SinPositionEncoding'](4, 5)()
    except RuntimeError as e:
        odd_real_error = str(e)
    else:
        raise AssertionError('odd real PE should fail stack')
    bug_scope['d_model'] = 5
    assert list(bug_scope['SinPositionEncoding'](4, 5)().shape) == [4, 6]
    assert list(official.precompute_freqs_cis(5, 4).shape) == [4, 2]

    screenshot = load_module(LESSON / '代码/配图_RoPE源码阅读.py', 'lesson18_screenshot_functions')
    perm = torch.tensor([0, 2, 1, 3])
    layout_table = official.precompute_freqs_cis(4, 7)
    ids = torch.tensor([[0, 1, 3], [2, 4, 6]])
    sample_q = torch.arange(48, dtype=torch.float64).reshape(2, 2, 3, 4) / 10
    sample_k = torch.arange(24, dtype=torch.float64).reshape(2, 1, 3, 4) / 10
    cos = torch.cat([layout_table.real, layout_table.real], dim=-1)[None, None]
    sin = torch.cat([layout_table.imag, layout_table.imag], dim=-1)[None, None]
    half_q, half_k = screenshot.apply_rotary_pos_emb(sample_q[..., perm], sample_k[..., perm], cos, sin, ids)
    def adjacent_reference(value):
        z = torch.view_as_complex(value.reshape(*value.shape[:-1], 2, 2))
        return torch.view_as_real(z * layout_table[ids][:, None]).flatten(-2)
    torch.testing.assert_close(half_q[..., perm], adjacent_reference(sample_q), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(half_k[..., perm], adjacent_reference(sample_k), atol=1e-12, rtol=1e-12)

    result = dict(date_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  torch=torch.__version__, device='cpu', official_model_sha256=sha(MODEL),
                  note_sha256=sha(NOTE), verifier_sha256=sha(Path(__file__)), blocks=trace,
                  notebook_function_ast_identical=True, notebook_real_class_ast_identical=True,
                  official_math_ast_identical=True, official_precompute_exact=True,
                  table_real_imag=torch.view_as_real(table).tolist(), shape=list(table.shape),
                  table_bytes=64, real_view_bytes=64, real_view_shares_storage=True, flip_new_storage=True,
                  reconstructed_pe_max_error=reconstruction_error,
                  score=s['score'].item(), shifted_score=s['shifted_score'].item(),
                  relative_score=s['relative_score'].item(), norm_max_error=norm_error,
                  q_gradient=q.grad.tolist(), k_gradient=k.grad.tolist(),
                  q_gradient_matrix_max_error=(q.grad - rm.T @ rn @ k.detach()).abs().max().item(),
                  k_gradient_matrix_max_error=(k.grad - rn.T @ rm @ q.detach()).abs().max().item(),
                  default_table_shape=list(default_table.shape), default_table_bytes=2097152,
                  slice_shape=list(sliced.shape), reset_position_max_error=reset_error,
                  missing_dim_second_frequency=wrong_frequencies[1].item(),
                  original_second_class=dict(global6_vs_instance4_max_error=wrong_global_error,
                      global6_to12_max_change=changed_global_error, missing_global_name_error=missing_global,
                      corrected_name_count=fixer.count, fixed_class_errors=fixed_errors),
                  odd_real_error=odd_real_error, odd_original_complex_shape=[4, 6], odd_official_shape=[4, 2],
                  screenshot_layout=dict(q_shape=list(sample_q.shape), k_shape=list(sample_k.shape),
                      permutation=perm.tolist(), position_ids=ids.tolist(),
                      q_max_error=(half_q[..., perm] - adjacent_reference(sample_q)).abs().max().item(),
                      k_max_error=(half_k[..., perm] - adjacent_reference(sample_k)).abs().max().item()),
                  scope='正文四代码块、完整官方频率函数、数学旋转及原截图转录；不重跑旧整notebook或模型任务')
    output = LESSON / '复现/outputs/基础讲解验证.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['shape', 'score', 'shifted_score', 'relative_score',
        'q_gradient_matrix_max_error', 'k_gradient_matrix_max_error', 'reconstructed_pe_max_error',
        'default_table_shape', 'screenshot_layout']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
