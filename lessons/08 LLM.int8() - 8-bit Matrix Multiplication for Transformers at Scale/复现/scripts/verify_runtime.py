# -*- coding: utf-8 -*-
"""真实执行原/注释数值格；整数补充与未量化小OPT生成单独报告。"""
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import OPTConfig, AutoModelForCausalLM, PreTrainedTokenizerFast, BitsAndBytesConfig

def value(item):
    if isinstance(item, torch.Tensor):
        return {'dtype': str(item.dtype), 'shape': list(item.shape), 'values': item.detach().cpu().tolist()}
    if isinstance(item, np.ndarray):
        return {'dtype': str(item.dtype), 'shape': list(item.shape), 'values': item.tolist()}
    if isinstance(item, tuple):
        return list(item)
    return item

def execute(source, context):
    # Notebook最后的表达式会显示，普通exec不会；单独eval最后Expr捕获真实返回值。
    tree = ast.parse(source)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop().value
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(tree, '<original-cell>', 'exec'), context)
        result = eval(compile(ast.Expression(last), '<original-cell-result>', 'eval'), context) if last is not None else None
    return result, output.getvalue()

def quantize(x, w):
    # 教学补充；不调用bitsandbytes。dtype区分存储int8与累加int32。
    sx = np.abs(x).max(axis=1, keepdims=True)
    sw = np.abs(w).max(axis=0, keepdims=True)
    safe_x = np.where(sx == 0, 1, sx)
    safe_w = np.where(sw == 0, 1, sw)
    qx = np.clip(np.rint(x * 127.0 / safe_x), -127, 127).astype(np.int8)
    qw = np.clip(np.rint(w * 127.0 / safe_w), -127, 127).astype(np.int8)
    acc = qx.astype(np.int32) @ qw.astype(np.int32)
    restored = acc.astype(np.float64) * (sx * sw) / 127**2
    return restored, qx, qw, acc

def run_variant(path, original, directory):
    nb = json.loads(path.read_text(encoding='utf-8'))
    cells = nb['cells'] if original else nb['cells'][2::2]
    torch.manual_seed(20261013)
    np.random.seed(20261013)
    torch.set_num_threads(1)
    context = {'__name__': '__lesson08_validation__'}
    snapshots = []
    proxy_before = {k: os.environ.get(k) for k in ('HTTP_PROXY', 'HTTPS_PROXY')}
    previous = Path.cwd()
    try:
        os.chdir(directory)
        for i, c in enumerate(cells):
            if c['cell_type'] != 'code' or i in (50, 51, 52, 53, 54):
                continue
            result, output = execute(''.join(c['source']), context)
            entry = {'original_cell': i, 'executed_entire_cell': True, 'stdout': output}
            if i in (2, 5, 24):
                assert result.format == 'png' and isinstance(result.data, bytes)
                image = PILImage.open(io.BytesIO(result.data))
                image.verify()
                entry['Image'] = {'format': result.format, 'width_argument': result.width,
                                  'bytes': len(result.data), 'sha256': hashlib.sha256(result.data).hexdigest()}
            else:
                entry['result'] = value(result)
            if i in (7, 8, 9, 10, 11, 12, 13, 16, 17):
                names = [k for k in ('a', 'a_max', 'scaled', 'int8val', 'int7val', 'fp32val') if k in context]
                entry['variables'] = {k: value(context[k]) for k in names}
            if i == 10:
                assert context['int8val'].dtype == torch.float32
                assert torch.equal(context['int8val'], context['int8val'].round())
            if i in (13, 17):
                assert torch.isfinite(context['fp32val']).all()
                step = context['a_max'] / 127
                assert (context['fp32val'] - context['a']).abs().max() <= step / 2 + 1e-6
            if i in (25, 27, 33, 36, 38, 43, 45, 46):
                names = [k for k in ('X', 'W', 'X_reg', 'W_reg', 'X_out', 'W_out', 'Cx', 'Cw', 'X_i8', 'W_i8', 'Out_i32', 'Out_f16_1', 'Out_f16_2') if k in context]
                entry['variables'] = {k: value(context[k]) for k in names}
            snapshots.append(entry)
    finally:
        os.chdir(previous)
    assert proxy_before == {k: os.environ.get(k) for k in proxy_before}
    x, w = context['X'], context['W']
    truth = x @ w
    assert truth.tolist() == [[38, 34], [-167, 124], [-174, 168]]
    assert context['X_i8'].dtype == context['W_i8'].dtype == context['Out_i32'].dtype == context['Out_f16_1'].dtype == np.float64
    original_result = context['Out_f16_1'] + context['Out_f16_2']
    np.testing.assert_allclose(original_result, truth, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.kron(context['Cx'], context['Cw']), context['Cx'] * context['Cw'])
    # 原数组复制后按阈值自动识别，不用手工索引作为检测结果。
    out_ids = np.flatnonzero(np.abs(x).max(axis=0) > 6)
    reg_ids = np.flatnonzero(np.abs(x).max(axis=0) <= 6)
    assert out_ids.tolist() == [1, 3] and reg_ids.tolist() == [0, 2, 4]
    regular, qx, qw, acc = quantize(x[:, reg_ids], w[reg_ids, :])
    outlier64 = x[:, out_ids].astype(np.float64) @ w[out_ids, :].astype(np.float64)
    mixed = regular + outlier64
    whole, _, _, _ = quantize(x, w)
    mixed_error = float(np.abs(mixed - truth).max())
    whole_error = float(np.abs(whole - truth).max())
    assert math.isclose(mixed_error, 0.01574803149605941, abs_tol=1e-12)
    assert math.isclose(whole_error, 1.0940541881083732, abs_tol=1e-12)
    assert qx.dtype == qw.dtype == np.int8 and acc.dtype == np.int32
    assert qx.tolist() == [[127, -64, -64], [0, 127, 85], [-127, -127, 0]]
    # 真实CPU FP16异常支路，浮点舍入与上述float64隔离；本例数值恰可表示。
    out16 = x[:, out_ids].astype(np.float16) @ w[out_ids, :].astype(np.float16)
    assert out16.dtype == np.float16
    np.testing.assert_array_equal(out16.astype(np.float64), outlier64)
    tx = torch.tensor(x[:, out_ids], dtype=torch.float16)
    tw = torch.tensor(w[out_ids, :], dtype=torch.float16)
    torch_out = tx @ tw
    assert torch_out.dtype == torch.float16
    np.testing.assert_array_equal(torch_out.float().numpy(), outlier64)
    # 全零行与全零权重列都应编码0；不是给原例偷偷加入保护。
    zero_x = np.array([[0., 0.], [1., -2.]])
    zero_w = np.array([[0., 3.], [0., 4.]])
    zero_result, zero_qx, zero_qw, _ = quantize(zero_x, zero_w)
    assert np.isfinite(zero_result).all() and (zero_result[0] == 0).all() and (zero_result[:, 0] == 0).all()
    assert (zero_qx[0] == 0).all() and (zero_qw[:, 0] == 0).all()
    # 不要用int8作为累加容器：2*127²=32258，int8 matmul回绕为2。
    huge_x = np.array([[127, 127]], dtype=np.int8)
    huge_w = np.array([[127], [127]], dtype=np.int8)
    bad = huge_x @ huge_w
    good = huge_x.astype(np.int32) @ huge_w.astype(np.int32)
    assert int(bad[0, 0]) == 2 and int(good[0, 0]) == 32258
    # 固定普通元素，只改末项，单独验证共同尺度造成信息损失。
    before = torch.tensor([.1, .2, -.1, -.2, 1.])
    after = before.clone(); after[-1] = 100.
    def recover(a):
        c = a.abs().max()
        return (a / c * 127).round() * c / 127
    small_error = (recover(before)[:4] - before[:4]).abs().mean().item()
    big_error = (recover(after)[:4] - after[:4]).abs().mean().item()
    assert big_error > small_error and torch.equal(recover(after)[:4], torch.zeros(4))
    assert torch.round(torch.tensor([.5, 1.5, 2.5, -.5, -1.5])).tolist() == [0., 2., 2., -0., -2.]
    vector_zero = torch.zeros(5)
    assert torch.isnan(vector_zero / vector_zero.abs().max()).all()
    # 独立本地教学上下文：不执行原51/52、不模拟from_pretrained、不把小模型说成8bit。
    vocabulary = {'[UNK]': 0, '[PAD]': 1, '[BOS]': 2, '[EOS]': 3,
                  'hello': 4, ',': 5, 'who': 6, 'are': 7, 'you': 8, '?': 9}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]')
    config = OPTConfig(vocab_size=len(tokenizer), hidden_size=16, word_embed_proj_dim=16,
             num_hidden_layers=1, num_attention_heads=2, ffn_dim=32,
             max_position_embeddings=32, dropout=0., attention_dropout=0.,
             pad_token_id=1, bos_token_id=2, eos_token_id=None)
    torch.manual_seed(808)
    model = AutoModelForCausalLM.from_config(config).eval()
    model.generation_config.eos_token_id = None
    input_ids = tokenizer('hello who are you', return_tensors='pt').input_ids
    assert input_ids.shape == (1, 4) and input_ids.dtype == torch.int64
    before_digest = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    ctx = {'model': model, 'tokenizer': tokenizer, 'input_ids': input_ids, 'MAX_NEW_TOKENS': 8}
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        decoded, output53 = execute(''.join(cells[53]['source']), ctx)
        _, output54 = execute(''.join(cells[54]['source']), ctx)
        new_ids = model.generate(input_ids, max_new_tokens=8, do_sample=False)
    total_ids = ctx['generated_ids']
    assert total_ids.shape == (1, 8) and new_ids.shape == (1, 12)
    assert torch.equal(total_ids[:, :4], input_ids) and torch.equal(new_ids[:, :4], input_ids)
    names = [name for name, _ in model.named_parameters()]
    assert len(output54.splitlines()) == len(names)
    assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in model.parameters())
    # 参数不变：generate不优化权重；保存两个生成调用后的参数摘要用于原/注释对照。
    digest = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    assert digest == before_digest
    return {'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
       'executed_numerical_and_import_cells': snapshots,
       'excluded_original_cells': {'50': '进程代理未改', '51': '未下载OPT tokenizer', '52': '未下载/量化OPT6.7B'},
       'original_arithmetic': {'actual_dtype_X': str(x.dtype), 'scaled_not_quantized_dtype': str(context['X_i8'].dtype),
             'outlier_dtype': str(context['Out_f16_2'].dtype), 'result': value(original_result),
             'max_error_without_rounding': float(np.abs(original_result - truth).max())},
       'integer_supplement': {'automatic_outlier_features': out_ids.tolist(), 'regular_features': reg_ids.tolist(),
             'qx': value(qx), 'qw': value(qw), 'accumulator': value(acc), 'mixed_result': value(mixed),
             'mixed_max_error': mixed_error, 'whole_max_error': whole_error,
             'int8_storage_bytes': qx.nbytes + qw.nbytes,
             'float64_same_values_storage_bytes': qx.size * 8 + qw.size * 8,
             'FP16_outlier_NumPy': value(out16), 'FP16_outlier_torch': value(torch_out),
             'zero_scale_test': value(zero_result), 'int8_accumulator_overflow': value(bad),
             'int32_accumulator_correct': value(good), 'raw_zero_vector_nan_confirmed': True,
             'fixed_ordinary_values_MAE_before': small_error, 'after_outlier100': big_error,
             'ties_to_even_confirmed': True, 'not_bitsandbytes_kernel': True},
       'small_OPT_supplement': {'executed_original_function_call_cells': [53, 54],
             'context': '随机CPU FP32 OPT；1层/H16/2heads/FFN32/vocab10，独立本地tokenizer；原51/52未执行。',
             'config': config.to_dict(), 'parameter_count': sum(p.numel() for p in model.parameters()),
             'input_ids': value(input_ids), 'max_length8_ids': value(total_ids),
             'max_new_tokens8_ids': value(new_ids), 'decode_full_sequence': decoded,
             'parameter_names': names, 'parameter_report_stdout': output54,
             'parameter_digest': digest, 'parameters_unchanged_by_generation': True, 'warnings': [str(w.message) for w in caught],
             'all_parameters_CPU_FP32': True, 'quantized': False,
             'semantic_quality_evaluated': False}}

def verify_bnb(directory):
    records = []
    for name, original in [('bitsandbytes.ipynb', True), ('bitsandbytes_代码注释.ipynb', False)]:
        nb = json.loads((directory / name).read_text(encoding='utf-8'))
        cells = nb['cells'] if original else nb['cells'][2::2]
        ctx = {}; execute(''.join(cells[1]['source']), ctx)
        cls = ctx['BitsAndBytesConfig']
        default = cls()
        assert not default.load_in_8bit and not default.load_in_4bit
        configured = cls(load_in_8bit=True)
        assert configured.load_in_8bit and not configured.load_in_4bit and configured.llm_int8_threshold == 6.
        try:
            cls(load_in_8bit=True, load_in_4bit=True)
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError('Expected mutually exclusive ValueError')
        records.append({'file': name, 'entire_cell1_executed': True,
                        'default_both_false': True, '8bit_config_threshold': 6., 'mutual_exclusion_error': message,
                        'only_config_not_model_load': True})
    assert records[0]['mutual_exclusion_error'] == records[1]['mutual_exclusion_error']
    return records

def main():
    lesson = Path(__file__).resolve().parents[2]
    directory = lesson / '代码/tutorials'
    a = run_variant(directory / 'LLM.int8.ipynb', True, directory)
    b = run_variant(directory / '代码注释.ipynb', False, directory)
    assert {k: v for k, v in a.items() if k != 'notebook_sha256'} == {k: v for k, v in b.items() if k != 'notebook_sha256'}
    record = {'environment': {'OS': platform.platform(), 'Python': platform.python_version(), 'torch': torch.__version__,
               'CUDA_available': torch.cuda.is_available(), 'packages': {p: version(p) for p in ('numpy', 'transformers', 'tokenizers', 'IPython', 'nbformat', 'Pillow')}},
              'original': a, 'annotated': b, 'runtime_results_equal': True,
              'bitsandbytes_config': verify_bnb(directory),
              'OPT6_7B_weights_downloaded': False, 'bitsandbytes_imported_or_installed': False,
              'CUDA_int8_kernel_executed': False, 'C4_or_zero_shot_metrics_reproduced': False}
    (lesson / '复现/outputs/runtime_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Original/commented numerical cells + 3 actual images match; true int8 storage/int32 arithmetic/FP16 branch/zero scales pass.')
    print('Real BitsAndBytesConfig mutual exclusion and original53/54 on separate unquantized tiny OPT pass; not OPT6.7B/CUDA reproduction.')

if __name__ == '__main__':
    main()
