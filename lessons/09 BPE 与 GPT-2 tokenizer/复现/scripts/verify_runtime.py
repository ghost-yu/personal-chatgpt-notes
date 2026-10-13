# -*- coding: utf-8 -*-
"""原单元实际执行与逐项审计补充；固定tokenizer，小JSONL，无权重/远程数据/上传。"""
import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import io
import itertools
import json
import os
import platform
import re
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import datasets
from datasets import Dataset, load_dataset, load_dataset_builder
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

def dump(node):
    return ast.dump(node, include_attributes=False)

def execute(source, ctx):
    tree = ast.parse(source)
    last = tree.body.pop().value if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(tree, '<lesson09-cell>', 'exec'), ctx)
        result = eval(compile(ast.Expression(last), '<lesson09-result>', 'eval'), ctx) if last is not None else None
    return result, output.getvalue()

def describe(result):
    if result is None or isinstance(result, (str, int, float, bool, list, tuple)):
        return result
    if isinstance(result, dict):
        return {'dict_size': len(result)}
    return {'class': type(result).__name__, 'repr': repr(result)}

def tokenizer_vocab(tokenizer):
    return tokenizer.get_vocab().keys()

def cells(path, original):
    nb = json.loads(path.read_text(encoding='utf-8'))
    return nb['cells'] if original else nb['cells'][2::2]

def loader_local(source, path):
    # 只改from_pretrained首参数，并加local_files_only，逆变换AST必须等于原格。
    tree = ast.parse(source); saved = copy.deepcopy(tree); changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'from_pretrained':
            old = node.args[0].value
            assert old in ('gpt2', 't5-base')
            node.args[0] = ast.Constant(str(path))
            node.keywords.append(ast.keyword(arg='local_files_only', value=ast.Constant(True)))
            changes.append((node, old))
    assert len(changes) == 1
    modified = ast.unparse(ast.fix_missing_locations(tree))
    for node, old in changes:
        node.args[0] = ast.Constant(old); node.keywords.pop()
    assert dump(tree) == dump(saved)
    return modified

def change_constants(source, replacements):
    tree = ast.parse(source); old = copy.deepcopy(tree); edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) == int and node.value in replacements:
            prior = node.value; node.value = replacements[prior]; edits.append((node, prior))
    assert sorted(v for _, v in edits) == sorted(replacements)
    result = ast.unparse(ast.fix_missing_locations(tree))
    for node, prior in edits:
        node.value = prior
    assert dump(tree) == dump(old)
    return result

class TrackingIterator:
    """小语料迭代器输入适配器；不改load_dataset/训练API，不伪造消费。"""
    def __init__(self, base):
        self.base = base; self.row_ids = []
    def __iter__(self):
        return self
    def __next__(self):
        row = next(self.base); self.row_ids.append(row['row_id']); return row

def source(cell_list, i):
    return ''.join(cell_list[i]['source'])

def functions_only(text):
    tree = ast.parse(text)
    return ast.unparse(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]))

def run_variant(folder, cache, original, jsonl, temp, asset_records):
    basics_path = folder / ('tokenizer_basics.ipynb' if original else '代码注释.ipynb')
    train_path = folder / ('gpt2_train_tokenizer.ipynb' if original else 'gpt2_train_tokenizer_代码注释.ipynb')
    aux_path = folder / ('gpt2.ipynb' if original else 'gpt2_代码注释.ipynb')
    basics, train, aux = cells(basics_path, original), cells(train_path, original), cells(aux_path, original)
    ctx = {'__name__': '__lesson09__'}
    records = []; oldcwd = Path.cwd()
    before_env = {key: os.environ.get(key) for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'CURL_CA_BUNDLE')}
    try:
        os.chdir(folder)
        for i, cell in enumerate(basics):
            if cell['cell_type'] != 'code' or i in (21, 22):
                continue
            text = source(basics, i); supplement = None
            if i in (5, 10, 31):
                text = loader_local(text, cache / ('t5' if i == 5 else 'gpt2'))
                supplement = '只改from_pretrained为固定本地资产+local_files_only=True；逆变换AST相等。'
            result, stdout = execute(text, ctx)
            item = {'cell': i, 'scope': supplement or '整个原格执行', 'stdout': stdout}
            if i == 17:
                assert result.format == 'png' and result.width == 600
                item['image'] = {'sha256': hashlib.sha256(result.data).hexdigest(), 'bytes': len(result.data)}
            else:
                item['result'] = describe(result)
            if i == 36:
                # 原sort只按长度；词表遍历来自后端字典，同长度项未定义二级顺序。
                lengths = [len(t) for t, _ in ctx['tokens'][:10]]
                expected = sorted((len(t) for t in tokenizer_vocab(ctx['tokenizer'])), reverse=True)[:10]
                assert lengths == expected
                assert result == [ctx['tokenizer'].convert_tokens_to_string([t]) for t, _ in ctx['tokens'][:10]]
                item['tie_aware_comparison'] = {'sort_key': 'len(token representation)', 'top10_key_lengths': lengths,
                                               'tie_order_not_defined': True, 'all_returned_items_match_original_sort': True}
            records.append(item)
        assert ctx['byte'] == [226, 130, 172]
        mapping = ctx['bytes_to_unicode_map']; inverse = ctx['unicode_to_bytes_map']
        assert len(mapping) == len(inverse) == 256
        assert mapping[32] == 'Ġ' and mapping[10] == 'Ċ' and mapping[13] == 'č'
        tokenizer = ctx['tokenizer']; old_json = tokenizer.backend_tokenizer.to_str()
        assert len(tokenizer) == 50257 and tokenizer.is_fast and tokenizer.backend_tokenizer.normalizer is None
        sample = '中文 € 😀\n    print(1)\t\r\n\u00a0'
        roundtrip = tokenizer.decode(tokenizer.encode(sample), clean_up_tokenization_spaces=False)
        assert roundtrip == sample
        byte_roundtrip = bytes(inverse[mapping[b]] for b in sample.encode('utf-8')).decode('utf-8')
        assert byte_roundtrip == sample
        emoji_ids = tokenizer.encode('😀', add_special_tokens=False)
        partial = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in emoji_ids]
        assert partial != ['😀'] and '�' in ''.join(partial)
        block = re.search(r'```[^\n]*\n(.*?)```', source(basics, 2), re.S)[1]
        local = AutoTokenizer.from_pretrained(str(cache / 'gpt2'), local_files_only=True)
        execute(block, {'tokenizer': local})
        assert local.padding_side == 'right' and local.pad_token == local.eos_token
        # 未调用原21的远程load_dataset；原生成器定义在明确小Dataset上真实执行。
        dataset = Dataset.from_dict({'text': [str(i) for i in range(2051)]})
        fn = {}; execute(functions_only(source(basics, 21)), fn)
        batches = list(fn['get_training_corpus'](dataset))
        assert list(map(len, batches)) == [1000, 1000, 51]
        assert [s for batch in batches for s in batch] == dataset['text']
        errors = []
        try:
            execute(source(basics, 22), {})
        except SyntaxError as error:
            errors.append({'type': type(error).__name__, 'message': error.msg})
        else:
            raise AssertionError('原22必须失败')
        corrected = source(basics, 22).replace('from tokenizers import Tokenizer, models,', 'from tokenizers import Tokenizer, models', 1)
        fixed = {}; execute(corrected, fixed)
        assert type(fixed['tokenizer'].model).__name__ == 'WordPiece'
        # 两轮小型真实流式训练。源数据固定2000行，每批默认1000。
        tc = {}; tr = []
        execute(source(train, 1), tc)
        stream = load_dataset('json', data_files=str(jsonl), split='train', streaming=True)
        assert isinstance(stream, datasets.IterableDataset)
        tc['dataset'] = stream; execute(source(train, 3), tc)
        tracked = TrackingIterator(tc['iter_dataset']); tc['iter_dataset'] = tracked
        execute(loader_local(source(train, 4), cache / 'gpt2'), tc)
        execute(source(train, 6), tc)
        original_template = tc['tokenizer'].backend_tokenizer.to_str()
        execute(source(train, 7), tc); tc['length'] = 1000
        # 这里只改目标大小；原默认batch_size=1000和生成器体保持。
        execute(change_constants(source(train, 8), {12500: 320}), tc)
        first = tc['new_tokenizer']; first_json = first.backend_tokenizer.to_str()
        assert len(first) == 320 and tracked.row_ids == list(range(1000))
        assert tc['tokenizer'].backend_tokenizer.to_str() == original_template
        for i in (9, 11, 12, 13, 14, 16, 17, 18):
            result, stdout = execute(source(train, i), tc)
            tr.append({'cell': i, 'scope': '整个原格在第一轮小型新tokenizer上下文执行', 'result': describe(result), 'stdout': stdout})
        # length原式100000*2；仅将100000改为500，目标32768改336，故本轮1000行。
        execute(change_constants(source(train, 20), {100000: 500, 32768: 336}), tc)
        second = tc['new_tokenizer']; second_json = second.backend_tokenizer.to_str()
        assert len(second) == 336 and tracked.row_ids == list(range(2000))
        # 原生成器刚好读完上限时未再次next；Windows上须显式关闭底层流句柄。
        if hasattr(tracked.base, 'close'):
            tracked.base.close()
        assert first.backend_tokenizer.to_str() == first_json and tc['tokenizer'].backend_tokenizer.to_str() == original_template
        _, stdout21 = execute(source(train, 21), tc)
        execute(source(train, 23), tc); execute(source(train, 26), tc)
        saved = []
        for name, tok in [('first', first), ('second', second)]:
            ids = tok(sample, add_special_tokens=False)['input_ids']
            assert tok.decode(ids, clean_up_tokenization_spaces=False) == sample
            target = temp / name; tok.save_pretrained(target)
            loaded = AutoTokenizer.from_pretrained(target, local_files_only=True)
            assert loaded(sample, add_special_tokens=False)['input_ids'] == ids
            assert loaded.decode(ids, clean_up_tokenization_spaces=False) == sample
            info = json.loads(tok.backend_tokenizer.to_str())
            saved.append({'name': name, 'actual_vocab': len(tok), 'merges': len(info['model']['merges']),
                          'special_ids': tok.all_special_ids, 'sample_ids': ids, 'sample_tokens': tok.convert_ids_to_tokens(ids),
                          'full_decode': sample, 'save_reload_equal': True})
        # 真正原batch_iterator体：超读、不满批丢失、第二轮继续。
        gc = {'tqdm': lambda x: x}; execute(functions_only(source(train, 7)), gc)
        rows = [{'content': str(i)} for i in range(7)]
        gc.update(length=5, iter_dataset=iter(rows))
        overshoot = list(gc['batch_iterator'](2)); assert overshoot == [['0', '1'], ['2', '3'], ['4', '5']]
        gc.update(length=8, iter_dataset=iter(rows))
        seen = []; exhaustion = None
        try:
            for batch in gc['batch_iterator'](3):
                seen.append(batch)
        except RuntimeError as error:
            exhaustion = str(error)
        assert seen == [['0', '1', '2'], ['3', '4', '5']] and exhaustion == 'generator raised StopIteration'
        def safe(data, limit, size):
            it = iter(data)
            for start in range(0, limit, size):
                chunk = list(itertools.islice(it, min(size, limit - start)))
                if not chunk: return
                yield [row['content'] for row in chunk]
        assert list(safe(rows, 5, 2)) == [['0', '1'], ['2', '3'], ['4']]
        assert list(safe(rows, 12, 3)) == [['0', '1', '2'], ['3', '4', '5'], ['6']]
        # 只把进度显示函数换成identity以静默边界小例子；主训练用真实tqdm。
        ac = {}; execute(source(aux, 2), ac); execute(source(aux, 3), ac)
        builder = load_dataset_builder('json', data_files=str(jsonl), cache_dir=str(temp / 'builder-cache'))
        assert isinstance(builder.cache_dir, str)
        weighted = Counter()
        words = {'aa': 4, 'ab': 3, 'aba': 2}
        for word, frequency in words.items():
            for pair in zip(word, word[1:]): weighted[pair] += frequency
        assert weighted[('a', 'b')] == 5 and weighted[('a', 'a')] == 4
        # 三种表示的计数分别计算，BPE初始byte映射不是Unicode码点0–255。
        assert tokenizer.backend_tokenizer.to_str() == old_json
    finally:
        os.chdir(oldcwd)
    assert before_env == {key: os.environ.get(key) for key in before_env}
    return {'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (basics_path, train_path, aux_path)},
      'basics_execution': records, 'basics21_original_function_on_small_Dataset': {'rows': 2051, 'batch_sizes': [1000, 1000, 51]},
      'basics22_expected_error': errors, 'basics22_independent_one_comma_fix': 'WordPiece对象成功，仅构造，不训练',
      'Markdown2_executed_on_separate_tokenizer': {'padding_side': 'right', 'pad_equals_eos': True},
      'byte_checks': {'mapping_count': 256, 'space': mapping[32], 'newline': mapping[10], 'carriage_return': mapping[13],
           'euro_bytes': [226, 130, 172], 'sample': sample, 'full_roundtrip': roundtrip,
           'emoji_ids': emoji_ids, 'emoji_individual_decode': partial, 'base_vocab10_is_not_byte10': ctx['base_vocab'][10] != mapping[10]},
      'training_execution': tr,
      'training_execution_scope': {'entire_unmodified_cells_with_small_global_context': [1, 3, 6, 7, 9, 11, 12, 13, 14, 16, 17, 18, 21, 23, 26],
          'audited_modified_cells': {'4': '本地tokenizer路径/local_files_only', '8': 'vocab12500→320', '20': '100000→500，32768→336'},
          'skipped_original_cells': {'2': '课程本地语料缺失，另用真实json流数据', '24': '代理环境未改', '25': '讲师Hub上传未执行'}},
      'small_training': {'real_dataset_type': type(stream).__name__, 'rows': 2000,
           'batch_default': 1000, 'original7_body_unchanged': True, 'supplement8_vocab12500_to320': True,
           'supplement20_constants_only': {'100000': 500, '32768': 336},
           'first_rows': [0, 999], 'second_rows': [1000, 1999], 'read_rows_total': len(tracked.row_ids),
           'old_template_unchanged': True, 'first_object_unchanged_after_second': True,
           'trained_tokenizers': saved, 'second_keyword_stdout': stdout21},
      'iterator_boundaries': {'original_length5_batch2_reads6': overshoot, 'original_length8_batch3_rows7_partial_lost': seen,
           'original_exhaustion': exhaustion, 'safe_partial': list(safe(rows, 12, 3)),
           'boundary_tqdm_display_only_silenced': True},
      'auxiliary': {'entire_original_cells_executed': [2, 3], 'independent_json_builder_cache_is_string': True,
           'shell1_4_not_executed': True, 'original5_missing_path_not_executed': True},
      'weighted_pair_counts': {' '.join(pair): n for pair, n in weighted.items()},
      'excluded_side_effects': {'course_datasets_downloaded': False, 'original_proxy24_executed': False,
           'original_Hub25_executed': False, 'model_weights_downloaded': False, 'LM_training': False}}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tokenizer-cache', type=Path, required=True)
    args = parser.parse_args(); cache = args.tokenizer_cache.resolve()
    lesson = Path(__file__).resolve().parents[2]; folder = lesson / '代码/tutorials'
    download = json.loads((lesson / '复现/outputs/source_download.json').read_text(encoding='utf-8'))
    asset_records = download['tokenizer_assets_only']
    for item in asset_records:
        assert hashlib.sha256((cache / item['tokenizer'] / item['file']).read_bytes()).hexdigest() == item['sha256']
    with tempfile.TemporaryDirectory(prefix='lesson09-') as tmp:
        temp = Path(tmp); jsonl = temp / 'code.jsonl'
        first = ["def say_hello():\n    print('Hello, World!')\n", "if value is None:\n    return False\n", "# 中文 € 😀\nx = 1\n", "for item in items:\n    print(item)\n"]
        second = ["async def task():\n    await task()\n", "def outer():\n    nonlocal_value = 1\n", "try:\n    raise ValueError()\nexcept Exception:\n    pass\n", "class Example:\n    def method(self):\n        return self\n"]
        rows = [{'row_id': i, 'content': (first if i < 1000 else second)[i % 4] + '# row ' + str(i) + '\n'} for i in range(2000)]
        jsonl.write_bytes(('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n').encode('utf-8'))
        a = run_variant(folder, cache, True, jsonl, temp / 'original', asset_records)
        b = run_variant(folder, cache, False, jsonl, temp / 'annotated', asset_records)
        # exec函数的globals会形成引用环，及时回收，释放临时JSONL/Arrow句柄。
        gc.collect()
    def comparable(r):
        c = copy.deepcopy({k: v for k, v in r.items() if k != 'source_hashes'})
        for entry in c['basics_execution']:
            if entry['cell'] == 36:
                entry['result'] = entry['tie_aware_comparison']
        return c
    def differences(x, y, path=''):
        if type(x) != type(y): return [path + ':type']
        if isinstance(x, dict):
            return [p for key in x for p in differences(x[key], y[key], path + '/' + str(key))]
        if isinstance(x, list):
            if len(x) != len(y): return [path + ':length']
            return [p for i, (v, w) in enumerate(zip(x, y)) for p in differences(v, w, path + '/' + str(i))]
        return [] if x == y else [path]
    mismatch = differences(comparable(a), comparable(b))
    assert not mismatch, mismatch[:12]
    record = {'environment': {'OS': platform.platform(), 'Python': platform.python_version(),
        'packages': {p: version(p) for p in ('transformers', 'tokenizers', 'datasets', 'pyarrow', 'sentencepiece', 'tqdm', 'IPython', 'nbformat')}},
        'assets': asset_records, 'original': a, 'annotated': b, 'runtime_results_equal_except_length_sort_ties': True,
        'length_sort_ties': 'basics36原sort未定义同长度二级顺序，保存两次原始输出；前10项长度序列及转换契约核对通过，其余结果一致。',
        'training_corpus_sha256': hashlib.sha256(('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n').encode('utf-8')).hexdigest(),
        'scope': '真实固定tokenizer与小语料两轮训练；非课程10/20万语料或语言模型复现'}
    (lesson / '复现/outputs/runtime_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Original/commented tokenizer cells match; local GPT2/T5 assets, byte/Unicode/offset, expected SyntaxError and Markdown settings verified.')
    print('Real IterableDataset 2000 rows -> two BPE trainings 320/336; actual consumption, save/reload and boundary failures passed; no Hub write.')

if __name__ == '__main__':
    main()
