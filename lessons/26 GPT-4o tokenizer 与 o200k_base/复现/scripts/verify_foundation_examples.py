"""Run all published examples against complete pinned tokenizers and corpus."""
from pathlib import Path
import ast
import base64
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re as std_re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get('LESSON26_LOCAL', 'G:/zzz/Personal ChatGPT 系列/26 GPT-4o tokenizer 与 o200k_base/资料'))
LIB = Path(os.environ.get('LESSON26_TIKTOKEN_LIBS', 'G:/zzz/Personal ChatGPT 系列/25 Llama 3 introduction/资料/runtime-libs'))
EXTRA = Path(os.environ.get('LESSON26_EXTRA_LIBS', str(LOCAL / 'runtime-libs')))
sys.path[:0] = [str(EXTRA), str(LIB)]
import regex
import tiktoken
import tiktoken._tiktoken
from tiktoken._educational import bpe_encode


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_sources():
    assert importlib.metadata.version('tiktoken') == '0.7.0'
    assert Path(tiktoken.__file__).resolve().parent == (LIB / 'tiktoken').resolve()
    manifest = json.loads((ROOT / '来源清单.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        assert sha(ROOT / item['published_path']) == item['sha256']
    for item in manifest['vocabularies'].values():
        assert sha(ROOT / item['published_path']) == item['sha256']
    rows = csv.reader(io.StringIO((LIB / 'tiktoken-0.7.0.dist-info/RECORD').read_text(encoding='utf-8')))
    record = {r[0]: r[1] for r in rows}
    for rel, expected in manifest['installed_sources_sha256'].items():
        p = LIB / rel
        assert sha(p) == expected, rel
        encoded = base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')
        assert record[rel] == 'sha256=' + encoded, rel
        assert p.read_bytes() == (LOCAL / 'tiktoken-source' / rel).read_bytes()
    binary = Path(tiktoken._tiktoken.__file__)
    rel = binary.relative_to(LIB).as_posix()
    encoded = base64.urlsafe_b64encode(hashlib.sha256(binary.read_bytes()).digest()).decode().rstrip('=')
    assert record[rel] == 'sha256=' + encoded
    manifest['verified_rust_binary'] = {'file': rel, 'sha256': sha(binary), 'matches_wheel_record': True}
    return manifest


def original_ast_checks(blocks):
    nb = json.loads((ROOT / '代码/tutorials/tokenizer/tokenizer_gpt4o.ipynb').read_text(encoding='utf-8'))
    actual = ast.parse(blocks[3])
    actual = ast.Module(body=[n for n in actual.body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id in ('token_ids_len1', 'token_ids_len2') for t in n.targets)]
                         + [next(n for n in actual.body if isinstance(n, ast.With))], type_ignores=[])
    original = ast.parse(''.join(nb['cells'][11]['source']))
    # The single path argument changes because the note runs from the lesson directory.
    actual.body[-1].items[0].context_expr.args[0] = original.body[-1].items[0].context_expr.args[0]
    assert ast.dump(actual) == ast.dump(original)
    original_regex = next(n for n in ast.parse(''.join(nb['cells'][19]['source'])).body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pattern' for t in n.targets))
    actual_regex = next(n for n in ast.parse(blocks[5]).body
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pattern' for t in n.targets))
    assert ast.dump(actual_regex) == ast.dump(original_regex)
    return nb


def corpus_checks(scope):
    assert [scope['token_ids_len1'], scope['token_ids_len2']] == [990508, 703303]
    assert scope['whole_counts'] == [989904, 702799]
    # Re-read by the iterator API, then independently sum in batches.
    with open(ROOT / '代码/tutorials/tokenizer/xyj.txt', 'r', encoding='utf-8') as f:
        lines = list(f)
    assert ''.join(lines) == scope['whole_text']
    batch_counts = []
    for enc in (scope['T1'], scope['T2']):
        batches = [lines[i:i+256] for i in range(0, len(lines), 256)]
        count = sum(len(ids) for batch in batches for ids in enc.encode_batch(batch, num_threads=2))
        batch_counts.append(count)
    assert batch_counts == [990508, 703303]
    return {'file_sha256': sha(ROOT / '代码/tutorials/tokenizer/xyj.txt'),
            'text_mode_characters': len(scope['whole_text']), 'lines': len(lines),
            'line_counts': batch_counts, 'whole_counts': scope['whole_counts'],
            'line_minus_whole': [a-b for a, b in zip(batch_counts, scope['whole_counts'])],
            'old_over_new': batch_counts[0]/batch_counts[1],
            'reduction': 1-batch_counts[1]/batch_counts[0], 'batch_iterator_recount_equal': True}


def vocabulary_checks(scope):
    result = []
    for enc, ordinary, valid, hole_count, broken in [
            (scope['T1'], 100256, 100261, 16, 773),
            (scope['T2'], 199998, 200000, 19, 1562)]:
        # Use actual dictionary membership as an independent enumeration of valid IDs.
        ids = set(enc._mergeable_ranks.values()) | set(enc._special_tokens.values())
        holes = sorted(set(range(enc.n_vocab)) - ids)
        assert len(enc._mergeable_ranks) == ordinary and len(ids) == valid
        assert len(holes) == hole_count
        byte_rows = [(i, enc.decode_single_token_bytes(i)) for i in sorted(ids)]
        invalid = 0
        han = []
        for i, raw in byte_rows:
            try:
                text = raw.decode('utf-8', errors='strict')
            except UnicodeDecodeError:
                invalid += 1
                continue
            if i not in enc._special_tokens.values() and any(regex.fullmatch(r'\p{Script=Han}', c) for c in text):
                han.append((len(text), i, text))
        assert invalid == broken
        han.sort(key=lambda row: (-row[0], row[1]))
        if enc is scope['T2']:
            assert scope['holes'] == holes
            assert scope['rows'] == han
            assert scope['incomplete_utf8'] == invalid
        result.append({'encoding': enc.name, 'ordinary': ordinary, 'special': len(enc._special_tokens),
                       'valid_ids': valid, 'n_vocab': enc.n_vocab, 'holes': holes,
                       'incomplete_single_token_utf8': invalid, 'han_candidates': len(han),
                       'longest_five_id_and_length': [[i, length] for length, i, text in han[:5]]})
    return result


def encoding_checks(scope, notebook):
    t1, t2 = scope['T1'], scope['T2']
    assert scope['ab_ids'] == [256, 99] and scope['bc_ids'] == [97, 256]
    assert list(scope['raw']) == [228, 189, 160, 229, 165, 189]
    assert len(scope['comparisons']['cl100k']['english']) == len(scope['comparisons']['o200k']['english']) == 5
    assert [len(scope['comparisons'][k]['chinese']) for k in ['cl100k', 'o200k']] == [34, 24]
    assert t1.encode('world') == [14957] and t1.encode(' world') == [1917]
    assert t1.decode([1917, 14957]) == ' worldworld'
    assert t1.encode('AIM') == [14660] and t2.encode('AIM') == [32, 2833]
    assert tiktoken.get_encoding('cl100k_base') is t1
    samples = ['i love chatgpt', scope['chinese'], '你好，世界！', 'hello world', ' world',
               '1234+4567', 'café e\u0301', '🧑🏽‍💻', '\t\n  ', 'HTTPServer_v2']
    comparisons = []
    for enc in (t1, t2):
        for text in samples:
            ids = enc.encode(text)
            pieces = regex.findall(enc._pat_str, text)
            assert ''.join(pieces) == text
            readable = [i for part in pieces for i in bpe_encode(enc._mergeable_ranks, part.encode('utf-8'), visualise=None)]
            assert readable == ids
            raw = b''.join(enc.decode_single_token_bytes(i) for i in ids)
            assert raw == text.encode('utf-8') and enc.decode(ids) == text
            comparisons.append({'encoding': enc.name, 'text': text, 'ids': ids, 'pretokenized': pieces})
        marker = '<|endoftext|>'
        rejected = False
        try:
            enc.encode(marker)
        except ValueError:
            rejected = True
        assert rejected
        assert enc.encode(marker, allowed_special='all') == [enc._special_tokens[marker]]
        assert enc.encode(marker, disallowed_special=()) != [enc._special_tokens[marker]]
        assert scope['emoji_checks'][enc.name]['per_token'] != scope['emoji_checks'][enc.name]['whole']
    # The observed long Chinese string is taken verbatim from the original cell.
    long_string = next(n.value for n in ast.walk(ast.parse(''.join(notebook['cells'][15]['source'])))
                       if isinstance(n, ast.Constant) and isinstance(n.value, str))
    assert t2.encode(long_string) == [177431] and len(t1.encode(long_string)) == 12
    assert 177431 not in t2._special_tokens.values()
    regex_results = {s: bool(regex.match(scope['pattern'], s)) for s in ['abc', '123', '中文', '___', '🙂']}
    assert regex_results == {'abc': False, '123': False, '中文': False, '___': True, '🙂': True}
    return {'short_samples_rust_and_full_official_python_equal': comparisons,
            'original_ordinary_long_token_id': 177431, 'original_ordinary_long_token_old_count': 12,
            'special_policy_default_rejected_allowed_and_ordinary_checked': True,
            'utf8_emoji_checks': scope['emoji_checks'], 'original_regex_results': regex_results,
            'encoding_registry_reuses_object': True}


def main():
    manifest = check_sources()
    text = (ROOT / '解释.md').read_text(encoding='utf-8').split('<details>', 1)[0]
    blocks = std_re.findall(r'```python\n(.*?)```', text, std_re.S)
    assert len(blocks) == 6
    scope = {'__name__': '_lesson26_examples_'}
    outputs = []
    cwd = os.getcwd()
    previous_cache = os.environ.get('TIKTOKEN_CACHE_DIR')
    with tempfile.TemporaryDirectory(prefix='lesson26-foundation-vocabs-') as td:
        cache = Path(td)
        for name, item in manifest['vocabularies'].items():
            key = hashlib.sha1(item['official_url'].encode()).hexdigest()
            shutil.copyfile(ROOT / item['published_path'], cache / key)
        os.environ['TIKTOKEN_CACHE_DIR'] = str(cache)
        try:
            os.chdir(ROOT)
            for i, block in enumerate(blocks):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    exec(compile(block, f'<lesson26-main-block-{i+1}>', 'exec'), scope)
                outputs.append(out.getvalue())
            notebook = original_ast_checks(blocks)
            corpus = corpus_checks(scope)
            vocabularies = vocabulary_checks(scope)
            encodings = encoding_checks(scope, notebook)
        finally:
            os.chdir(cwd)
            if previous_cache is None:
                os.environ.pop('TIKTOKEN_CACHE_DIR', None)
            else:
                os.environ['TIKTOKEN_CACHE_DIR'] = previous_cache
    figures = json.loads((ROOT / '复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
    for item in figures['crops']:
        assert sha(ROOT / item['file']) == item['sha256']
    report = {'date': '2026-10-18', 'note_sha256': sha(ROOT / '解释.md'), 'script_sha256': sha(__file__),
              'python': sys.version, 'packages': {n: importlib.metadata.version(n) for n in ['tiktoken', 'regex']},
              'official_source_sha256': manifest['installed_sources_sha256'],
              'loaded_rust_binary': manifest['verified_rust_binary'],
              'vocab_sha256': {k: v['sha256'] for k, v in manifest['vocabularies'].items()},
              'main_blocks': 6, 'stdout': outputs,
              'original_corpus_loop_ast_equal_except_path': True, 'original_regex_ast_equal': True,
              'corpus': corpus, 'vocabularies': vocabularies, 'encoding_checks': encodings,
              'scope': 'Complete fixed tiktoken 0.7.0, both complete public vocabularies and original novel; six CPU examples, all o200k IDs and independent both-vocabulary valid-ID scans. No language-model inference, model training, paid API or language-detector full rerun.'}
    path = ROOT / '复现/outputs/基础讲解验证.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'blocks': 6, 'novel_line_counts': corpus['line_counts'],
                      'vocabulary_valid_ids': [v['valid_ids'] for v in vocabularies],
                      'single_token_incomplete_utf8': [v['incomplete_single_token_utf8'] for v in vocabularies],
                      'result': str(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
