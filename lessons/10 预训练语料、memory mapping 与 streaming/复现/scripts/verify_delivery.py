# -*- coding: utf-8 -*-
"""原/注释6格真实图片执行、完整静态核对；另调用可阅读的独立数据教学补充。"""
import argparse
import ast
import copy
import difflib
import gc
import hashlib
import importlib
import io
import json
import os
import platform
import re
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
import nbformat
from PIL import Image as PILImage

FILENAME = 'pretrain_corpus_大模型预训练语料.ipynb'
TOKENIZER_FILES = {
    'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
}

def tree(source):
    return ast.dump(ast.parse(source), include_attributes=False)

def static(lesson):
    folder = lesson / '代码/tutorials'
    path = folder / FILENAME
    original = json.loads(path.read_text(encoding='utf-8'))
    manifest = json.loads((lesson / '来源清单.json').read_text(encoding='utf-8'))
    for item in manifest['published_originals']:
        assert hashlib.sha256((lesson / item['published_path']).read_bytes()).hexdigest() == item['sha256']
    nbformat.validate(nbformat.from_dict(original))
    reports = []
    code_indices = [i for i, c in enumerate(original['cells']) if c['cell_type'] == 'code']
    assert len(original['cells']) == 13 and code_indices == [0, 3, 5, 8, 11, 12]
    for name in ('原代码_带说明.ipynb', '代码注释.ipynb'):
        path = folder / name; nb = json.loads(path.read_text(encoding='utf-8'))
        nbformat.validate(nbformat.from_dict(nb))
        assert len(nb['cells']) == 27
        assert {k: v for k, v in nb.items() if k != 'cells'} == {k: v for k, v in original.items() if k != 'cells'}
        for i, c in enumerate(original['cells']):
            note, keep = nb['cells'][1+2*i:3+2*i]
            assert '## 单元 {}（'.format(i) in ''.join(note['source'])
            assert {k: v for k, v in c.items() if k != 'source'} == {k: v for k, v in keep.items() if k != 'source'}
            if name == '代码注释.ipynb' and c['cell_type'] == 'code':
                old, new = ''.join(c['source']), ''.join(keep['source'])
                a, b = old.splitlines(True), new.splitlines(True)
                for op, u, v, x, y in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                    if op != 'equal': assert op == 'insert' and all(s.lstrip().startswith('# ') for s in b[x:y])
                assert tree(old) == tree(new)
            else:
                assert c['source'] == keep['source']
        reports.append({'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'original13_cells_and_history_preserved': True, 'all6_AST_equal': True,
                        'original_Markdown_unchanged': True, 'schema_valid': True})
    py = folder / FILENAME.replace('.ipynb', '_代码注释.py')
    plain = '\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type'] == 'code')
    assert tree(py.read_text(encoding='utf-8')) == tree(plain)
    doc = (lesson / '代码单元解释.md').read_text(encoding='utf-8')
    assert [int(s) for s in re.findall(r'^## 单元 (\d+)（', doc, re.M)] == list(range(13))
    return {'all13_notes_present': True, 'all6_code_AST_equal': True, 'notebook_schemas_valid': 3,
            'original_and4_images_SHA256_match': True, 'reading_py_AST_equals_original6': True,
            'reading_py_sha256': hashlib.sha256(py.read_bytes()).hexdigest(), 'variants': reports}

def run_notebook(path, original, folder):
    nb = json.loads(path.read_text(encoding='utf-8'))
    cells = nb['cells'] if original else nb['cells'][2::2]
    context = {}; report = []; previous = Path.cwd()
    try:
        os.chdir(folder)
        for i, c in enumerate(cells):
            if c['cell_type'] != 'code': continue
            node = ast.parse(''.join(c['source']))
            last = node.body.pop().value if node.body and isinstance(node.body[-1], ast.Expr) else None
            exec(compile(node, '<original-cell>', 'exec'), context)
            result = eval(compile(ast.Expression(last), '<original-result>', 'eval'), context) if last is not None else None
            item = {'original_cell': i, 'entire_original_cell_executed': True}
            if i in (3, 5, 8, 11):
                assert result.format == 'png' and result.width == 500
                image = PILImage.open(io.BytesIO(result.data)); image.verify()
                item['Image'] = {'format': result.format, 'display_width': result.width,
                                 'bytes': len(result.data), 'sha256': hashlib.sha256(result.data).hexdigest()}
            else:
                assert result is None
            report.append(item)
    finally:
        os.chdir(previous)
    return {'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'all6_cells_executed': report}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tokenizer-dir', type=Path, required=True)
    args = parser.parse_args()
    lesson = Path(__file__).resolve().parents[2]; folder = lesson / '代码/tutorials'
    validation = static(lesson)
    original = run_notebook(folder / FILENAME, True, folder)
    annotated = run_notebook(folder / '代码注释.ipynb', False, folder)
    assert original['all6_cells_executed'] == annotated['all6_cells_executed']
    assets = []
    for name, expected in TOKENIZER_FILES.items():
        path = args.tokenizer_dir / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected
        assets.append({'file': name, 'sha256': actual, 'bytes': path.stat().st_size})
    # 常规模块导入确保Windows spawn的DataLoader worker能再次导入同名真实类。
    sys.path.insert(0, str(lesson / '代码'))
    supplement = importlib.import_module('数据读取_教学补充')
    with tempfile.TemporaryDirectory(prefix='lesson10-') as tmp:
        result = supplement.run_demo(Path(tmp), args.tokenizer_dir)
        gc.collect()
    out = lesson / '复现/outputs'
    (out / 'annotation_verification.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    record = {'environment': {'OS': platform.platform(), 'Python': platform.python_version(),
              'packages': {p: version(p) for p in ('torch', 'datasets', 'pyarrow', 'numpy', 'pandas', 'huggingface-hub', 'fsspec', 'transformers', 'tokenizers', 'IPython', 'nbformat', 'Pillow')}},
              'original': original, 'annotated': annotated, 'original_annotated_all6_results_equal': True,
              'supplement_path': '代码/数据读取_教学补充.py',
              'supplement_sha256': hashlib.sha256((lesson / '代码/数据读取_教学补充.py').read_bytes()).hexdigest(),
              'independent_teaching_supplement': result,
              'GPT2_tokenizer_revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
              'GPT2_assets_verified': assets, 'original_missing_data_code_not_invented': True,
              'C4_Pile_corpus_downloaded': False, 'full_paper_or_large_benchmark_reproduced': False}
    (out / 'runtime_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('All13 notes/6 ASTs/3 schemas/original history and 4 actual images passed; original/commented execution equal.')
    print('Independent real JSONL/Arrow maps/cache/mmap, GPT2 tokenization/padding and PyTorch loaders with2 workers passed.')

if __name__ == '__main__':
    main()
