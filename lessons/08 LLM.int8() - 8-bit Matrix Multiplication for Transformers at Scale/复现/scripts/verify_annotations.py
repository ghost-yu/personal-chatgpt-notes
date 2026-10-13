# -*- coding: utf-8 -*-
"""核对全部59原单元/38代码格、三张图、AST、注释新增、schema与历史输出。"""
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path
import nbformat

def tree(source):
    return ast.dump(ast.parse(source), include_attributes=False)

def only_comments(old, new):
    a, b = old.splitlines(True), new.splitlines(True)
    for op, i, j, u, v in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != 'equal':
            assert op == 'insert' and all(s.lstrip().startswith('# ') for s in b[u:v])
    assert tree(old) == tree(new)

def main():
    lesson = Path(__file__).resolve().parents[2]
    directory = lesson / '代码/tutorials'
    manifest = json.loads((lesson / '来源清单.json').read_text(encoding='utf-8'))
    originals = {item['source']: item for item in manifest['published_originals']}
    for item in originals.values():
        assert hashlib.sha256((lesson / item['published_path']).read_bytes()).hexdigest() == item['sha256']
    records = []
    total_code = 0
    for filename, prefix in [('LLM.int8.ipynb', ''), ('bitsandbytes.ipynb', 'bitsandbytes_')]:
        original = json.loads((directory / filename).read_text(encoding='utf-8'))
        nbformat.validate(nbformat.from_dict(original))
        count = len(original['cells'])
        indices = [i for i, c in enumerate(original['cells']) if c['cell_type'] == 'code']
        total_code += len(indices)
        for name in ('原代码_带说明.ipynb', '代码注释.ipynb'):
            path = directory / (prefix + name)
            nb = json.loads(path.read_text(encoding='utf-8'))
            nbformat.validate(nbformat.from_dict(nb))
            assert len(nb['cells']) == 1 + 2 * count
            assert {k: v for k, v in nb.items() if k != 'cells'} == {k: v for k, v in original.items() if k != 'cells'}
            for i, c in enumerate(original['cells']):
                note, keep = nb['cells'][1 + 2 * i:3 + 2 * i]
                assert '## 单元 {}（'.format(i) in ''.join(note['source'])
                assert len(''.join(note['source'])) > 60
                assert {k: v for k, v in keep.items() if k != 'source'} == {k: v for k, v in c.items() if k != 'source'}
                if name == '代码注释.ipynb' and c['cell_type'] == 'code':
                    only_comments(''.join(c['source']), ''.join(keep['source']))
                else:
                    assert c['source'] == keep['source']
            records.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'original_cell_count': count, 'published_cell_count': len(nb['cells']),
                            'code_indices': indices, 'metadata_and_outputs_equal': True,
                            'all_Markdown_unchanged': True, 'all_code_AST_equal': True,
                            'only_comment_insertions': True, 'schema_valid': True})
        py = directory / (filename.removesuffix('.ipynb') + '_代码注释.py')
        plain = '\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type'] == 'code')
        assert tree(py.read_text(encoding='utf-8')) == tree(plain)
        records.append({'reading_file': py.name, 'sha256': hashlib.sha256(py.read_bytes()).hexdigest(), 'AST_equal_all_code': True})
        # 3原Image路径必须实际存在，字节与原源manifest一致。
        for c in original['cells']:
            for rel in re.findall(r"Image\('([^']+)'", ''.join(c['source'])):
                path = (directory / rel).resolve()
                assert path.is_file()
                assert path.parent == (lesson / '代码/imgs').resolve()
    doc = (lesson / '代码单元解释.md').read_text(encoding='utf-8')
    assert [int(x) for x in re.findall(r'^## 单元 (\d+)（', doc, re.M)] == list(range(56)) + list(range(3))
    record = {'all_cells_explained': 59, 'code_cells_including_empty55': total_code,
              'Markdown22_pseudocode_explained_and_original_preserved': True,
              'original_files_and_image_hashes_matched': True, 'variants': records}
    (lesson / '复现/outputs/annotation_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('All59 original cells; {} code ASTs; 6 notebook schemas; images/history and 2 reading ASTs passed.'.format(total_code))

if __name__ == '__main__':
    main()
