# -*- coding: utf-8 -*-
"""核对原字节、所有原单元/历史输出、仅增注释、IPython转换AST和阅读版AST。"""
import ast
import hashlib
import json
import re
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager


def dump_ast(source):
    return ast.dump(ast.parse(TransformerManager().transform_cell(source)), include_attributes=False)


def main():
    lesson = Path(__file__).resolve().parents[2]
    directory = lesson/'代码/tutorials'
    raw_path = directory/'05_float16_bf16.ipynb'
    original = json.loads(raw_path.read_text(encoding='utf-8'))
    manifest = json.loads((lesson/'来源清单.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == manifest['upstream_sha256']
    for asset in manifest['images']:
        assert hashlib.sha256((lesson/'代码'/asset['path']).read_bytes()).hexdigest() == asset['sha256']
    reports = []
    nbformat.validate(nbformat.from_dict(original))
    for name in ('原代码_带说明.ipynb', '代码注释.ipynb'):
        path = directory/name
        nb = json.loads(path.read_text(encoding='utf-8'))
        nbformat.validate(nbformat.from_dict(nb))
        assert len(nb['cells']) == 55
        assert {k:v for k,v in nb.items() if k != 'cells'} == {k:v for k,v in original.items() if k != 'cells'}
        for index, old in enumerate(original['cells']):
            note, keep = nb['cells'][1+2*index:3+2*index]
            assert '## 单元 {}'.format(index) in ''.join(note['source'])
            assert {k:v for k,v in keep.items() if k != 'source'} == {k:v for k,v in old.items() if k != 'source'}
            source, new = ''.join(old['source']), ''.join(keep['source'])
            if name == '代码注释.ipynb' and old['cell_type']=='code':
                assert new.endswith(source) and len(new) > len(source)
                prefix = new[:-len(source)]
                assert all(line.startswith('# ') for line in prefix.splitlines())
                assert dump_ast(new) == dump_ast(source)
            else:
                assert keep['source'] == old['source']
        reports.append({'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'cells':55, 'original_cells':27, 'schema':True,
                        'all_original_metadata_and_outputs_preserved':True,
                        'all_8_code_ASTs_equal_after_IPython_transform':True})
    ordinary = '\n'.join(''.join(c['source']) for i,c in enumerate(original['cells']) if c['cell_type']=='code' and i != 20)
    py_path = directory/'05_float16_bf16_代码注释.py'
    py = py_path.read_text(encoding='utf-8')
    assert ast.dump(ast.parse(py), include_attributes=False) == ast.dump(ast.parse(ordinary), include_attributes=False)
    assert '# IPython原指令：torch.cuda.is_bf16_supported??' in py
    doc = (lesson/'代码单元解释.md').read_text(encoding='utf-8')
    ids = [int(x) for x in re.findall(r'^## 单元 (\d+)（', doc, re.M)]
    assert ids == list(range(27))
    record = {'original_sha256':manifest['upstream_sha256'], 'variants':reports,
              'code_cells':[0,5,16,17,18,19,20,26], 'markdown_cells':19,
              'all_27_cell_explanations_present':True, 'annotation_only_comments':True,
              'reading_py_AST_equals_7_ordinary_original_code_cells':True,
              'reading_py_IPython_directive_preserved_as_comment':True,
              'reading_py_sha256':hashlib.sha256(py_path.read_bytes()).hexdigest()}
    (lesson/'复现/outputs/annotation_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('27 cell explanations, 8 equivalent code ASTs, all original outputs/metadata and 3 notebook schemas passed.')


if __name__ == '__main__':
    main()
