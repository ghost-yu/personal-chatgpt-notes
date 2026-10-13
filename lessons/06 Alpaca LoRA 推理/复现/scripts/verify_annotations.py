# -*- coding: utf-8 -*-
"""核对57格派生文件：全部原单元/历史输出、仅增注释、IPython/普通Python AST。"""
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager


def ast_dump(source):
    return ast.dump(ast.parse(TransformerManager().transform_cell(source)),include_attributes=False)


def comment_only(old,new):
    for op,i,j,a,b in difflib.SequenceMatcher(None,old.splitlines(True),new.splitlines(True),autojunk=False).get_opcodes():
        if op=='equal':
            continue
        assert op=='insert', (op,old)
        assert all(s.lstrip().startswith('# ') for s in new.splitlines(True)[a:b])


def main():
    lesson=Path(__file__).resolve().parents[2]
    directory=lesson/'代码/tutorials'
    original_path=directory/'06_alpaca_llama_7b_lora.ipynb'
    original=json.loads(original_path.read_text(encoding='utf-8'))
    manifest=json.loads((lesson/'来源清单.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(original_path.read_bytes()).hexdigest()
    assert digest==manifest['upstream_sha256']
    nbformat.validate(nbformat.from_dict(original))
    reports=[]
    for name in ('原代码_带说明.ipynb','代码注释.ipynb'):
        path=directory/name
        result=json.loads(path.read_text(encoding='utf-8'))
        nbformat.validate(nbformat.from_dict(result))
        assert len(result['cells'])==57
        assert {k:v for k,v in result.items() if k!='cells'}=={k:v for k,v in original.items() if k!='cells'}
        for i,cell in enumerate(original['cells']):
            note,keep=result['cells'][1+2*i:3+2*i]
            assert '## 单元 {}'.format(i) in ''.join(note['source'])
            assert {k:v for k,v in keep.items() if k!='source'}=={k:v for k,v in cell.items() if k!='source'}
            if name=='代码注释.ipynb' and cell['cell_type']=='code':
                old,new=''.join(cell['source']),''.join(keep['source'])
                assert ast_dump(old)==ast_dump(new)
                comment_only(old,new)
            else:
                assert keep['source']==cell['source']
        reports.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'schema':True,'cells':57,
                        'all_28_original_metadata_and_outputs_preserved':True,'all_18_code_ASTs_equal':True})
    py_path=directory/'06_alpaca_llama_7b_lora_代码注释.py'
    source=py_path.read_text(encoding='utf-8')
    ordinary='\n'.join(''.join(c['source']) for i,c in enumerate(original['cells']) if c['cell_type']=='code' and i!=0)
    assert ast.dump(ast.parse(source),include_attributes=False)==ast.dump(ast.parse(ordinary),include_attributes=False)
    assert source.count('# IPython原指令：!pip install')==2
    doc=(lesson/'代码单元解释.md').read_text(encoding='utf-8')
    assert [int(x) for x in re.findall(r'^## 单元 (\d+)（',doc,re.M)]==list(range(28))
    record={'original_sha256':digest,'variants':reports,'code_cells':manifest['python_code_cells'],
            'all_28_cell_explanations_present':True,'annotation_only_comments':True,
            'literal_prompts_unchanged_by_AST_check':True,'reading_py_AST_equals_17_non_install_cells':True,
            'two_shell_commands_preserved_as_comments':True,'reading_py_sha256':hashlib.sha256(py_path.read_bytes()).hexdigest()}
    (lesson/'复现/outputs/annotation_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('28 cell explanations, 18 equivalent code ASTs, original histories, literal prompts and 3 schemas passed.')


if __name__=='__main__':
    main()
