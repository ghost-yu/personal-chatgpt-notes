# -*- coding: utf-8 -*-
"""核对19格原内容、12格代码及Markdown1源码块仅加注释，schema/AST/历史输出。"""
import ast,difflib,hashlib,json,re
from pathlib import Path
import nbformat

def dump(s):return ast.dump(ast.parse(s),include_attributes=False)
def comments(old,new):
    a,b=old.splitlines(True),new.splitlines(True)
    for op,i,j,u,v in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if op=='equal':continue
        assert op=='insert' and all(s.lstrip().startswith('# ') for s in b[u:v])
    assert dump(old)==dump(new)

def main():
    lesson=Path(__file__).resolve().parents[2];directory=lesson/'代码/tutorials'
    path=directory/'07_fp16_mixed_precision_training.ipynb'
    original=json.loads(path.read_text(encoding='utf-8'))
    manifest=json.loads((lesson/'来源清单.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['upstream_sha256']
    nbformat.validate(nbformat.from_dict(original));reports=[]
    fence=re.compile(r'(```[^\n]*\n)(.*?)(```)',re.S)
    for name in ('原代码_带说明.ipynb','代码注释.ipynb'):
        nbpath=directory/name;nb=json.loads(nbpath.read_text(encoding='utf-8'))
        nbformat.validate(nbformat.from_dict(nb))
        assert len(nb['cells'])==39
        assert {k:v for k,v in nb.items() if k!='cells'}=={k:v for k,v in original.items() if k!='cells'}
        for i,c in enumerate(original['cells']):
            note,keep=nb['cells'][1+2*i:3+2*i]
            assert '## 单元 {}'.format(i) in ''.join(note['source'])
            assert {k:v for k,v in keep.items() if k!='source'}=={k:v for k,v in c.items() if k!='source'}
            old,new=''.join(c['source']),''.join(keep['source'])
            if name=='代码注释.ipynb' and c['cell_type']=='code':comments(old,new)
            elif name=='代码注释.ipynb' and i==1:
                a,b=list(fence.finditer(old)),list(fence.finditer(new));assert len(a)==len(b)==1
                comments(a[0][2],b[0][2])
                assert old[:a[0].start()]==new[:b[0].start()] and old[a[0].end():]==new[b[0].end():]
            else:assert keep['source']==c['source']
        reports.append({'file':name,'sha256':hashlib.sha256(nbpath.read_bytes()).hexdigest(),'schema':True,
          'all19_original_cells_metadata_and_outputs_preserved':True,'all12_code_ASTs_equal':True,'Markdown1_AST_equal':True})
    py=directory/'07_fp16_mixed_precision_training_代码注释.py'
    plain='\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type']=='code')
    assert dump(py.read_text(encoding='utf-8'))==dump(plain)
    doc=(lesson/'代码单元解释.md').read_text(encoding='utf-8')
    assert [int(x) for x in re.findall(r'^## 单元 (\d+)（',doc,re.M)]==list(range(19))
    record={'original_sha256':manifest['upstream_sha256'],'variants':reports,'all19_notes_present':True,
      'annotation_only_comments':True,'12_code_cells_and1_Markdown_block_AST_equal':True,
      'reading_py_AST_equals_original12_code_cells':True,'reading_py_sha256':hashlib.sha256(py.read_bytes()).hexdigest()}
    (lesson/'复现/outputs/annotation_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('All19 notes;12 code ASTs;Markdown1 AST;original metadata/history and3schemas passed.')

if __name__=='__main__':main()
