"""不执行原pip/下载/训练；核对所有单元、注释-only、IPython转换AST和schema。"""
import argparse
import ast
import json
import re
from pathlib import Path
import nbformat
from build_annotated import source,ast_signature,COMMENTS,ordinary_python


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',required=True)
    parser.add_argument('--record',required=True)
    args=parser.parse_args()
    directory=Path(args.directory)
    original=nbformat.read(str(directory/'03_lora_practice.ipynb'),as_version=4)
    nbformat.validate(original)
    report={'status':'成功（注释静态核对）','cells':47,'code_cells':37,'artifacts':{}}
    for filename in ['原代码_带说明.ipynb','代码注释.ipynb']:
        current=nbformat.read(str(directory/filename),as_version=4);nbformat.validate(current)
        assert len(current.cells)==95
        for note in current.cells[1::2]:
            for link in re.findall(r'\]\(([^)]+)\)',source(note)):
                if '://' not in link and not link.startswith('#'):
                    assert (directory/link).exists(),('broken inserted note link',link)
        preserved=current.cells[2::2];assert len(preserved)==47
        checked=[]
        for i,(before,after) in enumerate(zip(original.cells,preserved)):
            a,b=dict(before),dict(after)
            if filename=='代码注释.ipynb' and before.cell_type=='code':
                assert ast_signature(source(before))==ast_signature(source(after)),i
                old_lines=source(before).splitlines();at=0;added=0
                for line in source(after).splitlines():
                    if at<len(old_lines) and line==old_lines[at]:at+=1
                    else:
                        assert line.lstrip().startswith('# '),(i,line)
                        added+=1
                assert at==len(old_lines) and added>0
                checked.append(i);a.pop('source');b.pop('source')
            assert a==b,(filename,i)
        if filename=='代码注释.ipynb':assert set(checked)==set(COMMENTS)
        report['artifacts'][filename]={'schema_valid':True,'original_cells_and_history_preserved':True,
                                       'comment_only_and_ast_equal':True,'code_cells_checked':checked}
    readable=(directory/'03_lora_practice_代码注释.py').read_text(encoding='utf-8')
    expected='\n'.join(ordinary_python(source(c)) for c in original.cells if c.cell_type=='code')
    assert ast.dump(ast.parse(readable))==ast.dump(ast.parse(expected))
    directives=[line for c in original.cells if c.cell_type=='code' for line in source(c).splitlines()
                if line.lstrip().startswith(('!', '%'))]
    assert len(directives)==5
    assert all('# IPython原指令（在notebook中执行）：'+line.strip() in readable for line in directives)
    report['readable_python']={'ast_equal_after_omitting_ipython_directives':True,
                              'ipython_directives_preserved_as_comments':len(directives),
                              'inserted_note_relative_links_valid':True}
    report['execution']='未执行原安装、magic、模型下载和训练；实际CPU验证另见verify_pipeline.py'
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('all original cells/history preserved; 37/37 code AST and comment-only checks passed')
