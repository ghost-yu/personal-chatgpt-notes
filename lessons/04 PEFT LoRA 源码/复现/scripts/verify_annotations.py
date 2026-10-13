"""核对两份notebook的全部原单元/输出、仅增注释及8格+6块AST。"""
import argparse
import ast
import json
from pathlib import Path
import nbformat
from build_annotated import source,blocks,CODE_NOTES


def comments_only(before,after):
    old=before.splitlines();at=0;added=0
    for line in after.splitlines():
        if at<len(old) and line==old[at]:at+=1
        else:
            assert line.lstrip().startswith('# '),line
            added+=1
    assert at==len(old) and added>0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',required=True)
    parser.add_argument('--record',required=True)
    args=parser.parse_args();directory=Path(args.directory)
    original=nbformat.read(str(directory/'04_PEFT_lora.ipynb'),as_version=4);nbformat.validate(original)
    report={'status':'成功（静态注释核对）','original_cells':18,'code_cells':8,'markdown_blocks':6,'artifacts':{}}
    for filename in ['原代码_带说明.ipynb','代码注释.ipynb']:
        current=nbformat.read(str(directory/filename),as_version=4);nbformat.validate(current)
        assert len(current.cells)==37
        preserved=current.cells[2::2];assert len(preserved)==18
        checked=[];block_count=0
        for i,(before,after) in enumerate(zip(original.cells,preserved)):
            a,b=dict(before),dict(after)
            if filename=='代码注释.ipynb' and (before.cell_type=='code' or blocks(source(before))):
                comments_only(source(before),source(after))
                if before.cell_type=='code':
                    assert ast.dump(ast.parse(source(before)))==ast.dump(ast.parse(source(after)))
                    checked.append(i)
                else:
                    x,y=blocks(source(before)),blocks(source(after));assert len(x)==len(y)
                    for u,v in zip(x,y):
                        assert ast.dump(ast.parse(u))==ast.dump(ast.parse(v))
                        block_count+=1
                a.pop('source');b.pop('source')
            assert a==b,(filename,i)
        if filename=='代码注释.ipynb':assert set(checked)==set(CODE_NOTES) and block_count==6
        report['artifacts'][filename]={'schema_valid':True,'original_cells_and_history_preserved':True,
            'comment_only':True,'code_ast_equal':checked,'markdown_blocks_ast_equal':block_count}
    report['execution']='未执行原7B加载；固定官方PEFT源码真实CPU验证另见verify_injection.py'
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('original cells/history preserved; 8 code ASTs + 6 Markdown ASTs and comment-only checks passed')
