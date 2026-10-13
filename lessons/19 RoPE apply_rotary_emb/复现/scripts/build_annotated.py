"""Insert an explanation before every original code cell without changing it."""
import argparse,copy,hashlib,json,re,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--notebook',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    m=json.loads((ROOT/'来源清单.json').read_text(encoding='utf-8'))
    raw=args.notebook.read_bytes() if args.notebook else urllib.request.urlopen('https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+m['tutorial_commit']+'/'+m['notebook'],timeout=30).read()
    assert hashlib.sha256(raw).hexdigest() in [m['notebook_sha256'],m['notebook_upstream_sha256']]
    nb=json.loads(raw);doc=(ROOT/'代码单元解释.md').read_text(encoding='utf-8')
    matches=list(re.finditer(r'^## 单元 (\d+)：.*$',doc,re.M));blocks={}
    for i,match in enumerate(matches):
        number=int(match.group(1));assert number not in blocks
        blocks[number]=doc[match.start():matches[i+1].start() if i+1<len(matches) else len(doc)]
    expected=[i for i,c in enumerate(nb['cells']) if c['cell_type']=='code']
    assert sorted(blocks)==expected,'Every code cell needs exactly one explanation'
    result=copy.deepcopy(nb);cells=[{'cell_type':'markdown','metadata':{},'id':'reading-header','source':['# 第19讲带说明原代码\n','原始单元编号保留在说明中。代码和历史输出未改写；插入说明会改变文件中的物理单元位置。\n','历史输出来自上游，不是本次结果；本次结果见复现记录。缺失helper在说明中明确记录。\n']}]
    for i,c in enumerate(nb['cells']):
        if i in blocks:cells.append({'cell_type':'markdown','metadata':{},'id':'reading-'+str(i),'source':blocks[i].splitlines(keepends=True)})
        cells.append(copy.deepcopy(c))
    result['cells']=cells
    original_code=[c for c in nb['cells'] if c['cell_type']=='code']
    output_code=[c for c in result['cells'] if c['cell_type']=='code']
    assert original_code==output_code
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    audit={'expected_code_cell_numbers':expected,'covered_code_cell_numbers':sorted(blocks),'all_code_cells_explained':True,'original_code_and_historical_outputs_unchanged':original_code==output_code,'original_cells':len(nb['cells']),'annotated_cells':len(cells),'annotated_notebook_filename':args.output.name,'publication':'Full original code with inserted explanations is a local companion artifact; public repository carries the explanations and builder.'}
    (ROOT/'复现/outputs/code_unit_coverage.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
