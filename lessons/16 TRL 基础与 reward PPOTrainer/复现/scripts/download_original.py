# -*- coding: utf-8 -*-
"""固定上游原件及五张配图；图片放到原相对路径实际能找到的位置。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen
COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
REL='tutorials/trl_hf/trl_basics.ipynb'
SHA='1d0ec8e4f196914257158d18a48746b12561fa68100aeb778105720bd8b8d474'
IMAGES=['trl_steps.png','rm_ppo.png','ppo_algo.png','trl-sft.png','trl_overview.png']

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream-clone',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];items=[]
    for rel in [REL]+['imgs/'+n for n in IMAGES]:
        url='https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+COMMIT+'/'+rel
        if a.upstream_clone:b=subprocess.check_output(['git','show',COMMIT+':'+rel],cwd=a.upstream_clone)
        else:
            with urlopen(url,timeout=45) as response:b=response.read()
        if rel==REL:assert hashlib.sha256(b).hexdigest()==SHA
        target=root/'代码'/rel if rel==REL else root/'代码/tutorials/imgs'/Path(rel).name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        items.append(dict(source=rel,url=url,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),published_path=str(target.relative_to(root)),acquisition='fixed Git blob' if a.upstream_clone else 'fixed raw URL'))
    out=root/'复现/outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'source_download.json').write_bytes((json.dumps(dict(commit=COMMIT,files=items,duplicate_root_notebook='tutorials/trl_basics.ipynb has identical Git blob; not duplicated'),ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    p=root/'来源清单.json';m=json.loads(p.read_bytes());m['published_originals']=items;m['legacy_hash_scope']='旧sha为CRLF工作副本；公开原件为固定Git blob。图片字节不改，放在原../imgs调用对应目录。'
    p.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('42-cell original and five images saved.')
if __name__=='__main__':main()
