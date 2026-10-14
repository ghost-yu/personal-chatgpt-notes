# -*- coding: utf-8 -*-
"""从固定上游Git blob保存两份课程notebook与配套图片。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
FILES=['tutorials/llama2_src_01.ipynb','tutorials/llama2_src_text_chat.ipynb','imgs/llama_weights.png','imgs/llama2-weights.png']
SHA={'tutorials/llama2_src_01.ipynb':'b537815b1f94a1be6d33d4e05d2bb290372408dfd9f2f4027b3c77400c0a839c',
     'tutorials/llama2_src_text_chat.ipynb':'fd822495540fb813a0bb4697a75e3abaf08aeefec2d84eada1123f20b84b686e'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream-clone',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];records=[]
    for rel in FILES:
        url='https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+COMMIT+'/'+rel
        if a.upstream_clone: data=subprocess.check_output(['git','show',COMMIT+':'+rel],cwd=a.upstream_clone)
        else:
            with urlopen(url,timeout=45) as response:data=response.read()
        digest=hashlib.sha256(data).hexdigest()
        if rel in SHA:assert digest==SHA[rel]
        target=root/'代码'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records.append(dict(source=rel,url=url,sha256=digest,bytes=len(data),published_path='代码/'+rel,
                            acquisition='fixed Git blob' if a.upstream_clone else 'fixed raw URL'))
    out=root/'复现/outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'source_download.json').write_bytes((json.dumps(dict(commit=COMMIT,files=records),ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    manifest=root/'来源清单.json';m=json.loads(manifest.read_bytes());m['published_originals']=records
    m['legacy_hash_scope']='旧sha为本地CRLF工作副本；公开原件为固定Git blob字节。'
    manifest.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('Two original notebooks and two PNG assets saved byte-exact.')

if __name__=='__main__':main()
