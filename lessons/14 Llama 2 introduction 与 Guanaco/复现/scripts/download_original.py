# -*- coding: utf-8 -*-
"""保存固定上游notebook原字节；支持从已下载的Git clone读取原blob。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
REL='tutorials/llama2_introduction.ipynb'
SHA='b27440095f45a608b79666bf967276621f256ab252e7b156b758e22b73435ab6'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--upstream-clone',type=Path);args=parser.parse_args()
    url='https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+COMMIT+'/'+REL
    if args.upstream_clone:
        raw=subprocess.check_output(['git','show',COMMIT+':'+REL],cwd=args.upstream_clone)
        acquisition='original Git blob from previously downloaded upstream clone'
    else:
        with urlopen(url,timeout=45) as response: raw=response.read()
        acquisition='fixed raw URL download'
    assert hashlib.sha256(raw).hexdigest()==SHA
    root=Path(__file__).resolve().parents[2];target=root/'代码'/REL
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    item=dict(source=REL,url=url,sha256=SHA,bytes=len(raw),published_path='代码/'+REL,
              original_bytes_preserved=True,acquisition=acquisition)
    p=root/'来源清单.json';manifest=json.loads(p.read_text(encoding='utf-8'))
    manifest['published_originals']=[item]
    manifest['legacy_hash_scope']='旧sha256为本地CRLF工作副本；published_originals为固定上游原Git blob字节。'
    p.write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    out=root/'复现/outputs/source_download.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes((json.dumps(dict(commit=COMMIT,files=[item],model_weights_downloaded=False),ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('Original23-cell notebook bytes verified:',SHA)


if __name__=='__main__': main()
