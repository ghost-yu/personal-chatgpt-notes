# -*- coding: utf-8 -*-
"""取得固定上游原notebook；远程失败时可从已下载Git仓库的原blob恢复。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
REL='tutorials/llm_training_sft.ipynb'
SHA='bcbb4c7c4e7e0310a07e8dccf0437eb3e0597031288e2a6332b3c677384dfa51'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-clone',type=Path,help='可选：已下载上游Git仓库，直接读固定commit的原blob，不读CRLF工作副本')
    args=parser.parse_args()
    url='https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+COMMIT+'/'+REL
    if args.upstream_clone is None:
        with urlopen(url,timeout=45) as response: data=response.read()
        acquisition='fixed raw URL download'
    else:
        data=subprocess.check_output(['git','show',COMMIT+':'+REL],cwd=args.upstream_clone)
        acquisition='original Git blob from previously downloaded upstream clone'
    assert hashlib.sha256(data).hexdigest()==SHA
    lesson=Path(__file__).resolve().parents[2]
    target=lesson/'代码'/REL
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    item={'source':REL,'url':url,'sha256':SHA,'bytes':len(data),'published_path':'代码/'+REL,
          'original_bytes_preserved':True,'acquisition':acquisition}
    manifest_path=lesson/'来源清单.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['published_originals']=[item]
    manifest['legacy_hash_scope']='旧sha256为本地CRLF工作副本；published_originals为上游固定Git blob原字节。'
    manifest_path.write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    (lesson/'复现/outputs/source_download.json').write_bytes((json.dumps({'commit':COMMIT,'files':[item],
        'model_weights_downloaded':False,'full_Alpaca_downloaded':False},ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('Fixed original27-cell notebook SHA256 matches:',SHA)

if __name__=='__main__':main()
