# -*- coding: utf-8 -*-
"""下载固定原版，不运行大CNN、下载MNIST或训练。"""
import hashlib,json
from pathlib import Path
from urllib.request import Request,urlopen

COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
REL='tutorials/07_fp16_mixed_precision_training.ipynb'
SHA='c007e4adba74d83cb577dc55c7df24697ff88c69b820e6b19996a2bfc4c852c7'

def main():
    lesson=Path(__file__).resolve().parents[2]
    url='https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+COMMIT+'/'+REL
    with urlopen(Request(url,headers={'User-Agent':'personal-chatgpt-notes'}),timeout=60) as response:
        raw=response.read()
    assert hashlib.sha256(raw).hexdigest()==SHA
    nb=json.loads(raw.decode('utf-8'))
    code=[i for i,c in enumerate(nb['cells']) if c['cell_type']=='code']
    assert len(nb['cells'])==19 and code==[3,4,6,7,8,10,11,12,14,15,17,18]
    target=lesson/'代码'/REL
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    manifest={'repository':'https://github.com/chunhuizhang/personal_chatgpt','commit':COMMIT,'notebook':REL,
      'source_url':url,'upstream_sha256':SHA,'published_original_path':'代码/'+REL,'published_original_sha256':SHA,
      'all_cells':list(range(19)),'python_code_cells':code,'markdown_code_cells':[1],'external_images':[],
      'original_environment':nb['metadata']['language_info'],
      'scope':'19格及Markdown1示例详注；原大CNN/真实MNIST/3090显存实验与CPU补充明确分开'}
    (lesson/'来源清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    out=lesson/'复现/outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'source_download.json').write_text(json.dumps({'url':url,'commit':COMMIT,'bytes':len(raw),
      'sha256':SHA,'hash_matched':True,'original_bytes_unmodified':True,'MNIST_downloaded':False},indent=2)+'\n',encoding='utf-8')
    print('Original19cells downloaded unchanged; fixed hash passed; no external PNG.')

if __name__=='__main__':main()
