# -*- coding: utf-8 -*-
"""只下载固定提交的原notebook，校验字节；不安装软件、下载模型或设置代理。"""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
RELATIVE = 'tutorials/06_alpaca_llama_7b_lora.ipynb'
EXPECTED = '501575b319fcbedcd97b10f80173d3433b2c3d21fe06b3e78138614ab415f1d1'


def main():
    lesson = Path(__file__).resolve().parents[2]
    url = 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+COMMIT+'/'+RELATIVE
    with urlopen(Request(url,headers={'User-Agent':'personal-chatgpt-notes'}),timeout=60) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    nb = json.loads(raw.decode('utf-8'))
    code = [i for i,c in enumerate(nb['cells']) if c['cell_type']=='code']
    assert len(nb['cells'])==28 and len(code)==18
    target = lesson/'代码'/RELATIVE
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    manifest = {'repository':'https://github.com/chunhuizhang/personal_chatgpt','commit':COMMIT,
       'notebook':RELATIVE,'source_url':url,'upstream_sha256':EXPECTED,
       'published_original_path':'代码/'+RELATIVE,'published_original_sha256':EXPECTED,
       'all_cells':list(range(28)),'python_code_cells':code,
       'installation_cell':0,'inactive_proxy_cell':8,'interactive_cell':26,'empty_cell':27,
       'external_images':[], 'original_python_metadata':nb['metadata']['language_info'],
       'original_remote_model_and_adapter_revisions':'not pinned in notebook',
       'verification_peft_commit':'7120c9a2636f93c8db3fc4ae466e02f338ecead8',
       'scope':'原单元与历史输出保留；注释所有代码，CPU教学验证独立于原7B/8bit/CUDA演示'}
    (lesson/'来源清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    out = lesson/'复现/outputs'
    out.mkdir(parents=True,exist_ok=True)
    (out/'source_download.json').write_text(json.dumps({'url':url,'commit':COMMIT,'sha256':EXPECTED,
        'bytes':len(raw),'hash_matched':True,'original_bytes_unmodified':True,'model_downloaded':False},indent=2)+'\n',encoding='utf-8')
    print('Downloaded original 28-cell notebook; fixed hash matched; no external images referenced.')


if __name__=='__main__':
    main()
