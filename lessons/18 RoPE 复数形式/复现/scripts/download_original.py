# -*- coding: utf-8 -*-
"""Preserve two fixed notebook blobs and their original image."""
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
FILES={
    'tutorials/llama2_src_03_RoPE_01_complex_position_encoding.ipynb':'678878a734027baf88ef14e9043b57ac96407cdf66c28e68c07bd98a46d87b4c',
    'tutorials/llama2_src_03_RoPE.ipynb':'000ecf521d2154e4483865ae2edd620829266416461c2a2888d6c75c977053c9',
    'imgs/hf_llama_rope.png':'9317d3375d0336efb8e48ab3d2aff8c4d6ab816943c3cecb3631015c67f87c68',
}

def main():
    root=Path(__file__).resolve().parents[2]
    m=json.loads((root/'来源清单.json').read_bytes())
    assets=[]
    for rel,expected in FILES.items():
        data=subprocess.check_output(['git','-C','G:/zzz/Llama 入门教程/代码/personal_chatgpt','show',COMMIT+':'+rel])
        assert hashlib.sha256(data).hexdigest()==expected
        path=root/'代码'/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        if rel.endswith('.ipynb'):
            entry=next(n for n in m['notebooks'] if n['notebook']==rel)
            entry.setdefault('legacy_local_crlf_sha256',entry['sha256'])
            entry.update(sha256=expected,bytes=len(data),line_endings='fixed Git blob bytes preserved; LF')
        else:
            assets.append(dict(upstream=rel,published='代码/'+rel,sha256=expected,bytes=len(data)))
    m['images']=assets
    m['transcript']='当前E:/根目录30个txt中未找到独立第18讲字幕；05717ac9为第28讲回顾，qwen3文件为其他模型材料，不冒用'
    (root/'来源清单.json').write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('Saved two original notebooks and one original PNG, all fixed hashes verified.')

if __name__=='__main__':main()
