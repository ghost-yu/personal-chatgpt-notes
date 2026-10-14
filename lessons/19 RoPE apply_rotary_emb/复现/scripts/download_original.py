# -*- coding: utf-8 -*-
"""Save the pinned notebook and its SentencePiece asset as exact Git blobs."""
import hashlib
import json
import subprocess
from pathlib import Path

FILES={
    'tutorials/llama2_src_03_RoPE_apply_rotary_emb.ipynb':'59fb6124d4c14dfe40c1cb1053a43458f0b745183447195abab18e3609d94f1e',
    'tutorials/tokenizer.model':'9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347',
}

def main():
    root=Path(__file__).resolve().parents[2];p=root/'来源清单.json';m=json.loads(p.read_bytes())
    for rel,expected in FILES.items():
        data=subprocess.check_output(['git','-C','G:/zzz/Llama 入门教程/代码/personal_chatgpt','show',m['tutorial_commit']+':'+rel])
        assert hashlib.sha256(data).hexdigest()==expected
        target=root/'代码'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        if rel.endswith('.ipynb'):
            m.setdefault('legacy_local_crlf_sha256',m['notebook_sha256'])
            m.update(notebook_sha256=expected,notebook_bytes=len(data),line_endings='upstream fixed Git blob; LF bytes preserved')
        else:m['tokenizer_bytes']=len(data)
    m['transcript']='当前E:/根目录30个txt未找到独立第19讲字幕；05717ac9为第28讲回顾，不冒用'
    p.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('Saved fixed notebook43654bytes and tokenizer499723bytes, exact SHA256 verified.')

if __name__=='__main__':main()
