# -*- coding: utf-8 -*-
"""Extract exact upstream blobs from the user's existing clone; never reserialize originals."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
FILES = ['tutorials/llama2_src_cache_kv.ipynb', 'tutorials/tokenizer.model',
         'imgs/gpt-2-autoregression-2.gif', 'imgs/gpt4-price.png', 'imgs/qkv.png',
         'imgs/decoding-qkv.png', 'imgs/kvmatrices.png', 'imgs/kv-cache.gif']
LESSON = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', required=True)
    args = parser.parse_args()
    entries = []
    for name in FILES:
        data = subprocess.check_output(['git', 'show', COMMIT + ':' + name], cwd=args.upstream)
        target = LESSON / '代码' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entries.append({'upstream_path': name, 'published_path': '代码/' + name,
                        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    report = {'repository': 'https://github.com/chunhuizhang/personal_chatgpt',
              'commit': COMMIT, 'files': entries}
    (LESSON / '来源清单.json').write_bytes((json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    print('Extracted exact blobs:', len(entries))

if __name__ == '__main__':
    main()
