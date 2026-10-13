# -*- coding: utf-8 -*-
"""下载固定课程原notebook和必要图片，按远程原字节校验。"""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
FILES = {
    'tutorials/gradient_accumulation.ipynb': '0d4ddca7593b5d01a3d2855411f6058b419029519f08c292103123e9d3cd131b',
    'imgs/grad-accumulation.png': '2ea922add3bb4c788a56d345afa965445382094e549e53b60f3f236881b48b2d',
}

def main():
    lesson = Path(__file__).resolve().parents[2]
    records = []
    for rel, expected in FILES.items():
        url = 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/' + COMMIT + '/' + rel
        with urlopen(url, timeout=45) as response:
            data = response.read()
        assert hashlib.sha256(data).hexdigest() == expected, rel
        target = lesson / '代码' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({'source': rel, 'url': url, 'bytes': len(data), 'sha256': expected,
                        'published_path': '代码/' + rel, 'original_bytes_preserved': True})
    manifest_path = lesson / '来源清单.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['published_originals'] = records
    manifest['legacy_hash_scope'] = '旧sha256为本地Git工作副本；published_originals为远程原字节。'
    manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    out = lesson / '复现/outputs/source_download.json'
    out.write_bytes((json.dumps({'commit': COMMIT, 'files': records, 'BERT_weights_downloaded': False},
                               ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print('Original notebook and image: fixed remote SHA256 hashes match.')

if __name__ == '__main__':
    main()
