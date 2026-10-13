# -*- coding: utf-8 -*-
"""下载固定提交的两个原 notebook 和三张必要图片，不下载模型。"""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
FILES = {
    'tutorials/LLM.int8.ipynb': '0b5ff8c81b5f5048eefc57a919bce7828899fe7c72a8bd24835c2b934628132f',
    'tutorials/bitsandbytes.ipynb': '4d1cfa595b0baa1e0c3865f2645d381bfba4d5d1a425c6737792b983642f1cdc',
    'imgs/floats-rep.png': 'f35ce5d6776b44643daf10c94a0299dd89df578e82c6a3e183cae972d266b068',
    'imgs/quant-freeze.png': '77f49147ee2f69632f62c744ed8b235f9efb7f25c123e7c093f9acfddd196523',
    'imgs/llm-int8.png': 'df22197e2a262f4cc05025be48638535215ee71ac56cffea1868c7c25663d048',
}

def main():
    lesson = Path(__file__).resolve().parents[2]
    manifest = json.loads((lesson / '来源清单.json').read_text(encoding='utf-8'))
    records = []
    for rel, expected in FILES.items():
        url = 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/' + COMMIT + '/' + rel
        with urlopen(Request(url, headers={'User-Agent': 'personal-chatgpt-notes'}), timeout=45) as response:
            raw = response.read()
        actual = hashlib.sha256(raw).hexdigest()
        assert actual == expected, (rel, actual)
        target = lesson / '代码' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        item = {'source': rel, 'url': url, 'published_path': '代码/' + rel,
                'bytes': len(raw), 'sha256': actual, 'original_bytes_preserved': True}
        if rel.endswith('.ipynb'):
            nb = json.loads(raw)
            item['cells'] = len(nb['cells'])
            item['code_cells'] = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
            item['original_environment'] = nb['metadata'].get('language_info', {})
        records.append(item)
    manifest['published_originals'] = records
    manifest['scope'] = 'LLM.int8全部56格与bitsandbytes全部3格；原代码只加注释；原输出保留。'
    (lesson / '来源清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (lesson / '复现/outputs/source_download.json').write_text(json.dumps({
        'commit': COMMIT, 'files': records, 'OPT_weights_downloaded': False,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('2 original notebooks and 3 required images downloaded; all fixed SHA256 hashes match.')

if __name__ == '__main__':
    main()
