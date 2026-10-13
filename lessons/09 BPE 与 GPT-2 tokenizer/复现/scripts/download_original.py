# -*- coding: utf-8 -*-
"""下载固定原notebook/配图；可选仅下载两套固定tokenizer资产到本地缓存。"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
FILES = {
    'tutorials/tokenizer_basics.ipynb': 'd5a46395802d3f9db31feb6e777ee727931841a9802f7f16599ef705f71a7569',
    'tutorials/gpt2_train_tokenizer.ipynb': 'b5916906f3f4246e6e0d1cb6c7a80712bba000d2f65f2b6d3b9ebe9f584ab9df',
    'tutorials/gpt2.ipynb': 'fbc5d24c92f71334cbee02f3293a51528469d4491a99d3824eadae9bbad88cdd',
    'imgs/token_steps.png': '4db0802ef4f6593b0120d5e3bdf29cdba7eab16324c57a1c9b54a9be4f842c66',
}
ASSETS = {
    'gpt2': ('openai-community/gpt2', '607a30d783dfa663caf39e06633721c8d4cfcd7e', ['config.json', 'tokenizer_config.json', 'tokenizer.json']),
    't5': ('google-t5/t5-base', 'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1', ['config.json', 'tokenizer.json', 'spiece.model']),
}

def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'personal-chatgpt-notes'}), timeout=45) as response:
        return response.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-cache', type=Path, help='可选缓存路径；只下载配置/tokenizer，绝不下载模型权重')
    args = parser.parse_args()
    lesson = Path(__file__).resolve().parents[2]
    manifest = json.loads((lesson / '来源清单.json').read_text(encoding='utf-8'))
    records = []
    for rel, expected in FILES.items():
        url = 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/' + COMMIT + '/' + rel
        raw = fetch(url)
        assert hashlib.sha256(raw).hexdigest() == expected, rel
        target = lesson / '代码' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        item = {'source': rel, 'url': url, 'published_path': '代码/' + rel,
                'sha256': expected, 'bytes': len(raw), 'original_bytes_preserved': True}
        if rel.endswith('.ipynb'):
            nb = json.loads(raw)
            item['cells'] = len(nb['cells'])
            item['code_indices'] = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
        records.append(item)
    manifest['published_originals'] = records
    manifest['legacy_notebook_hash_scope'] = 'notebooks字段保留早期本地Git工作副本（可能CRLF）哈希；published_originals是远程下载原字节SHA256，作为公开原文件核验依据。'
    manifest['t5_tokenizer_revision'] = ASSETS['t5'][1]
    manifest['scope'] = 'basics39格/train27格/辅助gpt2 6格；Markdown2设置块；原SyntaxError和IPython shell保留。'
    (lesson / '来源清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    result = {'commit': COMMIT, 'originals': records, 'model_weights_downloaded': False}
    if args.tokenizer_cache:
        assets = []
        for key, (repo, revision, names) in ASSETS.items():
            folder = args.tokenizer_cache / key
            folder.mkdir(parents=True, exist_ok=True)
            for name in names:
                url = 'https://huggingface.co/' + repo + '/resolve/' + revision + '/' + name
                raw = fetch(url)
                (folder / name).write_bytes(raw)
                assets.append({'tokenizer': key, 'repository': repo, 'revision': revision,
                               'file': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        result['tokenizer_assets_only'] = assets
        result['local_cache'] = str(args.tokenizer_cache.resolve())
    out = lesson / '复现/outputs/source_download.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('3 original notebooks + 1 PNG hashes match; optional fixed tokenizer assets only; no weights.')

if __name__ == '__main__':
    main()
