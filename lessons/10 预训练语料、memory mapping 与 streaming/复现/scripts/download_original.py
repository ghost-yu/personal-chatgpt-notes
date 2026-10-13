# -*- coding: utf-8 -*-
"""下载固定原notebook和4张必要图片，逐一核对原字节SHA256。"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
FILES = {
    'tutorials/pretrain_corpus_大模型预训练语料.ipynb': 'a72dab4de6111c0a41674dd41a9efd7593cf3498a282a60cd1107352517f72d3',
    'imgs/en-dataset.png': 'd2322618c13eb85cb8a8e688e84221310675967737f306cb880d804e538fadce',
    'imgs/ch-dataset.png': 'dc8cdeedef63df0e5dd9aa6cf04dbb845a33409bf2ddb85218a9a637209a8d71',
    'imgs/c4.png': '71ff6d783b5e623dc1832da24a2466c27d790db7c1e2166a15bec5975827461c',
    'imgs/pile.png': 'c0dc440e1e9b2af5d149563b9360f16c4436780a5fc6ad6136f6507e5d645fab',
}
GPT2_REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
TOKENIZER_FILES = {
    'config.json': '0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
    'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
}

def prepare_tokenizer(folder):
    """仅下载3个固定分词配置/词表文件；已存在且hash正确时直接复用，不下载权重。"""
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in TOKENIZER_FILES.items():
        target = folder / name
        url = 'https://huggingface.co/gpt2/resolve/' + GPT2_REVISION + '/' + name
        reused = target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected
        if not reused:
            with urlopen(Request(url, headers={'User-Agent': 'personal-chatgpt-notes'}), timeout=45) as response:
                raw = response.read()
            assert hashlib.sha256(raw).hexdigest() == expected, name
            target.write_bytes(raw)
        records.append({'file': name, 'url': url, 'sha256': expected,
                        'bytes': target.stat().st_size, 'existing_asset_reused': reused})
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', type=Path, help='可选：准备教学验证所用固定GPT2资产，建议放仓库外')
    args = parser.parse_args()
    lesson = Path(__file__).resolve().parents[2]
    manifest = json.loads((lesson / '来源清单.json').read_text(encoding='utf-8'))
    records = []
    for rel, expected in FILES.items():
        url = 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/' + COMMIT + '/' + quote(rel, safe='/')
        with urlopen(Request(url, headers={'User-Agent': 'personal-chatgpt-notes'}), timeout=45) as response:
            raw = response.read()
        assert hashlib.sha256(raw).hexdigest() == expected, rel
        target = lesson / '代码' / rel
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        records.append({'source': rel, 'url': url, 'bytes': len(raw), 'sha256': expected,
                        'published_path': '代码/' + rel, 'original_bytes_preserved': True})
    manifest['published_originals'] = records
    manifest['legacy_hash_scope'] = '旧sha256为本地Git工作副本；published_originals核对远程下载原字节。'
    manifest['scope'] = '原13格/6代码格/4图完整说明；缺失的数据演示独立标为教学补充。'
    (lesson / '来源清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    record = {
        'commit': COMMIT, 'files': records, 'corpus_downloaded': False, 'model_weights_downloaded': False,
    }
    if args.tokenizer_dir is not None:
        record['GPT2_tokenizer_revision'] = GPT2_REVISION
        record['GPT2_tokenizer_assets'] = prepare_tokenizer(args.tokenizer_dir)
    (lesson / '复现/outputs/source_download.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Original13 cells + 4 PNG images downloaded unchanged; all fixed SHA256 hashes match.')

if __name__ == '__main__':
    main()
