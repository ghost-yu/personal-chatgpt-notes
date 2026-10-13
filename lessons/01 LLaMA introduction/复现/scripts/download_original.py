"""下载固定提交的原 notebook 和配套图片，不执行 notebook。"""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
BASE = 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/' + COMMIT + '/'


def main():
    lesson = Path(__file__).resolve().parents[2]
    records = []
    for relative in ['tutorials/01_llama_intro.ipynb', 'imgs/llama_weights.png']:
        url = BASE + relative
        with urlopen(Request(url, headers={'User-Agent': 'personal-chatgpt-notes'}), timeout=60) as response:
            data = response.read()
        if relative.endswith('.ipynb'):
            notebook = json.loads(data.decode('utf-8'))
            assert len(notebook['cells']) == 9
        else:
            assert data.startswith(b'\x89PNG\r\n\x1a\n')
        target = lesson / '代码' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({'path': relative, 'url': url, 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
    report = {'upstream_commit': COMMIT, 'downloaded_files': records,
              'original_bytes_unmodified': True, 'notebook_executed_by_download': False}
    target = lesson / '复现' / 'outputs' / 'source_download.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
