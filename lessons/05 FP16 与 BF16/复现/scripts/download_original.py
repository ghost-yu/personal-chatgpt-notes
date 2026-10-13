# -*- coding: utf-8 -*-
"""下载固定版本的原notebook和两张图，按预先核对的哈希校验，原样保存。"""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
FILES = {
    'tutorials/05_float16_bf16.ipynb': '96346db1e5fde82fb4f66e74e2ba58a157236275dbbf8033665aef777dd2a712',
    'imgs/float16.png': '31e887c939136a581f40e485e7df99e67d504033e2733bc759d01762a268dd80',
    'imgs/bf-float.png': '9d2b2377fda2759a7835aa0eb786bf6b9aef6bcc362576a87d01f328c4c53b42',
}


def main():
    lesson = Path(__file__).resolve().parents[2]
    records = []
    for relative, expected in FILES.items():
        url = 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+COMMIT+'/'+relative
        with urlopen(Request(url, headers={'User-Agent': 'personal-chatgpt-notes'}), timeout=60) as response:
            raw = response.read()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == expected, relative+' source changed'
        target = lesson/'代码'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        records.append({'path': relative, 'url': url, 'bytes': len(raw), 'sha256': digest})
    original = lesson/'代码/tutorials/05_float16_bf16.ipynb'
    nb = json.loads(original.read_text(encoding='utf-8'))
    assert len(nb['cells']) == 27
    assert [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code'] == [0, 5, 16, 17, 18, 19, 20, 26]
    manifest = {
        'repository': 'https://github.com/chunhuizhang/personal_chatgpt', 'commit': COMMIT,
        'notebook': 'tutorials/05_float16_bf16.ipynb', 'source_url': records[0]['url'],
        'upstream_sha256': FILES['tutorials/05_float16_bf16.ipynb'],
        'published_original_path': '代码/tutorials/05_float16_bf16.ipynb',
        'published_original_sha256': FILES['tutorials/05_float16_bf16.ipynb'],
        'all_cells': list(range(27)), 'python_code_cells': [0, 5, 16, 17, 18, 19, 20, 26],
        'ipython_source_inspection_cells': [20], 'images': records[1:],
        'original_environment': nb['metadata']['language_info'],
        'scope': '全部单元、公式、图片与CUDA查询；历史GPU输出不是当前CPU验证结果',
    }
    (lesson/'来源清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    outputs = lesson/'复现/outputs'
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs/'source_download.json').write_text(json.dumps({
        'upstream_commit': COMMIT, 'downloaded_files': records,
        'hashes_matched': True, 'original_bytes_unmodified': True,
    }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Downloaded 27-cell original and 2 PNGs; all fixed hashes matched.')


if __name__ == '__main__':
    main()
