"""下载固定来源的第03讲原notebook，核对已登记的SHA-256，不执行代码。"""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    lesson = Path(__file__).resolve().parents[2]
    manifest = json.loads((lesson/'来源清单.json').read_text(encoding='utf-8-sig'))
    url = manifest['source_url']
    with urlopen(Request(url, headers={'User-Agent':'personal-chatgpt-notes'}), timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == manifest['upstream_sha256'], 'upstream hash mismatch'
    notebook = json.loads(data.decode('utf-8'))
    assert len(notebook['cells']) == 47
    assert sum(c['cell_type']=='code' for c in notebook['cells']) == 37
    target = lesson/'代码'/manifest['notebook']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    report = {'upstream_commit':manifest['commit'], 'downloaded_files':[
              {'path':manifest['notebook'], 'url':url, 'bytes':len(data), 'sha256':digest}],
              'manifest_hash_matched':True, 'original_bytes_unmodified':True,
              'notebook_executed_by_download':False}
    (lesson/'复现/outputs/source_download.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'downloaded_bytes':len(data), 'manifest_hash_matched':True}))


if __name__ == '__main__':
    main()
