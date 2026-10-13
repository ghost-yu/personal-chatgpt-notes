"""下载固定提交的第04讲原notebook，核对来源清单，不运行模型。"""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    lesson = Path(__file__).resolve().parents[2]
    manifest = json.loads((lesson/'来源清单.json').read_text(encoding='utf-8-sig'))
    with urlopen(Request(manifest['source_url'], headers={'User-Agent':'personal-chatgpt-notes'}), timeout=60) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == manifest['upstream_sha256'], 'source hash mismatch'
    notebook = json.loads(raw.decode('utf-8'))
    assert len(notebook['cells']) == 18
    assert sum(c['cell_type']=='code' for c in notebook['cells']) == 8
    target = lesson/'代码'/manifest['notebook']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    record = {'upstream_commit':manifest['commit'], 'downloaded_files':[
               {'path':manifest['notebook'], 'url':manifest['source_url'], 'bytes':len(raw), 'sha256':digest}],
              'manifest_hash_matched':True, 'original_bytes_unmodified':True, 'notebook_executed_by_download':False}
    (lesson/'复现/outputs/source_download.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'downloaded_bytes':len(raw), 'manifest_hash_matched':True}))


if __name__=='__main__':
    main()
