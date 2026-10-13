"""下载第02讲固定提交的原notebook与六张图，核对已有来源清单的哈希。"""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    lesson = Path(__file__).resolve().parents[2]
    manifest = json.loads((lesson/'来源清单.json').read_text(encoding='utf-8-sig'))
    commit = manifest['commit']
    base = 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+commit+'/'
    entries = [{'path':manifest['notebook'], 'sha256':manifest['upstream_sha256']}] + manifest['images']
    records = []
    for entry in entries:
        relative = entry['path']
        with urlopen(Request(base+relative, headers={'User-Agent':'personal-chatgpt-notes'}), timeout=60) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == entry['sha256'], (relative, 'source hash mismatch')
        if relative.endswith('.ipynb'):
            assert len(json.loads(data.decode('utf-8'))['cells']) == 16
        else:
            assert data.startswith(b'\x89PNG\r\n\x1a\n')
        target = lesson/'代码'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({'path':relative, 'url':base+relative, 'bytes':len(data), 'sha256':digest})
    record = {'upstream_commit':commit, 'downloaded_files':records, 'manifest_hashes_matched':True,
              'original_bytes_unmodified':True, 'notebook_executed_by_download':False}
    (lesson/'复现/outputs/source_download.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'files_downloaded':len(records), 'manifest_hashes_matched':True}))


if __name__ == '__main__':
    main()
