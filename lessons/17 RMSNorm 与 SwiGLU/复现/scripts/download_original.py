# -*- coding: utf-8 -*-
"""Save the fixed upstream Git blob without newline conversion."""
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
REL = 'tutorials/llama2_src_02_RMSNorm_SwiGLU.ipynb'
SHA = 'a90ee5e811cf42d3a3f3496f0d6449d7def728f497a57525b8bfe6bfd834a61f'

def main():
    root = Path(__file__).resolve().parents[2]
    data = subprocess.check_output(['git', '-C', 'G:/zzz/Llama 入门教程/代码/personal_chatgpt', 'show', COMMIT + ':' + REL])
    assert hashlib.sha256(data).hexdigest() == SHA
    folder = root / '代码/tutorials'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / Path(REL).name).write_bytes(data)
    p = root / '来源清单.json'
    m = json.loads(p.read_text(encoding='utf-8'))
    m['legacy_local_crlf_sha256'] = '93f2882a152fab153f31675c651b3747f282c81c814a2b817143b64eac655a12'
    m.update(sha256=SHA, bytes=len(data), code_cells=20, download='fixed upstream Git blob; bytes preserved')
    p.write_bytes((json.dumps(m, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print('Saved fixed original:', len(data), SHA)

if __name__ == '__main__':
    main()
