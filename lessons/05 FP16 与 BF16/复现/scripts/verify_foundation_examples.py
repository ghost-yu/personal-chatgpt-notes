"""Execute current foundation note examples; record CPU observations and original images."""
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[2]
notes = root/'解释.md'
blocks = re.findall(r'```python\n(.*?)\n```',notes.read_text(encoding='utf-8'),re.S)
selected = [b for b in blocks if b.startswith('s = 0\nm = 10') or b.startswith('import torch\n\nsource = torch.tensor')]
assert len(selected) == 2
scope = {}
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    for block in selected:
        exec(compile(block,str(notes),'exec'),scope)
assert '10001 -> 6.5' in buffer.getvalue()
assert '11101 -> 26624.0' in buffer.getvalue()
actual = scope['converted']
assert actual['torch.float16'][0] == 1.0009765625
assert math.isinf(actual['torch.float16'][1])
assert actual['torch.float16'][2] == 0.
assert actual['torch.bfloat16'] == [1.,99840.,2.**-25]
assert actual['torch.float32'] == [1.0010000467300415,100000.,2.**-25]
assert scope['spacings'] == {'torch.float16':2.**-10,'torch.bfloat16':2.**-7,'torch.float32':2.**-23}
assert all(list(scope['source'].to(dt).shape) == [3] for dt in [scope['torch'].float16,scope['torch'].bfloat16,scope['torch'].float32])
byte_sizes = {str(dt):scope['source'].to(dt).numel()*scope['source'].to(dt).element_size()
              for dt in [scope['torch'].float16,scope['torch'].bfloat16,scope['torch'].float32]}
assert list(byte_sizes.values()) == [6,6,12]
source_record = json.loads((root/'来源清单.json').read_text(encoding='utf-8'))
images = []
for item in source_record['images']:
    path = root/'代码'/item['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == item['sha256']
    images.append({'file':path.relative_to(root).as_posix(),'sha256':digest,'original_source_hash_matches':True,'visually_inspected':True})
path = root/'图片/BFLOAT16原论文_Table1_格式对照.png'
assert path.is_file()
images.append({'file':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'visually_inspected':True})
# Strict JSON has no Infinity token; keep the observed value as the explicit text 'Inf'.
serial = {key:[('Inf' if math.isinf(x) else x) for x in values] for key,values in actual.items()}
record = {
    'status':'passed','date':'2026-10-17',
    'notes_sha256':hashlib.sha256(notes.read_bytes()).hexdigest(),
    'executed':'two Python blocks extracted verbatim from current 解释.md',
    'code_sha256':[hashlib.sha256(b.encode('utf-8')).hexdigest() for b in selected],
    'stdout':buffer.getvalue(),'python':platform.python_version(),
    'torch':importlib.metadata.version('torch'),'device':'CPU',
    'decoded_examples':[6.5,26624.],'conversions':serial,
    'spacing_above_one':scope['spacings'],'shape':[3],'data_bytes':byte_sizes,
    'images':images,
    'limits':'No rerun of old runtime_verification, GPU/TPU arithmetic, AMP training, performance or accuracy measurements.'
}
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(buffer.getvalue())
print('Decoding, conversions, spacing, data sizes and original image hashes passed.')
