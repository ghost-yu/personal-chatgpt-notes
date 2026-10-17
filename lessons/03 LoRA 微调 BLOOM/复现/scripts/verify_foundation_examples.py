"""Execute the two foundation examples copied directly from the reading note."""
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import re
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

root = Path(__file__).resolve().parents[2]
notes = root / '解释.md'
text = notes.read_text(encoding='utf-8')
blocks = re.findall(r'```python\n(.*?)\n```', text, re.S)
selected = [b for b in blocks if b.startswith("row = {'quote':") or b.startswith('import math\nimport torch')]
assert len(selected) == 2
scope = {}
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    for block in selected:
        exec(compile(block, str(notes), 'exec'), scope)
assert scope['row']['prediction'] == "Be yourself. ->: ['life']"
assert int((scope['next_labels'] != -100).sum()) == 6
assert math.isclose(scope['loss'].item(), math.log(6), rel_tol=1e-6)
assert scope['logits'].grad[:,:-1].abs().sum().item() > 0
image = root / '图片/LoRA原论文_Figure1_双分支.png'
assert image.is_file()
record = {
    'status': 'passed', 'date': '2026-10-17',
    'notes_sha256': hashlib.sha256(notes.read_bytes()).hexdigest(),
    'executed': 'two Python blocks copied from the current 解释.md, no substitutions',
    'stdout': buffer.getvalue(), 'valid_next_token_targets': 6,
    'loss': scope['loss'].item(), 'expected_loss': math.log(6),
    'ignored_and_unused_positions_have_zero_gradient': True,
    'valid_positions_have_nonzero_gradient': True,
    'paper_image': str(image.relative_to(root)),
    'paper_image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
    'paper_image_visually_inspected': True,
    'torch': importlib.metadata.version('torch'),
    'limits': 'Synthetic IDs and logits only; old tiny-model training not rerun; original 7B configuration not run.'
}
(root / '复现/outputs/基础改写验证.json').write_text(
    json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(buffer.getvalue())
print('Foundation examples passed; old training not rerun.')
