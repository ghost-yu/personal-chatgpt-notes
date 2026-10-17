"""Execute current foundation examples, including real CPU scaler and autocast."""
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
starts = ['import torch\n\nsmall =','import torch\n\np = torch.nn.Parameter','import torch\n\ntorch.manual_seed(7)']
selected = [b for b in blocks if any(b.startswith(prefix) for prefix in starts)]
assert len(selected) == 3
scope = {}
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    for block in selected:
        exec(compile(block,str(notes),'exec'),scope)
actual_stdout = buffer.getvalue()
assert scope['direct'].item() == 0
assert math.isclose(scope['recovered'].item(),1e-8,rel_tol=.01)
assert scope['scaled_grad'] == 256 and scope['unscaled_grad'] == 2
finite = re.search(r'更新后参数/scale: ([^ ]+) ([^\n]+)',actual_stdout)
assert finite is not None
finite_p,finite_scale = map(float,finite.groups())
assert math.isclose(finite_p,.8,abs_tol=1e-7) and finite_scale == 256
assert math.isclose(scope['p'].item(),.8,abs_tol=1e-7)
assert scope['scaler'].is_enabled() and scope['scaler'].get_scale() == 128
assert scope['torch'].equal(scope['p'].detach(),scope['saved_p'])
assert scope['layer'].weight.dtype == scope['x'].dtype == scope['torch'].float32
assert scope['logits'].dtype == scope['torch'].float16
assert scope['classification_loss'].dtype == scope['torch'].float32
assert list(scope['logits'].shape) == [2,2]
assert list(scope['classification_loss'].shape) == []
assert math.isfinite(scope['classification_loss'].item())
image = root/'图片/MixedPrecision原论文_Figure1_单层训练.png'
image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
provenance = json.loads((root/'复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
assert image_hash == provenance['image_sha256']
record = {
    'status':'passed','date':'2026-10-17',
    'notes_sha256':hashlib.sha256(notes.read_bytes()).hexdigest(),
    'executed':'three Python blocks extracted verbatim from current 解释.md',
    'code_sha256':[hashlib.sha256(b.encode('utf-8')).hexdigest() for b in selected],
    'stdout':actual_stdout,'python':platform.python_version(),
    'torch':importlib.metadata.version('torch'),'device':'CPU',
    'small_conversion':{'input':scope['small'].item(),'direct':scope['direct'].item(),'scaled_then_recovered':scope['recovered'].item()},
    'real_scaler':{'enabled':True,'scaled_gradient':256.,'unscaled_gradient':2.,'initial_parameter':1.,
                   'finite_updated_parameter':finite_p,'initial_scale':128.,'after_finite_scale':finite_scale,
                   'after_injected_inf_parameter':scope['p'].item(),'after_injected_inf_scale':128.,
                   'inf_update_skipped':True,'optimizer':'SGD lr0.1','loss':'p squared'},
    'real_autocast':{'device':'cpu','dtype':'float16','parameter_dtype':str(scope['layer'].weight.dtype),
                     'input_dtype':str(scope['x'].dtype),'logits_dtype':str(scope['logits'].dtype),
                     'loss_dtype':str(scope['classification_loss'].dtype),'logits_shape':[2,2],
                     'loss_shape':[],'loss_value':scope['classification_loss'].item(),
                     'forward_only_no_optimizer_update':True},
    'paper_image':image.relative_to(root).as_posix(),'paper_image_sha256':image_hash,
    'paper_image_visually_inspected':True,
    'limits':'No old CNN/meta/training verifier rerun, MNIST download, original large CNN, original CUDA AMP, GPU/3090 memory, performance or quality experiment.'
}
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(actual_stdout)
print('Small-value conversion, real scaler update/skip, and CPU autocast dtypes passed.')
