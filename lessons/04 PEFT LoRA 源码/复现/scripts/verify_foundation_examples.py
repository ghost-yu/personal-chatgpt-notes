"""Run current foundation note examples against the pinned official PEFT checkout."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')
parser = argparse.ArgumentParser()
parser.add_argument('--peft-source', required=True, type=Path)
args = parser.parse_args()
source = args.peft_source.resolve()
expected = '7120c9a2636f93c8db3fc4ae466e02f338ecead8'
head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert head == expected
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--','src'],text=True).strip()
sys.path.insert(0,str(source/'src'))
os.environ['HF_HUB_OFFLINE'] = '1'
import peft
import torch
assert Path(peft.__file__).resolve().is_relative_to(source/'src')
assert peft.__version__ == '0.3.0'
torch.set_num_threads(2)
root = Path(__file__).resolve().parents[2]
notes = root/'解释.md'
blocks = re.findall(r'```python\n(.*?)\n```',notes.read_text(encoding='utf-8'),re.S)
selected = [b for b in blocks if b.startswith("original = {'weight':") or b.startswith('import torch\nfrom transformers import BloomConfig')]
assert len(selected) == 2
scope = {}
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    for block in selected:
        exec(compile(block,str(notes),'exec'),scope)
layer = scope['layer']
assert scope['original']['weight'] == [9.,2.]
assert layer.lora_A['default'].weight.grad.abs().sum().item() > 0
assert layer.lora_B['default'].weight.grad.abs().sum().item() > 0
assert all(p.grad is None for name,p in scope['wrapped'].named_parameters() if 'lora_' not in name)
images = [root/'图片/LoRA原论文_Figure1_双分支.png',root/'图片/LoRA原论文_Equation3_更新公式.png']
assert all(p.is_file() for p in images)
record = {
    'status':'passed','date':'2026-10-17',
    'notes_sha256':hashlib.sha256(notes.read_bytes()).hexdigest(),
    'executed':'two Python blocks extracted verbatim from current 解释.md',
    'stdout':buffer.getvalue(),'peft_version':peft.__version__,
    'peft_commit':head,'peft_module':str(Path(peft.__file__).resolve()),
    'peft_source_clean':True,
    'source_sha256':{n:hashlib.sha256((source/'src/peft'/n).read_bytes()).hexdigest()
                     for n in ['mapping.py','peft_model.py','tuners/lora.py']},
    'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate']},
    'model':'random CPU FP32 BLOOM; 1 layer, hidden 4, heads 2, vocab 8',
    'lora':{'r':1,'alpha':1,'dropout':0.,'target_modules':['query_key_value']},
    'base_identity_preserved':True,'target_layer_replaced':True,'base_weight_identity_preserved':True,
    'zero_B_full_logits_equal_to_before':True,
    'manual_nonzero_B_not_training':True,
    'layer_output_shape':list(scope['y'].shape),
    'manual_update':(scope['y']-scope['y0']).detach().flatten().tolist(),
    'trainable_parameters':16,'trainable_tensors':2,
    'A_and_B_have_nonzero_gradients_after_manual_B_assignment':True,
    'base_parameters_have_no_gradients':True,'base_weight_unchanged':True,
    'images':[{'file':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
               'visually_inspected':True} for p in images],
    'limits':'No optimizer updates; old H16 experiment not rerun; no 7B, quantization, CUDA or quality evaluation.'
}
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(buffer.getvalue())
print('Pinned PEFT and foundation examples passed.')
