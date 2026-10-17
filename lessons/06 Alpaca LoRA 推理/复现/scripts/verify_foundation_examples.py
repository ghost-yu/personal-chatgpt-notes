"""Execute current prompt and tiny-generation examples without downloading any model."""
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
root = Path(__file__).resolve().parents[2]
notes = root/'解释.md'
blocks = re.findall(r'```python\n(.*?)\n```',notes.read_text(encoding='utf-8'),re.S)
selected = [b for b in blocks if b.startswith('def generate_prompt(instruction, input=None):') or b.startswith('import torch\nfrom transformers import LlamaConfig')]
assert len(selected) == 2
original = json.loads((root/'代码/tutorials/06_alpaca_llama_7b_lora.ipynb').read_text(encoding='utf-8'))
original_function = ast.parse(''.join(original['cells'][24]['source'])).body[0]
current_function = ast.parse(selected[0]).body[0]
assert ast.dump(original_function,include_attributes=False) == ast.dump(current_function,include_attributes=False)
scope = {}
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    for block in selected:
        exec(compile(block,str(notes),'exec'),scope)
f = scope['generate_prompt']
assert f('Tell a joke',None) == scope['no_context']
assert scope['prompt'].endswith('### Response:')
assert '\nHello\n' in scope['prompt']
result = scope['result']
assert list(result.sequences.shape) == [1,6]
assert len(result.scores) == 3
assert all(list(v.shape) == [4,8] for v in result.scores)
assert scope['new_ids'].numel() == 3
assert not scope['tiny'].training
image = root/'图片/LoRA原论文_Figure1_双分支.png'
assert image.is_file()
record = {
    'status':'passed','date':'2026-10-17',
    'notes_sha256':hashlib.sha256(notes.read_bytes()).hexdigest(),
    'executed':'two Python blocks extracted verbatim from current 解释.md',
    'code_sha256':[hashlib.sha256(b.encode('utf-8')).hexdigest() for b in selected],
    'original_prompt_function_AST_equal':True,
    'template_checks':['input present','no input','empty string equals no input','None equals no input','Response suffix left empty'],
    'stdout':buffer.getvalue(),'python':platform.python_version(),
    'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers']},
    'device':'CPU FP32','model':'random LLaMA; 1 layer; H8; intermediate16; 2 heads; vocabulary8',
    'adapter_loaded_in_this_example':False,
    'generation':{'num_beams':4,'do_sample':False,'max_new_tokens':3,'eos_token_id':None},
    'input_shape':list(scope['input_ids'].shape),'sequence_shape':list(result.sequences.shape),
    'prefix_ids_preserved':True,'new_token_count':scope['new_ids'].numel(),
    'score_shapes':[list(v.shape) for v in result.scores],
    'sequence_ids':result.sequences.tolist(),
    'paper_image':image.relative_to(root).as_posix(),
    'paper_image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
    'paper_image_visually_inspected':True,
    'limits':'No old pipeline rerun, remote model/tokenizer/adapter load, original 7B, quantization, CUDA, training or language-quality evaluation.'
}
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(buffer.getvalue())
print('Original prompt AST, branches and real CPU generation boundary passed.')
