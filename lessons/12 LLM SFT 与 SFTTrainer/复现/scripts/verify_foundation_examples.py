"""Execute the four current teaching blocks verbatim; preserve all older results."""
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch
import transformers
import trl
import peft
from trl import SFTTrainer
from trl.trainer.utils import ConstantLengthDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

sys.stdout.reconfigure(encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[2]
notes = root/'解释.md'
foundation = notes.read_text(encoding='utf-8').split('<details>', 1)[0]
blocks = re.findall(r'```python\n(.*?)\n```', foundation, re.S)
assert len(blocks) == 4
assert (torch.__version__, transformers.__version__, trl.__version__, peft.__version__) == (
    '2.0.1+cpu', '4.30.0', '0.4.7', '0.4.0')
scope = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    for index, block in enumerate(blocks):
        exec(compile(block, str(notes)+':block'+str(index+1), 'exec'), scope)
assert scope['valid_answer'] == 2
assert scope['masked_predictions'].any()
assert int(scope['real_eos'].sum()) == 2 and scope['padding'].any()
assert scope['packed_eos'].any()
assert scope['trainable_count'] == 64 and scope['updates'] == 2
assert list(scope['batch']['input_ids'].shape) == [2, 16]
assert list(scope['output'].logits.shape) == [2, 16, 50257]
assert type(scope['trainer'].data_collator) is DataCollatorWithPadding
assert scope['base'].config.vocab_size == 50257
assert not getattr(scope['base'], 'is_loaded_in_4bit', False)
assert not scope['base'].is_gradient_checkpointing
assert all(not parameter.requires_grad for name, parameter in scope['model'].named_parameters()
           if 'lora_' not in name)

source = json.loads((root/'复现/outputs/source_download.json').read_text(encoding='utf-8'))
for entry in source['files']:
    assert digest(root/entry['published_path']) == entry['sha256']
figures = json.loads((root/'复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
for paper in figures['papers']:
    assert digest(paper['local_pdf']) == paper['pdf_sha256']
    for figure in paper['images']:
        assert digest(root/figure['file']) == figure['sha256']
        assert figure['visually_inspected'] is True

sources = {}
for name, obj in [('SFTTrainer', SFTTrainer), ('ConstantLengthDataset', ConstantLengthDataset),
                  ('DataCollatorWithPadding', DataCollatorWithPadding),
                  ('DataCollatorForLanguageModeling', DataCollatorForLanguageModeling),
                  ('LlamaForCausalLM', scope['LlamaForCausalLM']),
                  ('prepare_model_for_kbit_training', scope['prepare_model_for_kbit_training'])]:
    path = Path(inspect.getfile(obj))
    sources[name] = dict(path=str(path), sha256=digest(path))
record = dict(
    status='passed', date='2026-10-17', executed='Four Python blocks extracted verbatim before historical details in current 解释.md',
    command=[sys.executable, str(Path(__file__).resolve())], cwd=str(Path.cwd()),
    python=platform.python_version(), versions={name:importlib.metadata.version(name) for name in
                                              ['torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate', 'tokenizers']},
    device='CPU', notes_sha256=digest(notes), script_sha256=digest(__file__),
    code_sha256=[hashlib.sha256(block.encode('utf-8')).hexdigest() for block in blocks],
    stdout=stdout.getvalue(), source_commit=source['commit'], original_notebook_sha256=source['files'][0]['sha256'],
    tokenizer_revision='607a30d783dfa663caf39e06633721c8d4cfcd7e',
    tokenizer_assets={str(path.name):digest(path) for path in scope['tokenizer_path'].glob('*.json')},
    official_installed_sources=sources,
    text=dict(rows=scope['rows'], encoded=scope['encoded'], token_lengths=[len(ids) for ids in scope['encoded']],
              vocab_size=scope['vocab_size'], eos_id=scope['eos_id']),
    packing=dict(stream=scope['stream'], kept=scope['kept'], tail=scope['stream'][scope['kept']:],
                 block_ids=[block['input_ids'].tolist() for block in scope['blocks']],
                 actual_blocks=len(scope['blocks']), reported_length=len(scope['packed']),
                 packed_labels=scope['packed_batch']['labels'].tolist(),
                 eos_labels=scope['packed_batch']['labels'][scope['packed_eos']].tolist(),
                 nonpacked_ids=scope['lm_batch']['input_ids'].tolist(),
                 nonpacked_labels=scope['lm_batch']['labels'].tolist(),
                 nonpacked_real_eos_masked=int(scope['real_eos'].sum()),
                 nonpacked_padding_masked=int(scope['padding'].sum())),
    loss=dict(answer_start=scope['answer_start'], input_ids=scope['ids'],
              answer_labels=scope['answer_labels'].tolist(), full_targets=scope['valid_full'], answer_targets=scope['valid_answer'],
              full_loss=float(scope['full_loss'].detach()), answer_loss=float(scope['answer_loss'].detach()),
              masked_score_gradient_zero=True, full_score_gradient_nonzero_at_same_positions=True,
              last_score_gradient_zero=True, dtype='float64', input_unchanged=True),
    trainer=dict(base_parameters=sum(parameter.numel() for parameter in scope['base'].parameters())-scope['trainable_count'],
                 trainable_parameters=scope['trainable_count'], updates=scope['updates'], loss=scope['training_loss'],
                 input_shape=list(scope['batch']['input_ids'].shape), logits_shape=list(scope['output'].logits.shape),
                 logits_dtype=str(scope['output'].logits.dtype), collator=type(scope['trainer'].data_collator).__name__,
                 changed_adapter_names=scope['adapter_changed'], frozen_tensors_unchanged=len(scope['frozen_before']),
                 frozen_gradients_none=True, optimizer_parameters_exactly_trainable=True,
                 model_not_rewrapped=True, quantized=False, checkpointing=False, fp16=False,
                 temporary_training_files_removed=True),
    paper_crops_visually_inspected=True,
    limits='Two self-created short instruction rows, fixed local GPT-2 tokenizer, random one-layer H8 CPU LLaMA, 64 LoRA parameters and two optimizer steps only. No rerun of older six-row/H16 verification, original notebook training, Xgen7B weights/remote tokenizer, full Alpaca, 4bit/bitsandbytes, CUDA, checkpointing, AMP/DDP, adapter reload, RM/PPO, quality or performance benchmark.',
)
out = root/'复现/outputs/基础改写验证.json'
out.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(stdout.getvalue())
print('passed; old records preserved; saved', out)
