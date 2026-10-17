"""Execute all four current-note examples verbatim; never overwrite historical runs."""
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import platform
import re
import sys
import warnings
from pathlib import Path

import torch
import transformers
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel

sys.stdout.reconfigure(encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def max_difference(left, right):
    return max(float((left[name]-right[name]).abs().max()) for name in left)


root = Path(__file__).resolve().parents[2]
notes = root/'解释.md'
blocks = re.findall(r'```python\n(.*?)\n```', notes.read_text(encoding='utf-8').split('<details>', 1)[0], re.S)
assert len(blocks) == 4
assert torch.__version__ == '2.0.1+cpu' and transformers.__version__ == '4.30.0'
scope = {}
stdout = io.StringIO()
with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(stdout):
    warnings.simplefilter('always')
    for index, block in enumerate(blocks):
        exec(compile(block, str(notes)+':block'+str(index+1), 'exec'), scope)
for run in scope['scalar_runs'].values():
    assert run['prediction'] == 1 and run['loss'] == 0.5 and run['w_unchanged'] == 2
assert scope['sequential_runs']['ordinary']['calls'] == [[True], [True], [True], [True]]
assert sum(parameter.numel() for parameter in scope['template'].parameters()) == 24
plain, checked = scope['sequential_runs']['ordinary'], scope['sequential_runs']['checkpoint']
assert max_difference(plain['parameter_grads'], checked['parameter_grads']) == 0
rng = scope['rng_runs']
assert len(rng['ordinary']['replay']) == 1
assert len(rng['preserve']['replay']) == len(rng['no_preserve']['replay']) == 2
assert torch.equal(*rng['preserve']['replay'])
assert not torch.equal(*rng['no_preserve']['replay'])
bert = scope['bert_runs']
assert bert[False]['loss'] == bert[True]['loss']
assert torch.equal(bert[False]['logits'], bert[True]['logits'])
assert list(bert[True]['logits'].shape) == [2, 2]
assert scope['batch_ids'].dtype == scope['batch_labels'].dtype == torch.int64
assert max_difference(bert[False]['state'], bert[True]['state']) == 0
assert max_difference(bert[False]['gradients'], bert[True]['gradients']) == 0
assert bert[True]['parameters'] == bert[False]['parameters']

source = json.loads((root/'复现/outputs/source_download.json').read_text(encoding='utf-8'))
for entry in source['files']:
    assert digest(root/entry['published_path']) == entry['sha256']
figures = json.loads((root/'复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
assert figures['version'] == 'arXiv:1604.06174v2' and len(figures['images']) == 4
assert digest(figures['local_pdf']) == figures['pdf_sha256']
assert digest(root/'复现/scripts/extract_paper_images.py') == figures['script_sha256']
for figure in figures['images']:
    assert digest(root/figure['file']) == figure['sha256'] and figure['visually_inspected'] is True
sources = {}
for name, obj in [('checkpoint', checkpoint), ('checkpoint_sequential', checkpoint_sequential),
                  ('BertEncoder', BertEncoder), ('BertPreTrainedModel', BertPreTrainedModel),
                  ('BertForSequenceClassification', scope['BertForSequenceClassification'])]:
    path = Path(inspect.getfile(obj))
    sources[name] = dict(path=str(path), sha256=digest(path))

record = dict(
    status='passed', date='2026-10-17', executed='Four Python blocks extracted verbatim before historical details in current 解释.md',
    command=[sys.executable, str(Path(__file__).resolve())], cwd=str(Path.cwd()), python=platform.python_version(),
    versions={name:importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy']}, device='CPU',
    notes_sha256=digest(notes), script_sha256=digest(__file__),
    code_sha256=[hashlib.sha256(block.encode('utf-8')).hexdigest() for block in blocks],
    stdout=stdout.getvalue(), warnings=[str(warning.message) for warning in caught],
    official_installed_sources=sources, source_commit=source['commit'],
    original_notebook_sha256=source['files'][0]['sha256'],
    scalar=dict(runs=scope['scalar_runs'], dtype='float64', shape=[], optimizer_step=False),
    sequential=dict(calls={mode:run['calls'] for mode,run in scope['sequential_runs'].items()},
                    input_shape=[1, 2], dtype='float64', parameters=24,
                    output=checked['output'].tolist(), input_gradient=checked['input_grad'].tolist(),
                    output_exact=True, input_gradient_max_difference=0.0,
                    parameter_gradient_max_difference=max_difference(plain['parameter_grads'],checked['parameter_grads']),
                    total_forward_layer_calls=[4, 6], segments=2, last_segment_recomputed=False, optimizer_step=False),
    rng={mode:dict(first_output=run['first_output'].tolist(), weight_gradient=run['weight_grad'].tolist(),
                   replay_outputs=[item.tolist() for item in run['replay']],
                   gradient_max_difference=float((run['weight_grad']-rng['ordinary']['weight_grad']).abs().max()))
         for mode,run in rng.items()},
    bert=dict(input_ids=scope['batch_ids'].tolist(), labels=scope['batch_labels'].tolist(),
              input_shape=[2,4], logits_shape=[2,2], logits_dtype=str(bert[True]['logits'].dtype),
              parameters=bert[True]['parameters'], loss=bert[True]['loss'],
              calls_off=bert[False]['calls'], calls_on=bert[True]['calls'],
              logits_exact=True, gradient_max_difference=max_difference(bert[False]['gradients'],bert[True]['gradients']),
              post_update_state_max_difference=max_difference(bert[False]['state'],bert[True]['state']),
              all_parameter_gradients_present=True, parameters_unchanged_until_backward_completed=True,
              optimizer_steps_per_mode=1, optimizer='SGD', learning_rate=0.1,
              parameter_update_observed_each_mode=True, dropout=0.0, model_weights_downloaded=False),
    paper_crops_visually_inspected=True,
    storage_arithmetic_only=dict(hidden_fp32_bytes=4*512*1024*4, hidden_mib=8,
                                 explicit_attention_fp32_bytes=4*16*512*512*4, explicit_attention_mib=64,
                                 measured_peak_memory=False),
    limits='New scalar, 4-block H2 sequential, 8-element dropout replay and random 2-layer H8/V32 BERT CPU examples only. No rerun of older four Trainer groups/H16 models, saved-tensor byte counts, frozen-embedding/evaluation negative cases, original notebook/CUDA/NVML, BERT-large weights, AMP/DDP, non-reentrant migration, CNN/LSTM paper experiments or memory/speed/quality benchmarks. Diagnostic snapshots consume additional memory and are not memory measurements.',
)
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(stdout.getvalue())
print('passed; historical evidence preserved; saved new foundation verification')
