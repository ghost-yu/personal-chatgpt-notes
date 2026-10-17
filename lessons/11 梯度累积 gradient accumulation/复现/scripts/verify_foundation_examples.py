"""Execute current scalar accumulation examples verbatim; CPU SGD only."""
import contextlib
import hashlib
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
foundation = notes.read_text(encoding='utf-8').split('<details>',1)[0]
blocks = re.findall(r'```python\n(.*?)\n```',foundation,re.S)
assert len(blocks) == 3
scope = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    for block in blocks:
        exec(compile(block,str(notes),'exec'),scope)
assert scope['first_grad'] == 2 and scope['second_grad'] == 4
assert scope['before_step'] == 1
assert math.isclose(scope['single_updated'],.8,abs_tol=1e-12)
assert scope['gradient_after_step'] == 2 and scope['w'].grad is None
assert scope['full_loss'].item() == 7.5
assert scope['full_gradient'] == scope['micro_gradient'] == 15
assert scope['gradient_trace'] == [.5,2.5,7,15]
assert scope['parameter_trace'] == [1,1,1,1]
assert math.isclose(scope['full_w'].item(),-.5,abs_tol=1e-12)
assert math.isclose(scope['micro_w'].item(),-.5,abs_tol=1e-12)
assert scope['wrong_gradient'] == 60
assert math.isclose(scope['wrong_w'].item(),-5.,abs_tol=1e-12)
for key, expected in [('tail_correct',(5.,.5)),('tail_fixed_four',(2.5,.75)),
                      ('unequal_correct',(14.,-.4)),('unequal_mean_of_means',(18.5,-.85))]:
    assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(scope[key],expected))
torch = scope['torch']
assert torch.__version__ == '2.5.1+cpu'
assert scope['w'].device.type == 'cpu'
assert scope['xs'].dtype == torch.float64 and list(scope['xs'].shape) == [4]
assert list(scope['full_w'].shape) == []
source = json.loads((root/'复现/outputs/source_download.json').read_text(encoding='utf-8'))
for entry in source['files']:
    assert hashlib.sha256((root/entry['published_path']).read_bytes()).hexdigest() == entry['sha256']
references = json.loads((root/'复现/outputs/原图与引用来源.json').read_text(encoding='utf-8'))
assert references['course_image']['sha256'] == source['files'][1]['sha256']
assert references['paper_figure_status'] == 'No designated original algorithm paper or verified paper source for course PNG found'
assert 'accumulates gradients in the leaves' in torch.Tensor.backward.__doc__
record = {
    'status':'passed','date':'2026-10-17',
    'notes_sha256':hashlib.sha256(notes.read_bytes()).hexdigest(),
    'executed':'Three Python blocks extracted verbatim before historical details in current 解释.md',
    'code_sha256':[hashlib.sha256(b.encode('utf-8')).hexdigest() for b in blocks],
    'command':[sys.executable,str(Path(__file__).resolve())],'working_directory':str(Path.cwd()),
    'python':platform.python_version(),'torch':torch.__version__,'device':'CPU','dtype':'float64',
    'stdout':stdout.getvalue(),
    'scalar_buffer':{'first_backward':2.,'second_new_graph_backward':4.,'parameter_before_step':1.,
        'one_gradient_updated_parameter':scope['single_updated'],'gradient_still_after_step':2.,
        'final_zero_grad_none':True,'retain_graph_used':False},
    'complete_window':{'inputs':scope['xs'].tolist(),'targets':'all zero',
        'mean_loss':scope['full_loss'].item(),'full_gradient':scope['full_gradient'],
        'micro_gradient':scope['micro_gradient'],'gradient_trace':scope['gradient_trace'],
        'parameter_trace':scope['parameter_trace'],'full_updated':scope['full_w'].item(),
        'micro_updated':scope['micro_w'].item(),'missing_division_gradient':scope['wrong_gradient'],
        'missing_division_updated':scope['wrong_w'].item()},
    'tail_two_single_sample_microbatches':{'correct':scope['tail_correct'],
        'fixed_mean_divisor_4':scope['tail_fixed_four']},
    'unequal_two_plus_one_samples':{'inputs':[[1.,2.],[4.]],'correct':scope['unequal_correct'],
        'mean_of_means':scope['unequal_mean_of_means']},
    'tuple_order':'gradient before update, parameter after one SGD lr0.1 update',
    'course_image_sha256':references['course_image']['sha256'],'course_image_visually_inspected':True,
    'source_commit':source['commit'],
    'limits':'Independent one-parameter CPU float64 squared-error/SGD examples only. No old notebook/Markdown-loop/BERT/Trainer/NVML verifier rerun, model weights, CUDA, two-GPU, AMP/DDP, token loss, model quality or memory/speed benchmark. No original algorithm paper figure claimed.'
}
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(stdout.getvalue())
print('Scalar buffer, full-window equivalence, missing divisor, tail and unequal-batch checks passed.')
