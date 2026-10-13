# -*- coding: utf-8 -*-
"""全部原单元与AST审计；真实CPU autograd、原循环与小型BERT Trainer验证。"""
import ast
import contextlib
import difflib
import gc
import hashlib
import importlib
import io
import json
import os
import platform
import re
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
import nbformat
import numpy as np
import torch
import pynvml
from PIL import Image as PILImage
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainerCallback

CODE_INDICES = [0, 2, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21, 23, 26]

def tree(source):
    return ast.dump(ast.parse(source), include_attributes=False)

def inserted_comments_only(old, new):
    a, b = old.splitlines(True), new.splitlines(True)
    for op, u, v, x, y in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != 'equal':
            assert op == 'insert' and all(line.lstrip().startswith('# ') for line in b[x:y]), (op, a[u:v], b[x:y])

def loop(source):
    return re.search(r'```\n(.*?)\n```', source, re.S).group(1)

def audit(lesson):
    folder = lesson / '代码/tutorials'
    original = json.loads((folder / 'gradient_accumulation.ipynb').read_text(encoding='utf-8'))
    manifest = json.loads((lesson / '来源清单.json').read_text(encoding='utf-8'))
    for r in manifest['published_originals']:
        assert hashlib.sha256((lesson / r['published_path']).read_bytes()).hexdigest() == r['sha256']
    assert len(original['cells']) == 28
    assert [i for i,c in enumerate(original['cells']) if c['cell_type'] == 'code'] == CODE_INDICES
    nbformat.validate(nbformat.from_dict(original))
    reports = []
    commented = None
    for name in ('原代码_带说明.ipynb', '代码注释.ipynb'):
        nb = json.loads((folder / name).read_text(encoding='utf-8'))
        nbformat.validate(nbformat.from_dict(nb))
        assert len(nb['cells']) == 57
        assert {k:v for k,v in nb.items() if k!='cells'} == {k:v for k,v in original.items() if k!='cells'}
        for i,c in enumerate(original['cells']):
            note, keep = nb['cells'][1+2*i:3+2*i]
            assert '## 单元 {}（'.format(i) in ''.join(note['source'])
            assert {k:v for k,v in c.items() if k!='source'} == {k:v for k,v in keep.items() if k!='source'}
            old, new = ''.join(c['source']), ''.join(keep['source'])
            if name == '代码注释.ipynb' and (c['cell_type']=='code' or i in (5,7)):
                inserted_comments_only(old, new)
                assert tree(loop(old) if i in (5,7) else old) == tree(loop(new) if i in (5,7) else new)
            else: assert c['source'] == keep['source']
        reports.append({'file':name, 'sha256':hashlib.sha256((folder/name).read_bytes()).hexdigest(),
                        'all28_original_metadata_outputs_preserved':True, 'schema_valid':True,
                        'all15_code_AST_equal':True, 'Markdown5_7_AST_equal':True})
        if name=='代码注释.ipynb': commented = nb['cells'][2::2]
    py = folder / 'gradient_accumulation_代码注释.py'
    assert tree(py.read_text(encoding='utf-8')) == tree('\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type']=='code'))
    doc=(lesson/'代码单元解释.md').read_text(encoding='utf-8')
    assert [int(n) for n in re.findall(r'^## 单元 (\d+)（',doc,re.M)] == list(range(28))
    return original['cells'], commented, {'notes28_complete':True, 'code15_AST_equal':True,
        'Markdown5_7_comments_and_AST':True, 'schemas_valid':3, 'original_hashes_match':True,
        'reading_py_AST_equal':True, 'reading_py_sha256':hashlib.sha256(py.read_bytes()).hexdigest(), 'variants':reports}

def exec_cell(source, context):
    node = ast.parse(source)
    last = node.body.pop().value if node.body and isinstance(node.body[-1], ast.Expr) else None
    exec(compile(node, '<course-cell>', 'exec'), context)
    return eval(compile(ast.Expression(last), '<course-result>', 'eval'), context) if last is not None else None

def run_original_cells(cells):
    context = {}; reports = []
    np.random.seed(1234)
    for i in CODE_INDICES:
        if i in (18,19,23,26):
            reports.append({'cell':i,'status':'separate_scope' if i in (23,26) else 'not_run',
                            'reason':{18:'Do not set teacher localhost proxy',19:'No BERT-large download or CUDA build',
                                      23:'Entire source executed with controlled CPU context below',
                                      26:'Three explicit CPU teaching AST substitutions below'}[i]})
            continue
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            result=exec_cell(''.join(cells[i]['source']),context)
        entry={'cell':i,'status':'entire_original_source_executed','stdout':output.getvalue()}
        if i==2:
            assert result.width==400 and result.format=='png'
            image=PILImage.open(io.BytesIO(result.data)); image.verify()
            entry['image_sha256']=hashlib.sha256(result.data).hexdigest()
        if i in (9,10):
            assert result.item()==(2 if i==9 else 4) and result.shape==torch.Size([])
            entry['grad_value']=result.item();entry['dtype']=str(result.dtype)
        if i==15:
            ds=context['ds']; assert len(ds)==1024 and ds[0]['input_ids'].shape==torch.Size([512])
            assert ds[0]['input_ids'].dtype==torch.int64
            assert 100<=int(ds['input_ids'].min()) and int(ds['input_ids'].max())<30000
            entry['shape']=[1024,512];entry['numpy_input_sha256']=hashlib.sha256(context['dummy_data']['input_ids'].tobytes()).hexdigest()
        if i==16:
            assert result=={0,1}; entry['labels']=[int(v) for v in sorted(result)]
        reports.append(entry)
    return reports

def tiny_bert():
    # vocab覆盖原100–29999随机ID；仅改变层数/hidden尺寸/seq数据，并关闭dropout控制对照。
    torch.manual_seed(42)
    config=BertConfig(vocab_size=30000, hidden_size=16, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=32, num_labels=2,
                      hidden_dropout_prob=0., attention_probs_dropout_prob=0., max_position_embeddings=512)
    return BertForSequenceClassification(config)

class Trace(TrainerCallback):
    """真实Trainer回调：只观察micro/更新事件，不替换loss、optimizer或backward。"""
    def __init__(self): self.events=[]
    def on_substep_end(self,args,state,control,**kwargs): self.events.append('micro')
    def on_step_end(self,args,state,control,**kwargs): self.events.append('update')

class CPUFixture(ast.NodeTransformer):
    def __init__(self): self.changes=[]
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in ('num_seqs','seq_len'):
                assert isinstance(node.value,ast.Constant) and node.value.value==({'num_seqs':1024,'seq_len':512}[key])
                node.value=ast.Constant({'num_seqs':8,'seq_len':16}[key]); self.changes.append(key)
            elif key=='model':
                assert ast.unparse(node.value)=="AutoModelForSequenceClassification.from_pretrained('bert-large-uncased').to('cuda')"
                node.value=ast.Call(func=ast.Name(id='tiny_bert',ctx=ast.Load()),args=[],keywords=[])
                self.changes.append('BERT-large/CUDA load -> random CPU BertConfig fixture')
        return node

def training(cells,index):
    # 每次临时cwd保证原output_dir='tmp'只写临时目录；原公开文件不修改。
    previous=Path.cwd(); trace=Trace(); stdout=io.StringIO(); forward_shapes=[]
    with tempfile.TemporaryDirectory(prefix='lesson11-trainer-') as tmp:
        try:
            os.chdir(tmp)
            context={'tiny_bert':tiny_bert}
            exec(''.join(cells[0]['source']),context)
            exec(''.join(cells[12]['source']),context)
            np.random.seed(1234)
            if index==23:
                transform=CPUFixture(); data=transform.visit(ast.parse(''.join(cells[15]['source'])))
                ast.fix_missing_locations(data);exec(compile(data,'<data-CPU-fixture>','exec'),context)
                context['model']=tiny_bert()
                code=ast.parse(''.join(cells[23]['source']))
                changes=transform.changes+['provided random CPU BertConfig model in initial context']
            else:
                transform=CPUFixture();code=transform.visit(ast.parse(''.join(cells[26]['source'])))
                changes=transform.changes
                assert len(changes)==3
            ast.fix_missing_locations(code)
            # 插入只读真实callback要在原Trainer构造之后、train之前；语句本身不改。
            state_before=None
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stdout):
                for statement in code.body:
                    exec(compile(ast.Module(body=[statement],type_ignores=[]),'<course-training>','exec'),context)
                    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='trainer' for t in statement.targets):
                        trainer=context['trainer'];trainer.add_callback(trace)
                        def observe_forward(module,args,kwargs,output):
                            # 只保存普通列表/dtype，不保留Tensor或图，不修改forward结果。
                            forward_shapes.append({'input_ids':list(kwargs['input_ids'].shape),
                                'labels':list(kwargs['labels'].shape), 'logits':list(output.logits.shape),
                                'loss_shape':list(output.loss.shape), 'ids_dtype':str(kwargs['input_ids'].dtype),
                                'logits_dtype':str(output.logits.dtype)})
                        context['model'].register_forward_hook(observe_forward,with_kwargs=True)
                        state_before={k:v.detach().clone() for k,v in context['model'].state_dict().items()}
            result=context['result'];model=context['model'];trainer=context['trainer']
            assert result.global_step==2 and trainer.state.global_step==2
            assert trace.events==(['update']*2 if index==23 else ['micro']*3+['update']+['micro']*3+['update'])
            expected_batch=4 if index==23 else 1
            assert len(forward_shapes)==(2 if index==23 else 8)
            assert all(s['input_ids']==[expected_batch,16] and s['labels']==[expected_batch]
                       and s['logits']==[expected_batch,2] and s['loss_shape']==[] for s in forward_shapes)
            assert any(not torch.equal(v,state_before[k]) for k,v in model.state_dict().items())
            ds=context['ds']; assert len(ds)==8 and ds[0]['input_ids'].shape==torch.Size([16])
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
            initial=state_before
            report={'original_cell':index,'scope':'entire original cell with provided CPU model/data context' if index==23 else 'explicit three AST-substitution CPU teaching variant',
                    'changes':changes,'num_samples':8,'seq_len':16,'model_parameters':sum(p.numel() for p in model.parameters()),
                    'global_step':result.global_step,'training_loss':result.training_loss,'callback_events':trace.events,
                    'observed_forward_shapes':forward_shapes,
                    'per_device_batch':trainer.args.per_device_train_batch_size,'accumulation':trainer.args.gradient_accumulation_steps,
                    'torch_visible_cuda_devices':torch.cuda.device_count(),'actual_device':str(trainer.args.device),
                    'dataset_numpy_SHA256':hashlib.sha256(context['dummy_data']['input_ids'].tobytes()).hexdigest(),
                    'stdout':stdout.getvalue(),'real_Trainer_optimizer_backward_used':True,
                    'no_model_weights_downloaded':True,'not_original_GPU_benchmark':True}
            del trainer,model,context,ds
            gc.collect()
        finally: os.chdir(previous)
    return report,state,initial

def main():
    torch.set_num_threads(1)
    lesson=Path(__file__).resolve().parents[2];folder=lesson/'代码/tutorials'
    original,commented,audit_result=audit(lesson)
    previous=Path.cwd()
    try:
        os.chdir(folder)
        raw_results=run_original_cells(original);comment_results=run_original_cells(commented)
    finally: os.chdir(previous)
    assert raw_results==comment_results
    pynvml.nvmlInit()
    hardware={'torch_visible_cuda_devices':torch.cuda.device_count(),'real_NVML_devices':pynvml.nvmlDeviceGetCount(),
              'real_NVML_used_bytes':[pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used for i in range(pynvml.nvmlDeviceGetCount())],
              'warning':'Original helper loops over torch CUDA count; CPU torch prints 0 even when NVML sees hardware.'}
    sys.path.insert(0,str(lesson/'代码'))
    supplement=importlib.import_module('窗口归一化_教学补充')
    raw_loops={i:loop(''.join(original[i]['source'])) for i in (5,7)}
    comment_loops={i:loop(''.join(commented[i]['source'])) for i in (5,7)}
    raw_math=supplement.run_checks(raw_loops);comment_math=supplement.run_checks(comment_loops)
    assert raw_math==comment_math
    runs=[];states={};initials={}
    for name,cells in (('original',original),('annotated',commented)):
        for i in (23,26):
            report,state,initial=training(cells,i);report['version']=name
            runs.append(report);states[name,i]=state;initials[name,i]=initial
    for i in (23,26):
        assert all(torch.equal(states['original',i][k],states['annotated',i][k]) for k in states['original',i])
    assert all(torch.equal(initials['original',23][k],initials['original',26][k]) for k in initials['original',23])
    assert runs[0]['dataset_numpy_SHA256']==runs[1]['dataset_numpy_SHA256']
    delta=max((states['original',23][k]-states['original',26][k]).abs().max().item() for k in states['original',23])
    # 同样初始化/顺序、dropout0，两次AdamW/线性scheduler；浮点归约次序可能有微差。
    assert delta<3e-6, delta
    out=lesson/'复现/outputs'
    (out/'annotation_verification.json').write_bytes((json.dumps(audit_result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    result={'environment':{'OS':platform.platform(),'Python':platform.python_version(),
            'packages':{p:version(p) for p in ('torch','transformers','accelerate','datasets','numpy','nvidia-ml-py','IPython','nbformat','Pillow')}},
            'original_code_results':raw_results,'annotated_code_results':comment_results,
            'original_annotated_11_unmodified_cells_equal':True,'hardware_NVML':hardware,
            'actual_original_Markdown_loops_and_corrected':raw_math,'Markdown_annotations_results_equal':True,
            'CPU_BERT_Trainer_runs':runs,'original_annotated_CPU_Trainer_parameters_exactly_equal':True,
            'baseline_accumulation_initial_data_and_parameters_equal':True,'baseline_accumulation_max_parameter_delta':delta,
            'supplement_sha256':hashlib.sha256((lesson/'代码/窗口归一化_教学补充.py').read_bytes()).hexdigest(),
            'BERT_large_downloaded':False,'CUDA_training_or_original_benchmark_reproduced':False}
    (out/'runtime_verification.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    pynvml.nvmlShutdown()
    print('All28 notes/15 code ASTs/2 Markdown loop ASTs/3 schemas and original history passed.')
    print('Real original/annotated11 cells, autograd/SGD full-tail-unequal windows and four tiny BERT Trainer runs passed.')
    print('CPU baseline vs accumulation max parameter delta:',delta)

if __name__=='__main__': main()
