# -*- coding: utf-8 -*-
"""原件/全部单元审计，真实Torch重算边界及四组受控BERT Trainer对照。"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import platform
import re
import sys
import tempfile
import warnings
from pathlib import Path

import accelerate
import datasets
import nbformat
import numpy as np
import pynvml
import torch
import transformers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedModel, Trainer, TrainingArguments, set_seed
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.optimization import AdamW
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'代码/tutorials'
SHA='a2f49289c867c32bf2ac3795c2068586d0840420eccb98ebcc28bd2e67193711'
CODE=[4,6,7,8,9,11,13,14,16,17,19,22]


def text(cell): return ''.join(cell['source'])
def ast_text(src): return ast.dump(ast.parse(src))
def md_code(src): return re.search(r'```[^\n]*\n(.*?)```',src,re.S).group(1)
def write_json(path,result): path.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def comments_only(original,changed):
    lines=original.splitlines(keepends=True);i=0
    for line in changed.splitlines(keepends=True):
        if i<len(lines) and line==lines[i]: i+=1
        else: assert line.lstrip().startswith('#')
    assert i==len(lines)


def audit():
    paths=[FOLDER/n for n in ['gradient_checkpointing.ipynb','原代码_带说明.ipynb','代码注释.ipynb']]
    raw,explained,annotated=[json.loads(p.read_bytes()) for p in paths]
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest()==SHA
    assert len(raw['cells'])==23 and [i for i,c in enumerate(raw['cells']) if c['cell_type']=='code']==CODE
    for nb in (raw,explained,annotated):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');nbformat.validate(nbformat.from_dict(copy.deepcopy(nb)))
    for derived,changed in [(explained,False),(annotated,True)]:
        assert len(derived['cells'])==47
        assert {k:v for k,v in derived.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
        for i,old in enumerate(raw['cells']):
            note,saved=derived['cells'][1+2*i:3+2*i]
            assert note['cell_type']=='markdown' and '单元 {}（'.format(i) in text(note)
            assert {k:v for k,v in old.items() if k!='source'}=={k:v for k,v in saved.items() if k!='source'}
            if changed and old['cell_type']=='code':
                assert ast_text(text(old))==ast_text(text(saved));comments_only(text(old),text(saved))
            elif changed and i==21:
                assert ast_text(md_code(text(old)))==ast_text(md_code(text(saved)))
                comments_only(text(old),text(saved))
            else: assert saved==old
    py=(FOLDER/'gradient_checkpointing_代码注释.py').read_text(encoding='utf-8')
    assert ast_text(py)==ast_text('\n\n'.join(text(raw['cells'][i]) for i in CODE))
    doc=(ROOT/'代码单元解释.md').read_text(encoding='utf-8')
    assert all('## 单元 {}（'.format(i) in doc for i in range(23))
    report=dict(original_sha256=SHA,cells=23,code_indices=CODE,schemas_passed=3,derived_cells=47,
                code_AST_passed=12,markdown21_AST_passed=True,reading_AST_passed=True,
                all_notes_present=True,original_lines_metadata_historical_outputs_preserved=True)
    write_json(ROOT/'复现/outputs/annotation_verification.json',report)
    return raw,[annotated['cells'][2+2*i] for i in range(23)]


class CPUVariant(ast.NodeTransformer):
    def __init__(self,group): self.group=group
    def visit_Assign(self,node):
        self.generic_visit(node)
        if isinstance(node.targets[0],ast.Tuple) and [v.id for v in node.targets[0].elts]==['seq_len','dataset_size']:
            node.value=ast.Tuple(elts=[ast.Constant(16),ast.Constant(8)],ctx=ast.Load())
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='from_pretrained':
            node=ast.Call(func=ast.Name(id='BertForSequenceClassification',ctx=ast.Load()),
                          args=[ast.Name(id='teaching_config',ctx=ast.Load())],keywords=[])
        elif isinstance(node.func,ast.Attribute) and node.func.attr=='to':
            if node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='cuda': node.args[0]=ast.Constant('cpu')
        elif isinstance(node.func,ast.Name) and node.func.id=='TrainingArguments':
            # two extra controls differ only in the checkpoint flag; original14/16 remain whole-cell.
            if self.group=='B4K1_on': node.keywords.append(ast.keyword(arg='gradient_checkpointing',value=ast.Constant(True)))
            elif self.group=='B1K4_off':
                for kw in node.keywords:
                    if kw.arg=='gradient_checkpointing': kw.value=ast.Constant(False)
        return node


def config():
    return BertConfig(vocab_size=30000,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
                      intermediate_size=32,max_position_embeddings=512,num_labels=2,
                      hidden_dropout_prob=0,attention_probs_dropout_prob=0)


def run_version(cells,output):
    namespace={};stream=io.StringIO();whole=[6,7,8,9,13,17,19,22]
    with contextlib.redirect_stdout(stream):
        np.random.seed(1234)
        for i in whole: exec(compile(text(cells[i]),'original_cell_'+str(i),'exec'),namespace)
    assert namespace['dummy_data']['input_ids'].shape==(512,512)
    assert namespace['dummy_data']['labels'].shape==(512,)
    assert namespace['ds'][0]['input_ids'].dtype==torch.int64
    original_data_hash=hashlib.sha256(namespace['dummy_data']['input_ids'].tobytes()+namespace['dummy_data']['labels'].tobytes()).hexdigest()
    np.random.seed(1234)
    data_tree=CPUVariant('').visit(ast.parse(text(cells[6])))
    exec(compile(ast.fix_missing_locations(data_tree),'cell6_small_data','exec'),namespace)
    assert namespace['ds'][:]['input_ids'].shape==(8,16)
    teaching_config=config();namespace.update(BertForSequenceClassification=BertForSequenceClassification,teaching_config=teaching_config)
    runs={};states={};initial_reference=None;order_reference=None
    for group in ['B4K1_off','B4K1_on','B1K4_off','B1K4_on']:
        set_seed(42)
        with contextlib.redirect_stdout(stream):
            tree=CPUVariant(group).visit(ast.parse(text(cells[11])))
            exec(compile(ast.fix_missing_locations(tree),'cell11_CPU_random_config','exec'),namespace)
        model=namespace['model'];assert not model.is_gradient_checkpointing
        initial={n:p.detach().clone() for n,p in model.named_parameters()}
        if initial_reference is None: initial_reference=initial
        else: assert all(torch.equal(initial_reference[n],initial[n]) for n in initial)
        parameter_count=sum(p.numel() for p in model.parameters())
        # Original13 already ran unchanged; these explicit context updates bound/redirect the training.
        namespace['default_args'].update(output_dir=str(output/group),evaluation_strategy='no',max_steps=2,save_strategy='no')
        calls=[[] for _ in model.bert.encoder.layer];trace=[];handles=[]
        for index,layer in enumerate(model.bert.encoder.layer):
            def observe_layer(module,args,result,index=index): calls[index].append(torch.is_grad_enabled())
            handles.append(layer.register_forward_hook(observe_layer))
        def observe_model(module,args,kwargs,result):
            trace.append(dict(input_shape=list(kwargs['input_ids'].shape),label_shape=list(kwargs['labels'].shape),
                              sample_first_ids=kwargs['input_ids'][:,0].tolist(),labels=kwargs['labels'].tolist(),
                              logits_shape=list(result.logits.shape),loss=float(result.loss.detach()),dtype=str(result.logits.dtype)))
        handles.append(model.register_forward_hook(observe_model,with_kwargs=True))
        cell=14 if group.startswith('B4') else 16
        tree=CPUVariant(group).visit(ast.parse(text(cells[cell])))
        unchanged=group in ('B4K1_off','B1K4_on')
        if unchanged: assert ast.dump(tree)==ast_text(text(cells[cell]))
        with contextlib.redirect_stdout(stream):
            exec(compile(ast.fix_missing_locations(tree),'cell_'+str(cell)+'_'+group,'exec'),namespace)
        for handle in handles: handle.remove()
        on=group.endswith('_on');batch=4 if group.startswith('B4') else 1
        forwards=2 if batch==4 else 8
        assert len(trace)==forwards and all(t['input_shape']==[batch,16] and t['logits_shape']==[batch,2] for t in trace)
        assert all(len(c)==forwards*(2 if on else 1) for c in calls)
        assert all(c.count(False)==forwards if on else c.count(False)==0 for c in calls)
        assert model.is_gradient_checkpointing==on and namespace['trainer'].state.global_step==2
        assert isinstance(namespace['trainer'].optimizer.optimizer,AdamW)
        order=[v for item in trace for v in item['sample_first_ids']]
        if order_reference is None: order_reference=order
        else: assert order_reference==order
        assert all(p.grad is None for p in model.parameters())  # Trainer clears gradients after steps.
        state={n:p.detach().clone() for n,p in model.named_parameters()}
        assert any(not torch.equal(initial[n],state[n]) for n in initial)
        states[group]=state
        runs[group]=dict(training_cell=cell,training_cell_unaltered=unchanged,batch=batch,accumulation=1 if batch==4 else 4,
                         checkpoint=on,updates=2,micro_forwards=forwards,layer_grad_enabled_calls=calls,trace=trace,
                         optimizer='transformers.optimization.AdamW via AcceleratedOptimizer',
                         train_loss=namespace['result'].metrics['train_loss'],parameter_count=parameter_count)
    comparisons={}
    for left,right in [('B4K1_off','B4K1_on'),('B1K4_off','B1K4_on'),('B4K1_off','B1K4_off')]:
        error=max(float((states[left][n]-states[right][n]).abs().max()) for n in states[left])
        assert error<1e-6
        comparisons[left+' vs '+right]=dict(max_parameter_difference=error,state_exact=all(torch.equal(states[left][n],states[right][n]) for n in states[left]))
    # Original Markdown21 is executable only in this isolated test, not an original execution cell.
    policy_ns=dict(PreTrainedModel=PreTrainedModel,BertEncoder=BertEncoder)
    exec(compile(md_code(text(cells[21])),'markdown21_isolated','exec'),policy_ns)
    encoder=BertEncoder(teaching_config)
    policy=policy_ns['BertPreTrainedModel'](teaching_config)
    assert policy._set_gradient_checkpointing(encoder,True) is None and encoder.gradient_checkpointing
    policy._set_gradient_checkpointing(encoder);assert not encoder.gradient_checkpointing
    unrelated=torch.nn.Linear(2,2);policy._set_gradient_checkpointing(unrelated,True)
    assert not hasattr(unrelated,'gradient_checkpointing')
    # Actual scheduled evaluation with no eval_dataset, rather than a hypothetical warning.
    evaluation_model=BertForSequenceClassification(teaching_config)
    evaluation_args=TrainingArguments(output_dir=str(output/'missing_eval'),evaluation_strategy='steps',eval_steps=1,
                                     max_steps=1,per_device_train_batch_size=4,report_to='none',save_strategy='no',disable_tqdm=True)
    try:
        with contextlib.redirect_stdout(stream):
            Trainer(model=evaluation_model,args=evaluation_args,train_dataset=namespace['ds']).train()
    except ValueError as error:
        eval_failure=str(error);assert 'eval_dataset' in eval_failure
    else: raise AssertionError('scheduled evaluation should require a dataset')
    return dict(unaltered_common_whole_cells=whole,unaltered_training_cells=[14,16],original_data_shape=[512,512],
                original_data_sha256=original_data_hash,teaching_data_shape=[8,16],seed=dict(numpy=1234,model_and_Trainer=42),
                skipped_original_cells={'4':'CUDA_VISIBLE_DEVICES setting','11':'BERT-large remote/CUDA load'},
                variants={'6':'only size512/512 to16/8','11':'from_pretrained to random BertConfig construction, cuda to cpu',
                          'context_default_args':{'output_dir':'temporary directory','evaluation_strategy':'no','max_steps':2,'save_strategy':'no'},
                          'extra_controls':'14 checkpoint=True;16 checkpoint=False'},
                groups=runs,comparisons=comparisons,all_four_initial_states_exact=True,all_four_sample_orders_exact=True,
                markdown21_real_encoder_flag_test=True,
                scheduled_missing_eval_failure=eval_failure,stdout=stream.getvalue()),states


def load_teaching():
    spec=importlib.util.spec_from_file_location('checkpoint_teaching',ROOT/'代码/重算与梯度_教学补充.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    torch.set_num_threads(2)
    raw,annotated=audit();teaching=load_teaching()
    supplements=dict(sequential=teaching.sequential_experiment(),rng=teaching.rng_experiment(),
                     input_boundary=teaching.input_gradient_boundary(),frozen_BERT=teaching.frozen_bert_embeddings(config()))
    with tempfile.TemporaryDirectory(prefix='lesson13-checkpoint-') as work:
        original,s1=run_version(raw['cells'],Path(work)/'original')
        changed,s2=run_version(annotated,Path(work)/'annotated')
    assert original['original_data_sha256']==changed['original_data_sha256']
    for group in s1:
        assert all(torch.equal(s1[group][n],s2[group][n]) for n in s1[group])
        assert original['groups'][group]['trace']==changed['groups'][group]['trace']
        assert original['groups'][group]['layer_grad_enabled_calls']==changed['groups'][group]['layer_grad_enabled_calls']
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0);info=pynvml.nvmlDeviceGetMemoryInfo(handle)
    nvml=dict(physical_devices=pynvml.nvmlDeviceGetCount(),device_name=pynvml.nvmlDeviceGetName(handle),
              total_bytes=info.total,used_bytes=info.used,scope='physical GPU instant, not CPU training allocation or benchmark')
    sources={}
    for obj in [checkpoint,checkpoint_sequential,BertEncoder,Trainer]:
        p=Path(inspect.getfile(obj));sources[obj.__name__]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    scripts=[Path(__file__),ROOT/'复现/scripts/build_annotated.py',ROOT/'复现/scripts/download_original.py',ROOT/'代码/重算与梯度_教学补充.py']
    result=dict(status='部分成功',command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),python=sys.version,platform=platform.platform(),
                versions={m.__name__:m.__version__ for m in [torch,transformers,accelerate,datasets,np]},cuda_available=torch.cuda.is_available(),
                NVML=nvml,installed_sources=sources,script_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
                supplements=supplements,original=original,annotated=changed,
                original_annotated=dict(all_four_states_exact=True,all_forward_traces_exact=True,all_layer_call_traces_exact=True),
                not_run=['original CUDA-visible setting','BERT-large weights','CUDA training/memory peak/throughput benchmark','paper CNN/LSTM results'],
                memory_scope='sequential forward hook references only; no GPU memory claim')
    write_json(ROOT/'复现/outputs/runtime_verification.json',result)
    print(json.dumps(dict(schemas=3,code_AST=12,markdown_AST=True,four_groups_each_version=True,updates_each_group=2,
                          exact_original_annotated=True,sequential_call_counts=[2,2,1,1],rng_and_frozen_boundary=True),ensure_ascii=True))


if __name__=='__main__': main()
