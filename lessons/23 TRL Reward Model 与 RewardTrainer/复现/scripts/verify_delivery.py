# -*- coding: utf-8 -*-
"""Genuine fixed TRL/PEFT CPU training, all-cell audit and pairwise scoring checks."""
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
import re
import sys
import tempfile
import tokenize
from pathlib import Path

import nbformat
import numpy as np
import torch
import torch.nn.functional as F
from IPython.core.inputtransformer2 import TransformerManager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'代码/tutorials/reward_model'
TRANSFORM=TransformerManager()
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n):return ast.dump(n,include_attributes=False)
def plain(s):return TRANSFORM.transform_cell(s)
def tokens(s):return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline) if t.type not in (tokenize.COMMENT,tokenize.NL)]
def segment(src,node):
    first=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    return ''.join(src.splitlines(keepends=True)[first-1:node.end_lineno])+'\n'

def audit(sources):
    manifest=json.loads((ROOT/'来源清单.json').read_bytes())
    for e in manifest['files']:
        b=(ROOT/e['published_path']).read_bytes();assert sha(b)==e['sha256'] and len(b)==e['bytes']
    for name,h in manifest['official_source_sha256'].items():assert sha((sources/name).read_bytes())==h
    for name,h in manifest['tokenizer']['local_files_sha256'].items():assert sha((sources/'opt-tokenizer'/name).read_bytes())==h
    nbs=[json.loads((CODE/n).read_bytes()) for n in ('trl_reward_trainer.ipynb','原代码_带说明.ipynb','代码注释.ipynb')]
    for n in nbs:nbformat.validate(nbformat.from_dict(n))
    raw,explained,annotated=nbs
    assert len(raw['cells'])==37 and len(explained['cells'])==len(annotated['cells'])==75
    originals=[]
    for n in nbs[1:]:assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
    for i,c in enumerate(raw['cells']):
        assert explained['cells'][2*i+2]==c
        a=annotated['cells'][2*i+2]
        assert {k:v for k,v in a.items() if k!='source'}=={k:v for k,v in c.items() if k!='source'}
        if c['cell_type']=='code':
            s=plain(''.join(c['source']));t=plain(''.join(a['source']))
            assert dump(ast.parse(s))==dump(ast.parse(t)) and tokens(s)==tokens(t)
            originals.append(s)
        else:assert a==c
    assert len(originals)==29
    assert dump(ast.parse('\n'.join(originals)))==dump(ast.parse((CODE/'trl_reward_trainer_代码注释.py').read_text(encoding='utf-8')))
    reading=(CODE/'RewardTrainer核心源码_代码注释.py').read_text(encoding='utf-8');tree=ast.parse(reading)
    for name,file in [('RewardTrainer','reward_trainer.py'),('RewardDataCollatorWithPadding','trainer_utils.py'),('compute_accuracy','trainer_utils.py')]:
        src=(sources/file).read_text(encoding='utf-8');node=next(n for n in ast.parse(src).body if getattr(n,'name',None)==name)
        read=next(n for n in tree.body if getattr(n,'name',None)==name)
        assert dump(node)==dump(read) and tokens(segment(src,node))==tokens(segment(reading,read))
    return raw,annotated,{'cells':37,'code_cells':29,'schemas':3,'derived_cells_each':75,'ipython_pinfo2':True,
        'preserved':'raw bytes, all metadata/history, original cell ordering, transformed AST and noncomment tokens',
        'official_complete_objects':['RewardTrainer','RewardDataCollatorWithPadding','compute_accuracy']}

def dataset_fixture(Dataset):
    pairs=[('Name a primary color.','Red is a primary color.','I do not know.'),
      ('What is two plus two?','Two plus two is four.','It is five.'),
      ('Give a kind greeting.','Hello, I hope you have a good day.','Go away.'),
      ('What does a plant need?','Plants need water and light.','Only plastic.'),
      ('What is the capital of France?','Paris is the capital of France.','It is Madrid.'),
      ('How do I learn safely?','Practice gradually and ask for help.','Never ask questions.'),
      ('Give a short reply.','Yes.',' very'*600),
      ('Give another short reply.',' very'*600,'No.')]
    return Dataset.from_dict({k:[f'\n\nHuman: {p}\n\nAssistant: {v}' for p,*responses in pairs for v in [responses[0 if k=='chosen' else 1]]]
                              for k in ('chosen','rejected')})

class Variant(ast.NodeTransformer):
    def __init__(self,index,tokenizer_path,work):self.index=index;self.tokenizer_path=tokenizer_path;self.work=work;self.changes=[]
    def visit_Call(self,node):
        self.generic_visit(node)
        if self.index==13 and isinstance(node.func,ast.Attribute) and node.func.attr=='from_pretrained':
            assert node.args[0].value=='facebook/opt-350m';node.args[0]=ast.Constant(str(self.tokenizer_path));self.changes.append('model name -> pinned local tokenizer path')
        if self.index==15:
            for k in node.keywords:
                if k.arg=='num_proc':assert k.value.value==16;k.value=ast.Constant(1);self.changes.append('num_proc16 -> 1')
        if self.index==20:
            for k in node.keywords:
                if k.arg=='load_in_4bit':assert k.value.value is True;k.value=ast.Constant(False);self.changes.append('load_in_4bit True -> False (no quantization)')
        if self.index==21 and isinstance(node.func,ast.Attribute) and node.func.attr=='from_pretrained':
            self.changes.append('entire pretrained/quantized load -> actual OPTForSequenceClassification(tiny_config)')
            return ast.copy_location(ast.Call(func=ast.Name('OPTForSequenceClassification',ast.Load()),args=[ast.Name('tiny_config',ast.Load())],keywords=[]),node)
        if self.index==29 and isinstance(node.func,ast.Name) and node.func.id=='TrainingArguments':
            replace={'output_dir':str(self.work/'train_logs'),'max_steps':2,'per_device_train_batch_size':2,
                'learning_rate':.001,'save_steps':2,'logging_steps':1,'report_to':[]}
            for k in node.keywords:
                if k.arg in replace:
                    value=replace[k.arg];k.value=ast.parse(repr(value),mode='eval').body;self.changes.append(k.arg+' -> '+repr(value))
            for name,value in {'use_cpu':True,'disable_tqdm':True,'seed':23,'data_seed':23,'dataloader_num_workers':0,'save_safetensors':True}.items():
                node.keywords.append(ast.keyword(arg=name,value=ast.Constant(value)));self.changes.append('add '+name+'='+repr(value))
        if self.index==31 and isinstance(node.func,ast.Attribute) and node.func.attr=='save_pretrained':
            assert node.args[0].value=='./reward_model';node.args[0]=ast.Constant(str(self.work/'reward_model'));self.changes.append('save path -> temporary reward_model')
        return node

def run_notebook(nb,sources,work,annotated=False):
    from datasets import Dataset,disable_progress_bar
    from transformers import OPTConfig,OPTForSequenceClassification
    disable_progress_bar();torch.manual_seed(23)
    shell=InteractiveShell.instance();shell.reset(new_session=False)
    def pager(self,data,start=0,screen_lines=0):print(data if isinstance(data,str) else data.get('text/plain',''))
    shell.set_hook('show_in_pager',pager)
    ns=shell.user_ns
    cfg=OPTConfig(vocab_size=50272,hidden_size=16,word_embed_proj_dim=16,ffn_dim=32,num_hidden_layers=1,
        num_attention_heads=2,max_position_embeddings=64,pad_token_id=1,bos_token_id=2,eos_token_id=2,
        num_labels=1,dropout=0.,attention_dropout=0.,activation_dropout=0.,layerdrop=0.,do_layer_norm_before=False)
    ns.update(train_dataset=dataset_fixture(Dataset),OPTForSequenceClassification=OPTForSequenceClassification,tiny_config=cfg)
    records=[];images=[];literal=[];variants={};quant_error=None;before=None;trainable=None;base_init=None;loss_before=None;initial_logits=None
    cells=nb['cells'][2::2] if annotated else nb['cells']
    for i,c in enumerate(cells):
        if c['cell_type']!='code':continue
        if i in (0,9):records.append({'cell':i,'status':'skipped','reason':'proxy side effect' if i==0 else 'remote full dataset; explicit 8-row teaching fixture supplied'});continue
        source=''.join(c['source'])
        if i==20:
            with capture_output():failure=shell.run_cell(source,store_history=False)
            err=failure.error_before_exec or failure.error_in_exec
            assert isinstance(err,importlib.metadata.PackageNotFoundError)
            quant_error={'type':type(err).__name__,'message':str(err)}
        if i in (13,15,20,21,29,31):
            tr=Variant(i,sources/'opt-tokenizer',work);tree=tr.visit(ast.parse(source));assert tr.changes
            variants[str(i)]=tr.changes;source=ast.unparse(ast.fix_missing_locations(tree))
        with capture_output() as captured:result=shell.run_cell(source,store_history=False)
        err=result.error_before_exec or result.error_in_exec
        assert result.success,(i,type(err).__name__,str(err),captured.stdout,captured.stderr)
        if i not in (13,15,20,21,29,31):literal.append(i)
        rec={'cell':i,'status':'CPU/local teaching variant' if str(i) in variants else 'literal original statements executed'}
        if i in (5,6):
            html=result.result._repr_html_();assert 'https://miro.medium.com/' in html and 'width="600"' in html
            images.append(html);rec['image_html_sha256']=sha(html.encode())
        if i==13:
            tok=ns['tokenizer'];assert (tok.pad_token_id,tok.bos_token_id,tok.eos_token_id)==(1,2,2)
            tok.model_max_length=1024 # Explicit fixture limit, not an original preprocessing argument.
        if i==15:
            assert len(ns['train_dataset'])==6
            assert set(ns['train_dataset'].column_names)=={'chosen','rejected','input_ids_chosen','attention_mask_chosen','input_ids_rejected','attention_mask_rejected'}
            rec['filtered_rows']=6
        if i==21:
            base_init={k:v.detach().clone() for k,v in ns['model'].state_dict().items()}
            with torch.no_grad():initial_logits=ns['model'](input_ids=torch.tensor([[2,100,200]]),attention_mask=torch.ones(1,3,dtype=torch.long)).logits.clone()
        if i==23:
            displayed=captured.stdout+captured.stderr+str(captured.outputs)
            displayed=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', displayed)
            assert 'def forward' in displayed and 'sequence_lengths' in displayed,displayed[-3000:]
            rec['actual_OPT_forward_source_inspected']=True
        if i==29:
            trainer=ns['trainer'];m=trainer.model
            trainable={n:p.numel() for n,p in m.named_parameters() if p.requires_grad}
            assert sum(trainable.values())==1040 and all(('lora_' in n or 'score.modules_to_save' in n) for n in trainable)
            assert all(torch.count_nonzero(p)==0 for n,p in m.named_parameters() if 'lora_B' in n)
            with torch.no_grad():after_injection=m(input_ids=torch.tensor([[2,100,200]]),attention_mask=torch.ones(1,3,dtype=torch.long)).logits
            assert torch.equal(initial_logits,after_injection)
            before={n:p.detach().clone() for n,p in m.named_parameters()}
            batch=trainer.data_collator([ns['train_dataset'][j] for j in (0,1)])
            with torch.no_grad():loss_before=float(trainer.compute_loss(m,batch))
        if i==31:
            trainer=ns['trainer'];assert trainer.state.global_step==2
            for n,p in trainer.model.named_parameters():
                if n not in trainable:assert torch.equal(p,before[n]),n
            changed=[n for n,p in trainer.model.named_parameters() if not torch.equal(p,before[n])]
            assert changed and any('lora_B' in n for n in changed) and any('score.modules_to_save' in n for n in changed)
            assert all(n in trainable for n in changed)
            opt_ids={id(p) for g in trainer.optimizer.param_groups for p in g['params']}
            assert opt_ids=={id(p) for p in trainer.model.parameters() if p.requires_grad}
            rec['optimizer_steps']=2
        if i==33:assert list(result.result.shape)==[1,1];rec['score']=result.result.detach().tolist()
        if i==35:assert result.result.loss is None and list(result.result.logits.shape)==[1,1];rec['loss_is_None']=True
        if i in (34,36):rec['sigmoid_constant']=result.result.tolist()
        records.append(rec)
    trainer=ns['trainer'];m=trainer.model.eval();data=ns['train_dataset'];tok=ns['tokenizer']
    batch=trainer.data_collator([data[j] for j in (0,1)])
    with torch.no_grad():loss_after=float(trainer.compute_loss(m,batch))
    from peft import PeftModel
    reload_base=OPTForSequenceClassification(cfg);reload_base.load_state_dict(base_init)
    loaded=PeftModel.from_pretrained(reload_base,work/'reward_model').eval()
    with torch.no_grad():
        y=m(input_ids=batch['input_ids_chosen'],attention_mask=batch['attention_mask_chosen']).logits
        z=loaded(input_ids=batch['input_ids_chosen'],attention_mask=batch['attention_mask_chosen']).logits
    assert torch.equal(y,z)
    adapter=json.loads((work/'reward_model/adapter_config.json').read_text())
    assert 'score' in adapter['modules_to_save']
    from safetensors.torch import load_file
    saved=load_file(str(work/'reward_model/adapter_model.safetensors'))
    assert any(k.endswith('score.weight') for k in saved) and all('lora_' in k or 'score' in k for k in saved)
    state={n:p.detach().clone() for n,p in m.named_parameters()}
    report={'literal_success_cells':literal,'literal_count':len(literal),'teaching_variants':variants,'all_code_cells':records,
        'original_4bit_config_failure':quant_error,'fixture':'8 self-written pairs, tokenizer max1024, two long rows filtered to6; no original HH dataset',
        'tokenizer':{'length':len(tok),'vocab_size':tok.vocab_size,'pad':tok.pad_token_id,'bos':tok.bos_token_id,'eos':tok.eos_token_id},
        'tiny_base_parameters':sum(v.numel() for v in base_init.values()),'trainable_parameters':trainable,'trainable_total':sum(trainable.values()),
        'updated_parameters':changed,'initial_lora_B_zero_and_injection_logits_identical':True,'frozen_unchanged':True,'optimizer_trainable_only':True,'global_steps':trainer.state.global_step,
        'first_two_pair_loss_before':loss_before,'first_two_pair_loss_after':loss_after,'reload_max_logit_error':float((y-z).abs().max()),
        'saved_keys':list(saved),'saved_adapter_config':adapter,'image_html_only':images,
        'train_history':[{k:v for k,v in x.items() if k not in ('train_runtime','train_samples_per_second','train_steps_per_second')} for x in trainer.state.log_history]}
    return trainer,data,tok,state,report

def error_of(fn):
    try:fn()
    except Exception as e:return {'type':type(e).__name__,'message':str(e)[:500]}
    raise AssertionError('Expected exception')

def additional(trainer,data,tok):
    from trl.trainer.utils import RewardDataCollatorWithPadding,compute_accuracy
    model=trainer.model.eval();batch=trainer.data_collator([data[0],data[1]])
    loss,outputs=trainer.compute_loss(model,batch,return_outputs=True)
    chosen=outputs['rewards_chosen'];rejected=outputs['rewards_rejected'];diff=chosen-rejected
    assert list(chosen.shape)==list(rejected.shape)==[2,1]
    assert torch.allclose(loss,F.softplus(-diff).mean(),atol=1e-7)
    gc,gr=torch.autograd.grad(loss,(chosen,rejected),retain_graph=True)
    expected=-torch.sigmoid(-diff)/2
    assert torch.allclose(gc,expected,atol=1e-7) and torch.allclose(gr,-expected,atol=1e-7)
    prediction_loss,probs,labels=trainer.prediction_step(model,batch,False)
    assert list(probs.shape)==[2,2] and torch.equal(labels,torch.zeros(2))
    assert torch.allclose(probs[:,0],torch.sigmoid(diff[:,0]),atol=1e-7)
    shift=F.softplus(-((chosen+100)-(rejected+100))).mean()
    assert torch.allclose(shift,loss,atol=1e-5)
    single=torch.sigmoid(chosen);single_shift=torch.sigmoid(chosen+100)
    assert not torch.allclose(single,single_shift)
    # Padding on either side must preserve per-sequence score with mask and correct PAD id.
    oldside=tok.padding_side;padded=[]
    with torch.no_grad():
        single_scores=[]
        for j in (0,1):
            x=torch.tensor([data[j]['input_ids_chosen']]);mask=torch.ones_like(x)
            single_scores.append(model(input_ids=x,attention_mask=mask).logits.item())
        for side in ('right','left'):
            tok.padding_side=side;coll=RewardDataCollatorWithPadding(tok,max_length=512)
            b=coll([data[0],data[1]]);score=model(input_ids=b['input_ids_chosen'],attention_mask=b['attention_mask_chosen']).logits[:,0]
            assert torch.allclose(score,torch.tensor(single_scores),atol=1e-6)
            padded.append({'side':side,'chosen_shape':list(b['input_ids_chosen'].shape),'rejected_shape':list(b['input_ids_rejected'].shape),'scores':score.tolist()})
    tok.padding_side=oldside
    # Original preprocess function under a different explicit tokenizer limit:
    # truncating to512 first retains long rows that filter512 would otherwise remove.
    from datasets import Dataset
    original=json.loads((CODE/'trl_reward_trainer.ipynb').read_bytes())
    pre_ns={'tokenizer':tok};exec(''.join(original['cells'][14]['source']),pre_ns)
    old_limit=tok.model_max_length
    try:
        tok.model_max_length=512
        truncated=dataset_fixture(Dataset).map(pre_ns['preprocess'],batched=True)
        truncated=truncated.filter(lambda x:len(x['input_ids_chosen'])<=512 and len(x['input_ids_rejected'])<=512)
        assert len(truncated)==8
    finally:tok.model_max_length=old_limit
    # max_length passed with padding=True is not a truncation request.
    long={'input_ids_chosen':[2]+[100]*599,'attention_mask_chosen':[1]*600,
          'input_ids_rejected':[2,100],'attention_mask_rejected':[1,1]}
    longbatch=trainer.data_collator([long]);assert longbatch['input_ids_chosen'].shape[1]==600
    missing=error_of(lambda:trainer.data_collator([{'input_ids_chosen':[2]}]))
    # Genuine old margin broadcast: [B,1] minus [B] makes [B,B], not pair-specific.
    features=[{**data[j],'margin':v} for j,v in zip((0,1),(.2,1.3))]
    marginbatch=trainer.data_collator(features)
    assert list(marginbatch['margin'].shape)==[2]
    old_margin=trainer.compute_loss(model,marginbatch)
    crosspair=F.softplus(-(diff-marginbatch['margin'])).mean()
    assert torch.allclose(old_margin,crosspair)
    fixedbatch={**marginbatch,'margin':marginbatch['margin'][:,None]}
    fixed_margin=trainer.compute_loss(model,fixedbatch)
    correct=F.softplus(-(diff-marginbatch['margin'][:,None])).mean()
    assert torch.allclose(fixed_margin,correct)
    # Nonuniform scores exhibit the broadcast error clearly, independent of training luck.
    d=torch.tensor([[2.],[-1.]],dtype=torch.float64);margin=torch.tensor([.2,1.3],dtype=torch.float64)
    broadcast_example=float(F.softplus(-(d-margin)).mean());paired_example=float(F.softplus(-(d-margin[:,None])).mean())
    assert abs(broadcast_example-paired_example)>.01
    tie_metrics=compute_accuracy((np.array([[.5,.5],[.8,.2]]),np.zeros(2)))
    assert tie_metrics['accuracy']==1.
    z=torch.tensor([-1000.,0.,1000.],dtype=torch.float64,requires_grad=True)
    stable=F.softplus(-z);assert torch.allclose(stable,-F.logsigmoid(z),atol=1e-12);stable.sum().backward()
    assert torch.isfinite(stable).all() and torch.isfinite(z.grad).all()
    assert abs(float(stable[1])-np.log(2))<1e-15
    naive=-torch.log(torch.sigmoid(z.detach()));assert torch.isinf(naive[0])
    historical={'chosen':.6126,'rejected':1.4780,'pair_probability':torch.sigmoid(torch.tensor(.6126-1.4780)).item(),
        'constant_sigmoids':[torch.sigmoid(torch.tensor(.1196)).item(),torch.sigmoid(torch.tensor(.9642)).item()]}
    return {'scores':{'chosen':chosen.detach().tolist(),'rejected':rejected.detach().tolist(),'loss':loss.item(),
            'chosen_gradient':gc.tolist(),'rejected_gradient':gr.tolist()},
        'prediction':{'shape':list(probs.shape),'probabilities':probs.tolist(),'labels':labels.tolist()},
        'common_shift_invariance':True,'single_score_sigmoid_changes_under_shift':True,'padding':padded,
        'collator_max_length512_still_returns600':True,'preprocess_limit512_retains_all8_vs_limit1024_retains6':True,'missing_fields':missing,
        'margin':{'original_margin_shape':[2],'broadcast_intermediate_shape':[2,2],'old_loss':old_margin.item(),
            'explicit_column_margin_loss':fixed_margin.item(),'independent_nonuniform_example':{'broadcast':broadcast_example,'correct_pairwise':paired_example},
            'fix':'Only independent inputs use margin[:,None]; original collator/source preserved.'},
        'tie_accuracy_includes_first_column_tie':tie_metrics,'stable_loss':stable.detach().tolist(),'stable_gradient':z.grad.tolist(),
        'naive_negative1000_is_inf':True,'historical_numbers_only':historical}

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True);a=p.parse_args();sources=Path(a.sources)
    sys.path.insert(0,str(sources/'runtime-libs'));torch.set_num_threads(1)
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[name]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import trl
    from trl import RewardTrainer
    import trl.trainer.utils as utils
    assert trl.__version__=='0.8.6'
    manifest=json.loads((ROOT/'来源清单.json').read_bytes())
    assert sha(Path(inspect.getfile(RewardTrainer)).read_bytes())==manifest['official_source_sha256']['reward_trainer.py']
    assert sha(Path(inspect.getfile(utils.RewardDataCollatorWithPadding)).read_bytes())==manifest['official_source_sha256']['trainer_utils.py']
    raw,annotated,static=audit(sources)
    with tempfile.TemporaryDirectory(prefix='lesson23-') as td:
        t=Path(td);(t/'original').mkdir();(t/'annotated').mkdir()
        trainer,data,tok,state1,r1=run_notebook(raw,sources,t/'original')
        checks=additional(trainer,data,tok)
        _,_,_,state2,r2=run_notebook(annotated,sources,t/'annotated',True)
        assert state1.keys()==state2.keys() and all(torch.equal(state1[n],state2[n]) for n in state1)
        assert r1['first_two_pair_loss_after']==r2['first_two_pair_loss_after']
        assert r1['image_html_only']==r2['image_html_only']
    files=[Path(__file__),Path(__file__).with_name('build_delivery.py'),*CODE.glob('*.ipynb'),*CODE.glob('*.py')]
    report={'status':'passed','static':static,'original':r1,'annotated':r2,'states_exact_equal':True,'additional':checks,
        'environment':{'python':sys.version,'platform':platform.platform(),'cuda_available':torch.cuda.is_available(),
            'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','trl','peft','datasets','accelerate','numpy','pyarrow','nbformat','ipython','tokenizers')}},
        'artifact_sha256':{str(f.relative_to(ROOT)).replace('\\','/'):sha(f.read_bytes()) for f in files},
        'limits':'Two actual CPU OPT/LoRA RewardTrainer runs, 2 optimizer updates each, synthetic preferences. No HH full corpus, OPT350M weights, 4bit backend, original1000 steps, GPU/PPO, calibration or human preference benchmark.'}
    out=ROOT/'复现/outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'runtime_verification.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'status':'passed','static':static,'literal_cells_each':r1['literal_count'],'variants_each':len(r1['teaching_variants']),
        'trainable':r1['trainable_total'],'base_params':r1['tiny_base_parameters'],'steps_each':2,'reload_error':r1['reload_max_logit_error'],
        'loss_before':r1['first_two_pair_loss_before'],'loss_after':r1['first_two_pair_loss_after'],'states_exact':True},ensure_ascii=True))

if __name__=='__main__':main()
