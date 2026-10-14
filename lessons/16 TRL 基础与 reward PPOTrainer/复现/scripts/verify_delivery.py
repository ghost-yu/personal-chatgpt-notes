# -*- coding: utf-8 -*-
"""真实旧版PPOTrainer：原代码/注释等价，重复query与纯response修正分开执行。"""
import argparse
import ast
import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
import shlex
import re
import sys
import tempfile
import warnings
from pathlib import Path
import nbformat
import torch
import transformers
import trl
from transformers import GPT2Config,GPT2LMHeadModel,GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead,PPOConfig,PPOTrainer

ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'代码/tutorials/trl_hf'
RAW_SHA='1d0ec8e4f196914257158d18a48746b12561fa68100aeb778105720bd8b8d474'
TOKEN_SHA='8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
PPO_SHA='8142e1296171adc0a9f664ed0a241d4e7313ad0fac6987b2a093b0982489a407'
def src(c):return ''.join(c['source'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def astsrc(s):return ast.dump(ast.parse(s))
def save(name,r):(ROOT/'复现/outputs'/name).write_bytes((json.dumps(r,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def comment_only(a,b):
    lines=a.splitlines(True);i=0
    for line in b.splitlines(True):
        if i<len(lines) and line==lines[i]:i+=1
        else:assert line.lstrip().startswith('#')
    assert i==len(lines)
def audit():
    raw,notes,ann=[json.loads((FOLDER/name).read_bytes()) for name in ['trl_basics.ipynb','原代码_带说明.ipynb','代码注释.ipynb']]
    assert sha(FOLDER/'trl_basics.ipynb')==RAW_SHA
    codes=[i for i,c in enumerate(raw['cells']) if c['cell_type']=='code'];assert len(raw['cells'])==42 and len(codes)==28
    for n in (raw,notes,ann):
        with warnings.catch_warnings():warnings.simplefilter('ignore');nbformat.validate(nbformat.from_dict(copy.deepcopy(n)))
    for n,changed in [(notes,False),(ann,True)]:
        assert len(n['cells'])==85
        assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
        for i,c in enumerate(raw['cells']):
            note,saved=n['cells'][1+2*i:3+2*i];assert '单元 {}（'.format(i) in src(note)
            assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in saved.items() if k!='source'}
            if changed:
                comment_only(src(c),src(saved))
                if c['cell_type']=='code':assert astsrc(src(c))==astsrc(src(saved))
            else:assert c==saved
    assert astsrc((FOLDER/'trl_basics_代码注释.py').read_text(encoding='utf-8'))==astsrc('\n\n'.join(src(raw['cells'][i]) for i in codes))
    def shell(s):
        s=re.search(r'```[^\n]*\n(.*?)```',s,re.S).group(1)
        s='\n'.join(x for x in s.splitlines() if not x.lstrip().startswith('#'))
        return [shlex.split(x) for x in s.replace('\\\n','').splitlines() if x.strip()]
    assert shell(src(raw['cells'][10]))==shell(src(ann['cells'][22]))
    save('annotation_verification.json',dict(cells=42,code_indices=codes,derived_cells=85,schemas=3,AST=28,reading_AST=True,
        all_notes=True,metadata_historical_outputs_original_lines=True,markdown10_shell_commands=3,markdown31='incomplete overview preserved/comments only'))
    return raw['cells'],[ann['cells'][2+2*i] for i in range(42)],codes

class LocalLoad(ast.NodeTransformer):
    def visit_Constant(self,n):
        if n.value=='gpt2':return ast.Name(id='local_base',ctx=ast.Load())
        return n
def state(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}
def equal(a,b):return all(torch.equal(v,b[k]) for k,v in a.items())
def step_counter(trainer):
    calls=[];original=trainer.optimizer.step
    def counted(*args,**kwargs):
        result=original(*args,**kwargs);calls.append(1);return result
    trainer.optimizer.step=counted;return calls
def mask_audit(trainer,q,r):
    random_state=torch.get_rng_state();training=[(m,m.training) for m in trainer.model.modules()]
    try:
        inputs=trainer.prepare_model_inputs(q,r)
        with torch.no_grad():lp,_,values,mask=trainer.batched_forward_pass(trainer.model,q,r,inputs)
        return dict(input_ids=inputs['input_ids'].tolist(),input_shape=list(inputs['input_ids'].shape),logprob_shape=list(lp.shape),
                    value_shape=list(values.shape),mask_shape=list(mask.shape),mask=mask.tolist(),mask_count=int(mask.sum()),labels_removed='labels' not in inputs)
    finally:
        torch.set_rng_state(random_state)
        for m,mode in training:m.training=mode

def phase(cells,codes,base):
    ns=dict(local_base=str(base));out=io.StringIO();images={};calls=None;step={}
    old=Path.cwd();os.chdir(FOLDER)
    try:
        with contextlib.redirect_stdout(out):
            for i in codes:
                tree=ast.parse(src(cells[i]))
                if i==18:tree=LocalLoad().visit(tree)
                if i==28:
                    trainer=ns['ppo_trainer'];calls=step_counter(trainer)
                    before=state(trainer.model);reference=state(trainer.ref_model)
                    optids={id(p) for g in trainer.optimizer.param_groups for p in g['params']}
                    assert optids=={id(p) for p in trainer.model.parameters() if p.requires_grad}
                exec(compile(ast.fix_missing_locations(tree),'cell-'+str(i),'exec'),ns)
                if i in (3,7,8,12,14):
                    image=eval(compile(ast.Expression(tree.body[0].value),'image-cell-'+str(i),'eval'),ns)
                    assert image._repr_png_();images[str(i)]=dict(bytes=len(image.data),sha256=hashlib.sha256(image.data).hexdigest())
                if i==28:
                    after=state(trainer.model);refafter=state(trainer.ref_model)
                    assert len(calls)==4 and not equal(before,after) and equal(reference,refafter)
                    assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('v_head.'))
                    step=dict(optimizer_calls=len(calls),optimizer_class=type(trainer.optimizer.optimizer).__name__,active_changed=True,
                        value_head_changed=True,reference_unchanged=True,optimizer_only_active=True,
                        reference_requires_grad_count=sum(p.requires_grad for p in trainer.ref_model.parameters()),
                        reference_gradients_none=all(p.grad is None for p in trainer.ref_model.parameters()),
                        mask_audit=mask_audit(trainer,[ns['query_tensor'][0]],[ns['response_tensor'][0]]))
    finally:os.chdir(old)
    q=ns['query_tensor'][0];r=ns['response_tensor'][0]
    assert len(q)==7 and len(r)==27 and torch.equal(q,r[:7]) and len(ns['input_ids'])==34
    assert ns['last_hidden_state'].shape==(34,16) and ns['lm_logits'].shape==(34,50257) and ns['value'].shape==(34,)
    assert step['mask_audit']['mask_count']==27 and step['mask_audit']['input_shape']==[1,34]
    stats={k:float(ns['train_stats'][k]) for k in ['objective/kl','ppo/loss/policy','ppo/loss/value','ppo/learning_rate']}
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    result=dict(all_code_cells_executed=28,whole_unmodified_cells=27,variant18='three gpt2 identifiers replaced with local random checkpoint/tokenizer directory',
        query_ids=q.tolist(),response_ids=r.tolist(),full_response_text=ns['response_txt'],input_ids=ns['input_ids'].tolist(),
        hidden_shape=list(ns['last_hidden_state'].shape),logits_shape=list(ns['lm_logits'].shape),value_shape=list(ns['value'].shape),
        value_has_grad=ns['value'].requires_grad,images=images,step=step,stats=stats,config_seed=ns['config'].seed,
        policy_parameters=sum(p.numel() for p in ns['model'].parameters()),value_head_parameters=sum(p.numel() for p in ns['model'].v_head.parameters()),stdout=out.getvalue())
    return result,state(ns['model']),ns

def corrected(base,kwargs):
    base=str(base)  # TRL0.4.7 wrapper accepts str, not pathlib.Path.
    torch.manual_seed(42)
    active=AutoModelForCausalLMWithValueHead.from_pretrained(base);ref=AutoModelForCausalLMWithValueHead.from_pretrained(base)
    tokenizer=GPT2Tokenizer.from_pretrained(base);tokenizer.pad_token=tokenizer.eos_token
    trainer=PPOTrainer(PPOConfig(batch_size=1),active,ref,tokenizer);calls=step_counter(trainer)
    q=tokenizer.encode('This morning I went to the ',return_tensors='pt')[0]
    response=trainer.generate([q],return_prompt=False,**kwargs)
    assert len(response)==1 and len(response[0])==20
    snapshot=state(ref);stats=trainer.step([q],response,[torch.tensor(1.)])
    mask=mask_audit(trainer,[q],response)
    assert len(calls)==4 and equal(snapshot,state(ref)) and mask['mask_count']==20 and mask['input_shape']==[1,27]
    return dict(response_ids=response[0].tolist(),optimizer_calls=4,reference_unchanged=True,mask_audit=mask,
                stats={k:float(stats[k]) for k in ['objective/kl','ppo/loss/policy','ppo/loss/value']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer-json',type=Path,required=True);a=p.parse_args()
    assert sha(a.tokenizer_json)==TOKEN_SHA
    assert trl.__version__=='0.4.7' and transformers.__version__=='4.30.0'
    source=Path(inspect.getfile(PPOTrainer));assert sha(source)==PPO_SHA
    raw,ann,codes=audit();torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='lesson16-') as work:
        base=Path(work);tok=json.loads(a.tokenizer_json.read_bytes())
        (base/'vocab.json').write_bytes(json.dumps(tok['model']['vocab'],ensure_ascii=False).encode('utf-8'))
        (base/'merges.txt').write_bytes(('#version: 0.2\n'+'\n'.join(tok['model']['merges'])+'\n').encode('utf-8'))
        tokenizer=GPT2Tokenizer(vocab_file=str(base/'vocab.json'),merges_file=str(base/'merges.txt'));tokenizer.save_pretrained(base)
        assert tokenizer.encode('This morning I went to the ')==[1212,3329,314,1816,284,262,220]
        torch.manual_seed(42);model=GPT2LMHeadModel(GPT2Config(vocab_size=50257,n_positions=128,n_ctx=128,n_embd=16,n_layer=1,n_head=2))
        model.save_pretrained(base)
        original,s1,ns=phase(raw,codes,base);annotated,s2,_=phase(ann,codes,base)
        assert equal(s1,s2) and original['response_ids']==annotated['response_ids'] and original['images']==annotated['images']
        assert original['step']['mask_audit']==annotated['step']['mask_audit'] and original['stats']==annotated['stats']
        fixed=corrected(base,ns['generation_kwargs']);assert fixed['response_ids']==original['response_ids'][7:]
    scripts=[Path(__file__),ROOT/'复现/scripts/build_annotated.py',ROOT/'复现/scripts/download_original.py']
    report=dict(status='部分成功',command=[sys.executable,*sys.argv],python=sys.version,torch=torch.__version__,transformers=transformers.__version__,trl=trl.__version__,
        cuda_available=torch.cuda.is_available(),tokenizer_json_sha256=TOKEN_SHA,tokenizer_scope='real GPT2Tokenizer; vocab and merges exported from fixed original tokenizer.json',
        PPOTrainer_source=dict(path=str(source),sha256=PPO_SHA),scripts={str(q.relative_to(ROOT)):sha(q) for q in scripts},
        original=original,annotated=annotated,states_exact=True,generation_exact=True,numerical_stats_exact=True,corrected=fixed,
        not_run=['original GPT2 pretrained weights','Llama2/Guanaco SFT shell','learned reward model','full RLHF/human preference evaluation','GPU/multicard'])
    save('runtime_verification.json',report);print('42 notes/28 ASTs/3 schemas; both real PPO runs and corrected pure-response run each four optimizer updates passed.')
if __name__=='__main__':main()
