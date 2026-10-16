"""固定原件/完整源码静态审计；真实TRL小GPT2 PPO与独立数学/OpenRLHF核对。"""
import argparse
import ast
import base64
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import textwrap
import tokenize
import warnings

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'代码/tutorials/trl_hf'
BACK=Path('G:/zzz/Personal ChatGPT 系列/27 TRL RLHF PPOTrainer 与 OpenRLHF/资料')
UPSTREAM=Path('G:/zzz/Llama 入门教程/代码/personal_chatgpt')
LIB=Path('G:/zzz/Personal ChatGPT 系列/23 TRL Reward Model 与 RewardTrainer/资料/runtime-libs')
TOKEN=Path('G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/gpt2-tokenizer/tokenizer.json')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def source(c): return ''.join(c['source'])
def tree(s): return ast.dump(ast.parse(s))
def tokens(s): return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline) if t.type not in [tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,tokenize.ENCODING,tokenize.ENDMARKER]]
def state(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def same(a,b): return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)

def audit():
    import nbformat
    manifest=json.loads((ROOT/'来源清单.json').read_text('utf-8'))
    for path,expected in manifest['files'].items():
        p=ROOT/path;assert sha(p)==expected
        upstream=path.removeprefix('代码/')
        assert p.read_bytes()==subprocess.check_output(['git','show',manifest['upstream_commit']+':'+upstream],cwd=UPSTREAM)
    raw=json.loads((FOLDER/'trl_ppotrainer_helloworld.ipynb').read_bytes())
    variants=[json.loads((FOLDER/n).read_bytes()) for n in ['原代码_带说明.ipynb','代码注释.ipynb']]
    for n in [raw,*variants]:
        with warnings.catch_warnings(): warnings.simplefilter('ignore');nbformat.validate(nbformat.from_dict(copy.deepcopy(n)))
    for n,annotated in zip(variants,[False,True]):
        assert len(n['cells'])==67
        assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
        for i,c in enumerate(raw['cells']):
            note,saved=n['cells'][1+2*i:3+2*i]
            assert f'单元 {i}（' in source(note)
            assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in saved.items() if k!='source'}
            if annotated and c['cell_type']=='code':
                assert tree(source(c))==tree(source(saved));assert tokens(source(c))==tokens(source(saved))
            else:assert c==saved
    codes=[i for i,c in enumerate(raw['cells']) if c['cell_type']=='code']
    assert tree((FOLDER/'trl_ppotrainer_helloworld_代码注释.py').read_text('utf-8'))==tree('\n\n'.join(source(raw['cells'][i]) for i in codes))
    reading=ast.parse((FOLDER/'PPOTrainer核心源码_代码注释.py').read_text('utf-8'))
    reading_nodes={x.name:x for x in reading.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))}
    reading_text=(FOLDER/'PPOTrainer核心源码_代码注释.py').read_text('utf-8')
    record={row[0]:row[1] for row in csv.reader((LIB/'trl-0.8.6.dist-info/RECORD').open(encoding='utf-8'))}
    for rel,expected in manifest['trl_source_sha256'].items():
        p=BACK/'trl-0.8.6'/rel;assert sha(p)==expected==sha(LIB/rel)
        digest=base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).rstrip(b'=').decode()
        assert record[rel]=='sha256='+digest
        for node in ast.parse(p.read_text('utf-8')).body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in manifest['core_objects'][rel]:
                got=reading_nodes[node.name];assert ast.dump(got)==ast.dump(node)
                assert tokens(ast.get_source_segment(reading_text,got))==tokens(ast.get_source_segment(p.read_text('utf-8'),node))
    open_reading=ast.parse((FOLDER/'OpenRLHF核心源码_代码注释.py').read_text('utf-8'))
    open_nodes={x.name:x for x in open_reading.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))}
    for file,names in manifest['openrlhf_core_objects'].items():
        p=BACK/file;assert sha(p)==manifest['openrlhf_source_sha256'][file]
        for name in names:
            node=ast.parse(p.read_text('utf-8'))
            for part in name.split('.'):
                node=next(x for x in node.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==part)
            assert ast.dump(node)==ast.dump(open_nodes[name.split('.')[-1]])
    return raw['cells'],[variants[1]['cells'][2+2*i] for i in range(33)],dict(cells=33,code_cells=len(codes),schemas=3,derived_cells=67,raw_git_blobs=True,history_and_metadata=True,comment_only_ast_tokens=True,reading_ast=True,full_trl_objects=list(reading_nodes),trl_record_hashes=True,openrlhf_objects=list(open_nodes))

class LocalLoad(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=='gpt2': return ast.copy_location(ast.Constant(value=self.path),node)
        return node

def exec_display(s,ns):
    a=ast.parse(s);last=None
    if a.body and isinstance(a.body[-1],ast.Expr):last=a.body.pop().value
    exec(compile(a,'<notebook-cell>','exec'),ns)
    return eval(compile(ast.Expression(last),'<notebook-display>','eval'),ns) if last else None

def simplify(x):
    if torch.is_tensor(x):return {'dtype':str(x.dtype),'shape':list(x.shape),'value':simplify(x.detach().cpu().tolist())}
    if hasattr(x,'tolist'):return simplify(x.tolist())
    if isinstance(x,float) and x!=x:return 'NaN (observed undefined sample statistic)'
    if isinstance(x,(str,int,float,bool,type(None))):return x
    if isinstance(x,(list,tuple)):return [simplify(v) for v in x]
    return {'type':type(x).__name__}

def run(cells,local):
    import random
    import numpy as np
    random.seed(27);np.random.seed(27);torch.manual_seed(27)
    ns={};trace=[];calls=[];cache=[];before=None;ref_before=None;counter=[]
    for i,c in enumerate(cells):
        if c['cell_type']!='code':continue
        if i==0:
            from IPython.display import Image
            ns['Image']=Image;trace.append({'cell':i,'status':'skipped','reason':'proxy/NCCL environment; Image import provided externally'});continue
        text=source(c);variant=False
        if i in (7,12):
            transform=LocalLoad();transform.path=str(local);text=ast.unparse(ast.fix_missing_locations(transform.visit(ast.parse(text))));variant=True
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):value=exec_display(text,ns)
        trace.append({'cell':i,'status':'local CPU path variant' if variant else 'literal','stdout':buf.getvalue(),'display':simplify(value)})
        if i==15:
            trainer=ns['ppo_trainer'];before=state(ns['model']);ref_before=state(ns['model_ref'])
            original=trainer.compute_rewards
            def observed(scores,lp,rp,masks):
                result=original(scores,lp,rp,masks)
                cache.append({'scores':scores.detach().clone(),'old':lp.detach().clone(),'ref':rp.detach().clone(),'masks':masks.detach().clone(),'rewards':result[0].detach().clone()})
                return result
            trainer.compute_rewards=observed
            base_optimizer=trainer.optimizer.optimizer
            base_optimizer.register_step_post_hook(lambda *args: counter.append(len(counter)+1))
        if i==27:
            assert len(counter)==4
    trainer=ns['ppo_trainer'];active=state(ns['model']);ref=state(ns['model_ref'])
    assert not same(active,before);assert same(ref,ref_before)
    changed=[k for k in before if not torch.equal(before[k],active[k])]
    opt_ids={id(p) for group in trainer.optimizer.param_groups for p in group['params']}
    assert opt_ids=={id(p) for p in trainer.model.parameters() if p.requires_grad}
    assert not (opt_ids & {id(p) for p in trainer.ref_model.parameters()})
    # TRL state_dict保留基础模型原键，只有value head追加v_head前缀。
    assert any(k.startswith('v_head.') for k in changed)
    assert any(not k.startswith('v_head.') for k in changed)
    scores={k:simplify(v) for k,v in ns['train_stats'].items() if not k.startswith('time/')}
    return ns,active,ref,cache[0],dict(trace=trace,optimizer_updates=len(counter),parameter_count=sum(p.numel() for p in ns['model'].parameters()),changed_parameter_names=changed,reference_unchanged=True,reference_requires_grad=sorted({p.requires_grad for p in ns['model_ref'].parameters()}),optimizer_only_active=True,query_ids=ns['query_tensor'].tolist(),response_ids=ns['response_tensor'][0].tolist(),rollout={k:simplify(v) for k,v in cache[0].items()},stats_without_wall_clock=scores)

def load_open_objects():
    from typing import Optional,Tuple,Union
    import torch.nn as nn
    import torch.nn.functional as F
    namespace=dict(torch=torch,nn=nn,F=F,Optional=Optional,Tuple=Tuple,Union=Union)
    manifest=json.loads((ROOT/'来源清单.json').read_text('utf-8'))
    for file,names in manifest['openrlhf_core_objects'].items():
        text=(BACK/file).read_text('utf-8')
        for name in names:
            if name.endswith('make_experience'):continue
            node=ast.parse(text)
            for part in name.split('.'):
                node=next(x for x in node.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==part)
            # 原AST对象在提供真实torch命名空间内编译；不启动Ray、CUDA或DeepSpeed。
            exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),namespace)
    return namespace

def diagnostics(ns,cache):
    import math
    from trl.core import logprobs_from_logits,masked_mean,masked_var,masked_whiten
    from trl.trainer.utils import AdaptiveKLController
    t=ns['ppo_trainer'];q=[ns['query_tensor'][0]];r=ns['response_tensor'];o=load_open_objects();out={}
    inputs=t.prepare_model_inputs(q,r)
    assert torch.equal(inputs['input_ids'][0],torch.cat([q[0],r[0]]))
    with torch.no_grad():
        lp,logits,values,mask=t.batched_forward_pass(t.model,q,r,inputs,return_logits=True)
        direct_logits,_,direct_values=t.model(**inputs)
    assert lp.shape==values.shape==mask.shape==(1,inputs['input_ids'].shape[1]-1)
    assert mask.sum()==len(r[0]) and mask[0,:len(q[0])-1].sum()==0
    hand=direct_logits[:,:-1].log_softmax(-1).gather(-1,inputs['input_ids'][:,1:,None]).squeeze(-1)
    assert torch.equal(lp,hand) and torch.equal(values,direct_values[:,:-1])
    out['shape_shift_mask']={'query':list(q[0].shape),'response':list(r[0].shape),'input':list(inputs['input_ids'].shape),'logits':list(direct_logits.shape),'values':list(values.shape),'mask_indices':mask[0].nonzero().flatten().tolist(),'manual_logprob_exact':True,'no_duplicate_prompt':True}
    # 一次notebook step之后，当前/新rollout old相同，而固定ref已不同。
    with torch.no_grad(): ref_lp,*_=t.batched_forward_pass(t.ref_model,q,r,inputs)
    gap=float(((lp-ref_lp)*mask).abs().max());assert gap>0
    assert torch.equal(torch.exp(lp-lp),torch.ones_like(lp))
    out['old_vs_reference_after_update']={'new_rollout_ratio_exactly_one':True,'ref_logprob_max_difference':gap,'wrong_ref_denominator_ratio_max_difference':float((torch.exp(lp-ref_lp)-1).abs().max())}
    # 使用独立显式值：采样KL可以负，末token才加整句分数。
    old_config=t.config.kl_penalty;beta=t.kl_ctl.value;t.config.kl_penalty='kl';t.kl_ctl.value=.2
    logp=torch.tensor([[-.2,-1.,-.5,-.3]],dtype=torch.float64);refp=torch.tensor([[-.4,-.7,-.6,-.2]],dtype=torch.float64);m=torch.tensor([[0,1,1,0]])
    score=torch.tensor([1.],dtype=torch.float64);reward,non,kl=t.compute_rewards(score,logp,refp,m)
    expected=-.2*(logp-refp);expected[0,2]+=1
    torch.testing.assert_close(reward,expected);assert kl.min()<0
    open_reward,open_kl=o['compute_reward'](score,.2,logp,refp,m)
    torch.testing.assert_close(open_reward*m,reward*m)
    clipped_score=torch.tensor([20.],dtype=torch.float64)
    open_c,*_=o['compute_reward'](clipped_score,.2,logp,refp,m)
    trl_c,*_=t.compute_rewards(clipped_score,logp,refp,m)
    assert abs(float((trl_c-open_c)[0,2])-10)<1e-12
    p=torch.tensor([[[.25,.75]]],dtype=torch.float64);base=torch.tensor([[[.5,.5]]],dtype=torch.float64)
    t.config.kl_penalty='full';full=t._kl_penalty(p.log(),base.log());torch.testing.assert_close(full,(p*(p.log()-base.log())).sum(-1))
    out['kl_reward']={'sample_kl':kl.tolist(),'rewards':reward.tolist(),'full_kl':full.tolist(),'full_equals_manual':True,'OpenRLHF_valid_tokens_equal':True,'OpenRLHF_clamps_score_20_to_10':True}
    t.config.kl_penalty=old_config;t.kl_ctl.value=beta
    v=torch.tensor([[.2,.4,.1]],dtype=torch.float64);rew=torch.tensor([[0.,0.,1.]],dtype=torch.float64);m3=torch.ones_like(v)
    vv,adv,ret=t.compute_advantages(v,rew,m3)
    # gamma=1, lambda=.95 -> delta=[.2,-.3,.9], raw A=[.72725,.555,.9]
    raw=torch.tensor([[.72725,.555,.9]],dtype=torch.float64);torch.testing.assert_close(ret,raw+v)
    torch.testing.assert_close(adv,(raw-raw.mean())/torch.sqrt(raw.var(unbiased=True)+1e-8))
    oa,orr=o['get_advantages_and_returns'](None,v,rew,m3,1.,.95)
    torch.testing.assert_close(oa,raw);torch.testing.assert_close(orr,ret)
    out['GAE']={'delta':[.2,-.3,.9],'raw_advantages':raw.tolist(),'returns':ret.tolist(),'whitened_advantages':adv.tolist(),'targets_require_grad':adv.requires_grad,'OpenRLHF_raw_advantages_equal':True}
    # 正/负优势、概率比上/下界四个分支，直接调用真实TRL loss并反传。
    ratios=torch.tensor([[1.3,.7,1.3,.7]],dtype=torch.float64)
    new=ratios.log().requires_grad_();old=torch.zeros_like(new);a=torch.tensor([[1.,1.,-1.,-1.]],dtype=torch.float64)
    z=torch.zeros_like(new);vpred=torch.tensor([[.5,-.5,.1,-.1]],dtype=torch.float64,requires_grad=True);mm=torch.ones_like(new)
    pg,vf,stats=t.loss(old,z,torch.zeros(1,4,2,dtype=torch.float64),vpred,new,mm,a,z)
    manual_pg=-torch.minimum(ratios*a,ratios.clamp(.8,1.2)*a).mean();torch.testing.assert_close(pg,manual_pg)
    manual_vf=.1*.5*torch.maximum(vpred**2,vpred.clamp(-.2,.2)**2).mean();torch.testing.assert_close(vf,manual_vf)
    pg.backward(retain_graph=True);torch.testing.assert_close(new.grad,torch.tensor([[0.,-.175,.325,0.]],dtype=torch.float64))
    vf.backward();assert torch.isfinite(vpred.grad).all()
    policy=o['PolicyLoss'](clip_eps=.2);ov=o['ValueLoss'](clip_eps=.2)
    torch.testing.assert_close(policy(new.detach(),old,a,mm),pg.detach());torch.testing.assert_close(ov(vpred.detach(),z,z,mm),vf.detach()/.1)
    # 选新value接近target但跨旧value裁剪边界，max两支不等于普通MSE。
    close=torch.tensor([[.9,.9]],dtype=torch.float64);target=torch.ones_like(close);zero=torch.zeros_like(close)
    _,vf_clip,_=t.loss(zero,zero,torch.zeros(1,2,2,dtype=torch.float64),close,zero,torch.ones_like(close),target,target)
    assert float(vf_clip)>.1*.5*float(((close-target)**2).mean())
    out['clip_and_gradient']={'ratios':ratios.tolist(),'advantages':a.tolist(),'policy_loss':float(pg),'weighted_value_loss':float(vf),'d_policy_d_new_logprob':new.grad.tolist(),'OpenRLHF_policy_and_value_equal':True,'value_clip_pessimistic_example':float(vf_clip)}
    ctl=AdaptiveKLController(.2,6,10000);ctl.update(12,100);assert math.isclose(ctl.value,.2004);high=ctl.value;ctl.update(0,100)
    out['controller']={'highKL_beta':high,'then_lowKL_beta':ctl.value,'relative_error_clipped':True}
    # 变长/左右padding核对，无需改变模型或训练数据。
    padding={}
    for side in ['left','right']:
        ns['tokenizer'].padding_side=side;qs=[q[0],q[0][:-2]];rs=[r[0][:5],r[0][:3]]
        batch=t.prepare_model_inputs(qs,rs)
        with torch.no_grad():blp,_,bval,bm=t.batched_forward_pass(t.model,qs,rs,batch)
        assert bm.sum(-1).tolist()==[5,3]
        errors=[]
        for j in range(2):
            one=t.prepare_model_inputs([qs[j]],[rs[j]])
            with torch.no_grad():slp,_,_,sm=t.batched_forward_pass(t.model,[qs[j]],[rs[j]],one)
            errors.append(float((blp[j][bm[j].bool()]-slp[0][sm[0].bool()]).abs().max()))
        padding[side]={'mask_counts':bm.sum(-1).tolist(),'batch_vs_single_max_errors':errors}
        if side=='left':
            corrected=dict(batch);corrected['position_ids']=batch['attention_mask'].long().cumsum(-1)-1
            corrected['position_ids'].masked_fill_(batch['attention_mask']==0,0)
            with torch.no_grad():fixed,_,_,fm=t.batched_forward_pass(t.model,qs,rs,corrected)
            fixed_errors=[]
            for j in range(2):
                one=t.prepare_model_inputs([qs[j]],[rs[j]])
                with torch.no_grad():single,_,_,single_mask=t.batched_forward_pass(t.model,[qs[j]],[rs[j]],one)
                err=float((fixed[j][fm[j].bool()]-single[0][single_mask[0].bool()]).abs().max());assert err<2e-6;fixed_errors.append(err)
            padding[side]['independent_position_ids_fix_errors']=fixed_errors
    ns['tokenizer'].padding_side='right'
    out['variable_length_padding']=padding
    zz=torch.zeros(2,2,dtype=torch.float64);different_lengths=torch.tensor([[1.,1.],[1.,0.]],dtype=torch.float64)
    aa=torch.tensor([[-1.,-3.],[-9.,0.]],dtype=torch.float64)
    global_loss,*_=t.loss(zz,zz,torch.zeros(2,2,2,dtype=torch.float64),zz,zz,different_lengths,aa,zz)
    sequence_loss=policy(zz,zz,aa,different_lengths)
    assert math.isclose(float(global_loss),13/3) and math.isclose(float(sequence_loss),5.5)
    out['framework_mask_reduction_difference']={'TRL_global_token_mean':float(global_loss),'OpenRLHF_sequence_then_batch_mean':float(sequence_loss),'unequal_response_lengths':[2,1]}
    # 实际错误与旧版边界，不把NaN包装成通过的训练。
    failures={}
    for name,fn in [('zero_response_mask',lambda:t.compute_rewards(torch.ones(1),torch.zeros(1,2),torch.zeros(1,2),torch.zeros(1,2))),('scores_plain_float',lambda:t._step_safety_checker(1,q,r,[1.])),('one_valid_token_variance',lambda:masked_var(torch.ones(1,1),torch.ones(1,1)))]:
        try:
            value=fn();failures[name]={'returned':simplify(value),'finite':bool(torch.isfinite(value).all()) if torch.is_tensor(value) else None}
        except Exception as e:failures[name]={'exception':type(e).__name__,'message':str(e)}
    assert failures['one_valid_token_variance']['finite'] is False
    assert 'exception' in failures['zero_response_mask'] and 'exception' in failures['scores_plain_float']
    out['observed_boundaries']=failures
    return out

def main():
    global torch,BACK,UPSTREAM,LIB,TOKEN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=BACK,help='本讲资料，含trl-0.8.6及OpenRLHF固定源码')
    parser.add_argument('--upstream',type=Path,default=UPSTREAM,help='固定提交仍可访问的课程Git仓库')
    parser.add_argument('--libs',type=Path,default=LIB,help='含trl与trl-0.8.6.dist-info的实际安装目录')
    parser.add_argument('--tokenizer-json',type=Path,default=TOKEN,help='真实GPT2 tokenizer.json')
    args=parser.parse_args();BACK=args.source_dir;UPSTREAM=args.upstream;LIB=args.libs;TOKEN=args.tokenizer_json
    for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY']:os.environ.pop(k,None)
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,str(LIB))
    import torch
    import transformers,trl,accelerate,datasets
    torch.set_num_threads(1)
    assert trl.__version__=='0.8.6'
    raw,annotated,static=audit()
    from tokenizers import Tokenizer
    from transformers import GPT2Config,GPT2LMHeadModel,GPT2Tokenizer
    with tempfile.TemporaryDirectory(prefix='lesson27-') as tmp:
        local=Path(tmp)
        Tokenizer.from_file(str(TOKEN)).model.save(str(local))
        tok=GPT2Tokenizer(vocab_file=str(local/'vocab.json'),merges_file=str(local/'merges.txt'));tok.save_pretrained(str(local))
        torch.manual_seed(270)
        config=GPT2Config(vocab_size=50257,n_positions=64,n_ctx=64,n_embd=16,n_layer=1,n_head=2,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,summary_dropout_prob=0.,bos_token_id=50256,eos_token_id=50256)
        GPT2LMHeadModel(config).save_pretrained(str(local))
        a,sa,ra,ca,result_a=run(raw,local)
        b,sb,rb,cb,result_b=run(annotated,local)
        assert same(sa,sb) and same(ra,rb)
        assert result_a==result_b
        diag=diagnostics(b,cb)
        from trl import AutoModelForCausalLMWithValueHead
        saved=local/'trained';b['model'].save_pretrained(str(saved));b['tokenizer'].save_pretrained(str(saved))
        reloaded=AutoModelForCausalLMWithValueHead.from_pretrained(str(saved));assert same(state(reloaded),state(b['model']))
        reloaded.eval();b['model'].eval()
        check_input=b['ppo_trainer'].prepare_model_inputs([b['query_tensor'][0]],b['response_tensor'])
        with torch.no_grad():
            left=b['model'](**check_input);right=reloaded(**check_input)
        assert torch.equal(left[0],right[0]) and torch.equal(left[2],right[2])
        diag['checkpoint']={'all_base_and_value_head_state_exact':True,'reloaded_logits_and_values_exact':True,'temporary_checkpoint_not_published':True}
        report={'status':'passed','preservation':static,'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,'trl':trl.__version__,'accelerate':accelerate.__version__,'datasets':datasets.__version__,'cuda_available':torch.cuda.is_available(),'torch_threads':1,'external_seed':27,'checkpoint_seed':270,'random_gpt2_config':config.to_dict(),'tokenizer_json_sha256':sha(TOKEN)},'original':result_a,'annotated':result_b,'states_and_non_timing_outputs_exactly_equal':True,'diagnostics':diag,'not_run':['Original GPT2 pretrained checkpoint and original natural language quality','OpenRLHF Ray/DeepSpeed/CUDA distributed training or make_experience','Learned reward model, preference dataset or human evaluation','PPO paper Atari/MuJoCo or large scale speed benchmark'],'scripts_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
        (ROOT/'复现/outputs/delivery_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps({'status':report['status'],'cells':static,'parameters':result_a['parameter_count'],'updates_each':result_a['optimizer_updates'],'shape':diag['shape_shift_mask'],'boundaries':diag['observed_boundaries']},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
