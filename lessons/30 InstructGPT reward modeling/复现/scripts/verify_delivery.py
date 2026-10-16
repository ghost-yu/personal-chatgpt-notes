"""全单元保留/执行；BT似然/梯度/数值稳定性、真实SciPy与小Llama训练。"""
import ast
import base64
import contextlib
import copy
import hashlib
import importlib.metadata as metadata
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import tokenize
import warnings

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'代码/tutorials/reward_model'
BACK=Path(os.environ.get('LESSON30_DATA','G:/zzz/Personal ChatGPT 系列/30 InstructGPT reward modeling/资料'))
sys.path.insert(0,str(BACK/'runtime-libs'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import transformers
from transformers.models.llama import modeling_llama as hf

MAIN='reward_model_Bradley_Terry_model'
SUP='bradley_terry_modeling'
REPORT={}
torch.set_num_threads(1)
torch.manual_seed(30)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(c):return ''.join(c['source'])
def tree(s):return ast.dump(ast.parse(s),include_attributes=False)
def tokens(s):return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline) if t.type not in [tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,tokenize.ENDMARKER,tokenize.ENCODING]]
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def state(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}
def same(a,b):return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
def err(fn):
    try:fn()
    except Exception as e:return {'exception':type(e).__name__,'message':str(e)[:300]}
    raise AssertionError('Expected actual error')

def audit():
    manifest=json.loads((ROOT/'来源清单.json').read_text(encoding='utf-8'))
    for p,h in manifest['files'].items():
        b=(ROOT/p).read_bytes();assert hashlib.sha256(b).hexdigest()==h
        assert b==subprocess.check_output(['git','-C','G:/zzz/Llama 入门教程/代码/personal_chatgpt','show',manifest['upstream_commit']+':'+p.removeprefix('代码/')])
    nbs={};stats={}
    for name,count,ncode in [(MAIN,27,12),(SUP,8,5)]:
        raw=json.loads((CODE/(name+'.ipynb')).read_text(encoding='utf-8'));assert len(raw['cells'])==count
        versions=[]
        for kind in ['原代码_带说明','代码注释']:
            path=CODE/((SUP+'_' if name==SUP else '')+kind+'.ipynb')
            nb=json.loads(path.read_text(encoding='utf-8'));assert len(nb['cells'])==2*count+1
            assert {k:v for k,v in raw.items() if k!='cells'}=={k:v for k,v in nb.items() if k!='cells'}
            for i,c in enumerate(raw['cells']):
                d=nb['cells'][2*i+2];assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in d.items() if k!='source'}
                if kind=='原代码_带说明' or c['cell_type']!='code':assert source(c)==source(d)
                else:assert tree(source(c))==tree(source(d)) and tokens(source(c))==tokens(source(d))
            versions.append(nb)
        for n in [raw,*versions]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',nbformat.warnings.MissingIDFieldWarning);nbformat.validate(nbformat.from_dict(copy.deepcopy(n)))
        code='\n'.join(source(c) for c in raw['cells'] if c['cell_type']=='code')
        assert tree(code)==tree((CODE/(name+'_代码注释.py')).read_text(encoding='utf-8'))
        assert sum(c['cell_type']=='code' for c in raw['cells'])==ncode
        stats[name]={'cells':count,'code_cells':ncode,'derived_cells':2*count+1,'schemas':3,'original_git_bytes':True,'metadata_historical_outputs':True,'ast_tokens':True,'reading_ast':True}
        nbs[name]=(raw,{'cells':versions[1]['cells'][2::2]})
    core=load_module('lecture30_official',CODE/'Llama奖励打分源码_代码注释.py')
    s=inspect.getsource(hf.LlamaForSequenceClassification)
    assert hashlib.sha256(s.encode()).hexdigest()==manifest['official_class_sha256']
    assert tree(s)==tree(inspect.getsource(core.LlamaForSequenceClassification))
    dist=metadata.distribution('transformers');p=next(x for x in dist.files if str(x).replace('\\','/')=='transformers/models/llama/modeling_llama.py')
    assert base64.urlsafe_b64encode(hashlib.sha256(Path(dist.locate_file(p)).read_bytes()).digest()).rstrip(b'=').decode()==p.hash.value
    REPORT['preservation']={'notebooks':stats,'schemas':6,'full_official_class_ast':True,'core_import':True,'installed_source_record':True}
    return nbs,core

def execute(nb,name):
    ns={'__name__':'__lecture30__'};trace=[];figures={};team=None
    previous=Path.cwd();os.chdir(CODE)
    try:
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            s=source(c);parsed=ast.parse(s);last=parsed.body.pop() if parsed.body and isinstance(parsed.body[-1],ast.Expr) else None
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(parsed,f'{name}_cell_{i}','exec'),ns)
                v=eval(compile(ast.Expression(last.value),f'{name}_cell_{i}','eval'),ns) if last else None
            display={'type':'Image','url':v.url} if v is not None and type(v).__name__=='Image' else repr(v) if v is not None else None
            trace.append({'cell':i,'status':'literal','stdout':out.getvalue(),'display':display})
            if name==MAIN and i in [14,23]:
                fig=plt.gcf();lines=plt.gca().lines
                data=[{'x':np.asarray(l.get_xdata()).tolist(),'y':np.asarray(l.get_ydata()).tolist()} for l in lines[:(1 if i==14 else 2)]]
                p=ROOT/'图片'/('log_odds.png' if i==14 else 'pair_loss_gradient.png');p.parent.mkdir(exist_ok=True)
                fig.savefig(p,dpi=150,bbox_inches='tight');figures[str(i)]={'arrays_sha256':hashlib.sha256(json.dumps(data).encode()).hexdigest(),'png_sha256':sha(p),'lines':len(data)};plt.close(fig)
            if name==SUP and i==2:
                team={'objective':ns['neg_log_likelihood'],'theta':ns['theta_full'].copy(),'weights':ns['p'].copy(),
                  'normalized':ns['p_normalized'].copy(),'w':ns['w'].copy(),'n':ns['n'].copy(),
                  'outer_iteration':ns['iteration']+1,'last_BFGS_nit':ns['result'].nit,
                  'break_result_difference':float(np.max(np.abs(ns['theta_opt']-ns['result'].x)))}
    finally:os.chdir(previous);plt.close('all')
    return ns,trace,figures,team

def run_original(nbs):
    namespaces={}
    for name,(raw,ann) in nbs.items():
        a,ta,fa,team_a=execute(raw,name);b,tb,fb,team_b=execute(ann,name)
        assert ta==tb and fa==fb
        if name==MAIN:
            assert len(a['p'])==100 and len(a['delta'])==1000
            assert np.allclose(a['log_odds'],np.log(a['p'])-np.log1p(-a['p']))
            assert np.allclose(a['neg_log_sigma'],np.logaddexp(0,-a['delta']))
            REPORT['literal_execution_main']={'code_cells':12,'original_annotated_trace_equal':True,'traces':ta,'figures':fa,
              'arrays_checked':True,'remote_image_content_checked':False}
        else:
            assert np.allclose(team_a['theta'],team_b['theta']) and np.allclose(a['log_pis_est'],b['log_pis_est'])
            REPORT['literal_execution_supplement']={'code_cells':5,'original_annotated_trace_equal':True,'traces':ta,
              'team_strengths':team_a['weights'].tolist(),'team_normalized':team_a['normalized'].tolist(),
              'brand_normalized':a['pis_est_normalized'].tolist(),'team_outer_iterations':team_a['outer_iteration'],
              'team_last_BFGS_internal_nit':team_a['last_BFGS_nit'],'team_break_stale_difference':team_a['break_result_difference'],
              'brand_success':bool(a['result'].success),'brand_BFGS_nit':int(a['result'].nit)}
        namespaces[name]=(a,team_a)
    return namespaces

def loss_checks(demo):
    delta=torch.linspace(-20,20,81,dtype=torch.double,requires_grad=True)
    loss=-F.logsigmoid(delta);bce=F.binary_cross_entropy_with_logits(delta,torch.ones_like(delta),reduction='none')
    soft=F.softplus(-delta);ce=F.cross_entropy(torch.stack([delta,torch.zeros_like(delta)],-1),torch.zeros(81,dtype=torch.long),reduction='none')
    assert torch.allclose(loss,bce,atol=1e-14) and torch.allclose(loss,soft,atol=1e-14) and torch.allclose(loss,ce,atol=1e-14)
    grad=torch.autograd.grad(loss.sum(),delta,create_graph=True)[0];hess=torch.autograd.grad(grad.sum(),delta)[0]
    assert torch.allclose(grad,delta.sigmoid()-1,atol=1e-15)
    assert torch.allclose(hess,delta.sigmoid()*(1-delta.sigmoid()),atol=1e-15)
    rw=torch.tensor([.5,-2.,3.],dtype=torch.double,requires_grad=True);rl=torch.tensor([0.,1.,-1.],dtype=torch.double,requires_grad=True)
    check=torch.autograd.gradcheck(lambda w,l:-F.logsigmoid(w-l).mean(),(rw,rl))
    sample=[]
    for x in [-6.,0.,1.,3.,10.]:
        t=torch.tensor(x,dtype=torch.double,requires_grad=True);l=F.softplus(-t);g=torch.autograd.grad(l,t)[0]
        sample.append({'delta':x,'loss':l.item(),'probability':t.sigmoid().item(),'gradient':g.item()})
    extreme=torch.tensor([-1000.,1000.],dtype=torch.double,requires_grad=True)
    stable=-F.logsigmoid(extreme);gs=torch.autograd.grad(stable.sum(),extreme)[0];naive=-torch.log(torch.sigmoid(extreme))
    assert torch.isfinite(stable).all() and torch.isfinite(gs).all() and not torch.isfinite(naive).all()
    assert len(demo.strict_pairs([0,0,1,2]))==5 and len(demo.strict_pairs(list(range(9))))==36
    r=torch.tensor([0.,.2,.5,1.,0.,1.,3.],dtype=torch.double,requires_grad=True);groups=[(0,4),(4,7)];ranks=[[0,0,1,2],[2,1,0]]
    c=torch.tensor([10.,-8.],dtype=torch.double,requires_grad=True);shift=torch.cat([c[0].expand(4),c[1].expand(3)])
    L=demo.grouped_pair_loss(r,groups,ranks);Ls=demo.grouped_pair_loss(r+shift,groups,ranks)
    assert torch.allclose(L,Ls,atol=1e-15);cg=torch.autograd.grad(Ls,c)[0];assert cg.abs().max()<1e-15
    flat=torch.cat([F.softplus(-torch.stack([r[s+w]-r[s+l] for w,l in demo.strict_pairs(rank)])) for (s,_),rank in zip(groups,ranks)])
    flatmean=flat.mean();assert not torch.allclose(L,flatmean)
    assert not torch.allclose(L,demo.grouped_pair_loss(2*r,groups,ranks))
    REPORT['loss_math']={'four_loss_forms_equal':True,'first_second_derivative_equal':True,'double_gradcheck':check,
      'points':sample,'equal_scores_ln_loss':float(np.log(2)),'equal_scores_log10_loss':float(np.log10(2)),
      'extreme_stable_losses':stable.tolist(),'extreme_stable_gradients':gs.tolist(),'naive_nonfinite_count':int((~torch.isfinite(naive)).sum()),
      'tie_example_strict_pairs':5,'K9_pairs':36,'K9_B64_max_pairs':2304,
      'group_prompt_mean':L.item(),'flat_pair_mean':flatmean.item(),'common_prompt_shift_invariant':True,
      'prompt_shift_gradients':cg.tolist(),'scale_not_invariant':True,
      'empty_comparison_error':err(lambda:demo.grouped_pair_loss(torch.zeros(2),[(0,2)],[[1,1]]))}

def scipy_checks(ns):
    a,team=ns[SUP];w=torch.tensor(team['w'],dtype=torch.double);theta=torch.tensor(team['theta'],dtype=torch.double)
    def stable_teams(t):
        return sum(-w[i,j]*F.logsigmoid(t[i]-t[j])-w[j,i]*F.logsigmoid(t[j]-t[i]) for i in range(4) for j in range(i+1,4))
    free=theta[1:].clone().requires_grad_();anchored=lambda v:stable_teams(torch.cat([torch.zeros(1,dtype=torch.double),v]))
    independent=anchored(free);original=team['objective'](free.detach().numpy());assert abs(independent.item()-original)<1e-12
    grad=torch.autograd.grad(independent,free)[0];assert grad.abs().max()<2e-5
    Hess=torch.autograd.functional.hessian(anchored,free);eigs=torch.linalg.eigvalsh(Hess);assert eigs.min()>0
    assert torch.allclose(stable_teams(theta+1000),stable_teams(theta),atol=1e-11)
    pair_prob=theta[:,None].sub(theta[None,:]).sigmoid();assert torch.allclose(pair_prob+pair_prob.T,torch.ones_like(pair_prob))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');huge=team['objective'](np.array([1000.,-1000.,500.]))
    assert not np.isfinite(huge)
    big_brand=a['neg_log_likelihood'](np.array([1000.,-1000.,500.,0.]));assert np.isfinite(big_brand)
    tb=torch.tensor(a['log_pis_est'],dtype=torch.double,requires_grad=True)
    def brand_loss(t):return sum(-wij*F.logsigmoid(t[i]-t[j])-wji*F.logsigmoid(t[j]-t[i]) for i,j,wij,wji in a['data'])
    independent_brand=brand_loss(tb);g=torch.autograd.grad(independent_brand,tb)[0]
    assert abs(independent_brand.item()-a['neg_log_likelihood'](tb.detach().numpy()[:-1]))<1e-10 and g.abs().max()<5e-5
    REPORT['scipy_likelihood']={'team_observations':int(w.sum()),'team_negative_log_likelihood':original,
      'team_independent_max_free_gradient':grad.abs().max().item(),'team_anchored_hessian_eigenvalues':eigs.tolist(),
      'team_pair_probabilities':pair_prob.tolist(),'brand_observations':1000,'brand_negative_log_likelihood':independent_brand.item(),
      'brand_independent_max_gradient':g.abs().max().item(),'naive_team_extreme_finite':bool(np.isfinite(huge)),
      'naive_team_warning_types':sorted(set(type(x.message).__name__ for x in caught)),
      'stable_brand_extreme_finite':True,'full_log_strength_common_shift_invariant':True}

def neural_checks(demo,core):
    model,ids,mask,groups,ranks=demo.make_experiment();model.eval();params=tuple(model.parameters())
    seen=[];hook=model.register_forward_pre_hook(lambda m,args,kwargs:seen.append(kwargs['input_ids'].shape[0]),with_kwargs=True)
    rewards=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.squeeze(-1)
    unique=demo.grouped_pair_loss(rewards,groups,ranks)
    gu=torch.autograd.grad(unique,params)
    indices=[];counts=[]
    for (start,_),rank in zip(groups,ranks):
        pairs=demo.strict_pairs(rank);counts.append(len(pairs))
        for w,l in pairs:indices.extend([start+w,start+l])
    ix=torch.tensor(indices);rep=model(input_ids=ids[ix],attention_mask=mask[ix],use_cache=False).logits.reshape(-1,2)
    perpair=F.softplus(-(rep[:,0]-rep[:,1]));duplicate=torch.stack([perpair[:counts[0]].mean(),perpair[counts[0]:].mean()]).mean()
    gd=torch.autograd.grad(duplicate,params);hook.remove()
    assert torch.allclose(unique,duplicate,atol=1e-7)
    gradient_error=max((a-b).abs().max().item() for a,b in zip(gu,gd));assert gradient_error<2e-6
    assert seen==[7,16]
    copy_model=core.LlamaForSequenceClassification(model.config).eval();copy_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        pred=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
        assert torch.equal(pred,copy_model(input_ids=ids,attention_mask=mask,use_cache=False).logits)
        hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        last=mask.sum(-1)-1;selected=model.score(hidden)[torch.arange(7),last]
        assert torch.equal(pred,selected)
        padded_ids=F.pad(ids,(0,2));padded_mask=F.pad(mask,(0,2))
        assert torch.allclose(pred,model(input_ids=padded_ids,attention_mask=padded_mask,use_cache=False).logits,atol=1e-7)
    init=state(model);trained,trace,data=demo.train_demo();trained.eval();final=state(trained)
    assert trace[-1]['loss_before_update']<trace[0]['loss_before_update']
    changed=[k for k,v in final.items() if not torch.equal(init[k],v)];assert 'score.weight' in changed and any('self_attn' in k for k in changed)
    ids,mask,groups,ranks=data
    with torch.no_grad():
        scores=trained(input_ids=ids,attention_mask=mask,use_cache=False).logits.squeeze(-1)
        last_loss=demo.grouped_pair_loss(scores,groups,ranks).item()
    with tempfile.TemporaryDirectory(prefix='lesson30_') as tmp:
        trained.save_pretrained(tmp);reloaded=hf.LlamaForSequenceClassification.from_pretrained(tmp,local_files_only=True).eval()
        assert same(trained.state_dict(),reloaded.state_dict())
        with torch.no_grad():assert torch.equal(scores,reloaded(input_ids=ids,attention_mask=mask,use_cache=False).logits.squeeze(-1))
    REPORT['real_reward_model']={'class':'LlamaForSequenceClassification','parameters':sum(p.numel() for p in trained.parameters()),
      'unique_sequence_batch_size':7,'strict_pair_counts':counts,'duplicated_sequence_batch_size':16,
      'grouped_reuse_loss':unique.item(),'duplicated_loss':duplicate.item(),'all_weight_gradient_max_error':gradient_error,
      'last_nonpad_head_selection_exact':True,'right_padding_append_invariant':True,'exported_complete_class_logits_exact':True,
      'optimizer':'AdamW','lr':0.005,'updates':8,'trace':trace,'loss_after_last_update':last_loss,
      'changed_state_entries':len(changed),'score_and_backbone_changed':True,'save_reload_state_logits_exact':True,
      'trained_rewards':scores.tolist(),'config':trained.config.to_dict(),
      'limitation':'Random model and self-made ID rankings only, not original6B RM or human preference generalization'}

def main():
    nbs,core=audit();ns=run_original(nbs);demo=load_module('lecture30_demo',CODE/'偏好损失_教学补充.py')
    loss_checks(demo);scipy_checks(ns);neural_checks(demo,core)
    import psutil
    REPORT['environment']={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
      'numpy':np.__version__,'matplotlib':matplotlib.__version__,'scipy':scipy.__version__,'transformers':transformers.__version__,
      'nbformat':nbformat.__version__,'cuda_available':torch.cuda.is_available(),'threads':torch.get_num_threads(),
      'physical_RAM_bytes':psutil.virtual_memory().total,'workspace_free_bytes':shutil.disk_usage(ROOT).free}
    REPORT['not_run']=['Original InstructGPT6B/GPT3 weights or data','Human annotation/quality/generalization',
      'Original fullRM training or PPO','Remote image contents','Private chatgpt.com conversation']
    REPORT['scripts_sha256']={p.name:sha(p) for p in (ROOT/'复现/scripts').glob('*.py')}
    REPORT['teaching_code_sha256']=sha(CODE/'偏好损失_教学补充.py')
    REPORT['status']='passed';(ROOT/'复现/outputs').mkdir(exist_ok=True)
    (ROOT/'复现/outputs/delivery_verification.json').write_text(json.dumps(REPORT,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in REPORT.items() if k not in ['literal_execution_main','literal_execution_supplement']},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
