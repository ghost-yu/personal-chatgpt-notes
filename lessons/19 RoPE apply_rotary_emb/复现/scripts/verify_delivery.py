# -*- coding: utf-8 -*-
"""Real notebook execution and complete fixed Meta module, including missing-helper audit."""
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import tokenize
import types
from pathlib import Path

os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import sentencepiece
import torch

ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'代码/tutorials';OUTPUT=ROOT/'复现/outputs'
META=Path('G:/zzz/Personal ChatGPT 系列/19 RoPE apply_rotary_emb/资料/llama-model.py')
sys.path.insert(0,'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local')

def sha(b):return hashlib.sha256(b).hexdigest()

def tokens(s):
    return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline)
        if t.type not in (tokenize.COMMENT,tokenize.NL)]

def without_docs(node):
    node=copy.deepcopy(node)
    if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
    return node

def audit():
    m=json.loads((ROOT/'来源清单.json').read_bytes())
    data=(FOLDER/'llama2_src_03_RoPE_apply_rotary_emb.ipynb').read_bytes()
    assert sha(data)==m['notebook_upstream_sha256']==m['notebook_sha256']
    assert sha((FOLDER/'tokenizer.model').read_bytes())==m['tokenizer_sha256']
    raw=json.loads(data);ann=json.loads((FOLDER/'代码注释.ipynb').read_bytes());exp=json.loads((FOLDER/'原代码_带说明.ipynb').read_bytes())
    assert len(raw['cells'])==27 and len(ann['cells'])==len(exp['cells'])==55
    from build_delivery import NOTES
    assert set(NOTES)==set(range(27))
    for n in [raw,ann,exp]:
        nbformat.validate(nbformat.from_dict(n))
        assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
    ids=[]
    for i,c in enumerate(raw['cells']):
        a=ann['cells'][2+2*i]
        assert exp['cells'][2+2*i]==c
        assert {k:v for k,v in a.items() if k!='source'}=={k:v for k,v in c.items() if k!='source'}
        s=''.join(c['source']);t=''.join(a['source'])
        if c['cell_type']=='code':
            ids.append(i);assert tokens(s)==tokens(t)
            assert ast.dump(ast.parse(s))==ast.dump(ast.parse(t))
            assert '# ' in t
        else:assert s==t
    assert len(ids)==15
    joined='\n'.join(''.join(raw['cells'][i]['source']) for i in ids)
    reading=(FOLDER/'llama2_src_03_RoPE_apply_rotary_emb_代码注释.py').read_text(encoding='utf-8')
    assert ast.dump(ast.parse(joined))==ast.dump(ast.parse(reading))
    official_tree=ast.parse(META.read_bytes())
    reading_tree=ast.parse((ROOT/'代码/RoPE_官方函数阅读.py').read_bytes())
    names=['precompute_freqs_cis','reshape_for_broadcast','apply_rotary_emb']
    for name in names:
        a=next(n for n in official_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in reading_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(without_docs(a))==ast.dump(without_docs(b))
    a=next(n for n in official_tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_rotary_emb')
    b=ast.parse(''.join(raw['cells'][16]['source'])).body[0]
    assert ast.dump(without_docs(a))==ast.dump(b)
    return raw,ann,dict(original_bytes=len(data),sha256=sha(data),cells=27,code_ids=ids,derived_cells=55,
        schemas=3,comment_only_ast_tokens=True,all_history_metadata_execution_preserved=True,reading_ast=True,
        three_official_reading_functions_execution_ast=True,notebook_apply_matches_official_execution_ast=True)

def summarize(x):
    if isinstance(x,torch.Tensor):a=x.detach().cpu().numpy()
    elif isinstance(x,np.ndarray):a=x
    elif isinstance(x,list) and x and isinstance(x[0],matplotlib.lines.Line2D):
        return [dict(x=summarize(i.get_xdata()),y=summarize(i.get_ydata())) for i in x]
    elif isinstance(x,list):return [summarize(i) for i in x]
    elif isinstance(x,(int,float,np.integer,np.floating)):return float(x)
    else:return None
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=sha(a.tobytes()))

def execute(n,annotated=False):
    plt.close('all');scope={'__name__':'lesson19_notebook'};trace=[]
    previous=Path.cwd();os.chdir(FOLDER)
    try:
        for i in range(27):
            c=n['cells'][2+2*i] if annotated else n['cells'][i]
            if c['cell_type']!='code':continue
            tree=ast.parse(''.join(c['source']));result=None;captured=io.StringIO()
            with contextlib.redirect_stdout(captured):
                if tree.body and isinstance(tree.body[-1],ast.Expr):
                    exec(compile(ast.Module(body=tree.body[:-1],type_ignores=[]),'cell-'+str(i),'exec'),scope)
                    result=eval(compile(ast.Expression(tree.body[-1].value),'cell-'+str(i),'eval'),scope)
                else:exec(compile(tree,'cell-'+str(i),'exec'),scope)
            trace.append(dict(cell=i,stdout=captured.getvalue(),result=summarize(result),
                variables={k:summarize(scope[k]) for k in ['R1','R2','xs','ys'] if k in scope}))
            if i==25:plt.savefig(OUTPUT/'notebook_phase_curve.png')
    finally:os.chdir(previous)
    ids=[scope['sp_model'].bos_id(),*scope['sp_model'].encode(scope['prompt'])]
    assert ids==[1,306,4658,278,6593,310,2834,338]
    np.testing.assert_allclose(scope['R1']@scope['R2'],scope['R'](90),rtol=0,atol=1e-15)
    np.testing.assert_allclose(scope['R1'].T@scope['R2'],scope['R'](30),rtol=0,atol=1e-15)
    return trace,scope,ids

def official():
    data=META.read_bytes();assert sha(data)=='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    m=types.ModuleType('lesson19_official_full_model');m.__file__=str(META);sys.modules[m.__name__]=m
    exec(compile(data,str(META),'exec'),m.__dict__)
    return m

def expect_error(fn,kind):
    try:fn()
    except kind as exc:return type(exc).__name__+': '+str(exc)
    raise AssertionError('Expected '+kind.__name__)

def reference(x,phase):
    c=phase.real[None,:,None,:];s=phase.imag[None,:,None,:]
    a=x[...,0::2];b=x[...,1::2]
    return torch.stack([a*c-b*s,a*s+b*c],dim=-1).flatten(3)

def check_rotary(mod,scopes):
    torch.manual_seed(19)
    q=torch.randn(2,4,4,8);k=torch.randn(2,4,2,8)
    table=mod.precompute_freqs_cis(8,32);phase=table[3:7];originals=[q.clone(),k.clone(),phase.clone()]
    missing=[];outputs=[];gradients=[]
    gq=torch.randn_like(q);gk=torch.randn_like(k)
    for scope in scopes:
        assert 'reshape_for_broadcast' not in scope
        missing.append(expect_error(lambda:scope['apply_rotary_emb'](q,k,phase),NameError))
        # Explicit supplement after all original cells, not a hidden notebook modification.
        scope['reshape_for_broadcast']=mod.reshape_for_broadcast
        qg=q.clone().requires_grad_(True);kg=k.clone().requires_grad_(True)
        oq,ok=scope['apply_rotary_emb'](qg,kg,phase)
        ((oq*gq).sum()+(ok*gk).sum()).backward()
        outputs.append((oq.detach(),ok.detach()));gradients.append((qg.grad,kg.grad))
    assert missing[0]==missing[1]
    assert all(torch.equal(a,b) for a,b in zip(outputs[0],outputs[1]))
    assert all(torch.equal(a,b) for a,b in zip(gradients[0],gradients[1]))
    qg=q.clone().requires_grad_(True);kg=k.clone().requires_grad_(True)
    oq,ok=mod.apply_rotary_emb(qg,kg,phase)
    ((oq*gq).sum()+(ok*gk).sum()).backward()
    assert all(torch.equal(a,b) for a,b in zip(outputs[0],(oq.detach(),ok.detach())))
    assert all(torch.equal(a,b) for a,b in zip(gradients[0],(qg.grad,kg.grad)))
    forward_error=max(float((oq.detach()-reference(q,phase)).abs().max()),float((ok.detach()-reference(k,phase)).abs().max()))
    grad_error=max(float((qg.grad-reference(gq,phase.conj())).abs().max()),float((kg.grad-reference(gk,phase.conj())).abs().max()))
    assert forward_error<1e-6 and grad_error<1e-6
    assert all(torch.equal(a,b) for a,b in zip(originals,(q,k,phase)))
    z=torch.view_as_complex(q.reshape(2,4,4,4,2));bc=mod.reshape_for_broadcast(phase,z)
    assert list(bc.shape)==[1,4,1,4] and bc.untyped_storage().data_ptr()==phase.untyped_storage().data_ptr()
    dtypes=[]
    for dtype in [torch.float32,torch.float16,torch.bfloat16,torch.float64]:
        qq=q.detach().clone().to(dtype).requires_grad_(True);kk=k.detach().clone().to(dtype).requires_grad_(True)
        rq,rk=mod.apply_rotary_emb(qq,kk,phase)
        error=max(float((rq-reference(qq.float(),phase).to(dtype)).abs().max()),float((rk-reference(kk.float(),phase).to(dtype)).abs().max()))
        assert error<1e-6 and rq.dtype==rk.dtype==dtype
        (rq.float().sum()+rk.float().sum()).backward()
        assert torch.isfinite(qq.grad).all() and torch.isfinite(kk.grad).all()
        dtypes.append(dict(input=str(dtype),output=str(rq.dtype),reference_max_error=error,
            norm_max_error=float((rq.float().norm(dim=-1)-qq.detach().float().norm(dim=-1)).abs().max()),finite_backward=True))
    double_shift=q.double()+1e-10
    double_out,_=mod.apply_rotary_emb(double_shift,k.double(),phase)
    assert torch.equal(double_out,oq.detach().double())
    bad=torch.randn(2,4,4,16)[...,::2]
    stride_error=expect_error(lambda:mod.apply_rotary_emb(bad,k,phase),RuntimeError)
    fixed,_=mod.apply_rotary_emb(bad.contiguous(),k,phase)
    torch.testing.assert_close(fixed,reference(bad,phase),rtol=1e-6,atol=1e-7)
    valid_noncontig=torch.randn(2,4,8,8)[:,:,::2,:]
    assert not valid_noncontig.is_contiguous()
    valid,_=mod.apply_rotary_emb(valid_noncontig,k,phase)
    torch.testing.assert_close(valid,reference(valid_noncontig,phase),rtol=1e-6,atol=1e-7)
    odd_error=expect_error(lambda:mod.apply_rotary_emb(torch.randn(2,4,4,7),k,phase),RuntimeError)
    phase_error=expect_error(lambda:mod.apply_rotary_emb(q,k,table[:3]),AssertionError)
    hq=torch.randn(1,3,3,8);hp=table[:3];hz=torch.view_as_complex(hq.reshape(1,3,3,4,2))
    wrong=torch.view_as_real(hz*hp).flatten(3);correct,_=mod.apply_rotary_emb(hq,hq,hp)
    silent_error=float((wrong-correct).abs().max());assert silent_error>.01
    short_k=k[:,:1]
    _,short_output=mod.apply_rotary_emb(q,short_k,phase)
    assert list(short_output.shape)==[2,4,2,8] and short_output.shape!=short_k.shape
    rank3=q[:,:,0,:]
    rank3_out,_=mod.apply_rotary_emb(rank3,rank3,phase)
    assert list(rank3_out.shape)==[2,4,4,2]
    omega=(1/(10000**(torch.arange(0,8,2).float()/8))).double()
    scores=oq.detach()[0,:,0]@ok.detach()[0,:,0].T;oracle=torch.zeros(4,4,dtype=torch.float64)
    for m in range(4):
        for n in range(4):
            a=k[0,n,0].double().reshape(4,2);angle=(n-m)*omega
            rot=torch.stack([a[:,0]*angle.cos()-a[:,1]*angle.sin(),a[:,0]*angle.sin()+a[:,1]*angle.cos()],-1).flatten()
            oracle[m,n]=q[0,m,0].double()@rot
    relative_error=float((scores.double()-oracle).abs().max());assert relative_error<3e-6
    shifted_q,shifted_k=mod.apply_rotary_emb(q,k,table[10:14])
    shift_error=float((scores-shifted_q[0,:,0]@shifted_k[0,:,0].T).abs().max());assert shift_error<3e-6
    restarted,_=mod.apply_rotary_emb(q,k,table[:4])
    restart_error=float((oq.detach()-restarted).abs().max());assert restart_error>.01
    return dict(full_official_module_unmodified=True,seed=19,q_shape=list(q.shape),k_shape=list(k.shape),phase_shape=list(phase.shape),
        original_and_annotated_missing_helper_errors=missing,official_helper_supplied_only_after_notebook_execution=True,
        original_annotated_official_outputs_and_gradients_exact_equal=True,independent_forward_max_error=forward_error,
        independent_inverse_rotation_gradient_max_error=grad_error,inputs_unchanged=True,broadcast_shape=list(bc.shape),broadcast_shares_storage=True,
        dtype_checks=dtypes,fp64_1e_minus10_change_lost_in_fp32_cast=True,strided_last_axis_error=stride_error,
        contiguous_repair_passed=True,valid_noncontiguous_input_passed=True,odd_dim_error=odd_error,wrong_phase_shape_error=phase_error,
        silent_wrong_head_broadcast_max_error=silent_error,unchecked_key_length=dict(input_shape=list(short_k.shape),output_shape=list(short_output.shape)),
        unsupported_rank3=dict(input_shape=list(rank3.shape),output_shape=list(rank3_out.shape)),relative_score_max_error=relative_error,
        common_shift_score_max_error=shift_error,restarting_position_changes_output_by=restart_error,
        limits='only three RoPE functions and teaching inputs; not full Attention/Transformer execution')

def curves(scope):
    xs=np.asarray(scope['xs']);ys=np.asarray(scope['ys'])
    omega=10000**(-2*np.arange(64)/128)
    independent=np.abs(np.cumsum(np.exp(1j*xs[:,None]*omega),axis=1)[:,:63]).sum(axis=1)/64
    error=float(np.max(np.abs(independent-ys)));assert error<1e-12 and ys[0]==31.5
    increases=int(np.count_nonzero(np.diff(ys)>0));assert increases==113
    # Changing only caller global d does not alter the function's bound default.
    scope['d']=64
    mismatched=float(scope['s'](0)/(scope['d']//2))
    explicit=float(scope['s'](0,d=64)/(64//2));scope['d']=128
    assert mismatched==63 and explicit==15.5
    return dict(points=len(xs),independent_cumsum_max_abs_error=error,at_zero=float(ys[0]),at_255=float(ys[-1]),
        increasing_steps=increases,global64_without_passing_to_s_at0=mismatched,explicit_s_d64_at0=explicit,
        meaning='partial-phase-sum statistic; no Q/K content or attention probabilities')

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    raw,ann,audit_result=audit()
    trace1,s1,ids1=execute(raw);trace2,s2,ids2=execute(ann,True)
    assert trace1==trace2 and len(trace1)==15 and ids1==ids2
    rotary=check_rotary(official(),[s1,s2]);curve=curves(s1)
    artifacts={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted((ROOT/'复现/scripts').glob('*.py'))
        if p.name in ('download_original.py','build_delivery.py','verify_delivery.py')}
    report=dict(status='passed',environment=dict(os=platform.platform(),python=sys.version,torch=torch.__version__,numpy=np.__version__,
        matplotlib=matplotlib.__version__,nbformat=nbformat.__version__,sentencepiece=sentencepiece.__version__,fairscale=importlib.metadata.version('fairscale'),
        cuda_available=torch.cuda.is_available()),audit=audit_result,execution=dict(original_and_annotated_each=15,all_traces_exact_equal=True,
        cwd=str(FOLDER),token_ids=ids1,original_helper_not_defined_or_called=True,traces=trace1),rotary=rotary,curve=curve,
        official_source_sha256=sha(META.read_bytes()),verification_script_sha256=artifacts,
        official_reading_sha256=sha((ROOT/'代码/RoPE_官方函数阅读.py').read_bytes()),
        limitations=['No Llama checkpoint or complete Attention/Transformer execution','No KV cache/repeat_kv or model parallel communication',
            'No GPU kernels, throughput or trained long-context quality','No RoFormer full training/metrics',
            'Missing helper supplied only for explicit supplemental calls after original execution','Historical outputs preserved; new curve separate'])
    (OUTPUT/'完整交付验证.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(status='passed',cells=27,code=15,schemas=3,forward_error=rotary['independent_forward_max_error'],
        gradient_error=rotary['independent_inverse_rotation_gradient_max_error'],relative_score_error=rotary['relative_score_max_error'],token_ids=ids1),ensure_ascii=True))

if __name__=='__main__':main()
