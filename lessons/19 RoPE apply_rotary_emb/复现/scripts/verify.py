"""Pinned notebook + official RoPE methods, real CPU torch; no model weights."""
import argparse,ast,hashlib,json,os,platform,sys,tempfile,urllib.request
from pathlib import Path
from typing import Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import sentencepiece

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]


def source(path,url,hashes):
    data=path.read_bytes() if path else urllib.request.urlopen(url,timeout=30).read()
    assert hashlib.sha256(data).hexdigest() in hashes,'Source hash mismatch'
    return data


def reference(x,phase):
    c=phase.real[None,:,None,:];s=phase.imag[None,:,None,:]
    a=x[...,0::2];b=x[...,1::2]
    return torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(3)


def caught(f,kind):
    try:f()
    except kind as exc:return str(exc)
    raise AssertionError('Expected '+kind.__name__)


def main():
    p=argparse.ArgumentParser();p.add_argument('--notebook',type=Path);p.add_argument('--official-model',type=Path);p.add_argument('--tokenizer',type=Path);args=p.parse_args()
    m=json.loads((ROOT/'来源清单.json').read_text(encoding='utf-8'))
    base='https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/'+m['tutorial_commit']+'/'
    nb=json.loads(source(args.notebook,base+m['notebook'],[m['notebook_sha256'],m['notebook_upstream_sha256']]))
    model=source(args.official_model,'https://raw.githubusercontent.com/meta-llama/llama/'+m['meta_commit']+'/llama/model.py',[m['meta_model_sha256']]).decode('utf-8')
    tok=source(args.tokenizer,base+m['tokenizer'],[m['tokenizer_sha256']])
    state={};executed=[];old=Path.cwd()
    # ASCII temporary folder lets the original SentencePiece file-path cell run unchanged.
    with tempfile.TemporaryDirectory(prefix='rope19_') as tmp:
        Path(tmp,'tokenizer.model').write_bytes(tok)
        os.chdir(tmp)
        try:
            for i,c in enumerate(nb['cells']):
                if c['cell_type']=='code':
                    exec(compile(''.join(c['source']),'notebook cell '+str(i),'exec'),state);executed.append(i)
        finally:os.chdir(old)
    plt.close('all')
    np.testing.assert_allclose(state['R1']@state['R2'],state['R'](90),atol=1e-15)
    np.testing.assert_allclose(state['R1'].T@state['R2'],state['R'](30),atol=1e-15)
    token_ids=[state['sp_model'].bos_id(),*state['sp_model'].encode(state['prompt'])]
    assert token_ids==[1,306,4658,278,6593,310,2834,338]
    ns={'torch':torch,'Tuple':Tuple}
    tree=ast.parse(model)
    names=['precompute_freqs_cis','reshape_for_broadcast','apply_rotary_emb']
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(funcs)==3
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),'pinned Meta RoPE','exec'),ns)
    pre=ns['precompute_freqs_cis'];apply=ns['apply_rotary_emb'];broadcast=ns['reshape_for_broadcast']
    torch.manual_seed(19)
    q=torch.randn(2,4,4,8);k=torch.randn(2,4,2,8)
    table=pre(8,32);start=3;phase=table[start:start+4]
    missing=caught(lambda:state['apply_rotary_emb'](q,k,phase),NameError)
    state['reshape_for_broadcast']=broadcast
    notebook_q,notebook_k=state['apply_rotary_emb'](q,k,phase)
    before_q=q.clone();before_k=k.clone();before_phase=phase.clone()
    qr,kr=apply(q,k,phase)
    torch.testing.assert_close(notebook_q,qr);torch.testing.assert_close(notebook_k,kr)
    expected_q=reference(q,phase);expected_k=reference(k,phase)
    forward_error=float((qr-expected_q).abs().max())
    torch.testing.assert_close(qr,expected_q,atol=3e-7,rtol=1e-6)
    torch.testing.assert_close(kr,expected_k,atol=3e-7,rtol=1e-6)
    assert torch.equal(q,before_q) and torch.equal(k,before_k) and torch.equal(phase,before_phase)
    z=torch.view_as_complex(q.reshape(2,4,4,4,2));bc=broadcast(phase,z)
    assert bc.shape==(1,4,1,4)
    qg=q.clone().requires_grad_(True);kg=k.clone().requires_grad_(True)
    gq=torch.randn_like(q);gk=torch.randn_like(k)
    outq,outk=apply(qg,kg,phase);((outq*gq).sum()+(outk*gk).sum()).backward()
    grad_error=max(float((qg.grad-reference(gq,phase.conj())).abs().max()),float((kg.grad-reference(gk,phase.conj())).abs().max()))
    assert grad_error<1e-6
    # Independent block-diagonal float64 relative-position oracle using identical FP32 omega.
    omega=(1/(10000**(torch.arange(0,8,2).float()/8))).double()
    scores=qr[0,:,0]@kr[0,:,0].T
    oracle=torch.zeros(4,4,dtype=torch.float64)
    for mi in range(4):
        for ni in range(4):
            a=k[0,ni,0].double().reshape(4,2);ang=(ni-mi)*omega;c=ang.cos();s=ang.sin()
            rot=torch.stack((a[:,0]*c-a[:,1]*s,a[:,0]*s+a[:,1]*c),-1).flatten()
            oracle[mi,ni]=q[0,mi,0].double()@rot
    score_error=float((scores.double()-oracle).abs().max());assert score_error<3e-6
    shifted_q,shifted_k=apply(q,k,table[start+7:start+11])
    shift_error=float((scores-(shifted_q[0,:,0]@shifted_k[0,:,0].T)).abs().max());assert shift_error<3e-6
    restart_q,_=apply(q,k,table[:4]);offset_error=float((qr-restart_q).abs().max());assert offset_error>1e-2
    dtype_results=[]
    for dtype in (torch.float32,torch.float16,torch.bfloat16,torch.float64):
        qq=q.to(dtype);kk=k.to(dtype);oq,ok=apply(qq,kk,phase)
        expected=reference(qq.float(),phase).to(dtype)
        diff=float((oq.float()-expected.float()).abs().max())
        torch.testing.assert_close(oq,expected,atol=0.003 if dtype==torch.bfloat16 else .001,rtol=.001)
        assert oq.dtype==ok.dtype==dtype
        dtype_results.append({'input':str(dtype),'output':str(oq.dtype),'reference_max_error':diff,'norm_max_error':float((oq.float().norm(dim=-1)-qq.float().norm(dim=-1)).abs().max())})
    noncontig=torch.randn(2,4,4,16)[...,::2]
    stride_failure=caught(lambda:apply(noncontig,k,phase),RuntimeError)
    apply(noncontig.contiguous(),k,phase)
    odd_failure=caught(lambda:apply(torch.randn(2,4,4,7),k,phase),RuntimeError)
    shape_failure=caught(lambda:apply(q,k,table[:3]),AssertionError)
    hidden_q=torch.randn(1,3,3,8);hidden_phase=table[:3]
    complex_q=torch.view_as_complex(hidden_q.reshape(1,3,3,4,2))
    bad=torch.view_as_real(complex_q*hidden_phase).flatten(3)
    good,_=apply(hidden_q,hidden_q,hidden_phase)
    silent_error=float((bad-good).abs().max());assert silent_error>1e-2
    # Original curve: partial-phase sums, not a token attention score.
    xs=np.asarray(state['xs']);ys=np.asarray(state['ys']);assert len(xs)==256
    theta=10000**(-2*np.arange(64)/128)
    partial=np.cumsum(np.exp(1j*xs[:,None]*theta[None,:]),axis=1)
    independent=np.abs(partial[:,:63]).sum(axis=1)/64
    np.testing.assert_allclose(ys,independent,atol=1e-12)
    assert ys[0]==31.5
    increases=int(np.count_nonzero(np.diff(ys)>0));assert increases>0
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot(xs,ys);axes[0].set(xlabel='Relative distance',ylabel='Sum of partial magnitudes / 64',title='Notebook phase-sum statistic (d=128)')
    axes[1].plot(xs,np.cos(xs));axes[1].set(xlabel='Relative distance',ylabel='Unscaled dot product',title='Counterexample: q=k=(1,0), omega=1')
    for ax in axes:ax.grid(alpha=.3)
    fig.savefig(ROOT/'图片/phase_sum_and_counterexample.png',dpi=160);plt.close(fig)
    report={'status':'部分成功','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sentencepiece':sentencepiece.__version__,'device':'cpu','seed':19,'executed_code_cells':executed,'token_ids':token_ids,'original_missing_helper_error':missing,'official_functions':names,'query_shape':list(q.shape),'key_shape':list(k.shape),'phase_shape':list(phase.shape),'broadcast_shape':list(bc.shape),'forward_max_error':forward_error,'gradient_max_error':grad_error,'relative_score_max_error':score_error,'common_shift_max_error':shift_error,'restart_position_output_error':offset_error,'inputs_unchanged':True,'dtype_checks':dtype_results,'strided_input_error':stride_failure,'odd_dim_error':odd_failure,'wrong_phase_shape_error':shape_failure,'silent_wrong_broadcast_max_error':silent_error,'phase_statistic_at_zero':float(ys[0]),'phase_statistic_at_255':float(ys[-1]),'phase_statistic_increasing_steps':increases,'limitations':['No Llama model weights or complete Attention/FairScale execution','No CUDA, throughput or trained long-context quality measurement','Original notebook helper supplied only for the explicit follow-up rotation calls']}
    out=ROOT/'复现/outputs';(out/'教学验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'curve_data.json').write_text(json.dumps({'distance':xs.tolist(),'partial_sum_statistic':ys.tolist(),'single_pair_dot_counterexample':np.cos(xs).tolist()},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
