# -*- coding: utf-8 -*-
"""Audit original preservation and run genuine CPU Torch/FairScale modules."""
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import sys
import tempfile
import tokenize
import types
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / '代码/tutorials'
OUTPUT = ROOT / '复现/outputs'
META = Path('G:/zzz/Personal ChatGPT 系列/17 RMSNorm 与 SwiGLU/资料/llama-model.py')
FAIRSCALE = Path('G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local')
sys.path.insert(0, str(FAIRSCALE))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def tokens(s):
    return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline)
            if t.type not in (tokenize.COMMENT,tokenize.NL)]

def tensor_summary(x):
    if isinstance(x,torch.Tensor):
        a=x.detach().cpu().numpy()
    elif isinstance(x,np.ndarray):
        a=x
    else:
        return None
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=sha(a.tobytes()))

def audit():
    data=(FOLDER/'llama2_src_02_RMSNorm_SwiGLU.ipynb').read_bytes()
    assert sha(data)=='a90ee5e811cf42d3a3f3496f0d6449d7def728f497a57525b8bfe6bfd834a61f'
    raw=json.loads(data)
    annotated=json.loads((FOLDER/'代码注释.ipynb').read_bytes())
    explained=json.loads((FOLDER/'原代码_带说明.ipynb').read_bytes())
    assert len(raw['cells'])==28 and len(annotated['cells'])==len(explained['cells'])==57
    from build_annotated import NOTES
    assert set(NOTES)==set(range(28))
    for n in [raw,annotated,explained]:
        nbformat.validate(nbformat.from_dict(n))
        assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
    code_ids=[]
    for i,c in enumerate(raw['cells']):
        a=annotated['cells'][2+2*i]
        e=explained['cells'][2+2*i]
        assert c==e
        assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in a.items() if k!='source'}
        s=''.join(c['source']);t=''.join(a['source'])
        if c['cell_type']=='code':
            code_ids.append(i)
            assert ast.dump(ast.parse(s))==ast.dump(ast.parse(t))
            assert tokens(s)==tokens(t), ('not comment-only',i)
            assert '# ' in t
        elif i==3:
            sb=s.split('```')[1];tb=t.split('```')[1]
            assert tokens(sb)==tokens(tb)
            assert ast.dump(ast.parse(sb))==ast.dump(ast.parse(tb))
        else:
            assert s==t
    reading=(FOLDER/'llama2_src_02_RMSNorm_SwiGLU_代码注释.py').read_text(encoding='utf-8')
    joined='\n'.join(''.join(raw['cells'][i]['source']) for i in code_ids)
    assert ast.dump(ast.parse(joined))==ast.dump(ast.parse(reading))
    return raw,annotated,dict(original_bytes=len(data),original_sha256=sha(data),cells=28,code_cells=code_ids,
        derived_cells=57,schemas=3,comments_only=True,metadata_outputs_execution_counts_preserved=True,
        markdown_class_ast=True,reading_python_ast=True)

def execute_notebook(n,annotated=False):
    torch.manual_seed(17);np.random.seed(17)
    plt.close('all')
    scope={'__name__':'lesson17_notebook'}
    trace=[]
    for i in range(28):
        c=n['cells'][2+2*i] if annotated else n['cells'][i]
        if c['cell_type']!='code':continue
        s=''.join(c['source']);tree=ast.parse(s)
        result=None
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            # Preserve notebook execution order and evaluate its displayed final expression.
            if tree.body and isinstance(tree.body[-1],ast.Expr):
                prefix=ast.Module(body=tree.body[:-1],type_ignores=[])
                exec(compile(prefix,'cell-'+str(i),'exec'),scope)
                result=eval(compile(ast.Expression(tree.body[-1].value),'cell-'+str(i),'eval'),scope)
            else:
                exec(compile(tree,'cell-'+str(i),'exec'),scope)
        row=dict(cell=i,stdout=stream.getvalue(),x=tensor_summary(scope.get('x')),result=tensor_summary(result))
        for key in ['x_ln','x_rms']:
            if key in scope:row[key]=tensor_summary(scope[key])
        if isinstance(result,(np.floating,float,int)):row['scalar']=float(result)
        if i==23:
            curve=result[0]
            row['curve_x']=tensor_summary(curve.get_xdata())
            row['curve_y']=tensor_summary(curve.get_ydata())
            plt.savefig(OUTPUT/'silu_curve.png')
        trace.append(row)
    return trace,scope

def official_module():
    data=META.read_bytes()
    assert sha(data)=='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    mod=types.ModuleType('lesson17_official_model')
    mod.__file__=str(META)
    sys.modules[mod.__name__]=mod
    exec(compile(data,str(META),'exec'),mod.__dict__)
    return mod

def rms_checks(mod,scope):
    norm=mod.RMSNorm(3,eps=1e-6)
    x=torch.tensor([.7,-1.2,2.1],requires_grad=True)
    with torch.no_grad():norm.weight.copy_(torch.tensor([1.1,.8,1.3]))
    jac=torch.autograd.functional.jacobian(norm,x)
    a=x.detach().numpy().astype(np.float64);g=norm.weight.detach().numpy().astype(np.float64)
    s=np.sqrt(np.mean(a*a)+norm.eps)
    expected=g[:,None]*(np.eye(3)/s-np.outer(a,a)/(3*s**3))
    jacerr=float(np.max(np.abs(jac.numpy()-expected)))
    assert jacerr<1e-6
    norm(x).sum().backward()
    werr=float(np.max(np.abs(norm.weight.grad.numpy()-a/s)))
    assert werr<1e-6 and torch.isfinite(x.grad).all()
    notebook_norm=scope['RMSNorm'](3)
    with torch.no_grad():notebook_norm.weight.copy_(norm.weight)
    assert torch.equal(norm(x),notebook_norm(x))
    before=norm(x).detach();norm.eval();after=norm(x).detach()
    assert torch.equal(before,after)
    dtype_cases=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for convert_weight in [False,True]:
            module=mod.RMSNorm(3)
            if convert_weight:module=module.to(dtype=dtype)
            z=torch.tensor([.3,-.4,.9],dtype=dtype,requires_grad=True)
            y=module(z);y.float().sum().backward()
            assert torch.isfinite(y).all() and torch.isfinite(z.grad).all()
            assert y.dtype==(dtype if convert_weight else torch.float32)
            dtype_cases.append(dict(input=str(dtype),weight=str(module.weight.dtype),output=str(y.dtype),
                finite_forward_and_backward=True))
    unit=mod.RMSNorm(2)
    z=torch.tensor([3.,4.])
    ordinary=unit(z);zero=unit(torch.zeros(2))
    assert torch.equal(zero,torch.zeros(2))
    noeps=mod.RMSNorm(2,eps=0)
    invalid=noeps(torch.zeros(2))
    assert torch.isnan(invalid).all()
    near=torch.tensor([1e-8,2e-8])
    near_difference=float((unit(near)-noeps(near)).abs().max())
    assert near_difference>1
    with torch.no_grad():unit.weight.fill_(2)
    doubled=unit(z)
    assert torch.equal(doubled,ordinary*2)
    dtype_probe=mod.RMSNorm(3).double()
    double_x=torch.tensor([.7,-1.2,2.1],dtype=torch.float64)
    internal=dtype_probe._norm(double_x.float()).type_as(double_x)*dtype_probe.weight
    assert torch.equal(dtype_probe(double_x),internal)
    return dict(official_source_unmodified=True,notebook_class_exact_output=True,
        autograd_jacobian_max_abs_error=jacerr,autograd_weight_grad_max_abs_error=werr,
        dtype_cases=dtype_cases,train_eval_exact_equal=True,zero_with_eps=[float(t) for t in zero],
        zero_without_eps_all_nan=True,near_zero_omitted_eps_max_difference=near_difference,
        gain2_doubles_output=True,fp64_input_still_fp32_internal=True,
        fresh_notebook_norm_range=[float(scope['x_rms'].norm(dim=-1).min()),float(scope['x_rms'].norm(dim=-1).max())])

def silu_checks():
    z=torch.tensor([-5.,-2.,0.,2.,5.],dtype=torch.float64,requires_grad=True)
    expected=z/(1+torch.exp(-z))
    output=torch.nn.functional.silu(z)
    err=float((output-expected).abs().max())
    output.sum().backward()
    sig=1/(1+torch.exp(-z.detach()))
    derivative=sig+z.detach()*sig*(1-sig)
    graderr=float((z.grad-derivative).abs().max())
    assert err<1e-14 and graderr<1e-14
    return dict(values=output.detach().tolist(),input=z.detach().tolist(),forward_max_abs_error=err,
        autograd_max_abs_error=graderr,original_input_unchanged=True)

def ff_checks(mod):
    import fairscale.nn.model_parallel.initialize as mp
    assert torch.distributed.is_gloo_available()
    with tempfile.TemporaryDirectory(prefix='lesson17-gloo-') as temporary:
        uri=(Path(temporary)/'rendezvous').as_uri()
        torch.distributed.init_process_group('gloo',init_method=uri,rank=0,world_size=1)
        try:
            mp.initialize_model_parallel(1)
            torch.manual_seed(17)
            ff=mod.FeedForward(dim=4,hidden_dim=16,multiple_of=4,ffn_dim_multiplier=None).double()
            # Official lambda x:x intentionally leaves weights for checkpoint loading.
            # Explicitly initialize every actual parameter before this teaching run.
            with torch.no_grad():
                for p in ff.parameters():p.normal_(mean=0,std=.2)
            x=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
            y=ff(x)
            upstream=torch.randn_like(y)
            (y*upstream).sum().backward()
            params=dict(ff.named_parameters())
            assert sum(p.numel() for p in params.values())==144
            a=x.detach().numpy().copy();u=upstream.numpy().copy()
            w={k:p.detach().numpy().copy() for k,p in params.items()}
            def formula(a,w):
                left=a@w['w1.weight'].T
                right=a@w['w3.weight'].T
                return (left/(1+np.exp(-left))*right)@w['w2.weight'].T
            expected=formula(a,w)
            outputerr=float(np.max(np.abs(y.detach().numpy()-expected)))
            assert outputerr<1e-12
            def objective():return float(np.sum(formula(a,w)*u))
            def finite_difference(arr):
                grad=np.zeros_like(arr);h=1e-5
                for index in np.ndindex(arr.shape):
                    old=arr[index];arr[index]=old+h;plus=objective()
                    arr[index]=old-h;minus=objective();arr[index]=old
                    grad[index]=(plus-minus)/(2*h)
                return grad
            inputerr=float(np.max(np.abs(x.grad.numpy()-finite_difference(a))))
            graderrs={k:float(np.max(np.abs(p.grad.numpy()-finite_difference(w[k])))) for k,p in params.items()}
            assert inputerr<1e-8 and max(graderrs.values())<1e-8
            assert all(torch.isfinite(p.grad).all() for p in params.values())
            no_gate=(a@w['w1.weight'].T)@w['w2.weight'].T
            no_gate_error=float(np.max(np.abs(expected-no_gate)))
            assert no_gate_error>.001
            shapes={k:list(p.shape) for k,p in params.items()}
            # Only logical meta shapes for the 4096-dimensional model, no weight data or forward.
            with torch.device('meta'):
                large=mod.FeedForward(dim=4096,hidden_dim=16384,multiple_of=256,ffn_dim_multiplier=None)
            large_count=sum(p.numel() for p in large.parameters())
            assert large_count==135266304
            assert list(large.w1.weight.shape)==[11008,4096]
            return dict(world_size=1,backend='gloo',official_full_module_unmodified=True,
                explicit_random_weight_initialization='normal std0.2 seed17; no checkpoint',input_shape=list(x.shape),
                intermediate_shape=[2,3,12],output_shape=list(y.shape),parameter_shapes=shapes,parameters=144,
                numpy_formula_forward_max_abs_error=outputerr,numpy_fd_input_grad_max_abs_error=inputerr,
                numpy_fd_all_weight_grad_max_abs_errors=graderrs,omitting_gate_changes_output_by=no_gate_error,
                meta_only_4096=dict(hidden=11008,parameters=large_count,forward_executed=False))
        finally:
            mp.destroy_model_parallel()
            torch.distributed.destroy_process_group()

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    raw,ann,checks=audit()
    trace1,scope1=execute_notebook(raw)
    trace2,scope2=execute_notebook(ann,True)
    assert trace1==trace2
    mod=official_module()
    official_tree=ast.parse(META.read_bytes())
    reading_tree=ast.parse((ROOT/'代码/SwiGLU_源码阅读.py').read_bytes())
    class_node=lambda tree:next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeedForward')
    assert ast.dump(class_node(official_tree))==ast.dump(class_node(reading_tree))
    checks['supplementary_official_feedforward_class_ast']=True
    rms=rms_checks(mod,scope1)
    silu=silu_checks()
    ff=ff_checks(mod)
    artifacts={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted((ROOT/'复现/scripts').glob('*.py'))
        if p.name in ('build_annotated.py','download_original.py','verify_delivery.py')}
    report=dict(status='passed',scope='original notebook CPU execution plus actual official modules; no full paper/model reproduction',
        environment=dict(os=platform.platform(),python=sys.version,torch=torch.__version__,numpy=np.__version__,
            matplotlib=matplotlib.__version__,nbformat=nbformat.__version__,fairscale=importlib.metadata.version('fairscale'),
            cuda_available=torch.cuda.is_available(),fairscale_target=str(FAIRSCALE)),
        external_execution_controls=dict(torch_seed=17,numpy_seed=17,matplotlib_backend='Agg',original_cells_modified=False),
        audit=checks,notebook=dict(original_and_annotated_executed_each=20,all_traces_exact_equal=True,original_trace=trace1),
        rmsnorm=rms,silu=silu,swiglu=ff,official_source_sha256=sha(META.read_bytes()),verification_script_sha256=artifacts,
        limits=['No original Llama checkpoint','No full model generation or optimizer training in this lecture',
            'No GPU/NCCL or model parallel world_size>1','No paper dataset, benchmark, throughput or quality claim',
            'Original historical outputs were preserved, not replaced by new random values',
            'No strict float64 gradcheck claim for official RMSNorm because forward casts statistics to FP32'])
    (OUTPUT/'完整交付验证.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(status=report['status'],cells=28,code_cells=20,derived_cells=57,
        rms_jacobian_error=rms['autograd_jacobian_max_abs_error'],swiglu_input_grad_error=ff['numpy_fd_input_grad_max_abs_error'],
        swiglu_weight_grad_errors=ff['numpy_fd_all_weight_grad_max_abs_errors']),ensure_ascii=False))

if __name__=='__main__':
    main()
