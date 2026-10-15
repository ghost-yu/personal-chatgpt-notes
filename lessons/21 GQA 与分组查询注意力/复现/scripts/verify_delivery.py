# -*- coding: utf-8 -*-
"""Real fixed Meta/FairScale GQA forward, autograd, grouping, storage and checkpoint conversion tests."""
import argparse
import ast
import ctypes
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import tokenize
import types
from pathlib import Path

import nbformat
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'代码/tutorials'
META_SHA='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(tree):return ast.dump(tree,include_attributes=False)
def tokens(src):return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type not in (tokenize.COMMENT,tokenize.NL)]

def audit(source):
    manifest=json.loads((ROOT/'来源清单.json').read_bytes())
    for entry in manifest['files']:
        b=(ROOT/entry['published_path']).read_bytes();assert len(b)==entry['bytes'] and sha(b)==entry['sha256']
    nbs=[json.loads((CODE/name).read_bytes()) for name in ('llama2_src_grouped_query_attention.ipynb','原代码_带说明.ipynb','代码注释.ipynb')]
    for n in nbs:nbformat.validate(nbformat.from_dict(n))
    original,explained,annotated=nbs
    assert len(original['cells'])==5 and len(explained['cells'])==len(annotated['cells'])==11
    for n in nbs[1:]:assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in original.items() if k!='cells'}
    executions=[];previous=os.getcwd()
    try:
        os.chdir(CODE)
        for label,nb in [('original',original),('annotated',annotated)]:
            ns={};records=[]
            for i,c in enumerate(original['cells']):
                if label=='annotated':c=nb['cells'][2+2*i]
                if c['cell_type']!='code':continue
                tree=ast.parse(''.join(c['source']));last=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
                exec(compile(tree,label+str(i),'exec'),ns)
                value=eval(compile(ast.Expression(last.value),label+str(i),'eval'),ns) if last else None
                record={'cell':i,'whole_original_cell':True}
                if value is not None:
                    record.update(image_bytes=len(value.data),image_sha256=sha(value.data),width=value.width)
                    assert sha(value.data)==manifest['files'][1]['sha256']
                records.append(record)
            executions.append(records)
    finally:os.chdir(previous)
    assert executions[0]==executions[1]
    for i,c in enumerate(original['cells']):
        assert explained['cells'][2+2*i]==c
        a=annotated['cells'][2+2*i]
        assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in a.items() if k!='source'}
        if c['cell_type']=='code':
            assert dump(ast.parse(''.join(c['source'])))==dump(ast.parse(''.join(a['source'])))
            assert tokens(''.join(c['source']))==tokens(''.join(a['source']))
        else:assert c==a
    joined='\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type']=='code')
    assert dump(ast.parse(joined))==dump(ast.parse((CODE/'llama2_src_grouped_query_attention_代码注释.py').read_text(encoding='utf-8')))
    reading=ast.parse((CODE/'GQA核心源码_代码注释.py').read_text(encoding='utf-8'))
    original_tree=ast.parse(source)
    for n in reading.body:
        official=next(o for o in original_tree.body if isinstance(o,(ast.FunctionDef,ast.ClassDef)) and o.name==n.name)
        assert dump(n)==dump(official)
    official_text='\n\n'.join(ast.get_source_segment(source,next(o for o in original_tree.body if isinstance(o,(ast.FunctionDef,ast.ClassDef)) and o.name==n.name)) for n in reading.body)+'\n'
    assert tokens(official_text)==tokens((CODE/'GQA核心源码_代码注释.py').read_text(encoding='utf-8'))
    mdsrc=''.join(original['cells'][4]['source']);snippets=ast.parse((CODE/'Markdown源码_代码注释.py').read_text(encoding='utf-8'))
    assert len(snippets.body)==6
    # These are six genuine assignments, not statements invented to replace the Markdown source.
    for node in snippets.body:
        assert ast.unparse(node) in mdsrc
    ns={'torch':torch,'bsz':1,'seqlen':2,'model_parallel_size':1,
        'args':types.SimpleNamespace(n_heads=64),
        'self':types.SimpleNamespace(n_local_heads=64,n_local_kv_heads=8,n_kv_heads=8,head_dim=128),
        'xq':torch.randn(1,2,8192),'xk':torch.randn(1,2,1024),'xv':torch.randn(1,2,1024)}
    exec(compile(snippets,'Markdown4 actual CPU shape context','exec'),ns)
    assert ns['xq'].shape==(1,2,64,128) and ns['xk'].shape==ns['xv'].shape==(1,2,8,128) and ns['self'].n_rep==8
    return {'original_cells':5,'code_cells':2,'markdown_cells':3,'derivative_cells':[11,11],'schemas':3,
            'original_bytes_history_metadata_ast_tokens_preserved':True,'all_reading_ast_and_tokens_equal':True,
            'original_and_annotated_execution_equal':True,'execution':executions,
            'markdown4_assignments_executed':6,'markdown_context':'explicit CPU tensors, world_size arithmetic=1; no multi-rank simulation',
            'markdown_shapes':{'xq':list(ns['xq'].shape),'xk':list(ns['xk'].shape),'xv':list(ns['xv'].shape),'n_rep':8}}

class CPU(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            assert isinstance(node.func.value,ast.Call) and isinstance(node.func.value.func,ast.Attribute) and node.func.value.func.attr=='zeros'
            node.func.attr='to';node.args=[ast.Constant('cpu')];self.count+=1
        return node

def load(source,name,cpu=False):
    tree=ast.parse(source)
    if cpu:
        transform=CPU();tree=transform.visit(tree);assert transform.count==2
    ast.fix_missing_locations(tree);m=types.ModuleType(name);sys.modules[name]=m
    exec(compile(tree,name,'exec'),m.__dict__);return m

def repeat_tests(m):
    torch.manual_seed(21)
    rows=[]
    for groups,rep in [(4,1),(2,2),(2,4),(1,4)]:
        x=torch.randn(2,3,groups,4,dtype=torch.float64,requires_grad=True)
        y=m.repeat_kv(x,rep);expected=x.repeat_interleave(rep,dim=2)
        assert torch.equal(y,expected)
        cotangent=torch.randn_like(y)
        grad=torch.autograd.grad((y*cotangent).sum(),x)[0]
        expected_grad=cotangent.reshape(2,3,groups,rep,4).sum(3)
        assert torch.equal(grad,expected_grad)
        if rep==1:assert y is x
        expanded=x[:,:,:,None,:].expand(2,3,groups,rep,4)
        shared=y.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()
        rows.append({'groups':groups,'n_rep':rep,'shape':list(y.shape),'output_is_input':y is x,
                     'reshape_shares_input_storage':shared,'expanded_shares_input_storage':expanded.untyped_storage().data_ptr()==x.untyped_storage().data_ptr(),
                     'input_storage_bytes':x.untyped_storage().nbytes(),'output_storage_bytes':y.untyped_storage().nbytes(),'output_stride':list(y.stride()),
                     'gradient_sum_exact':True})
    x=torch.randn(1,2,2,2,dtype=torch.float64,requires_grad=True)
    gradcheck=bool(torch.autograd.gradcheck(lambda z:m.repeat_kv(z,3),(x,),eps=1e-6,atol=1e-6,rtol=1e-5));assert gradcheck
    raw=torch.arange(2,dtype=torch.float64).view(1,1,2,1)
    grouped=m.repeat_kv(raw,4);cyclic=raw.repeat(1,1,4,1)
    assert grouped.flatten().tolist()==[0,0,0,0,1,1,1,1] and not torch.equal(grouped,cyclic)
    nc=torch.randn(2,3,2,8)[...,::2];assert not nc.is_contiguous()
    assert torch.equal(m.repeat_kv(nc,3),nc.repeat_interleave(3,dim=2))
    failures={}
    for name,fn in [('rank3',lambda:m.repeat_kv(torch.zeros(2,3,4),2)),('negative_rep',lambda:m.repeat_kv(torch.zeros(1,2,2,4),-1))]:
        try:fn()
        except (RuntimeError,ValueError) as e:failures[name]={'type':type(e).__name__,'message':str(e)}
        assert name in failures
    zero=m.repeat_kv(torch.zeros(1,2,2,4),0);assert zero.shape==(1,2,0,4)
    return {'cases':rows,'float64_gradcheck':gradcheck,'head_order':grouped.flatten().tolist(),
            'wrong_cyclic_order':cyclic.flatten().tolist(),'noncontiguous_passed':True,'failures':failures,
            'zero_repeat_shape':list(zero.shape),'zero_repeat_is_not_valid_gqa':True}

def factory(m,groups,dim=16,heads=4):
    args=m.ModelArgs(dim=dim,n_heads=heads,n_kv_heads=groups,max_batch_size=2,max_seq_len=8)
    a=m.Attention(args).double()
    with torch.no_grad():
        for p in a.parameters():p.normal_(0,0.1)
    return a

def manual(m,x,weights,groups,freqs,mask,grouped=False,cyclic=False):
    b,t,dmodel=x.shape;heads=4;d=4;rep=heads//groups
    q=F.linear(x,weights['wq']).view(b,t,heads,d)
    k=F.linear(x,weights['wk']).view(b,t,groups,d)
    v=F.linear(x,weights['wv']).view(b,t,groups,d)
    q,k=m.apply_rotary_emb(q,k,freqs)
    if grouped:
        qg=q.transpose(1,2).reshape(b,groups,rep,t,d);kg=k.transpose(1,2);vg=v.transpose(1,2)
        scores=torch.einsum('bgrtd,bgnd->bgrtn',qg,kg)/math.sqrt(d)
        weights_attention=(scores+mask).float().softmax(-1).type_as(q)
        out=torch.einsum('bgrtn,bgnd->bgrtd',weights_attention,vg).reshape(b,heads,t,d).transpose(1,2)
    else:
        parts=[]
        for h in range(heads):
            group=h%groups if cyclic else h//rep
            score=(q[:,:,h]@k[:,:,group].transpose(-1,-2))/math.sqrt(d)
            attention=(score+mask).float().softmax(-1).type_as(q)
            parts.append(attention@v[:,:,group])
        out=torch.stack(parts,dim=2)
    return F.linear(out.reshape(b,t,dmodel),weights['wo'])

def forward_tests(m):
    torch.manual_seed(21);freq=m.precompute_freqs_cis(4,8)
    mask=torch.triu(torch.full((3,3),-float('inf'),dtype=torch.float64),diagonal=1)
    results=[]
    for groups in (4,2,1):
        a=factory(m,groups);x=torch.randn(2,3,16,dtype=torch.float64,requires_grad=True)
        y=a(x,0,freq[:3],mask);cotangent=torch.randn_like(y)
        parameters=[getattr(a,key).weight for key in ('wq','wk','wv','wo')]
        actual_grads=torch.autograd.grad((y*cotangent).sum(),[x]+parameters)
        xr=x.detach().clone().requires_grad_(True)
        weights={key:getattr(a,key).weight.detach().clone().requires_grad_(True) for key in ('wq','wk','wv','wo')}
        ref=manual(m,xr,weights,groups,freq[:3],mask)
        ref_grads=torch.autograd.grad((ref*cotangent).sum(),[xr]+list(weights.values()))
        error=float((y-ref).abs().max());assert torch.allclose(y,ref,atol=1e-6,rtol=1e-5)
        grad_errors={key:float((left-right).abs().max()) for key,left,right in zip(('x','wq','wk','wv','wo'),actual_grads,ref_grads)}
        for left,right in zip(actual_grads,ref_grads):assert torch.allclose(left,right,atol=1e-6,rtol=1e-5)
        grouped=manual(m,xr,weights,groups,freq[:3],mask,grouped=True)
        assert torch.allclose(y,grouped,atol=1e-6,rtol=1e-5)
        incorrect=manual(m,xr,weights,groups,freq[:3],mask,cyclic=True)
        wrong_error=float((incorrect-y).abs().max())
        if groups==2:assert wrong_error>1e-5
        cache_bytes=sum(z.numel()*z.element_size() for z in (a.cache_k,a.cache_v))
        results.append({'query_heads':4,'kv_heads':groups,'n_rep':4//groups,'output_shape':list(y.shape),
                        'projection_weight_shapes':{k:list(getattr(a,k).weight.shape) for k in weights},
                        'parameter_count':sum(p.numel() for p in a.parameters()),'cache_shape':list(a.cache_k.shape),'cache_allocated_bytes_after_double_forward':cache_bytes,
                        'head_loop_error':error,'gradient_errors':grad_errors,'grouped_einsum_error':float((y-grouped).abs().max()),'wrong_cyclic_group_error':wrong_error})
    return results

def tied_and_pooling(m):
    torch.manual_seed(210);g=factory(m,2);tied=factory(m,4)
    with torch.no_grad():
        tied.wq.weight.copy_(g.wq.weight);tied.wo.weight.copy_(g.wo.weight)
        for name in ('wk','wv'):
            getattr(tied,name).weight.copy_(getattr(g,name).weight.reshape(2,4,16).repeat_interleave(2,dim=0).reshape(16,16))
    x=torch.randn(2,3,16,dtype=torch.float64,requires_grad=True);xx=x.detach().clone().requires_grad_(True)
    mask=torch.triu(torch.full((3,3),-float('inf'),dtype=torch.float64),1);freq=m.precompute_freqs_cis(4,8)[:3]
    yg=g(x,0,freq,mask);ym=tied(xx,0,freq,mask);cotangent=torch.randn_like(yg)
    keys=('wq','wk','wv','wo')
    gg=torch.autograd.grad((yg*cotangent).sum(),[x]+[getattr(g,k).weight for k in keys])
    mg=torch.autograd.grad((ym*cotangent).sum(),[xx]+[getattr(tied,k).weight for k in keys])
    comparisons={}
    for index,key in enumerate(('x',)+keys):
        right=mg[index]
        if key in ('wk','wv'):right=right.reshape(2,2,4,16).sum(1).reshape(8,16)
        comparisons[key]=float((gg[index]-right).abs().max());assert torch.allclose(gg[index],right,atol=1e-6,rtol=1e-5)
    assert torch.allclose(yg,ym,atol=1e-6,rtol=1e-5)
    pooled=factory(m,2)
    with torch.no_grad():
        pooled.wq.weight.copy_(tied.wq.weight);pooled.wo.weight.copy_(tied.wo.weight)
        for name in ('wk','wv'):getattr(pooled,name).weight.copy_(getattr(tied,name).weight.reshape(2,2,4,16).mean(1).reshape(8,16))
    yp=pooled(x.detach(),0,freq,mask);assert torch.allclose(yp,yg,atol=1e-6,rtol=1e-5)
    random_mha=factory(m,4);random_pool=factory(m,2)
    with torch.no_grad():
        random_pool.wq.weight.copy_(random_mha.wq.weight);random_pool.wo.weight.copy_(random_mha.wo.weight)
        for name in ('wk','wv'):getattr(random_pool,name).weight.copy_(getattr(random_mha,name).weight.reshape(2,2,4,16).mean(1).reshape(8,16))
    before=random_mha(x.detach(),0,freq,mask);after=random_pool(x.detach(),0,freq,mask)
    changed=float((before-after).abs().max());assert changed>1e-5
    return {'tied_mha_forward_error':float((yg-ym).abs().max()),'tied_mha_gradient_sum_errors':comparisons,
            'identical_head_pooling_error':float((yp-yg).abs().max()),'general_random_mha_pooling_changes_output':changed,
            'conversion_scope':'actual CPU matrix mean-pooling and Attention forward; no uptraining or quality evaluation'}

def incremental_and_invalid(m):
    torch.manual_seed(211);cached=factory(m,2);ref=factory(m,2);ref.load_state_dict(cached.state_dict())
    default=factory(m,None)
    assert default.n_kv_heads==4 and default.n_rep==1 and default.wk.weight.shape==(16,16)
    freq=m.precompute_freqs_cis(4,8);x=torch.randn(2,6,16,dtype=torch.float64)
    rows=[]
    with torch.no_grad():
        for end in (3,4,5,6):
            start=0 if end==3 else end-1
            mask=torch.triu(torch.full((end,end),-float('inf'),dtype=torch.float64),1)
            y=cached(x[:,start:end],start,freq[start:end],mask if start==0 else None)
            full=ref(x[:,:end],0,freq[:end],mask)
            error=float((y-full[:,start:end]).abs().max());assert torch.allclose(y,full[:,start:end],atol=1e-6,rtol=1e-5)
            for field in ('cache_k','cache_v'):
                assert torch.allclose(getattr(cached,field)[:,:end],getattr(ref,field)[:,:end],atol=1e-6,rtol=1e-5)
            assert cached.cache_k.shape==(2,8,2,4)
            rows.append({'start_pos':start,'new_tokens':end-start,'logit_free_attention_output_error':error,'persistent_kv_heads':2})
    failures={}
    # A nondivisible ratio is floored to1 but later head matmul really fails.
    bad=factory(m,4,dim=24,heads=6)
    try:bad(torch.randn(1,2,24,dtype=torch.float64),0,freq[:2],None)
    except RuntimeError as e:failures['query6_kv4']={'n_rep':bad.n_rep,'type':type(e).__name__,'message':str(e)}
    assert failures['query6_kv4']['n_rep']==1
    # Source uses //, so hidden dimension divisibility is not asserted here either.
    truncated=factory(m,2,dim=18,heads=4)
    y=truncated(torch.randn(1,2,18,dtype=torch.float64),0,freq[:2],None)
    assert y.shape==(1,2,18) and truncated.head_dim==4 and truncated.wq.weight.shape==(16,18)
    return {'incremental':rows,'nondivisible_heads_failure':failures,
            'none_default_is_mha':True,
            'dim18_heads4_runs_with_truncated_inner_width':{'head_dim':4,'wq_weight_shape':list(truncated.wq.weight.shape),'output_shape':list(y.shape)},
            'not_a_validation_of_noncausal_multitoken_use':'dim18 case only checks shape, mask=None is not a causal sequence experiment',
            'multi_rank_tested':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--meta-source',required=True);p.add_argument('--fairscale',required=True);args=p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8');data=Path(args.meta_source).read_bytes();assert sha(data)==META_SHA
    source=data.decode('utf-8');static=audit(source);sys.path.insert(0,args.fairscale)
    import fairscale.nn.model_parallel.initialize as fs
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='lesson21-gloo-') as directory:
        torch.distributed.init_process_group('gloo',init_method=(Path(directory)/'rendezvous').as_uri(),rank=0,world_size=1)
        fs.initialize_model_parallel(1)
        try:
            original=load(source,'lesson21_meta_original')
            failure=None
            try:original.Attention(original.ModelArgs(dim=16,n_heads=4,n_kv_heads=2,max_batch_size=2,max_seq_len=8))
            except (AssertionError,RuntimeError) as e:failure={'type':type(e).__name__,'message':str(e)}
            assert failure and not torch.cuda.is_available()
            m=load(source,'lesson21_meta_cpu',True)
            repeat=repeat_tests(m);forward=forward_tests(m);tied=tied_and_pooling(m);incremental=incremental_and_invalid(m)
        finally:fs.destroy_model_parallel();torch.distributed.destroy_process_group()
    class Memory(ctypes.Structure):
        _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','available','totalPage','availPage','totalVirtual','availVirtual','extended')]
    memory=Memory();memory.dwLength=ctypes.sizeof(memory);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
         'packages':{n:importlib.metadata.version(n) for n in ('fairscale','nbformat','ipython')},'ram_bytes':memory.total,'free_disk_bytes':shutil.disk_usage(ROOT).free}
    report={'status':'部分成功','static':static,'environment':env,'official_source_sha256':META_SHA,'original_cuda_failure':failure,
            'cpu_changes':['exactly two cache torch.zeros(...).cuda() calls changed to .to(cpu)'],'world_size':1,'backend':'gloo',
            'repeat_kv':repeat,'attention_forward_backward':forward,'tied_mha_and_pooling':tied,'incremental_and_invalid':incremental,
            'dtype_scope':'Attention parameters float64, official RoPE and softmax internally float32; repeat_kv gradcheck genuinely float64',
            'scripts_sha256':{q.name:sha(q.read_bytes()) for q in Path(__file__).parent.glob('*.py')},
            'reading_sha256':{q.name:sha(q.read_bytes()) for q in CODE.glob('*代码注释.py')},
            'not_run':['Llama2 pretrained weights','GQA paper T5 uptraining or benchmark','GPU inference timing','multi-rank communication','quality metrics']}
    out=ROOT/'复现/outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'完整交付验证.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'passed':True,'static':static,'attention_cases':forward,'tied_and_pooling':tied},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
