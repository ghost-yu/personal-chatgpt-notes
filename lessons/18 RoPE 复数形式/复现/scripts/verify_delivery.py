# -*- coding: utf-8 -*-
"""Audit both notebooks and run their original/annotated code with actual Torch."""
import ast
import contextlib
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
import torch

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'代码/tutorials'
OUTPUT=ROOT/'复现/outputs'
META=Path('G:/zzz/Personal ChatGPT 系列/18 RoPE 复数形式/资料/llama-model.py')
FAIRSCALE=Path('G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local')
sys.path.insert(0,str(FAIRSCALE))
SPECS=[('main','llama2_src_03_RoPE_01_complex_position_encoding','',16),
       ('overview','llama2_src_03_RoPE','_RoPE总览',19)]

def sha(b):return hashlib.sha256(b).hexdigest()

def tokens(s):
    return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline)
        if t.type not in (tokenize.COMMENT,tokenize.NL)]

def summarize(x):
    if isinstance(x,torch.Tensor):
        a=x.detach().cpu().numpy()
        return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=sha(a.tobytes()))
    if isinstance(x,np.ndarray):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=sha(x.tobytes()))
    if isinstance(x,torch.Size):return dict(torch_size=list(x))
    if isinstance(x,tuple):return [summarize(i) for i in x]
    if isinstance(x,list) and x and isinstance(x[0],matplotlib.lines.Line2D):
        return [dict(x=summarize(i.get_xdata()),y=summarize(i.get_ydata())) for i in x]
    if isinstance(x,(float,int,np.floating)):return float(x)
    # Actual IPython image bytes are compared, not its memory-address repr.
    if hasattr(x,'data') and isinstance(x.data,bytes):return dict(image_bytes=len(x.data),sha256=sha(x.data))
    return None

def audit():
    manifest=json.loads((ROOT/'来源清单.json').read_bytes())
    pairs={};report={}
    from build_annotated import MAIN,OVERVIEW
    for role,base,suffix,count in SPECS:
        data=(FOLDER/(base+'.ipynb')).read_bytes()
        record=next(n for n in manifest['notebooks'] if Path(n['notebook']).stem==base)
        assert sha(data)==record['upstream_sha256']==record['sha256']
        raw=json.loads(data)
        ann=json.loads((FOLDER/('代码注释'+suffix+'.ipynb')).read_bytes())
        exp=json.loads((FOLDER/('原代码_带说明'+suffix+'.ipynb')).read_bytes())
        assert len(raw['cells'])==29 and len(ann['cells'])==len(exp['cells'])==59
        assert set(MAIN if role=='main' else OVERVIEW)==set(range(29))
        for n in [raw,ann,exp]:
            nbformat.validate(nbformat.from_dict(n))
            assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
        ids=[]
        for i,c in enumerate(raw['cells']):
            a=ann['cells'][2+2*i]
            assert c==exp['cells'][2+2*i]
            assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in a.items() if k!='source'}
            s=''.join(c['source']);t=''.join(a['source'])
            if c['cell_type']=='code':
                ids.append(i)
                assert tokens(s)==tokens(t), (role,i)
                assert ast.dump(ast.parse(s))==ast.dump(ast.parse(t))
                assert '# ' in t
            elif role=='overview' and i==21:
                assert tokens(s.split('```')[1])==tokens(t.split('```')[1])
                assert ast.dump(ast.parse(s.split('```')[1]))==ast.dump(ast.parse(t.split('```')[1]))
            else:assert s==t
        assert len(ids)==count
        reading=(FOLDER/(base+'_代码注释.py')).read_text(encoding='utf-8')
        joined='\n'.join(''.join(raw['cells'][i]['source']) for i in ids)
        assert ast.dump(ast.parse(joined))==ast.dump(ast.parse(reading))
        pairs[role]=(raw,ann)
        report[role]=dict(cells=29,code_ids=ids,derived_cells_each=59,original_bytes=len(data),sha256=sha(data),
            schemas=3,comment_only_tokens_and_ast=True,history_metadata_execution_preserved=True,reading_ast=True)
    image=manifest['images'][0]
    assert sha((ROOT/image['published']).read_bytes())==image['sha256']
    return pairs,report,image

def execute(n,role,annotated=False):
    plt.close('all');scope={'__name__':'lesson18_'+role};trace=[];saved={}
    previous=Path.cwd();os.chdir(FOLDER)
    try:
        for i in range(29):
            c=n['cells'][2+2*i] if annotated else n['cells'][i]
            if c['cell_type']!='code':continue
            tree=ast.parse(''.join(c['source']));result=None;capture=io.StringIO()
            with contextlib.redirect_stdout(capture):
                if tree.body and isinstance(tree.body[-1],ast.Expr):
                    exec(compile(ast.Module(body=tree.body[:-1],type_ignores=[]),'cell-'+str(i),'exec'),scope)
                    result=eval(compile(ast.Expression(tree.body[-1].value),'cell-'+str(i),'eval'),scope)
                else:exec(compile(tree,'cell-'+str(i),'exec'),scope)
            if i==15:
                saved['real_class']=scope['SinPositionEncoding'];saved['old_spe']=scope['spe']
                saved['real_pe']=result.detach().clone()
            if role=='main' and i==27:
                saved['complex_class']=scope['SinPositionEncoding'];saved['complex_pe']=result.detach().clone()
            trace.append(dict(cell=i,stdout=capture.getvalue(),result=summarize(result),
                variables={k:summarize(scope[k]) for k in ['thetas','xs','ys','position','even_i','odd_i','even_pe','odd_pe','stacked','pe','freqs_cis','complex_tensor'] if k in scope}))
            if role=='main' and i==5:plt.savefig(OUTPUT/'unit_circle.png')
    finally:os.chdir(previous)
    return trace,scope,saved

def official():
    data=META.read_bytes()
    assert sha(data)=='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    module=types.ModuleType('lesson18_full_meta_model');module.__file__=str(META)
    sys.modules[module.__name__]=module
    exec(compile(data,str(META),'exec'),module.__dict__)
    return module

def global_bug(raw,scope,saved):
    real=saved['real_class'];original=saved['complex_class'];global_dim=scope['d_model']
    reference=real(10,4)();wrong=original(10,4)()
    error=float((wrong-reference).abs().max());assert error>.001
    scope['d_model']=12
    altered=original(10,4)()
    dependency=float((altered-wrong).abs().max());assert dependency>.001
    del scope['d_model']
    try:
        original(10,4)();raise AssertionError('Expected missing-global failure')
    except NameError as exc:missing=str(exc)
    finally:scope['d_model']=global_dim
    # A separate teaching variant changes exactly one Name load in cell26.
    tree=ast.parse(''.join(raw['cells'][26]['source']))
    class ReplaceGlobal(ast.NodeTransformer):
        changes=0
        def visit_FunctionDef(self,node):
            return self.generic_visit(node) if node.name=='forward' else node
        def visit_Name(self,node):
            if node.id=='d_model' and isinstance(node.ctx,ast.Load):
                self.changes+=1
                return ast.copy_location(ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='d_model',ctx=ast.Load()),node)
            return node
    transformer=ReplaceGlobal();corrected_tree=ast.fix_missing_locations(transformer.visit(tree))
    assert transformer.changes==1
    isolated={'torch':torch,'nn':torch.nn}
    exec(compile(corrected_tree,'explicit cell26 teaching variant','exec'),isolated)
    corrected=isolated['SinPositionEncoding'];errors={}
    for width in [2,4,6,8]:
        errors[str(width)]=float((corrected(10,width)()-real(10,width)()).abs().max())
        assert errors[str(width)]<1e-6
    try:
        real(10,5)();raise AssertionError('Expected odd-dimension stack failure')
    except RuntimeError as exc:odd_error=str(exc)
    odd_shape=list(original(10,5)().shape)
    assert odd_shape==[10,6]
    assert saved['old_spe'].__class__ is real and real is not original
    assert sum(p.numel() for p in real(10,6).parameters())==0
    assert sum(p.numel() for p in original(10,6).parameters())==0
    return dict(original_global_d_model=global_dim,instance_dim4_wrong_max_abs_error=error,
        changing_global_to12_changes_output_by=dependency,missing_global_name_error=missing,
        separate_variant='only cell26 global d_model Name changed to self.d_model Attribute',ast_replacements=1,
        corrected_even_dimensions_max_errors=errors,original_odd_dim5_shape=odd_shape,real_odd_dim5_error=odd_error,
        old_instance_class_preserved=True,parameters_each=0)

def mathematical_checks(mod,scopes,saved):
    f=mod.precompute_freqs_cis;c=f(6,10)
    assert c.dtype==torch.complex64 and list(c.shape)==[10,3]
    with contextlib.redirect_stdout(io.StringIO()):
        main_c=scopes['main']['precompute_freqs_cis'](6,10)
    overview_c=scopes['overview']['precompute_freqs_cis'](6,10)
    assert torch.equal(main_c,c) and torch.equal(overview_c,c)
    real=saved['main']['real_pe'];complex_pe=saved['main']['complex_pe'];broadcast=saved['overview']['real_pe']
    broadcast_error=float((real-broadcast).abs().max())
    torch.testing.assert_close(real,broadcast,rtol=1e-6,atol=1e-7)
    restored=torch.view_as_real(c).flip((2,)).flatten(1,2)
    pe_error=float((restored-real).abs().max());assert pe_error<1e-6
    assert torch.equal(restored,complex_pe)
    view=torch.view_as_real(c);flipped=view.flip((2,))
    assert view.untyped_storage().data_ptr()==c.untyped_storage().data_ptr()
    assert flipped.untyped_storage().data_ptr()!=c.untyped_storage().data_ptr()
    assert c.numel()*c.element_size()==view.numel()*view.element_size()==240
    wrong=scopes['overview']['complex_tensor'];right=scopes['overview']['freqs_cis']
    missing_dim_error=float((wrong-right).abs().max());assert missing_dim_error>.01
    zero_end=f(6,0);odd=f(5,10)
    assert list(zero_end.shape)==[0,3] and list(odd.shape)==[10,2]
    a=mod.ModelArgs();large=f(a.dim//a.n_heads,a.max_seq_len*2)
    assert list(large.shape)==[4096,64]
    start=7;length=3;slice_c=c[start:start+length]
    omega32=10000**(-torch.arange(0,6,2).float()/6)
    expected=torch.polar(torch.ones(3,3),torch.outer(torch.arange(start,start+length),omega32))
    torch.testing.assert_close(slice_c,expected,rtol=1e-6,atol=1e-7)
    reset_error=float((slice_c-f(6,length)).abs().max());assert reset_error>.1
    q=torch.tensor([.3,-1.2,2.,.5,-.7,.8],dtype=torch.float64,requires_grad=True)
    k=torch.tensor([1.1,.4,-.6,1.5,2.,-.3],dtype=torch.float64,requires_grad=True)
    omega=10000**(-torch.arange(0,6,2,dtype=torch.float64)/6)
    def rotate(v,pos):
        z=torch.view_as_complex(v.reshape(3,2))
        return torch.view_as_real(z*torch.polar(torch.ones_like(omega),pos*omega)).flatten()
    def matrix(pos):
        r=torch.zeros(6,6,dtype=torch.float64)
        for j,w in enumerate(omega):
            angle=pos*w;r[2*j:2*j+2,2*j:2*j+2]=torch.stack([torch.stack([angle.cos(),-angle.sin()]),torch.stack([angle.sin(),angle.cos()])])
        return r
    m,n=3,7
    qr=rotate(q,m);kr=rotate(k,n);actual=qr@kr
    matrix_error=float((qr-matrix(m)@q).detach().abs().max());assert matrix_error<1e-12
    zq=torch.view_as_complex(q.reshape(3,2));zk=torch.view_as_complex(k.reshape(3,2))
    relative=(zq*zk.conj()*torch.exp(1j*(m-n)*omega)).real.sum()
    relative_matrix=q@matrix(n-m)@k
    shifted=rotate(q,m+11)@rotate(k,n+11)
    positional_error=max(float((actual-t).detach().abs()) for t in [relative,relative_matrix,shifted])
    assert positional_error<1e-12
    no_conjugate=(zq*zk*torch.exp(1j*(m-n)*omega)).real.sum()
    wrong_sign=(zq*zk.conj()*torch.exp(1j*(n-m)*omega)).real.sum()
    assert abs(float((actual-no_conjugate).detach()))>.001 and abs(float((actual-wrong_sign).detach()))>.001
    actual.backward()
    q_expected=matrix(n-m)@k.detach();k_expected=matrix(m-n)@q.detach()
    q_error=float((q.grad-q_expected).abs().max());k_error=float((k.grad-k_expected).abs().max())
    assert max(q_error,k_error)<1e-12
    norm_error=float(abs(qr.detach().norm()-q.detach().norm()));assert norm_error<1e-12
    angles=torch.tensor([.2,-.7],dtype=torch.float64,requires_grad=True)
    p=torch.polar(torch.ones_like(angles),angles);(p.real+2*p.imag).sum().backward()
    polar_error=float((angles.grad-(-angles.detach().sin()+2*angles.detach().cos())).abs().max());assert polar_error<1e-12
    gradcheck=torch.autograd.gradcheck(lambda angle:torch.polar(torch.ones_like(angle),angle),(torch.tensor([.2,-.7],dtype=torch.float64,requires_grad=True),))
    assert gradcheck
    return dict(complete_official_module_unmodified=True,official_vs_two_notebook_frequency_exact=True,
        real_outer_vs_overview_broadcast_max_abs_error=broadcast_error,real_complex_pe_max_abs_error=pe_error,
        original_overview_missing_dim_max_error=missing_dim_error,complex_and_pair_bytes=240,
        view_shares_storage=True,flip_allocates_storage=True,zero_end_shape=list(zero_end.shape),odd_dim5_official_shape=list(odd.shape),
        official_default_table=dict(dim=a.dim,heads=a.n_heads,max_seq_len=a.max_seq_len,shape=list(large.shape),bytes=large.numel()*large.element_size()),
        start_pos_slice=dict(start_pos=start,length=length,shape=list(slice_c.shape),resetting_position_changes_by=reset_error),
        rotation_matrix_max_abs_error=matrix_error,norm_max_abs_error=norm_error,rotated_dot=float(actual.detach()),
        relative_or_shift_max_abs_error=positional_error,wrong_without_conjugate=float(no_conjugate.detach()),
        wrong_complex_phase_sign=float(wrong_sign.detach()),autograd_q_max_abs_error=q_error,autograd_k_max_abs_error=k_error,
        polar_angle_autograd_max_error=polar_error,polar_complex_gradcheck=gradcheck,
        limits='manual small Q/K mathematical rotation, not Attention/apply_rotary_emb model execution')

def screenshot_checks(mod):
    # Execute the exact functions transcribed from the preserved upstream image.
    source=ROOT/'代码/配图_RoPE源码阅读.py'
    scope={};exec(compile(source.read_bytes(),str(source),'exec'),scope)
    table=mod.precompute_freqs_cis(6,10).to(torch.complex128)
    cos=torch.cat([table.real,table.real],dim=-1)[None,None,:,:]
    sin=torch.cat([table.imag,table.imag],dim=-1)[None,None,:,:]
    ids=torch.tensor([[1,3,7],[2,4,9]],dtype=torch.int64)
    q=torch.arange(72,dtype=torch.float64).reshape(2,2,3,6)/100
    k=torch.arange(36,dtype=torch.float64).reshape(2,1,3,6)/100
    perm=[0,2,4,1,3,5];inverse=[0,3,1,4,2,5]
    q_half=q[...,perm];k_half=k[...,perm]
    q_out,k_out=scope['apply_rotary_pos_emb'](q_half,k_half,cos,sin,ids)
    phase=table[ids].unsqueeze(1)
    def adjacent(x):
        z=torch.view_as_complex(x.reshape(*x.shape[:-1],3,2))
        return torch.view_as_real(z*phase).flatten(-2)
    q_error=float((q_out[...,inverse]-adjacent(q)).abs().max())
    k_error=float((k_out[...,inverse]-adjacent(k)).abs().max())
    assert max(q_error,k_error)<1e-12
    wrong_q,_=scope['apply_rotary_pos_emb'](q,k,cos,sin,ids)
    mismatch=float((wrong_q-adjacent(q)).abs().max());assert mismatch>.01
    return dict(source='fixed upstream PNG transcription; exact HF commit not found',source_sha256=sha(source.read_bytes()),
        actual_functions_executed=True,q_shape=list(q.shape),k_shape=list(k.shape),position_ids=ids.tolist(),
        matching_permutation=perm,q_max_error=q_error,k_max_error=k_error,mixed_layout_max_error=mismatch,
        limitation='two functions only; no HF model, no Meta apply_rotary_emb or Attention execution')

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    pairs,audits,image=audit();scopes={};saved={};traces={}
    for role,_,_,count in SPECS:
        raw,ann=pairs[role]
        trace1,scope1,saved1=execute(raw,role)
        trace2,scope2,saved2=execute(ann,role,True)
        assert trace1==trace2 and len(trace1)==count
        scopes[role]=scope1;saved[role]=saved1;traces[role]=trace1
    mod=official()
    bug=global_bug(pairs['main'][0],scopes['main'],saved['main'])
    math=mathematical_checks(mod,scopes,saved)
    screenshot=screenshot_checks(mod)
    artifacts={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted((ROOT/'复现/scripts').glob('*.py'))
        if p.name in ('download_original.py','build_annotated.py','verify_delivery.py')}
    report=dict(status='passed',environment=dict(os=platform.platform(),python=sys.version,torch=torch.__version__,numpy=np.__version__,
        matplotlib=matplotlib.__version__,nbformat=nbformat.__version__,fairscale=importlib.metadata.version('fairscale'),cuda_available=torch.cuda.is_available()),
        audit=audits,original_image=image,execution=dict(original_and_annotated_each={'main':16,'overview':19},
            all_traces_exact_equal=True,backend='Agg',cell_statement_changes=0,traces=traces),global_variable_bug=bug,
        mathematical_checks=math,official_source_sha256=sha(META.read_bytes()),verification_script_sha256=artifacts,
        original_screenshot_code=screenshot,
        limitations=['No Llama checkpoint or full Transformer/Attention execution','No actual apply_rotary_emb call in this lecture',
            'No CUDA/low-precision model execution or model parallel communication','No RoFormer paper training, quality, speed or long-context benchmark',
            'Historical notebook outputs preserved; new traces separate','Correction of global variable is a separate one-Name teaching variant'])
    (OUTPUT/'完整交付验证.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(status='passed',cells=58,code=35,schemas=6,main_global_bug_error=bug['instance_dim4_wrong_max_abs_error'],
        corrected_errors=bug['corrected_even_dimensions_max_errors'],relative_error=math['relative_or_shift_max_abs_error']),ensure_ascii=True))

if __name__=='__main__':main()
