"""真实PyTorch执行原/注释全部代码与Markdown类；独立RoPE/CoPE前向梯度及边界。"""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import textwrap
import tokenize
import warnings
import nbformat
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'代码/tutorials/position_encoding'
UPSTREAM=Path('G:/zzz/Llama 入门教程/代码/personal_chatgpt')
def src(c):return ''.join(c['source'])
def tree(s):return ast.dump(ast.parse(textwrap.dedent(s)))
def blocks(s):return re.findall(r'```[^\n]*\n(.*?)```',s,re.S)
def tokens(s):return [(x.type,x.string) for x in tokenize.generate_tokens(io.StringIO(s).readline) if x.type not in [tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,tokenize.ENDMARKER,tokenize.ENCODING]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def simplify(x):
    if torch.is_tensor(x):return {'dtype':str(x.dtype),'shape':list(x.shape),'value':torch.view_as_real(x).tolist() if x.is_complex() else x.tolist()}
    if isinstance(x,(list,tuple)):return [simplify(v) for v in x]
    if isinstance(x,(str,float,int,bool,type(None))):return x
    return {'type':type(x).__name__,'url':getattr(x,'url',None)}

def audit():
    m=json.loads((ROOT/'来源清单.json').read_text('utf-8'))
    for rel,expected in m['files'].items():
        p=ROOT/rel;assert sha(p)==expected
        assert p.read_bytes()==subprocess.check_output(['git','show',m['upstream_commit']+':'+rel.removeprefix('代码/')],cwd=UPSTREAM)
    raw=json.loads((FOLDER/'rope_cope.ipynb').read_bytes());variants=[json.loads((FOLDER/f).read_bytes()) for f in ['原代码_带说明.ipynb','代码注释.ipynb']]
    for n in [raw,*variants]:
        with warnings.catch_warnings():warnings.simplefilter('ignore');nbformat.validate(nbformat.from_dict(copy.deepcopy(n)))
    for n,annotated in zip(variants,[False,True]):
        assert len(n['cells'])==81
        assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
        for i,c in enumerate(raw['cells']):
            note,saved=n['cells'][1+2*i:3+2*i];assert f'单元 {i}（' in src(note)
            assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in saved.items() if k!='source'}
            if not annotated:assert c==saved
            elif c['cell_type']=='code':assert tree(src(c))==tree(src(saved));assert tokens(src(c))==tokens(src(saved))
            elif i in [5,9,39]:
                assert len(blocks(src(c)))==len(blocks(src(saved)))
                for a,b in zip(blocks(src(c)),blocks(src(saved))):assert tree(a)==tree(b);assert tokens(a)==tokens(b)
            else:assert c==saved
    codes=[i for i,c in enumerate(raw['cells']) if c['cell_type']=='code'];assert len(codes)==25
    assert tree((FOLDER/'rope_cope_代码注释.py').read_text('utf-8'))==tree('\n'.join(src(raw['cells'][i]) for i in codes))
    class_src=blocks(src(raw['cells'][39]))[0];core=ast.parse((FOLDER/'CoPE核心源码_代码注释.py').read_text('utf-8'))
    assert [ast.dump(x) for x in core.body if isinstance(x,ast.ClassDef)]==[ast.dump(x) for x in ast.parse(class_src).body]
    return raw['cells'],[variants[1]['cells'][2+2*i] for i in range(40)],dict(cells=40,code_cells=25,schemas=3,derived_cells=81,raw_git_bytes=True,metadata_historical_outputs=True,code_ast_and_tokens=True,markdown_code_blocks_5_9_39=True,python_reading_ast=True,complete_cope_and_selfattn_ast=True)

def run(cells):
    torch.manual_seed(28);np.random.seed(28);ns={};trace=[]
    for i,c in enumerate(cells):
        if c['cell_type']!='code':continue
        a=ast.parse(src(c));last=None
        if a.body and isinstance(a.body[-1],ast.Expr):last=a.body.pop().value
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(a,f'<original-cell-{i}>','exec'),ns)
            value=eval(compile(ast.Expression(last),f'<display-{i}>','eval'),ns) if last is not None else None
        trace.append({'cell':i,'status':'literal','stdout':output.getvalue(),'display':simplify(value)})
    # 原notebook的Markdown类单独执行：只提供原块缺失的真实nn导入。
    ns['nn']=nn;exec(compile(blocks(src(cells[39]))[0],'<markdown39>','exec'),ns)
    return ns,trace

def rope_tests(ns):
    compute=ns['precompute_freqs_cis'];matrix=ns['get_rope_matrix'];out={}
    phases=compute(4,3);angles=torch.tensor([[0.,0.],[1.,.01],[2.,.02]])
    torch.testing.assert_close(phases,torch.complex(angles.cos(),angles.sin()))
    assert torch.equal(phases,ns['e_j_theta_matrix'])
    perm=torch.eye(4)[[0,2,1,3]]
    same_angles=torch.stack([ns['theta1'],ns['theta2']])
    half=matrix(4,same_angles);torch.testing.assert_close(half,perm@ns['R']@perm.T)
    torch.testing.assert_close(ns['R'].T@ns['R'],torch.eye(4))
    q=ns['q'];k=ns['k'];rm=ns['R_m'];rn=ns['R_n'];rel=ns['R_n_minus_m']
    scores=[torch.dot(rm@q,rn@k),q@(rm.T@rn)@k,q@rel@k]
    for x in scores[1:]:torch.testing.assert_close(x,scores[0])
    # 独立可微double旋转，不用原函数的float32分配路径做double声称。
    def rotate(x,position):
        freq=torch.tensor([1.,.01],dtype=x.dtype);c=(position*freq).cos();s=(position*freq).sin()
        real=x[:2];imag=x[2:];return torch.cat([real*c-imag*s,real*s+imag*c])
    qq=q.double().requires_grad_();kk=k.double().requires_grad_()
    lhs=(rotate(qq,2)*rotate(kk,5)).sum();rhs=(qq*rotate(kk,3)).sum()
    torch.testing.assert_close(lhs,rhs);left=torch.autograd.grad(lhs,(qq,kk),retain_graph=True);right=torch.autograd.grad(rhs,(qq,kk))
    for a,b in zip(left,right):torch.testing.assert_close(a,b)
    shared=(rotate(qq,9)*rotate(kk,12)).sum();torch.testing.assert_close(lhs,shared)
    normal=torch.diag(torch.tensor([2.,3.]));assert torch.equal(normal@normal.T,normal.T@normal) and not torch.equal(normal@normal.T,torch.eye(2))
    dtype=matrix(4,torch.tensor([.2,.3],dtype=torch.float64)).dtype
    failures={}
    for name,fn in [('odd_precompute',lambda:compute(3,3)),('odd_matrix',lambda:matrix(3,torch.ones(1))),('wrong_angle_length',lambda:matrix(4,torch.ones(3)))]:
        try:failures[name]={'result':simplify(fn())}
        except Exception as e:failures[name]={'exception':type(e).__name__,'message':str(e)}
    odd=matrix(3,torch.ones(1));failures['odd_matrix']['rank']=int(torch.linalg.matrix_rank(odd));failures['odd_matrix']['orthogonal']=bool(torch.allclose(odd.T@odd,torch.eye(3)))
    assert failures['odd_matrix']['rank']==2 and not failures['odd_matrix']['orthogonal']
    out.update(phase_shape=list(phases.shape),phase_manual_exact=True,adjacent_half_permutation=[0,2,1,3],permutation_verified=True,three_original_scores=[float(x) for x in scores],independent_double_relative_score=float(lhs),relative_forward_gradient_and_shared_shift=True,normal_not_orthogonal_counterexample=True,original_matrix_with_double_angles_dtype=str(dtype),boundaries=failures)
    # 无位置无mask的attention置换等变；同步排列输出，不是输出完全不变。
    torch.manual_seed(280);x=torch.randn(4,3,dtype=torch.float64);p=torch.eye(4,dtype=torch.float64)[[2,0,3,1]]
    att=lambda z:torch.softmax(z@z.T/math.sqrt(3),-1)@z
    error=float((att(p@x)-p@att(x)).abs().max());assert error<1e-12
    out['positionless_permutation_equivariance_max_error']=error
    return out

def dense_reference(q,k,v,embedding,mask):
    """独立先插值完整位置向量再点积；不调用原CoPE/gather优化路径。"""
    logits=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])+mask.log()
    gate=torch.sigmoid(logits);b,n,s=gate.shape;p=embedding.shape[-1]
    rows=[]
    for batch in range(b):
        rr=[]
        for i in range(n):
            cols=[]
            for j in range(s):
                position=gate[batch,i,j:].sum().clamp(max=p-1)
                low=int(position.floor());high=int(position.ceil());w=position-low
                vec=(1-w)*embedding[0,:,low]+w*embedding[0,:,high]
                cols.append((q[batch,i]*vec).sum())
            rr.append(torch.stack(cols))
        rows.append(torch.stack(rr))
    bias=torch.stack(rows)
    return torch.softmax(logits+bias,-1)@v

def cope_tests(ns):
    CoPE=ns['CoPE'];SelfAttn=ns['SelfAttn'];torch.manual_seed(281)
    q=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True);k=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True);v=torch.randn(2,4,2,dtype=torch.float64,requires_grad=True)
    mask=torch.ones(4,4,dtype=torch.float64).tril();model=SelfAttn(8,3).double()
    zero=model(q,k,v,mask);ordinary=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(3)+mask.log(),-1)@v
    assert torch.equal(zero,ordinary)
    # 零表时CoPE的gate路径无梯度；位置表自身仍可获得梯度。
    raw_logits=torch.randn(1,2,4,dtype=torch.float64,requires_grad=True);query=torch.randn(1,2,3,dtype=torch.float64,requires_grad=True)
    module=CoPE(8,3).double();bias=module(query,raw_logits);grads=torch.autograd.grad(bias.sum(),(raw_logits,module.pos_emb))
    assert torch.equal(grads[0],torch.zeros_like(grads[0])) and grads[1].abs().sum()>0
    with torch.no_grad():model.cope.pos_emb.copy_(torch.randn_like(model.cope.pos_emb)*.2);module.pos_emb.copy_(torch.randn_like(module.pos_emb)*.2)
    nonzero_gate_grad=torch.autograd.grad(module(query,raw_logits).sum(),raw_logits)[0];assert nonzero_gate_grad.abs().max()>0
    actual=model(q,k,v,mask);expected=dense_reference(q,k,v,model.cope.pos_emb,mask)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    inputs=(q,k,v,model.cope.pos_emb);weight=torch.randn_like(actual)
    ag=torch.autograd.grad((actual*weight).sum(),inputs);eg=torch.autograd.grad((expected*weight).sum(),inputs)
    gradient_errors=[]
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11);gradient_errors.append(float((a-b).abs().max()))
    # 用原类functional_call核对四种输入的数值梯度，避开整数插值结点与饱和边界。
    small=SelfAttn(8,2).double();sq=torch.randn(1,3,2,dtype=torch.float64,requires_grad=True)*.2;sk=torch.randn_like(sq,requires_grad=True)*.2;sv=torch.randn(1,3,2,dtype=torch.float64,requires_grad=True)
    emb=(torch.randn(1,2,8,dtype=torch.float64)*.1).requires_grad_();sm=torch.ones(3,3,dtype=torch.float64).tril()
    call=lambda a,b,c,e:torch.func.functional_call(small,{'cope.pos_emb':e},(a,b,c,sm))
    gradcheck=torch.autograd.gradcheck(call,(sq,sk,sv,emb),eps=1e-6,atol=2e-5,rtol=1e-4);assert gradcheck
    # 因果性与完整前缀重算，不声称实现了KV cache。
    with torch.no_grad():
        all_out=model(q,k,v,mask);prefix_errors=[]
        for end in range(1,5):
            pre=model(q[:,:end],k[:,:end],v[:,:end],mask[:end,:end]);prefix_errors.append(float((pre-all_out[:,:end]).abs().max()))
        changed_q=q.detach().clone();changed_k=k.detach().clone();changed_v=v.detach().clone()
        changed_q[:,2:]+=100;changed_k[:,2:]-=100;changed_v[:,2:]+=100
        modified=model(changed_q,changed_k,changed_v,mask);assert torch.equal(modified[:,:2],all_out[:,:2])
    for err in prefix_errors:assert err<1e-12
    # 仅优化原始位置表，明确为教学微训练，不冒充论文LM或Flip-Flop。
    train=SelfAttn(8,3).double();optimizer=torch.optim.Adam(train.parameters(),lr=.02)
    target=torch.randn(2,4,2,dtype=torch.float64);losses=[];before=train.cope.pos_emb.detach().clone()
    for step in range(5):
        optimizer.zero_grad();pred=train(q.detach(),k.detach(),v.detach(),mask);loss=F.mse_loss(pred,target);losses.append(float(loss.detach()));loss.backward();assert torch.isfinite(train.cope.pos_emb.grad).all();optimizer.step()
    assert not torch.equal(before,train.cope.pos_emb) and losses[-1]<losses[0]
    # 饱和、整数结点、错误mask与非法表大小均真实调用原实现。
    saturated=CoPE(1,2).double()
    with torch.no_grad():saturated.pos_emb.fill_(1)
    s_logits=torch.ones(1,1,3,dtype=torch.float64,requires_grad=True);s_query=torch.ones(1,1,2,dtype=torch.float64)
    sbias=saturated(s_query,s_logits);sgrad=torch.autograd.grad(sbias.sum(),s_logits)[0];assert torch.equal(sgrad,torch.zeros_like(sgrad))
    node=CoPE(4,1).double()
    with torch.no_grad():node.pos_emb.copy_(torch.tensor([[[0.,1.,3.,5.]]],dtype=torch.float64))
    # 两个sigmoid(0)=.5，第一位置和恰为1，ceil=floor时本代码选0斜率。
    nlogits=torch.zeros(1,1,2,dtype=torch.float64,requires_grad=True);nquery=torch.ones(1,1,1,dtype=torch.float64)
    deriv=torch.autograd.grad(node(nquery,nlogits)[0,0,0],nlogits)[0];assert torch.equal(deriv,torch.zeros_like(deriv))
    boundaries={}
    for name,fn in [('all_masked',lambda:model(q,k,v,torch.zeros_like(mask))),('additive_mask_in_wrong_api',lambda:model(q,k,v,torch.where(mask.bool(),0.,-float('inf')))),('npos_zero',lambda:CoPE(0,3).double()(q,torch.zeros(2,4,4,dtype=torch.float64))),('wrong_query_dim',lambda:CoPE(8,4).double()(q,torch.zeros(2,4,4,dtype=torch.float64)))]:
        try:result=fn();boundaries[name]={'finite':bool(torch.isfinite(result).all()),'shape':list(result.shape)}
        except Exception as e:boundaries[name]={'exception':type(e).__name__,'message':str(e)}
    assert boundaries['all_masked']['finite'] is False
    wrong=boundaries['additive_mask_in_wrong_api'];assert wrong.get('finite') is False or 'exception' in wrong
    return dict(zero_initialized_equals_plain_attention=True,zero_table_gate_gradient_zero=True,zero_table_embedding_gradient_nonzero=True,learned_table_gate_gradient_max=float(nonzero_gate_grad.abs().max()),dense_vector_reference_max_error=float((actual-expected).abs().max()),query_key_value_table_gradient_max_errors=gradient_errors,double_gradcheck=gradcheck,causal_prefix_max_errors=prefix_errors,future_inputs_do_not_change_prefix=True,small_training={'parameter_count':sum(p.numel() for p in train.parameters()),'optimizer':'Adam','updates':5,'losses':losses,'position_table_changed':True,'only_teaching_random_target':True},npos_one_saturates_and_gate_gradient_zero=True,integer_knot_selected_gate_derivative=deriv.tolist(),boundaries=boundaries)

def main():
    torch.set_num_threads(1);raw,ann,static=audit();a,ta=run(raw);b,tb=run(ann);assert ta==tb
    spec=importlib.util.spec_from_file_location('cope_reading',FOLDER/'CoPE核心源码_代码注释.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    assert tree(__import__('inspect').getsource(module.CoPE))==tree(ast.get_source_segment(blocks(src(raw[39]))[0],ast.parse(blocks(src(raw[39]))[0]).body[0]))
    static['standalone_core_python_import']=True
    ra=rope_tests(a);rb=rope_tests(b);assert ra==rb
    ca=cope_tests(a);cb=cope_tests(b);assert ca==cb
    report={'status':'passed','preservation':static,'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'nbformat':nbformat.__version__,'cuda_available':torch.cuda.is_available(),'threads':1,'external_seed_original_cells':28},'original_trace':ta,'annotated_trace':tb,'all25_original_code_cells_literal':True,'Markdown39_classes_executed_both_versions':True,'all_traces_and_diagnostics_equal':True,'RoPE':ra,'CoPE':ca,'not_run':['Remote image cell4 rendering/content','BERT/GPT Markdown5 or partial Llama Markdown9 as full pretrained model','Official CoPE full repository, pretrained model or language quality','Paper selective-copy/counting/Flip-Flop/LM training benchmarks','GPU, multihead/full Transformer, KV cache or speed/memory benchmark'],'scripts_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    (ROOT/'复现/outputs/delivery_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':report['status'],'static':static,'CoPE':ca},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
