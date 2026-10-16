"""保留性审计、完整meta计数、真实本地小Llama加载/前向/梯度/训练/缓存。"""
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
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize

import nbformat
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama as hf

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'代码/tutorials/llama'
REPORT={}
torch.set_num_threads(1)
torch.manual_seed(29)

def dump_ast(s):return ast.dump(ast.parse(s),include_attributes=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tokens(s):
    return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline)
            if t.type not in [tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,tokenize.ENCODING,tokenize.ENDMARKER]]
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def error(fn):
    try:fn()
    except Exception as e:return {'exception':type(e).__name__,'message':str(e)[:400]}
    raise AssertionError('Expected an actual exception')

def preservation():
    original=json.loads((FOLDER/'llama2_7b_llama3_8b.ipynb').read_bytes())
    manifest=json.loads((ROOT/'来源清单.json').read_text(encoding='utf-8'))
    raw=subprocess.check_output(['git','-C','G:/zzz/Llama 入门教程/代码/personal_chatgpt','show',manifest['upstream_commit']+':'+manifest['upstream_path']])
    assert raw==(FOLDER/'llama2_7b_llama3_8b.ipynb').read_bytes()
    assert sha(FOLDER/'llama2_7b_llama3_8b.ipynb')==manifest['original_sha256']
    nbformat.validate(nbformat.reads(raw.decode(),as_version=4))
    annotated=None
    for filename in ['代码注释.ipynb','原代码_带说明.ipynb']:
        d=json.loads((FOLDER/filename).read_text(encoding='utf-8'));nbformat.validate(nbformat.from_dict(d))
        assert len(d['cells'])==49
        for i,c in enumerate(original['cells']):
            n=d['cells'][2*i+2];assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in n.items() if k!='source'}
            a=''.join(c['source']);b=''.join(n['source'])
            if filename=='原代码_带说明.ipynb':assert a==b
            elif c['cell_type']=='code':assert dump_ast(a)==dump_ast(b) and tokens(a)==tokens(b)
            else:
                aa=re.findall(r'```[^\n]*\n(.*?)```',a,re.S);bb=re.findall(r'```[^\n]*\n(.*?)```',b,re.S)
                assert len(aa)==len(bb)
                for x,y in zip(aa,bb):assert dump_ast(x)==dump_ast(y) and tokens(x)==tokens(y)
        if filename=='代码注释.ipynb':annotated=d
    source='\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type']=='code')
    assert dump_ast(source)==dump_ast((FOLDER/'llama2_7b_llama3_8b_代码注释.py').read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('lecture29_core',FOLDER/'Llama核心实现_代码注释.py')
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    for name,h in manifest['official_object_source_sha256'].items():
        s=inspect.getsource(getattr(hf,name));assert hashlib.sha256(s.encode()).hexdigest()==h
        assert dump_ast(s)==dump_ast(inspect.getsource(getattr(core,name)))
    dist=metadata.distribution('transformers');entry=next(x for x in dist.files if str(x).replace('\\','/')=='transformers/models/llama/modeling_llama.py')
    assert base64.urlsafe_b64encode(hashlib.sha256(Path(dist.locate_file(entry)).read_bytes()).digest()).rstrip(b'=').decode()==entry.hash.value
    REPORT['preservation']={'cells':24,'code_cells':sum(c['cell_type']=='code' for c in original['cells']),
      'schemas':3,'derived_cells':49,'original_git_bytes':True,'metadata_historical_outputs':True,
      'code_ast_tokens':True,'markdown_blocks_9_21_ast_tokens':True,'reading_ast':True,
      'official_complete_objects':list(manifest['official_object_source_sha256']),
      'official_installed_record_match':True,'core_import':True,'modeling_llama_sha256':sha(Path(hf.__file__))}
    return original,annotated,core

def full_config(which):
    old=which==2
    return LlamaConfig(vocab_size=32000 if old else 128256,hidden_size=4096,
      intermediate_size=11008 if old else 14336,num_hidden_layers=32,num_attention_heads=32,
      num_key_value_heads=32 if old else 8,tie_word_embeddings=False,
      attention_bias=False,mlp_bias=False,rms_norm_eps=1e-6 if old else 1e-5,
      rope_theta=10000.0 if old else 500000.0,max_position_embeddings=4096 if old else 8192)

def breakdown(model):
    d={'embedding':model.model.embed_tokens.weight.numel(),'lm_head':model.lm_head.weight.numel(),
       'attention':sum(p.numel() for l in model.model.layers for p in l.self_attn.parameters()),
       'mlp':sum(p.numel() for l in model.model.layers for p in l.mlp.parameters()),
       'norms':model.model.norm.weight.numel()+sum(p.numel() for l in model.model.layers for n in [l.input_layernorm,l.post_attention_layernorm] for p in n.parameters())}
    assert sum(d.values())==sum(p.numel() for p in model.parameters())
    return d

def execute_notebook(nb,mode,paths=None):
    ns={'__name__':'__lecture29__'};traces=[]
    env={k:os.environ.get(k) for k in ['http_proxy','https_proxy']}
    try:
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            src=''.join(c['source']);status='literal'
            if i==4 and mode=='tiny':
                src=f'llama2_id={str(paths[0])!r}\nllama3_id={str(paths[1])!r}\n';status='local_id_variant'
            if i in [5,7] and mode=='meta':
                which=2 if i==5 else 3;name='llama'+str(which)
                ns['cfg']=full_config(which)
                with torch.device('meta'):ns[name]=AutoModelForCausalLM.from_config(ns['cfg'],attn_implementation='sdpa')
                traces.append({'cell':i,'status':'explicit_from_config_meta_variant','stdout':'','display':None});continue
            tree=ast.parse(src);last=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
            buf=io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(tree,f'cell_{i}','exec'),ns)
                value=eval(compile(ast.Expression(last.value),f'cell_{i}','eval'),ns) if last else None
            display={'type':'Image','url':value.url} if value is not None and type(value).__name__=='Image' else repr(value) if value is not None else None
            traces.append({'cell':i,'status':status,'stdout':buf.getvalue(),'display':display})
    finally:
        for k,v in env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    return ns,traces

def run_meta(original,annotated):
    a,ta=execute_notebook(original,'meta');b,tb=execute_notebook({'cells':annotated['cells'][2::2]},'meta')
    assert ta==tb
    allstats={}
    for n,expected in [(2,6738415616),(3,8030261248)]:
        model=a['llama'+str(n)];assert all(p.device.type=='meta' for p in model.parameters())
        d=breakdown(model);assert sum(d.values())==expected
        allstats[str(n)]={'total':expected,'breakdown':d,'bf16_weights_bytes':2*expected,
          'bf16_weights_GiB':2*expected/1024**3,'config':full_config(n).to_dict()}
        assert d==breakdown(b['llama'+str(n)])
    delta={k:allstats['3']['breakdown'][k]-v for k,v in allstats['2']['breakdown'].items()}
    assert sum(delta.values())==1291845632
    REPORT['full_meta']={'models':allstats,'parameter_delta':delta,'traces':ta,'original_annotated_equal':True,
                        'not_weight_inference':True}

def tiny_config(which):
    return LlamaConfig(vocab_size=64 if which==2 else 128,hidden_size=32,
      intermediate_size=86 if which==2 else 112,num_hidden_layers=2,num_attention_heads=4,
      num_key_value_heads=4 if which==2 else 1,tie_word_embeddings=False,
      max_position_embeddings=64,bos_token_id=1,eos_token_id=2,pad_token_id=0,attention_dropout=0.0,
      rms_norm_eps=1e-6 if which==2 else 1e-5,rope_theta=10000.0 if which==2 else 500000.0)

def tensor_equal(a,b):return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
def clone_state(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}

def train_model(model):
    # 原notebook不含训练；以下是明确的随机数据补充，FP32改善CPU数值诊断。
    model.float().train();x=torch.tensor([[1,3,4,5,6,7],[1,8,9,10,11,12]])
    initial=clone_state(model);opt=torch.optim.AdamW(model.parameters(),lr=1e-3);losses=[]
    for _ in range(3):
        opt.zero_grad();out=model(x,labels=x,use_cache=False);out.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step();losses.append(out.loss.item())
    assert losses[-1]<losses[0]
    final=clone_state(model);changed=sum(not torch.equal(initial[k],v) for k,v in final.items());assert changed
    model.eval()
    with torch.no_grad():
        out=model(x,use_cache=False);manual=F.cross_entropy(out.logits[:,:-1].reshape(-1,model.config.vocab_size),x[:,1:].reshape(-1))
        assert torch.allclose(manual,model(x,labels=x).loss)
        pre=model(x[:,:3],use_cache=True);cache=pre.past_key_values
        assert list(cache[0][0].shape)==[2,model.config.num_key_value_heads,3,8]
        errors=[]
        for t in range(3,6):
            step=model(x[:,t:t+1],past_key_values=cache,use_cache=True);cache=step.past_key_values
            full=model(x[:,:t+1],use_cache=False);err=(step.logits[:,-1]-full.logits[:,-1]).abs().max().item();errors.append(err)
            assert err<1e-6
        greedy=model.generate(x[:1,:3],max_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=None)
    return final,{'parameters':sum(p.numel() for p in model.parameters()),'AdamW_updates':3,'losses':losses,
      'changed_state_entries':changed,'shift_loss_match':True,'cache_shape':list(cache[0][0].shape),
      'cached_prefix_logits_max_errors':errors,'greedy_ids':greedy.tolist()}

def run_tiny(original,annotated):
    with tempfile.TemporaryDirectory(prefix='lecture29_') as tmp:
        paths=[Path(tmp)/'small2',Path(tmp)/'small3']
        for n,p in zip([2,3],paths):
            torch.manual_seed(290+n);LlamaForCausalLM(tiny_config(n)).save_pretrained(p)
        a,ta=execute_notebook(original,'tiny',paths);b,tb=execute_notebook({'cells':annotated['cells'][2::2]},'tiny',paths)
        assert ta==tb
        result={}
        for n in [2,3]:
            ma,mb=a['llama'+str(n)],b['llama'+str(n)]
            assert tensor_equal(ma.state_dict(),mb.state_dict())
            assert all(p.dtype==torch.bfloat16 and p.device.type=='cpu' for p in ma.parameters())
            with torch.no_grad():assert torch.isfinite(ma(torch.tensor([[1,3,4]])).logits).all()
            device_map={str(k):str(v) for k,v in ma.hf_device_map.items()}
            sa,ra=train_model(ma);sb,rb=train_model(mb);assert tensor_equal(sa,sb) and ra==rb
            outdir=Path(tmp)/('trained'+str(n));ma.save_pretrained(outdir)
            loaded=AutoModelForCausalLM.from_pretrained(outdir,local_files_only=True).eval()
            assert tensor_equal(ma.state_dict(),loaded.state_dict())
            with torch.no_grad():assert torch.equal(ma(torch.tensor([[1,3,4]])).logits,loaded(torch.tensor([[1,3,4]])).logits)
            result[str(n)]={**ra,'local_original_loading_dtype':'bfloat16','hf_device_map':device_map,
              'bf16_cpu_forward_finite':True,'original_annotated_state_exact':True,'save_reload_exact':True}
        result['traces']=ta;result['local_id_cell4_only_variant']=True
        REPORT['tiny_real']=result
        REPORT['boundaries']={'empty_parameter_count':error(lambda:a['print_trainable_parameters'](nn.Identity())),
            'strict_false_does_not_ignore_shape':error(lambda:a['llama2'].load_state_dict({'lm_head.weight':torch.zeros(2,32)},strict=False)),
            'out_of_range_token':error(lambda:a['llama2'](torch.tensor([[64]])))}

def core_checks(core):
    # 官方完整eager Attention 对照独立逐query头循环，包含Q/K位置旋转与所有投影。
    torch.manual_seed(299);stats={}
    for G in [4,2,1]:
        cfg=LlamaConfig(hidden_size=16,intermediate_size=24,num_attention_heads=4,num_key_value_heads=G,
                        num_hidden_layers=1,attention_dropout=0.0);cfg._attn_implementation='eager'
        module=core.LlamaAttention(cfg,layer_idx=0).double().eval()
        x=torch.randn(2,3,16,dtype=torch.double,requires_grad=True)
        pos=torch.arange(3)[None].expand(2,-1);phase=pos.double()[:,:,None]*torch.tensor([1.,.01],dtype=torch.double)
        cos=torch.cat([phase.cos(),phase.cos()],-1);sin=torch.cat([phase.sin(),phase.sin()],-1)
        mask=torch.full((3,3),float('-inf'),dtype=torch.double).triu(1)[None,None]
        got=module(x,attention_mask=mask,position_ids=pos,position_embeddings=(cos,sin),output_attentions=True)[0]
        q=F.linear(x,module.q_proj.weight).reshape(2,3,4,4).transpose(1,2)
        k=F.linear(x,module.k_proj.weight).reshape(2,3,G,4).transpose(1,2)
        v=F.linear(x,module.v_proj.weight).reshape(2,3,G,4).transpose(1,2)
        def rotate(t):return t*cos[:,None]+torch.cat([-t[...,2:],t[...,:2]],-1)*sin[:,None]
        q,k=rotate(q),rotate(k);heads=[]
        for h in range(4):
            g=h//(4//G);score=q[:,h]@k[:,g].transpose(-1,-2)/2+mask[:,0]
            prob=score.float().softmax(-1).double();heads.append(prob@v[:,g])
        ref=F.linear(torch.stack(heads,2).reshape(2,3,16),module.o_proj.weight)
        assert torch.allclose(got,ref,atol=1e-12,rtol=1e-12)
        inputs=(x,*module.parameters());gg=torch.autograd.grad(got.square().sum(),inputs,retain_graph=True);gr=torch.autograd.grad(ref.square().sum(),inputs)
        graderr=max((a-b).abs().max().item() for a,b in zip(gg,gr));assert graderr<1e-10
        sdpa=core.LlamaSdpaAttention(cfg,layer_idx=0).double().eval();sdpa.load_state_dict(module.state_dict())
        ys=sdpa(x.detach(),attention_mask=mask,position_ids=pos,position_embeddings=(cos,sin))[0]
        sdpaerr=(ys-got.detach()).abs().max().item();assert sdpaerr<1e-6
        stats[str(G)]={'parameters':sum(p.numel() for p in module.parameters()),
          'independent_forward_max_error':(got-ref).abs().max().item(),'all_input_weight_gradient_max_error':graderr,
          'sdpa_eager_max_error':sdpaerr,'group_mapping':[h//(4//G) for h in range(4)]}
    mlp=core.LlamaMLP(cfg).double();x=torch.randn(2,3,16,dtype=torch.double,requires_grad=True)
    got=mlp(x);gate=x@mlp.gate_proj.weight.T;up=x@mlp.up_proj.weight.T
    ref=((gate*torch.sigmoid(gate))*up)@mlp.down_proj.weight.T
    gg=torch.autograd.grad(got.square().sum(),(x,*mlp.parameters()),retain_graph=True)
    gr=torch.autograd.grad(ref.square().sum(),(x,*mlp.parameters()))
    mlpgrad=max((a-b).abs().max().item() for a,b in zip(gg,gr));assert mlpgrad<1e-12
    assert torch.allclose(got,ref,atol=1e-14,rtol=1e-14)
    check=torch.autograd.gradcheck(lambda xx,wg,wu,wd:torch.func.functional_call(mlp,
        {'gate_proj.weight':wg,'up_proj.weight':wu,'down_proj.weight':wd},(xx,)),
        (x,*[mlp.gate_proj.weight,mlp.up_proj.weight,mlp.down_proj.weight]),fast_mode=True)
    REPORT['core']={'attention':stats,'mlp':{'parameters':sum(p.numel() for p in mlp.parameters()),
      'forward_max_error':(got-ref).abs().max().item(),'all_input_weight_gradient_max_error':mlpgrad,'double_gradcheck':check}}
    REPORT['boundaries']['hidden_not_divisible']=error(lambda:core.LlamaAttention(LlamaConfig(hidden_size=15,num_attention_heads=4,num_key_value_heads=2),layer_idx=0))
    bad=core.LlamaAttention(LlamaConfig(hidden_size=16,num_attention_heads=4,num_key_value_heads=3),layer_idx=0).double()
    REPORT['boundaries']['heads_not_divisible_by_kv']=error(lambda:bad(torch.randn(1,3,16,dtype=torch.double),position_ids=torch.arange(3)[None]))

def main():
    original,annotated,core=preservation();run_meta(original,annotated);run_tiny(original,annotated);core_checks(core)
    REPORT['environment']={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
      'transformers':transformers.__version__,'accelerate':metadata.version('accelerate'),'nbformat':metadata.version('nbformat'),
      'cuda_available':torch.cuda.is_available(),'threads':torch.get_num_threads()}
    import psutil
    REPORT['environment']['physical_RAM_bytes']=psutil.virtual_memory().total
    REPORT['environment']['available_RAM_bytes_at_end']=psutil.virtual_memory().available
    REPORT['environment']['workspace_disk_free_bytes_at_end']=shutil.disk_usage(ROOT).free
    REPORT['not_run']=['Original remote 7B/8B checkpoint loading or inference','Two RTX4090 device_map behavior',
      'Remote image content','Full model training, benchmarks, language quality','Markdown2 Meta checkpoint loading and Markdown9 undefined-context fragment']
    REPORT['scripts_sha256']={p.name:sha(p) for p in (ROOT/'复现/scripts').glob('*.py')}
    REPORT['status']='passed';(ROOT/'复现/outputs').mkdir(exist_ok=True);save(ROOT/'复现/outputs/delivery_verification.json',REPORT)
    print(json.dumps({'status':'passed','preservation':REPORT['preservation'],'delta':REPORT['full_meta']['parameter_delta'],
                     'tiny':{k:v for k,v in REPORT['tiny_real'].items() if k not in ['traces']},'core':REPORT['core'],'boundaries':REPORT['boundaries']},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
