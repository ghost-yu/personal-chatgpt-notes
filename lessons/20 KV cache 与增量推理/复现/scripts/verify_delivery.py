# -*- coding: utf-8 -*-
"""Audit exact originals/comment-only derivatives and run genuine Meta/FairScale and GPT2 CPU paths."""
import argparse
import ast
import contextlib
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import tokenize
import types
from pathlib import Path

import nbformat
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'代码/tutorials'
META_SHA='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'

def sha(data): return hashlib.sha256(data).hexdigest()
def dump(tree): return ast.dump(tree,include_attributes=False)
def tokens(src):
    return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type not in (tokenize.COMMENT,tokenize.NL)]

def audit():
    manifest=json.loads((ROOT/'来源清单.json').read_bytes())
    for item in manifest['files']:
        data=(ROOT/item['published_path']).read_bytes()
        assert sha(data)==item['sha256'] and len(data)==item['bytes']
    original=json.loads((CODE/'llama2_src_cache_kv.ipynb').read_bytes())
    explained=json.loads((CODE/'原代码_带说明.ipynb').read_bytes())
    annotated=json.loads((CODE/'代码注释.ipynb').read_bytes())
    for nb in (original,explained,annotated): nbformat.validate(nbformat.from_dict(nb))
    count=0
    for derived in (explained,annotated):
        assert len(derived['cells'])==51
        assert {k:v for k,v in derived.items() if k!='cells'}=={k:v for k,v in original.items() if k!='cells'}
    for i,c in enumerate(original['cells']):
        assert explained['cells'][2+2*i]==c
        actual=annotated['cells'][2+2*i]
        assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in actual.items() if k!='source'}
        src=''.join(c['source']); dst=''.join(actual['source'])
        if c['cell_type']=='code':
            assert dump(ast.parse(src))==dump(ast.parse(dst))
            assert tokens(src)==tokens(dst)
            count+=1
        else: assert c==actual
    joined='\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type']=='code')
    assert dump(ast.parse(joined))==dump(ast.parse((CODE/'llama2_src_cache_kv_代码注释.py').read_text(encoding='utf-8')))
    excerpt=''.join(original['cells'][18]['source']).split('```')[1].strip('\n')+'\n'
    read=(CODE/'Attention摘录_代码注释.py').read_text(encoding='utf-8')
    assert dump(ast.parse(excerpt))==dump(ast.parse(read)) and tokens(excerpt)==tokens(read)
    return {'original_cells':25,'code_cells':count,'markdown_cells':11,
            'derivative_cells':[51,51],'schema_count':3,'all_original_metadata_and_outputs_preserved':True,
            'all_code_ast_and_noncomment_tokens_equal':True,'markdown18_excerpt_ast_equal':True,
            'reading_python_ast_equal':True,'files_hashed':len(manifest['files'])}, original,annotated

class CpuCache(ast.NodeTransformer):
    def __init__(self): self.changes=0
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            assert isinstance(node.func.value,ast.Call) and isinstance(node.func.value.func,ast.Attribute)
            assert node.func.value.func.attr=='zeros'
            node.func.attr='to';node.args=[ast.Constant('cpu')]; self.changes+=1
        return node

class ChunkMask(ast.NodeTransformer):
    def __init__(self): self.changes=0
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='mask' for t in node.targets):
            if 'diagonal=start_pos + 1' in ast.unparse(node.value):
                new=ast.parse('mask = torch.cat([torch.zeros((1, 1, seqlen, start_pos), device=tokens.device), torch.triu(mask, diagonal=1)], dim=-1).type_as(h)').body[0]
                self.changes+=1
                return ast.copy_location(new,node)
        return node

def module(src,name,cpu=False,chunk=False):
    tree=ast.parse(src);changes=[]
    if cpu:
        adapt=CpuCache();tree=adapt.visit(tree);assert adapt.changes==2;changes.append('exactly two zeros.cuda() -> zeros.to(cpu)')
    if chunk:
        adapt=ChunkMask();tree=adapt.visit(tree);assert adapt.changes==1;changes.append('exactly one Transformer.forward mask assignment')
    ast.fix_missing_locations(tree)
    m=types.ModuleType(name);sys.modules[name]=m
    exec(compile(tree,name,'exec'),m.__dict__)
    return m,changes

def meta_tests(source,fairscale):
    assert sha(source.read_bytes())==META_SHA
    sys.path.insert(0,str(fairscale))
    import fairscale.nn.model_parallel.initialize as fs
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='lesson20-gloo-') as directory:
        torch.distributed.init_process_group('gloo',init_method=(Path(directory)/'rendezvous').as_uri(),rank=0,world_size=1)
        fs.initialize_model_parallel(1)
        try:
            return meta_inner(source)
        finally:
            fs.destroy_model_parallel();torch.distributed.destroy_process_group()

def meta_inner(source):
    src=source.read_text(encoding='utf-8')
    official=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='Attention')
    reading=ast.parse((CODE/'Attention官方源码_代码注释.py').read_text(encoding='utf-8')).body[0]
    assert dump(official)==dump(reading)
    assert tokens(ast.get_source_segment(src,official)+'\n')==tokens((CODE/'Attention官方源码_代码注释.py').read_text(encoding='utf-8'))
    raw,_=module(src,'lesson20_meta_original')
    conf=dict(dim=16,n_layers=2,n_heads=4,n_kv_heads=2,vocab_size=32,multiple_of=4,max_batch_size=2,max_seq_len=16)
    failure=None
    try: raw.Transformer(raw.ModelArgs(**conf))
    except (AssertionError,RuntimeError) as e: failure={'type':type(e).__name__,'message':str(e)}
    assert failure and not torch.cuda.is_available()
    cpu,changes=module(src,'lesson20_meta_cpu',cpu=True)
    torch.manual_seed(20)
    cached=cpu.Transformer(cpu.ModelArgs(**conf)).eval()
    # Official init_method=lambda x:x does not initialize weights; explicitly seed every parameter.
    with torch.no_grad():
        for p in cached.parameters():
            if p.ndim==1:p.fill_(1)
            else:p.normal_(0,0.1)
    reference=cpu.Transformer(cpu.ModelArgs(**conf)).eval();reference.load_state_dict(cached.state_dict())
    ids=torch.tensor([[1,2,3,4,5,6],[1,7,8,9,10,11]],dtype=torch.long)
    hooks=[];calls={'cached':[],'reference':[]};context={'mode':'cached'}
    for label,model in [('cached',cached),('reference',reference)]:
        for layer_index,layer in enumerate(model.layers):
            for projection in ('wq','wk','wv'):
                def hook(mod,args, label=label,layer_index=layer_index,projection=projection):
                    calls[label].append({'layer':layer_index,'projection':projection,'shape':list(args[0].shape)})
                hooks.append(getattr(layer.attention,projection).register_forward_pre_hook(hook))
    comparisons=[];max_error=0.0
    for end in (3,4,5,6):
        start=0 if end==3 else end-1
        previous=[(l.attention.cache_k[:,:start].clone(),l.attention.cache_v[:,:start].clone()) for l in cached.layers]
        new=cached(ids[:,start:end],start)
        dense=reference(ids[:,:end],0)
        expected=dense if start==0 else dense[:,-1:]
        error=float((new-expected).abs().max()); assert torch.allclose(new,expected,atol=1e-6,rtol=1e-5)
        max_error=max(max_error,error)
        for layer_index,(a,b) in enumerate(zip(cached.layers,reference.layers)):
            a=a.attention;b=b.attention
            for field in ('cache_k','cache_v'):
                assert torch.allclose(getattr(a,field)[:,:end],getattr(b,field)[:,:end],atol=1e-6,rtol=1e-5)
                assert torch.count_nonzero(getattr(a,field)[:,end:])==0
            if start:
                assert torch.equal(a.cache_k[:,:start],previous[layer_index][0])
                assert torch.equal(a.cache_v[:,:start],previous[layer_index][1])
        comparisons.append({'start_pos':start,'input_shape':list(ids[:,start:end].shape),'logits_shape':list(new.shape),'max_logit_error':error})
    for h in hooks:h.remove()
    assert [c['shape'][1] for c in calls['cached'] if c['layer']==0 and c['projection']=='wk']==[3,1,1,1]
    assert [c['shape'][1] for c in calls['reference'] if c['layer']==0 and c['projection']=='wk']==[3,4,5,6]
    # Independent score/value route: compare raw post-RoPE K and unrotated V from projection hooks.
    attention=cached.layers[0].attention; hidden=torch.randn(2,1,16);pos=6
    capture={}
    handles=[getattr(attention,name).register_forward_hook(lambda mod,args,out,name=name:capture.__setitem__(name,out.detach().clone())) for name in ('wq','wk','wv')]
    f=cached.freqs_cis[pos:pos+1]
    out=attention(hidden,pos,f,None)
    for h in handles:h.remove()
    q=capture['wq'].view(2,1,4,4);k=capture['wk'].view(2,1,2,4);v=capture['wv'].view(2,1,2,4)
    q,k=cpu.apply_rotary_emb(q,k,f)
    assert torch.equal(attention.cache_k[:,pos:pos+1],k) and torch.equal(attention.cache_v[:,pos:pos+1],v)
    ks=attention.cache_k[:,:pos+1].repeat_interleave(2,dim=2).transpose(1,2)
    vs=attention.cache_v[:,:pos+1].repeat_interleave(2,dim=2).transpose(1,2)
    scores=(q.transpose(1,2)@ks.transpose(-1,-2))/2
    independent=attention.wo((scores.softmax(-1)@vs).transpose(1,2).contiguous().view(2,1,16))
    assert torch.allclose(out,independent,atol=1e-6,rtol=1e-5)
    cache_bytes=sum(x.numel()*x.element_size() for l in cached.layers for x in (l.attention.cache_k,l.attention.cache_v))
    assert cache_bytes==4096
    assert not any('cache' in key for key in cached.state_dict())
    assert not any('cache' in key for key,_ in cached.named_buffers())
    # Fixed old source fails on P>0, S>1; do not silently repair it.
    chunk_failure=None
    cached(ids[:,:3],0)
    try:cached(ids[:,3:5],3)
    except RuntimeError as e:chunk_failure={'type':type(e).__name__,'message':str(e)}
    assert chunk_failure
    repaired,repair_changes=module(src,'lesson20_meta_chunk',cpu=True,chunk=True)
    chunk=repaired.Transformer(repaired.ModelArgs(**conf)).eval();chunk.load_state_dict(cached.state_dict())
    chunk_errors=[]
    for start,end in ((0,3),(3,5),(5,6)):
        y=chunk(ids[:,start:end],start);expected=reference(ids[:,:end],0)[:,start:end]
        err=float((y-expected).abs().max());assert torch.allclose(y,expected,atol=1e-6,rtol=1e-5);chunk_errors.append(err)
    # Causality: changing future token cannot affect earlier position logits.
    altered=ids.clone();altered[:,4:]=torch.tensor([[20,21],[22,23]])
    original_prefix=reference(ids,0)[:,:4].clone(); changed_prefix=reference(altered,0)[:,:4].clone()
    assert torch.equal(original_prefix,changed_prefix)
    # New session only reads the effective prefix, even if old capacity tail is dirty.
    clean=reference(ids[:,:3],0).clone()
    for l in reference.layers:
        l.attention.cache_k[:,3:].fill_(123);l.attention.cache_v[:,3:].fill_(456)
    assert torch.equal(clean,reference(ids[:,:3],0))
    # Reusing mismatched prefixes and jumping positions gives real wrong outputs.
    wrong=cpu.Transformer(cpu.ModelArgs(**conf)).eval();wrong.load_state_dict(cached.state_dict())
    wrong(ids[:,:3].flip(1),0); mismatch=wrong(ids[:,3:4],3)
    correct=reference(ids[:,:4],0)[:,-1:]
    mismatch_error=float((mismatch-correct).abs().max());assert mismatch_error>1e-5
    fresh=cpu.Transformer(cpu.ModelArgs(**conf)).eval();fresh.load_state_dict(cached.state_dict())
    gap_error=float((fresh(ids[:,3:4],3)-correct).abs().max());assert gap_error>1e-5
    capacity_failure=None
    try:fresh(torch.ones(2,2,dtype=torch.long),15)
    except RuntimeError as e:capacity_failure={'type':type(e).__name__,'message':str(e)}
    assert capacity_failure
    # Greedy generation consumes prior sampled tokens one at a time; no fake generation stub.
    seq=ids[:,:3].clone();generated=[];gen_errors=[]
    for step in range(4):
        start=0 if step==0 else seq.shape[1]-1
        small=cached(seq[:,start:],start)[:,-1];dense=reference(seq,0)[:,-1]
        assert torch.allclose(small,dense,atol=1e-6,rtol=1e-5)
        assert torch.equal(small.argmax(-1),dense.argmax(-1))
        nxt=small.argmax(-1);generated.append(nxt.tolist());gen_errors.append(float((small-dense).abs().max()))
        seq=torch.cat([seq,nxt[:,None]],dim=1)
    return {'official_sha256':META_SHA,'official_commit':'7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e',
            'cuda_original_failure':failure,'cpu_changes':changes,'config':conf,
            'official_attention_reading_ast_and_tokens_equal':True,
            'parameter_count':sum(p.numel() for p in cached.parameters()),'cache_bytes':cache_bytes,
            'active_prefix_6_bytes':2*2*2*6*2*4*4,'cache_absent_from_state_dict_and_buffers':True,
            'incremental_comparisons':comparisons,'projection_call_shapes':calls,
            'post_rope_k_unrotated_v_equal':True,'independent_attention_error':float((out-independent).abs().max()),
            'chunk_original_failure':chunk_failure,'chunk_teaching_changes':repair_changes,'chunk_errors':chunk_errors,
            'causal_future_invariance_exact':True,'dirty_tail_new_session_equal':True,
            'mismatched_prefix_error':mismatch_error,'unfilled_gap_error':gap_error,'capacity_failure':capacity_failure,
            'greedy_new_ids':generated,'greedy_errors':gen_errors,'world_size':1,'backend':'gloo'}

def snapshot(value):
    from IPython.display import Image
    if value is None:return None
    if isinstance(value,Image):return {'type':'Image','sha256':sha(value.data),'bytes':len(value.data),'width':value.width}
    if isinstance(value,list):return value
    return {'type':type(value).__name__,'vocab_size':getattr(value,'vocab_size',None)}

def run_cells(nb,model_dir,annotated=False):
    namespace={};trace=[];generated=[];forward_lengths=[];old=os.getcwd()
    try:
        os.chdir(CODE)
        for i in range(25):
            cell=nb['cells'][2+2*i] if annotated else nb['cells'][i]
            if cell['cell_type']!='code':continue
            if i==20:
                trace.append({'cell':i,'status':'skipped','reason':'preserve process proxy environment'});continue
            src=''.join(cell['source']);tree=ast.parse(src);changes=[]
            if i==21:
                replacements=0
                for node in ast.walk(tree):
                    if isinstance(node,ast.Constant) and node.value=='gpt2':node.value=str(model_dir);replacements+=1
                assert replacements==2;changes=['exactly two loading strings -> local random tinyGPT2 checkpoint']
            if i==23:
                replacements=[]
                for node in ast.walk(tree):
                    if isinstance(node,ast.Constant) and type(node.value)==int and node.value in (10,1000):
                        replacements.append(node.value);node.value={10:2,1000:16}[node.value]
                assert sorted(replacements)==[10,1000];changes=['range(10)->range(2)','max_new_tokens=1000->16']
                model=namespace['model'];original_generate=model.generate
                def capture(*args,**kwargs):
                    output=original_generate(*args,**kwargs);generated.append({'use_cache':kwargs['use_cache'],'ids':output.tolist()});return output
                model.generate=capture
                handle=model.transformer.h[0].attn.c_attn.register_forward_pre_hook(lambda mod,args:forward_lengths.append(list(args[0].shape)))
            ast.fix_missing_locations(tree)
            last=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(compile(tree,f'original-cell-{i}','exec'),namespace)
                result=eval(compile(ast.Expression(last.value),f'original-cell-{i}-last','eval'),namespace) if last else None
            if i==23:handle.remove()
            trace.append({'cell':i,'status':'executed','scope':'teaching_variant' if changes else 'whole_original_cell',
                          'changes':changes,'result':snapshot(result),'stdout':stdout.getvalue()})
    finally:os.chdir(old)
    return {'cells':trace,'generations':generated,'projection_input_shapes':forward_lengths}

def hf_tests(tokenizer_json,original,annotated):
    from transformers import GPT2Config,GPT2LMHeadModel,GPT2Tokenizer,AutoTokenizer
    assert sha(tokenizer_json.read_bytes())=='8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
    with tempfile.TemporaryDirectory(prefix='lesson20-gpt2-') as directory:
        base=Path(directory);token=json.loads(tokenizer_json.read_bytes())
        (base/'vocab.json').write_bytes(json.dumps(token['model']['vocab'],ensure_ascii=False).encode('utf-8'))
        (base/'merges.txt').write_bytes(('#version: 0.2\n'+'\n'.join(' '.join(m) if isinstance(m,list) else m for m in token['model']['merges'])+'\n').encode('utf-8'))
        GPT2Tokenizer(vocab_file=str(base/'vocab.json'),merges_file=str(base/'merges.txt'),model_max_length=64).save_pretrained(base)
        torch.manual_seed(20)
        config=GPT2Config(vocab_size=50257,n_positions=64,n_embd=16,n_layer=2,n_head=2,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)
        config._attn_implementation='eager'
        model=GPT2LMHeadModel(config).eval();model.save_pretrained(base)
        model.generation_config.pad_token_id=50256
        model.generation_config.save_pretrained(base)
        left=run_cells(original,base);right=run_cells(annotated,base,True)
        for a,b in zip(left['cells'],right['cells']):
            assert {k:v for k,v in a.items() if k!='stdout'}=={k:v for k,v in b.items() if k!='stdout'}
        assert left['generations']==right['generations'] and left['projection_input_shapes']==right['projection_input_shapes']
        assert left['cells'][-2]['cell']==23
        input_ids=AutoTokenizer.from_pretrained(str(base))('What is KV caching?',return_tensors='pt')['input_ids']
        plen=input_ids.shape[1]
        assert len(left['generations'])==4
        for g in left['generations']:assert len(g['ids'][0])==plen+16
        assert all(g['ids']==left['generations'][0]['ids'] for g in left['generations'])
        with torch.inference_mode():
            full=model(input_ids,use_cache=True)
            past=full.past_key_values
            assert isinstance(past,tuple) and len(past)==2
            nxt=full.logits[:,-1].argmax(-1)[:,None]
            mask=torch.ones(1,plen+1,dtype=torch.long)
            inc=model(nxt,past_key_values=past,attention_mask=mask,use_cache=True)
            dense=model(torch.cat([input_ids,nxt],1),attention_mask=mask,use_cache=False)
            error=float((inc.logits[:,-1]-dense.logits[:,-1]).abs().max())
            assert torch.allclose(inc.logits[:,-1],dense.logits[:,-1],atol=1e-6,rtol=1e-5)
            assert dense.past_key_values is None
        shape=[list(pair[0].shape) for pair in past]
        for p in past:assert p[0].shape==p[1].shape==(1,2,plen,8)
        no_cache=[v[1] for v in left['projection_input_shapes'][32:]]
        cached=[v[1] for v in left['projection_input_shapes'][:32]]
        assert cached==([plen]+[1]*15)*2
        assert no_cache==list(range(plen,plen+16))*2
        return {'tokenizer_json_sha256':sha(tokenizer_json.read_bytes()),'tokenizer_vocab_size':50257,
                'random_model':True,'model_config':config.to_dict(),'parameter_count':sum(p.numel() for p in model.parameters()),
                'prompt_ids':input_ids.tolist(),'legacy_past_type':'tuple[layer][K,V]','past_shapes':shape,
                'one_token_logit_error':error,'no_cache_past_is_none':True,
                'original_execution':left,'annotated_execution':right,'contract_results_equal_except_wallclock_stdout':True,
                'all_four_generations_equal':True,'executed_cells_per_notebook':13,'whole_original_cells_per_notebook':11,
                'skipped_cells':[20],'variant_cells':[21,23],'benchmark_claim':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--meta-source',required=True);parser.add_argument('--fairscale',required=True);parser.add_argument('--tokenizer-json',required=True)
    args=parser.parse_args();sys.stdout.reconfigure(encoding='utf-8')
    static,original,annotated=audit()
    meta=meta_tests(Path(args.meta_source),Path(args.fairscale))
    hf=hf_tests(Path(args.tokenizer_json),original,annotated)
    environment={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
                 'packages':{name:importlib.metadata.version(name) for name in ('fairscale','transformers','tokenizers','sentencepiece','numpy','nbformat')}}
    class MemoryStatus(ctypes.Structure):
        _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
    if sys.platform=='win32':
        memory=MemoryStatus();memory.dwLength=ctypes.sizeof(memory)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        environment['ram_bytes']=memory.ullTotalPhys;environment['available_ram_bytes']=memory.ullAvailPhys
    environment['disk_free_bytes']=shutil.disk_usage(ROOT).free
    report={'status':'部分成功','static':static,'environment':environment,'meta':meta,'transformers':hf,
            'script_sha256':{p.name:sha(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')},
            'reading_artifact_sha256':{p.name:sha(p.read_bytes()) for p in CODE.glob('*代码注释.py')},
            'not_run':['original GPT2 pretrained weights','original 1000-new-token x10 benchmark','Llama2 pretrained weights','CUDA','model_parallel_world_size>1','model quality or speed benchmark']}
    output=ROOT/'复现/outputs';output.mkdir(parents=True,exist_ok=True)
    (output/'完整交付验证.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'static':static,'meta_max_logit_error':max(x['max_logit_error'] for x in meta['incremental_comparisons']),
                      'chunk_original_failure':meta['chunk_original_failure'],'chunk_errors':meta['chunk_errors'],
                      'hf_logit_error':hf['one_token_logit_error'],'executed_per_notebook':13,'passed':True},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
