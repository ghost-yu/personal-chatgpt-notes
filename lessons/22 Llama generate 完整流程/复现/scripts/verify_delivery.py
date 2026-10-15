# -*- coding: utf-8 -*-
"""Audit preserved cells and execute complete fixed Meta generation with real tiny CPU models."""
import argparse
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
import tempfile
import textwrap
import tokenize
import types
from pathlib import Path

import nbformat
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / '代码/tutorials'

def sha(b): return hashlib.sha256(b).hexdigest()
def dump(n): return ast.dump(n, include_attributes=False)
def significant(s):
    return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline)
            if t.type not in (tokenize.COMMENT,tokenize.NL)]
def extract(src,node):
    start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    return textwrap.dedent(''.join(src.splitlines(keepends=True)[start-1:node.end_lineno]))

def audit(sources):
    manifest=json.loads((ROOT/'来源清单.json').read_bytes())
    for e in manifest['files']:
        b=(ROOT/e['published_path']).read_bytes()
        assert len(b)==e['bytes'] and sha(b)==e['sha256']
    for name,h in manifest['official_source_sha256'].items(): assert sha((sources/name).read_bytes())==h
    raw,explained,annotated=[json.loads((CODE/n).read_bytes()) for n in
        ('llama2_src_generate.ipynb','原代码_带说明.ipynb','代码注释.ipynb')]
    for n in (raw,explained,annotated): nbformat.validate(nbformat.from_dict(n))
    assert len(raw['cells'])==17 and len(explained['cells'])==len(annotated['cells'])==35
    for n in (explained,annotated):
        assert {k:v for k,v in n.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
    code=[]
    for i,c in enumerate(raw['cells']):
        assert explained['cells'][2*i+2]==c
        a=annotated['cells'][2*i+2]
        assert {k:v for k,v in a.items() if k!='source'}=={k:v for k,v in c.items() if k!='source'}
        if c['cell_type']=='code':
            s=''.join(c['source']);t=''.join(a['source'])
            assert dump(ast.parse(s))==dump(ast.parse(t)) and significant(s)==significant(t)
            code.append(s)
        else: assert a==c
    assert dump(ast.parse('\n'.join(code)))==dump(ast.parse((CODE/'llama2_src_generate_代码注释.py').read_text(encoding='utf-8')))
    snippets=(CODE/'Markdown片段_代码注释.py').read_text(encoding='utf-8')
    for i in (4,10,13,16):
        s=textwrap.dedent(''.join(raw['cells'][i]['source']).split('```')[1].strip('\n')+'\n')
        t=snippets.split(f'# %% 原Markdown {i}\n')[1].split('# %%')[0]
        assert dump(ast.parse(s))==dump(ast.parse(t)) and significant(s.strip()+'\n')==significant(t.strip()+'\n')
    src=(sources/'llama-generation.py').read_text(encoding='utf-8')
    orig=ast.parse(src);readsrc=(CODE/'generate核心源码_代码注释.py').read_text(encoding='utf-8');read=ast.parse(readsrc)
    oc=next(n for n in orig.body if isinstance(n,ast.ClassDef) and n.name=='Llama')
    rc=next(n for n in read.body if isinstance(n,ast.ClassDef))
    for method in rc.body:
        match=next(n for n in oc.body if getattr(n,'name',None)==method.name)
        assert dump(method)==dump(match)
        assert significant(extract(src,match))==significant(extract(readsrc,method))
    of=next(n for n in orig.body if getattr(n,'name',None)=='sample_top_p')
    rf=next(n for n in read.body if getattr(n,'name',None)=='sample_top_p')
    assert dump(of)==dump(rf) and significant(extract(src,of))==significant(extract(readsrc,rf))
    return raw,annotated,{'cells':17,'code_cells':4,'schemas':3,'derived_cells_each':35,
        'markdown_snippets':[4,10,13,16],'official_complete_methods':['__init__','generate','text_completion','sample_top_p'],
        'preserved':'raw bytes / metadata / outputs / AST / noncomment tokens'}

def run_cells(nb):
    ns={};out=[];old=os.getcwd()
    try:
        os.chdir(CODE)
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            s=''.join(c['source']);tree=ast.parse(s);buf=io.StringIO();value=None
            with contextlib.redirect_stdout(buf):
                if tree.body and isinstance(tree.body[-1],ast.Expr):
                    last=tree.body.pop();exec(compile(tree,'<notebook>','exec'),ns)
                    value=eval(compile(ast.Expression(last.value),'<notebook>','eval'),ns)
                else:exec(compile(tree,'<notebook>','exec'),ns)
            out.append({'stdout':buf.getvalue(),'value':value})
    finally: os.chdir(old)
    assert ns['prompt_tokens']==[1,306,4658,278,6593,310,2834,338]
    assert out[2]['stdout']=='1\n2\n-1\n32000\n'
    return out

class DeviceOnly(ast.NodeTransformer):
    def __init__(self,mode): self.mode=mode;self.count=0
    def visit_Call(self,node):
        self.generic_visit(node)
        if self.mode=='model' and isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            assert isinstance(node.func.value,ast.Call) and isinstance(node.func.value.func,ast.Attribute) and node.func.value.func.attr=='zeros'
            node.func.attr='to';node.args=[ast.Constant('cpu')];self.count+=1
        if self.mode=='generation':
            for k in node.keywords:
                if k.arg=='device' and isinstance(k.value,ast.Constant) and k.value.value=='cuda':
                    k.value=ast.Constant('cpu');self.count+=1
        return node

def load(name,src):
    m=types.ModuleType(name);m.__file__=name+'.py';sys.modules[name]=m
    exec(compile(ast.fix_missing_locations(src),m.__file__,'exec'),m.__dict__)
    return m

def error_of(fn):
    try:fn()
    except Exception as e:return {'type':type(e).__name__,'message':str(e)[:500]}
    raise AssertionError('Expected real exception')

def observe(engine,**kwargs):
    """Observe genuine forward and generate frames; neither tokens nor logits are replaced."""
    calls=[];steps=[];fwd=engine.model.forward;oldtrace=sys.gettrace()
    def forward(ids,pos):
        assert (ids>=0).all(), 'PAD was sent into embedding'
        y=fwd(ids,pos)
        calls.append({'start':pos,'input':ids.tolist(),'shape':list(ids.shape),'logits_shape':list(y.shape),
                      'requires_grad':y.requires_grad})
        return y
    def tracer(frame,event,arg):
        if event=='line' and frame.f_code.co_name=='generate' and frame.f_globals.get('__name__')==engine.__class__.__module__:
            loc=frame.f_locals
            if frame.f_lineno==engine._observe_line:
                ids=loc['next_token'];cur=loc['cur_pos'];z=loc['logits'][:,-1]
                steps.append({'cur':cur,'prev':loc['prev_pos'],'chosen':ids.tolist(),
                    'protected':loc['input_text_mask'][:,cur].tolist(),'eos':loc['eos_reached'].tolist(),
                    'tokens':loc['tokens'].tolist(),'actual_logp':F.log_softmax(z,dim=-1).gather(1,ids[:,None]).squeeze(1).tolist(),
                    'recorded_logp':loc['token_logprobs'][:,cur].tolist() if kwargs.get('logprobs') else None,
                    'last_logits':z.detach().clone()})
        return tracer
    engine.model.forward=forward
    try:
        sys.settrace(tracer);result=engine.generate(**kwargs)
    finally:
        sys.settrace(oldtrace);engine.model.forward=fwd
    assert len(calls)==len(steps) and all(not c['requires_grad'] for c in calls)
    return result,calls,steps

def gen_line(tree):
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Llama')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='cur_pos')
    line=next(n.lineno for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='prev_pos')
    return loop,line

def serial_steps(steps):return [{k:v for k,v in s.items() if k!='last_logits'} for s in steps]

def verify_runtime(sources):
    sys.path.insert(0,str(sources/'fairscale-local'))
    import fairscale.nn.model_parallel.initialize as mp
    pkg=types.ModuleType('llama');pkg.__path__=[];sys.modules['llama']=pkg
    modeltree=ast.parse((sources/'llama-model.py').read_text(encoding='utf-8'))
    mt=DeviceOnly('model');mt.visit(modeltree);assert mt.count==2
    model=load('llama.model',modeltree)
    tokenizer=load('llama.tokenizer',ast.parse((sources/'llama-tokenizer.py').read_text(encoding='utf-8')))
    gentree=ast.parse((sources/'llama-generation.py').read_text(encoding='utf-8'))
    gt=DeviceOnly('generation');gt.visit(gentree);assert gt.count==3
    _,line=gen_line(gentree);gen=load('llama.generation',gentree)
    fixedtree=copy.deepcopy(gentree);loop,_=gen_line(fixedtree)
    block=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='logprobs')
    loop.body.remove(block)
    write=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript))
    loop.body.insert(write+1,block)
    fixed=load('llama.generation_teaching_fix',fixedtree)
    previous_cwd=os.getcwd()
    try:
        os.chdir(CODE)
        tok=tokenizer.Tokenizer('tokenizer.model')
    finally: os.chdir(previous_cwd)
    assert (tok.bos_id,tok.eos_id,tok.pad_id,tok.n_words)==(1,2,-1,32000)
    def make(controlled=False):
        torch.manual_seed(22)
        args=model.ModelArgs(dim=16,n_layers=2,n_heads=4,n_kv_heads=2,vocab_size=32000,multiple_of=4,max_batch_size=2,max_seq_len=32)
        m=model.Transformer(args).eval()
        with torch.no_grad():
            for n,p in m.named_parameters():
                if controlled:p.zero_()
                elif p.ndim==1:p.fill_(1.)
                else:p.normal_(0,.02)
            if controlled:
                # Zero attention/MLP, explicit residual embeddings and real output projection.
                m.norm.weight.fill_(1.)
                m.tok_embeddings.weight[3,0]=1.;m.tok_embeddings.weight[6,0]=1.
                m.tok_embeddings.weight[4,1]=1.;m.tok_embeddings.weight[2,1]=1.
                m.output.weight[2,0]=1.;m.output.weight[6,1]=1.
        return m
    def engine(m,cls=gen.Llama):
        e=cls(m,tok);e._observe_line=line;return e
    with tempfile.TemporaryDirectory(prefix='lesson22-') as td:
        torch.distributed.init_process_group('gloo',init_method=(Path(td)/'rendezvous').as_uri(),rank=0,world_size=1)
        mp.initialize_model_parallel(1)
        try:
            m=make();e=engine(m);params=sum(p.numel() for p in m.parameters())
            prompts=[[1,7],[1,8,9,10]]
            kw=dict(prompt_tokens=prompts,max_gen_len=3,temperature=0,top_p=.9,logprobs=False,echo=False)
            result,calls,steps=observe(e,**kw)
            assert [c['shape'][1] for c in calls]==[2,1,1,1,1]
            assert result[1] is None and [len(t) for t in result[0]]==[3,3]
            assert steps[0]['protected']==steps[1]['protected']==[False,True]
            ref=make();reference_error=0.
            with torch.inference_mode():
                for s in steps:
                    prefix=torch.tensor(s['tokens'])[:,:s['cur']]
                    z=ref(prefix,0)[:,-1];reference_error=max(reference_error,(z-s['last_logits']).abs().max().item())
                    assert torch.allclose(z,s['last_logits'],atol=2e-6,rtol=1e-5)
                    greedy=z.argmax(-1).tolist()
                    for b,protected in enumerate(s['protected']):
                        assert s['chosen'][b]==(prompts[b][s['cur']] if protected else greedy[b])
            individual=[e.generate([p],3,temperature=0)[0][0] for p in prompts]
            assert individual==result[0]
            echo=e.generate(prompts,3,temperature=0,echo=True)[0]
            assert echo==[p+t for p,t in zip(prompts,result[0])]
            old,_,oldsteps=observe(e,**{**kw,'logprobs':True})
            repaired,_,fixsteps=observe(engine(m,fixed.Llama),**{**kw,'logprobs':True})
            assert old[0]==repaired[0]==result[0]
            for s in oldsteps:
                for b,protected in enumerate(s['protected']):
                    if not protected: assert s['recorded_logp'][b]==0 and s['actual_logp'][b]<-1
            for s in fixsteps:
                assert torch.allclose(torch.tensor(s['recorded_logp']),torch.tensor(s['actual_logp']),atol=2e-6)
            assert e.generate(prompts,3,temperature=-1,top_p=-100)[0]==result[0]
            assert e.generate(prompts,3,temperature=.6,top_p=0)[0]==result[0]
            torch.manual_seed(220);sample_a=e.generate(prompts,3,temperature=.7,top_p=.8)
            torch.manual_seed(220);sample_b=e.generate(prompts,3,temperature=.7,top_p=.8)
            assert sample_a==sample_b
            eos_result,eos_calls,eos_steps=observe(engine(make(True)),prompt_tokens=[[1,3],[1,4]],max_gen_len=4,temperature=0,logprobs=True)
            assert eos_result[0]==[[],[6]] and len(eos_calls)==2
            assert [s['chosen'] for s in eos_steps]==[[2,6],[6,2]]
            assert [s['eos'] for s in eos_steps]==[[True,False],[True,True]]
            empty,emptycalls,_=observe(e,prompt_tokens=[[1,7]],max_gen_len=0,temperature=0)
            assert empty==([[]],None) and not emptycalls
            zero_unequal,zero_calls,_=observe(e,prompt_tokens=prompts,max_gen_len=0,temperature=0)
            assert zero_unequal==([[],[]],None) and len(zero_calls)==2
            prompt_eos=engine(make(True)).generate([[1,2,3]],1,temperature=0,echo=True)[0]
            assert prompt_eos==[[1]]
            capped,capcalls,_=observe(e,prompt_tokens=[[1]+[7]*29],max_gen_len=8,temperature=0)
            assert len(capped[0][0])==2 and [c['shape'][1] for c in capcalls]==[30,1]
            errors={
                'empty_batch':error_of(lambda:e.generate([],1)),
                'flat_prompt':error_of(lambda:e.generate([1,7],1)),
                'over_batch':error_of(lambda:e.generate([[1]]*3,1)),
                'over_context':error_of(lambda:e.generate([[1]*33],1)),
                'negative_budget':error_of(lambda:e.generate([[1,7]],-1)),
                'empty_prompt_row':error_of(lambda:e.generate([[]],1)),
            }
            assert errors['empty_batch']['type']=='ValueError' and errors['flat_prompt']['type']=='TypeError'
            text=e.text_completion(['I believe the meaning of life is'],temperature=0,max_gen_len=2,logprobs=True,echo=True)
            assert len(text)==1 and len(text[0]['tokens'])==len(text[0]['logprobs'])==10
            # Literal original Markdown snippets, with all missing owner/context supplied explicitly.
            raw=json.loads((CODE/'llama2_src_generate.ipynb').read_bytes())
            def snippet(i):return textwrap.dedent(''.join(raw['cells'][i]['source']).split('```')[1].strip('\n')+'\n')
            ns=dict(self=e,prompt_tokens=[[1,306,4658,278,6593,310,2834,338]],max_gen_len=2,temperature=0,top_p=.9,logprobs=False,echo=False)
            exec(snippet(4),ns);assert len(ns['generation_tokens'][0])==2
            literal_calls=[];saved=m.forward
            def literal_fwd(ids,pos):
                literal_calls.append([pos,ids.shape[1]]);return saved(ids,pos)
            m.forward=literal_fwd
            try:
                ns=dict(self=e,tokens=torch.tensor([[1,7,8,9,10]]),prev_pos=0,min_prompt_len=2,total_len=5)
                with torch.inference_mode():exec(snippet(10),ns)
            finally:m.forward=saved
            assert literal_calls==[[0,2],[0,3],[0,4]]
            attention=m.layers[0].attention
            ns=dict(self=attention,bsz=1,start_pos=0,seqlen=4)
            exec(snippet(13),ns);assert list(ns['keys'].shape)==[1,4,2,4]
            for temp,shape in ((0,[1]),(.6,[1,1])):
                ns=dict(torch=torch,logits=torch.tensor([[[2.,1.,0.]]]),temperature=temp,top_p=.9,sample_top_p=gen.sample_top_p)
                exec(snippet(16),ns);assert list(ns['next_token'].shape)==shape
            # Observe actual multinomial input, retaining its original randomness and output.
            sampling=[];saved_sample=torch.multinomial
            def capture(probs,*a,**k):
                sampling.append(probs.clone().tolist());return saved_sample(probs,*a,**k)
            p=torch.tensor([[.3,.5,.2]],dtype=torch.float64);unchanged=p.clone();boundary={}
            torch.multinomial=capture
            try:
                for threshold in (.49,.5,0.,1.):
                    torch.manual_seed(22);chosen=gen.sample_top_p(p,threshold)
                    boundary[str(threshold)]={'sorted_normalized':sampling[-1],'chosen_original_id':chosen.tolist()}
                    if threshold in (.49,0.):assert chosen.item()==1 and sampling[-1]==[[1.,0.,0.]]
                assert sampling[1][0][1]>0 and sampling[1][0][2]==0
                negative_p=error_of(lambda:gen.sample_top_p(p,-.1))
            finally:torch.multinomial=saved_sample
            assert torch.equal(p,unchanged)
            entropy=[]
            for temp in (.5,1.,2.):
                prob=F.softmax(torch.tensor([2.,1.,0.],dtype=torch.float64)/temp,dim=-1)
                entropy.append({'T':temp,'p':prob.tolist(),'entropy':-(prob*prob.log()).sum().item()})
            assert entropy[0]['entropy']<entropy[1]['entropy']<entropy[2]['entropy']
            return {'cpu_changes':{'model_cache_cuda_to_cpu':2,'generation_device_cuda_to_cpu':3,
                    'teaching_logprobs_fix':'move one complete CE if-block after tokens[:,cur_pos] assignment; original untouched'},
                'tiny_model':{'dim':16,'layers':2,'q_heads':4,'kv_heads':2,'vocab':32000,'max_seq_len':32,'batch_capacity':2,'parameters':params,'init':'seed22 normal(0,.02), norm1; eval'},
                'unequal_prompt':{'prompts':prompts,'output':result,'calls':calls,'steps':serial_steps(steps),'individual_same':True,'prefix_reference_max_logit_error':reference_error},
                'echo':echo,'old_logprobs':{'output':old,'steps':serial_steps(oldsteps)},
                'teaching_fixed_logprobs':{'output':repaired,'steps':serial_steps(fixsteps)},
                'sampling_seed_replay':{'seed':220,'output':sample_a},'negative_T_greedy_and_top_p0_matches_greedy':True,
                'controlled_real_weights_EOS':{'output':eos_result,'calls':eos_calls,'steps':serial_steps(eos_steps)},
                'zero_budget':{'equal_output':empty,'equal_calls':emptycalls,'unequal_output':zero_unequal,'unequal_calls':zero_calls},
                'prompt_EOS_echo_crops_prompt':prompt_eos,'context_clip':{'output':capped,'calls':capcalls},
                'invalid_inputs':errors,'text_completion':text,'literal_markdown':{'4':'actual generate with explicit batch context','10_calls':literal_calls,'13_shape':[1,4,2,4],'16_branch_shapes':[[1],[1,1]]},
                'nucleus_boundary':boundary,'negative_top_p':negative_p,'temperature_entropy':entropy,
                'limits':'Random/controlled tiny real Transformer, actual FairScale/Gloo world1; no original checkpoint, build/NCCL, CUDA, multi-rank, language-quality benchmark or training.'}
        finally:
            mp.destroy_model_parallel();torch.distributed.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True);a=p.parse_args();sources=Path(a.sources)
    torch.set_num_threads(1)
    raw,annotated,static=audit(sources);r=run_cells(raw);ann=run_cells(annotated);assert r==ann
    runtime=verify_runtime(sources)
    files=[Path(__file__),Path(__file__).with_name('build_delivery.py'),*CODE.glob('*.py'),*CODE.glob('*.ipynb'),CODE/'tokenizer.model']
    result={'status':'passed','static':static,'original_and_annotated_execution':r,'runtime':runtime,
        'environment':{'platform':platform.platform(),'python':sys.version,'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
          'versions':{name:importlib.metadata.version(name) for name in ('sentencepiece','nbformat','IPython')},'fairscale':'0.4.13 local target'},
        'artifact_sha256':{str(f.relative_to(ROOT)).replace('\\','/'):sha(f.read_bytes()) for f in files}}
    out=ROOT/'复现/outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'runtime_verification.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'status':'passed','static':static,'parameters':runtime['tiny_model']['parameters'],
        'max_logit_error':runtime['unequal_prompt']['prefix_reference_max_logit_error'],'eos':runtime['controlled_real_weights_EOS']['output']},ensure_ascii=True))

if __name__=='__main__': main()
