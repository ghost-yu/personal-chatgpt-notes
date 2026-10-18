"""Run the note's examples against complete fixed Meta modules on CPU."""
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(os.environ.get('LESSON22_META_SOURCES',
    'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料'))
EXPECTED = {
    'llama-generation.py': 'acc739aeda4aa222305ca5e7ae020df38d0fa841530d966f041c13e065db05c4',
    'llama-model.py': '56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754',
    'llama-tokenizer.py': 'a675762526cd5f6e9d742a642e33a3560ea825522b8a9efb531d36b7c7a69e6c',
}
LOADER = ROOT.parent/'20 KV cache 与增量推理/复现/scripts/verify_foundation_examples.py'
spec = importlib.util.spec_from_file_location('_lesson22_model_loader', LOADER)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, tree, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


class GenerationCPU(ast.NodeTransformer):
    def __init__(self):
        self.changes = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        for keyword in node.keywords:
            if keyword.arg == 'device' and isinstance(keyword.value, ast.Constant) and keyword.value.value == 'cuda':
                keyword.value.value = 'cpu'
                self.changes += 1
        return node


def generate_node(tree):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Llama')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')


def move_logprob_block(tree):
    loop = next(n for n in generate_node(tree).body if isinstance(n, ast.For) and ast.unparse(n.target) == 'cur_pos')
    block = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'logprobs')
    loop.body.remove(block)
    write = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'tokens[:, cur_pos]')
    loop.body.insert(write + 1, block)
    return tree


@contextlib.contextmanager
def generation_session():
    """Keep complete source files; adapt 2 model allocations and 3 generation devices."""
    for name, digest in EXPECTED.items():
        assert sha(SOURCES/name) == digest, name
    names = ['llama', 'llama.model', 'llama.tokenizer', 'llama.generation', 'llama.generation_teaching_fix']
    saved = {name:sys.modules.get(name) for name in names}
    previous_env = {name:os.environ.get(name) for name in ['LESSON20_META_SOURCE','LESSON20_FAIRSCALE']}
    os.environ['LESSON20_META_SOURCE'] = str(SOURCES/'llama-model.py')
    os.environ['LESSON20_FAIRSCALE'] = str(SOURCES/'fairscale-local')
    try:
        with loader.cpu_llama_session() as model, tempfile.TemporaryDirectory(prefix='lesson22-tokenizer-') as td:
            pkg = types.ModuleType('llama'); pkg.__path__ = []
            sys.modules['llama'] = pkg
            sys.modules['llama.model'] = model
            tokmod = load('llama.tokenizer', ast.parse((SOURCES/'llama-tokenizer.py').read_text(encoding='utf-8')),
                          SOURCES/'llama-tokenizer.py')
            tree = ast.parse((SOURCES/'llama-generation.py').read_text(encoding='utf-8'))
            adapter = GenerationCPU();tree = adapter.visit(tree)
            assert adapter.changes == 3
            gen = load('llama.generation', tree, SOURCES/'llama-generation.py')
            fixed = load('llama.generation_teaching_fix', move_logprob_block(copy.deepcopy(tree)),
                         ROOT/'复现/scripts/教学移动logprobs内存版本.py')
            target = Path(td)/'tokenizer.model'
            shutil.copyfile(ROOT/'代码/tutorials/tokenizer.model', target)
            tokenizer = tokmod.Tokenizer(str(target))
            assert (tokenizer.bos_id,tokenizer.eos_id,tokenizer.pad_id,tokenizer.n_words) == (1,2,-1,32000)
            yield model, gen, fixed, tokenizer
    finally:
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
        for name,value in previous_env.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def make_model(module, controlled=False):
    torch.manual_seed(22)
    args = module.ModelArgs(dim=16,n_layers=2,n_heads=4,n_kv_heads=2,vocab_size=32000,
                           multiple_of=4,max_batch_size=2,max_seq_len=32)
    model = module.Transformer(args).eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if controlled:p.zero_()
            elif p.ndim == 1:p.fill_(1.)
            else:p.normal_(0,.02)
        if controlled:
            model.norm.weight.fill_(1.)
            model.tok_embeddings.weight[3,0]=1.;model.tok_embeddings.weight[6,0]=1.
            model.tok_embeddings.weight[4,1]=1.;model.tok_embeddings.weight[2,1]=1.
            model.output.weight[2,0]=1.;model.output.weight[6,1]=1.
    return model


def trace_generate(engine, **kwargs):
    """Record genuine forward calls and post-write state without changing results."""
    method = engine.generate.__func__.__wrapped__
    # The moved block keeps original locations, so both versions share this event.
    original_tree = ast.parse((SOURCES/'llama-generation.py').read_text(encoding='utf-8'))
    loop = next(n for n in generate_node(original_tree).body if isinstance(n,ast.For) and ast.unparse(n.target)=='cur_pos')
    line = next(n.lineno for n in loop.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='prev_pos')
    calls=[];steps=[];saved_forward=engine.model.forward;previous_trace=sys.gettrace()

    def forward(ids,position):
        assert (ids>=0).all(), 'PAD must never reach embedding'
        logits=saved_forward(ids,position)
        calls.append({'start':position,'input':ids.tolist(),'shape':list(ids.shape),
                      'logits_shape':list(logits.shape),'grad_enabled':torch.is_grad_enabled(),
                      '_logits':logits[:,-1].detach().clone()})
        return logits

    def tracer(frame,event,arg):
        if frame.f_code is method.__code__ and event=='line' and frame.f_lineno==line:
            local=frame.f_locals;cur=local['cur_pos']
            step={'cur':cur,'prev':local['prev_pos'],'tokens':local['tokens'].tolist(),
                  'chosen':local['next_token'].tolist(),
                  'protected':local['input_text_mask'][:,cur].tolist(),
                  'eos':local['eos_reached'].tolist()}
            if local['logprobs']:
                step['recorded_logp']=local['token_logprobs'][:,cur].tolist()
                step['actual_logp']=F.log_softmax(calls[-1]['_logits'],-1).gather(
                    1,local['next_token'][:,None]).flatten().tolist()
            steps.append(step)
        return tracer

    engine.model.forward=forward;sys.settrace(tracer)
    try:result=engine.generate(**kwargs)
    finally:engine.model.forward=saved_forward;sys.settrace(previous_trace)
    assert len(calls)==len(steps)
    return result,calls,steps


def function_ast(node):
    node=copy.deepcopy(node)
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
        node.body.pop(0)
    return ast.dump(node,include_attributes=False)


def additional_checks(scope):
    reference_error=0.
    with generation_session() as (module,gen,fixed,tok):
        model=make_model(module);snapshot={name:p.clone()for name,p in model.state_dict().items()}
        engine=gen.Llama(model,tok)
        prompts=[[1,7],[1,8,9,10]]
        kw=dict(prompt_tokens=prompts,max_gen_len=3,temperature=0,echo=False)
        result,calls,steps=trace_generate(engine,**kw)
        reference=make_model(module)
        with torch.inference_mode():
            for step,call in zip(steps,calls):
                prefix=torch.tensor(step['tokens'])[:,:step['cur']]
                logits=reference(prefix,0)[:,-1]
                error=(logits-call['_logits']).abs().max().item();reference_error=max(reference_error,error)
                assert torch.allclose(logits,call['_logits'],atol=2e-6,rtol=1e-5)
                expected=logits.argmax(-1).tolist()
                for b,protected in enumerate(step['protected']):
                    assert step['chosen'][b] == (prompts[b][step['cur']] if protected else expected[b])
        assert [c['start']for c in calls]==[0,2,3,4,5]
        assert [c['shape'][1]for c in calls]==[2,1,1,1,1]
        assert [s['protected']for s in steps]==[[False,True],[False,True],[False,False],[False,False],[False,False]]
        assert result[0]==[[17528,7163,2814],[4553,4870,27233]]
        assert engine.generate(prompts,3,temperature=0,echo=True)[0]==[p+t for p,t in zip(prompts,result[0])]
        assert [engine.generate([p],3,temperature=0)[0][0]for p in prompts]==result[0]
        original,_,old_steps=trace_generate(engine,**kw,logprobs=True)
        repaired,_,new_steps=trace_generate(fixed.Llama(model,tok),**kw,logprobs=True)
        assert original[0]==repaired[0]==result[0]
        for old,new in zip(old_steps,new_steps):
            for b,protected in enumerate(old['protected']):
                if not protected:assert old['recorded_logp'][b]==0 and old['actual_logp'][b]<-1
            assert torch.allclose(torch.tensor(new['recorded_logp']),torch.tensor(new['actual_logp']),atol=2e-6)
        eos,eos_calls,eos_steps=trace_generate(gen.Llama(make_model(module,True),tok),
            prompt_tokens=[[1,3],[1,4]],max_gen_len=4,temperature=0)
        assert eos[0]==[[],[6]]
        assert [s['chosen']for s in eos_steps]==[[2,6],[6,2]]
        assert [s['eos']for s in eos_steps]==[[True,False],[True,True]]
        assert len(eos_calls)==2
        text=engine.text_completion(['I believe the meaning of life is'],max_gen_len=2,temperature=0,echo=True)
        assert isinstance(text[0]['generation'],str)
        assert engine.generate(prompts,3,temperature=-1,top_p=-100)[0]==result[0]
        torch.manual_seed(220);sample1=engine.generate(prompts,3,temperature=.7,top_p=.8)
        torch.manual_seed(220);sample2=engine.generate(prompts,3,temperature=.7,top_p=.8)
        assert sample1==sample2
        assert len(engine.generate([[1]+[7]*29],8,temperature=0)[0][0])==2
        assert engine.generate([[1,7]],0,temperature=0)==([[]],None)
        for name,p in model.state_dict().items():assert torch.equal(snapshot[name],p),name
        assert all(p.grad is None for p in model.parameters())
    # A separate arithmetic construction checks support without observing or replacing multinomial.
    top_p_cases=[]
    for probability,threshold,expected_ids in [
        ([.3,.5,.2],.6,[1,0]),([.3,.5,.2],.49,[1]),
        ([.5,.25,.125,.125],.5,[0,1]),([.25]*4,.9,[0,1,2,3])]:
        p=torch.tensor([probability],dtype=torch.float64);unchanged=p.clone()
        order=sorted(range(len(probability)),key=lambda i:-probability[i])
        retained=[];mass=0.
        for i in order:
            if mass<=threshold:retained.append(i)
            mass+=probability[i]
        assert retained==expected_ids
        torch.manual_seed(220)
        repeated=p.expand(4096,-1).clone()
        before_sampling=repeated.clone()
        selected=scope['sample_top_p'](repeated,threshold).flatten()
        assert torch.equal(repeated,before_sampling)
        observed=sorted(set(selected.tolist()))
        assert observed==sorted(retained)
        counts=torch.bincount(selected,minlength=len(probability)).double()/len(selected)
        expected=torch.tensor([probability[i]/sum(probability[j]for j in retained)if i in retained else 0 for i in range(len(probability))])
        assert torch.allclose(counts,expected.double(),atol=.035,rtol=0)
        assert torch.equal(p,unchanged)
        top_p_cases.append({'probabilities':probability,'threshold':threshold,'support':observed,'empirical':counts.tolist()})
    return {'parameters':sum(p.numel()for p in model.parameters()),'calls':[{k:v for k,v in c.items()if not k.startswith('_')}for c in calls],
            'steps':steps,'generated':result[0],'full_prefix_max_error':reference_error,
            'old_generated_logprobs':original[1],'repaired_generated_logprobs':repaired[1],
            'eos_steps':eos_steps,'eos_return':eos[0],'sampling_replay':sample1[0],
            'top_p_cases':top_p_cases,'parameters_unchanged':True,'all_parameter_grad_none':True,
            'first_repaired_logp':new_steps[0]['recorded_logp'][0]}


def main():
    torch.set_num_threads(1)
    note=ROOT/'解释.md';main_text=note.read_text(encoding='utf-8').split('<details>',1)[0]
    blocks=re.findall(r'```python\n(.*?)```',main_text,re.S)
    assert len(blocks)==6,len(blocks)
    scope={'__name__':'_lesson22_note_examples_'}
    stdout=[];previous=os.getcwd()
    try:
        os.chdir(ROOT)
        for i,block in enumerate(blocks):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):exec(compile(block,f'<lesson22-main-block-{i+1}>','exec'),scope)
            stdout.append(out.getvalue())
    finally:os.chdir(previous)
    official=ast.parse((SOURCES/'llama-generation.py').read_text(encoding='utf-8'))
    actual=next(n for n in ast.parse(blocks[2]).body if isinstance(n,ast.FunctionDef))
    original=next(n for n in official.body if isinstance(n,ast.FunctionDef)and n.name=='sample_top_p')
    assert function_ast(actual)==function_ast(original)
    displayed=next(n for n in ast.parse(blocks[4]).body if isinstance(n,ast.FunctionDef))
    assert function_ast(displayed)==function_ast(generate_node(official))
    assert scope['encoded']==[1,306,4658,278,6593,310,2834,338]
    assert scope['tokens'].tolist()==[[1,7,-1,-1,-1,-1,-1],[1,8,9,10,-1,-1,-1]]
    assert scope['input_text_mask'].tolist()==[[True,True,False,False,False,False,False],[True,True,True,True,False,False,False]]
    assert scope['kept'].tolist()==[21,9]
    assert scope['ignored'].item()==0 and abs(scope['written'].item()+torch.log_softmax(torch.tensor([2.,1.,0.]),0)[0].item())<1e-7
    checks=additional_checks(scope)
    figures=json.loads((ROOT/'复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
    for crop in figures['crops']:assert sha(ROOT/crop['file'])==crop['sha256']
    report={'date':'2026-10-18','note_sha256':sha(note),'script_sha256':sha(__file__),
            'loader_sha256':sha(LOADER),'official_source_sha256':EXPECTED,
            'tokenizer_sha256':sha(ROOT/'代码/tutorials/tokenizer.model'),
            'python':sys.version,'torch':torch.__version__,
            'sentencepiece':importlib.metadata.version('sentencepiece'),
            'fairscale':importlib.metadata.version('fairscale'),'device':'cpu','gloo_world_size':1,
            'model_cpu_allocations':2,'generation_cpu_devices':3,'main_blocks':len(blocks),
            'stdout':stdout,'sampling_and_generate_ast_match':True,'checks':checks,
            'scope':'Complete fixed Meta modules, initialized small model; no original pretrained weights or paper task evaluation.'}
    output=ROOT/'复现/outputs/基础讲解验证.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'blocks':len(blocks),'parameters':checks['parameters'],'full_prefix_error':checks['full_prefix_max_error'],
                      'first_repaired_logp':checks['first_repaired_logp'],'result':str(output)},ensure_ascii=False))


if __name__=='__main__':
    main()
