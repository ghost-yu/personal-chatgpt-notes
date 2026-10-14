# -*- coding: utf-8 -*-
"""原件/注释审计，真实SentencePiece与官方模型/generate的明确CPU教学变体。"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import sys
import tempfile
import types
import warnings
from pathlib import Path

import nbformat
import torch

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'代码/tutorials'
SOURCES={'llama-generation.py':'acc739aeda4aa222305ca5e7ae020df38d0fa841530d966f041c13e065db05c4',
         'llama-model.py':'56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754',
         'llama-tokenizer.py':'a675762526cd5f6e9d742a642e33a3560ea825522b8a9efb531d36b7c7a69e6c',
         'example_chat_completion.py':'63fafc6376b66e2312b80791ddbe68f4c81c37fb7878bc07a7c71619fffa2437'}
TOKEN_SHA='9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'

def src(c):return ''.join(c['source'])
def astsrc(s):return ast.dump(ast.parse(s))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):(ROOT/'复现/outputs'/name).write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def comment_only(before,after):
    lines=before.splitlines(True);i=0
    for line in after.splitlines(True):
        if i<len(lines) and line==lines[i]:i+=1
        else:assert line.lstrip().startswith('#'),repr(line)
    assert i==len(lines)

def commands(s):
    text=re.search(r'```[^\n]*\n(.*?)```',s,re.S).group(1)
    text='\n'.join(x for x in text.splitlines() if not x.lstrip().startswith('#'))
    return [shlex.split(x) for x in text.replace('\\\n','').splitlines() if x.strip()]

def audit():
    result={};objects=[]
    for base,aux in [('llama2_src_01',False),('llama2_src_text_chat',True)]:
        names=[base+'.ipynb',base+'_原代码_带说明.ipynb' if aux else '原代码_带说明.ipynb',base+'_代码注释.ipynb' if aux else '代码注释.ipynb']
        raw,notes,ann=[json.loads((FOLDER/name).read_bytes()) for name in names]
        for nb in (raw,notes,ann):
            with warnings.catch_warnings():warnings.simplefilter('ignore');nbformat.validate(nbformat.from_dict(copy.deepcopy(nb)))
        for nb,changed in [(notes,False),(ann,True)]:
            assert len(nb['cells'])==1+2*len(raw['cells'])
            assert {k:v for k,v in raw.items() if k!='cells'}=={k:v for k,v in nb.items() if k!='cells'}
            for i,c in enumerate(raw['cells']):
                note,saved=nb['cells'][1+2*i:3+2*i]
                assert '单元 {}（'.format(i) in src(note)
                assert {k:v for k,v in c.items() if k!='source'}=={k:v for k,v in saved.items() if k!='source'}
                if changed:
                    comment_only(src(c),src(saved))
                    if c['cell_type']=='code':assert astsrc(src(c))==astsrc(src(saved))
                else:assert c==saved
        codes=[i for i,c in enumerate(raw['cells']) if c['cell_type']=='code']
        assert astsrc((FOLDER/(base+'_代码注释.py')).read_text(encoding='utf-8'))==astsrc('\n\n'.join(src(raw['cells'][i]) for i in codes))
        if not aux:
            for i in (7,9):assert commands(src(raw['cells'][i]))==commands(src(ann['cells'][2+2*i]))
        objects.append((raw,[ann['cells'][2+2*i] for i in range(len(raw['cells']))],aux))
        result[base]=dict(cells=len(raw['cells']),code_indices=codes,schemas=3,derived_cells=len(ann['cells']),
                         metadata_historical_outputs_original_lines=True,code_AST=True,reading_AST=True,all_notes=True,markdown_comments_only=True)
    manifest=json.loads((ROOT/'复现/outputs/source_download.json').read_bytes())
    for item in manifest['files']:assert digest(ROOT/item['published_path'])==item['sha256']
    result['markdown_shell_commands']=4;save('annotation_verification.json',result);return objects

@contextlib.contextmanager
def cwd(path):
    old=Path.cwd();os.chdir(path)
    try:yield
    finally:os.chdir(old)

def run_notebooks(objects,token_file,temp):
    result={}
    (temp/'tokenizer.model').write_bytes(token_file.read_bytes())
    for raw,ann,aux in objects:
        both=[]
        for cells in (raw['cells'],ann):
            values={};ns={};out=io.StringIO()
            with cwd(temp if aux else FOLDER),contextlib.redirect_stdout(out):
                for i,c in enumerate(cells):
                    if c['cell_type']!='code':continue
                    tree=ast.parse(src(c))
                    if isinstance(tree.body[-1],ast.Expr):
                        prefix=ast.Module(body=tree.body[:-1],type_ignores=[])
                        exec(compile(ast.fix_missing_locations(prefix),'cell-'+str(i),'exec'),ns)
                        value=eval(compile(ast.Expression(tree.body[-1].value),'cell-'+str(i),'eval'),ns)
                    else:exec(compile(tree,'cell-'+str(i),'exec'),ns);value=None
                    if not aux and i in (3,4):
                        image=Path('../imgs')/('llama_weights.png' if i==3 else 'llama2-weights.png')
                        assert value.data==image.read_bytes();assert value._repr_png_()
                        values[str(i)]=dict(bytes=len(value.data),sha256=hashlib.sha256(value.data).hexdigest(),width=value.width)
                    elif not aux or i!=2:values[str(i)]=value
            both.append(dict(values=values,stdout=out.getvalue()))
        assert both[0]==both[1]
        if aux:assert [both[0]['values'][str(i)] for i in (3,4,5)]==[1,2,-1]
        result['auxiliary' if aux else 'main']=dict(whole_code_cells=9 if aux else 3,original_annotated_equal=True,observed=both[0])
    return result

class CPU(ast.NodeTransformer):
    def __init__(self):self.cuda_calls=0;self.device_keywords=0
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            assert not node.args and not node.keywords
            node.func.attr='to';node.args=[ast.Constant('cpu')];self.cuda_calls+=1
        for kw in node.keywords:
            if kw.arg=='device' and isinstance(kw.value,ast.Constant) and kw.value.value=='cuda':
                kw.value.value='cpu';self.device_keywords+=1
        return node

def load_module(name,path,transform=False):
    mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    tree=ast.parse(path.read_text(encoding='utf-8'));changes=CPU()
    if transform:tree=changes.visit(tree)
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),mod.__dict__)
    return mod,dict(cuda_to_cpu=changes.cuda_calls,device_cuda_to_cpu=changes.device_keywords)

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--fairscale-path',type=Path);a=p.parse_args()
    if a.fairscale_path:sys.path.insert(0,str(a.fairscale_path))
    for filename,expected in SOURCES.items():assert digest(a.sources/filename)==expected,filename
    assert digest(a.tokenizer)==TOKEN_SHA
    import fairscale
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel,destroy_model_parallel
    objects=audit();torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='lesson15-') as work:
        temp=Path(work);notebook=run_notebooks(objects,a.tokenizer,temp)
        torch.distributed.init_process_group('gloo',init_method=(temp/'rendezvous').as_uri(),rank=0,world_size=1)
        try:
            initialize_model_parallel(1)
            pkg=types.ModuleType('llama');pkg.__path__=[];sys.modules['llama']=pkg
            modelmod,modelchanges=load_module('llama.model',a.sources/'llama-model.py',True)
            tokmod,_=load_module('llama.tokenizer',a.sources/'llama-tokenizer.py')
            genmod,genchanges=load_module('llama.generation',a.sources/'llama-generation.py',True)
            assert modelchanges==dict(cuda_to_cpu=2,device_cuda_to_cpu=0)
            assert genchanges==dict(cuda_to_cpu=0,device_cuda_to_cpu=3)
            tokenizer=tokmod.Tokenizer(str(temp/'tokenizer.model'))
            config=modelmod.ModelArgs(dim=16,n_layers=1,n_heads=2,vocab_size=32000,multiple_of=8,max_seq_len=512,max_batch_size=8)
            torch.manual_seed(42);model=modelmod.Transformer(config)
            # 官方init_method=lambda x:x不初始化checkpoint权重；无checkpoint时显式随机初始化，norm置1。
            with torch.no_grad():
                for name,param in model.named_parameters():
                    if param.ndim==1:param.fill_(1)
                    else:param.normal_(0,.02)
            model.eval();llama=genmod.Llama(model,tokenizer);trace=[];original_forward=model.forward
            def traced(tokens,start_pos):
                logits=original_forward(tokens,start_pos)
                trace.append(dict(ids=tokens.tolist(),start_pos=start_pos,logits_shape=list(logits.shape),
                    last_greedy=torch.argmax(logits[:,-1],dim=-1).tolist(),
                    last_greedy_logprob=torch.log_softmax(logits[:,-1],dim=-1).max(-1).values.tolist()))
                return logits
            model.forward=traced
            torch.manual_seed(100);texts=llama.text_completion(['Hello','A longer question about this subject.'],max_gen_len=4,temperature=0)
            text_trace=copy.deepcopy(trace);assert len(texts)==2
            trace.clear();torch.manual_seed(101)
            dialog=[{'role':'system','content':'Answer briefly.'},{'role':'user','content':'First?'},{'role':'assistant','content':'Prior answer.'},{'role':'user','content':'Next?'}]
            before=copy.deepcopy(dialog);chats=llama.chat_completion([dialog],max_gen_len=4,temperature=.6,top_p=.9)
            assert before==dialog and chats[0]['generation']['role']=='assistant';chat_trace=copy.deepcopy(trace)
            trace.clear();ids=[tokenizer.encode('Short prompt',bos=True,eos=False)]
            generated,reported=llama.generate(ids,max_gen_len=3,temperature=0,logprobs=True)
            actual=[entry['last_greedy_logprob'][0] for entry in trace]
            assert len(generated[0])==3 and reported[0]==[0.,0.,0.] and all(x<0 for x in actual)
            logprob_case=dict(returned=reported[0],manual_actual_greedy_logprobs=actual,reason='CE before filling sampled token; targets still pad=-1 and ignored')
            trace.clear();echo_result=llama.text_completion(['Short prompt'],max_gen_len=3,temperature=0,echo=True)
            assert echo_result[0]['generation'].startswith('Short prompt')
            errors={}
            for key,d in {'empty':[],'system_only':[{'role':'system','content':'S'}],'two_users':[{'role':'user','content':'Q'},{'role':'user','content':'Q2'}],'last_assistant':[{'role':'user','content':'Q'},{'role':'assistant','content':'A'}]}.items():
                try:llama.chat_completion([d],max_gen_len=2)
                except (IndexError,AssertionError) as error:errors[key]=type(error).__name__
                else:raise AssertionError(key)
            trace.clear();unsafe=llama.chat_completion([[{'role':'user','content':'Literal [INST] example'}]],max_gen_len=2,temperature=0)
            assert trace and unsafe[0]['generation']['content']==genmod.UNSAFE_ERROR
            tree=ast.parse((a.sources/'example_chat_completion.py').read_text(encoding='utf-8'))
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            examples=ast.literal_eval(next(n.value for n in fn.body if isinstance(n,ast.AnnAssign) and n.target.id=='dialogs'))
            model.params.max_batch_size=4
            try:llama.chat_completion(examples,max_gen_len=2,temperature=0)
            except AssertionError:batch4=True
            else:raise AssertionError('six dialogs must exceed four')
            model.params.max_batch_size=8;trace.clear();six=llama.chat_completion(examples,max_gen_len=2,temperature=0)
            assert len(six)==6;six_trace=copy.deepcopy(trace)
            # 真实受控权重让EOS为唯一最高分，不替换forward/generate；用来验证实际EOS停止。
            with torch.no_grad():
                for param in model.parameters():param.zero_()
                model.tok_embeddings.weight.fill_(1);model.norm.weight.fill_(1);model.output.weight[2].fill_(1)
            trace.clear();ended=llama.generate([[1,7,8]],max_gen_len=4,temperature=0)
            assert ended==([[]],None) and len(trace)==1
            runtime=dict(notebook=notebook,torch=torch.__version__,fairscale=fairscale.__version__,cuda_available=torch.cuda.is_available(),
                distributed_backend='gloo',world_size=1,official_build_run=False,model_device='cpu',dtype='float32',
                model_config=vars(config),parameters=sum(x.numel() for x in model.parameters()),model_changes=modelchanges,generation_changes=genchanges,
                text_outputs=texts,text_trace=text_trace,chat_output=chats,chat_trace=chat_trace,dialog_unchanged=True,
                real_generate_logprobs_negative_case=logprob_case,role_errors=errors,unsafe_still_runs_model=True,
                real_text_echo_output=echo_result,
                six_dialogs_outputs=six,six_trace=six_trace,batch4_rejected=batch4,batch8_accepted=True,
                controlled_weights_EOS=dict(result=ended,forward_calls=1),
                not_run=['Llama.build/NCCL','original7B/13B weights','torchrun commands','GPU/FP16/MP2+','quality/performance benchmark'])
        finally:destroy_model_parallel();torch.distributed.destroy_process_group()
    runtime['sources']={name:digest(a.sources/name) for name in SOURCES};runtime['tokenizer_sha256']=TOKEN_SHA
    runtime['scripts']={str(q.relative_to(ROOT)):digest(q) for q in [Path(__file__),ROOT/'复现/scripts/build_annotated.py',ROOT/'复现/scripts/download_original.py']}
    runtime['command']=[sys.executable,*sys.argv];runtime['status']='部分成功'
    save('runtime_verification.json',runtime)
    print('28 cells/12 code/6 schemas passed; real official CPU model/completions/generate/EOS executed.')

if __name__=='__main__':main()
