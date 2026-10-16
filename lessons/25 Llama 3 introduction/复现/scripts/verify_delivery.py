# -*- coding: utf-8 -*-
"""Preservation audit, real original-tokenizer tests, and explicit CPU variants."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tokenize
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LOCAL=Path(os.environ.get("LESSON25_LOCAL","G:/zzz/Personal ChatGPT 系列/25 Llama 3 introduction/资料"))
sys.path.insert(0,os.environ.get("LESSON25_RUNTIME_LIBS",str(LOCAL/"runtime-libs")))
sys.path.insert(0,os.environ.get("LESSON25_FAIRSCALE","G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local"))
import nbformat
import numpy as np
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter
from build_delivery import FILES, COMMIT, META_COMMIT

FOLDER=ROOT/"代码/tutorials/llama"
OUTPUT=ROOT/"复现/outputs"
torch.set_num_threads(2)
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
REPORT={"status":"running","scope":"2024-04 initial Llama3; no pretrained weights",
        "environment":{"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
                       "transformers":transformers.__version__,"cuda":torch.cuda.is_available()},
        "skipped":["Original proxy cell 1","Original 7B/8B checkpoints",
                   "CUDA/NCCL and multi-rank execution","Original benchmark/data/full training"]}

def sha(x):
    return hashlib.sha256(x).hexdigest()

def tokens(s):
    return [(t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(s).readline)
            if t.type not in (tokenize.COMMENT,tokenize.NL)]

def ast_equal(a,b):
    assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def audit():
    manifest=json.loads((ROOT/"来源清单.json").read_text(encoding="utf-8"))
    assert manifest["commit"]==COMMIT and manifest["meta_commit"]==META_COMMIT
    for f in manifest["files"]:
        data=(ROOT/f["published_path"]).read_bytes()
        assert len(data)==f["bytes"] and sha(data)==f["sha256"]
        # Compare against the pinned Git blob, not only the builder's own manifest.
        upstream=os.environ.get("LESSON25_UPSTREAM","G:/zzz/Llama 入门教程/代码/personal_chatgpt")
        assert data==subprocess.check_output(["git","-C",upstream,"show",COMMIT+":"+f["upstream_path"]])
    for rel,h in manifest["official_sources_sha256"].items():
        assert sha((LOCAL/"meta-llama3"/rel).read_bytes())==h,rel
    assert (FOLDER/"tokenizer.py").read_bytes()==(LOCAL/"meta-llama3/llama/tokenizer.py").read_bytes()
    source=(FOLDER/"tokenizer.py").read_text(encoding="utf-8")
    reading=(FOLDER/"tokenizer_代码注释.py").read_text(encoding="utf-8")
    ast_equal(source,reading);assert tokens(source)==tokens(reading)
    core=(FOLDER/"Llama3核心源码_代码注释.py").read_text(encoding="utf-8")
    official=ast.parse((LOCAL/"meta-llama3/llama/model.py").read_text(encoding="utf-8"))
    for node in ast.parse(core).body:
        assert ast.dump(node)==ast.dump(next(n for n in official.body if getattr(n,"name",None)==node.name))
    results={}
    for stem,(prefix,notes) in FILES.items():
        raw=json.loads((FOLDER/(stem+".ipynb")).read_bytes())
        annotated=json.loads((FOLDER/(prefix+"代码注释.ipynb")).read_bytes())
        explained=json.loads((FOLDER/(prefix+"原代码_带说明.ipynb")).read_bytes())
        assert len(raw["cells"])==len(notes)
        assert len(annotated["cells"])==len(explained["cells"])==2*len(notes)+1
        for nb in (raw,annotated,explained):
            nbformat.validate(nbformat.from_dict(nb))
            assert {k:v for k,v in nb.items() if k!="cells"}=={k:v for k,v in raw.items() if k!="cells"}
        for i,c in enumerate(raw["cells"]):
            a=annotated["cells"][2+2*i];e=explained["cells"][2+2*i]
            assert c==e
            assert {k:v for k,v in c.items() if k!="source"}=={k:v for k,v in a.items() if k!="source"}
            s="".join(c["source"]);t="".join(a["source"])
            if c["cell_type"]=="code":
                ast_equal(s,t);assert tokens(s)==tokens(t)
                assert "# " in t
            elif s!=t:
                assert t.startswith(s) and ((stem=="llama3_supp_data_config" and i in (3,5,7))
                                            or (stem=="llama3_0419" and i in (10,32)))
                blocks=re.findall(chr(96)*3+r"[^\n]*\n(.*?)"+chr(96)*3,t,re.S)
                assert len(blocks)==2
                jsonc=(stem=="llama3_supp_data_config" and i==3) or (stem=="llama3_0419" and i==32)
                if jsonc:
                    clean=lambda b:re.sub(r"^\s*//.*$","",b,flags=re.M)
                    assert json.loads(clean(blocks[0]))==json.loads(clean(blocks[1]))
                else:
                    ast_equal(blocks[0],blocks[1]);assert tokens(blocks[0])==tokens(blocks[1])
        joined="\n".join("".join(c["source"]) for c in raw["cells"] if c["cell_type"]=="code")
        ast_equal(joined,(FOLDER/(stem+"_代码注释.py")).read_text(encoding="utf-8"))
        results[stem]={"cells":len(notes),"code_cells":sum(c["cell_type"]=="code" for c in raw["cells"]),
                       "derived_cells":len(annotated["cells"]),"schemas":3}
    REPORT["preservation"]={"notebooks":results,"fixed_git_blob_bytes":True,
        "all_metadata_history_preserved":True,"all_code_ast_noncomment_tokens_equal":True,
        "markdown_complete_snippets_equal":True,"official_core_ast_equal":True}

def full_meta(which):
    old=which=="2"
    cfg=LlamaConfig(vocab_size=32000 if old else 128256,hidden_size=4096,
        intermediate_size=11008 if old else 14336,num_hidden_layers=32,
        num_attention_heads=32,num_key_value_heads=32 if old else 8,
        max_position_embeddings=4096 if old else 8192,
        rms_norm_eps=1e-6 if old else 1e-5,rope_theta=10000.0 if old else 500000.0,
        tie_word_embeddings=False,attention_bias=False,mlp_bias=False)
    with torch.device("meta"):
        model=LlamaForCausalLM(cfg)
    expected=6738415616 if old else 8030261248
    assert sum(p.numel() for p in model.parameters())==expected
    assert all(p.is_meta for p in model.parameters())
    d=4096;v=cfg.vocab_size;m=cfg.intermediate_size;g=cfg.num_key_value_heads
    breakdown={"embedding_and_head":2*v*d,"attention":32*(2*d*d+2*d*g*128),
               "mlp":32*3*d*m,"norms":32*2*d+d}
    assert sum(breakdown.values())==expected
    REPORT.setdefault("full_meta_configs",{})[which]=dict(parameters=expected,
        breakdown=breakdown,config=cfg.to_dict(),physical_weight_data=False,
        kv_weight_shape=list(model.model.layers[0].self_attn.k_proj.weight.shape),
        mlp_weight_shape=list(model.model.layers[0].mlp.gate_proj.weight.shape))
    return model

def make_fast(tokenizer):
    backend=TikTokenConverter(vocab_file=str(FOLDER/"llama3_tokenizer.model"),
                             pattern=tokenizer.pat_str).converted()
    fast=PreTrainedTokenizerFast(tokenizer_object=backend,model_max_length=512,
         bos_token="<|begin_of_text|>",eos_token="<|end_of_text|>")
    # Constructor may already register bos/eos; remaining IDs must retain official order.
    fast.add_special_tokens({"additional_special_tokens":list(tokenizer.special_tokens)})
    fast.pad_token=fast.eos_token
    for text,idx in tokenizer.special_tokens.items():
        assert fast.convert_tokens_to_ids(text)==idx,(text,idx)
    assert len(fast)==128256
    fast.chat_template=(
        "{{ bos_token }}{% for m in messages %}{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\\n\\n' + (m['content']|trim) + '<|eot_id|>' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}")
    return fast

FAST=None
def cpu_pipeline():
    cfg=LlamaConfig(vocab_size=128256,hidden_size=16,intermediate_size=56,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=1,max_position_embeddings=512,
        rms_norm_eps=1e-5,rope_theta=500000.0,bos_token_id=128000,eos_token_id=128001,
        pad_token_id=128001,tie_word_embeddings=False,attention_dropout=0.0)
    model=LlamaForCausalLM(cfg).eval()
    return transformers.pipeline("text-generation",model=model,tokenizer=copy.deepcopy(FAST),
        device=-1,max_new_tokens=4,pad_token_id=128001)

def normalize(x):
    if isinstance(x,torch.Tensor):
        a=x.detach().cpu().contiguous()
        return {"shape":list(a.shape),"dtype":str(a.dtype),"sha256":sha(a.numpy().tobytes())}
    if isinstance(x,datetime):
        return {"type":"datetime","actual_clock_excluded_from_equality":True}
    if isinstance(x,LlamaForCausalLM):
        return repr(x)
    if hasattr(x,"_repr_png_"):
        return {"type":type(x).__name__,"png_sha256":sha(x.data)}
    if isinstance(x,(dict,list,tuple)):
        return {str(k):normalize(v) for k,v in x.items()} if isinstance(x,dict) else [normalize(v) for v in x]
    if x is None or isinstance(x,(int,float,str,bool)):
        return x
    return {"type":type(x).__name__}

def execute(source,ns):
    tree=ast.parse(source)
    last=None
    if tree.body and isinstance(tree.body[-1],ast.Expr):
        last=tree.body.pop().value
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(tree,"<original-notebook-cell>","exec"),ns)
        val=eval(compile(ast.Expression(last),"<cell-display>","eval"),ns) if last else None
    return {"display":normalize(val),"stdout":capture.getvalue()},val

def notebook_run(annotated):
    torch.manual_seed(25)
    mod=load_module("lesson25_tokenizer_"+str(annotated),
                    FOLDER/("tokenizer_代码注释.py" if annotated else "tokenizer.py"))
    sys.modules["tokenizer"]=mod
    ns={"full_meta":full_meta,"cpu_pipeline":cpu_pipeline}
    results={};counts={"literal":0,"cpu_variants":0,"skipped":1}
    actual={}
    for stem,(prefix,_) in FILES.items():
        path=FOLDER/(prefix+"代码注释.ipynb" if annotated else stem+".ipynb")
        nb=json.loads(path.read_bytes())
        cells=nb["cells"][2::2] if annotated else nb["cells"]
        traces=[]
        for i,c in enumerate(cells):
            if c["cell_type"]!="code":continue
            if stem=="llama3_0419" and i==1:continue
            s="".join(c["source"]);mode="literal"
            if stem=="llama3_0419" and i in (15,19,34,36):
                tree=ast.parse(s)
                for stmt in tree.body:
                    if isinstance(stmt,ast.Assign) and isinstance(stmt.targets[0],ast.Name):
                        name=stmt.targets[0].id
                        if name in ("llama2_7b","llama3_8b"):
                            stmt.value=ast.parse("full_meta("+repr("2" if i==15 else "3")+")",mode="eval").body
                        if name=="pipeline":stmt.value=ast.parse("cpu_pipeline()",mode="eval").body
                if i==36:
                    for node in ast.walk(tree):
                        if isinstance(node,ast.keyword) and node.arg=="max_new_tokens":node.value=ast.Constant(4)
                s=ast.unparse(ast.fix_missing_locations(tree))
                mode="cpu_variants";actual[f"{stem}:{i}"]=s
            counts[mode]+=1
            trace,value=execute(s,ns)
            if isinstance(value,datetime):
                actual["clock"]=value.isoformat()
            traces.append(dict(cell=i,mode=mode,**trace))
        results[stem]=traces
    assert ns["total_params"]==8030261248 and ns["hidden_dim"]==14336
    assert ns["terminators"]==[128001,128009]
    assert len(ns["outputs"])==1 and ns["outputs"][0]["generated_text"].startswith(ns["prompt"])
    # The real random CPU model performs forward and returns reduced KV-head caches.
    m=ns["pipeline"].model
    ids=torch.tensor([[128000,9906,128009]],dtype=torch.long)
    with torch.no_grad():out=m(ids,use_cache=True)
    assert out.logits.shape==(1,3,128256)
    assert out.past_key_values[0][0].shape==(1,1,3,4)
    state={name:normalize(p) for name,p in m.state_dict().items()}
    return dict(counts=counts,traces=results,explicit_variants_and_clock=actual,
        final_state=state,prompt=ns["prompt"],small_model_parameters=sum(p.numel() for p in m.parameters()),
        logits_shape=list(out.logits.shape),kv_shape=list(out.past_key_values[0][0].shape))

def tokenizer_checks():
    global FAST
    original=load_module("tokenizer_reference",FOLDER/"tokenizer.py")
    annotated=load_module("tokenizer_annotated",FOLDER/"tokenizer_代码注释.py")
    t=original.Tokenizer(str(FOLDER/"llama3_tokenizer.model"))
    a=annotated.Tokenizer(str(FOLDER/"llama3_tokenizer.model"))
    FAST=make_fast(t)
    import tiktoken,blobfile
    REPORT["environment"].update(tiktoken=importlib.metadata.version("tiktoken"),blobfile=importlib.metadata.version("blobfile"))
    texts=["Hello world!","你好，世界！","  A\n\nB\t","1234567890","🙂🧑🏽‍💻","café e\u0301","",
           "Llama 3's tokenizer: 2024-04-18.\n数学 x² ≠ x"]
    rows=[]
    from sentencepiece import SentencePieceProcessor
    sp=SentencePieceProcessor(model_file="./llama2_tokenizer.model")
    for text in texts:
        ids=t.encode(text,bos=False,eos=False)
        assert ids==a.encode(text,bos=False,eos=False)==FAST(text,add_special_tokens=False).input_ids
        assert t.decode(ids)==a.decode(ids)==FAST.decode(ids,clean_up_tokenization_spaces=False)==text
        assert t.encode(text,bos=True,eos=True)==[128000]+ids+[128001]
        rows.append({"text":text,"llama3_ids":ids,"llama2_ids":sp.encode(text),"llama3_tokens":len(ids),
                     "llama2_tokens":len(sp.encode(text))})
    for text in ("x"*50001," "*50001,"ab "*140000):
        ids=t.encode(text,bos=False,eos=False)
        assert t.decode(ids)==text and ids==a.encode(text,bos=False,eos=False)
    assert t.n_words==128256 and len(t.special_tokens)==256 and sp.vocab_size()==32000
    assert t.stop_tokens=={128001,128009} and t.pad_id==-1
    marker="<|eot_id|>"
    assert t.encode(marker,bos=False,eos=False)!=[128009]
    assert t.encode(marker,bos=False,eos=False,allowed_special="all")==[128009]
    try:t.encode(marker,bos=False,eos=False,disallowed_special="all")
    except ValueError:pass
    else:raise AssertionError("disallowed special not rejected")
    dialogs=[
        [{"role":"system","content":"You are a pirate chatbot who always responds in pirate speak!"},
         {"role":"user","content":"Who are you?"}],
        [{"role":"user","content":"  你好！ \n"},{"role":"assistant","content":" 回复 "},
         {"role":"user","content":"再见"}], []]
    for dialog in dialogs:
        ids=original.ChatFormat(t).encode_dialog_prompt(dialog)
        assert ids==annotated.ChatFormat(a).encode_dialog_prompt(dialog)
        ids_fast=FAST.apply_chat_template(dialog,tokenize=True,add_generation_prompt=True) if dialog else None
        if dialog:
            assert ids==ids_fast
            s=FAST.apply_chat_template(dialog,tokenize=False,add_generation_prompt=True)
            assert s==t.decode(ids)
            assert FAST(s).input_ids==ids # No duplicate BOS in this explicit local adaptation.
        assert ids.count(128000)==1 and ids.count(128009)==len(dialog)
    REPORT["tokenizers"]={"real_original_rank_files":True,"ordinary_samples":rows,
        "special_count":len(t.special_tokens),"special_ids":{k:t.special_tokens[k] for k in
            ("<|begin_of_text|>","<|end_of_text|>","<|start_header_id|>","<|end_header_id|>","<|eot_id|>")},
        "long_character_cases":[50001,50001,420000],"roundtrips":True,
        "fast_adaptation":"Transformers4.44.2 TikTokenConverter + explicit original-order special tokens + explicit Jinja template; not downloaded Hub tokenizer_config",
        "ordinary_and_chat_ids_equal":True,"single_bos":True}

def fairscale_checks():
    import fairscale.nn.model_parallel.initialize as mp
    from fairscale.nn.model_parallel.layers import ParallelEmbedding,VocabParallelEmbedding,ColumnParallelLinear,RowParallelLinear
    from typing import Optional
    import torch.nn.functional as F
    mod=load_module("lesson25_official_model",LOCAL/"meta-llama3/llama/model.py")
    supp=json.loads((FOLDER/"llama3_supp_data_config.ipynb").read_bytes())
    fenced=lambda i:re.findall(chr(96)*3+r"[^\n]*\n(.*?)"+chr(96)*3,"".join(supp["cells"][i]["source"]),re.S)[0]
    config=json.loads(fenced(3))
    assert config==dict(dim=4096,n_layers=32,n_heads=32,n_kv_heads=8,vocab_size=128256,
                       multiple_of=1024,ffn_dim_multiplier=1.3,norm_eps=1e-5,rope_theta=500000.0)
    ns={"nn":torch.nn,"F":F,"Optional":Optional,"ColumnParallelLinear":ColumnParallelLinear,
        "RowParallelLinear":RowParallelLinear}
    exec(fenced(7),ns)
    ast_equal(fenced(7),ast.get_source_segment((LOCAL/"meta-llama3/llama/model.py").read_text(encoding="utf-8"),
        next(n for n in ast.parse((LOCAL/"meta-llama3/llama/model.py").read_text(encoding="utf-8")).body if getattr(n,"name","")=="FeedForward")))
    with tempfile.TemporaryDirectory(prefix="lesson25-gloo-") as td:
        torch.distributed.init_process_group("gloo",init_method=(Path(td)/"rendezvous").as_uri(),rank=0,world_size=1)
        try:
            mp.initialize_model_parallel(1);torch.manual_seed(25)
            ff=mod.FeedForward(16,64,8,1.3).double()
            for p in ff.parameters():torch.nn.init.normal_(p,mean=0,std=.1)
            copy_ff=ns["FeedForward"](16,64,8,1.3).double();copy_ff.load_state_dict(ff.state_dict())
            x=torch.randn(2,3,16,dtype=torch.float64,requires_grad=True)
            y=ff(x);y2=copy_ff(x)
            assert torch.equal(y,y2) and y.shape==x.shape
            w1=ff.w1.weight.detach().numpy();w2=ff.w2.weight.detach().numpy();w3=ff.w3.weight.detach().numpy()
            xn=x.detach().numpy()
            def numpy_ff(z):
                gate=z@w1.T
                return (gate/(1+np.exp(-gate))*(z@w3.T))@w2.T
            err=float(np.max(np.abs(y.detach().numpy()-numpy_ff(xn))))
            assert err<1e-14
            y.square().sum().backward()
            for p in [x,*ff.parameters()]:assert p.grad is not None and torch.isfinite(p.grad).all()
            reference_x=x.detach().clone().requires_grad_()
            reference_weights=[p.detach().clone().requires_grad_() for p in (ff.w1.weight,ff.w2.weight,ff.w3.weight)]
            u,v,w=reference_weights
            ref=(F.silu(reference_x@u.T)*(reference_x@w.T))@v.T
            ref.square().sum().backward()
            assert torch.equal(x.grad,reference_x.grad)
            for p,r in zip((ff.w1.weight,ff.w2.weight,ff.w3.weight),reference_weights):assert torch.equal(p.grad,r.grad)
            graderr=[]
            for idx in [(0,0,0),(1,2,15),(0,1,7)]:
                plus=xn.copy();minus=xn.copy();plus[idx]+=1e-6;minus[idx]-=1e-6
                fd=(np.square(numpy_ff(plus)).sum()-np.square(numpy_ff(minus)).sum())/2e-6
                graderr.append(abs(float(x.grad[idx])-fd))
            assert max(graderr)<1e-8
            assert sum(p.numel() for p in ff.parameters())==2688
            args=mod.ModelArgs(**config)
            with torch.device("meta"):
                large=mod.FeedForward(args.dim,4*args.dim,args.multiple_of,args.ffn_dim_multiplier)
                # Execute the actual Markdown constructor snippet in its complete declared context.
                holder=type("Holder",(),{})()
                exec(fenced(5),{"self":holder,"args":args,"FeedForward":mod.FeedForward})
            assert large.w1.weight.shape==(14336,4096)
            assert sum(p.numel() for p in large.parameters())==176160768
            assert holder.feed_forward.w1.weight.shape==large.w1.weight.shape
            embed_checks={}
            weight=torch.randn(32,16,dtype=torch.float64)
            ids=torch.tensor([[0,1,31],[2,2,4]],dtype=torch.long)
            for cls in (ParallelEmbedding,VocabParallelEmbedding):
                emb=cls(32,16,init_method=lambda x:x).double()
                emb.weight.data.copy_(weight)
                ey=emb(ids);ry=F.embedding(ids,weight)
                assert torch.equal(ey,ry)
                ey.sum().backward()
                expected=torch.zeros_like(weight);expected.index_add_(0,ids.flatten(),torch.ones(6,16,dtype=torch.float64))
                assert torch.equal(emb.weight.grad,expected)
                embed_checks[cls.__name__]={"shape":list(ey.shape),"lookup_and_weight_gradient_equal":True}
            REPORT["fairscale"]={"backend":"gloo","world_size":1,"official_model_module_unmodified":True,
                "course_complete_class_ast_same":True,"explicit_weight_init":"seed25 normal std.1, double",
                "tiny_input_output":[2,3,16],"tiny_intermediate":[2,3,56],"tiny_parameters":2688,
                "numpy_forward_error":err,"input_fd_max_error":max(graderr),"all_autograd_exact_reference":True,
                "large_meta_width":14336,"large_meta_parameters":176160768,"embeddings":embed_checks,
                "full_meta_constructor_snippet":True,"multi_rank_tested":False}
        finally:
            mp.destroy_model_parallel()
            torch.distributed.destroy_process_group()

def markdown_checks():
    nb=json.loads((FOLDER/"llama3_0419.ipynb").read_bytes())
    block=lambda i:re.findall(chr(96)*3+r"[^\n]*\n(.*?)"+chr(96)*3,"".join(nb["cells"][i]["source"]),re.S)[0]
    config=json.loads(re.sub(r"^\s*//.*$","",block(32),flags=re.M))
    entry=config["configurations"][0]
    assert entry["module"]=="torch.distributed.launch"
    assert entry["args"]==["--nproc_per_node","1","example_text_completion.py","--ckpt_dir","Meta-Llama-3-8B",
        "--tokenizer_path","Meta-Llama-3-8B/tokenizer.model","--max_seq_len","512","--max_batch_size","6"]
    child_env={**os.environ,"PYTHONIOENCODING":"utf-8"}
    probe=subprocess.run([sys.executable,"-c","import torch\n"+block(10)],
                         capture_output=True,text=True,encoding="utf-8",errors="replace",env=child_env)
    assert probe.returncode!=0 and "CUDA" in probe.stderr
    small=torch.tensor([1.,.1])
    assert small.to(torch.bfloat16).element_size()==small.to(torch.float16).element_size()==2
    REPORT["markdown"]={"jsonc_fields_parsed":entry,"cuda_default_snippet_cpu_returncode":probe.returncode,
        "cuda_default_snippet_error":probe.stderr,"cpu_dtype_bytes":2}
    deltas={k:REPORT["full_meta_configs"]["3"]["breakdown"][k]-v
            for k,v in REPORT["full_meta_configs"]["2"]["breakdown"].items()}
    assert deltas==dict(embedding_and_head=788529152,attention=-805306368,mlp=1308622848,norms=0)
    assert sum(deltas.values())==1291845632
    REPORT["derived_math"]={"parameter_differences":deltas,"weights_fp16_gib":8030261248*2/1024**3,
        "kv_cache_2_4096_gib":2*32*1*4096*32*128*2/1024**3,
        "kv_cache_3_4096_gib":2*32*1*4096*8*128*2/1024**3,
        "kv_cache_3_8192_gib":2*32*1*8192*8*128*2/1024**3,
        "cache_numbers_are_shape_estimates":True}

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    audit()
    previous=Path.cwd();os.chdir(FOLDER)
    try:
        tokenizer_checks()
        a=notebook_run(False);b=notebook_run(True)
        assert a["counts"]==b["counts"]=={"literal":22,"cpu_variants":4,"skipped":1}
        for key in ("traces","final_state","prompt","small_model_parameters","logits_shape","kv_shape"):
            assert a[key]==b[key],key
        REPORT["notebook_execution"]={"original":a,"annotated":b,
            "traces_and_final_state_equal":True,"clock_values_only_type_compared":True,
            "full_original_checkpoint_loading":False}
        assert a["prompt"]==load_module("final_template",FOLDER/"tokenizer.py").Tokenizer(
            "./llama3_tokenizer.model").decode(load_module("final_format",FOLDER/"tokenizer.py").ChatFormat(
            load_module("final_token",FOLDER/"tokenizer.py").Tokenizer("./llama3_tokenizer.model")).encode_dialog_prompt([
                {"role":"system","content":"You are a pirate chatbot who always responds in pirate speak!"},
                {"role":"user","content":"Who are you?"}]))
        fairscale_checks();markdown_checks()
    finally:
        os.chdir(previous)
    REPORT["scripts_sha256"]={n:sha((ROOT/"复现/scripts"/n).read_bytes()) for n in
                              ("build_delivery.py","verify_delivery.py")}
    REPORT["status"]="passed"
    (OUTPUT/"delivery_verification.json").write_bytes((json.dumps(REPORT,ensure_ascii=False,indent=2)+"\n").encode())
    print(json.dumps({"status":"passed","static":REPORT["preservation"],
        "executions_per_version":a["counts"],"small_model_parameters":a["small_model_parameters"],
        "fairscale":REPORT["fairscale"],"math":REPORT["derived_math"]},ensure_ascii=True,indent=2))

if __name__=="__main__":
    main()
