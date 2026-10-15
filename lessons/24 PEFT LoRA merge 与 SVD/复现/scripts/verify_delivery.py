# -*- coding: utf-8 -*-
"""Audit all cells, then execute actual PEFT injection and local Llama/SVD workflows."""
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
import re
import string
import sys
import tempfile
import tokenize
from pathlib import Path

if os.environ.get("LESSON24_RUNTIME_LIBS"):
    sys.path.insert(0,os.environ["LESSON24_RUNTIME_LIBS"])

import nbformat
import sentencepiece as spm
import torch
from IPython.core.inputtransformer2 import TransformerManager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "代码/tutorials/peft"
TRANSFORM = TransformerManager()
STEMS = [("lora_inject", "inject"), ("lora_svd_basics", "svd"), ("lora_merge", "")]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def dump(node):
    return ast.dump(node, include_attributes=False)

def tokens(src):
    return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type not in (tokenize.COMMENT, tokenize.NL)]

def tree_text(src):
    return TRANSFORM.transform_cell(src)

def audit(sources):
    manifest = json.loads((ROOT/"来源清单.json").read_bytes())
    for e in manifest["files"]:
        raw = (ROOT/e["published_path"]).read_bytes()
        assert sha(raw) == e["sha256"] and len(raw) == e["bytes"]
    import peft
    peft_root = Path(inspect.getfile(peft)).parent
    for rel, expected in manifest["peft_installed_sources_sha256"].items():
        assert sha((sources/rel).read_bytes()) == expected == sha((peft_root/rel).read_bytes())
    assert sha((sources/"字幕原文.txt").read_bytes()) == manifest["subtitle"]["sha256"]
    notebooks, info = {}, {}
    for stem, prefix in STEMS:
        prefix = prefix+"_" if prefix else ""
        raw, explained, annotated = [json.loads((CODE/n).read_bytes()) for n in
            [stem+".ipynb", prefix+"原代码_带说明.ipynb", prefix+"代码注释.ipynb"]]
        for nb in [raw, explained, annotated]:
            nbformat.validate(nbformat.from_dict(nb))
        for nb in [explained, annotated]:
            assert len(nb["cells"]) == 2*len(raw["cells"])+1
            assert {k:v for k,v in nb.items() if k!="cells"} == {k:v for k,v in raw.items() if k!="cells"}
        py = []
        code_count = 0
        for i,c in enumerate(raw["cells"]):
            assert explained["cells"][2*i+2] == c
            a = annotated["cells"][2*i+2]
            assert {k:v for k,v in a.items() if k!="source"} == {k:v for k,v in c.items() if k!="source"}
            assert str(i) in "".join(annotated["cells"][2*i+1]["source"])
            if c["cell_type"] == "code":
                s, t = map(tree_text, ["".join(c["source"]), "".join(a["source"])])
                assert dump(ast.parse(s)) == dump(ast.parse(t)), (stem,i,"AST")
                assert tokens(s) == tokens(t), (stem,i,"noncomment tokens")
                py.append(s)
                code_count += 1
            else:
                assert "".join(a["source"]).startswith("".join(c["source"]))
        reading = (CODE/(stem+"_代码注释.py")).read_text(encoding="utf-8")
        assert dump(ast.parse("\n".join(py))) == dump(ast.parse(reading))
        notebooks[stem] = (raw, annotated)
        info[stem] = {"cells":len(raw["cells"]), "code_cells":code_count, "derived_cells_each":len(annotated["cells"]), "schemas":3}
    # Complete methods in the reading file must match their installed owner, not truncated excerpts.
    core = ast.parse((CODE/"PEFT合并核心源码_代码注释.py").read_text(encoding="utf-8"))
    official = [ast.parse((sources/r).read_text(encoding="utf-8")) for r in
                ["mapping.py", "tuners/lora/model.py", "tuners/lora/layer.py"]]
    methods = []
    for node in core.body:
        if isinstance(node, ast.FunctionDef):
            found = next(n for t in official for n in t.body if getattr(n,"name",None)==node.name)
            assert dump(node) == dump(found)
            methods.append(node.name)
        elif isinstance(node, ast.ClassDef):
            owner = next(n for t in official for n in t.body if isinstance(n,ast.ClassDef) and n.name==node.name)
            for method in node.body:
                if not isinstance(method,ast.FunctionDef):
                    continue
                src = next(n for n in owner.body if getattr(n,"name",None)==method.name)
                assert dump(method)==dump(src)
                methods.append(node.name+"."+method.name)
    return notebooks, {"notebooks":info,"total_cells":sum(v["cells"] for v in info.values()),
                       "total_code_cells":sum(v["code_cells"] for v in info.values()),
                       "preserved":"raw bytes, original metadata/history, ordering, all code AST/noncomment tokens, reading AST",
                       "complete_official_functions":methods}

def tensor_summary(x):
    if torch.is_tensor(x):
        y=x.detach().contiguous().cpu()
        return {"shape":list(y.shape),"dtype":str(y.dtype),"sha256":sha(y.numpy().tobytes())}
    return type(x).__name__

def execute(src, env):
    captured_source=[]
    if "??" in src:
        shell=InteractiveShell.instance()
        shell.user_ns.update(env)
        def pager(self,data,start=0,screen_lines=0):
            captured_source.append(data if isinstance(data,str) else data.get("text/plain",""))
        # set_hook appends a dispatcher callback; a prior run would retain the first closure.
        # Replace only the presentation callback so each run captures its own actual pinfo2 text.
        shell.hooks.show_in_pager=__import__("types").MethodType(pager,shell)
        with capture_output() as cap:
            result=shell.run_cell(src,store_history=False)
        assert result.success,(result.error_before_exec,result.error_in_exec,cap.stdout,cap.stderr)
        shown=re.sub(r"\x1b\[[0-9;]*m","",cap.stdout+cap.stderr+str(cap.outputs)+"\n".join(captured_source))
        assert "def forward" in shown and "lora_B" in shown,shown[:2500]
        return {"pinfo2_source_verified":True}
    tree=ast.parse(src)
    expr=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
    with capture_output():
        exec(compile(tree,"<course-cell>","exec"),env)
        value=eval(compile(ast.Expression(expr.value),"<course-last-expression>","eval"),env) if expr else None
    return tensor_summary(value)

def cells(nb, annotated=False):
    return [(i,c) for i,c in enumerate(nb["cells"]) if c["cell_type"]=="code"] if not annotated else [
        (i,nb["cells"][2*i+2]) for i in range((len(nb["cells"])-1)//2)
        if nb["cells"][2*i+2]["cell_type"]=="code"]

def run_inject(nb, annotated):
    torch.manual_seed(24)
    env={"__name__":"__lecture24_inject__"}
    trace=[]
    before=None
    for i,c in cells(nb,annotated):
        value=execute("".join(c["source"]),env)
        trace.append({"cell":i,"mode":"literal","value":value})
        if i==6:
            model=env["model"]; identity=id(model); linear=model.linear; emb=model.embedding; head=model.lm_head
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            baseline=model(torch.arange(8).reshape(1,8)).detach().clone()
        if i==10:
            model=env["model"]
            assert id(model)==identity and model.linear.base_layer is linear
            assert model.embedding is emb and model.lm_head is head
            assert not isinstance(model,PeftModel)
            torch.testing.assert_close(model(torch.arange(8).reshape(1,8)),baseline,rtol=0,atol=0)
    model=env["model"]; model.eval()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==128000
    assert [n for n,p in model.named_parameters() if p.requires_grad]==[
        "linear.lora_A.default.weight","linear.lora_B.default.weight"]
    exported=env["get_peft_model_state_dict"](model)
    assert set(exported)=={"linear.lora_A.weight","linear.lora_B.weight"}
    loss=model(torch.arange(8).reshape(1,8)).square().mean()
    loss.backward()
    a=model.linear.lora_A["default"].weight; b=model.linear.lora_B["default"].weight
    assert torch.count_nonzero(a.grad)==0 and torch.count_nonzero(b.grad)>0
    for n,p in model.named_parameters():
        if "lora_" not in n:assert p.grad is None
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
    opt.step()
    assert not torch.equal(b,initial["linear.lora_B.default.weight"])
    assert torch.equal(a,initial["linear.lora_A.default.weight"])
    for name,x in before.items():
        mapped="linear.base_layer."+name.split(".",1)[1] if name.startswith("linear.") else name
        assert torch.equal(model.state_dict()[mapped],x)
    return initial,{"trace":trace,"same_object":True,"base_layer_reference_preserved":True,"trainable":128000,
                    "initial_output_error":0,"B0_initial_A_gradient_zero":True,"one_supplemental_SGD_step_updates_B_only":True,
                    "adapter_export_shapes":{k:list(v.shape) for k,v in exported.items()},
                    "training_scope":"One extra CPU step on synthetic squared logits; original notebook has no optimizer/training."}

def run_svd(nb,annotated):
    env={"__name__":"__lecture24_svd__"};trace=[]
    for i,c in cells(nb,annotated):
        value=execute("".join(c["source"]),env);trace.append({"cell":i,"mode":"literal","value":value})
        if i==14:low_error=float(torch.dist(env["U"]@env["Vh"],env["W"]))
    W,W2=env["W"],env["W2"]
    full_s=torch.linalg.svdvals(W2)
    error=float(torch.dist(env["U"]@env["Vh"],W2))
    tail=float(full_s[20:].square().sum().sqrt())
    assert low_error<.001 and error>10
    assert abs(error-tail)<.001
    diagonal=torch.diag(torch.tensor([5.,2.,1.],dtype=torch.float64))
    u,s,v=torch.linalg.svd(diagonal,full_matrices=False)
    assert torch.dist((u[:,:2]*s[:2])@v[:2],diagonal)==1
    # A rectangular example makes full/reduced shapes visible; no giant vocabulary matrix allocated.
    rectangle=torch.arange(15,dtype=torch.float64).reshape(3,5)
    uf,sf,vf=torch.linalg.svd(rectangle,full_matrices=True)
    ur,sr,vr=torch.linalg.svd(rectangle,full_matrices=False)
    assert (uf.shape,vf.shape,ur.shape,vr.shape)==(torch.Size([3,3]),torch.Size([5,5]),torch.Size([3,3]),torch.Size([3,5]))
    torch.testing.assert_close((ur*sr)@vr,rectangle)
    return {"W":W,"W2":W2},{"trace":trace,"low_rank":int(torch.linalg.matrix_rank(W)),
                "full_rank":int(torch.linalg.matrix_rank(W2)),"rank20_low_rank_error":low_error,
                "rank20_full_rank_error":error,"discarded_singular_value_norm":tail,
                "diagonal_5_2_1_rank2_error":1,"rectangular_full_reduced_verified":True}

def fixture(work):
    torch.manual_seed(240)
    corpus=("Write an essay about Generative AI. Create a text ad given product and description. "
            "Product Sony PS5 PlayStation Console. Table Team Head Coach President Home Ground Location. "
            "Who is the Head Coach of the team whose President is Mario Volarevic? SQL Query SELECT FROM WHERE.\n")
    (work/"corpus.txt").write_text((corpus+string.ascii_letters+string.digits+string.punctuation+"\n")*8,encoding="utf-8")
    old=os.getcwd()
    try:
        os.chdir(work)
        spm.SentencePieceTrainer.train(input="corpus.txt",model_prefix="tiny",vocab_size=128,
             model_type="bpe",character_coverage=1.0,hard_vocab_limit=True,num_threads=1,minloglevel=2)
        tokenizer=LlamaTokenizer(vocab_file="tiny.model",legacy=True)
        base=work/"base";base.mkdir()
        tokenizer.save_pretrained(str(base))
        vocab=len(tokenizer)
        cfg=LlamaConfig(vocab_size=vocab,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
              num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
              tie_word_embeddings=False,bos_token_id=1,eos_token_id=2,pad_token_id=None,attention_dropout=0)
        model=LlamaForCausalLM(cfg);model.save_pretrained(str(base))
        extras=["<pad>","<|im_end|>","<|im_start|>assistant","<|im_start|>system","<|im_start|>user"]
        tokenizer.add_special_tokens({"pad_token":extras[0],"additional_special_tokens":extras[1:]})
        tokenizer.chat_template="{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        model=LlamaForCausalLM.from_pretrained(str(base),local_files_only=True)
        model.resize_token_embeddings(len(tokenizer))
        all_targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","embed_tokens","lm_head"]
        paths={}
        for j,name in enumerate(["norobots","sql","adcopy"]):
            torch.manual_seed(241+j)
            targets=all_targets[:7] if name=="sql" else all_targets
            pm=get_peft_model(copy.deepcopy(model),LoraConfig(r=8,lora_alpha=16,lora_dropout=.1,
                target_modules=targets,bias="none",task_type="CAUSAL_LM"))
            with torch.no_grad():
                for n,p in pm.named_parameters():
                    if "lora_" in n:p.normal_(0,.05)
            path=work/name
            pm.save_pretrained(str(path),save_embedding_layers=True)
            tokenizer.save_pretrained(str(path))
            paths[name]=str(path)
        return paths,{"base":str(base),"base_vocab":vocab,"adapter_vocab":len(tokenizer),
            "extra_tokens":extras,"hidden_size":16,"layers":1,"rank":8,"alpha":16,
            "description":"Random FP32 Llama and trained-on-local-text SentencePiece, nonzero seeded synthetic adapters; no pretrained model or task quality claim."}
    finally:
        os.chdir(old)

def variant(index,src,paths,meta):
    changes=[]
    for before,after in [
        ('"smangrul/tinyllama_lora_norobots"',repr(paths["norobots"])),
        ('"smangrul/tinyllama_lora_sql"',repr(paths["sql"])),
        ('"smangrul/tinyllama_lora_adcopy"',repr(paths["adcopy"])),
        ('"TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"',repr(meta["base"])),
        ('"cuda"','"cpu"'),
        ('load_in_4bit=True, device_map="auto"','local_files_only=True'),
        ('max_new_tokens=256','max_new_tokens=4'),
        ('max_new_tokens=64','max_new_tokens=4'),
    ]:
        if before in src:
            count=src.count(before);src=src.replace(before,after);changes.append({"from":before,"to":after,"count":count})
    return src,changes

def run_merge(nb,annotated,paths,meta):
    torch.manual_seed(24)
    env={"__name__":"__lecture24_merge__"};trace=[];original_active=None
    for i,c in cells(nb,annotated):
        if i==1:
            trace.append({"cell":i,"mode":"skipped","reason":"Do not set GPU/proxy environment for offline CPU run"})
            continue
        src,changes=variant(i,"".join(c["source"]),paths,meta)
        value=execute(src,env)
        trace.append({"cell":i,"mode":"explicit_cpu_variant" if changes else "literal","changes":changes,"value":value})
        if i==22:original_active=list(env["model"].active_adapters)
        if i==25:assert list(env["model"].active_adapters)==original_active==["norobots"]
    model=env["model"];model.eval()
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    embedding=model.base_model.model.model.embed_tokens
    assert set(embedding.scaling)=={"norobots","adcopy","merge"}
    assert embedding.scaling["merge"]==1 and model.peft_config["merge"].r==8
    corrected=2*embedding.get_delta_weight("norobots")+.3*embedding.get_delta_weight("adcopy")
    historical=env["delta_lora"].T
    torch.testing.assert_close(historical,2*corrected,rtol=1e-6,atol=1e-7)
    manual_product=(env["U"]@env["Vh"]).T
    merged=embedding.get_delta_weight("merge")
    torch.testing.assert_close(manual_product,merged,rtol=5e-5,atol=2e-6)
    layer_checks=[]
    names=["norobots","adcopy","sql"];weights=[2.,.3,.7]
    for key,layer in model.named_modules():
        if not isinstance(layer,LoraLayer):continue
        if "merge" not in layer.scaling:continue
        valid=[(n,w) for n,w in zip(names,weights) if n in layer.scaling]
        intended=sum(w*layer.get_delta_weight(n) for n,w in valid)
        biased=sum(w*layer.scaling[n]*layer.get_delta_weight(n) for n,w in valid)
        assert torch.allclose(biased,2*intended,rtol=1e-5,atol=1e-7)
        got=layer.get_delta_weight("merge")
        sv=torch.linalg.svdvals(biased)
        err=torch.dist(got,biased);tail=sv[8:].square().sum().sqrt()
        assert abs(float(err-tail))<2e-5,(key,err,tail)
        layer_checks.append({"layer":key,"valid_adapters":[n for n,w in valid],
              "weight_shape":list(got.shape),"svd_error":float(err),"tail_norm":float(tail),
              "error_against_intended":float(torch.dist(got,intended))})
    # The Markdown regex really executes in this context, but does not give reliable target paths.
    snippets=re.findall(chr(96)*3+r"[^\n]*\n(.*?)"+chr(96)*3,"".join(nb["cells"][8]["source"]) if not annotated else "".join(nb["cells"][18]["source"]),re.S)
    snippet=snippets[0];exec(snippet,env)
    regex_names=sorted(env["target_modules"])
    assert "norobots" in regex_names and "merge" in regex_names
    proper=[k for k,m in model.named_modules() if isinstance(m,LoraLayer)]
    assert regex_names!=proper
    inputs=torch.tensor([[1,4,7,9,2]])
    with torch.no_grad():before=model(inputs).logits.detach().clone()
    model.set_adapter("merge")
    with torch.no_grad():after=model(inputs).logits.detach().clone()
    active_change=float((before-after).abs().max())
    assert active_change>1e-5
    # New teaching comparisons operate on a clone; original notebook remains unchanged.
    extra=copy.deepcopy(model);extra.eval()
    source_state={n:p.detach().clone() for n,p in extra.named_parameters()
                  if any("."+a+"." in n or n.endswith("."+a) for a in names)}
    for new,typ,kw in [("cat_exact","cat",{}),("density_other","svd",{"density":.9}),
                        ("rank16","svd",{"svd_rank":16,"svd_full_matrices":False}),("linear_mix","linear",{})]:
        extra.add_weighted_adapter(names,weights,new,combination_type=typ,**kw)
    cat_errors=[];full_errors=[];linear_cross=[]
    for key,layer in extra.named_modules():
        if not isinstance(layer,LoraLayer) or "merge" not in layer.scaling:continue
        valid=[(n,w) for n,w in zip(names,weights) if n in layer.scaling]
        intended=sum(w*layer.get_delta_weight(n) for n,w in valid)
        biased=sum(w*layer.scaling[n]*layer.get_delta_weight(n) for n,w in valid)
        torch.testing.assert_close(layer.get_delta_weight("cat_exact"),intended,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(layer.get_delta_weight("density_other"),layer.get_delta_weight("merge"),rtol=1e-4,atol=2e-6)
        torch.testing.assert_close(layer.get_delta_weight("rank16"),biased,rtol=1e-4,atol=2e-6)
        cat_errors.append(float(torch.dist(layer.get_delta_weight("cat_exact"),intended)))
        full_errors.append(float(torch.dist(layer.get_delta_weight("rank16"),biased)))
        linear_cross.append(float(torch.dist(layer.get_delta_weight("linear_mix"),intended)))
    assert max(linear_cross)>.01
    for n,p in extra.named_parameters():
        if n in source_state:assert torch.equal(p,source_state[n])
    negative_error=None
    try:extra.add_weighted_adapter(names,[2.,-.3,.7],"negative_linear",combination_type="linear")
    except ValueError as e:negative_error=str(e)
    assert negative_error and "math domain" in negative_error
    # A corrected formula is demonstrated independently, not patched into original PEFT.
    d=corrected.T;u,s,vh=torch.linalg.svd(d,full_matrices=False)
    intended_rank8=((u[:,:8]*s[:8])@vh[:8]).T
    correction_error=float(torch.dist(intended_rank8,corrected))
    correction_tail=float(s[8:].square().sum().sqrt())
    assert abs(correction_error-correction_tail)<2e-5
    # Folding the active adapter is separate from creating a weighted adapter.
    folded=copy.deepcopy(model);folded.eval()
    with torch.no_grad():expected=folded(inputs).logits.detach().clone()
    merged_base=folded.merge_and_unload(safe_merge=True)
    assert isinstance(merged_base,LlamaForCausalLM)
    assert not any(isinstance(m,LoraLayer) for m in merged_base.modules())
    with torch.no_grad():got=merged_base(inputs).logits
    fold_error=float((got-expected).abs().max())
    torch.testing.assert_close(got,expected,rtol=5e-5,atol=2e-6)
    return state,{"trace":trace,"original_active_after_weighted_adapter":original_active,
        "source_scaling":2,"merge_scaling":1,"embedding_original_hand_code_matches_library":True,
        "peft012_svd_repeats_source_scaling":True,"historical_matrix_is_twice_intended":True,
        "layers":layer_checks,"regex_results":regex_names,"named_module_target_paths":proper,
        "explicit_activation_max_logit_change":active_change,"cat_exact_max_matrix_error":max(cat_errors),
        "rank16_reconstructs_biased_not_intended_max_error":max(full_errors),
        "linear_cross_term_max_difference":max(linear_cross),"linear_negative_weight_error":negative_error,
        "density_0_2_vs_0_9_same_svd_product":True,"source_adapters_unchanged":True,
        "corrected_rank8_error":correction_error,"corrected_rank8_tail":correction_tail,
        "merge_and_unload_max_logit_error":fold_error,"folded_model_has_no_LoraLayer":True,
        "original_full_vh_FP32_bytes":32005*32005*4}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--sources",required=True);args=parser.parse_args()
    assert importlib.metadata.version("peft")=="0.12.0"
    torch.set_num_threads(1)
    notebooks,static=audit(Path(args.sources))
    report={"status":"部分成功","static":static,"environment":{"os":platform.platform(),"python":sys.version,
       "torch":torch.__version__,"cuda_available":torch.cuda.is_available(),
       "versions":{n:importlib.metadata.version(n) for n in ["peft","transformers","sentencepiece","protobuf","nbformat","IPython"]}},
       "verification_script_sha256":sha(Path(__file__).read_bytes()),
       "builder_sha256":sha((ROOT/"复现/scripts/build_delivery.py").read_bytes())}
    for stem,fn,key in [("lora_inject",run_inject,"injection"),("lora_svd_basics",run_svd,"svd_basics")]:
        first,a=fn(notebooks[stem][0],False);second,b=fn(notebooks[stem][1],True)
        assert first.keys()==second.keys()
        for k in first:assert torch.equal(first[k],second[k]),(stem,k)
        assert a==b,(stem,"trace mismatch")
        report[key]={"raw":a,"annotated":b,"states_and_trace_identical":True}
    with tempfile.TemporaryDirectory(prefix="peft24_") as tmp:
        paths,meta=fixture(Path(tmp));report["cpu_fixture"]=meta
        first,a=run_merge(notebooks["lora_merge"][0],False,paths,meta)
        second,b=run_merge(notebooks["lora_merge"][1],True,paths,meta)
        assert first.keys()==second.keys()
        for k in first:assert torch.equal(first[k],second[k]),k
        assert a==b,"merge trace mismatch"
        report["merge"]={"raw":a,"annotated":b,"states_and_trace_identical":True}
    report["limits"]=["No original TinyLlama1.1B pretrained weights, remote trained adapters or 4bit CUDA run",
        "No original full-size 32005-by-32005 full SVD allocation; small matrix variant is explicit",
        "No language-task quality, TIES/DARE/Conv2d, GPU latency/memory or LoRA paper benchmark reproduction",
        "Remote Image HTML only; remote image content and lecturer exact dependency version unverified",
        "Official tag network fetch failed with TLS EOF; executable source fixed by installed distribution version/RECORD/SHA"]
    out=ROOT/"复现/outputs";out.mkdir(parents=True,exist_ok=True)
    (out/"delivery_verification.json").write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+"\n").encode())
    print(json.dumps({"status":report["status"],"static":static,
        "injection_trainable":report["injection"]["raw"]["trainable"],
        "svd_errors":[report["svd_basics"]["raw"]["rank20_low_rank_error"],report["svd_basics"]["raw"]["rank20_full_rank_error"]],
        "merge_facts":{k:v for k,v in report["merge"]["raw"].items() if k not in ["trace","layers","named_module_target_paths"]}},
        ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
