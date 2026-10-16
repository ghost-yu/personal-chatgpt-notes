# -*- coding: utf-8 -*-
"""Audit all originals; execute both unmodified code streams using the real full vocabularies."""
import ast
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/"代码/tutorials/tokenizer"
LOCAL=Path(os.environ.get("LESSON26_LOCAL","G:/zzz/Personal ChatGPT 系列/26 GPT-4o tokenizer 与 o200k_base/资料"))
LIB=Path(os.environ.get("LESSON26_TIKTOKEN_LIBS","G:/zzz/Personal ChatGPT 系列/25 Llama 3 introduction/资料/runtime-libs"))
EXTRA=Path(os.environ.get("LESSON26_EXTRA_LIBS",str(LOCAL/"runtime-libs")))
sys.path[:0]=[str(EXTRA),str(LIB)]
from build_delivery import COMMIT, FILES, VOCABS, segment
OUTPUT=ROOT/"复现/outputs"

def sha(b):return hashlib.sha256(b).hexdigest()

def tokens(s):
    return [(t.type,t.string)for t in tokenize.generate_tokens(io.StringIO(s).readline)
            if t.type not in (tokenize.COMMENT,tokenize.NL)]

def equal_ast(s,t):
    assert ast.dump(ast.parse(s))==ast.dump(ast.parse(t))

def audit():
    import nbformat
    manifest=json.loads((ROOT/"来源清单.json").read_bytes())
    assert manifest["commit"]==COMMIT
    upstream=os.environ.get("LESSON26_UPSTREAM","G:/zzz/Llama 入门教程/代码/personal_chatgpt")
    for item in manifest["files"]:
        data=(ROOT/item["published_path"]).read_bytes()
        assert sha(data)==item["sha256"] and len(data)==item["bytes"]
        assert data==subprocess.check_output(["git","-C",upstream,"show",COMMIT+":"+item["upstream_path"]])
    for name,record in manifest["vocabularies"].items():
        assert sha((ROOT/record["published_path"]).read_bytes())==VOCABS[name][1]==record["sha256"]
    rows={row[0]:row[1]for row in csv.reader(io.StringIO((LIB/"tiktoken-0.7.0.dist-info/RECORD").read_text(encoding="utf-8")))}
    for rel,h in manifest["installed_sources_sha256"].items():
        data=(LIB/rel).read_bytes()
        assert sha(data)==h
        assert rows[rel]=="sha256="+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        assert data==(LOCAL/"tiktoken-source"/rel).read_bytes()
    assert (LOCAL/"openai_public_remote.py").read_text(encoding="utf-8")==(LIB/"tiktoken_ext/openai_public.py").read_text(encoding="utf-8")
    results={}
    for stem,(prefix,notes)in FILES.items():
        raw=json.loads((FOLDER/(stem+".ipynb")).read_bytes())
        explained=json.loads((FOLDER/(prefix+"原代码_带说明.ipynb")).read_bytes())
        annotated=json.loads((FOLDER/(prefix+"代码注释.ipynb")).read_bytes())
        assert len(raw["cells"])==len(notes)
        assert len(explained["cells"])==len(annotated["cells"])==2*len(notes)+1
        for nb in (raw,explained,annotated):
            nbformat.validate(nbformat.from_dict(nb))
            assert {k:v for k,v in nb.items()if k!="cells"}=={k:v for k,v in raw.items()if k!="cells"}
        for i,c in enumerate(raw["cells"]):
            a=annotated["cells"][2+2*i];e=explained["cells"][2+2*i]
            assert c==e
            assert {k:v for k,v in c.items()if k!="source"}=={k:v for k,v in a.items()if k!="source"}
            s="".join(c["source"]);t="".join(a["source"])
            if c["cell_type"]=="code":
                equal_ast(s,t);assert tokens(s)==tokens(t);assert "# " in t
            else:assert s==t
        joined="\n".join("".join(c["source"])for c in raw["cells"]if c["cell_type"]=="code")
        equal_ast(joined,(FOLDER/(stem+"_代码注释.py")).read_text(encoding="utf-8"))
        results[stem]=dict(cells=len(notes),code_cells=sum(c["cell_type"]=="code"for c in raw["cells"]),
                          derived_cells=len(annotated["cells"]),schemas=3)
    core=(FOLDER/"tiktoken核心源码_代码注释.py").read_text(encoding="utf-8")
    locations={"get_encoding":"tiktoken/registry.py","cl100k_base":"tiktoken_ext/openai_public.py",
               "o200k_base":"tiktoken_ext/openai_public.py","bpe_encode":"tiktoken/_educational.py"}
    core_tree=ast.parse(core);verified=[]
    for node in core_tree.body:
        if isinstance(node,ast.ClassDef):
            src=(LIB/"tiktoken/core.py").read_text(encoding="utf-8")
            owner=next(n for n in ast.parse(src).body if getattr(n,"name","")=="Encoding")
            for method in node.body:
                ref=next(n for n in owner.body if getattr(n,"name","")==method.name)
                assert ast.dump(method)==ast.dump(ref)
                assert tokens(segment(core,method))==tokens(segment(src,ref))
                verified.append("Encoding."+method.name)
        else:
            src=(LIB/locations[node.name]).read_text(encoding="utf-8")
            ref=next(n for n in ast.parse(src).body if getattr(n,"name","")==node.name)
            assert ast.dump(node)==ast.dump(ref)
            assert tokens(segment(core,node))==tokens(segment(src,ref))
            verified.append(node.name)
    return dict(notebooks=results,raw_fixed_git_blobs=True,metadata_and_history_preserved=True,
        comment_only_ast_and_tokens=True,reading_python_ast=True,official_complete_objects=verified,
        installed_record_sources_verified=True,remote_tag_constructor_text_equal=True)

def normalize(x):
    if isinstance(x,(tuple,list)):return [normalize(v)for v in x]
    if isinstance(x,dict):return {str(k):normalize(v)for k,v in x.items()}
    if x is None or isinstance(x,(str,int,float,bool)):return x
    return {"type":type(x).__name__}

def worker(which,path):
    import contextlib
    import requests
    from langdetect import DetectorFactory
    # Only an external reproducibility context; no original cell is edited.
    DetectorFactory.seed=26
    def fail_network(*args,**kwargs):raise AssertionError("Offline verification attempted network access")
    requests.sessions.Session.request=fail_network
    ns={};traces={};literal=0
    os.chdir(FOLDER)
    for stem,(prefix,_)in FILES.items():
        file=FOLDER/(prefix+"代码注释.ipynb"if which=="annotated"else stem+".ipynb")
        nb=json.loads(file.read_bytes())
        cells=nb["cells"][2::2]if which=="annotated"else nb["cells"]
        rows=[]
        for i,c in enumerate(cells):
            if c["cell_type"]!="code":continue
            source="".join(c["source"]);tree=ast.parse(source);last=None
            if tree.body and isinstance(tree.body[-1],ast.Expr):last=tree.body.pop().value
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(tree,"<literal-notebook-cell>","exec"),ns)
                value=eval(compile(ast.Expression(last),"<display>","eval"),ns)if last else None
            row=dict(cell=i,display=normalize(value),stdout=out.getvalue())
            if stem=="tokenizer_gpt4o"and i in (17,18,19):
                mapping=ns["length_dict"]
                digest=hashlib.sha256()
                for token_id,length in mapping.items():digest.update(struct.pack("<II",token_id,length))
                row["vocabulary_scan"]=dict(valid_ids=len(mapping),ordered_id_length_sha256=digest.hexdigest(),
                    printed_count=ns["tot"],first20_by_length=list(mapping.items())[:20])
                assert ns["tot"]==20
            rows.append(row);literal+=1
        traces[stem]=rows
    assert literal==25
    assert ns["T1"].n_vocab==100277 and ns["T2"].n_vocab==200019
    assert ns["T"]is ns["T1"]
    result=dict(literal_cells=literal,cpu_variants=0,skips=0,traces=traces,
        external_context="langdetect.DetectorFactory.seed=26; full real vocab cached; requests disabled",
        corpus_counts=[ns["token_ids_len1"],ns["token_ids_len2"]],encoding_cache_same_object=True)
    Path(path).write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode())

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    report=dict(status="running",preservation=audit(),environment=dict(python=sys.version,platform=platform.platform(),
        versions={name:importlib.metadata.version(name)for name in ["tiktoken","langdetect","regex","nbformat","requests"]}),
        not_run=["GPT-4o API/model weights","Tokenizer or Transformer training","Training data audit","Model quality/price benchmark"])
    with tempfile.TemporaryDirectory(prefix="lesson26-real-cache-")as td:
        cache=Path(td)/"cache";cache.mkdir()
        for name,(key,h)in VOCABS.items():
            data=(FOLDER/"词表"/(name+".tiktoken")).read_bytes();assert sha(data)==h
            (cache/key).write_bytes(data)
        env={**os.environ,"TIKTOKEN_CACHE_DIR":str(cache),"PYTHONIOENCODING":"utf-8"}
        os.environ["TIKTOKEN_CACHE_DIR"]=str(cache)
        executed=[]
        for which in ("original","annotated"):
            result_path=Path(td)/(which+".json")
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker",which,str(result_path)],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
            assert child.returncode==0,(which,child.stderr.decode("utf-8",errors="replace")[-4000:])
            result=json.loads(result_path.read_bytes())
            native=child.stderr.decode("utf-8",errors="replace")
            result["native_stderr"]={"bytes":len(child.stderr),"sha256":sha(child.stderr),
                "panic_count":native.count("panicked at"),"first_lines":native.splitlines()[:6]}
            assert result["native_stderr"]["panic_count"]==54 # 19+16+19 absent ID decode calls.
            report[which]=result;executed.append(result)
        assert executed[0]["traces"]==executed[1]["traces"]
        assert executed[0]["corpus_counts"]==executed[1]["corpus_counts"]
        assert [r["vocabulary_scan"]["valid_ids"]for r in executed[0]["traces"]["tokenizer_gpt4o"]if "vocabulary_scan"in r]==[200000,100261,200000]
        report["all_literal_outputs_and_scan_order_equal"]=True
        # Supplement uses safe KeyError handling and strict UTF8, independently of the original loops.
        from tokenizer_diagnostics import run
        report["diagnostics"]=run()
        import tiktoken
        corpus=(FOLDER/"xyj.txt").read_text(encoding="utf-8")
        report["corpus"]=dict(bytes=(FOLDER/"xyj.txt").stat().st_size,unicode_chars=len(corpus),
            line_counts=executed[0]["corpus_counts"],
            historical_line_counts=[990508,703303],
            historical_equal=executed[0]["corpus_counts"]==[990508,703303],
            whole_string_counts=[len(tiktoken.get_encoding(name).encode(corpus))for name in VOCABS],
            universal_newline_text_mode=True)
        script=("import json,tiktoken\nenc=tiktoken.get_encoding('cl100k_base')\n"
            "try:enc.decode([100256])\n"
            "except BaseException as e:print(json.dumps(dict(type=type(e).__name__,module=type(e).__module__,is_exception=isinstance(e,Exception),is_base_exception=isinstance(e,BaseException))))\n")
        probe=subprocess.run([sys.executable,"-c","import sys;sys.path[:0]="+repr([str(EXTRA),str(LIB)])+"\n"+script],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
        assert probe.returncode==0
        error=json.loads(probe.stdout)
        assert error["type"]=="PanicException"and error["is_exception"]is False and error["is_base_exception"]is True
        report["invalid_id_original_decode_exception"]=error
        from langdetect import detect,DetectorFactory,LangDetectException
        DetectorFactory.seed=26
        try:detect("")
        except LangDetectException as e:report["empty_language_detection_error"]=str(e)
        else:raise AssertionError("Expected language detection failure")
    report["scripts_sha256"]={name:sha((ROOT/"复现/scripts"/name).read_bytes())
        for name in ["build_delivery.py","verify_delivery.py","tokenizer_diagnostics.py"]}
    report["status"]="passed"
    (OUTPUT/"delivery_verification.json").write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+"\n").encode())
    print(json.dumps(dict(status="passed",cells=25,variants=0,skips=0,
        corpus=report["corpus"],gaps=[len(s["holes"])for s in report["diagnostics"]["scans"]],
        native_panics_per_version=54,more_tokens_examples=report["diagnostics"]["larger_vocab_can_give_more_tokens"],
        utf8_partial_counts=[s["single_token_invalid_utf8"]for s in report["diagnostics"]["scans"]]),
        ensure_ascii=True,indent=2))

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--worker":worker(sys.argv[2],sys.argv[3])
    else:main()
