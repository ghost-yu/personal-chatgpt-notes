# -*- coding: utf-8 -*-
"""全部原件/注释审计，真实pipeline与固定TRL0.5示例CLI的CPU小模型运行。"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import platform
import re
import runpy
import shlex
import sys
import tempfile
import warnings
from pathlib import Path

import accelerate
import datasets
import nbformat
import peft
import sentencepiece
import torch
import transformers
import trl
from datasets import load_dataset
from peft import PeftModel
from transformers import HfArgumentParser,LlamaConfig,LlamaForCausalLM,LlamaTokenizerFast,set_seed

ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'代码/tutorials'
CODE=[3,4,5,6,7,8,9,10,11,12,17,18,19,20,21,22]
WHOLE=[3,4,7,8,9,10,11,12,17,19,20,21,22]
TOKENIZER_SHA='9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'
SAMPLE_SHA='0c59149204ecf17b600fa8bfc59a6308e7abc1f985e8ef0efa6ae302360d3561'
SCRIPT_SHA='e75c36b75daa03445ef9e763847b8e1450b561fbc222f345e1daa65f46687aec'
RAW_SHA='b27440095f45a608b79666bf967276621f256ab252e7b156b758e22b73435ab6'


def src(cell):return ''.join(cell['source'])
def astsrc(text):return ast.dump(ast.parse(text))
def write_json(path,r):path.write_bytes((json.dumps(r,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def shell_commands(text):
    block=re.search(r'```[^\n]*\n(.*?)```',text,re.S).group(1)
    lines='\n'.join(line for line in block.splitlines() if not line.lstrip().startswith('#'))
    return [shlex.split(command) for command in lines.replace('\\\n','').splitlines() if command.strip()]


def comments_only(original,changed):
    lines=original.splitlines(True);i=0
    for line in changed.splitlines(True):
        if i<len(lines) and line==lines[i]:i+=1
        else:assert line.lstrip().startswith('#')
    assert i==len(lines)


def audit():
    names=['llama2_introduction.ipynb','原代码_带说明.ipynb','代码注释.ipynb']
    raw,explained,annotated=[json.loads((FOLDER/n).read_bytes()) for n in names]
    assert hashlib.sha256((FOLDER/names[0]).read_bytes()).hexdigest()==RAW_SHA
    assert len(raw['cells'])==23 and [i for i,c in enumerate(raw['cells']) if c['cell_type']=='code']==CODE
    for nb in (raw,explained,annotated):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');nbformat.validate(nbformat.from_dict(copy.deepcopy(nb)))
    for nb,changed in [(explained,False),(annotated,True)]:
        assert len(nb['cells'])==47
        assert {k:v for k,v in nb.items() if k!='cells'}=={k:v for k,v in raw.items() if k!='cells'}
        for i,c in enumerate(raw['cells']):
            note,saved=nb['cells'][1+2*i:3+2*i]
            assert note['cell_type']=='markdown' and '单元 {}（'.format(i) in src(note)
            assert {k:v for k,v in saved.items() if k!='source'}=={k:v for k,v in c.items() if k!='source'}
            if changed and c['cell_type']=='code':
                assert astsrc(src(c))==astsrc(src(saved));comments_only(src(c),src(saved))
            elif changed and i==15:
                comments_only(src(c),src(saved));assert shell_commands(src(c))==shell_commands(src(saved))
            else:assert c==saved
    assert astsrc((FOLDER/'llama2_introduction_代码注释.py').read_text(encoding='utf-8'))==astsrc('\n\n'.join(src(raw['cells'][i]) for i in CODE))
    doc=(ROOT/'代码单元解释.md').read_text(encoding='utf-8');assert all('## 单元 {}（'.format(i) in doc for i in range(23))
    commands=shell_commands(src(raw['cells'][15]));assert len(commands)==3
    write_json(ROOT/'复现/outputs/annotation_verification.json',dict(original_sha256=RAW_SHA,cells=23,code_indices=CODE,
        schemas_passed=3,derived_cells=47,code_AST_passed=16,reading_AST_passed=True,markdown15_shell_commands=commands,
        shell_semantics_preserved=True,all_23_notes=True,original_lines_metadata_historical_outputs_preserved=True))
    return raw,[annotated['cells'][2+2*i] for i in range(23)],commands[2][2:]


class CPUVariant(ast.NodeTransformer):
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='float16':node.attr='float32'
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='pipeline':
            for kw in node.keywords:
                if kw.arg=='device_map':kw.value=ast.Constant(None)
        if isinstance(node.func,ast.Name) and node.func.id=='load_dataset':
            node.args=[ast.Constant('json')]
            node.keywords.append(ast.keyword(arg='data_files',value=ast.Name(id='sample_path',ctx=ast.Load())))
        return node


def run_notebook(cells,base_dir,sample_path):
    context=dict(sample_path=str(sample_path));out=io.StringIO();results={};preprocess={}
    with contextlib.redirect_stdout(out):
        for i in CODE:
            if i==5:continue
            if i==6:context['model_name']=str(base_dir)
            if i in (10,12):set_seed(100+i)
            if i in (6,18):tree=CPUVariant().visit(ast.parse(src(cells[i])))
            else:tree=ast.parse(src(cells[i]))
            exec(compile(ast.fix_missing_locations(tree),'cell_'+str(i),'exec'),context)
            if i==6:
                pipeline=context['pipeline'];tokenizer=context['tokenizer']
                assert pipeline.tokenizer is not tokenizer
                assert len(tokenizer)==32000 and tokenizer.bos_token_id==1 and tokenizer.eos_token_id==2
                assert next(pipeline.model.parameters()).dtype==torch.float32
            if i in (10,12):
                generated=context['sequences'];assert len(generated)==3
                prompt=ast.literal_eval(next(n for n in ast.walk(ast.parse(src(cells[i]))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='pipeline').args[0])
                assert all(item['generated_text'].startswith(prompt) for item in generated)
                ids=context['pipeline'].preprocess(prompt)['input_ids'][0].tolist()
                ordinary=context['tokenizer'](prompt)['input_ids']
                assert ordinary==[1]+ids
                results[str(i)]=generated
                preprocess[str(i)]=dict(prompt_tokens=len(ids),ordinary_tokenizer_tokens=len(ordinary),pipeline_adds_BOS=False,
                                       max_length=200,max_new_budget=200-len(ids),num_return_sequences=3)
    assert context['all_param']==context['trainable_params']==1026608
    assert len(context['dataset'])==5
    # 修改外部分词实例的pad，不改变pipeline内部实例。
    tokenizer=context['tokenizer'];pipeline=context['pipeline'];before=pipeline.tokenizer.pad_token
    tokenizer.pad_token=tokenizer.eos_token
    assert pipeline.tokenizer.pad_token==before
    return dict(unaltered_whole_cells=WHOLE,skipped_original_cells={'5':'proxy environment'},
                variants={'6':'context model_name local random checkpoint; float16→float32, device_map auto→None',
                          '18':'Hub dataset→json/data_files fixed local five rows'},
                parameters=context['all_param'],vocab=32000,independent_tokenizer_objects=True,
                external_pad_change_does_not_change_pipeline=True,preprocessing=preprocess,generation=results,
                rows=5,first_row_role_split=context['dataset'][0]['text'].split('###'),stdout=out.getvalue())


def run_cli(script,argv):
    old_argv=sys.argv;out=io.StringIO()
    try:
        sys.argv=[str(script),*argv]
        with contextlib.redirect_stdout(out): result=runpy.run_path(str(script),run_name='__main__')
    finally:sys.argv=old_argv
    return result,out.getvalue()


def import_teaching():
    spec=importlib.util.spec_from_file_location('llama2_teaching',ROOT/'代码/分词与监督_教学补充.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tokenizer',type=Path,required=True);parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--sft-script',type=Path,required=True);args=parser.parse_args()
    for p,sha in [(args.tokenizer,TOKENIZER_SHA),(args.samples,SAMPLE_SHA),(args.sft_script,SCRIPT_SHA)]:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,p
    torch.set_num_threads(2)
    raw,annotated,original_flags=audit();teaching=import_teaching()
    with tempfile.TemporaryDirectory(prefix='lesson14-local-') as work:
        temp=Path(work);token_file=temp/'tokenizer.model';token_file.write_bytes(args.tokenizer.read_bytes())
        tokenizer=LlamaTokenizerFast(vocab_file=str(token_file))
        assert len(tokenizer)==32000
        set_seed(42);config=LlamaConfig(vocab_size=32000,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=4096,bos_token_id=1,eos_token_id=2)
        model=LlamaForCausalLM(config);base=temp/'base';model.save_pretrained(base);tokenizer.save_pretrained(base)
        before={k:v.clone() for k,v in model.state_dict().items()}
        # JSONL目录由真实load_dataset自动识别，不替换任何数据/模型API。
        local_data=temp/'guanaco';local_data.mkdir();jsonl=local_data/'train.jsonl'
        rows=json.loads(args.samples.read_text(encoding='utf-8'));assert len(rows)==5
        jsonl.write_bytes(('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n').encode('utf-8'))
        original=run_notebook(raw['cells'],base,jsonl);changed=run_notebook(annotated,base,jsonl)
        assert original['generation']==changed['generation'] and original['preprocessing']==changed['preprocessing']
        argv=['--model_name',str(base),'--dataset_name',str(local_data),'--use_peft',
              '--batch_size','1','--gradient_accumulation_steps','2','--max_steps','2',
              '--output_dir',str(temp/'adapter'),'--use_auth_token','False','--trust_remote_code','False']
        set_seed(42);ns,stdout=run_cli(args.sft_script,argv)
        trainer=ns['trainer'];trained=trainer.model
        assert isinstance(trained,PeftModel) and trainer.state.global_step==2
        assert ns['script_args'].seq_length==512 and ns['script_args'].peft_lora_r==64 and ns['script_args'].peft_lora_alpha==16
        assert not ns['script_args'].load_in_4bit and not ns['script_args'].load_in_8bit
        trainable={n:p for n,p in trained.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in trainable.values())==4096
        targets=[n for n,m in trained.named_modules() if hasattr(m,'lora_A')]
        assert len(targets)==2 and all(n.endswith(('q_proj','v_proj')) for n in targets)
        current=trained.get_base_model().state_dict()
        assert all(torch.equal(v,current[k]) for k,v in before.items())
        assert any(p.detach().ne(0).any() for n,p in trainable.items() if 'lora_B' in n)
        opt_ids={id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        assert opt_ids=={id(p) for p in trainable.values()}
        supervision=teaching.inspect_supervision(trainer,ns['dataset'])
        parsed=HfArgumentParser(ns['ScriptArguments']).parse_args_into_dataclasses(args=original_flags)[0]
        assert parsed.load_in_4bit and parsed.use_peft and parsed.batch_size==16 and parsed.gradient_accumulation_steps==2
        adapter=temp/'adapter';saved=torch.load(adapter/'adapter_model.bin',map_location='cpu',weights_only=True)
        assert len(saved)==4 and all('lora_' in k for k in saved)
        fresh=LlamaForCausalLM.from_pretrained(base)
        loaded=PeftModel.from_pretrained(fresh,adapter);trained.eval();loaded.eval()
        inputs=trainer.tokenizer('Hello, explain addition.',return_tensors='pt')
        with torch.no_grad():left=trained(**inputs).logits;right=loaded(**inputs).logits
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
        assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in trained.state_dict().items())
        negative=argv+['--load_in_4bit','--load_in_8bit']
        try:run_cli(args.sft_script,negative)
        except ValueError as error:quant_error=str(error);assert "at the same time" in quant_error
        else:raise AssertionError('mutually exclusive quantization flags should fail')
        # meta只计配置逻辑参数，无7B权重分配/forward/能力声明。
        with torch.device('meta'):
            meta=LlamaForCausalLM(LlamaConfig(vocab_size=32000,hidden_size=4096,intermediate_size=11008,
                num_hidden_layers=32,num_attention_heads=32,max_position_embeddings=4096))
        meta_count=sum(p.numel() for p in meta.parameters())
        assert meta_count==teaching.parameter_formula(32000,4096,11008,32)==6738415616
        cli=dict(script_sha256=SCRIPT_SHA,actual_argv=argv,original_shell_cli_parsed=vars(parsed),actual_arguments=vars(ns['script_args']),
                 optimizer_updates=2,raw_rows=5,rank=64,alpha=16,scaling=.25,trainable_parameters=4096,targets=targets,
                 base_unchanged=True,adapter_B_changed=True,optimizer_only_adapter=True,packing=False,
                 supervision=supervision,training_log=trainer.state.log_history,stdout=stdout,
                 saved_adapter_keys=sorted(saved),save_reload=dict(state_tensors_exact=True,
                    logits_max_abs_difference=float((left-right).abs().max()),tolerance=dict(rtol=1e-5,atol=1e-6)),
                 mutually_exclusive_quantization_failure=quant_error)
    sources={}
    for obj in [trl.SFTTrainer,transformers.pipelines.TextGenerationPipeline,peft.PeftModel]:
        p=Path(inspect.getfile(obj));sources[obj.__name__]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    scripts=[Path(__file__),ROOT/'复现/scripts/build_annotated.py',ROOT/'复现/scripts/download_original.py',ROOT/'代码/分词与监督_教学补充.py']
    report=dict(status='部分成功',command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),python=sys.version,platform=platform.platform(),
        versions={m.__name__:m.__version__ for m in [torch,transformers,trl,peft,accelerate,datasets,sentencepiece]},cuda_available=torch.cuda.is_available(),
        assets={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.tokenizer,args.samples,args.sft_script]},
        installed_sources=sources,script_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
        original=original,annotated=changed,original_annotated_generation_exact=True,
        cli=cli,meta_7B_logical_parameter_count=meta_count,
        not_run=['original proxy','remote gated7B/chat weights','CUDA/4bit/bitsandbytes','fullGuanaco9846 rows','original924 updates/5hour estimate','chat ability/quality benchmark'])
    write_json(ROOT/'复现/outputs/runtime_verification.json',report)
    print(json.dumps(dict(schemas=3,code_AST=16,all23notes=True,original_whole_cells_each=13,
                         real_pipeline_generations_equal=True,real_reference_cli_updates=2,trainable=4096,meta7B=meta_count),ensure_ascii=True))


if __name__=='__main__':main()
