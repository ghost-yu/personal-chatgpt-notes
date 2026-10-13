# -*- coding: utf-8 -*-
"""固定官方PEFT源码+真实本地小型LLaMA/tokenizer/adapter；明确区分原CUDA失败与CPU补充。"""
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
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
PEFT_COMMIT='7120c9a2636f93c8db3fc4ae466e02f338ecead8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(cell,scope,index):
    tree=ast.parse(''.join(cell['source']))
    value=None
    with contextlib.redirect_stdout(io.StringIO()) as out:
        if tree.body and isinstance(tree.body[-1],ast.Expr):
            last=tree.body.pop()
            exec(compile(tree,'<original-cell-%d>'%index,'exec'),scope)
            value=eval(compile(ast.Expression(last.value),'<original-cell-%d>'%index,'eval'),scope)
        else:
            exec(compile(tree,'<original-cell-%d>'%index,'exec'),scope)
    return {'cell':index,'original_statements_executed':True,'result_type':type(value).__name__,
            'printed_lines':len(out.getvalue().splitlines())}, value, out.getvalue()


def cpu_function(cell,scope):
    tree=ast.parse(''.join(cell['source']))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='cuda']
    assert len(calls)==1
    target=calls[0]
    assert not target.args and not target.keywords
    target.func.attr='to'
    target.args=[ast.Constant(value='cpu')]
    # 仅替换设备表达式；反向还原后必须与原AST完全相等。
    restore=copy.deepcopy(tree)
    restored=[node for node in ast.walk(restore) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='to']
    assert len(restored)==1
    restored[0].func.attr='cuda'
    restored[0].args=[]
    assert ast.dump(restore,include_attributes=False)==ast.dump(ast.parse(''.join(cell['source'])),include_attributes=False)
    ast.fix_missing_locations(tree)
    exec(compile(tree,'<supplementary-CPU-only-device-change>','exec'),scope)


def verify(path,peft_root):
    import torch
    import transformers
    import sentencepiece as spm
    import peft
    from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer,GenerationConfig
    from peft import LoraConfig,get_peft_model,PeftModel
    from peft.utils import other
    assert torch.__version__=='2.5.1+cpu' and not torch.cuda.is_available()
    assert Path(peft.__file__).resolve().is_relative_to((peft_root/'src').resolve())
    assert peft.__version__=='0.3.0'
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    nb=json.loads(path.read_text(encoding='utf-8'))
    annotated=len(nb['cells'])==57
    cells=nb['cells'][2::2] if annotated else nb['cells']
    assert len(cells)==28
    scope={}
    records=[]
    for i in (1,2,7,8,18,24):
        rec,_,_=execute(cells[i],scope,i)
        records.append(rec)
    prompt_fn=scope['generate_prompt']
    p_none=prompt_fn('Tell a joke')
    p_empty=prompt_fn('Tell a joke','')
    p_input=prompt_fn('Translate to French','Hello')
    assert p_none==p_empty and '### Input:' not in p_none
    assert '\n### Input:\nHello\n\n### Response:' in p_input
    assert p_none.endswith('### Response:') and p_input.endswith('### Response:')
    # 完整英文前缀及换行沿用此前原函数结果，逐字对照，不重写prompt函数。
    old_record=json.loads((path.parents[2]/'复现/outputs/教学验证.json').read_text(encoding='utf-8'))
    assert p_none==old_record['original_prompt_no_input'] and p_input==old_record['original_prompt_with_input']
    with tempfile.TemporaryDirectory(prefix='lesson06-offline-') as temp:
        directory=Path(temp)
        corpus=directory/'corpus.txt'
        corpus.write_text('\n'.join([p_none,p_input,prompt_fn('Explain LoRA'),prompt_fn('Return a number','42'),
          'A short local teaching answer. The model is randomly initialized. Hello Bonjour.']),encoding='utf-8')
        spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(directory/'tokenizer'),vocab_size=96,
           model_type='unigram',character_coverage=1.0,hard_vocab_limit=False,
           unk_id=0,bos_id=1,eos_id=2,pad_id=3,shuffle_input_sentence=False,num_threads=1,minloglevel=2)
        tokenizer=LlamaTokenizer(vocab_file=str(directory/'tokenizer.model'),pad_token='<pad>',legacy=True)
        tokenizer.save_pretrained(directory/'tokenizer_files')
        tokenizer=LlamaTokenizer.from_pretrained(directory/'tokenizer_files',local_files_only=True)
        cfg=LlamaConfig(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=512,
            bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
            attention_dropout=0.,_attn_implementation='eager')
        base=LlamaForCausalLM(cfg).eval()
        base.save_pretrained(directory/'base')
        encoded=tokenizer(p_none,return_tensors='pt')
        with torch.no_grad():
            base_logits=base(**encoded).logits.clone()
        config=LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',
                          target_modules=['q_proj','k_proj','v_proj','o_proj'])
        teaching=get_peft_model(base,config).eval()
        with torch.no_grad():
            for name,module in teaching.named_modules():
                if hasattr(module,'lora_B'):
                    module.lora_A['default'].weight.normal_(0.,0.03)
                    module.lora_B['default'].weight.normal_(0.,0.02)
            adapted_logits=teaching(**encoded).logits.clone()
        assert (adapted_logits-base_logits).abs().max().item()>0
        teaching.save_pretrained(directory/'adapter')
        # 补充本地加载，而非执行原远程标识/8bit/auto/CUDA条件。
        loaded_base=LlamaForCausalLM.from_pretrained(directory/'base',local_files_only=True,attn_implementation='eager').eval()
        scope.update(model=loaded_base,tokenizer=tokenizer)
        for i in (11,12,15):
            rec,_,_=execute(cells[i],scope,i)
            records.append(rec)
        loaded=PeftModel.from_pretrained(loaded_base,str(directory/'adapter')).eval()
        assert loaded.peft_config['default'].target_modules==['q_proj','k_proj','v_proj','o_proj']
        assert loaded.peft_config['default'].inference_mode
        assert not any(p.requires_grad for p in loaded.parameters())
        assert other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']==['q_proj','v_proj']
        with torch.no_grad():
            restored_logits=loaded(**encoded).logits
        reload_error=(restored_logits-adapted_logits).abs().max().item()
        assert torch.allclose(restored_logits,adapted_logits,rtol=1e-5,atol=1e-6), reload_error
        scope['model']=loaded
        for i in (20,21,25,27):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                rec,_,text=execute(cells[i],scope,i)
            rec['warnings']=[str(w.message) for w in caught]
            records.append(rec)
            if i==21:
                assert "['q_proj', 'k_proj', 'v_proj', 'o_proj']" in text and 'model_type llama' in text
        original_config=scope['generation_config']
        assert original_config.temperature==1.5 and original_config.top_p==0.8 and original_config.num_beams==4
        assert original_config.do_sample is False and original_config.num_return_sequences==1
        try:
            scope['inference']('Tell a joke')
        except AssertionError as error:
            assert 'Torch not compiled with CUDA enabled' in str(error)
            cuda_failure={'type':type(error).__name__,'message':str(error),'location':'inputs["input_ids"].cuda()',
                          'model_generate_not_reached':True}
        else:
            raise AssertionError('原函数不应在CPU包中成功执行cuda')
        cpu_function(cells[25],scope)
        observed=[]
        handle=loaded.base_model.model.register_forward_pre_hook(lambda module,args,kwargs:
               observed.append(list(kwargs['input_ids'].shape)),with_kwargs=True)
        # 只观察实际调用结果：方法逻辑原样执行；wrapper透明调用真实PEFT.generate。
        real_generate=loaded.generate
        outputs=[]
        def observe_generate(*args,**kwargs):
            result=real_generate(*args,**kwargs)
            outputs.append(result)
            return result
        loaded.generate=observe_generate
        try:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                returned=scope['inference']('Tell a joke')
            text=captured.getvalue()
        finally:
            loaded.generate=real_generate
            handle.remove()
        assert returned is None and text.startswith('Response:')
        result=outputs[0]
        ids=encoded['input_ids']
        assert result.sequences.shape[0]==1
        assert torch.equal(result.sequences[0,:ids.shape[1]],ids[0])
        assert 0<result.sequences.shape[1]-ids.shape[1]<=256
        assert len(result.scores)>0 and all(s.shape==(4,len(tokenizer)) for s in result.scores)
        assert observed[0]==[4,ids.shape[1]] and any(s==[4,1] for s in observed[1:])
        decoded=tokenizer.decode(result.sequences[0])
        assert text=='Response: '+decoded.split('### Response:')[1].strip()+'\n'
        # 错误反例：标签不存在或在指令里重复，不能只测正常样例。
        try:
            'missing marker'.split('### Response:')[1]
        except IndexError:
            missing_marker=True
        marked=prompt_fn('User text ### Response: misleading marker')
        wrongly_selected=marked.split('### Response:')[1]
        assert 'misleading marker' in wrongly_selected
        response={'sequence_shape':list(result.sequences.shape),'input_shape':list(ids.shape),
                  'new_tokens':result.sequences.shape[1]-ids.shape[1],'returned_sequences':1,
                  'score_steps':len(result.scores),'score_shape':list(result.scores[0].shape),
                  'first_forward_shape':observed[0],'subsequent_forward_shapes':sorted({tuple(s) for s in observed[1:]}),
                  'printed_response_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
                  'sequences_sha256':hashlib.sha256(result.sequences.numpy().tobytes()).hexdigest(),
                  'returns_None':True,'prompt_prefix_preserved':True,'decode_and_original_split_executed':True,
                  'missing_marker_IndexError':missing_marker,'instruction_marker_selects_wrong_segment':True}
        targets=[name for name,module in loaded.named_modules() if hasattr(module,'lora_A')]
        assert len(targets)==4
        for name,module in loaded.named_modules():
            if name in targets:
                assert module.lora_A['default'].weight.shape==(2,16) and module.lora_B['default'].weight.shape==(16,2)
        result_record={'source_sha256':sha(path),'is_annotated':annotated,'original_code_cells_executed':[r['cell'] for r in records],
         'original_code_records':records,'excluded_original_cells':[0,10,14,19,26],
         'original_prompt_branches':{'none_equals_empty':True,'with_input':True,'exact_prior_template_match':True},
         'original_generation_config':{'temperature':1.5,'top_p':0.8,'num_beams':4,'do_sample':False,'num_return_sequences':1},
         'original_inference_CPU_failure':cuda_failure,
         'supplementary_setup':{'random_Llama_layers':1,'hidden_size':16,'intermediate_size':32,'heads':2,'KV_heads':2,
          'real_LlamaTokenizer_SentencePiece_vocab':len(tokenizer),'local_base_reload':True,'local_adapter_reload':True,
          'adapter_random_not_trained':True,'adapter_rank':2,'adapter_alpha':4,'adapter_targets':targets,
          'inference_mode':True,'all_parameters_frozen':True,'A_shape':[2,16],'B_shape':[16,2],
          'logits_shape':list(restored_logits.shape),'nonzero_adapter_logit_difference':(adapted_logits-base_logits).abs().max().item(),
          'save_reload_forward_max_error':reload_error,'save_reload_tolerance':{'rtol':1e-5,'atol':1e-6},
          'loaded_module_merged_flags':{name:module.merged for name,module in loaded.named_modules() if name in targets},
          'attention_implementation':'eager','model_and_adapter_files_in_temporary_directory_only':True},
         'CPU_function_adaptation':{'only_device_expression_changed':'.cuda()→.to("cpu")','original_AST_recoverable_after_reverse_change':True,
           'all_generation_options_including_max_new_tokens_256_preserved':True,'transparent_result_observer_no_generation_mock':True},
         'actual_CPU_generation':response}
    return result_record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--peft-source',required=True,type=Path)
    args=parser.parse_args()
    root=args.peft_source.resolve()
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    assert head==PEFT_COMMIT
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    sys.path.insert(0,str(root/'src'))
    lesson=Path(__file__).resolve().parents[2]
    directory=lesson/'代码/tutorials'
    original=verify(directory/'06_alpaca_llama_7b_lora.ipynb',root)
    annotated=verify(directory/'代码注释.ipynb',root)
    for key in original:
        if key not in ('source_sha256','is_annotated'):
            assert original[key]==annotated[key], key
    record={'status':'部分成功','environment':{'OS':platform.platform(),'python':platform.python_version(),
          'packages':{key:importlib.metadata.version(key) for key in ['torch','transformers','sentencepiece','accelerate','IPython','nbformat']}},
      'fixed_PEFT':{'commit':head,'version':'0.3.0','actual_import_path':str(root/'src/peft'),
        'official_url':'https://github.com/huggingface/peft/tree/'+PEFT_COMMIT,
        'source_sha256':{p:sha(root/p) for p in ['src/peft/peft_model.py','src/peft/mapping.py','src/peft/tuners/lora.py','src/peft/utils/other.py']}},
      'original':original,'annotated':annotated,'structural_and_numerical_results_equal':True,
      'limits':['No original remote 7B model/tokenizer/Alpaca adapter loaded; no bitsandbytes/8bit/CUDA/auto multi-device reproduction',
        'Original inference called in CPU package and fails at .cuda(); supplementary actual generation changes only device expression',
        'Random small LLaMA, locally trained tokenizer and random nonzero adapter are teaching data, not evaluated Alpaca capability',
        'Installation and interactive notebook cells not executed; proxy remains disabled; original histories preserved']}
    (lesson/'复现/outputs/pipeline_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Both original/annotated: 13 code cells executed in explicit CPU teaching context; original CUDA failure recorded.')
    print('True local LLaMA/tokenizer/adapter save-reload and device-adapted 4-beam generation agree; no model-quality claim.')


if __name__=='__main__':
    main()
