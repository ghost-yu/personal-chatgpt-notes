"""直接导入固定官方PEFT v0.3.0，离线验证真实模块替换/前向/梯度/状态。"""
import argparse
import ast
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import List,Optional,Union

EXPECTED_PEFT_SHA='7120c9a2636f93c8db3fc4ae466e02f338ecead8'
os.environ['HF_HUB_OFFLINE']='1'


def source(cell):
    value=cell.get('source','')
    return ''.join(value) if isinstance(value,list) else value


def blocks(cell):
    return [textwrap.dedent(m.group(1)).strip() for m in re.finditer(r'```[^\n]*\n(.*?)```',source(cell),re.S)]


def execute(cell,scope,index):
    tree=ast.parse(source(cell));value=None
    if tree.body and isinstance(tree.body[-1],ast.Expr):
        expression=tree.body.pop()
        exec(compile(tree,'<original-cell-%d>'%index,'exec'),scope)
        value=eval(compile(ast.Expression(expression.value),'<original-cell-%d>'%index,'eval'),scope)
    else:exec(compile(tree,'<original-cell-%d>'%index,'exec'),scope)
    return {'cell':index,'executed_with_teaching_model':True,'result_type':type(value).__name__}


def scan_original_fragment(fragment,model,targets):
    # trace只收集原循环赋值后的结果，不重写匹配表达式或模块列表。
    code=compile(fragment,'<original-markdown8-scan>','exec')
    results={}
    def trace(frame,event,arg):
        if frame.f_code is code and (event=='return' or (event=='line' and frame.f_lineno==2)):
            if 'key' in frame.f_locals and 'target_module_found' in frame.f_locals:
                results[frame.f_locals['key']]=bool(frame.f_locals['target_module_found'])
        return trace
    previous=sys.gettrace()
    try:
        sys.settrace(trace)
        exec(code,{'self':SimpleNamespace(model=model),'lora_config':SimpleNamespace(target_modules=targets),'re':re})
    finally:sys.settrace(previous)
    assert len(results)==len(list(model.named_modules()))
    return [key for key,value in results.items() if value]


def verify(notebook,checkout):
    checkout=Path(checkout).resolve()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(checkout)).decode().strip()
    assert commit==EXPECTED_PEFT_SHA,commit
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=str(checkout)).strip()
    sys.path.insert(0,str(checkout/'src'))
    import torch
    from transformers import BloomConfig,BloomForCausalLM
    import peft
    assert peft.__version__=='0.3.0'
    assert Path(peft.__file__).resolve().is_relative_to(checkout/'src')
    mapping=importlib.import_module('peft.mapping')
    model_module=importlib.import_module('peft.peft_model')
    lora_module=importlib.import_module('peft.tuners.lora')
    from peft.utils import PeftConfig,TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    torch.manual_seed(2026);torch.set_num_threads(2)
    raw=Path(notebook).read_bytes();cells=json.loads(raw.decode('utf-8-sig'))['cells'];assert len(cells)==18
    source_hashes={name:hashlib.sha256((checkout/name).read_bytes()).hexdigest()
        for name in ['src/peft/mapping.py','src/peft/peft_model.py','src/peft/tuners/lora.py','src/peft/utils/other.py']}
    # 原Markdown的两张类型字典，使用真实官方类对象验证，不实例化其他适配方法。
    namespace=dict(model_module.__dict__);namespace.update(mapping.__dict__)
    for block in blocks(cells[2]):exec(compile(block,'<markdown2>','exec'),namespace)
    assert namespace['MODEL_TYPE_TO_PEFT_MODEL_MAPPING']==mapping.MODEL_TYPE_TO_PEFT_MODEL_MAPPING
    assert namespace['PEFT_TYPE_TO_MODEL_MAPPING']==model_module.PEFT_TYPE_TO_MODEL_MAPPING
    target_scope={};exec(compile(blocks(cells[8])[0],'<markdown8-map>','exec'),target_scope)
    assert target_scope['TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING']==TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    # 提供真正父类和标准库依赖后，原缩略数据类仍按预期失败。
    class_scope={'dataclass':dataclasses.dataclass,'field':dataclasses.field,'PeftConfig':PeftConfig,
                 'Optional':Optional,'Union':Union,'List':List}
    try:exec(compile(blocks(cells[7])[0],'<markdown7>','exec'),class_scope)
    except TypeError as error:
        assert 'target_modules' in str(error) and 'default' in str(error)
        dataclass_failure={'type':'TypeError','message':str(error)}
    else:raise AssertionError('节选漏默认值的缺陷不应被修复或隐去')
    tiny_config=BloomConfig(hidden_size=16,n_layer=2,n_head=2,vocab_size=32,
                            hidden_dropout=0.,attention_dropout=0.)
    base=BloomForCausalLM(tiny_config);base.eval()
    attribute_holder=SimpleNamespace()
    exec(compile(blocks(cells[4])[0],'<markdown4>','exec'),
         {'self':attribute_holder,'model':base,'adapter_name':'default'})
    assert attribute_holder.base_model is base and attribute_holder.config is base.config
    assert attribute_holder.active_adapter=='default'
    fragment=blocks(cells[8])[1]
    suffix_hits=scan_original_fragment(fragment,base,['query_key_value'])
    regex_hits=scan_original_fragment(fragment,base,'.*query_key_value')
    bare_hits=scan_original_fragment(fragment,base,'query_key_value')
    wrong_hits=scan_original_fragment(fragment,base,['q_proj'])
    assert len(suffix_hits)==2 and suffix_hits==regex_hits and not bare_hits and not wrong_hits
    original_targets={name:base.get_submodule(name) for name in suffix_hits}
    ids=torch.tensor([[4,5,6]],dtype=torch.int64)
    with torch.no_grad():original_logits=base(input_ids=ids).logits.detach().clone()
    scope={'model':base};runs=[]
    for index in [9,11,13,14]:runs.append(execute(cells[index],scope,index))
    calls=[];functions={'get_peft_model','add_adapter','_prepare_lora_config','_find_and_replace',
                       '_replace_module','mark_only_lora_as_trainable'}
    source_prefix=str(checkout/'src').replace('\\','/').lower()
    def profile(frame,event,arg):
        if event=='call' and frame.f_code.co_name in functions and frame.f_code.co_filename.replace('\\','/').lower().startswith(source_prefix):
            calls.append(frame.f_code.co_name)
    previous=sys.getprofile()
    try:
        sys.setprofile(profile)
        runs.append(execute(cells[15],scope,15))
    finally:sys.setprofile(previous)
    for index in [16,17]:runs.append(execute(cells[index],scope,index))
    wrapped=scope['model'];wrapped.eval()
    assert type(wrapped).__name__=='PeftModelForCausalLM'
    assert isinstance(wrapped.base_model,lora_module.LoraModel) and wrapped.base_model.model is base
    assert wrapped.config is base.config
    assert calls.count('_replace_module')==2 and '_find_and_replace' in calls and 'mark_only_lora_as_trainable' in calls
    replacements=[]
    for name,old in original_targets.items():
        new=base.get_submodule(name)
        assert new is not old and isinstance(new,lora_module.Linear)
        assert new.weight is old.weight and new.bias is old.bias
        assert new.lora_A['default'].weight.shape==(16,16)
        assert new.lora_B['default'].weight.shape==(48,16)
        assert new.scaling['default']==2
        replacements.append({'path':name,'old_class':type(old).__name__,'new_class':type(new).__name__,
                             'old_class_module':type(old).__module__,'new_class_module':type(new).__module__,
                             'weight_and_bias_identity_preserved':True,'a_shape':[16,16],'b_shape':[48,16]})
    trainable=[name for name,p in wrapped.named_parameters() if p.requires_grad]
    assert len(trainable)==4 and all('lora_' in name for name in trainable)
    with torch.no_grad():initial_logits=wrapped(input_ids=ids).logits
    torch.testing.assert_close(initial_logits,original_logits,rtol=0,atol=0)
    layer=base.get_submodule(suffix_hits[0])
    x=torch.randn(2,3,16)
    initial=layer(x);initial.sum().backward()
    a=layer.lora_A['default'].weight;b=layer.lora_B['default'].weight
    assert a.grad is not None and torch.count_nonzero(a.grad)==0
    assert b.grad is not None and b.grad.norm()>0
    assert layer.weight.grad is None and layer.bias.grad is None
    b_grad=float(b.grad.norm());wrapped.zero_grad(set_to_none=True)
    # 人工设置非零B用于验证分支，不称为训练后的adapter。
    with torch.no_grad():b.normal_(mean=0.,std=.05)
    actual=layer(x)
    baseline=torch.nn.functional.linear(x,layer.weight,layer.bias)
    expected=baseline+2*torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    formula_error=float((actual-expected).abs().max().detach())
    with wrapped.disable_adapter():
        disabled=layer(x)
        torch.testing.assert_close(disabled,baseline,rtol=0,atol=0)
    assert not layer.disable_adapters
    saved_weight=layer.weight.detach().clone()
    layer.merge();assert layer.merged
    merged=layer(x)
    torch.testing.assert_close(merged,actual,rtol=1e-5,atol=1e-6)
    merge_error=float((merged-actual).abs().max().detach())
    layer.unmerge();assert not layer.merged
    torch.testing.assert_close(layer.weight,saved_weight,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(layer(x),actual,rtol=1e-5,atol=1e-6)
    # 错误目标在真实官方注入中报错；使用新模型避免失败副作用影响上面结果。
    failures={}
    for label,targets in [('bare_regex','query_key_value'),('wrong_suffix',['q_proj'])]:
        fresh=BloomForCausalLM(tiny_config)
        config=peft.LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=targets)
        try:peft.get_peft_model(fresh,config)
        except ValueError as error:
            assert 'not found' in str(error)
            failures[label]={'type':'ValueError','message':str(error)}
        else:raise AssertionError('错误目标应该失败')
    return {'status':'部分成功','python':platform.python_version(),'torch':torch.__version__,
        'transformers':importlib.metadata.version('transformers'),'accelerate':importlib.metadata.version('accelerate'),
        'peft_imported_version':peft.__version__,'peft_checkout_commit':commit,'source_hashes':source_hashes,
        'device':'cpu','quantized':False,'tiny_model':{'hidden_size':16,'layers':2,'heads':2,'vocab_size':32,'random_weights':True},
        'original_code_cells_executed':runs,'markdown_mapping_dictionaries_equal_to_official':True,
        'markdown4_attribute_fragment':'独立属性上下文执行成功；最终包装结构另由官方构造器验证',
        'markdown7_expected_failure':dataclass_failure,'suffix_hits':suffix_hits,'regex_hits':regex_hits,
        'bare_regex_hits':bare_hits,'wrong_suffix_hits':wrong_hits,'official_injection_failures':failures,
        'official_call_sequence':calls,'wrapper_reuses_original_model':True,'replacements':replacements,
        'trainable_parameters':sum(p.numel() for p in wrapped.parameters() if p.requires_grad),
        'only_lora_trainable':True,'zero_b_initial_logits_equal':True,'logits_shape':list(initial_logits.shape),
        'initial_a_gradient_zero':True,'initial_b_gradient_norm':b_grad,'base_weight_bias_gradients_none':True,
        'adapter_output_shape':list(actual.shape),'manual_forward_max_error':formula_error,'disable_matches_base':True,
        'merge_max_error':merge_error,'unmerge_restores_weight_within_tolerance':True,
        'limits':'直接执行官方PEFT v0.3.0普通Linear路径，搭配较新torch/Transformers；未加载7B、未测8bit/CUDA、未训练或复现论文指标'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--notebook',required=True)
    parser.add_argument('--peft-source',required=True)
    parser.add_argument('--record',required=True)
    args=parser.parse_args();report=verify(args.notebook,args.peft_source)
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
