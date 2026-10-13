"""真实库的离线CPU调用链；提取原单元执行，小模型/本地数据/训练参数明确另设。"""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import tempfile
from pathlib import Path

# 保证没有把原7B下载或远程数据加载偷偷带入教学验证。
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from torch import nn
import transformers
import peft
from peft import LoraConfig,get_peft_model
from transformers import BloomConfig,BloomForCausalLM,PreTrainedTokenizerFast
from transformers import Trainer,TrainingArguments,DataCollatorForLanguageModeling
from datasets import Dataset,DatasetDict,disable_progress_bars
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def source(cell):
    x=cell.get('source','')
    return ''.join(x) if isinstance(x,list) else x


def execute(cell,scope,index):
    # 模拟notebook末尾表达式，但不把对象完整repr写进公开验证记录。
    tree=ast.parse(source(cell))
    value=None
    if tree.body and isinstance(tree.body[-1],ast.Expr):
        expr=tree.body.pop()
        exec(compile(tree,'<original-cell-%d>'%index,'exec'),scope)
        value=eval(compile(ast.Expression(expr.value),'<original-cell-%d>'%index,'eval'),scope)
    else:
        exec(compile(tree,'<original-cell-%d>'%index,'exec'),scope)
    return {'cell':index,'original_source_executed':True,'result_type':type(value).__name__}


def verify(notebook):
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    disable_progress_bars()
    raw=Path(notebook).read_bytes();cells=json.loads(raw.decode('utf-8-sig'))['cells']
    is_annotated=len(cells)==95
    if is_annotated:
        # 生成版在每个原格前插入说明；提取原编号位置，仍不运行安装/下载格。
        cells=cells[2::2]
    assert len(cells)==47
    # 明确新编数据，不下载Abirate数据；至少5行，覆盖原第五行索引。
    quotes=['Be yourself.','Keep learning every day.','Build small useful things.',
            'Try again.','Read carefully and test the code.']
    rows={'quote':quotes,'author':['Teaching example']*5,
          'tags':[['life'],['learning'],['work'],['life'],['learning','work']]}
    texts=[q+' ->: '+str(t) for q,t in zip(rows['quote'],rows['tags'])]
    backend=Tokenizer(WordLevel({'[PAD]':0,'[UNK]':1,'[BOS]':2,'[EOS]':3},unk_token='[UNK]'))
    backend.pre_tokenizer=Whitespace()
    # 使用真实tokenizers训练小词表，避免用假tokenizer接口冒充原分词。
    from tokenizers.trainers import WordLevelTrainer
    backend.train_from_iterator(texts,WordLevelTrainer(special_tokens=['[PAD]','[UNK]','[BOS]','[EOS]']))
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',unk_token='[UNK]',
                                     bos_token='[BOS]',eos_token='[EOS]',
                                     model_input_names=['input_ids','attention_mask'])
    config=BloomConfig(vocab_size=len(tokenizer),hidden_size=16,n_layer=2,n_head=2,
                       hidden_dropout=0.,attention_dropout=0.,bos_token_id=2,eos_token_id=3,pad_token_id=0)
    model=BloomForCausalLM(config)
    scope={'torch':torch,'nn':nn,'transformers':transformers,'model':model,'tokenizer':tokenizer,
           'dataset':DatasetDict({'train':Dataset.from_dict(rows)}),
           'LoraConfig':LoraConfig,'get_peft_model':get_peft_model}
    before_wrap=[]
    for index in [10,11,12,14,15,16,17,19,20,21,22]:
        if index==21:
            assert all(not p.requires_grad for p in scope['model'].parameters())
            assert scope['model'].is_gradient_checkpointing
            emb=scope['model'].get_input_embeddings()
            assert not emb.weight.requires_grad
            assert emb(torch.tensor([[4]])).requires_grad
        before_wrap.append(execute(cells[index],scope,index))
    model=scope['model']
    assert model.peft_config['default'].r==16
    assert model.peft_config['default'].lora_alpha==32
    targets=[name for name,module in model.named_modules() if hasattr(module,'lora_A') and len(module.lora_A)]
    assert len(targets)==2 and all(name.endswith('query_key_value') for name in targets)
    for name,module in model.named_modules():
        if name in targets:
            assert module.lora_A['default'].weight.shape==(16,16)
            assert module.lora_B['default'].weight.shape==(48,16)
            assert module.scaling['default']==2
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in name for name in trainable)
    data_records=[]
    # 独立检查merge确实修改传入dict；不执行整个33以免重复map。
    function_tree=ast.Module(body=[node for node in ast.parse(source(cells[33])).body if isinstance(node,ast.FunctionDef)],type_ignores=[])
    exec(compile(function_tree,'<merge>','exec'),scope)
    item={'quote':'Be yourself.','tags':['life']}
    assert scope['merge'](item) is item and item['prediction']=="Be yourself. ->: ['life']"
    for index in [26,27,28,29,30,31,32,33,34,35,36,38,39,41]:
        data_records.append(execute(cells[index],scope,index))
    dataset=scope['dataset']
    collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)
    features=[{key:dataset['train'][i][key] for key in ['input_ids','attention_mask']} for i in [0,4]]
    batch=collator(features)
    assert batch['input_ids'].dtype==torch.int64
    pads=batch['input_ids'].eq(tokenizer.pad_token_id)
    assert pads.any() and torch.all(batch['labels'][pads]==-100)
    assert torch.equal(batch['labels'][~pads],batch['input_ids'][~pads])
    # 真实BLOOM forward返回loss，与手动shift交叉熵比较。
    model.config.use_cache=False
    model.eval()
    result=model(**batch)
    shifted_labels=batch['labels'][:,1:].contiguous()
    manual_loss=torch.nn.functional.cross_entropy(result.logits[:,:-1,:].contiguous().view(-1,len(tokenizer)),
                                                  shifted_labels.view(-1),ignore_index=-100)
    torch.testing.assert_close(result.loss,manual_loss,rtol=1e-6,atol=1e-6)
    result.loss.backward()
    assert all(p.grad is None for name,p in model.named_parameters() if 'lora_' not in name)
    nonzero=[name for name,p in model.named_parameters() if p.grad is not None and p.grad.abs().sum()>0]
    assert nonzero and all('lora_B' in name for name in nonzero)
    model.zero_grad(set_to_none=True)
    snapshot={name:p.detach().clone() for name,p in model.named_parameters()}
    # 原42参数不能在CPU直接执行，下面明确设置教学差异，不改原文件。
    with tempfile.TemporaryDirectory(prefix='lesson03-trainer-') as temp:
        arguments=TrainingArguments(output_dir=temp,use_cpu=True,fp16=False,
             per_device_train_batch_size=2,gradient_accumulation_steps=2,max_steps=2,
             warmup_steps=0,learning_rate=2e-3,logging_steps=1,save_strategy='no',
             report_to=[],disable_tqdm=True,dataloader_pin_memory=False,optim='adamw_torch',seed=2026)
        trainer=Trainer(model=model,args=arguments,train_dataset=dataset['train'],data_collator=collator)
        train_result=trainer.train()
        assert trainer.state.global_step==2 and train_result.training_loss>0
        scope['trainer']=trainer
        data_records.append(execute(cells[46],scope,46))
    changed=[name for name,p in model.named_parameters() if not torch.equal(p,snapshot[name])]
    assert changed and all('lora_' in name for name in changed)
    assert any('lora_B' in name for name in changed)
    model.eval()
    prompt=tokenizer('Be yourself. ->:',return_tensors='pt')
    with torch.no_grad():
        generated=model.generate(**prompt,max_new_tokens=3,do_sample=False)
    added=generated.shape[1]-prompt['input_ids'].shape[1]
    assert 1<=added<=3
    packages=['torch','transformers','peft','accelerate','datasets','tokenizers','pyarrow','pandas','numpy']
    return {'status':'部分成功','python':platform.python_version(),
        'versions':{name:importlib.metadata.version(name) for name in packages},
        'source_sha256':hashlib.sha256(raw).hexdigest(),'annotated_notebook':is_annotated,
        'device':'cpu','offline':True,
        'tiny_model':{'hidden_size':16,'n_layer':2,'n_head':2,'vocab_size':len(tokenizer),'random_weights':True,'quantized':False},
        'original_cells_executed':before_wrap+data_records,
        'lora_targets':targets,'lora_scale':2,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'embedding_weight_frozen_output_grad_enabled':True,'gradient_checkpointing_enabled':True,
        'synthetic_dataset_rows':len(dataset['train']),'tokenized_fields':dataset['train'].column_names,
        'collator_shape':list(batch['input_ids'].shape),'padding_targets_ignored':int(pads.sum()),
        'shift_valid_targets':int(shifted_labels.ne(-100).sum()),'clm_loss':float(result.loss.detach()),
        'manual_shift_loss_error':float((result.loss-manual_loss).abs().detach()),
        'initial_nonzero_gradient_names':nonzero,'base_weights_unchanged':True,'changed_adapter_names':changed,
        'teaching_training':{'global_step':2,'micro_batch':2,'accumulation':2,'fp16':False,'warmup':0,
                             'learning_rate':2e-3,'training_loss':train_result.training_loss},
        'generated_new_tokens':added,'generated_shape':list(generated.shape),
        'not_reproduced':['7B pretrained/8bit/device_map加载','原BLOOM tokenizer','Abirate数据集','watermark原环境',
                          '原FP16/200步训练','两条原CUDA生成','论文任务指标'],
        'limits':'新版固定库、随机小型BLOOM、新编5条数据和自建词表；真实调用链验证，不是原7B或论文性能复现'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--notebook',required=True)
    parser.add_argument('--record',required=True)
    parser.add_argument('--annotated-notebook',help='可选：再执行注释版中的相同26格，比较真实CPU调用链结果')
    args=parser.parse_args()
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        report=verify(args.notebook)
        if args.annotated_notebook:
            annotated=verify(args.annotated_notebook)
            assert not report['annotated_notebook'] and annotated['annotated_notebook']
            keys=['tiny_model','lora_targets','lora_scale','trainable_parameters',
                  'tokenized_fields','collator_shape','padding_targets_ignored','shift_valid_targets',
                  'initial_nonzero_gradient_names','changed_adapter_names','base_weights_unchanged',
                  'generated_new_tokens','generated_shape']
            for key in keys:
                assert report[key]==annotated[key],('annotation execution mismatch',key)
            assert [r['cell'] for r in report['original_cells_executed']]==[r['cell'] for r in annotated['original_cells_executed']]
            for key in ['clm_loss','manual_shift_loss_error']:
                assert abs(report[key]-annotated[key])<1e-7,('annotation execution mismatch',key)
            assert abs(report['teaching_training']['training_loss']-annotated['teaching_training']['training_loss'])<1e-7
            report['annotated_execution_comparison']={
                'source_sha256':annotated['source_sha256'],
                'executed_original_cell_indices':[r['cell'] for r in annotated['original_cells_executed']],
                'structural_keys_equal':keys,'clm_loss':annotated['clm_loss'],
                'training_loss':annotated['teaching_training']['training_loss'],
                'losses_equal_within_1e_minus_7':True,
                'same_two_step_cpu_pipeline_passed':True}
    report['captured_library_stdout']=captured.getvalue()
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
