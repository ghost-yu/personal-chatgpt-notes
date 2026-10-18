"""Execute the four teaching blocks with complete fixed TRL and a real tiny OPT."""
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import re
import sys
import warnings

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
SOURCES=Path(os.environ.get('LESSON23_SOURCES',
    'G:/zzz/Personal ChatGPT 系列/23 TRL Reward Model 与 RewardTrainer/资料'))
sys.path.insert(0,str(SOURCES/'runtime-libs'))
for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:
    os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tokenizer():
    from transformers import AutoTokenizer
    manifest=json.loads((ROOT/'来源清单.json').read_text(encoding='utf-8'))
    for name,digest in manifest['tokenizer']['local_files_sha256'].items():
        assert sha(SOURCES/'opt-tokenizer'/name)==digest,name
    return AutoTokenizer.from_pretrained(SOURCES/'opt-tokenizer',local_files_only=True)


def check_runtime_sources():
    from trl import RewardTrainer
    import trl.trainer.utils as utils
    from trl.trainer.reward_config import RewardConfig
    manifest=json.loads((ROOT/'来源清单.json').read_text(encoding='utf-8'))
    for obj,name in [(RewardTrainer,'reward_trainer.py'),(utils,'trainer_utils.py'),(RewardConfig,'reward_config.py')]:
        assert sha(inspect.getfile(obj))==manifest['official_source_sha256'][name],name
    assert importlib.metadata.version('trl')=='0.8.6'
    return manifest['official_source_sha256']


def strip_doc(node):
    node=copy.deepcopy(node)
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
    return ast.dump(node,include_attributes=False)


def additional_checks(scope):
    from trl.trainer.utils import compute_accuracy
    trainer=scope['trainer'];model=trainer.model.eval();tok=scope['tokenizer'];rows=scope['rows'];batch=scope['batch']
    assert trainer.state.global_step==2
    assert sum(p.numel()for p in model.parameters()if p.requires_grad)==72
    changed=[];frozen=[];trainable=[]
    for name,p in model.named_parameters():
        before=scope['before_state'][name]
        if p.requires_grad:
            trainable.append(name)
            assert not torch.equal(before,p),name
            changed.append(name)
        else:
            assert torch.equal(before,p),name
            frozen.append(name)
    optimized={id(p)for group in trainer.optimizer.param_groups for p in group['params']}
    assert optimized=={id(p)for p in model.parameters()if p.requires_grad}
    assert len(changed)==5 and any('modules_to_save.default.weight' in name for name in changed)
    loss,out=trainer.compute_loss(model,batch,return_outputs=True)
    chosen=out['rewards_chosen'];rejected=out['rewards_rejected'];difference=chosen-rejected
    gc,gr=torch.autograd.grad(loss,(chosen,rejected),retain_graph=True)
    assert torch.allclose(gc,-torch.sigmoid(-difference)/len(rows),atol=1e-7)
    assert torch.allclose(gr,-gc,atol=1e-7)
    assert torch.allclose(loss,F.softplus(-difference).mean(),atol=1e-7)
    prediction_loss,probability,labels=trainer.prediction_step(model,batch,False)
    assert list(probability.shape)==[2,2]
    assert torch.equal(labels,torch.zeros(2))
    assert torch.allclose(probability[:,0],torch.sigmoid(difference[:,0]),atol=1e-7)
    base=model.get_base_model();padding_errors={};pool_errors={}
    trained_state=model.state_dict();reloaded_state=scope['reloaded'].state_dict()
    assert trained_state.keys()==reloaded_state.keys()
    for name,value in trained_state.items():assert torch.equal(value,reloaded_state[name]),name
    with torch.no_grad():
        for side in ['chosen','rejected']:
            ids=batch['input_ids_'+side];mask=batch['attention_mask_'+side]
            hidden=base.model.decoder(input_ids=ids,attention_mask=mask).last_hidden_state
            weights=base.score.modules_to_save['default'].weight
            all_scores=hidden@weights.T
            # Select the last mask=1 position independently of OPT's pooling implementation.
            positions=torch.arange(ids.shape[1]).expand_as(ids)
            last=positions.masked_fill(mask==0,-1).max(-1).values
            reference=all_scores[torch.arange(len(rows)),last]
            actual=out['rewards_'+side]
            restored=scope['reloaded'](input_ids=ids,attention_mask=mask).logits
            assert torch.equal(restored,actual)
            pool_errors[side]=(reference-actual).abs().max().item()
            assert torch.allclose(reference,actual,atol=2e-7,rtol=1e-5)
            singles=[]
            for row in rows:
                singles.append(model(input_ids=torch.tensor([row['input_ids_'+side]]),
                                     attention_mask=torch.tensor([row['attention_mask_'+side]])).logits)
            independent=torch.cat(singles)
            padding_errors[side]=(independent-actual).abs().max().item()
            assert torch.allclose(independent,actual,atol=2e-7,rtol=1e-5)
        tok.padding_side='left';left=trainer.data_collator(rows)
        _,left_out=trainer.compute_loss(model,left,return_outputs=True)
        tok.padding_side='right'
        left_error=max((left_out[k]-out[k]).abs().max().item()for k in out)
        assert left_error<2e-7
        assert scope['reload_error']==0
    # Exercise the optional fixed-version margin branch with actual default collation.
    margins=[.2,.8];margin_rows=[{**row,'margin':margin}for row,margin in zip(rows,margins)]
    margin_batch=trainer.data_collator(margin_rows)
    assert list(margin_batch['margin'].shape)==[2]
    actual_margin=trainer.compute_loss(model,margin_batch)
    assert torch.allclose(actual_margin,F.softplus(-(difference-margin_batch['margin'])).mean(),atol=1e-7)
    assert list((difference-margin_batch['margin']).shape)==[2,2]
    column_batch={**margin_batch,'margin':margin_batch['margin'][:,None]}
    corrected_margin=trainer.compute_loss(model,column_batch)
    assert torch.allclose(corrected_margin,F.softplus(-(difference-column_batch['margin'])).mean(),atol=1e-7)
    # Nonuniform independent scores make row/column broadcasting visibly different.
    d=torch.tensor([[.2],[1.5]])
    mixed=F.softplus(-(d-torch.tensor(margins))).mean().item()
    paired=F.softplus(-(d-torch.tensor(margins)[:,None])).mean().item()
    assert abs(mixed-paired)>.005
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tie=compute_accuracy((torch.tensor([[.5,.5],[.8,.2]]).numpy(),torch.zeros(2).numpy()))
    assert tie['accuracy']==1.
    shifted=F.softplus(-((chosen.double()+100)-(rejected.double()+100))).mean()
    assert torch.allclose(shifted,loss.double(),atol=1e-7)
    extremes=torch.tensor([-1000.,0.,1000.],requires_grad=True)
    stable=F.softplus(-extremes);stable.sum().backward()
    assert torch.isfinite(stable).all()and torch.isfinite(extremes.grad).all()
    assert torch.isinf(-torch.log(torch.sigmoid(torch.tensor(-1000.))))
    # Default padding pads; it does not truncate an already-tokenized long row.
    long_row={k:([2]*600 if k.startswith('input_ids')else [1]*600)for k in rows[0]}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always');long_batch=trainer.data_collator([long_row])
    assert long_batch['input_ids_chosen'].shape[1]==600
    long_examples={'chosen':[' very'*600],'rejected':['No.']}
    processed=scope['preprocess'](long_examples)
    assert len(processed['input_ids_chosen'][0])>512
    assert not (len(processed['input_ids_chosen'][0])<=512 and len(processed['input_ids_rejected'][0])<=512)
    assert tok.model_max_length==1024
    for name,value in model.state_dict().items():
        assert torch.equal(value,scope['after_training_state'][name]),name
    for p in model.parameters():p.grad=None
    return {'base_parameters':scope['base_count'],'wrapped_parameters':sum(p.numel()for p in model.parameters()),
            'trainable_parameters':72,'optimizer_steps':trainer.state.global_step,'changed_trainable_names':changed,
            'frozen_parameter_tensors':len(frozen),'optimizer_matches_trainable':True,
            'loss_before':scope['training_loss_before'],'loss_after':scope['training_loss_after'],
            'batch_shapes':{k:list(v.shape)for k,v in batch.items()if torch.is_tensor(v)},
            'chosen_scores':chosen.detach().tolist(),'rejected_scores':rejected.detach().tolist(),
            'pair_probabilities':probability.tolist(),'chosen_score_gradients':gc.tolist(),'rejected_score_gradients':gr.tolist(),
            'independent_pool_errors':pool_errors,'single_sequence_padding_errors':padding_errors,
            'left_right_padding_error':left_error,'reload_error':scope['reload_error'],'checkpoint_keys':scope['checkpoint_keys'],
            'reload_all_state_tensors_equal':True,'reload_chosen_and_rejected_scores_equal':True,
            'margin_shape':[2],'broadcast_shape':[2,2],'margin_loss':actual_margin.item(),
            'column_margin_loss':corrected_margin.item(),'independent_mixed_margin_loss':mixed,
            'independent_paired_margin_loss':paired,'tie_accuracy':tie,'tie_warning':[str(w.message)for w in caught],
            'stable_loss':stable.tolist(),'stable_gradients':extremes.grad.tolist(),
            'collator_did_not_truncate_600_tokens':True,
            'tokenizer_max_length':tok.model_max_length,
            'long_pair_chosen_length':len(processed['input_ids_chosen'][0]),
            'scoring_and_reload_preserve_trained_parameters':True}


def main():
    torch.set_num_threads(1)
    sources=check_runtime_sources()
    note=ROOT/'解释.md';text=note.read_text(encoding='utf-8').split('<details>',1)[0]
    blocks=re.findall(r'```python\n(.*?)```',text,re.S)
    assert len(blocks)==4,len(blocks)
    scope={'__name__':'_lesson23_teaching_'};stdout=[];previous=os.getcwd()
    try:
        os.chdir(ROOT)
        for i,block in enumerate(blocks):
            out=io.StringIO()
            with contextlib.redirect_stdout(out):exec(compile(block,f'<lesson23-main-block-{i+1}>','exec'),scope)
            stdout.append(out.getvalue())
    finally:os.chdir(previous)
    actual=next(n for n in ast.parse(blocks[1]).body if isinstance(n,ast.FunctionDef)and n.name=='preprocess')
    notebook=json.loads((ROOT/'代码/tutorials/reward_model/trl_reward_trainer.ipynb').read_text(encoding='utf-8'))
    original=ast.parse(''.join(notebook['cells'][14]['source'])).body[0]
    assert strip_doc(actual)==strip_doc(original)
    assert abs(scope['scalar_before']-.6931471805599453)<1e-7
    assert abs(scope['scalar_after']-.6443966600735709)<1e-7
    assert scope['scalar_gradients']==[-.5,.5]
    checks=additional_checks(scope)
    figures=json.loads((ROOT/'复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
    for item in figures['crops']:assert sha(ROOT/item['file'])==item['sha256']
    report={'date':'2026-10-18','note_sha256':sha(note),'script_sha256':sha(__file__),
            'official_source_sha256':sources,'tokenizer_source':str(SOURCES/'opt-tokenizer'),
            'python':sys.version,'packages':{name:importlib.metadata.version(name)for name in
                ['torch','transformers','trl','peft','accelerate','datasets']},
            'main_blocks':4,'stdout':stdout,'preprocess_ast_matches_original':True,'checks':checks,
            'scope':'Real CPU OPT/LoRA and full fixed TRL, two optimizer updates on two synthetic preference pairs; no original weights, HH corpus, 4bit, PPO, human quality evaluation or GPU.'}
    path=ROOT/'复现/outputs/基础讲解验证.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'blocks':4,'base_parameters':checks['base_parameters'],'trainable_parameters':72,
                      'steps':checks['optimizer_steps'],'loss_before':checks['loss_before'],'loss_after':checks['loss_after'],
                      'reload_error':checks['reload_error'],'result':str(path)},ensure_ascii=False))


if __name__=='__main__':main()
