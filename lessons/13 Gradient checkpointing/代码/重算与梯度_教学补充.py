# -*- coding: utf-8 -*-
"""真实Torch2.0.1 CPU教学实验；不是原课程单元或GPU显存benchmark。"""
import copy
import warnings
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class Block(nn.Module):
    """继承Module的8维线性+sigmoid块，记录调用时是否启用autograd。"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8,8)
        self.calls = []

    def forward(self,x):
        self.calls.append(torch.is_grad_enabled())
        return self.linear(x).sigmoid()


def saved_forward(function, model):
    """仅观察初次forward保存事件；不把它当进程峰值内存。"""
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    events = []
    def pack(tensor):
        storage = tensor.untyped_storage()
        # 只保存元数据用于报告；真正用于backward的tensor仍由autograd持有。
        events.append(dict(shape=list(tensor.shape),dtype=str(tensor.dtype),
                           pointer=storage.data_ptr(),bytes=storage.nbytes(),parameter=storage.data_ptr() in params))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        output = function()
    nonparam = {e['pointer']:e['bytes'] for e in events if not e['parameter']}
    return output, dict(saved_tensor_events=len(events),unique_nonparameter_storage_bytes=sum(nonparam.values()),
                        scope="initial forward saved-tensor hooks, includes input; excludes parameter storage; not live/peak/RSS/CUDA memory",
                        events=[{k:v for k,v in e.items() if k!='pointer'} for e in events])


def sequential_experiment():
    torch.manual_seed(41)
    template = nn.Sequential(*[Block() for _ in range(4)]).double()
    initial_x = torch.randn(2,8,dtype=torch.float64)
    runs = {}
    tensors = {}
    for mode in ('ordinary','checkpoint'):
        model = copy.deepcopy(template)
        x = initial_x.clone().requires_grad_()
        function = (lambda:model(x)) if mode=='ordinary' else (
            lambda:checkpoint_sequential(model,2,x,use_reentrant=True))
        output, saved = saved_forward(function,model)
        loss = output.square().mean()
        loss.backward()
        grads = {n:p.grad.clone() for n,p in model.named_parameters()}
        tensors[mode]=(output.detach(),x.grad.clone(),grads)
        runs[mode]=dict(output=output.detach().tolist(),loss=float(loss),input_gradient=x.grad.tolist(),
                        calls=[m.calls for m in model],saved_forward=saved)
    left,right=tensors['ordinary'],tensors['checkpoint']
    torch.testing.assert_close(left[0],right[0],rtol=0,atol=0)
    torch.testing.assert_close(left[1],right[1],rtol=1e-12,atol=1e-12)
    error=max(float((left[2][n]-right[2][n]).abs().max()) for n in left[2])
    assert error<1e-12
    assert [len(c) for c in runs['ordinary']['calls']]==[1,1,1,1]
    assert runs['checkpoint']['calls']==[[False,True],[False,True],[True],[True]]
    assert runs['checkpoint']['saved_forward']['unique_nonparameter_storage_bytes'] < runs['ordinary']['saved_forward']['unique_nonparameter_storage_bytes']
    return dict(runs=runs,parameter_gradient_max_difference=error,output_exact=True,
                input_gradient_max_difference=float((left[1]-right[1]).abs().max()))


def rng_experiment():
    """同一次初始forward、不同重算mask：输出可相同，但梯度会改变。"""
    torch.manual_seed(55)
    template=nn.Sequential(nn.Linear(8,8),nn.Dropout(.5)).double()
    values=torch.randn(3,8,dtype=torch.float64)
    runs={}; tensors={}
    for mode in ('ordinary','preserve','no_preserve'):
        model=copy.deepcopy(template);x=values.clone().requires_grad_(); replay=[]
        def function(inp):
            result=model(inp)
            replay.append(result.detach().clone())
            return result.square()
        torch.manual_seed(777)
        result=function(x) if mode=='ordinary' else checkpoint(
            function,x,use_reentrant=True,preserve_rng_state=(mode=='preserve'))
        result.mean().backward()
        tensors[mode]=(result.detach(),{n:p.grad.clone() for n,p in model.named_parameters()},x.grad.clone())
        runs[mode]=dict(function_calls=len(replay),replay_output_equal=(None if len(replay)==1 else torch.equal(replay[0],replay[1])))
    for mode in ('preserve','no_preserve'):
        assert torch.equal(tensors['ordinary'][0],tensors[mode][0])
        error=max(float((tensors['ordinary'][1][n]-tensors[mode][1][n]).abs().max()) for n in tensors['ordinary'][1])
        runs[mode]['parameter_gradient_max_difference']=error
    assert runs['preserve']['replay_output_equal'] and runs['preserve']['parameter_gradient_max_difference']==0
    assert not runs['no_preserve']['replay_output_equal'] and runs['no_preserve']['parameter_gradient_max_difference']>0
    return runs


def input_gradient_boundary():
    """reentrant要求边界输入参与autograd；参数可训练不替代此条件。"""
    torch.manual_seed(9); template=nn.Linear(4,4).double(); values=torch.randn(2,4,dtype=torch.float64)
    bad=copy.deepcopy(template)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=checkpoint(bad,values,use_reentrant=True)
        assert not result.requires_grad
        try: result.square().mean().backward()
        except RuntimeError as error: failure=str(error)
        else: raise AssertionError('reentrant input boundary should fail')
    runs={}
    gradients={}
    for mode in ('ordinary','requires_grad_input','non_reentrant'):
        model=copy.deepcopy(template)
        x=values.clone().requires_grad_(mode=='requires_grad_input')
        if mode=='ordinary': result=model(x)
        else: result=checkpoint(model,x,use_reentrant=(mode!='non_reentrant'))
        result.square().mean().backward()
        gradients[mode]={n:p.grad.clone() for n,p in model.named_parameters()}
        runs[mode]=dict(output_requires_grad=result.requires_grad,parameter_gradient_norm=float(model.weight.grad.norm()),input_has_grad=x.grad is not None)
    for mode in ('requires_grad_input','non_reentrant'):
        assert all(torch.equal(gradients['ordinary'][n],gradients[mode][n]) for n in gradients['ordinary'])
    return dict(reentrant_no_grad_input_failure=failure,warnings=[str(w.message) for w in caught],
                corrected_runs=runs,corrected_parameter_gradients_match=True)


def frozen_bert_embeddings(config):
    """冻结embedding并保持encoder可训练，检查真实BERT旧版重算的梯度边界。"""
    from transformers import BertForSequenceClassification
    torch.manual_seed(101); template=BertForSequenceClassification(config)
    ids=torch.tensor([[100,101,102,103],[104,105,106,107]])
    labels=torch.tensor([0,1]); reports={}
    for repaired in (False,True):
        model=copy.deepcopy(template);model.train()
        for p in model.bert.embeddings.parameters(): p.requires_grad=False
        assert model.gradient_checkpointing_enable() is None
        if repaired: model.enable_input_require_grads()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model(input_ids=ids,labels=labels).loss.backward()
        encoder=[p for p in model.bert.encoder.parameters()]
        count=sum(p.grad is not None for p in encoder)
        assert all(p.grad is None for p in model.bert.embeddings.parameters())
        assert model.classifier.weight.grad is not None
        assert count==len(encoder) if repaired else count==0
        reports['input_hook' if repaired else 'frozen_without_input_hook']=dict(
            encoder_parameters_with_grad=count,total_encoder_parameter_tensors=len(encoder),
            embedding_stays_frozen=True,classifier_has_grad=True,warnings=[str(w.message) for w in caught])
    return reports
