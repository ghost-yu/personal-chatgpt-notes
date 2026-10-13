# -*- coding: utf-8 -*-
"""原大CNN仅meta验证；四常量改小后执行原训练/评估函数，核对真正CPU FP16 AMP与跳步。"""
import ast,contextlib,copy,hashlib,inspect,io,json,math,platform,re,warnings
from importlib.metadata import version
from pathlib import Path
import torch
import torchvision
import torch.nn.functional as F
from torch import nn,optim
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(c):return ''.join(c['source'])

def definitions(c):
    tree=ast.parse(source(c))
    return ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])

def training(c,ctx,cpu=False):
    tree=definitions(c)
    if cpu:
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.amp.autocast']
        assert len(calls)==1
        n=calls[0]
        n.func=ast.parse('torch.amp.autocast',mode='eval').body
        n.args=[ast.Constant(value='cpu')]
        n.keywords=[ast.keyword(arg='dtype',value=ast.parse('torch.float16',mode='eval').body)]
        restore=copy.deepcopy(tree)
        call=next(n for n in ast.walk(restore) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.amp.autocast')
        call.func=ast.parse('torch.cuda.amp.autocast',mode='eval').body;call.args=[];call.keywords=[]
        assert ast.dump(restore,include_attributes=False)==ast.dump(definitions(c),include_attributes=False)
    ast.fix_missing_locations(tree)
    exec(compile(tree,'<original-functions-CPU-adaptation>' if cpu else '<original-functions>','exec'),ctx)

def run_train(c,cls,loader,mode):
    torch.manual_seed(123)
    model=cls();criterion=nn.CrossEntropyLoss();optimizer=optim.Adam(model.parameters(),lr=3e-4)
    ctx={'torch':torch,'model':model,'criterion':criterion,'optimizer':optimizer,'train_loader':loader,
      'device':torch.device('cpu'),'num_epochs':1,'tqdm':lambda x:tqdm(x,disable=True)}
    observed=[];losses=[];steps=[]
    h=model.register_forward_hook(lambda m,a,o:observed.append({'shape':list(o.shape),'dtype':str(o.dtype),'requires_grad':o.requires_grad}))
    k=criterion.register_forward_hook(lambda m,a,o:losses.append({'value':o.item(),'dtype':str(o.dtype)}))
    step_hook=optimizer.register_step_post_hook(lambda *args:steps.append(True))
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if mode=='cuda_fallback':ctx['scaler']=torch.cuda.amp.GradScaler()
        if mode=='cpu_amp':ctx['scaler']=torch.amp.GradScaler('cpu',init_scale=128,growth_interval=2)
        training(c,ctx,cpu=mode=='cpu_amp')
        assert ctx['train']() is None
    h.remove();k.remove();step_hook.remove()
    assert len(steps)==2 and len(losses)==2 and all(math.isfinite(x['value']) for x in losses)
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert all(p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in model.parameters())
    expected=torch.float16 if mode=='cpu_amp' else torch.float32
    assert all(o['dtype']==str(expected) and o['shape']==[4,10] for o in observed)
    assert all(l['dtype']=='torch.float32' for l in losses)
    if mode=='cuda_fallback':assert not ctx['scaler'].is_enabled() and ctx['scaler'].get_scale()==1
    if mode=='cpu_amp':assert ctx['scaler'].is_enabled() and ctx['scaler'].get_scale()==256
    # 验证eval/no_grad/准确率定义，并检查原函数无条件切train的副作用。
    model.eval();eval_outputs=[]
    e=model.register_forward_hook(lambda m,a,o:eval_outputs.append(o.requires_grad))
    accuracy=ctx['evalute'](model,loader);e.remove()
    assert model.training and eval_outputs==[False,False]
    with torch.no_grad():
        correct=sum((model(x).argmax(1)==y).sum().item() for x,y in loader)
    assert accuracy.item()==correct/8
    returned={'mode':mode,'updates':2,'losses':losses,'training_outputs':observed,
      'parameters_and_gradients_dtype':'torch.float32','weights_changed':True,'eval_accuracy_on_synthetic_only':accuracy.item(),
      'eval_no_grad':True,'eval_unconditionally_sets_train':True,'warnings':[str(w.message) for w in caught],
      'scaler_enabled':ctx['scaler'].is_enabled() if 'scaler' in ctx else None,
      'scale_after_two_finite_steps':ctx['scaler'].get_scale() if 'scaler' in ctx else None}
    if mode=='cpu_amp':
        scaler=ctx['scaler'];snapshot={n:p.detach().clone() for n,p in model.named_parameters()}
        opt_steps={n:s['step'].item() for n,s in enumerate(optimizer.state.values())}
        optimizer.zero_grad();x,y=next(iter(loader))
        with torch.amp.autocast('cpu',dtype=torch.float16):loss=criterion(model(x),y)
        scaler.scale(loss).backward()
        next(model.parameters()).grad.fill_(float('inf'))
        scaler.step(optimizer);scaler.update()
        assert all(torch.equal(snapshot[n],p) for n,p in model.named_parameters())
        assert opt_steps=={n:s['step'].item() for n,s in enumerate(optimizer.state.values())}
        assert scaler.get_scale()==128
        returned['injected_inf_gradient']={'weights_unchanged':True,'Adam_step_counters_unchanged':True,'scale_backoff_256_to_128':True}
    return returned,{n:p.detach().clone() for n,p in model.named_parameters()},ctx

def verify(path):
    raw=json.loads(path.read_text(encoding='utf-8'));cells=raw['cells'][2::2] if len(raw['cells'])==39 else raw['cells']
    assert len(cells)==19
    ctx={};executed=[]
    for i in [3,4,6,7,10]:
        exec(compile(source(cells[i]),'<original-cell-%d>'%i,'exec'),ctx);executed.append(i)
    assert ctx['device']==torch.device('cpu')
    cls=ctx['CNN']
    # 元数据执行使用原类/forward，无真实大权重或激活存储。
    with torch.device('meta'):
        model=cls();meta_out=model(torch.empty(2,1,28,28))
        assert list(meta_out.shape)==[2,10]
        params=sum(p.numel() for p in model.parameters());assert params==476938250
        wrong=cls(in_channels=3)
        try:wrong(torch.empty(2,3,224,224))
        except RuntimeError as error:
            mismatch=str(error);assert '32112640' in mismatch and '501760' in mismatch
        else:raise AssertionError('RGB224不应适配原fc')
    # 只改四处通道常量，并核对forward完全原样。
    original_tree=ast.parse(source(cells[6]));tree=copy.deepcopy(original_tree);changes=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Constant) and n.value in [5120,10240]:
            old=n.value;n.value=8 if old==5120 else 16;changes.append(old)
    assert len(changes)==4
    restore=copy.deepcopy(tree)
    for n in ast.walk(restore):
        if isinstance(n,ast.Constant) and n.value in [8,16]:n.value=5120 if n.value==8 else 10240
    assert ast.dump(restore,include_attributes=False)==ast.dump(original_tree,include_attributes=False)
    exec(compile(tree,'<explicit-small-channel-CNN>','exec'),ctx);small=ctx['CNN']
    torch.manual_seed(2026)
    # 真实ToTensor处理自编uint8图，构造8样本，不下载或替换MNIST类。
    import numpy as np
    from torchvision.transforms import ToTensor
    x=torch.stack([ToTensor()(((np.arange(784).reshape(28,28)+i*17)%256).astype('uint8')) for i in range(8)])
    y=torch.arange(8,dtype=torch.int64)
    assert x.shape==(8,1,28,28) and x.dtype==torch.float32
    loader=DataLoader(TensorDataset(x,y),batch_size=4,shuffle=False)
    ctx['train_loader']=loader
    with contextlib.redirect_stdout(io.StringIO()) as out:exec(source(cells[12]),ctx)
    executed.append(12)
    assert ctx['batch_x'].shape==(4,1,28,28) and ctx['batch_y'].shape==(4,)
    # torchsummary的实际虚拟batch=2，与报告batch参数区分。
    sample_batches=[];m=small();hook=m.register_forward_hook(lambda m,a,o:sample_batches.append(a[0].shape[0]))
    with contextlib.redirect_stdout(io.StringIO()):ctx['summary'](m,input_size=(1,28,28),batch_size=32,device='cpu')
    hook.remove();assert sample_batches==[2]
    fp32,state_fp,ctx_fp=run_train(cells[14],small,loader,'fp32')
    fallback,state_fb,ctx_fb=run_train(cells[17],small,loader,'cuda_fallback')
    assert all(torch.equal(state_fp[k],state_fb[k]) for k in state_fp)
    active,state_active,ctx_active=run_train(cells[17],small,loader,'cpu_amp')
    # 打印原15/18格，合成loader是显式上下文替代，未冒充MNIST结果。
    for i,scope in [(15,ctx_fp),(18,ctx_fb)]:
        scope['test_loader']=loader
        with contextlib.redirect_stdout(io.StringIO()) as printed:exec(source(cells[i]),scope)
        assert len(printed.getvalue().splitlines())==2;executed.append(i)
    snippet=re.search(r'```[^\n]*\n(.*?)```',source(cells[1]),re.S)[1]
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always');scope={'torch':torch};exec(snippet,scope)
    assert not scope['scaler'].is_enabled()
    uniform=nn.CrossEntropyLoss()(torch.zeros(2,10),torch.tensor([1,3])).item()
    assert abs(uniform-math.log(10))<1e-6
    # 真实autograd验证放大/还原梯度，不只做标量算术。
    p=nn.Parameter(torch.tensor([1.]));opt=optim.SGD([p],lr=.1)
    scaler=torch.amp.GradScaler('cpu',init_scale=128)
    scaler.scale((p*2).sum()).backward();scaled=p.grad.item();assert scaled==256
    scaler.unscale_(opt);unscaled=p.grad.item();assert unscaled==2
    scaler.step(opt);scaler.update();assert abs(p.item()-.8)<1e-6
    return {'notebook_sha256':sha(path),'executed_whole_original_code_cells':executed,
      'excluded_whole_original_cells':[8,11,14,17],'extracted_original_functions':[14,17],
      'original_CNN_meta':{'parameter_count':params,'MNIST_logits_shape':list(meta_out.shape),'RGB224_error':mismatch,
        'real_large_weights_or_activations_allocated':False,'fp32_weights_GiB':params*4/2**30},
      'small_CNN_adaptation':{'four_channel_constants_changed':'5120→8,10240→16','forward_AST_unchanged':True,
        'parameters':sum(p.numel() for p in small().parameters()),'synthetic_samples':8,'batch_size':4,'epochs':1,
        'ToTensor_dtype':str(x.dtype),'MNIST_downloaded':False,'torchsummary_virtual_batch':sample_batches[0]},
      'FP32_original_functions':fp32,'CUDA_AMP_original_functions_CPU_fallback':fallback,
      'FP32_and_disabled_CUDA_AMP_weights_equal':True,'CPU_FP16_active_autocast_supplement':active,
      'CPU_AMP_adaptations':{'autocast_only_expression_changed':"torch.cuda.amp.autocast()→torch.amp.autocast('cpu',dtype=torch.float16)",
        'reverse_AST_check':True,'scaler':'torch.amp.GradScaler(cpu,init_scale128,growth_interval2)',
        'synthetic_data_small_model_epoch_and_batch_are_explicit_alternatives':True},
      'Markdown1_actual_CPU_disabled_scaler':{'enabled':False,'warnings':[str(w.message) for w in warns]},
      'uniform_10class_CE':uniform,'autograd_loss_scaling':{'scaled_grad':scaled,'unscaled_grad':unscaled,'SGD_parameter':p.item()},
      'no_GPU_memory_or_MNIST_accuracy_benchmark':True}

def main():
    assert torch.__version__=='2.5.1+cpu' and not torch.cuda.is_available()
    torch.set_num_threads(2)
    lesson=Path(__file__).resolve().parents[2];directory=lesson/'代码/tutorials'
    original=verify(directory/'07_fp16_mixed_precision_training.ipynb')
    annotated=verify(directory/'代码注释.ipynb')
    for key in original:
        if key!='notebook_sha256':assert original[key]==annotated[key],key
    record={'status':'部分成功','environment':{'OS':platform.platform(),'python':platform.python_version(),
      'torch':torch.__version__,'torchvision':torchvision.__version__,
      'packages':{k:version(k) for k in ['torchsummary','numpy','tqdm','IPython','nbformat']}},
      'original':original,'annotated':annotated,'all_structural_and_numeric_results_equal':True,
      'limits':['No real original large CNN storage, MNIST download/full training or RTX3090 experiment',
        'Original CUDA AMP disabled on CPU, not labelled as active AMP',
        'Supplement uses4channel constants changed,8synthetic samples,batch4/epoch1 andCPU FP16 APIs',
        'Inf gradient deliberately injected to verify step skip; synthetic accuracy not MNIST performance']}
    (lesson/'复现/outputs/training_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Original meta CNN shape/count/error verified; original FP32 and disabled CUDA AMP functions executed.')
    print('Active CPU FP16 autocast/FP32 loss+grad,2Adam updates,scale growth,Inf skip+backoff andevaluation passed in both variants.')

if __name__=='__main__':main()
