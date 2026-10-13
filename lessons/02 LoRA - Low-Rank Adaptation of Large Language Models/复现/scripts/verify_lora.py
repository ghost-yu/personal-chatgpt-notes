"""新编CPU LoRA最小示例：矩阵/梯度/冻结与小型更新核对，不加载BLOOM。"""
import argparse
import json
import math
import platform
from pathlib import Path
import torch
from torch import nn


def forward(x, weight, factor_a, factor_b, scale):
    # 行向量约定：[N,k]@[k,d]，LoRA支路先降维到r再升回d。
    return x @ weight + scale * ((x @ factor_a) @ factor_b)


def verify():
    torch.manual_seed(2026)
    dtype = torch.float64
    # 可手算的例子：降维结果为5，总输出应为[16,22]。
    x = torch.tensor([[1.,2.]],dtype=dtype)
    w = torch.eye(2,dtype=dtype)
    a = torch.tensor([[1.],[2.]],dtype=dtype)
    b = torch.tensor([[3.,4.]],dtype=dtype)
    output = forward(x,w,a,b,1.)
    torch.testing.assert_close(output,torch.tensor([[16.,22.]],dtype=dtype))
    # 多维输入最后一维是k；与显式增量、合并及nn.Linear方向比较。
    x3 = torch.randn(2,3,4,dtype=dtype)
    w3 = torch.randn(4,5,dtype=dtype)
    a3 = torch.randn(4,2,dtype=dtype)
    b3 = torch.randn(2,5,dtype=dtype)
    scale = 4./2
    factored = forward(x3,w3,a3,b3,scale)
    explicit = x3@w3 + scale*(x3@(a3@b3))
    merged = x3@(w3+scale*(a3@b3))
    linear = torch.nn.functional.linear(x3,w3.T) + scale*torch.nn.functional.linear(
        torch.nn.functional.linear(x3,a3.T),b3.T)
    for other in [explicit,merged,linear]:
        torch.testing.assert_close(factored,other,rtol=1e-12,atol=1e-12)
    assert factored.shape == (2,3,5)
    # 直接与手推矩阵梯度比较；G是输出处给定的上游梯度。
    xa = torch.randn(3,4,dtype=dtype)
    aa = nn.Parameter(torch.randn(4,2,dtype=dtype))
    bb = nn.Parameter(torch.randn(2,5,dtype=dtype))
    g = torch.randn(3,5,dtype=dtype)
    h = forward(xa,w3,aa,bb,scale)
    (h*g).sum().backward()
    expected_a=scale*xa.T@g@bb.detach().T
    expected_b=scale*(xa@aa.detach()).T@g
    torch.testing.assert_close(aa.grad,expected_a,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(bb.grad,expected_b,rtol=1e-12,atol=1e-12)
    # 随机A+零B初始化：初始输出等于基分支，A先为零梯度、B可更新。
    base=nn.Parameter(torch.eye(2,dtype=dtype),requires_grad=False)
    first=nn.Parameter(torch.empty(2,1,dtype=dtype))
    second=nn.Parameter(torch.zeros(1,2,dtype=dtype))
    nn.init.kaiming_uniform_(first,a=math.sqrt(5))
    base_snapshot=base.detach().clone()
    target=torch.tensor([[3.,4.]],dtype=dtype)
    optimizer=torch.optim.SGD([first,second],lr=.05)
    prediction=forward(x,base,first,second,1.)
    torch.testing.assert_close(prediction,x@base,rtol=0,atol=0)
    initial_loss=((prediction-target)**2).mean()
    initial_loss.backward()
    first_grad=float(first.grad.norm()); second_grad=float(second.grad.norm())
    assert first_grad==0 and second_grad>0 and base.grad is None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    next_loss=((forward(x,base,first,second,1.)-target)**2).mean()
    next_loss.backward()
    later_first_grad=float(first.grad.norm())
    assert later_first_grad>0 and base.grad is None
    # 做20步教学更新，确认冻结权重不变；不据此推断真实任务能力。
    for _ in range(20):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss=((forward(x,base,first,second,1.)-target)**2).mean()
        loss.backward()
    final_loss=float(loss.detach())
    assert final_loss<float(initial_loss.detach())
    assert torch.equal(base,base_snapshot) and base.grad is None
    # 两因子均为零的失败边界：两边初始梯度都为零。
    zero_a=nn.Parameter(torch.zeros(2,1,dtype=dtype))
    zero_b=nn.Parameter(torch.zeros(1,2,dtype=dtype))
    ((forward(x,base,zero_a,zero_b,1.)-target)**2).mean().backward()
    assert zero_a.grad.norm()==0 and zero_b.grad.norm()==0
    return {'status':'成功（教学验证）','python':platform.python_version(),'torch':torch.__version__,
            'device':'cpu','dtype':'float64','numeric_output':output.tolist(),
            'batch_shape':list(factored.shape),'max_merge_error':float((merged-factored).abs().max()),
            'analytic_gradient_max_error':float(max((aa.grad-expected_a).abs().max(),(bb.grad-expected_b).abs().max())),
            'initial_factor_a_grad_norm':first_grad,'initial_factor_b_grad_norm':second_grad,
            'factor_a_grad_after_first_update':later_first_grad,'base_unchanged_and_no_grad':True,
            'initial_mse':float(initial_loss.detach()),'final_mse':final_loss,
            'both_zero_init_gradients_zero':True,
            'parameter_examples':[{'k':k,'d':d,'r':r,'base':k*d,'lora':r*(k+d),'ratio':r*(k+d)/(k*d)}
                                  for k,d,r in [(100,500,5),(768,768,8),(4096,12288,16)]],
            'limits':'自编小型线性层验证，非原伪代码执行，非PEFT/BLOOM或论文任务复现'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',required=True)
    args=parser.parse_args()
    result=verify()
    Path(args.record).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))
