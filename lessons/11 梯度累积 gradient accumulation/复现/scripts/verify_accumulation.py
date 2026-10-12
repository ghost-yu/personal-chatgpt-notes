# -*- coding: utf-8 -*-
"""Exact arithmetic gradient checks. No torch/autograd/model/GPU run."""
import json
import platform
from fractions import Fraction as F
from pathlib import Path

def loss(points,w,b):
    return sum((w*x+b-y)**2 for x,y in points)/len(points)

def gradient(points,w=F(1),b=F(0)):
    return (sum(2*(w*x+b-y)*x for x,y in points)/len(points),
            sum(2*(w*x+b-y) for x,y in points)/len(points))

def add(a,b):return tuple(x+y for x,y in zip(a,b))
def scale(a,s):return tuple(x*s for x in a)
def numbers(a):return [float(x) for x in a]

def main():
    points=[(F(i),F(0)) for i in [1,2,3,4]]
    big=gradient(points)
    micro=[gradient([p]) for p in points]
    accum=scale(tuple(map(sum,zip(*micro))),F(1,4))
    assert accum==big==(F(15),F(5))
    # Independent central finite difference exactly matches quadratic derivatives.
    eps=F(1,1000)
    finite=((loss(points,F(1)+eps,F(0))-loss(points,F(1)-eps,F(0)))/(2*eps),
            (loss(points,F(1),eps)-loss(points,F(1),-eps))/(2*eps))
    assert finite==big
    # Step only after the whole window gives the same SGD update.
    correct_params=add((F(1),F(0)),scale(big,F(-1,10)))
    sequential=(F(1),F(0))
    for point in points:
        sequential=add(sequential,scale(gradient([point],*sequential),F(-1,10)))
    assert sequential!=correct_params
    tail=gradient(points[:2])
    original_tail=scale(add(micro[0],micro[1]),F(1,4))
    assert original_tail==scale(tail,F(1,2))
    unequal=[points[:2],[(F(10),F(0))]]
    unequal_big=gradient(unequal[0]+unequal[1])
    means=[gradient(p) for p in unequal]
    naive=scale(add(*means),F(1,2))
    weighted=add(scale(means[0],F(2,3)),scale(means[1],F(1,3)))
    assert weighted==unequal_big and naive!=unequal_big
    # Ten equal micro-batches: three windows of lengths 4,4,2.
    boundaries=[i for i in range(1,11) if i%4==0 or i==10]
    assert boundaries==[4,8,10]
    raw=[2,2];cleared=[];g=0
    for value in raw:g+=value;cleared.append(g)
    assert cleared==[2,4]
    output={'status':'passed','scope':'Fraction arithmetic and analytic/finite-difference gradients; not PyTorch backward or BERT training',
      'python':platform.python_version(),'big_batch_gradient':numbers(big),'micro_accumulated_gradient':numbers(accum),
      'central_difference_gradient':numbers(finite),'one_update_parameters':numbers(correct_params),'step_each_micro_parameters':numbers(sequential),
      'tail_actual_gradient':numbers(tail),'tail_divide_by_4_gradient':numbers(original_tail),'tail_ratio':0.5,
      'unequal_big_gradient':numbers(unequal_big),'unequal_average_of_means':numbers(naive),'unequal_weighted_gradient':numbers(weighted),
      'ten_micro_batch_update_boundaries':boundaries,'two_backward_buffer_additions':cleared,
      'historical_metrics':{'memory_drop_mib':21296-13166,'memory_drop_percent':100*(21296-13166)/21296,'time_ratio':102.8176/38.0194,'throughput_drop_percent':100*(26.934-9.959)/26.934}}
    out=Path(__file__).resolve().parents[1]/'outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'教学验证.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
