# -*- coding: utf-8 -*-
"""PPO arithmetic and response-boundary checks, not an actual PPOTrainer run."""
import argparse
import ast
import hashlib
import json
import math
import pathlib


def gae(rewards, values, gamma=1.0, lam=0.95):
    acc = 0.0
    advantages = [0.0]*len(rewards)
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1<len(values) else 0.0
        delta = rewards[t]+gamma*next_value-values[t]
        acc = delta+gamma*lam*acc
        advantages[t]=acc
    return advantages,[a+v for a,v in zip(advantages,values)]


def policy_loss(ratio, advantage, clip=0.2):
    bounded = min(1+clip,max(1-clip,ratio))
    return max(-advantage*ratio,-advantage*bounded)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tokenizer-json',type=pathlib.Path)
    parser.add_argument('--sources',type=pathlib.Path)
    parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('outputs/教学验证.json'))
    args=parser.parse_args()
    query=[1212,3329,314,1816,284,262,220]
    tokenizer_status='historical ids fixture; tokenizer not requested'
    tokenizer_hash=None
    if args.tokenizer_json:
        from tokenizers import Tokenizer
        tok=Tokenizer.from_str(args.tokenizer_json.read_text(encoding='utf-8'))
        encoded=tok.encode('This morning I went to the ').ids
        assert encoded==query
        assert tok.decode(encoded)=='This morning I went to the '
        tokenizer_status='real tokenizers encoding and roundtrip passed'
        tokenizer_hash=hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest()
    # Twenty integer fixture continuation IDs; no language-model output.
    generated=[100+i for i in range(20)]
    full_response=query+generated
    wrong_input=query+full_response
    correct_input=query+generated
    assert len(wrong_input)==34 and len(correct_input)==27
    start=len(query)-1
    wrong_mask=[int(start<=i<start+len(full_response)) for i in range(len(wrong_input)-1)]
    correct_mask=[int(start<=i<start+len(generated)) for i in range(len(correct_input)-1)]
    assert sum(wrong_mask)==27 and sum(correct_mask)==20
    pi,reference=[0.8,0.2],[0.5,0.5]
    sampled=[math.log(p/q) for p,q in zip(pi,reference)]
    true_kl=sum(p*k for p,k in zip(pi,sampled))
    assert sampled[1]<0 and true_kl>0
    rewards=[-0.2*sampled[0],-0.2*sampled[1]+1.0]
    assert abs(rewards[-1]-1.183258146374831)<1e-12
    advantages,returns=gae([-0.1,0.0,0.9],[0.2,0.3,0.4])
    assert max(abs(a-b) for a,b in zip(advantages,[0.54625,0.575,0.5]))<1e-12
    assert max(abs(a-b) for a,b in zip(returns,[0.74625,0.875,0.9]))<1e-12
    positive_loss=policy_loss(1.5,2.0)
    negative_loss=policy_loss(1.5,-2.0)
    assert positive_loss==-2.4 and negative_loss==3.0
    # Finite difference confirms positive-advantage clipping gives a flat region.
    eps=1e-6
    clipped_derivative=(policy_loss(1.5+eps,2)-policy_loss(1.5-eps,2))/(2*eps)
    assert clipped_derivative==0
    defaults={}
    if args.sources:
        for f in args.sources.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
        tree=ast.parse((args.sources/'ppo-config-v0.4.7.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PPOConfig')
        for n in cls.body:
            if isinstance(n,ast.AnnAssign) and isinstance(n.value,ast.Call):
                for kw in n.value.keywords:
                    if kw.arg=='default':defaults[n.target.id]=ast.literal_eval(kw.value)
        assert defaults['mini_batch_size']==1 and defaults['ppo_epochs']==4
        assert defaults['init_kl_coef']==0.2 and defaults['seed']==0
        source=(args.sources/'ppo-trainer-v0.4.7.py').read_text(encoding='utf-8')
        assert 'input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]' in source
        assert 'start = len(query_batch[j]) - 1' in source
    report={'status':'partial','torch_training':False,'tokenizer':tokenizer_status,'tokenizer_sha256':tokenizer_hash,
            'query_ids':query,'query_length':7,'wrong_response_length':27,'wrong_combined_length':34,
            'correct_response_length':20,'correct_combined_length':27,'wrong_mask_target_count':sum(wrong_mask),
            'correct_mask_target_count':sum(correct_mask),'sampled_log_ratios':sampled,'full_distribution_kl':true_kl,
            'kl_shaped_rewards_with_score_on_last':rewards,'raw_gae_advantages':advantages,'returns':returns,
            'positive_advantage_clipped_loss':positive_loss,'negative_advantage_loss':negative_loss,
            'finite_difference_clipped_derivative':clipped_derivative,'value_head_parameters':768+1,
            'ppo_defaults':defaults,'source_checks':'AST and boundary statements' if args.sources else 'not requested'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
