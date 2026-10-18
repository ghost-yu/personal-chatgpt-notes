"""Execute the six published blocks and compare complete pinned implementations."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get('LESSON27_LOCAL', 'G:/zzz/Personal ChatGPT 系列/27 TRL RLHF PPOTrainer 与 OpenRLHF/资料'))
LIB = Path(os.environ.get('LESSON27_LIBS', 'G:/zzz/Personal ChatGPT 系列/23 TRL Reward Model 与 RewardTrainer/资料/runtime-libs'))
TOKEN = Path(os.environ.get('GPT2_TOKENIZER_JSON', 'G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/gpt2-tokenizer/tokenizer.json'))
sys.path.insert(0, str(LIB))
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', GPT2_TOKENIZER_JSON=str(TOKEN))
import torch
import transformers
import trl
import accelerate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_previous_verifier():
    path = ROOT / '复现/scripts/verify_delivery.py'
    spec = importlib.util.spec_from_file_location('lesson27_previous_verifier', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.torch = torch
    mod.BACK = LOCAL
    mod.LIB = LIB
    mod.TOKEN = TOKEN
    return mod


def main():
    torch.set_num_threads(1)
    assert trl.__version__ == '0.8.6'
    assert Path(trl.__file__).resolve().parent == (LIB / 'trl').resolve()
    previous = load_previous_verifier()
    _, _, static = previous.audit()
    manifest = json.loads((ROOT / '来源清单.json').read_text('utf-8'))
    note = (ROOT / '解释.md').read_text('utf-8').split('<details>', 1)[0]
    blocks = re.findall(r'^```python\n(.*?)^```', note, re.M | re.S)
    assert len(blocks) == 6
    scope = {'__name__': '__foundation_examples__'}
    execution = []
    for i, block in enumerate(blocks, 1):
        output = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(output):
            warnings.simplefilter('always')
            exec(compile(block, f'解释.md:block{i}', 'exec'), scope)
        execution.append({'block': i, 'stdout': output.getvalue(), 'warnings': [str(w.message) for w in caught]})
    trainer, model, reference = (scope[k] for k in ('trainer', 'model', 'reference'))
    query, response, inputs = (scope[k] for k in ('query', 'response', 'inputs'))
    assert query.shape == (7,) and response.shape == (3,)
    assert inputs['input_ids'].shape == (1, 10)
    assert torch.equal(inputs['input_ids'][0], torch.cat([query, response]))
    assert scope['response_mask'].shape == scope['rollout_logp'].shape == scope['rollout_values'].shape == (1, 9)
    assert scope['response_mask'][0].nonzero().flatten().tolist() == [6, 7, 8]
    with torch.no_grad():
        logits, _, values = model(**inputs)
        lp, _, forward_values, mask = trainer.batched_forward_pass(model, [query], [response], inputs)
        rlp, _, _, _ = trainer.batched_forward_pass(reference, [query], [response], inputs)
    manual_lp = logits[:, :-1].log_softmax(-1).gather(-1, inputs['input_ids'][:, 1:, None]).squeeze(-1)
    assert logits.shape == (1, 10, 50257) and values.shape == (1, 10)
    torch.testing.assert_close(lp, manual_lp, rtol=0, atol=0)
    torch.testing.assert_close(forward_values, values[:, :-1], rtol=0, atol=0)
    total_parameters = sum(p.numel() for p in model.parameters())
    head_parameters = sum(p.numel() for p in model.v_head.parameters())
    assert total_parameters == 403465 and head_parameters == 9
    assert len(scope['optimizer_steps']) == 4 and scope['reference_same']
    assert any(k.startswith('v_head.') for k in scope['changed'])
    assert any(not k.startswith('v_head.') for k in scope['changed'])
    assert all(not p.requires_grad for p in reference.parameters())
    opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g['params']}
    assert opt_ids == {id(p) for p in model.parameters() if p.requires_grad}
    assert not opt_ids.intersection(id(p) for p in reference.parameters())
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    valid = mask.bool()
    gap = (lp - rlp)[valid].abs().max().item()
    assert gap > 0
    initial_state_equal = all(torch.equal(v, scope['reference_before'][k]) for k, v in scope['actor_before'].items())
    assert initial_state_equal
    torch.testing.assert_close(scope['rollout_logp'][valid], rlp[valid], rtol=0, atol=0)
    # A newly collected rollout has the current actor as its old policy.
    next_rollout_old = lp.detach().clone()
    assert torch.equal((lp - next_rollout_old).exp(), torch.ones_like(lp))

    old_beta = trainer.kl_ctl.value
    trainer.kl_ctl.value = 0.2
    p = torch.tensor([[-0.2, -1., -0.5]], dtype=torch.float64)
    q = torch.tensor([[-0.4, -0.7, -0.6]], dtype=torch.float64)
    m = torch.ones_like(p)
    shaped, non_score, sample_kl = trainer.compute_rewards(torch.tensor([1.], dtype=torch.float64), p, q, m)
    torch.testing.assert_close(shaped, torch.tensor([scope['rewards']], dtype=torch.float64))
    torch.testing.assert_close(non_score, -0.2 * (p - q))
    assert sample_kl[0, 1] < 0
    # The complete method places the score at the last valid position, not the padded tail.
    pm = torch.tensor([[0., 1., 1., 0.]], dtype=torch.float64)
    pp = torch.tensor([[-0.2, -1., -0.5, -0.3]], dtype=torch.float64)
    qq = torch.tensor([[-0.4, -0.7, -0.6, -0.2]], dtype=torch.float64)
    padded, padded_non, _ = trainer.compute_rewards(torch.tensor([1.], dtype=torch.float64), pp, qq, pm)
    injection = padded - padded_non
    torch.testing.assert_close(injection, torch.tensor([[0., 0., 1., 0.]], dtype=torch.float64))
    trainer.kl_ctl.value = old_beta

    # Independent mathematical branches, including gradients through real TRL loss.
    new_lp = scope['new_logp'].detach().clone().requires_grad_()
    zero = torch.zeros_like(new_lp)
    new_v = torch.tensor([[0.5, -0.5, 0.1, -0.1]], dtype=torch.float64, requires_grad=True)
    a = scope['advantage_tensor']
    old_lp = scope['old_logp_tensor']
    pg, vf, _ = trainer.loss(old_lp, zero, torch.zeros(1, 4, 2, dtype=torch.float64), new_v, new_lp, torch.ones_like(zero), a, zero)
    pg.backward(retain_graph=True)
    vf.backward()
    torch.testing.assert_close(new_lp.grad, torch.tensor([[0., -0.175, 0.325, 0.]], dtype=torch.float64))
    torch.testing.assert_close(new_v.grad, torch.tensor([[0.0125, -0.0125, 0.0025, -0.0025]], dtype=torch.float64))
    torch.testing.assert_close(pg, torch.tensor(0.05, dtype=torch.float64))
    torch.testing.assert_close(vf, torch.tensor(0.0065, dtype=torch.float64))
    # Pessimistic value clipping: old=0, target=1, new=.9 -> .1 * .5 * .64 = .032.
    one = torch.ones(1, 2, dtype=torch.float64)
    z2 = torch.zeros_like(one)
    _, clipped_vf, _ = trainer.loss(z2, z2, torch.zeros(1, 2, 2, dtype=torch.float64), 0.9 * one, z2, one, one, one)
    torch.testing.assert_close(clipped_vf, torch.tensor(0.032, dtype=torch.float64))
    assert not scope['actual_adv'].requires_grad

    objects = previous.load_open_objects()
    open_reward, _ = objects['compute_reward'](torch.tensor([1.], dtype=torch.float64), 0.2, p, q, m)
    torch.testing.assert_close(open_reward, shaped)
    open_adv, open_returns = objects['get_advantages_and_returns'](None, scope['v'], scope['r'], scope['mask'], 1., .95)
    torch.testing.assert_close(open_adv, torch.tensor([scope['raw_advantages']], dtype=torch.float64))
    torch.testing.assert_close(open_returns, scope['actual_returns'])
    open_policy = objects['PolicyLoss'](clip_eps=.2)
    open_value = objects['ValueLoss'](clip_eps=.2)
    torch.testing.assert_close(open_policy(new_lp.detach(), old_lp, a, torch.ones_like(a)), pg.detach())
    torch.testing.assert_close(open_value(new_v.detach(), zero, zero, torch.ones_like(a)), vf.detach() / .1)
    twenty = torch.tensor([20.], dtype=torch.float64)
    old_beta = trainer.kl_ctl.value
    trainer.kl_ctl.value = .2
    trl_twenty, *_ = trainer.compute_rewards(twenty, p, q, m)
    open_twenty, _ = objects['compute_reward'](twenty, .2, p, q, m)
    torch.testing.assert_close((trl_twenty - open_twenty)[0, -1], torch.tensor(10., dtype=torch.float64))
    trainer.kl_ctl.value = old_beta
    lengths = torch.tensor([[1., 1.], [1., 0.]], dtype=torch.float64)
    aa = torch.tensor([[-1., -3.], [-9., 0.]], dtype=torch.float64)
    zz = torch.zeros_like(aa)
    trl_reduction, *_ = trainer.loss(zz, zz, torch.zeros(2, 2, 2, dtype=torch.float64), zz, zz, lengths, aa, zz)
    open_reduction = open_policy(zz, zz, aa, lengths)
    torch.testing.assert_close(trl_reduction, torch.tensor(13/3, dtype=torch.float64))
    torch.testing.assert_close(open_reduction, torch.tensor(5.5, dtype=torch.float64))
    from trl.trainer.utils import AdaptiveKLController
    ctl = AdaptiveKLController(.2, 6, 10000)
    ctl.update(12, 100)
    assert abs(ctl.value - .2004) < 1e-12
    figures = json.loads((ROOT / '复现/outputs/论文截图来源.json').read_text('utf-8'))
    for item in figures:
        assert sha(LOCAL / item['paper']) == item['pdf_sha256']
        assert sha(ROOT / item['published_path']) == item['png_sha256']
        assert '](' + item['published_path'] + ')' in note
    result = {
        'status': 'passed', 'date': '2026-10-18',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'trl': trl.__version__, 'accelerate': accelerate.__version__, 'device': str(trainer.accelerator.device)},
        'note_sha256': sha(ROOT / '解释.md'), 'script_sha256': sha(__file__),
        'tokenizer_sha256': sha(TOKEN), 'trl_sources_sha256': manifest['trl_source_sha256'],
        'openrlhf_sources_sha256': manifest['openrlhf_source_sha256'],
        'source_fidelity': static, 'published_blocks': execution,
        'model': {'parameters': total_parameters, 'value_head_parameters': head_parameters,
                  'query_ids': query.tolist(), 'response_ids': response.tolist(),
                  'input_shape': list(inputs['input_ids'].shape), 'logits_shape': list(logits.shape),
                  'shift_shape': list(lp.shape), 'mask_indices': [6, 7, 8], 'no_duplicate_query': True,
                  'manual_logprob_exact': True, 'value_shift_exact': True,
                  'optimizer_updates': len(scope['optimizer_steps']), 'changed_parameters': scope['changed'],
                  'reference_unchanged': scope['reference_same'], 'initial_actor_reference_equal': initial_state_equal,
                  'optimizer_only_actor': True, 'next_rollout_ratio_one': True,
                  'current_reference_logprob_max_difference': gap, 'current_rollout_probability_ratios': scope['new_old_ratio'].tolist()},
        'mathematics': {'sample_log_ratios': sample_kl.tolist(), 'shaped_reward': shaped.tolist(),
                        'score_last_valid_once': injection.tolist(), 'raw_advantages': scope['raw_advantages'],
                        'returns': scope['actual_returns'].tolist(), 'whitened_advantages': scope['actual_adv'].tolist(),
                        'policy_loss': pg.item(), 'policy_logprob_gradient': new_lp.grad.tolist(),
                        'weighted_value_loss': vf.item(), 'value_gradient': new_v.grad.tolist(),
                        'pessimistic_value_clip_loss': clipped_vf.item(), 'adaptive_beta': ctl.value},
        'openrlhf': {'complete_source_math_objects': list(objects.keys() - {'torch', 'nn', 'F', 'Optional', 'Tuple', 'Union'}),
                    'reward_gae_equal': True, 'single_sequence_policy_value_equal': True,
                    'score20_clipped_to10': True, 'TRL_token_mean': trl_reduction.item(),
                    'OpenRLHF_sequence_mean': open_reduction.item(), 'distributed_training_run': False},
        'paper_crops_verified': len(figures),
        'scope': 'Random CPU GPT2 with real GPT2 tokenizer and complete TRL 0.8.6; complete pinned OpenRLHF mathematical objects. Original pretrained GPT2, distributed OpenRLHF and paper task metrics not run.'
    }
    path = ROOT / '复现/outputs/基础讲解验证.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps({'status': 'passed', 'blocks': len(blocks), 'parameters': total_parameters, 'optimizer_steps': len(scope['optimizer_steps']), 'query_ids': query.tolist(), 'response_ids': response.tolist(), 'reference_gap': gap, 'paper_crops': len(figures)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
