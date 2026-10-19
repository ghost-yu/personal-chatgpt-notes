"""Execute all published examples, compare real score pooling, losses and gradients."""
from pathlib import Path
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import platform
import re
import sys
import tempfile
import warnings

import torch
import torch.nn.functional as F
import transformers

ROOT = Path(__file__).resolve().parents[2]
MATERIAL = Path('G:/zzz/Personal ChatGPT 系列/30 InstructGPT reward modeling/资料')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('lecture30_previous', ROOT / '复现/scripts/verify_delivery.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    notebooks, core = previous.audit()
    note = (ROOT / '解释.md').read_text('utf-8').split('<details>', 1)[0]
    blocks = re.findall(r'^```python\n(.*?)^```', note, re.M | re.S)
    assert len(blocks) == 6
    assert previous.tree(blocks[2]) == previous.tree(previous.source(notebooks[previous.MAIN][0]['cells'][23]))
    assert previous.tree(blocks[5]) == previous.tree(previous.source(notebooks[previous.SUP][0]['cells'][2]))
    scope = {'__name__': '__foundation_examples__'}
    executions = []
    for i, block in enumerate(blocks, 1):
        out = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out):
            warnings.simplefilter('always')
            exec(compile(block, f'解释.md:block{i}', 'exec'), scope)
        executions.append({'block': i, 'stdout': out.getvalue(), 'warnings': [str(x.message) for x in caught]})
        if i == 1:
            assert math.isclose(scope['probability'].item(), 1/(1+math.exp(-1)), abs_tol=1e-15)
            assert math.isclose(scope['pair_loss'].item(), math.log1p(math.exp(-1)), abs_tol=1e-15)
            assert math.isclose(scope['rw'].grad.item(), -1/(1+math.exp(1)), abs_tol=1e-15)
            assert scope['rw'].grad.item() == -scope['rl'].grad.item()
        if i == 3:
            np, plt = scope['np'], scope['plt']
            np.testing.assert_allclose(scope['neg_log_sigma'], np.logaddexp(0, -scope['delta']), atol=1e-11)
            with tempfile.TemporaryDirectory(prefix='lesson30_plot_') as tmp:
                rendered = Path(tmp) / 'plot.png'
                plt.gcf().savefig(rendered, dpi=150, bbox_inches='tight')
                assert sha(rendered) == sha(ROOT / '图片/pair_loss_gradient.png')
            plt.close('all')
        if i == 5:
            # Keep the new neural trace before the original team cell reuses generic names.
            assert sum(p.numel() for p in scope['reward_model'].parameters()) == 2880
    assert math.isclose(scope['equal_scores_base10'] * math.log(10), math.log(2), abs_tol=1e-15)
    torch.testing.assert_close(scope['stable_losses'], scope['bce_losses'], atol=1e-15, rtol=0)
    assert torch.isfinite(scope['stable_losses']).all()
    assert len(scope['make_pairs']([0, 0, 1, 2])) == 5
    assert len(scope['make_pairs'](list(range(9)))) == 36
    demo = previous.load_module('lecture30_demo_foundation', previous.CODE / '偏好损失_教学补充.py')
    previous.loss_checks(demo)
    ns, traces, _, team = previous.execute(notebooks[previous.SUP][0], previous.SUP)
    previous.scipy_checks({previous.SUP: (ns, team)})
    previous.np.testing.assert_allclose(scope['theta_full'], team['theta'], atol=1e-15)
    assert scope['result'].success and ns['result'].success
    model, ids, mask = (scope[n] for n in ['reward_model', 'input_ids', 'attention_mask'])
    groups, ranks = scope['groups'], scope['ranks']
    assert scope['final_loss'] < scope['training_trace'][0]
    assert len(scope['training_trace']) == 5
    assert sorted({int(s['step']) for s in scope['optimizer'].state.values()}) == [5]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    initial, _, _, _, _ = demo.make_experiment()
    initial_state = initial.state_dict()
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, initial_state[k])]
    assert 'score.weight' in changed and any('self_attn' in k for k in changed)
    with torch.no_grad():
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        assert list(hidden.shape) == [7, 6, 16]
        last = mask.sum(-1) - 1
        manual_scores = (hidden[torch.arange(7), last] * model.score.weight[0]).sum(-1)
        torch.testing.assert_close(manual_scores, scope['final_rewards'], atol=1e-7, rtol=0)
        padded_scores = model(input_ids=F.pad(ids, (0, 2)), attention_mask=F.pad(mask, (0, 2)), use_cache=False).logits.squeeze(-1)
        torch.testing.assert_close(padded_scores, scope['final_rewards'], atol=1e-7, rtol=0)
    params = tuple(model.parameters())
    rewards = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.squeeze(-1)
    reused_loss = scope['prompt_mean_loss'](rewards, groups, ranks)
    reused_grad = torch.autograd.grad(reused_loss, params)
    ix = [start + i for (start, _), rank in zip(groups, ranks) for pair in scope['make_pairs'](rank) for i in pair]
    pair_scores = model(input_ids=ids[ix], attention_mask=mask[ix], use_cache=False).logits.reshape(-1, 2)
    pair_bce = F.binary_cross_entropy_with_logits(pair_scores[:, 0] - pair_scores[:, 1], torch.ones(8), reduction='none')
    independent_loss = (pair_bce[:5].mean() + pair_bce[5:].mean()) / 2
    independent_grad = torch.autograd.grad(independent_loss, params)
    gradient_error = max((a-b).abs().max().item() for a, b in zip(reused_grad, independent_grad))
    assert abs(reused_loss.item() - independent_loss.item()) < 1e-7 and gradient_error < 2e-6
    complete = core.LlamaForSequenceClassification(model.config).eval()
    complete.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.equal(complete(input_ids=ids, attention_mask=mask, use_cache=False).logits.squeeze(-1), scope['final_rewards'])
    with tempfile.TemporaryDirectory(prefix='lesson30_save_') as tmp:
        model.save_pretrained(tmp)
        reloaded = transformers.LlamaForSequenceClassification.from_pretrained(tmp, local_files_only=True).eval()
        assert previous.same(model.state_dict(), reloaded.state_dict())
        with torch.no_grad():
            assert torch.equal(reloaded(input_ids=ids, attention_mask=mask, use_cache=False).logits.squeeze(-1), scope['final_rewards'])
    crops = json.loads((ROOT / '复现/outputs/论文截图来源.json').read_text('utf-8'))
    for item in crops:
        assert sha(MATERIAL / item['paper']) == item['pdf_sha256']
        assert sha(ROOT / item['published_path']) == item['png_sha256']
        assert item['published_path'] in note
    report = {
        'status': 'passed', 'environment': {'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'transformers': transformers.__version__, 'scipy': previous.scipy.__version__, 'threads': torch.get_num_threads()},
        'preservation': previous.REPORT['preservation'], 'executions': executions,
        'original_plot_and_team_cell_AST_equal': True, 'original_plot_png_equal': True,
        'loss_math': previous.REPORT['loss_math'], 'scipy_likelihood': previous.REPORT['scipy_likelihood'],
        'original_supplement_execution': traces,
        'new_neural': {'parameters': 2880, 'ids_shape': list(ids.shape), 'hidden_shape': [7, 6, 16],
            'head_shape': list(model.score.weight.shape), 'updates': 5, 'trace_before_updates': scope['training_trace'],
            'loss_after_last_update': scope['final_loss'], 'rewards': scope['final_rewards'].tolist(),
            'changed_state_entries': changed, 'all_parameter_gradients_finite': True,
            'manual_last_valid_pooling_equal': True, 'extra_right_padding_invariant': True,
            'unique_candidate_rows': 7, 'duplicated_pair_rows': len(ix), 'reused_loss': reused_loss.item(),
            'independent_pair_BCE_loss': independent_loss.item(), 'all_parameter_gradient_max_error': gradient_error,
            'complete_exported_class_outputs_equal': True, 'save_reload_states_and_scores_equal': True},
        'paper_crops': crops, 'note_sha256': sha(ROOT / '解释.md'), 'script_sha256': sha(__file__),
        'not_run': ['Original GPT-3/InstructGPT 6B weights and human preference training data', 'System quality or generalization assessment', 'PPO or GPU measurements'],
    }
    (ROOT / '复现/outputs/基础讲解验证.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
    print(json.dumps({'status': report['status'], 'blocks': len(blocks), 'new_neural': report['new_neural'],
                      'scipy_likelihood': report['scipy_likelihood']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
