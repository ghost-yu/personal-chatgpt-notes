"""Execute the five published examples and check actual fixed PEFT merge paths."""
from pathlib import Path
import ast
import base64
import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer, Embedding
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_sources():
    assert peft.__version__ == '0.12.0'
    assert transformers.__version__ == '4.44.2'
    manifest = json.loads((ROOT / '来源清单.json').read_text(encoding='utf-8'))
    package = Path(peft.__file__).parent
    dist = importlib.metadata.distribution('peft')
    rows = list(csv.reader(io.StringIO(dist.read_text('RECORD'))))
    record = {r[0]: r[1] for r in rows}
    for rel, expected in manifest['peft_installed_sources_sha256'].items():
        p = package / rel
        assert sha(p) == expected, rel
        encoded = base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')
        assert record['peft/' + rel] == 'sha256=' + encoded, rel
    for item in manifest['files']:
        assert sha(ROOT / item['published_path']) == item['sha256']
    return manifest['peft_installed_sources_sha256']


def check_main(scope, blocks):
    # Compare the entire class and configuration, not merely dimensions or names.
    nb = json.loads((ROOT / '代码/tutorials/peft/lora_inject.ipynb').read_text(encoding='utf-8'))
    original_class = ast.parse(''.join(nb['cells'][2]['source'])).body[0]
    current_class = next(x for x in ast.parse(blocks[2]).body if isinstance(x, ast.ClassDef))
    assert ast.dump(original_class) == ast.dump(current_class)
    original_config = ast.parse(''.join(nb['cells'][3]['source'])).body[0]
    current_config = next(x for x in ast.parse(blocks[2]).body
                          if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'lora_config' for t in x.targets))
    assert ast.dump(original_config) == ast.dump(current_config)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    close = torch.testing.assert_close
    close(scope['base_output'], t([[5, 11]]))
    close(scope['delta1'], t([[1, 0], [.5, 0]]))
    close(scope['M'], t([[1, .5], [.5, 1]]))
    close(scope['S'], t([1.5, .5]))
    close(scope['M_rank1'], torch.full((2, 2), .75, dtype=torch.float64))
    assert abs(scope['error'].item() - .5) < 1e-14
    layer = scope['layer']
    close(layer.get_delta_weight('cat_exact'), scope['M'])
    close(layer.get_delta_weight('svd_old'), 2 * scope['M_rank1'])
    close(layer.get_delta_weight('svd_once'), scope['M_rank1'])
    expected = {'left': [[6, 11.5]], 'cat_exact': [[7, 13.5]],
                'svd_old': [[9.5, 15.5]], 'svd_once': [[7.25, 13.25]]}
    for name, values in expected.items():
        close(scope['outputs'][name], t(values))
    assert scope['trainable_count'] == 128000
    dummy = scope['dummy']
    trained = {n for n, p in dummy.named_parameters() if p.requires_grad}
    assert trained == {'linear.lora_A.default.weight', 'linear.lora_B.default.weight'}
    assert torch.count_nonzero(dummy.linear.lora_B['default'].weight) == 0
    dummy.zero_grad(set_to_none=True)
    loss = dummy(scope['ids']).square().mean()
    loss.backward()
    assert torch.count_nonzero(dummy.linear.lora_A['default'].weight.grad) == 0
    assert torch.count_nonzero(dummy.linear.lora_B['default'].weight.grad) > 0
    assert all(p.grad is None for n, p in dummy.named_parameters() if n not in trained)

    # Independent outer-product reconstruction of the numerical SVD example.
    q1 = t([1, 1]) / 2**.5
    q2 = t([1, -1]) / 2**.5
    close(1.5 * torch.outer(q1, q1) + .5 * torch.outer(q2, q2), scope['M'])
    close(1.5 * torch.outer(q1, q1), scope['M_rank1'])
    assert list(scope['vh_full'].shape) == [5, 5]
    assert list(scope['vh_reduced'].shape) == [2, 5]
    return {'dummy_class_and_config_ast_match_original': True,
            'dummy_trainable_parameters': 128000, 'zero_B_initial_output_preserved': True,
            'first_A_gradient_zero_B_gradient_nonzero_base_gradients_none': True,
            'singular_values': scope['S'].tolist(), 'rank1_error': scope['error'].item(),
            'actual_peft_matrix_outputs': {n: v.tolist() for n, v in scope['outputs'].items()},
            'folded_weight_error': float(torch.dist(scope['plain_model'].proj.weight, scope['W0'] + scope['M_rank1'])),
            'source_deltas_unchanged_during_switch_and_fold': True}


def extra_matrix_checks(scope):
    mix = copy.deepcopy(scope['mix'])
    layer = mix.base_model.model.proj
    source = {n: p.detach().clone() for n, p in mix.named_parameters()}
    mix.add_weighted_adapter(['left', 'right'], [1., 1.], 'linear_mix', combination_type='linear')
    # Independent formula with cross terms in this fixed version.
    a = 2**.5 * (scope['A1'] + scope['A2'])
    b = 2**.5 * (scope['B1'] + scope['B2'])
    torch.testing.assert_close(layer.get_delta_weight('linear_mix'), b @ a)
    cross_error = torch.dist(layer.get_delta_weight('linear_mix'), scope['M']).item()
    assert cross_error > .5
    mix.add_weighted_adapter(['left', 'right'], [1., -.3], 'negative_cat', combination_type='cat')
    negative = scope['delta1'] - .3 * scope['delta2']
    torch.testing.assert_close(layer.get_delta_weight('negative_cat'), negative)
    failed = False
    try:
        mix.add_weighted_adapter(['left', 'right'], [1., -.3], 'negative_linear', combination_type='linear')
    except ValueError as exc:
        assert 'math domain' in str(exc)
        failed = True
    assert failed
    for name, p in mix.named_parameters():
        if name in source:
            assert torch.equal(p, source[name]), name

    # Unequal source ranks and unequal scalings: cat preserves each source's delta.
    cfg = LoraConfig(r=2, lora_alpha=6, lora_dropout=0, target_modules=['proj'])
    mix.add_adapter('rank2', cfg)
    with torch.no_grad():
        layer.lora_A['rank2'].weight.copy_(torch.tensor([[.2, .3], [-.4, .1]], dtype=torch.float64))
        layer.lora_B['rank2'].weight.copy_(torch.tensor([[.1, .5], [.2, -.3]], dtype=torch.float64))
    mix.add_weighted_adapter(['left', 'rank2'], [.4, -.7], 'unequal_cat', combination_type='cat')
    target = .4 * layer.get_delta_weight('left') - .7 * layer.get_delta_weight('rank2')
    torch.testing.assert_close(layer.get_delta_weight('unequal_cat'), target)
    assert mix.peft_config['unequal_cat'].r == 3
    mix.add_weighted_adapter(['left', 'rank2'], [.4, .7], 'unequal_svd', combination_type='svd',
                             svd_rank=2, svd_full_matrices=False)
    target_once = .4 * layer.get_delta_weight('left') + .7 * layer.get_delta_weight('rank2')
    target_twice = .4 * 2 * layer.get_delta_weight('left') + .7 * 3 * layer.get_delta_weight('rank2')
    torch.testing.assert_close(layer.get_delta_weight('unequal_svd'), target_twice)
    assert not torch.allclose(target_twice, 2 * target_once)
    return {'linear_cross_term_error': cross_error, 'linear_negative_weight_math_domain_error': failed,
            'cat_negative_weight_exact': True, 'cat_unequal_rank_and_scaling_exact': True,
            'unequal_svd_reconstructs_sourcewise_extra_scaling_not_global_factor': True}


def tiny_llama_checks():
    torch.manual_seed(240)
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=32, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                         attention_dropout=0., tie_word_embeddings=False)
    base = LlamaForCausalLM(config).eval()
    old_embedding = base.model.embed_tokens.weight.detach().clone()
    old_head = base.lm_head.weight.detach().clone()
    base.resize_token_embeddings(21)
    assert base.config.vocab_size == 21
    assert torch.equal(base.model.embed_tokens.weight[:16], old_embedding)
    assert torch.equal(base.lm_head.weight[:16], old_head)
    base_parameters = sum(p.numel() for p in base.parameters())
    base_state = {n: p.detach().clone() for n, p in base.named_parameters()}
    projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    all_targets = projections + ['embed_tokens', 'lm_head']
    def cfg(targets, alpha=4):
        return LoraConfig(r=2, lora_alpha=alpha, lora_dropout=.1,
                          bias='none', task_type='CAUSAL_LM', target_modules=targets)
    model = get_peft_model(base, cfg(all_targets), adapter_name='norobots')
    model.add_adapter('adcopy', cfg(all_targets))
    model.add_adapter('sql', cfg(projections))
    generator = torch.Generator().manual_seed(241)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_' in n:
                p.copy_(.05 * torch.randn(p.shape, generator=generator, dtype=p.dtype))
    names = ['norobots', 'adcopy', 'sql']
    coeffs = [2., .3, .7]
    source = {n: p.detach().clone() for n, p in model.named_parameters()}
    active_before = model.active_adapter
    assert active_before == 'norobots'
    model.add_weighted_adapter(names, coeffs, 'cat_exact', combination_type='cat')
    model.add_weighted_adapter(names, coeffs, 'svd_old', combination_type='svd',
                               svd_rank=2, svd_full_matrices=False, density=.2)
    model.add_weighted_adapter(names, coeffs, 'density_other', combination_type='svd',
                               svd_rank=2, svd_full_matrices=False, density=.9)
    model.add_adapter('svd_once', cfg(all_targets, alpha=2))
    assert model.active_adapter == active_before
    rows = []
    for path, layer in model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        valid = [(n, c) for n, c in zip(names, coeffs) if n in layer.scaling]
        assert len(valid) == (2 if isinstance(layer, Embedding) or path.endswith('lm_head') else 3)
        intended = sum(c * layer.get_delta_weight(n).detach() for n, c in valid)
        cat_error = torch.dist(layer.get_delta_weight('cat_exact'), intended).item()
        torch.testing.assert_close(layer.get_delta_weight('cat_exact'), intended, rtol=1e-5, atol=1e-7)
        # get_delta_weight returns the embedding in [V,H]; factorization needs [H,V].
        matrix = intended.T if isinstance(layer, Embedding) else intended
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        left = u[:, :2] * s[:2]
        right = vh[:2, :]
        with torch.no_grad():
            if isinstance(layer, Embedding):
                layer.lora_embedding_A['svd_once'].copy_(right)
                layer.lora_embedding_B['svd_once'].copy_(left)
            else:
                layer.lora_A['svd_once'].weight.copy_(right)
                layer.lora_B['svd_once'].weight.copy_(left)
        corrected = (left @ right).T if isinstance(layer, Embedding) else left @ right
        got = layer.get_delta_weight('svd_once')
        torch.testing.assert_close(got, corrected, rtol=1e-5, atol=1e-7)
        error = torch.dist(got, intended).item()
        tail = s[2:].square().sum().sqrt().item()
        assert abs(error - tail) < 2e-7
        torch.testing.assert_close(layer.get_delta_weight('svd_old'), 2 * corrected, rtol=1e-4, atol=1e-7)
        torch.testing.assert_close(layer.get_delta_weight('density_other'), layer.get_delta_weight('svd_old'))
        rows.append({'layer': path, 'embedding': isinstance(layer, Embedding),
                     'source_adapters': [n for n, c in valid], 'delta_shape': list(intended.shape),
                     'cat_error': cat_error, 'svd_error': error, 'singular_value_tail': tail})
    assert len(rows) == 9
    model.eval()
    ids = torch.tensor([[1, 4, 16, 20]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        old_logits = model(ids, attention_mask=mask).logits.detach().clone()
        model.set_adapter('svd_once')
        selected_logits = model(ids, attention_mask=mask).logits.detach().clone()
        generated = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
    active_error = (old_logits - selected_logits).abs().max().item()
    assert active_error > 1e-6
    for n, p in model.named_parameters():
        if n in source:
            assert torch.equal(p, source[n]), n
    # Independent pooling of the embedding delta checks orientation via actual ID lookup.
    embedding = model.base_model.model.model.embed_tokens
    with torch.no_grad():
        expected_embed = torch.nn.functional.embedding(ids, embedding.base_layer.weight + embedding.get_delta_weight('svd_once'))
        actual_embed = embedding(ids)
    torch.testing.assert_close(actual_embed, expected_embed, rtol=1e-5, atol=1e-7)
    folded = copy.deepcopy(model).merge_and_unload(safe_merge=True)
    assert isinstance(folded, LlamaForCausalLM)
    assert not any(isinstance(m, LoraLayer) for m in folded.modules())
    with torch.no_grad():
        folded_logits = folded(ids, attention_mask=mask).logits
        folded_generated = folded.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
    fold_error = (folded_logits - selected_logits).abs().max().item()
    torch.testing.assert_close(folded_logits, selected_logits, rtol=1e-4, atol=2e-7)
    assert torch.equal(generated, folded_generated)
    for n, p in base.named_parameters():
        # The still-wrapped original base has retained every base weight value.
        if '.base_layer.' in n:
            original_name = n.replace('.base_layer.', '.')
        elif 'lora_' in n:
            continue
        else:
            original_name = n
        assert torch.equal(p, base_state[original_name]), n
    return {'config': {'initial_vocab': 16, 'resized_vocab': 21, 'hidden': 8, 'intermediate': 16,
                       'layers': 1, 'query_heads': 2, 'kv_heads': 1, 'source_rank': 2, 'source_scaling': 2},
            'base_parameters': base_parameters, 'shared_new_embedding_and_head_rows': True,
            'active_after_creation': active_before, 'layers': rows,
            'embedding_lookup_matches_folded_weight': True,
            'activation_max_logit_change': active_error, 'fold_max_logit_error': fold_error,
            'input_ids': ids.tolist(), 'generated_ids': generated.tolist(),
            'folded_greedy_ids_equal': True, 'source_and_base_parameters_unchanged': True}


def main():
    torch.set_num_threads(1)
    sources = check_sources()
    note = ROOT / '解释.md'
    text = note.read_text(encoding='utf-8').split('<details>', 1)[0]
    blocks = re.findall(r'```python\n(.*?)```', text, re.S)
    assert len(blocks) == 5, len(blocks)
    scope = {'__name__': '_lesson24_examples_'}
    stdout = []
    for i, block in enumerate(blocks):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(block, f'<lesson24-main-block-{i+1}>', 'exec'), scope)
        stdout.append(out.getvalue())
    checks = check_main(scope, blocks)
    extra = extra_matrix_checks(scope)
    llama = tiny_llama_checks()
    figures = json.loads((ROOT / '复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
    for item in figures['crops']:
        assert sha(ROOT / item['file']) == item['sha256']
    report = {'date': '2026-10-18', 'note_sha256': sha(note), 'script_sha256': sha(__file__),
              'python': sys.version, 'packages': {n: importlib.metadata.version(n) for n in
                  ['torch', 'peft', 'transformers', 'accelerate', 'numpy']},
              'official_source_sha256': sources, 'original_notebook_commit': '7fbc29824171fd07dc2c30fe922e61ebc018ad1d',
              'main_blocks': 5, 'stdout': stdout, 'main_checks': checks,
              'additional_matrix_checks': extra, 'tiny_llama': llama,
              'scope': 'CPU FP64 hand-computable PEFT examples and FP32 random one-layer Llama; no original pretrained weights, 4bit, GPU or downstream quality evaluation.'}
    out = ROOT / '复现/outputs/基础讲解验证.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'blocks': 5, 'matrix_outputs': checks['actual_peft_matrix_outputs'],
                      'llama_parameters': llama['base_parameters'], 'llama_layers_checked': len(llama['layers']),
                      'fold_error': llama['fold_max_logit_error'], 'result': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
