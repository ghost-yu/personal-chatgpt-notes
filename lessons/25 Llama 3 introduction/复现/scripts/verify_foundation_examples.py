"""Execute the published introduction and verify complete fixed components."""
from pathlib import Path
import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get('LESSON25_LOCAL', 'G:/zzz/Personal ChatGPT 系列/25 Llama 3 introduction/资料'))
sys.path.insert(0, os.environ.get('LESSON25_RUNTIME_LIBS', str(LOCAL / 'runtime-libs')))
sys.path.insert(0, os.environ.get('LESSON25_FAIRSCALE', 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
import transformers


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_sources():
    manifest = json.loads((ROOT / '来源清单.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        assert sha(ROOT / item['published_path']) == item['sha256']
    for rel, expected in manifest['official_sources_sha256'].items():
        assert sha(LOCAL / 'meta-llama3' / rel) == expected, rel
    assert (ROOT / '代码/tutorials/llama/tokenizer.py').read_bytes() == (LOCAL / 'meta-llama3/llama/tokenizer.py').read_bytes()
    return manifest['official_sources_sha256']


def parameter_checks(scope, blocks):
    totals = {}
    breakdown = {}
    for name, expected in [('Llama2', 6738415616), ('Llama3', 8030261248)]:
        model = scope['full_models'][name]
        assert all(p.is_meta for p in model.parameters())
        totals[name] = sum(p.numel() for p in model.parameters())
        assert totals[name] == expected
        cfg = model.config
        v, d, m = cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size
        l, h, g = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        pieces = {'embedding_and_head': 2*v*d, 'attention': l*(2*d*d + 2*d*g*(d//h)),
                  'feed_forward': l*3*d*m, 'norm': l*2*d+d}
        assert sum(pieces.values()) == expected
        assert model.model.embed_tokens.weight is not model.lm_head.weight
        assert list(model.model.layers[0].self_attn.k_proj.weight.shape) == [g*(d//h), d]
        assert list(model.model.layers[0].mlp.gate_proj.weight.shape) == [m, d]
        breakdown[name] = pieces
    deltas = {k: breakdown['Llama3'][k] - breakdown['Llama2'][k] for k in breakdown['Llama2']}
    assert deltas == {'embedding_and_head': 788529152, 'attention': -805306368,
                      'feed_forward': 1308622848, 'norm': 0}
    notebook = json.loads((ROOT / '代码/tutorials/llama/llama3_supp_data_config.ipynb').read_text(encoding='utf-8'))
    original = ast.parse(''.join(notebook['cells'][6]['source']) + '\n' + ''.join(notebook['cells'][8]['source']))
    expected = [n for n in original.body if isinstance(n, ast.Assign)]
    actual = [n for n in ast.parse(blocks[1]).body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'hidden_dim' for t in n.targets)]
    assert [ast.dump(n) for n in expected] == [ast.dump(n) for n in actual]
    assert scope['hidden_dim'] == 14336
    kv_bytes = {'Llama2_4096': 2*32*4096*32*128*2, 'Llama3_4096': 2*32*4096*8*128*2,
                'Llama3_8192': 2*32*8192*8*128*2}
    assert kv_bytes == {'Llama2_4096': 2*1024**3, 'Llama3_4096': 512*1024**2, 'Llama3_8192': 1024**3}
    return {'totals': totals, 'breakdown': breakdown, 'deltas': deltas,
            'width_assignments_ast_match_original': True, 'intermediate_width': 14336,
            'full_weight_gib_at_two_bytes': totals['Llama3']*2/1024**3,
            'physical_pretrained_weight_data': False, 'kv_bytes': kv_bytes}


def tokenizer_checks(scope):
    t2, t3, fast = scope['t2'], scope['t3'], scope['fast']
    assert t2.vocab_size() == 32000 and t3.n_words == len(fast) == 128256
    assert scope['ids2'] == [15043, 29892, 3186, 29991]
    assert scope['ids3'] == [9906, 11, 1917, 0]
    assert t3.bos_id == 128000 and t3.eos_id == 128001
    assert t3.special_tokens['<|eot_id|>'] == 128009
    samples = ['', '你好，世界！', '1234567890', 'cafe\u0301', '🙂', '  a\n\tb  ']
    results = []
    for text in samples:
        ids = t3.encode(text, bos=False, eos=False)
        assert t3.decode(ids) == text
        assert fast.encode(text, add_special_tokens=False) == ids, repr(text)
        results.append({'text': text, 'ids': ids})
    # Construct the expected sequence directly from the protocol, independently of ChatFormat.
    messages = scope['messages']
    expected = [128000]
    for msg in messages:
        expected += [128006] + t3.encode(msg['role'], bos=False, eos=False) + [128007]
        expected += t3.encode('\n\n', bos=False, eos=False)
        expected += t3.encode(msg['content'].strip(), bos=False, eos=False) + [128009]
    expected += [128006] + t3.encode('assistant', bos=False, eos=False) + [128007]
    expected += t3.encode('\n\n', bos=False, eos=False)
    assert expected == scope['chat_ids'] == scope['hf_chat_ids']
    assert expected.count(128000) == 1 and expected.count(128009) == 2
    assert fast.encode(scope['prompt'], add_special_tokens=False) == expected
    for token, idx in t3.special_tokens.items():
        assert fast.convert_tokens_to_ids(token) == idx
    assert t3.encode('<|eot_id|>', bos=False, eos=False, allowed_special='all') == [128009]
    assert t3.encode('<|eot_id|>', bos=False, eos=False) != [128009]
    rejected = False
    try:
        t3.encode('<|eot_id|>', bos=False, eos=False, disallowed_special='all')
    except ValueError:
        rejected = True
    assert rejected
    return {'english_text': scope['sample_text'], 'llama2_ids': scope['ids2'], 'llama3_ids': scope['ids3'],
            'roundtrip_samples': results, 'chat_ids': expected, 'chat_length': len(expected),
            'prompt': scope['prompt'], 'meta_hf_chat_ids_equal': True,
            'all_256_special_ids_preserved': True, 'literal_special_token_branches_checked': True}


def inference_checks(scope):
    model = scope['tiny']
    input_ids = scope['input_ids']
    result = scope['result']
    t = input_ids.shape[1]
    assert list(result.hidden_states[-1].shape) == [1, t, 8]
    assert list(result.logits.shape) == [1, t, 128256]
    assert list(result.past_key_values[0][0].shape) == [1, 1, t, 4]
    assert list(result.past_key_values[0][1].shape) == [1, 1, t, 4]
    with torch.no_grad():
        # Actual final hidden state times the output matrix gives the stored logits.
        projected = result.hidden_states[-1] @ model.lm_head.weight.T
        torch.testing.assert_close(projected, result.logits)
        probability = result.logits[:, -1].softmax(-1)
        torch.testing.assert_close(probability.sum(-1), torch.ones(1))
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    cache_error = (scope['one_more'].logits[:, -1] - scope['full'].logits[:, -1]).abs().max().item()
    # Greedy selection written as a full-prefix loop, independently of generate's cache loop.
    ids = input_ids.clone()
    end_ids = {128001, 128009}
    with torch.no_grad():
        for _ in range(3):
            logits = model(ids, use_cache=False).logits[:, -1]
            next_id = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() in end_ids:
                break
    assert torch.equal(ids, scope['generated'])
    pipe_ids = scope['pipe'](scope['prompt'], max_new_tokens=3, do_sample=False,
                             add_special_tokens=False, eos_token_id=list(sorted(end_ids)), return_tensors=True)
    assert pipe_ids[0]['generated_token_ids'] == ids[0].tolist()
    assert scope['full_text'].startswith(scope['prompt'])
    assert scope['continuation'] == scope['full_text'][len(scope['prompt']):]
    assert all(torch.equal(p, state[n]) and p.grad is None for n, p in model.named_parameters())
    assert scope['toy_next'] == 1
    torch.testing.assert_close(scope['toy_logits'], torch.tensor([[[0., 3., 1., 0.]]]))
    return {'base_parameters': sum(p.numel() for p in model.parameters()),
            'input_shape': list(input_ids.shape), 'hidden_shape': list(result.hidden_states[-1].shape),
            'logit_shape': list(result.logits.shape), 'kv_shape': list(result.past_key_values[0][0].shape),
            'cache_full_prefix_logit_error': cache_error, 'final_head_matmul_equal': True,
            'generated_new_ids': ids[0, t:].tolist(), 'pipeline_and_manual_generate_ids_equal': True,
            'pipeline_result_fields': list(scope['pipeline_outputs'][0]),
            'inference_replay_preserves_all_parameters_and_no_gradients': True}


def full_official_ffn_checks():
    import fairscale.nn.model_parallel.initialize as mp
    source = LOCAL / 'meta-llama3/llama/model.py'
    spec = importlib.util.spec_from_file_location('lesson25_full_official_model', source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    notebook = json.loads((ROOT / '代码/tutorials/llama/llama3_supp_data_config.ipynb').read_text(encoding='utf-8'))
    original = re.findall(r'```[^\n]*\n(.*?)```', ''.join(notebook['cells'][7]['source']), re.S)[0]
    official_node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if getattr(n, 'name', '') == 'FeedForward')
    assert ast.dump(ast.parse(original).body[0]) == ast.dump(official_node)
    with tempfile.TemporaryDirectory(prefix='lesson25-foundation-gloo-') as td:
        torch.distributed.init_process_group('gloo', init_method=(Path(td)/'rendezvous').as_uri(), rank=0, world_size=1)
        try:
            mp.initialize_model_parallel(1)
            torch.manual_seed(2525)
            ff = mod.FeedForward(16, 64, 8, 1.3).double()
            for p in ff.parameters():
                torch.nn.init.normal_(p, mean=0, std=.1)
            x = torch.randn(2, 3, 16, dtype=torch.float64, requires_grad=True)
            y = ff(x)
            y.square().sum().backward()
            w1, w2, w3 = [w.weight.detach().numpy() for w in (ff.w1, ff.w2, ff.w3)]
            xn = x.detach().numpy()
            z = xn @ w1.T
            sigmoid = 1 / (1 + np.exp(-z))
            a = z * sigmoid
            b = xn @ w3.T
            h = a * b
            reference = h @ w2.T
            dy = 2 * reference
            dh = dy @ w2
            dz = dh * b * sigmoid * (1 + z*(1-sigmoid))
            db = dh * a
            gradients = [dz @ w1 + db @ w3,
                         dz.reshape(-1, 56).T @ xn.reshape(-1, 16),
                         dy.reshape(-1, 16).T @ h.reshape(-1, 56),
                         db.reshape(-1, 56).T @ xn.reshape(-1, 16)]
            actual = [x.grad, ff.w1.weight.grad, ff.w2.weight.grad, ff.w3.weight.grad]
            forward_error = float(np.max(np.abs(y.detach().numpy() - reference)))
            errors = [float(np.max(np.abs(p.numpy()-q))) for p, q in zip(actual, gradients)]
            assert forward_error < 1e-13 and max(errors) < 1e-12
            assert sum(p.numel() for p in ff.parameters()) == 2688
            with torch.device('meta'):
                large = mod.FeedForward(4096, 4*4096, 1024, 1.3)
            # FairScale's legacy torch.Tensor constructor allocates on CPU even
            # inside a meta context. Only inspect shapes/counts; do not use its
            # intentionally uninitialized weights in a forward pass.
            large_devices = sorted({p.device.type for p in large.parameters()})
            assert large_devices == ['cpu']
            assert list(large.w1.weight.shape) == [14336, 4096]
            assert sum(p.numel() for p in large.parameters()) == 176160768
        finally:
            mp.destroy_model_parallel()
            torch.distributed.destroy_process_group()
    return {'full_source_sha256': sha(source), 'original_class_ast_equal': True,
            'small_input_shape': [2, 3, 16], 'small_intermediate_width': 56,
            'small_parameters': 2688, 'numpy_forward_error': forward_error,
            'numpy_analytic_input_and_three_weight_gradient_errors': errors,
            'full_width_parameters': 176160768, 'full_width_weight_devices': large_devices,
            'full_width_weights_uninitialized_no_forward': True, 'full_intermediate_width': 14336,
            'world_size': 1, 'backend': 'gloo'}


def main():
    torch.set_num_threads(1)
    assert transformers.__version__ == '4.44.2'
    sources = check_sources()
    note = ROOT / '解释.md'
    blocks = re.findall(r'```python\n(.*?)```', note.read_text(encoding='utf-8').split('<details>', 1)[0], re.S)
    assert len(blocks) == 5
    scope = {'__name__': '_lesson25_examples_'}
    previous = os.getcwd()
    outputs = []
    try:
        os.chdir(ROOT)
        for i, block in enumerate(blocks):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(block, f'<lesson25-main-block-{i+1}>', 'exec'), scope)
            outputs.append(out.getvalue())
    finally:
        os.chdir(previous)
    parameters = parameter_checks(scope, blocks)
    tokenizer = tokenizer_checks(scope)
    inference = inference_checks(scope)
    ffn = full_official_ffn_checks()
    figures = json.loads((ROOT / '复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
    for item in figures['crops']:
        assert sha(ROOT/item['file']) == item['sha256']
    report = {'date': '2026-10-18', 'note_sha256': sha(note), 'script_sha256': sha(__file__),
              'python': sys.version, 'packages': {n: importlib.metadata.version(n) for n in
                  ['torch', 'transformers', 'tiktoken', 'sentencepiece', 'numpy', 'fairscale']},
              'official_source_sha256': sources, 'main_blocks': 5, 'stdout': outputs,
              'parameters': parameters, 'tokenizer': tokenizer, 'inference': inference,
              'full_official_feed_forward': ffn,
              'scope': 'Initial April 2024 Llama 3; original tokenizers; meta full configurations; random CPU one-layer model, complete fixed tokenizer and FFN, single-rank Gloo. No pretrained 7B/8B weights, GPU, benchmarks or language-quality claim.'}
    output = ROOT / '复现/outputs/基础讲解验证.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'blocks': 5, 'chat_length': tokenizer['chat_length'],
                      'tokenizer_ids': [tokenizer['llama2_ids'], tokenizer['llama3_ids']],
                      'tiny_parameters': inference['base_parameters'], 'generated_ids': inference['generated_new_ids'],
                      'cache_error': inference['cache_full_prefix_logit_error'],
                      'ffn_gradient_error': max(ffn['numpy_analytic_input_and_three_weight_gradient_errors']),
                      'result': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
