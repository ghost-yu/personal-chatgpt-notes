"""Execute the four teaching blocks and compare real CPU-adapted Meta cache paths."""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / '代码/tutorials'
META_SHA = '56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
DEFAULT_SOURCE = 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/llama-model.py'
DEFAULT_FAIRSCALE = 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CpuAllocation(ast.NodeTransformer):
    def __init__(self):
        self.changes = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            assert isinstance(node.func.value, ast.Call)
            assert isinstance(node.func.value.func, ast.Attribute)
            assert ast.unparse(node.func.value.func) == 'torch.zeros'
            node.func.attr = 'to'
            node.args = [ast.Constant('cpu')]
            self.changes += 1
        return node


class RectangularMask(ast.NodeTransformer):
    def __init__(self):
        self.changes = 0

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == 'mask' for t in node.targets):
            if 'diagonal=start_pos + 1' in ast.unparse(node.value):
                replacement = ast.parse(
                    'mask = make_causal_mask(start_pos, seqlen).to(h).view(1, 1, seqlen, start_pos + seqlen)'
                ).body[0]
                self.changes += 1
                return ast.copy_location(replacement, node)
        return self.generic_visit(node)


@contextlib.contextmanager
def cpu_llama_session(mask_factory=None):
    """Load the full fixed file, adapting exactly two allocations (and optionally one mask)."""
    source = Path(os.environ.get('LESSON20_META_SOURCE', DEFAULT_SOURCE))
    fairscale = Path(os.environ.get('LESSON20_FAIRSCALE', DEFAULT_FAIRSCALE))
    assert sha(source) == META_SHA
    sys.path.insert(0, str(fairscale))
    import fairscale.nn.model_parallel.initialize as fs
    tree = ast.parse(source.read_text(encoding='utf-8'))
    adapter = CpuAllocation()
    tree = adapter.visit(tree)
    assert adapter.changes == 2
    mask_changes = 0
    if mask_factory is not None:
        repair = RectangularMask()
        tree = repair.visit(tree)
        mask_changes = repair.changes
        assert mask_changes == 1
    ast.fix_missing_locations(tree)
    name = '_lesson20_foundation_meta_' + str(id(tree))
    module = types.ModuleType(name)
    module.__file__ = str(source)
    module.make_causal_mask = mask_factory
    module.cpu_allocation_changes = adapter.changes
    module.mask_assignment_changes = mask_changes
    sys.modules[name] = module
    exec(compile(tree, str(source), 'exec'), module.__dict__)
    assert not torch.distributed.is_initialized(), 'Run each teaching session outside another process group.'
    with tempfile.TemporaryDirectory(prefix='lesson20-foundation-gloo-') as directory:
        torch.distributed.init_process_group(
            'gloo', init_method=(Path(directory) / 'rendezvous').as_uri(), rank=0, world_size=1)
        fs.initialize_model_parallel(1)
        try:
            yield module
        finally:
            fs.destroy_model_parallel()
            torch.distributed.destroy_process_group()
            sys.modules.pop(name, None)


def fresh_model(module, scope):
    model = module.Transformer(module.ModelArgs(**vars(scope['args']))).eval()
    model.load_state_dict(scope['cached'].state_dict())
    return model


def slice_assignments(tree):
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = ast.unparse(node.targets[0])
            if target.startswith(('self.cache_k[', 'self.cache_v[')) or (
                    target in ('keys', 'values') and isinstance(node.value, ast.Subscript)):
                result.append(ast.dump(node, include_attributes=False))
    return result


def model_checks(scope):
    ids = scope['ids']
    with cpu_llama_session() as module, torch.inference_mode():
        cached, reference = fresh_model(module, scope), fresh_model(module, scope)
        state = {k: v.clone() for k, v in cached.state_dict().items()}
        prefix_results, lengths = [], {}
        handles = []
        for label, model in [('cached', cached), ('reference', reference)]:
            for layer_id, layer in enumerate(model.layers):
                for projection in ('wq', 'wk', 'wv'):
                    key = f'{label}/layer{layer_id}/{projection}'
                    lengths[key] = []
                    handles.append(getattr(layer.attention, projection).register_forward_pre_hook(
                        lambda mod, inputs, key=key: lengths[key].append(inputs[0].shape[1])))
        for end in (3, 4, 5, 6):
            start = 0 if end == 3 else end - 1
            previous = [(l.attention.cache_k[:, :start].clone(),
                         l.attention.cache_v[:, :start].clone()) for l in cached.layers]
            actual = cached(ids[:, start:end], start)
            expected = reference(ids[:, :end], 0)[:, start:end]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            for layer_id, (ca, re) in enumerate(zip(cached.layers, reference.layers)):
                a, b = ca.attention, re.attention
                for name in ('cache_k', 'cache_v'):
                    torch.testing.assert_close(getattr(a, name)[:, :end], getattr(b, name)[:, :end],
                                               atol=1e-6, rtol=1e-5)
                    assert not torch.count_nonzero(getattr(a, name)[:, end:])
                assert torch.equal(a.cache_k[:, :start], previous[layer_id][0])
                assert torch.equal(a.cache_v[:, :start], previous[layer_id][1])
            prefix_results.append({'start_pos': start, 'input_length': end-start,
                                   'logits_shape': list(actual.shape),
                                   'max_error': (actual-expected).abs().max().item()})
        for handle in handles:
            handle.remove()
        for key, value in lengths.items():
            assert value == ([3, 1, 1, 1] if key.startswith('cached') else [3, 4, 5, 6])
        allocation = sum(t.numel()*t.element_size() for l in cached.layers
                         for t in (l.attention.cache_k, l.attention.cache_v))
        assert allocation == 512
        effective_3 = 2 * 2 * 1 * 3 * 1 * 4 * 4
        assert effective_3 == 192
        assert not any('cache' in k for k in cached.state_dict())
        assert not any('cache' in k for k, _ in cached.named_buffers())

        # Compare the actual post-RoPE K / unrotated V against a componentwise rotation.
        attention = cached.layers[0].attention
        hidden = torch.tensor([[[.1, .2, .3, .4, -.1, -.2, -.3, -.4]]])
        qraw = attention.wq(hidden).view(1, 1, 2, 4)
        kraw = attention.wk(hidden).view(1, 1, 1, 4)
        vraw = attention.wv(hidden).view(1, 1, 1, 4)
        angles = 6 * torch.tensor([1., .01], dtype=torch.float64)
        def rotate(x):
            pair = x.double().reshape(*x.shape[:-1], 2, 2)
            a, b = pair[..., 0], pair[..., 1]
            return torch.stack([a*angles.cos()-b*angles.sin(),
                                a*angles.sin()+b*angles.cos()], -1).flatten(-2)
        actual = attention(hidden, 6, cached.freqs_cis[6:7], None)
        k_error = (attention.cache_k[:, 6:7].double()-rotate(kraw)).abs().max().item()
        assert k_error < 1e-6
        assert torch.equal(attention.cache_v[:, 6:7], vraw)
        key = attention.cache_k[:, :7].double().repeat_interleave(2, 2).transpose(1, 2)
        value = attention.cache_v[:, :7].double().repeat_interleave(2, 2).transpose(1, 2)
        prob = (rotate(qraw).transpose(1, 2) @ key.transpose(-1, -2) / 2).softmax(-1)
        joined = (prob @ value).transpose(1, 2).reshape(1, 1, 8)
        independent = joined @ attention.wo.weight.double().T
        attention_error = (actual.double()-independent).abs().max().item()
        assert attention_error < 1e-6

        clean = reference(ids[:, :3], 0).clone()
        for layer in reference.layers:
            layer.attention.cache_k[:, 3:].fill_(123)
            layer.attention.cache_v[:, 3:].fill_(456)
        assert torch.equal(clean, reference(ids[:, :3], 0))
        original = reference(ids, 0)[:, :4].clone()
        altered = ids.clone()
        altered[:, 4:] = torch.tensor([[14, 15]])
        assert torch.equal(original, reference(altered, 0)[:, :4])
        wrong = fresh_model(module, scope)
        wrong(ids[:, :3].flip(1), 0)
        mismatch = (wrong(ids[:, 3:4], 3)-reference(ids[:, :4], 0)[:, -1:]).abs().max().item()
        assert mismatch > 1e-5
        failure = None
        cached(ids[:, :3], 0)
        try:
            cached(ids[:, 3:5], 3)
        except RuntimeError as error:
            failure = str(error)
        assert failure and 'size' in failure
        generated = []
        seq = ids[:, :3].clone()
        for step in range(4):
            start = 0 if step == 0 else seq.shape[1]-1
            a = cached(seq[:, start:], start)[:, -1]
            b = reference(seq, 0)[:, -1]
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            assert torch.equal(a.argmax(-1), b.argmax(-1))
            nxt = a.argmax(-1)
            generated.append(nxt.item())
            seq = torch.cat([seq, nxt[:, None]], 1)
        assert all(torch.equal(state[k], v) for k, v in cached.state_dict().items())
        assert all(p.grad is None for p in cached.parameters())
        parameter_count = sum(p.numel() for p in cached.parameters())

    mask = scope['make_causal_mask']
    for p in range(5):
        for s in range(1, 4):
            expected = torch.tensor([[0. if j <= p+i else -float('inf')
                                      for j in range(p+s)] for i in range(s)])
            assert torch.equal(mask(p, s), expected)
    with cpu_llama_session(mask_factory=mask) as module, torch.inference_mode():
        chunk, reference = fresh_model(module, scope), fresh_model(module, scope)
        chunk_errors = []
        for start, end in [(0, 3), (3, 5), (5, 6)]:
            a = chunk(ids[:, start:end], start)
            b = reference(ids[:, :end], 0)[:, start:end]
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            chunk_errors.append((a-b).abs().max().item())
    return {'config': vars(scope['args']), 'parameters': parameter_count,
            'all_qkv_projection_lengths': lengths, 'prefix_comparisons': prefix_results,
            'every_layer_old_kv_unchanged': True, 'every_layer_future_capacity_unwritten': True,
            'allocated_cache_bytes': allocation, 'effective_3_token_bytes': effective_3,
            'post_rope_k_componentwise_error': k_error, 'v_is_unrotated_projection': True,
            'independent_attention_float64_error': attention_error,
            'future_changes_do_not_affect_prefix': True, 'dirty_tail_excluded': True,
            'mismatched_prefix_logit_error': mismatch, 'old_mask_error': failure,
            'rectangular_mask_cases': 15, 'chunk_mask_assignment_changes': 1,
            'chunk_3_2_1_errors': chunk_errors, 'greedy_ids': generated,
            'all_parameters_unchanged': True, 'all_parameter_grads_none': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-source', default=DEFAULT_SOURCE)
    parser.add_argument('--fairscale', default=DEFAULT_FAIRSCALE)
    args = parser.parse_args()
    os.environ['LESSON20_META_SOURCE'] = args.meta_source
    os.environ['LESSON20_FAIRSCALE'] = args.fairscale
    os.chdir(ROOT.parents[1])
    torch.set_num_threads(1)
    note = ROOT / '解释.md'
    body = note.read_text(encoding='utf-8').split('<details>', 1)[0]
    blocks = re.findall(r'```python\n(.*?)```', body, re.S)
    assert len(blocks) == 4
    scope = {'__name__': '_lesson20_teaching_examples'}
    block_outputs = []
    for i, block in enumerate(blocks, 1):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(block, f'lesson20-note-block-{i}', 'exec'), scope)
        block_outputs.append({'block': i, 'stdout': stream.getvalue()})
    source_tree = ast.parse(Path(args.meta_source).read_text(encoding='utf-8'))
    attention = next(n for n in source_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Attention')
    teaching = ast.parse(blocks[1])
    assert slice_assignments(attention) == slice_assignments(teaching)
    score = np.array([1., 0., 1., 1.]) / np.sqrt(2)
    probability = np.exp(score-score.max()); probability /= probability.sum()
    independent = probability @ np.array([[2., 0.], [0., 4.], [2., 2.], [4., 0.]])
    manual_error = np.abs(scope['output'].numpy()[0]-independent).max().item()
    assert manual_error < 1e-6
    assert scope['cached_lengths'] == [3, 1, 1, 1]
    assert scope['reference_lengths'] == [3, 4, 5, 6]
    results = model_checks(scope)
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor(model_proto=(CODE / 'tokenizer.model').read_bytes())
    ids = tokenizer.encode('I believe the meaning of life is', out_type=int)
    assert [tokenizer.bos_id()] + ids == [1,306,4658,278,6593,310,2834,338]
    report = {'note_sha256': sha(note), 'script_sha256': sha(__file__),
              'official_source_sha256': sha(args.meta_source),
              'official_commit': '7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e',
              'cpu_allocation_changes': 2, 'teaching_cache_four_assignments_ast_equal': True,
              'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
              'fairscale': '0.4.13 (local source target)',
              'blocks': block_outputs, 'manual_attention_output': scope['output'].tolist(),
              'manual_attention_numpy_error': manual_error, 'model': results,
              'sentencepiece_ids_with_bos': [tokenizer.bos_id()]+ids,
              'scope': 'Four current teaching blocks; full fixed Meta file with two CPU allocation edits; independent one-mask repair comparison. Random small model, no pretrained weights or GPU timing.'}
    target = ROOT / '复现/outputs/基础讲解验证.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'blocks': 4, 'parameters': results['parameters'],
                      'max_prefix_error': max(x['max_error'] for x in results['prefix_comparisons']),
                      'chunk_errors': results['chunk_3_2_1_errors'],
                      'report': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
