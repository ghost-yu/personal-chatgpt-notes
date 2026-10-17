# -*- coding: utf-8 -*-
"""Execute the four reading examples and compare complete official CPU modules."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

LESSON = Path(__file__).resolve().parents[2]
NOTE = LESSON / '解释.md'
MODEL = Path('G:/zzz/Personal ChatGPT 系列/17 RMSNorm 与 SwiGLU/资料/llama-model.py')
FAIRSCALE = Path('G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local')
sys.path.insert(0, str(FAIRSCALE))
import fairscale
import fairscale.nn.model_parallel.initialize as fs_init


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def without_docs(node):
    for n in ast.walk(node):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.body:
            if isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                if isinstance(n.body[0].value.value, str):
                    n.body.pop(0)
    return ast.dump(node)


def main():
    assert sha(MODEL) == '56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    text = NOTE.read_text(encoding='utf-8')
    blocks = re.findall(r'```python\n(.*?)```', text.split('<details>', 1)[0], re.S)
    assert len(blocks) == 4
    torch.manual_seed(17)
    scope = {'__name__': 'lesson17_foundation'}
    trace = []
    for i, block in enumerate(blocks, 1):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(block, str(NOTE) + ':block' + str(i), 'exec'), scope)
        trace.append({'block': i, 'stdout': capture.getvalue()})
    s = scope
    assert list(s['mean_square'].shape) == [1, 2, 1]
    torch.testing.assert_close(s['mean_square'], torch.tensor([[[12.5], [12.5]]]))
    torch.testing.assert_close(s['rms_result'], torch.tensor([[[3., 4.], [0., 5.]]]) / (12.5 ** .5))
    torch.testing.assert_close(s['ln_result'], torch.tensor([[[-1., 1.], [-1., 1.]]]), atol=3e-5, rtol=0)
    torch.testing.assert_close(s['y_with_gain'], s['y'] * torch.tensor([2., 1.]))
    assert float(s['loss_after']) < float(s['loss'])
    assert all(not torch.equal(p, s['before_step'][name]) for name, p in s['ffn'].named_parameters())

    raw = json.loads((LESSON / '代码/tutorials/llama2_src_02_RMSNorm_SwiGLU.ipynb').read_bytes())
    original_class = ast.parse(''.join(raw['cells'][11]['source'])).body[0]
    note_class = next(n for n in ast.parse(blocks[1]).body if isinstance(n, ast.ClassDef))
    assert ast.dump(original_class) == ast.dump(note_class)
    spec = importlib.util.spec_from_file_location('lesson17_fixed_meta_model', MODEL)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    official_class = next(n for n in ast.parse(MODEL.read_text(encoding='utf-8')).body
                          if isinstance(n, ast.ClassDef) and n.name == 'RMSNorm')
    assert without_docs(original_class) == without_docs(official_class)
    norm = official.RMSNorm(2)
    torch.testing.assert_close(norm(s['x']), s['y'], atol=0, rtol=0)
    assert torch.equal(norm(torch.zeros(1, 1, 2)), torch.zeros(1, 1, 2))
    norm.eval()
    eval_output = norm(s['x']).detach()
    norm.train()
    assert torch.equal(norm(s['x']).detach(), eval_output)
    dtype_cases = []
    for input_dtype in (torch.float16, torch.bfloat16, torch.float32):
        for weight_dtype in (torch.float32, input_dtype):
            layer = official.RMSNorm(2).to(weight_dtype)
            out = layer(s['x'].to(input_dtype))
            dtype_cases.append({'input': str(input_dtype), 'weight': str(weight_dtype), 'output': str(out.dtype)})
            assert out.dtype == weight_dtype

    import torch.distributed as dist
    with tempfile.TemporaryDirectory(prefix='lesson17-foundation-') as tmp:
        # File rendezvous avoids a fixed port and uses a fresh absolute filename.
        rendezvous = (Path(tmp) / 'rendezvous').resolve().as_uri()
        dist.init_process_group('gloo', init_method=rendezvous, rank=0, world_size=1)
        try:
            fs_init.initialize_model_parallel(1)
            actual = official.FeedForward(dim=2, hidden_dim=4, multiple_of=1, ffn_dim_multiplier=None)
            reference = s['TinySwiGLU']()
            with torch.no_grad():
                for name, p in actual.named_parameters():
                    p.copy_(s['before_step'][name])
                reference.load_state_dict(s['before_step'])
            actual_x = s['ffn_input'].clone().requires_grad_()
            reference_x = s['ffn_input'].clone().requires_grad_()
            actual_y, reference_y = actual(actual_x), reference(reference_x)
            torch.testing.assert_close(actual_y, reference_y, atol=0, rtol=0)
            torch.testing.assert_close(actual_y, s['ffn_output'], atol=0, rtol=0)
            actual_loss, reference_loss = actual_y.square().mean(), reference_y.square().mean()
            actual_loss.backward()
            reference_loss.backward()
            torch.testing.assert_close(actual_x.grad, reference_x.grad, atol=0, rtol=0)
            grad_errors = {}
            for name, p in actual.named_parameters():
                rp = dict(reference.named_parameters())[name]
                torch.testing.assert_close(p.grad, rp.grad, atol=0, rtol=0)
                torch.testing.assert_close(p.grad, dict(s['ffn'].named_parameters())[name].grad, atol=0, rtol=0)
                assert torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) == p.numel()
                assert torch.equal(p.detach(), s['before_step'][name])
                grad_errors[name] = (p.grad - rp.grad).abs().max().item()
            optimizer = torch.optim.SGD(actual.parameters(), lr=.01)
            assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in actual.parameters()}
            optimizer.step()
            for name, p in actual.named_parameters():
                torch.testing.assert_close(p, dict(s['ffn'].named_parameters())[name], atol=0, rtol=0)
            after = actual(s['ffn_input']).square().mean().item()
            assert after == s['loss_after'].item()
            actual_result = dict(input_shape=list(actual_x.shape), branch_shape=list(actual.w1(actual_x).shape),
                                 output=actual_y.detach().tolist(), parameters=sum(p.numel() for p in actual.parameters()),
                                 input_gradient=actual_x.grad.tolist(), weight_gradient_max_errors=grad_errors,
                                 loss_before=actual_loss.item(), loss_after=after, all_three_weights_changed=True,
                                 optimizer_parameters_exact=True, forward_max_error=0.)
        finally:
            fs_init.destroy_model_parallel()
            dist.destroy_process_group()

    width = ((int(2 * (4 * 4096) / 3) + 255) // 256) * 256
    assert width == 11008 and 3 * 4096 * width == 135266304
    result = dict(date_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  torch=torch.__version__, fairscale=fairscale.__version__, device='cpu',
                  official_source=str(MODEL), official_sha256=sha(MODEL), note_sha256=sha(NOTE),
                  verifier_sha256=sha(Path(__file__)), original_notebook_class_ast_identical=True,
                  official_class_ast_identical_ignoring_docstrings=True, blocks=trace,
                  rms_mean_square=s['mean_square'].tolist(), rms_output=s['y'].tolist(),
                  with_gain=s['y_with_gain'].tolist(), layernorm_output=s['ln_result'].tolist(),
                  zero_and_train_eval_checked=True, dtype_cases=dtype_cases, official_ffn=actual_result,
                  width=width, width_parameters=3 * 4096 * width,
                  scope='正文四块代码、完整固定官方CPU单rank模块；不重跑旧notebook与论文任务实验')
    output = LESSON / '复现/outputs/基础讲解验证.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'blocks': len(blocks), 'official_ffn': actual_result, 'report': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
