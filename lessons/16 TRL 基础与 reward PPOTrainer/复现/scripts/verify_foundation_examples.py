"""从当前讲解提取三段Python，真实运行并保存结果。"""
from pathlib import Path
import contextlib
import hashlib
import inspect
import io
import json
import os
import re
import sys
import importlib.metadata as metadata

sys.stdout.reconfigure(encoding='utf-8')
lesson = Path(__file__).resolve().parents[2]
note = lesson / '解释.md'
text = note.read_text(encoding='utf-8')
prefix = text.split('<details>', 1)[0]
blocks = re.findall(r'```python\n(.*?)\n```', prefix, flags=re.S)
assert len(blocks) == 3
assert metadata.version('trl') == '0.4.7'
assert metadata.version('transformers') == '4.30.0'
from trl import PPOTrainer
import torch

source = Path(inspect.getfile(PPOTrainer))
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == '8142e1296171adc0a9f664ed0a241d4e7313ad0fac6987b2a093b0982489a407'
tokenizer_file = Path(os.environ['LESSON16_TOKENIZER'])
assert hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() == '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
scope = {'__name__': '__main__'}
outputs = []
for index, code in enumerate(blocks, 1):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, f'{note.name}:block{index}', 'exec'), scope)
    outputs.append({'block': index, 'sha256': hashlib.sha256(code.encode()).hexdigest(),
                    'stdout': capture.getvalue()})
    print(capture.getvalue(), end='')

stats = scope['stats']
# 用正文手算的三步GAE，对照旧版完整loss方法中的实际结果。
trainer = scope['trainer']
old_values = scope['values'][None, :]
old_rewards = scope['rewards'][None, :]
mask = torch.ones((1, 3), dtype=torch.float64)
old_logprob = torch.full((1, 3), -0.6931471805599453, dtype=torch.float64)
with torch.no_grad():
    _, actual_value_loss, gae_stats = trainer.loss(
        old_logprob, old_values, old_rewards,
        torch.zeros((1, 3, 2), dtype=torch.float64), old_values,
        old_logprob, mask)
raw = scope['advantages']
expected_whitened = (raw - raw.mean()) / torch.sqrt(raw.var(unbiased=True) + 1e-8)
torch.testing.assert_close(gae_stats['policy/advantages'], expected_whitened[None, :])
torch.testing.assert_close(gae_stats['returns/mean'], scope['returns'].mean())
torch.testing.assert_close(actual_value_loss, 0.1 * 0.5 * raw.square().mean())
# 两候选分布的KL奖励：检查所选token的差值与末位score。
beta = trainer.kl_ctl.value
sample_logprob = torch.tensor([[0.8, 0.2]], dtype=torch.float64).log()
ref_logprob = torch.tensor([[0.5, 0.5]], dtype=torch.float64).log()
shaped, penalty = trainer.compute_rewards(
    torch.tensor([1.0], dtype=torch.float64), sample_logprob, ref_logprob,
    torch.ones((1, 2), dtype=torch.float64))
torch.testing.assert_close(penalty, -beta * (sample_logprob - ref_logprob))
torch.testing.assert_close(shaped - penalty, torch.tensor([[0.0, 1.0]], dtype=torch.float64))
report = {
    'status': '部分成功',
    'note_sha256': hashlib.sha256(note.read_bytes()).hexdigest(),
    'python': sys.version,
    'packages': {name: metadata.version(name) for name in ['torch', 'transformers', 'trl', 'accelerate']},
    'ppo_trainer_sha256': source_hash,
    'tokenizer_sha256': hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
    'blocks': outputs,
    'model': {'parameters': sum(p.numel() for p in scope['model'].parameters()),
              'hidden': 8, 'layers': 1, 'heads': 2, 'vocab': 50257,
              'head_parameters': sum(p.numel() for p in scope['model'].v_head.parameters())},
    'query': scope['query'].tolist(), 'full': scope['full'].tolist(),
    'response': scope['response'].tolist(),
    'wrong_targets': int(scope['wrong_mask'].sum()),
    'correct_mask': scope['mask'].tolist(),
    'optimizer_steps': scope['step_count'][0],
    'changed_names': scope['changed'],
    'reference_unchanged': all(torch.equal(value, scope['ref_before'][name])
                               for name, value in scope['reference'].state_dict().items()),
    'policy_loss': float(stats['ppo/loss/policy']),
    'value_loss': float(stats['ppo/loss/value']),
    'official_loss_gae_matches_hand_calculation': True,
    'official_compute_rewards_matches_two_candidate_formula': True,
    'scope': '随机CPU小模型、固定score1.0、贪心新增3个token；本次仅执行正文三块，不重跑历史原单元实验。'
}
output = lesson / '复现/outputs/基础讲解验证.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('结果已写入', output)
