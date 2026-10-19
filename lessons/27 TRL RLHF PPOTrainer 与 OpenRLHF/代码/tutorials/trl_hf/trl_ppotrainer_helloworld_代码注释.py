# 第27讲 Python 阅读版：来自固定上游notebook，新增内容均为注释。
# 实际可执行验证请使用复现/scripts/verify_delivery.py；本文件保留远程加载及环境设置。

# ===== 原单元 0 =====
# 设置本进程代理与 NCCL 环境变量，再导入图片展示类。代理127.0.0.1:7890不是通用网络配置；两条NCCL开关只在相应后端生效，不能让CPU版PyTorch获得CUDA。复现跳过此格，使用离线文件。
import os
# 中文说明：修改当前Python进程及其子进程环境；不是修改系统全局设置。
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# 中文说明：修改当前Python进程及其子进程环境；不是修改系统全局设置。
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# 中文说明：修改当前Python进程及其子进程环境；不是修改系统全局设置。
os.environ['NCCL_P2P_DISABLE'] = '1'
# 中文说明：修改当前Python进程及其子进程环境；不是修改系统全局设置。
os.environ['NCCL_IB_DISABLE'] = '1'
from IPython.display import Image

# ===== 原单元 1 =====
# 导入PyTorch、GPT2Tokenizer和旧版TRL三个核心类。AutoModelForCausalLMWithValueHead包装语言模型，并从最后一层hidden states另接标量价值头；PPOConfig保存超参，PPOTrainer组织生成和优化。本讲锁定TRL0.8.6，不将新版API套入旧代码。
import torch
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# ===== 原单元 2 =====
# 只有注释，不执行shell。!export是Unix/IPython风格，Windows不能直接照搬；上一格使用os.environ才实际修改进程环境。
# !export NCCL_P2P_DISABLE="1"

# ===== 原单元 3 =====
# 整体结构标题。先分清Actor、Reference、Critic和外部Reward Model，再读PPO更新。
# ## overall

# ===== 原单元 4 =====
# 原链接指向可变main；本交付将TRL锁到0.8.6、OpenRLHF锁到2024-05历史提交。PPO-penalty和PPO-clip是原论文两种surrogate。本讲实际实现使用clip优化，另外在逐token reward中惩罚偏离固定reference，不能把这两个KL约束混为一谈。
# - https://github.com/huggingface/trl/blob/main/examples/hello_world.py
# - OpenRLHF: https://github.com/OpenLLMAI/OpenRLHF
#     - https://arxiv.org/abs/2405.11143
# - PPO-penalty（PPO1）
#
#     $$
#     \begin{split}
#     &J^{\theta'}_{PPO}=J^{\theta'}(\theta)-\beta KL(\theta,\theta'),\quad J^{\theta'}(\theta)=\mathbb E_{s_t,a_t\sim \pi_{\theta'}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta'}(a_t|s_t)}A^{\theta'}(s_t,a_t)\right]\\
#     &\mathcal{L}^{\text{PENALTY}}(\theta) = \mathbb{E}_t \left[ \hat{A}_t \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} - \beta D_{KL} \left( \pi_{\theta_{\text{old}}}(\cdot | s_t) \parallel \pi_\theta(\cdot | s_t) \right) \right]
#     \end{split}
#     $$
#
# - PPO-clip（PPO2）
#
#     $$
#     J_{PPO2}^{\theta^k}(\theta) \approx \sum_{(s_t, a_t)} \min \left( \frac{p_\theta(a_t | s_t)}{p_{\theta^k}(a_t | s_t)} A^{\theta^k}(s_t, a_t), \
#     \text{clip} \left( \frac{p_\theta(a_t | s_t)}{p_{\theta^k}(a_t | s_t)}, 1 - \epsilon, 1 + \epsilon \right) A^{\theta^k}(s_t, a_t) \right)
#     $$

# ===== 原单元 5 =====
# 展示原OpenRLHF框架图。url参数只保存相对URL，构造Image不等于浏览器已加载图片；本交付提供真实图片。Actor生成response，Actor/Reference计算logprob，Critic计算value，Reward Model计算整句score，组装experience后更新Actor/Critic。
# 中文说明：构造展示对象；相对路径以notebook位置为基准。
Image(url='../../imgs/openrlhf.png', width=600)

# ===== 原单元 6 =====
# 模型对比标题。ref是固定参考模型；old是当前rollout的行为策略快照。
# ## model vs. model_ref

# ===== 原单元 7 =====
# 分别加载两份GPT2包装器。不是变量别名；两份语言模型从同一预训练权重起步，两个随机value head不保证相同。ref的value输出不会参与GAE。CPU验证仅替换两个加载路径，使用同一随机小GPT2 checkpoint，不能当原GPT2预训练效果。
# 1. load a pretrained model
# 中文说明：读取模型/分词器资产；默认远程名称可能访问网络，验证只使用明确本地路径。
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
# 中文说明：读取模型/分词器资产；默认远程名称可能访问网络，验证只使用明确本地路径。
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")

# ===== 原单元 8 =====
# 价值头作用于[B,L,H]，不是把词表概率[B,L,V]压到1。Linear(H,1)后squeeze(-1)得[B,L]。纠正原文：model_ref对应固定pi_ref/SFT；old_logprobs才对应每轮更新前的pi_old。自适应KL控制器截断的是相对误差，不是把KL值直接限制到±0.2。
# - `AutoModelForCausalLMWithValueHead`
#     - `ValueHead`:  `self.summary = nn.Linear(hidden_size, 1)` (hidden_size => 1)
#         - `value = self.v_head(last_hidden_state).squeeze(-1)`
# - `model` vs. `model_ref`
#     - `model`: $\pi_\theta$, `model_ref`: $\pi_{\theta_{old}}$（$\pi_{\text{sft}}$）
# - `AdaptiveKLController`（https://arxiv.org/pdf/1909.08593）
#     - 如下公式所示，$\pi_t, \rho$ 分别表示新旧策略，与 KL_target 的偏差在 -0.2 到 0.2 之间
#     - log-space controller
#
#     $$
#     \log\beta_{t+1}=\log\beta_t+\log(1+K_\beta e_t)
#     $$

# ===== 原单元 9 =====
# 自适应控制器按 beta×(1+clip(KL/target−1,−0.2,0.2)×n_steps/horizon) 调整 KL 惩罚强度。n_steps 是经验条数；在通常步长下，KL 偏高会提高 beta，增强后续更新的惩罚。
# 中文说明：构造展示对象；相对路径以notebook位置为基准。
Image(url='../../imgs/adap_kl_coef.png', width=500)

# ===== 原单元 10 =====
# 只有注释# model，Notebook不会展示模型，也不改变状态。
# model

# ===== 原单元 11 =====
# 读取语言模型参数所在设备；这是属性访问，不是搬运模型。无GPU的实测为cpu。
model.pretrained_model.device

# ===== 原单元 12 =====
# 从GPT2加载真实BPE tokenizer；最初pad_token为None，EOS为50256的特殊字符串。把pad指向EOS复用现有ID，不增加词表；attention_mask仍须区分真实EOS与padding。
# 中文说明：读取模型/分词器资产；默认远程名称可能访问网络，验证只使用明确本地路径。
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# 中文说明：输出用于人工检查，不改变模型权重。
print(tokenizer.pad_token, tokenizer.eos_token)
# 中文说明：复用EOS作为padding，不新增embedding行。
tokenizer.pad_token = tokenizer.eos_token

# ===== 原单元 13 =====
# encode将字符串变成ID列表，decode做逆映射。特殊EOS字符串在该tokenizer里编码成单个50256，不等于整句只允许该ID。
# 中文说明：输出用于人工检查，不改变模型权重。
print(tokenizer.encode('<|endoftext|>'))
# 中文说明：输出用于人工检查，不改变模型权重。
print(tokenizer.decode(tokenizer.encode('<|endoftext|>')))

# ===== 原单元 14 =====
# vocab_size返回50257，不是输入长度、hidden size或模型参数量。
tokenizer.vocab_size

# ===== 原单元 15 =====
# 字典**ppo_config解包为关键字参数。mini_batch_size=1/batch_size=1表示每次收一条经验、每个mini batch一条；ppo_epochs仍取旧库默认4，所以一次step不只一次Adam更新。PPOTrainer通过Accelerate准备模型/优化器/参考模型。
# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
# 中文说明：构造旧API配置对象；未指定字段仍保留该版本默认值。
config = PPOConfig(**ppo_config)
# 中文说明：绑定active模型、固定参考模型、tokenizer及优化器，建立训练状态。
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# ===== 原单元 16 =====
# 只有注释：提示真正的设备处理发生在PPOTrainer.__init__，此格不执行prepare。
# model, ... = self.accelerator.prepare(model, ...)

# ===== 原单元 17 =====
# print两个设备。CPU算例中两者均位于cpu。参与更新的权重由optimizer的参数集合决定；ref模型保持固定，为策略变化提供比较基准。
# 中文说明：输出用于人工检查，不改变模型权重。
print(model.pretrained_model.device)
# 中文说明：输出用于人工检查，不改变模型权重。
print(model_ref.pretrained_model.device)

# ===== 原单元 18 =====
# query_txt为prompt字符串。encode(return_tensors="pt")得int64[1,Q]，to只转设备；真实GPT2 BPE此句Q=7。尾部空格也参与分词。
# 3. encode a query
query_txt = "This morning I went to the "
# 中文说明：先得到int64[1,Q]，再放到语言模型设备。
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)
query_tensor

# ===== 原单元 19 =====
# 生成配置：min_length=-1不要求正最短长度；top_k=0关闭top-k筛选，top_p=1关闭nucleus截断，do_sample=True进行随机采样；max_new_tokens=20是新增token上限，不是固定生成20。EOS可提前结束。
# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}

# ===== 原单元 20 =====
# 进入生成阶段：使用active model，不使用ref，也不更新参数。
# ## `ppo_trainer.generate`

# ===== 原单元 21 =====
# list(query_tensor)沿batch维拆成一个int64[Q]张量的列表，不是把token转换成Python整数列表。
# 中文说明：沿第0维拆batch；列表里的每个元素仍是一维tensor。
list(query_tensor)

# ===== 原单元 22 =====
# generate接收list[Tensor[Q]]。return_prompt=False明确只返回新增response；与第16讲return_prompt=True不同，此处再拼query不会重复prompt。decode接受第一条响应ID，产出str。
# 中文说明：返回纯新增token，后续step再拼prompt时不会重复。
response_tensor = ppo_trainer.generate(list(query_tensor), return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])

# ===== 原单元 23 =====
# Notebook展示response_txt。随机小模型输出只用于核对接口，不能评价语言能力；历史输出属于上游运行。
response_txt

# ===== 原单元 24 =====
# 整格均为注释。直接model.generate通常会返回prompt+response，不能把该注释示例当作本格实际执行或等价的纯response接口。
# tokenizer.decode(model.generate(
#     input_ids=query_tensor,
#     **generation_kwargs
# )[0])

# ===== 原单元 25 =====
# 用常数1.0模拟奖励接口。scores 是长度为 B 的列表，元素为 float32 标量张量；每条回答收到相同分数，便于先观察 PPO 的接口与张量流。
# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
# 中文说明：零维浮点分数，没有从生成文本得到的计算图；不是序列标签。
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# ===== 原单元 26 =====
# 进入PPO step：准备输入、缓存旧策略logprob/value、固定ref前向、KL形状奖励、GAE、多个epoch的clip/value loss和反向更新。
# ## `ppo_trainer.step`

# ===== 原单元 27 =====
# 关键词queries/responses/scores明确三种输入。外层list长度B=1，内层query/response是各自一维ID。step返回包含loss、KL、ratio和耗时等统计的dict，同时实际修改active模型/optimizer及KL控制器状态。
# 中文说明：真正执行PPO训练，返回统计字典且更新参数。
train_stats = ppo_trainer.step(queries=[query_tensor[0]],
                               responses=[response_tensor[0]],
                               scores=reward)

# ===== 原单元 28 =====
# Markdown保存的是不完整step摘录，不能当独立函数运行。实际完整PPOTrainer见源码阅读版。令L=Q+T，logits[B,L,V]；log_softmax+gather和shift得[B,L-1]，不是词表维分布。response mask只保留T个动作；KL默认采样log比可负。外部score只加到最后有效token。GAE先用未白化优势算returns，再白化优势用于policy loss。
# ```
# def step(
#         self,
#         queries: List[torch.LongTensor],
#         responses: List[torch.LongTensor],
#         scores: List[torch.FloatTensor],
#         response_masks: Optional[List[torch.LongTensor]] = None,
#     ):
#
#     all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
#                     self.model,
#                     queries,
#                     responses,
#                     model_inputs,
# ```
#
# - `input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]`：拼接 queries & responses；
# - `ppo_trainer.batched_forward_pass`
#     - logprobs: $\log\pi_\theta(a_t|s_t)$
#
#
# ```
#
# # ppo_trainer.batched_forward_pass
# # logits.shape == (1, 27, 50257), values.shape == (1, 27)
# logits, _, values = model(**input_kwargs)
#
# # shift labels, next token predicition
# # lopprobs.shape == (1, 26)
# logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
# ```
#
# - 同样地对于 model_ref 再算一遍
#
#     ```
#     ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
#                         self.model if self.is_peft_model else self.ref_model,
#                         queries,
#                         responses,
#                         model_inputs,
#     ```
#
# - 计算 rewards
#
#     ```
#     rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
#     ```
#
#     - kl-penalty
#
#         $$
#         \text{KL}_{\text{penalty}} =\log\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{ref}}}(a_t|s_t)}=\log \pi_{\theta}(a_t|s_t) - \log \pi_{\theta_{\text{ref}}}(a_t|s_t)
#         $$
#      - `-self.kl_ctl.value * kl`
#      - reward is Preference Model (external RM) score + KL penalty
# - values, advantages, returns = self.compute_advantages(values, rewards, masks)
#     - `delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]`
#
#         $$
#         \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
#         $$
#     - gae lam (`lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam`)
#
#         $$
#         \hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}}
#         $$
#     - `returns = advantages + values`

# ===== 原单元 29 =====
# 展示原TRL PPO loss图。两支相加：policy clip loss与vf_coef乘value loss；旧0.8.6代码记录entropy，但没有默认把它作为第三个entropy bonus加进loss。
# 中文说明：构造展示对象；相对路径以notebook位置为基准。
Image(url='../../imgs/trl_ppo_loss.png', width=500)

# ===== 原单元 30 =====
# ratio=exp(new_logprob−old_logprob) 比较当前策略和 rollout 旧策略对同一动作的概率。用 log 差计算可绕开两个极小概率直接相除；最后的 exp 仍需留意浮点范围。
# - `ratio = torch.exp(logprobs - old_logprobs)`
#
# $$
# \exp(\log\pi_\theta-\log\pi_{\theta_{old}})=\exp\left(\log\frac{\pi_\theta}{\pi_{\theta_{old}}}\right)=\frac{\pi_\theta}{\pi_{\theta_{old}}}
# $$

# ===== 原单元 31 =====
# for逐一迭代统计字典的键并打印；keys()返回可迭代视图。某些字段只是诊断指标，不都参与autograd或loss。
# 中文说明：迭代字典键；这里只打印已有统计，不追加一次训练。
for key in train_stats.keys():
    # 中文说明：输出用于人工检查，不改变模型权重。
    print(key)

# ===== 原单元 32 =====
# 这是一个空白代码单元，保留在原编号位置，可用于追加计算。
