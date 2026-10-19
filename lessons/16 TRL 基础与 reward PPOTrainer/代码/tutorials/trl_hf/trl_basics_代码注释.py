# -*- coding: utf-8 -*-
# 第16讲原PPO代码阅读版，远程GPT2与重复query保留，修正验证另记。

# %% 0
# ## 单元 0（code）：图片展示导入
#
# IPython.display.Image类读取图片，不执行训练。
# 图片展示导入；原语句/历史输出保留。
# 导入图片展示类，不启动模型。
from IPython.display import Image

# %% 1
# ## 单元 1（markdown）：TRL基础章节
#
# SFT、RM训练、PPO是不同阶段，不能把Trainer名字当完成RLHF证据。
# ## basics

# %% 2
# ## 单元 2（markdown）：来源链接
#
# 原链接指main会变化，本讲参考固定TRL0.4.7；作者版本未锁定。
# - https://github.com/huggingface/trl/tree/main
# - https://github.com/huggingface/trl/tree/main/examples/notebooks

# %% 3
# ## 单元 3（code）：三阶段示意图
#
# SFT目标文本、RewardTrainer偏好对、PPO策略优化；固定reward1.0不等于训练RM。图来自上游，历史输出保留。
# 三阶段示意图；原语句/历史输出保留。
# 读取原配图；宽度仅展示，原相对路径需要正确工作目录。
Image('../imgs/trl_steps.png')

# %% 4
# ## 单元 4（markdown）：value head章节
#
# critic的value与外部reward model不同。
# ### value head

# %% 5
# ## 单元 5（markdown）：value输出接口
#
# 拼写foward指forward。包装返回(logits,loss,value)；hidden[B,T,H]经Linear(H,1)→[B,T,1]再squeeze→[B,T]，不是从词表概率生成reward。
# - `AutoModelForCausalLMWithValueHead`
#     - `foward()`
#         - `return (lm_logits, loss, value)`
#             - `lm_logits.shape`: (bs, seq_len, vocab_size)
#             - `value.shape`: (bs, seq_len)
# - `AutoModelForCausalLMWithValueHead` & `AutoModelForSeq2SeqLMWithValueHead`:
#     - A transformer model with an additional scalar output for each token which can be used as a value function in reinforcement learning.

# %% 6
# ## 单元 6（markdown）：PPO step入口章节
#
# step消费queries、纯responses、每样本scalar score，内部KL/GAE/loss/update。
# ### ppo_trainer.step()

# %% 7
# ## 单元 7（code）：偏好训练与策略图
#
# 打分训练与PPO是两阶段；本例没有实例化RM，不以图证明完成摘要任务。
# 偏好训练与策略图；原语句/历史输出保留。
# 读取原配图；宽度仅展示，原相对路径需要正确工作目录。
Image('../imgs/rm_ppo.png')

# %% 8
# ## 单元 8（code）：PPO算法图
#
# rollout保存old logprob，训练多轮clip policy/value；reference是KL锚点，不是old policy。width600仅展示。
# PPO算法图；原语句/历史输出保留。
# 读取原配图；宽度仅展示，原相对路径需要正确工作目录。
Image('../imgs/ppo_algo.png', width=600)

# %% 9
# ## 单元 9（markdown）：SFT示例章节
#
# 这里的Llama2/Guanaco命令属于SFT，不是后面的GPT2 PPO。
# ## examples 与示例代码（sft_trainer on llama2）

# %% 10
# ## 单元 10（markdown）：SFT安装与启动命令
#
# 三条shell逻辑命令：pip安装、clone源码、python执行示例。原未锁版本；model base7B、Guanaco、4bit/PEFT、micro4/K2单卡有效batch8。Bash用反斜杠续行；在PowerShell中执行时，需要改用其续行语法或写成一行。GPU运行环境见复现说明。
# ```
# # 安装库；原未锁版本。
# pip install trl
# # 取示例源码；需与库版本匹配。
# git clone https://github.com/lvwerra/trl
# # SFT示例base7B/Guanaco/4bit/PEFT/micro4/K2；非本PPO路径，Bash续行保留。
# python trl/examples/scripts/sft_trainer.py \
#     --model_name meta-llama/Llama-2-7b-hf \
#     --dataset_name timdettmers/openassistant-guanaco \
#     --load_in_4bit \
#     --use_peft \
#     --batch_size 4 \
#     --gradient_accumulation_steps 2
# ```

# %% 11
# ## 单元 11（markdown）：SFT默认优化器与scheduler
#
# AdamW/linear/warmup0属于前面SFT路径；本PPO0.4.7默认Adam，不能挪用。
# - 默认情况下
#     - optimize：AdamW
#     - lr scheduler：linear
#         - warm up ratio：0

# %% 12
# ## 单元 12（code）：SFT历史截图
#
# 展示课程保存的训练材料截图，结合图中字段阅读。
# SFT历史截图；原语句/历史输出保留。
# 读取原配图；宽度仅展示，原相对路径需要正确工作目录。
Image('../imgs/trl-sft.png')

# %% 13
# ## 单元 13（markdown）：PPO quick start章节
#
# 接下来GPT2/value/reference/固定score路径。
# ## quick start

# %% 14
# ## 单元 14（code）：rollout evaluation optimization图
#
# LM生成、外部score、active/ref与value优化分工；KL是惩罚偏离，value不是RM。width500为展示。
# rollout evaluation optimization图；原语句/历史输出保留。
# 读取原配图；宽度仅展示，原相对路径需要正确工作目录。
Image('../imgs/trl_overview.png', width=500)

# %% 15
# ## 单元 15（markdown）：三个模型名的含义
#
# active可更新，reference约束偏离，RM提供样本分数；本例RM缺失，另有value head。
# - model: active model
# - ref model: reference model
# - reward model: RM

# %% 16
# ## 单元 16（code）：导入框架、随机模块和TRL对象
#
# torch/numpy/random 提供计算与随机状态，GPT2Tokenizer 处理文本，PPOConfig 保存配置，PPOTrainer 组织生成与策略更新。
# 导入框架、随机模块和TRL对象；原语句/历史输出保留。
# 导入Tensor、自动求导和随机状态模块。
import torch
# NumPy模块别名np。
import numpy as np
# Python随机模块。
import random
# GPT2 slow tokenizer类，加载词表/merges。
from transformers import GPT2Tokenizer
# 因果LM工厂类，本例实际使用TRL包装。
from transformers import AutoModelForCausalLM
# value包装、PPO配置与训练器，固定0.4.7。
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# %% 17
# ## 单元 17（code）：设置随机种子函数
#
# 定义设种函数；传入42时设置Torch/CUDA/NumPy/Python的随机状态。cudnn.deterministic约束相关kernel；Trainer初始化还会使用PPOConfig的seed，默认是0。
# 设置随机种子函数；原语句/历史输出保留。
# 定义函数，调用时修改几类随机状态。
def setup_seed(seed):
    # 设置Torch随机状态。
    torch.manual_seed(seed)
    # 设置CUDA随机种子，不等于CUDA可用或运行GPU。
    torch.cuda.manual_seed_all(seed)
    # 设置NumPy随机状态。
    np.random.seed(seed)
    # 设置Python随机状态。
    random.seed(seed)
    # 约束相关cuDNN算法；不保证所有环境全确定性。
    torch.backends.cudnn.deterministic = True
# 调用函数；Trainer之后可能按config.seed0重新设种。
setup_seed(42)

# %% 18
# ## 单元 18（code）：active/ref包装和分词器加载
#
# 两次from_pretrained加载相同基础来源，但新增value head随机，不能说两个包装全状态相等。pad=eos50256不扩词表。CPU变体只把三处gpt2标识改本地随机小GPT2/真实词表目录。
# active/ref包装和分词器加载；原语句/历史输出保留。
# active基础LM+新增随机value head，CPU变体本地小模型。
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
# 独立reference包装；新增head不保证与active相等。
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
# 载入slow GPT2分词器；CPU变体使用相同词表导出的本地文件。
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# pad复用EOS50256，词表大小不变。
tokenizer.pad_token = tokenizer.eos_token

# %% 19
# ## 单元 19（code）：查看tokenizer
#
# 显示对象，不能据repr证明已训练语言模型。
# 查看tokenizer；原语句/历史输出保留。
# 查看分词器对象。
tokenizer

# %% 20
# ## 单元 20（code）：查看基础LM
#
# model.pretrained_model为GPT2LMHeadModel，输出词表logits与hidden；没有value head。
# 查看基础LM；原语句/历史输出保留。
# 查看基础LM，不含value head。
model.pretrained_model

# %% 21
# ## 单元 21（code）：查看value head
#
# Dropout+Linear(H,1)，GPT2的H=768时有769个参数，H=16时有17个；forward的具体操作见下方方法定义。
# 查看value head；原语句/历史输出保留。
# 查看critic的标量输出头。
model.v_head

# %% 22
# ## 单元 22（code）：PPO配置与Trainer初始化
#
# 字典**展开batch_size1，保留mini1/ppo_epochs4/seed0/lr1e-5等参考默认。位置参数按0.4.7签名；无dataset需自行送样本。active optimizer默认Adam，显式ref只作reference。
# PPO配置与Trainer初始化；原语句/历史输出保留。
# 2. initialize trainer
# 覆盖rollout batch为1，其余仍默认。
ppo_config = {"batch_size": 1}
# **展开字典为关键字参数，默认seed0/ppo_epochs4。
config = PPOConfig(**ppo_config)
# 真实初始化Trainer、optimizer和reference路径。
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# %% 23
# ## 单元 23（code）：查看active设备
#
# device属性返回设备，不移动模型。本次实际cpu，不当GPU证据。
# 查看active设备；原语句/历史输出保留。
# 设备属性，CPU小模型返回cpu。
model.pretrained_model.device

# %% 24
# ## 单元 24（code）：query字符串到tensor
#
# 末尾空格保留，encode return_tensors pt→int64[1,7]，to移动到active设备。固定GPT2 IDs1212/3329/314/1816/284/262/220。
# query字符串到tensor；原语句/历史输出保留。
# 原query末尾空格参与分词。
query_txt = "This morning I went to the "
# 编码int64[1,7]并移动到active设备。
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)
# 查看实际ID tensor。
query_tensor

# %% 25
# ## 单元 25（code）：生成kwargs字典
#
# min_length-1/top_k0.0是历史设置，兼容性按固定版本实测；top_p1、采样、pad=eos、新增20。字典创建不生成文本。
# 生成kwargs字典；原语句/历史输出保留。
# 构造生成关键字参数字典。
generation_kwargs = {
    # 历史设置，兼容性按固定4.30环境实测。
    "min_length": -1,
    # 原意关闭top-k，是否接受浮点0按固定版本验证。
    "top_k": 0.0,
    # 不额外截取nucleus概率质量。
    "top_p": 1.0,
    # 按随机概率采样，不是argmax。
    "do_sample": True,
    # padding采用EOS ID。
    "pad_token_id": tokenizer.eos_token_id,
    # 最多新增20个token，不含query。
    "max_new_tokens": 20,
}

# %% 26
# ## 单元 26（code）：真实rollout包含prompt
#
# 列表推导按batch行给list[tensor[7]]；**kwargs展开，return_prompt=True返回query+续写，通常[27]。这是后面重复prompt根源，CPU运行保留原参数。
# 真实rollout包含prompt；原语句/历史输出保留。
# 真实rollout列表；True返回query+续写，后面不能原样当纯response。
response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=True, **generation_kwargs)

# %% 27
# ## 单元 27（code）：decode完整返回
#
# response_tensor[0]取第一样本完整ID，decode返回含query的字符串；变量名response不证明它是纯续写。
# decode完整返回；原语句/历史输出保留。
# 第一返回样本ID解码为完整字符串。
response_txt = tokenizer.decode(response_tensor[0])
# 显示包含query的生成结果，便于对照输入与续写部分。
response_txt

# %% 28
# ## 单元 28（code）：固定score与真实PPO更新
#
# reward为list[0维float tensor]，值1.0，没有RM。step把query和response拼接，原7+27=34；第二份query被优化。一次step默认4个ppo_epochs，不是只一次optimizer更新。修正版另测试，不改原格。
# 固定score与真实PPO更新；原语句/历史输出保留。
# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
# got from a reward model
# 每样本0维float score，没有reward model forward。
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
# 原错误边界保留：7+27拼34，默认四epoch真实更新。
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

# %% 29
# ## 单元 29（code）：查看训练统计
#
# dict含KL、policy/value等及时间；不凭固定score/单次stats宣称偏好能力改善。历史未保存可读stats。
# 查看训练统计；原语句/历史输出保留。
# 统计dict，时间字段不可当数值等价依据。
train_stats

# %% 30
# ## 单元 30（markdown）：内部路线章节
#
# 下面是缩略源码概览，不是完整PPO实现。
# ## from scartch

# %% 31
# ## 单元 31（markdown）：PPO概览片段
#
# Markdown缩略拼接不同层级调用，缩进不构成可执行完整程序；按原文保留。batched_forward active/ref→compute_rewards→四轮minibatch→loss/更新；省略GAE与上下文，不能当完整算法或把kl loss当一个独立CE。
# ```
# # 这是缩略伪代码，不完整可执行；上下文、GAE和缩进层级被省略。
# # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
# train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
#
#     # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#     all_logprobs, _, values, masks = self.batched_forward_pass(self.model, queries, responses, model_inputs)
#
#     # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#     ref_logprobs, _, _, _ = self.batched_forward_pass(self.ref_model, queries, responses, model_inputs)
#
#     # kl loss
#     # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#     rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
#
#     # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#     for _ in range(self.config.ppo_epochs):
#         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#         logprobs, logits, vpreds, _ = self.batched_forward_pass(
#                         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                         self.model, batch["queries"], batch["responses"], model_inputs, return_logits=True
#                     # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                     )
#         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#         train_stats = self.train_minibatch(
#                         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                         batch["logprobs"],
#                         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                         batch["values"],
#                         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                         batch["rewards"],
#                         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                         logprobs,
#                         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                         logits,
#                         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                         vpreds,
#                         # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                         batch["masks"],
#                     # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#                     )
#             # active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。
#             loss_p, loss_v, train_stats = self.loss(old_logprobs, values, rewards, logits, vpreds, logprobs, mask)
# ```

# %% 32
# ## 单元 32（code）：重复prompt长度核对
#
# torch.cat沿唯一轴拼[7]+[27]→[34]，三次print展示；标准Trainer内部以batch输入。正确纯response20时总27。
# 重复prompt长度核对；原语句/历史输出保留。
# 沿一维拼接query与已含query的返回，重复输入。
input_ids = torch.cat([query_tensor[0], response_tensor[0]])
# 打印query长度7。
print(query_tensor[0].shape)
# 打印完整返回长度，通常27。
print(response_tensor[0].shape)
# 打印拼接总长，通常34。
print(input_ids.shape)

# %% 33
# ## 单元 33（code）：decode重复文本
#
# 拼接IDs解码可见query两遍，证明不只是打印重复，而是输入被重复。
# decode重复文本；原语句/历史输出保留。
# 解码可见query重复。
tokenizer.decode(input_ids)

# %% 34
# ## 单元 34（markdown）：基础模型前向章节
#
# 不带labels，不是SFT loss。
# ### base model forward

# %% 35
# ## 单元 35（code）：实际基础LM forward
#
# 一维IDs[34]输入，output_hidden_states True返回各层hidden和词表logits；此原打印不带显式batch轴。本次随机H16/V50257模型，不能照抄历史H768。
# 实际基础LM forward；原语句/历史输出保留。
# 真实基础LM forward，提供hidden但没有labels。
base_model_output = model.pretrained_model(input_ids, output_hidden_states=True)

# %% 36
# ## 单元 36（code）：取末层hidden与logits
#
# hidden_states[-1]最后层，[34,H]；logits[34,V]。原注释的loss访问没有执行；未传labels通常loss None。
# 取末层hidden与logits；原语句/历史输出保留。
# 取最后一层hidden，[T,H]。
last_hidden_state = base_model_output.hidden_states[-1]
# 显示T/H，CPU小模型H16与历史H768分开。
print(last_hidden_state.shape)
# 取词表分数[T,50257]。
lm_logits = base_model_output.logits
# 打印logits的词表维度；value由独立的value head输出。
print(lm_logits.shape)
# loss = base_model_output.loss
# print(loss.shape)

# %% 37
# ## 单元 37（markdown）：value计算章节
#
# 从hidden映射标量，不从词表logits或RM打分。
# ### value head

# %% 38
# ## 单元 38（code）：再次查看value模块
#
# 同一个v_head引用，不新建critic。
# 再次查看value模块；原语句/历史输出保留。
# 查看critic的标量输出头。
model.v_head

# %% 39
# ## 单元 39（code）：无梯度value forward
#
# with上下文禁止记录梯度，不自动关Dropout；Linear[34,H]→[34,1]，squeeze(-1)→[34]。历史H768说明保留，CPU实际H16另记录。
# 无梯度value forward；原语句/历史输出保留。
# 显示T/H，CPU小模型H16与历史H768分开。
print(last_hidden_state.shape)
# 上下文关闭梯度记录，不自动关闭dropout。
with torch.no_grad():
    # (34, 768) * (768, 1) => (34, 1)
    # Linear(H,1)后只去掉最后单维，[T]。
    value = model.v_head(last_hidden_state).squeeze(-1)
    # 打印每位置value shape。
    print(value.shape)

# %% 40
# ## 单元 40（code）：查看每位置value
#
# tensor[34]可正可负，不是34个完整回答reward或概率。
# 查看每位置value；原语句/历史输出保留。
# 查看标量tensor序列，不是RM评分。
value

# %% 41
# ## 单元 41（code）：空代码格
#
# 原空格保留，有注释说明但不编造计算。
# 空代码格；原语句/历史输出保留。
