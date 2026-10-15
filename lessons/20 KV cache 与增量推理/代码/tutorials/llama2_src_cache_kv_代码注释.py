# -*- coding: utf-8 -*-
# 第20讲原代码的中文注释阅读版；原加载及benchmark参数保持不变。
# 本文件含代理设置与远程模型加载，完整原版未作为CPU教学验证运行。

# %% 原代码单元 0
# Image 是 notebook 展示对象，后面用它读取配套图。
from IPython.display import Image

# %% 原代码单元 5
# 仅展示原自回归动画；width 是显示像素宽度。
Image('../imgs/gpt-2-autoregression-2.gif', width=700)

# %% 原代码单元 6
# 这是历史价格截图，不是当前报价或本次性能实验。
Image('../imgs/gpt4-price.png', width=600)

# %% 原代码单元 8
# 图中 x 经三组可训练投影分别得到 Q、K、V。
Image('../imgs/qkv.png', width=600)

# %% 原代码单元 10
# S=1 的 decode 每 head 新增一行 Q/K/V，旧 KV 可以复用。
Image('../imgs/decoding-qkv.png', width=600)

# %% 原代码单元 12
# KV 的 token 轴随有效长度增加，缓存不保存整张 attention score。
Image('../imgs/kvmatrices.png', width=600)

# %% 原代码单元 13
# 首次 prefill 建缓存，后续 decode 只新增当前 token 的 K/V。
Image('../imgs/kv-cache.gif', width=900)

# %% 原代码单元 16
# 导入真实 SentencePiece 分词器类。
from sentencepiece import SentencePieceProcessor
# 相对路径要求工作目录为代码/tutorials；加载的是词表模型。
sp_model = SentencePieceProcessor(model_file='./tokenizer.model')

# %% 原代码单元 17
# 输入是 Python 字符串，尚未成为 token Tensor。
prompt = 'I believe the meaning of life is'
# * 展开7个普通整数ID，前置BOS得到长度8的 list[int]。
[sp_model.bos_id(), *sp_model.encode(prompt)]

# %% 原代码单元 20
# os 提供环境变量映射；本次验证跳过整格。
import os
# 设置本进程HTTP代理；此端口必须已有服务。
os.environ["http_proxy"] = "http://127.0.0.1:7890"
# HTTPS代理同理；这不是缓存算法的一部分。
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# %% 原代码单元 21
# np用于统计秒数；不是模型后端。
import numpy as np
# time用于墙钟计时，CUDA时还要另考虑异步执行。
import time
# torch提供张量和设备检测。
import torch
# Auto类按配置选择具体Tokenizer/因果LM。
from transformers import AutoModelForCausalLM, AutoTokenizer

# 条件表达式选择当前torch可用的设备。
device = "cuda" if torch.cuda.is_available() else "cpu"
# 原版要求GPT-2 tokenizer；验证只将此字符串换为本地路径。
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# 原版加载预训练GPT-2；验证明确使用随机小GPT2并迁到CPU。
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# %% 原代码单元 22
# 在Jupyter显示tokenizer repr；普通脚本不会自动展示。
tokenizer

# %% 原代码单元 23
# 依次测试开缓存、关缓存；开关影响generate的past_key_values复用。
for use_cache in (True, False):
    # 每种模式重新创建浮点秒数列表。
    times = []
    # 10次重复；教学验证仅将这个常量改成2。
    for _ in range(10):  # measuring 10 generations
        # 计时从分词与输入迁移之前开始。
        start = time.time()
        # ** 将input_ids等字段解包成generate关键字参数。
        model.generate(**tokenizer("What is KV caching?", return_tensors="pt").to(device),
                       # True复用旧KV；False每步重新处理完整前缀。
                       use_cache=use_cache,
                       # 最多新增1000个token，教学验证仅将此常量改成16。
                       max_new_tokens=1000)
        # 记录一次调用总墙钟时间，原代码未保存生成结果。
        times.append(time.time() - start)
    # 均值与总体标准差不是置信区间。
    mu = round(np.mean(times), 3)
    # ddof默认0；保留三位小数仅影响打印。
    std = round(np.std(times), 3)
    # f-string中条件表达式选择with/without标签。
    print(f"{'with' if use_cache else 'without'} KV caching: {mu} +- {std} seconds")

# %% 原代码单元 24
