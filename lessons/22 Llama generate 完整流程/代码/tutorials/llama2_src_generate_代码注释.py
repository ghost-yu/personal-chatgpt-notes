# -*- coding: utf-8 -*-
# 原课程四格分词示例；完整生成方法见generate核心源码_代码注释.py。

# %% 原代码单元 0
# 输入文本是str，还不是token Tensor。
text = "I believe the meaning of life is"

# %% 原代码单元 1
# 读取真实SentencePiece模型，相对路径取决于工作目录。
from sentencepiece import SentencePieceProcessor
sp_model = SentencePieceProcessor(model_file='./tokenizer.model')

# %% 原代码单元 2
# 打印BOS/EOS/PAD及词表大小，PAD=-1只作占位。
print(sp_model.bos_id())
print(sp_model.eos_id())
print(sp_model.pad_id())
print(sp_model.vocab_size())

# %% 原代码单元 3
# *展开编码ID；这是单条列表，generate需要再包成二维batch。
prompt_tokens = [sp_model.bos_id(), *sp_model.encode(text)]
prompt_tokens
