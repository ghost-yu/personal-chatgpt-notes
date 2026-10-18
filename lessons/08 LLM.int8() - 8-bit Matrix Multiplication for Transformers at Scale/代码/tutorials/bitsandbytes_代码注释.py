# -*- coding: utf-8 -*-
# 第08讲 bitsandbytes配置 Python阅读版；原执行语句未修改。
# 按# %%分格阅读；最后OPT6.7B段可能下载权重，原代理格会修改环境。
# 实际CPU验证范围和跳过单元见复现/README.md。

# %% [markdown]
# ## 单元 0（markdown）：量化配置标题
#
# config是配置入口；该辅助notebook只有3格，不是bitsandbytes底层实现。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：

# %% [markdown] 原单元 0
# ## config

# %% [markdown]
# ## 单元 1（code）：导入BitsAndBytesConfig
#
# transformers中的配置类保存4/8-bit加载选项；导入不会构建配置、替换模型或下载权重。构造器默认load_in_8bit=False、load_in_4bit=False，llm_int8_threshold=6.0。主notebook沿用旧布尔参数，参考新写法见主解释。本次真实导入并构造8bit配置，检查相互排斥的ValueError，不等于运行量化内核。

# %% 原单元 1
# 导入BitsAndBytesConfig：详见本格中文说明，原执行语句保持不变。
# 导入配置类，本格未构造配置或调用量化加载。
from transformers import BitsAndBytesConfig

# %% [markdown]
# ## 单元 2（markdown）：4-bit与8-bit互斥
#
# 原Markdown提示load_in_8bit/load_in_4bit二选一为true。更准确是两者不能同时True，可以都False（不开这两种量化）；配置校验不是模型能力证明。实际BitsAndBytesConfig(load_in_8bit=True,load_in_4bit=True)应抛ValueError；开启4bit的配置还可能查询bitsandbytes安装版本。本次不安装4bit后端。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# - `load_in_8bit`/`load_in_4bit`: 最多一个为 true

# %% [markdown] 原单元 2
# - `load_in_8bit`/`load_in_4bit`: 最多一个为 true
