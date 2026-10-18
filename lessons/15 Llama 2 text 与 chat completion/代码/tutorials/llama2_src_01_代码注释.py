# -*- coding: utf-8 -*-
# 第15讲 llama2_src_01 Python阅读版；原语句保留。

# %% 0
# 导入图片展示类
# IPython.display.Image是展示对象，不是模型；后面从相对路径读取PNG。
# 导入IPython图片展示类，不加载模型。
from IPython.display import Image

# %% 1
# 基础章节
# 先分清原始分片、模型参数与推理入口。
# ## basics

# %% 2
# 仓库、模型规格与head_dim
# main对应llama2是当时描述，本次固定llama_v2 commit。7B dim4096/n_heads32，每头128；multiple_of256用于MLP维度取整。Markdown中的self.head_dim整数除法是源码片段，需要args/self上下文；下载链接是资料，不执行下载或申请。
# - 源码地址：https://github.com/facebookresearch/llama.git
#     - 默认 main 分支为 llama2
#     - llama_v1 分支为 llama1
#     - weights 申请地址（非常容易）：
#         - https://ai.meta.com/resources/models-and-libraries/llama-downloads/
# - 参数
#     - 7B
#         - dim: 4096
#         - multiple_of: 256 (SwiGLU)
#         - n_layers: 32
#         - n_heads: 32
#
# ```
# # 4096 / 32 == 2^12/2^5 == 2^7 == 128
# # 源码介绍片段；所需对象/类型上下文和每行用途见前置说明。
# self.head_dim = args.dim // args.n_heads
# ```
#

# %% 3
# 展示早期权重目录
# Image读取../imgs/llama_weights.png，width400是展示宽度，不改模型。原输出为历史图片，真实重做从本讲代码/tutorials工作目录执行。
# 相对本讲代码/tutorials读取图片，400是显示宽度。
Image('../imgs/llama_weights.png', width=400)

# %% 4
# 展示Llama2权重目录
# base/chat与7B/13B/70B、pth分片和params.json属于原始格式；图片总大小不等单模型显存。width300只控制展示。
# 展示原始Llama2权重目录，300不改变文件或权重。
Image('../imgs/llama2-weights.png', width=300)

# %% 5
# 两个completion入口
# text/chat共享Llama.generate，差别在权重和输入组织，不是两个独立解码算法。
# ## text-completion & chat-completion

# %% 6
# text入口
# base续写普通字符串，BOS=True/EOS=False；echo默认False，与HF pipeline完整文本默认不同。
# ### text-completion

# %% 7
# 7B与13B文本启动命令
# 两条torchrun逻辑命令；nproc1/2对应MP分片，不是batch。ckpt_dir是原始pth目录，tokenizer_path是独立词表。max_seq_len128/max_batch_size4是buffer/cache上限；示例main另默认max_gen_len64。反斜杠是Bash续行，PowerShell应换语法；原权重和NCCL路径本次未执行。
# ```
# # torchrun启动模型并行进程；nproc与原始checkpoint分片一致，不是batch。
# # ckpt_dir指定base或chat权重；tokenizer_path为单独词表。
# # max_seq_len/max_batch_size限制序列与batch，反斜杠是Bash续行。
# torchrun --nproc_per_node 1 example_text_completion.py \
# # 源码介绍片段；所需对象/类型上下文和每行用途见前置说明。
#     --ckpt_dir ./weights/llama-2-7b/ \
#     --tokenizer_path ./weights/tokenizer.model \
#     --max_seq_len 128 --max_batch_size 4
#
# # torchrun启动模型并行进程；nproc与原始checkpoint分片一致，不是batch。
# # ckpt_dir指定base或chat权重；tokenizer_path为单独词表。
# # max_seq_len/max_batch_size限制序列与batch，反斜杠是Bash续行。
# torchrun --nproc_per_node 2 example_text_completion.py \
#     --ckpt_dir ./weights/llama-2-13b/ \
#     --tokenizer_path ./weights/tokenizer.model \
#     --max_seq_len 128 --max_batch_size 4
# ```

# %% 8
# chat入口
# messages先经角色检查、system合并和INST模板，再进入相同生成器。
# ### chat-completion

# %% 9
# 7B与13B聊天启动命令
# 两个chat checkpoint目录，nproc1/2对应原始分片。max_seq_len512、max_batch_size8；官方例子六个dialog，batch4会超限。B是对话条数，不是消息数或GPU数；不把CPU随机验证当GPU命令执行。
# ```
# # torchrun启动模型并行进程；nproc与原始checkpoint分片一致，不是batch。
# # ckpt_dir指定base或chat权重；tokenizer_path为单独词表。
# # max_seq_len/max_batch_size限制序列与batch，反斜杠是Bash续行。
# torchrun --nproc_per_node 1 example_chat_completion.py \
# # 源码介绍片段；所需对象/类型上下文和每行用途见前置说明。
#     --ckpt_dir ./weights/llama-2-7b-chat/ \
#     --tokenizer_path ./weights/tokenizer.model \
#     --max_seq_len 512 --max_batch_size 8
#
# # torchrun启动模型并行进程；nproc与原始checkpoint分片一致，不是batch。
# # ckpt_dir指定base或chat权重；tokenizer_path为单独词表。
# # max_seq_len/max_batch_size限制序列与batch，反斜杠是Bash续行。
# torchrun --nproc_per_node 2 example_chat_completion.py \
#     --ckpt_dir ./weights/llama-2-13b-chat/ \
#     --tokenizer_path ./weights/tokenizer.model \
#     --max_seq_len 512 --max_batch_size 8
# ```

# %% 10
# 辅助工具章节
# 终端工作流与模型算法分开。
# ## misc

# %% 11
# 终端资料链接
# 外部终端使用文章是辅助链接，本讲没有复制其操作或执行命令。
# - https://blog.csdn.net/weixin_44826203/article/details/129255185
