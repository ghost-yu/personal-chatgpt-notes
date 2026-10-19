# 第29讲 Python 阅读版：原代码详注，原Markdown作为注释保留。
# 原5/7格会加载远程7B/8B权重；日常直接阅读。验证请使用复现/scripts/verify_delivery.py。

# %% 原单元 0 · code
# 导入Transformers、PyTorch、自动因果语言模型工厂与Image。os.environ两行会修改当前进程代理，127.0.0.1:7890只适用于讲师环境；不是加载模型的必需设置。本次在独立验证进程执行并恢复原代理。
# 导入后续使用的模块或对象。
import transformers
# 导入后续使用的模块或对象。
import torch
# 导入后续使用的模块或对象。
from transformers import AutoModelForCausalLM

# 导入后续使用的模块或对象。
from IPython.display import Image
# 导入后续使用的模块或对象。
import os
# 修改进程环境变量；讲师本地代理地址不通用。
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# 修改进程环境变量；讲师本地代理地址不通用。
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


# %% 原单元 1 · markdown
# 区分Meta原生权重与Hugging Face转换格式；同一个模型的不同键名/分片格式不意味着模型结构不同。
# ## llama meta vs. hf

# %% 原单元 2 · markdown
# Markdown三条加载语句：torch.load返回checkpoint字典，Transformer(model_args)创建架构，load_state_dict复制权重并返回missing/unexpected keys；strict=False不等于忽略同名尺寸不匹配。索引JSON把张量名映射到文件，不是device_map。原文存在乱码，原件保留；新增解释按源码语义校正。
#
# - llama2-7b
#     - https://huggingface.co/meta-llama/Llama-2-7b/tree/main
#         - `checkpoint = torch.load(ckpt_path, map_location="cpu")`
#         - `model = Transformer(model_args)`
#         - `model.load_state_dict(checkpoint, strict=False)`
#         - 自动地走 `bfloat16`（如果支持的话）
#     - https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main
#         - `pytorch_model.bin.index.json`
# - https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md

# %% 原单元 3 · markdown
# 比较对象是早期Llama2-7b-hf和Meta-Llama-3-8B基础模型，不是Chat/Instruct，也不是后来的3.1/3.2。
# ## llama2-7b vs. llama3-8b (hf)

# %% 原单元 4 · code
# llama2_id和llama3_id都是str类型Hub标识符，不是已经加载的模型；后续from_pretrained才解析配置与checkpoint。
# Llama2 Hub标识字符串，不是模型对象。
llama2_id = 'meta-llama/Llama-2-7b-hf'
# Llama3 Hub标识字符串，不是模型对象。
llama3_id = "meta-llama/Meta-Llama-3-8B"


# %% 原单元 5 · code
# AutoModelForCausalLM.from_pretrained类工厂按config选择LlamaForCausalLM；torch_dtype要求BF16权重（每元素2 bytes），device_map=auto让Accelerate分配设备。14GB只是7×10^9×2的粗估，不含KV/激活/临时峰值。配套算例分别用meta模型查看参数形状，用本地随机CPU权重演示加载。
# bfloat16: 2 byte
# 7b => 14GB
# 真正读取配置与checkpoint；原语句要求远程模型访问和足够内存。
llama2 = AutoModelForCausalLM.from_pretrained(llama2_id, torch_dtype=torch.bfloat16, device_map='auto')


# %% 原单元 6 · code
# 原格只有注释，不会打印任何参数。named_parameters遍历(name,Parameter)，para.device是所在设备；不要把原注释误称实际多卡输出。
# for name, para in llama2.named_parameters():
#     print(name, para.device)


# %% 原单元 7 · code
# 第二次加载Llama3；8B的16GB也只是近似权重体积。两次赋值让两模型同时驻留，不能把16GB当同时装入两者的内存。真实小模型只在原4格改成本地模型路径，5/7格保留BF16与device_map=auto参数；补充训练才转FP32。
# 8b => 16GB
# 真正读取配置与checkpoint；原语句要求远程模型访问和足够内存。
llama3 = AutoModelForCausalLM.from_pretrained(llama3_id, torch_dtype=torch.bfloat16, device_map='auto')


# %% 原单元 8 · markdown
# 自动设备映射标题；它与数据并行/张量并行/训练optimizer分片是不同机制。
# ### `device_map='auto'`

# %% 原单元 9 · markdown
# get_balanced_memory摘录计算各设备可用预算，infer_auto_device_map另负责分配；不是这个if块直接搬运权重。low_zero只在balanced_low_0为True，auto并不承诺每卡严格一半。上下文变量未定义，摘录不是独立可运行脚本。
# - 自动地模型并行；
#
# ```
# # sequential分配策略不走此处平衡预算。
# if device_map != "sequential":
#     # Compute a `max_memory` dictionary for [`infer_auto_device_map`] that will balance the use of each available GPU.
#     # 计算设备内存预算，不直接生成最终device_map。
#     max_memory = get_balanced_memory(
#         model,
#         dtype=target_dtype,
#         # 只有balanced_low_0策略才特意给GPU0少放权重。
#         low_zero=(device_map == "balanced_low_0"),
#         max_memory=max_memory,
#         **device_map_kwargs,
#     )
# ```

# %% 原单元 10 · markdown
# 可训练参数统计标题：requires_grad为True表示autograd资格，不表示此刻已经训练或optimizer已包含该参数。
# ### learnable parameters

# %% 原单元 11 · code
# 定义普通Python函数，不是模型方法。model是nn.Module；named_parameters默认去重，numel返回int元素数，requires_grad是bool。两个int累计器原地更新，函数只print、隐式返回None，不会改变模型权重或开关；空模型分母为0，另记录反例。
# 定义或调用参数统计函数，输入是模型对象。
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    # 初始化可训练元素数的int累计器。
    trainable_params = 0
    # 初始化全部元素数的int累计器。
    all_params = 0
    # 遍历去重的参数对象，名字与Parameter配成一对。
    for _, param in model.named_parameters():
        # 统计元素个数，不是字节数；字节还需乘dtype大小。
        all_params += param.numel()
        # 只检查是否允许求梯度，不会执行训练。
        if param.requires_grad:
            # 统计元素个数，不是字节数；字节还需乘dtype大小。
            trainable_params += param.numel()
    # 格式化打印统计结果；函数仍隐式返回None。
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")


# %% 原单元 12 · code
# 对Llama2实际模型对象调用统计函数。历史6738415616是精确元素数；本次完整meta架构能复核这个数，但meta没有权重数值，不能用于真实forward。
# 定义或调用参数统计函数，输入是模型对象。
print_trainable_parameters(llama2)


# %% 原单元 13 · code
# 同一函数用于Llama3；历史8030261248与8B近似命名区分。100%只表示所有参数requires_grad=True，不证明训练发生。
# 定义或调用参数统计函数，输入是模型对象。
print_trainable_parameters(llama3)


# %% 原单元 14 · code
# 末尾表达式在Jupyter触发模型repr；展示Embedding、32个DecoderLayer、norm和lm_head。文本repr不会执行forward。普通.py直接执行此表达式不会自动显示，需要print。
# Jupyter展示对象结构；不是前向或文本生成。
llama2


# %% 原单元 15 · code
# Llama3结构：Embedding[128256,4096]、Q/O4096、K/V1024、MLP14336；K/V总投影宽不同，但每个head仍是128，所以QK点积合法。
# Jupyter展示对象结构；不是前向或文本生成。
llama3


# %% 原单元 16 · code
# 属性链定位输入embedding的weight，numel按V×D计数；f-string的:,是千位分隔，返回两个str组成的tuple，不是Tensor。131072000与525336576只算输入表，未算独立lm_head。
# 统计元素个数，不是字节数；字节还需乘dtype大小。
f'{llama2.model.embed_tokens.weight.numel():,}', f'{llama3.model.embed_tokens.weight.numel():,}'


# %% 原单元 17 · code
# Python / 返回float；11008/4096=2.6875，14336/4096=3.5。比较的是MLP中间维与D的比值，不是模型大小或head数。
# 中间层宽度除以模型宽度，得到两个float比值。
11008/4096, 14336/4096


# %% 原单元 18 · markdown
# 完整attention公式：Q/K每head宽必须同为dh，V宽可不同（本实现也为dh）；softmax沿key位置j归一化。原式省略batch/head/causal mask；GQA先按g(h)=floor(h/R)选择共享K/V，再逐head做点积。
# - attention mechanism （Transformer 最特色的）
#     - $X\in\mathbb R^{\ell\times d}$
#     - $W_k\in\mathbb R^{d\times d_k},W_q\in\mathbb R^{d\times d_k},W_v\in\mathbb R^{d\times d_v}$
#     - $Q=XW_q\in\mathbb R^{\ell\times d_k}, K=XW_k\in\mathbb R^{\ell\times d_k}, V=XW_v\in\mathbb R^{\ell\times d_v}$
#
# $$
# \text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
# $$
#
# $$
# A_{ij}=\frac{\exp(\frac{q^T_ik_j}{\sqrt{d_k}})}{\sum_{j'}\exp(\frac{q^T_ik_{j'}}{\sqrt{d_k}})}
# $$

# %% 原单元 19 · code
# Image只构造远程HTML图对象，本次未验证外站图片内容。quality/efficiency图不能直接证明本机速度或所有MHA都质量最高；本讲用分组张量与实际梯度说明机制。
# quality & computation efficiency
# 构造远程图展示对象；没有验证图片内容。
Image(url='https://pbs.twimg.com/media/FzjhZk5X0AYAs_-?format=jpg&name=4096x4096', width=500)


# %% 原单元 20 · code
# 4096/32=128.0是每query head的宽度；GQA的KV head减少不改变它。公式中的dk对应单head维，不是K投影拼接总宽1024。
# 模型宽度除以query头数，得到单头128维。
(4096/32)


# %% 原单元 21 · markdown
# Markdown完整摘录需要逐句读：nn.Linear(in,out)实际weight[out,in]；Q为H×dh，K/V为G×dh；H//G为R，32/8=4个query heads共享一份KV。MLP gate/up先升维，SiLU(gate)*up是逐元素乘法，再down投影；Meta w1/w3/w2分别对应gate/up/down。self表示实例，config字段来自LlamaConfig，这里不是独立类定义。
# ```
# # self attn
# # Q投影：D→H×dh；Linear权重存储为[out,in]。
# self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
# # GQA，llama2: 32*(4096/32) = 4096
# # GQA，llama3: 8*(4096/32) = 1024
# # K投影：D→G×dh；GQA只保留G个KV头。
# self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
# # V投影：D→G×dh；与K共用头数，不共享参数矩阵。
# self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
# # 拼接H个query头输出后映射回D。
# self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
#
# # GQA
# # llama2: 32/32
# # llama3: 32/8 = 4, 4对1
# # R=H//G，每份KV服务R个连续query头。
# self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#
# # mlp
# # gate分支D→F；前向中施加SiLU激活。
# self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
# # up分支D→F，随后与激活后的gate逐元素相乘。
# self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
# # down分支F→D，恢复残差流宽度。
# self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
#
# # mlp forward
# # hf
# # gate分支D→F；前向中施加SiLU激活。
# down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
# # meta
# # Meta w1/w3/w2对应HF gate/up/down。
# self.w2(F.silu(self.w1(x)) * self.w3(x))
# ```

# %% 原单元 22 · markdown
# 参数表heads列是query head数，不是KV head数；Llama2家族不能一概称MHA，7B/13B与70B不同。两代本例head_dim都是128。Llama3 K/V单个矩阵参数剩1/4即减少75%，但整个attention剩5/8，不能说整个模型减少75%。
# |     **model**    | **heads** | **layers** | **dim** | **head_dim** |
# |------------------|-----------|------------|---------|--------------|
# | llama2-7b        | 32        | 32         | 4096    | 4096/32      |
# | llama2-13b       | 40        | 40         | 5120    | 5120/40      |
# | llama2-70b       | 64        | 80         | 8192    | 8192/64      |
#
# |     **model**    | **heads** | **layers** | **dim** | **head_dim** |
# |------------------|-----------|------------|---------|--------------|
# | llama3-8b        | 32        | 32         | 4096    | 4096/32      |
#
# -  vocab_size (Embedding):
#     - llama2: 32000
#     - llama3: 128256
# - GQA (k_proj, v_proj)
#     - head_dim: hidden_size/num_heads
#         - llama2: 4096/32 = 128
#         - llama3: 4096/32 = 128
#     - llama2: **32***(4096/32) = 4096
#     - llama3: 8*128 = 1024 (k_proj, v_proj 可以看到一个 learnable parameters 的一个降低)

# %% 原单元 23 · code
# 原最后一格为空，保留空格和历史元数据，不编造其执行结果。
