# 第 01 讲代码注释阅读版
# 来源：chunhuizhang/personal_chatgpt；固定提交 7fbc29824171fd07dc2c30fe922e61ebc018ad1d。
# 建议在配套 notebook 中运行；本文件仅导入 Image 并创建展示对象。
# Markdown 和六条 shell 命令均以 Python 注释保留，不会执行模型启动。

# %% [markdown]
# # 第 01 讲中文说明
#
# 插入的说明为新编内容；原输出是历史记录，不表示本次执行。
# 工作目录应为本讲代码/tutorials；原版、图片和 Python 阅读版均随仓库提供。

# %% [markdown]
# ## 单元 0：导入 Image
#
# 【必须会】原代码是 `from IPython.display import Image`。它从IPython的display模块导入Image类，让当前会话能够使用这个名字。`from ... import ...` 不等于创建模型，也不会读取图片或加载权重。
#
# Image是类，不是torch.Tensor；下一格实际调用它才产生对象。这里没有B、T、D、dtype或模型shape，没有可学习参数。状态变化只是在notebook命名空间绑定Image。若跳过本格，单元4会因为Image未定义而失败。
#
# 代码中的中文注释会直接写明“导入notebook图片展示类；仅建立名字，尚未加载LLaMA或图片”。无需给这一行虚构attention公式对应关系；它是展示辅助代码，Paper ↔ Code模型映射为不适用。

# %%
# 【必须会】这行准备 notebook 的图片展示工具，后面的单元 4 会用到。
# from 包.模块 import 名称：从 IPython.display 模块导入 Image 类。
# Image 是类；本行只把这个类绑定到当前内核的名字 Image，还没有创建实例。
# import 会加载展示模块，但不会读取 llama_weights.png、下载模型或分配模型张量。
# 本单元没有输入参数和返回表达式；状态变化是命名空间中新增 Image。
# 先运行本格，再运行单元 4；否则 Image 未定义会触发 NameError。
# 它是展示辅助代码，未实现论文中的 Attention、RMSNorm 或训练损失。
from IPython.display import Image

# %% [markdown]
# ## 单元 1：summary 标题
#
# 【可以暂时不掌握】这是Markdown标题，不是Python语句。`##` 控制渲染层级，不创建变量、不调用函数。它为后面家族介绍分节；它没有模型运行结果。

# %% [markdown]
# ## summary

# %% [markdown]
# ## 单元 2：模型家族和历史链接
#
# 【必须会】本格是Markdown资料。LLaMA指基础预训练模型；Alpaca在LLaMA基础上进行指令微调；Vicuna在LLaMA基础上进行对话微调。原来的链式箭头可用于表示教程介绍顺序，不能据此要求Vicuna一定从Alpaca权重继续训练。
#
# `self instruct` 指数据构造思想，SFT才是以监督样本更新参数。prompt/answer为训练样本的输入/目标，不是本notebook定义的Python变量。ShareGPT对话也不是checkpoint文件。
#
# 链接指向外部仓库与发布说明，打开链接不会自动安装依赖。原文字段有拼写和历史措辞，例如`text-davini-003`，应按历史的`text-davinci-003`类接口类比理解，不能推断所有基础模型有相同能力。关于权重许可的表述是2023年材料，不作为后续所有版本的现行使用结论。本篇未改写原Markdown，只在其前添加校正说明。

# %% [markdown]
# - llama => alpaca => vicuna
#     - llama: pretrained model, 作用类似于 text-davini-003（gpt3）
#         - https://github.com/facebookresearch/llama
#     - self instruct (fine tune)
#         - alpaca：https://github.com/tatsu-lab/stanford_alpaca
#             - prompt, answer：from chatgpt
#         - vicuna：https://github.com/lm-sys/FastChat
#             - from shareGPT
#             - long conversation/context window
# - llama 开源了又没完全开源
#     - 开源的是代码
#     - weights 没有完全开源，不可以商用；
#     - vicuna weights：https://github.com/lm-sys/FastChat
# - https://lmsys.org/blog/2023-03-30-vicuna/

# %% [markdown]
# ## 单元 3：llama weights 标题
#
# 【可以暂时不掌握】同样为Markdown。引出checkpoint目录截图，不产生weights变量、不分配GPU显存。下一格读的是截图，不是截图显示的任何权重文件。

# %% [markdown]
# ## llama weights

# %% [markdown]
# ## 单元 4：显示权重目录截图
#
# 【必须会】原语句是 `Image('../imgs/llama_weights.png', width=300)`。调用Image构造器，位置实参为str图片路径，关键字实参width为int展示宽度。返回IPython.display.Image对象；notebook会渲染最后表达式，普通Python脚本不会自动显示它。
#
# 相对路径根据内核当前工作目录解析：如果工作目录为教程仓库tutorials，`..`先退到仓库根，再进入imgs。它不天然根据`.ipynb`文件所在目录解析。本地注释版保存在配套的`代码/tutorials`下，邻近`代码/imgs`包含原图片，因此在该tutorials工作目录可运行原语句。
#
# 数据流为 `路径字符串 → PNG文件字节 → Image对象 → 展示输出`。width=300并不修改PNG原始像素，也不是embedding_dim；本次实际读取333093字节，PNG签名有效，width属性为300。没有`[B,C,H,W]`模型输入张量，也没有视觉编码器。
#
# 截图中的7B/13B等为参数规模目录；`.pth`是权重分片，params.json为结构配置，tokenizer.model为分词模型。截图不能证明本机持有这些文件。图中的30B与命令的33B还有历史命名差别，需要核对真实目录；不要根据图片强行改写路径。
#
# 中文代码注释会明确相对路径、300的含义和返回对象，不把显示操作注释成“加载LLaMA”。

# %%
# 【必须会】本行展示权重目录的 PNG 截图，便于认识 checkpoint 文件布局。
# Image(...) 调用上一格导入的类，构造 IPython.display.Image 实例。
# 第一个位置实参是 str 路径；在本例中它被当作本地文件读取。
# 相对路径基于内核当前工作目录：先由 .. 返回上一级，再进入 imgs。
# 请从本讲的代码/tutorials 目录启动内核；配套图片位于代码/imgs。
# 若从仓库根目录执行，此路径可能指向错误位置，并触发 FileNotFoundError。
# width=300 是关键字实参，类型为 int，表示前端展示宽度为 300 像素。
# 该参数不重新采样 PNG，也不是 embedding_dim、batch_size 或模型参数量。
# 数据流：路径字符串 -> PNG 文件字节 -> Image 对象 -> notebook 展示输出。
# 默认 embed=None；本地文件情形会嵌入图片数据，文件内容保存在对象的 data 属性。
# 这里没有 torch.Tensor，也没有 [B,C,H,W] 或 [B,T,D] 的模型 shape。
# 返回对象没有赋给变量；IPython 会显示本格最后一个表达式的展示表示。
# 普通 Python 脚本运行这行只创建对象；要显示需额外使用 display 或 notebook。
# 截图中的 .pth、params.json、tokenizer.model 只是图片内容，并未被本行加载。
# 原格保留的图片输出是作者历史记录；本次重新读取的结果另记在复现文档。
Image('../imgs/llama_weights.png', width=300)

# %% [markdown]
# ## 单元 5：inference/sample/generator 标题
#
# 【建议会】它是Markdown标题，用于引出外部example.py启动命令。inference为使用既有权重做前向；sample是从概率分布选择token；generator为生成循环。此处未定义名叫generator的Python对象。
#
# 主教程目录没有本格命令所需的第一代example.py与大模型权重，所以不能在tutorials中直接照抄就完成推理。主文读取官方llama_v1源码补充调用链，两份材料的角色不同。

# %% [markdown]
# ## inference/sample/generator：example.py

# %% [markdown]
# ## 单元 6：三条空格赋值风格的 torchrun 命令
#
# 【必须会】代码块在Markdown中，只展示命令；“Run Cell”不会在终端启动它。它含三条独立命令，分别面向7B、13B、33B。
#
# | 部分 | 类型/作用 | 常见误读 |
# |---|---|---|
# | torchrun | 进程启动器，可执行命令 | 不是LLaMA模型类 |
# | --nproc_per_node 1/2/4 | 每节点启动进程数，供启动器消费 | 不是batch_size或token数量 |
# | example.py | 启动器运行的外部脚本 | 不会因文件名而自动下载源码 |
# | --ckpt_dir ./weights/7B等 | 传给example.py的checkpoint目录 | 不是单个tokenizer文件 |
# | --tokenizer_path ./weights/tokenizer.model | 传给脚本的分词模型路径 | 不是Transformer权重 |
# | 行尾反斜杠 | Bash/POSIX shell的续行写法 | 不是PowerShell续行符 |
#
# 参数在脚本名前后的位置决定由启动器还是被启动脚本消费。官方第一代加载路径要求模型并行进程数与checkpoint分片相匹配，不能随意把2改1以适配显存。命令的1/2/4是历史示例，应核对实际权重与配置。
#
# Windows/PowerShell查看这些命令时，可将每条完整命令写为一行理解；不应直接把Bash的反斜杠跨行写法复制过去。这里不实际启动模型，因为没有配套checkpoint；只核对参数解析和依赖位置，不产生推理结果。

# %% [markdown]
# ```
# # 【必须会】下面展示 7B 的历史推理启动命令；本次没有执行。
# # torchrun 是分布式进程启动器；不是模型类，也不会自动提供权重。
# # --nproc_per_node 的值 1 表示每节点启动 1 个进程，不是 batch_size。
# # example.py 是外部入口；启动器之后的参数由该脚本解析。
# # --ckpt_dir 指向模型 checkpoint 目录；--tokenizer_path 指向分词模型文件。
# # ./weights 相对于启动时的工作目录；主教程未包含这些权重和该入口脚本。
# # 第一代模型并行需核对分片数与进程数，不能仅为省显存随意改进程数。
# # 行末反斜杠是 Bash 续行符；续行内部不要插入独立注释。
# # PowerShell 不使用此续行符；这里保留原 Bash 写法供阅读。
# # 参数可写为 --名称 值 或 --名称=值；两种格式不表示不同生成算法。
# torchrun --nproc_per_node 1 example.py \
#     --ckpt_dir ./weights/7B \
#     --tokenizer_path ./weights/tokenizer.model
#
# # 【必须会】下面展示 13B 的历史推理启动命令；本次没有执行。
# # torchrun 是分布式进程启动器；不是模型类，也不会自动提供权重。
# # --nproc_per_node 的值 2 表示每节点启动 2 个进程，不是 batch_size。
# # example.py 是外部入口；启动器之后的参数由该脚本解析。
# # --ckpt_dir 指向模型 checkpoint 目录；--tokenizer_path 指向分词模型文件。
# # ./weights 相对于启动时的工作目录；主教程未包含这些权重和该入口脚本。
# # 第一代模型并行需核对分片数与进程数，不能仅为省显存随意改进程数。
# # 行末反斜杠是 Bash 续行符；续行内部不要插入独立注释。
# # PowerShell 不使用此续行符；这里保留原 Bash 写法供阅读。
# # 参数可写为 --名称 值 或 --名称=值；两种格式不表示不同生成算法。
# torchrun --nproc_per_node 2 example.py \
#     --ckpt_dir ./weights/13B \
#     --tokenizer_path ./weights/tokenizer.model
#
# # 【必须会】下面展示 33B 的历史推理启动命令；本次没有执行。
# # torchrun 是分布式进程启动器；不是模型类，也不会自动提供权重。
# # --nproc_per_node 的值 4 表示每节点启动 4 个进程，不是 batch_size。
# # example.py 是外部入口；启动器之后的参数由该脚本解析。
# # --ckpt_dir 指向模型 checkpoint 目录；--tokenizer_path 指向分词模型文件。
# # ./weights 相对于启动时的工作目录；主教程未包含这些权重和该入口脚本。
# # 第一代模型并行需核对分片数与进程数，不能仅为省显存随意改进程数。
# # 行末反斜杠是 Bash 续行符；续行内部不要插入独立注释。
# # PowerShell 不使用此续行符；这里保留原 Bash 写法供阅读。
# # 参数可写为 --名称 值 或 --名称=值；两种格式不表示不同生成算法。
# torchrun --nproc_per_node 4 example.py \
#     --ckpt_dir ./weights/33B \
#     --tokenizer_path ./weights/tokenizer.model
# ```

# %% [markdown]
# ## 单元 7：等号赋值风格的三条命令
#
# 【必须会】它对应前格的三条命令，将参数写成`--名称=值`。对于这里的启动器参数解析，`--nproc_per_node=1`与`--nproc_per_node 1`表示同一数值；不是新增模型配置，也不是赋值给Python局部变量。
#
# ckpt_dir、tokenizer_path在这个历史入口由Fire解析为main参数。两格强调命令参数书写风格，没有两个不同生成算法，也不能把重复展示算作两次运行。续行规则和工作目录问题仍与前格相同。
#
# 注释版在每条完整 Markdown 命令前插入 Bash 注释，逐项解释进程数、入口脚本、checkpoint 和 tokenizer 路径、续行及等号格式。不在反斜杠续行中间插入注释；去除新增注释后，两个原命令块逐字一致。六条命令均记录为“展示/静态核对，未启动”。

# %% [markdown]
# ```
# # 【必须会】下面展示 7B 的历史推理启动命令；本次没有执行。
# # torchrun 是分布式进程启动器；不是模型类，也不会自动提供权重。
# # --nproc_per_node 的值 1 表示每节点启动 1 个进程，不是 batch_size。
# # example.py 是外部入口；启动器之后的参数由该脚本解析。
# # --ckpt_dir 指向模型 checkpoint 目录；--tokenizer_path 指向分词模型文件。
# # ./weights 相对于启动时的工作目录；主教程未包含这些权重和该入口脚本。
# # 第一代模型并行需核对分片数与进程数，不能仅为省显存随意改进程数。
# # 行末反斜杠是 Bash 续行符；续行内部不要插入独立注释。
# # PowerShell 不使用此续行符；这里保留原 Bash 写法供阅读。
# # 参数可写为 --名称 值 或 --名称=值；两种格式不表示不同生成算法。
# torchrun --nproc_per_node=1 example.py \
#     --ckpt_dir=./weights/7B \
#     --tokenizer_path=./weights/tokenizer.model
#
# # 【必须会】下面展示 13B 的历史推理启动命令；本次没有执行。
# # torchrun 是分布式进程启动器；不是模型类，也不会自动提供权重。
# # --nproc_per_node 的值 2 表示每节点启动 2 个进程，不是 batch_size。
# # example.py 是外部入口；启动器之后的参数由该脚本解析。
# # --ckpt_dir 指向模型 checkpoint 目录；--tokenizer_path 指向分词模型文件。
# # ./weights 相对于启动时的工作目录；主教程未包含这些权重和该入口脚本。
# # 第一代模型并行需核对分片数与进程数，不能仅为省显存随意改进程数。
# # 行末反斜杠是 Bash 续行符；续行内部不要插入独立注释。
# # PowerShell 不使用此续行符；这里保留原 Bash 写法供阅读。
# # 参数可写为 --名称 值 或 --名称=值；两种格式不表示不同生成算法。
# torchrun --nproc_per_node=2 example.py \
#     --ckpt_dir=./weights/13B \
#     --tokenizer_path=./weights/tokenizer.model
#
# # 【必须会】下面展示 33B 的历史推理启动命令；本次没有执行。
# # torchrun 是分布式进程启动器；不是模型类，也不会自动提供权重。
# # --nproc_per_node 的值 4 表示每节点启动 4 个进程，不是 batch_size。
# # example.py 是外部入口；启动器之后的参数由该脚本解析。
# # --ckpt_dir 指向模型 checkpoint 目录；--tokenizer_path 指向分词模型文件。
# # ./weights 相对于启动时的工作目录；主教程未包含这些权重和该入口脚本。
# # 第一代模型并行需核对分片数与进程数，不能仅为省显存随意改进程数。
# # 行末反斜杠是 Bash 续行符；续行内部不要插入独立注释。
# # PowerShell 不使用此续行符；这里保留原 Bash 写法供阅读。
# # 参数可写为 --名称 值 或 --名称=值；两种格式不表示不同生成算法。
# torchrun --nproc_per_node=4 example.py \
#     --ckpt_dir=./weights/33B \
#     --tokenizer_path=./weights/tokenizer.model
# ```

# %% [markdown]
# ## 单元 8：启动器和环境变量
#
# 【必须会】此格依然是Markdown。现代torchrun对应`python -m torch.distributed.run`，原格列出的`python -m torch.distributed.launch`是旧启动器，不能当成完全相同模块名。`python -m`表示把已安装包中的模块作为程序运行，不是读取任意名叫torch.py的文件。
#
# LOCAL_RANK是当前节点内的进程编号，WORLD_SIZE是参与任务的总进程数；它们由分布式启动环境提供，不是模型权重或可训练参数。单节点两进程通常local_rank为0和1，world_size=2。多节点时每节点进程数与总进程数不同。
#
# 这一格不设置环境变量，也没有真正初始化分布式通信。需要在启动后的脚本中读取这些值并选择设备/并行组，详见主文的setup_model_parallel调用链。

# %% [markdown]
# - `torchrun`
#     - `python -m torch.distributed.launch`
#     - 会自动设置 `LOCAL_RANK` 与 `WORLD_SIZE`
