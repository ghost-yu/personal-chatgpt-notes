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
# ## 单元0：导入 Image
#
# `from IPython.display import Image` 从IPython的display模块导入Image类。`from`后面写模块路径，`import`后面写要使用的名字。执行后，当前notebook就能通过`Image(...)`创建图片展示对象；单元4会用到它。

# %%
# 从IPython.display导入Image类，后面用它显示配图。
from IPython.display import Image

# %% [markdown]
# ## 单元1：summary 标题
#
# 这是Markdown标题，用来组织后面的模型介绍。Markdown中的`##`表示二级标题，渲染后显示为标题文字。

# %% [markdown]
# ## summary

# %% [markdown]
# ## 单元2：LLaMA、Alpaca 与 Vicuna
#
# LLaMA是基础预训练模型；Alpaca在LLaMA上做指令微调；Vicuna在LLaMA上做对话微调。原格的箭头可按介绍顺序阅读：先认识基础模型，再看它的两个应用方向。
#
# `self instruct`指构造指令数据的方法，SFT指用输入与目标回答进行监督微调。prompt/answer表示训练样本中的两部分；ShareGPT提供对话数据。
#
# 这格的链接指向外部仓库和发布说明。原文`text-davini-003`应读作历史接口名`text-davinci-003`。许可描述对应2023年的材料，具体模型的使用条件见对应版本的发布说明。

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
# ## 单元3：llama weights 标题
#
# 引出下一格的权重目录截图。读图时要区分模型结构、训练后的参数和分词器文件。

# %% [markdown]
# ## llama weights

# %% [markdown]
# ## 单元4：展示权重目录图片
#
# `Image('../imgs/llama_weights.png', width=300)`调用Image构造器。第一个参数是图片路径字符串，`width=300`指定展示宽度。
#
# 相对路径从notebook内核的当前工作目录开始解析。工作目录为`代码/tutorials`时，`..`回到`代码`，随后进入`imgs`读取原图。复制notebook后若图片无法显示，先检查工作目录和图片位置。
#
# 这一步的数据流是：路径字符串 → 读取PNG字节 → 创建Image对象 → notebook显示图片。`width`控制页面上的展示尺寸，原PNG像素保持原样。普通Python脚本中要显式调用展示接口，才会出现notebook里的显示效果。
#
# 截图中，7B、13B等目录名表示模型参数规模；`.pth`保存权重分片，`params.json`保存结构配置，`tokenizer.model`保存分词模型。运行推理需要在本地准备这些配套文件。图中的30B和后面命令的33B有历史命名差别，路径应以实际文件为准。

# %%
# 读取权重目录截图，以300像素宽度显示。
# 从代码/tutorials运行时，..返回代码目录，再进入imgs。
Image('../imgs/llama_weights.png', width=300)

# %% [markdown]
# ## 单元5：inference / sample / generator 标题
#
# 这三个词对应生成过程的不同环节：inference使用已有权重计算输出；sample从概率分布选出一个token；generator重复计算、选择和追加token。
#
# 后面的命令调用外部`example.py`。主讲解补充了官方第一代LLaMA实现，可沿入口脚本、模型加载和生成循环继续读。

# %% [markdown]
# ## inference/sample/generator：example.py

# %% [markdown]
# ## 单元6：三条 torchrun 启动命令
#
# 本格用Markdown展示7B、13B、33B的三条命令。需要在准备好源码和权重后，把命令交给终端执行。
#
# | 部分 | 作用 |
# |---|---|
# | `torchrun` | 启动运行脚本所需的进程 |
# | `--nproc_per_node 1/2/4` | 每个节点启动的进程数 |
# | `example.py` | 各进程执行的入口脚本 |
# | `--ckpt_dir ./weights/7B`等 | 入口脚本读取的权重目录 |
# | `--tokenizer_path ./weights/tokenizer.model` | 分词模型路径 |
# | 行末反斜杠 | Bash/POSIX shell中的续行符 |
#
# 脚本名前的参数由torchrun处理，脚本名后的参数交给`example.py`。第一代官方加载逻辑要求模型并行进程数与checkpoint分片匹配，因此选择进程数时要同时查看权重文件。
#
# PowerShell的续行语法与Bash不同；阅读这些命令时，可以先将每条命令写成一行，理清启动器、脚本和参数的关系。

# %% [markdown]
# ```
# # 下面展示 7B 的历史推理启动命令；本次没有执行。
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
# # 下面展示 13B 的历史推理启动命令；本次没有执行。
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
# # 下面展示 33B 的历史推理启动命令；本次没有执行。
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
# ## 单元7：等号形式的命令参数
#
# 这里再次展示三条命令，把参数写成`--名称=值`。例如`--nproc_per_node=1`与前格的`--nproc_per_node 1`表达同一项配置。
#
# 历史入口通过Fire将`ckpt_dir`、`tokenizer_path`等命令行字段传给`main`。把两格并排读，可以看出命令行参数的两种书写方式。
#
# 注释版在完整命令前解释各字段。续行中的反斜杠应保持连贯，说明性注释放在整条命令之前。

# %% [markdown]
# ```
# # 下面展示 7B 的历史推理启动命令；本次没有执行。
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
# # 下面展示 13B 的历史推理启动命令；本次没有执行。
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
# # 下面展示 33B 的历史推理启动命令；本次没有执行。
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
# ## 单元8：启动器与环境变量
#
# `python -m`把已安装包中的模块作为程序运行。现代torchrun对应`python -m torch.distributed.run`；原格还列出了旧启动器`python -m torch.distributed.launch`。
#
# `LOCAL_RANK`表示进程在当前节点内的编号，`WORLD_SIZE`表示整个任务的总进程数。例如单节点启动两个进程，LOCAL_RANK分别为0和1，WORLD_SIZE均为2。多节点时，WORLD_SIZE还要乘上参与的节点数。
#
# 启动器提供这些环境变量，入口脚本再读取它们，选择设备并初始化并行组。主讲解的`setup_model_parallel`调用链解释了后续步骤。

# %% [markdown]
# - `torchrun`
#     - `python -m torch.distributed.launch`
#     - 会自动设置 `LOCAL_RANK` 与 `WORLD_SIZE`
