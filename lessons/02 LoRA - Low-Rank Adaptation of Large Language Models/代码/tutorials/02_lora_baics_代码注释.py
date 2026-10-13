# 第02讲 LoRA 代码注释阅读版
# 来源：chunhuizhang/personal_chatgpt，提交 7fbc29824171fd07dc2c30fe922e61ebc018ad1d。
# 不含 JSON 或历史图片编码；# %% 分格，建议使用配套 notebook 执行展示。
# 原 Markdown13 伪代码保留为注释，含原缩进缺陷；可运行教学例另见 verify_lora.py。

# %% [markdown]
# # 第02讲中文代码说明
#
# 新增说明与注释为教学内容；原输出为历史记录。
# 原版另存。工作目录为代码/tutorials。Markdown13保留原伪代码缺陷；可运行示例另见复现脚本。

# %% [markdown]
# ## 单元 0：导入图片展示类
#
# 【必须会】`from IPython.display import Image` 从模块导入Image类，在当前命名空间绑定Image名字。它不创建神经网络，也不读取图片。后续六次Image调用依赖这一格；跳过导入会触发NameError。
#
# `from` 指模块来源，`import` 指绑定对象；Image是类，调用它才会构造实例。这一格没有Tensor、dtype或模型shape。中文注释直接写明它仅准备图片展示API，没有加载BLOOM或LoRA。

# %%
# 导入notebook图片展示类；不创建模型，也不读取权重。
# from 模块 import 名称：把 IPython.display 中的 Image 类绑定到当前内核名字。
# 类是创建对象的模板；这里只导入类，尚未调用 Image(...) 创建展示对象。
# 后续六格需要先运行本格；否则会因为 Image 未定义触发 NameError。
# 本格没有参数张量、dtype、shape、loss 或参数更新；它是展示辅助代码。
from IPython.display import Image

# %% [markdown]
# ## 单元 1：summary 标题
#
# 【可以暂时不掌握】Markdown的`##`表示二级标题，不是Python注释单元。运行/渲染本格不会修改参数，只为后面的概念概述分节。

# %% [markdown]
# ## summary

# %% [markdown]
# ## 单元 2：LoRA 的定义与冻结关系
#
# 【必须会】本格是文字与论文链接。Low-Rank Adaptation指用两个小矩阵表示权重增量；不是要求原始大权重本身低秩。原文Adaption为拼写差异，论文用Adaptation。
#
# `freezes`对应基权重不参与参数更新；`injects`对应在选定线性映射中加可训练分支。本格没调用`requires_grad_(False)`或PEFT注入API，不能从概念文字推断冻结代码已经执行。
#
# “随机投影到小子空间”只描述部分初始化直觉，LoRA因子随后随任务训练，通常并非固定随机投影。论文公式为列向量`W0 x + B A x`；本教程后面用行向量，两者必须按转置关系对应。PEFT是参数高效微调类别及库名，LoRA是其中一种方法。不是所有Transformer层或所有矩阵都必须加LoRA。

# %% [markdown]
# - LoRA: **Lo**w-**R**ank **A**daption of large language models
#     - A random projection to a smaller subspace
#     - parameter efficient
#         - PEFT
#     - https://arxiv.org/abs/2106.09685
# - 实现细节上
#     - 是一个 adaptor of pretrained model
#         - adaptor 是小的
#         - pretrained model 是大的
#             - large language models
#             - large vision models
#     - **freezes** pre-trained model weights
#     - **injects** trainable rank decomposition matrices
#         - into each layer of transformer Architecture

# %% [markdown]
# ## 单元 3：从完整增量到少量训练参数
#
# 【必须会】本格中的Phi是模型参数集合，Delta Phi是微调后的增量，Theta是编码增量的小参数集合。`|Phi|`这里表示参数个数，不是一个标量参数的绝对值。完整微调与LoRA的比较是需要优化/保存的任务参数规模；不能按比例推导整个模型显存。
#
# 对于一个输入宽k、输出宽d的线性层，完整增量有kd个参数；低秩因子有r(k+d)个参数。基权重仍需加载，激活也仍参与前向和必要的反向。Q/K/V/O是注意力投影矩阵，不是输入token或输出文本；本格没有具体实现attention。

# %% [markdown]
# - 基本思想
#     - 对于 transformer，最为重要的 self attention module
#         - Wq、Wk、Wv、Wo 表示 learnable query/key/value/output projection matrices
#         - 将这些记为模型的参数 $\Phi$
#     - 在 full fine-tune （不进行任何的 freeze）时，model 会初始化为预训练好的权重 $\Phi$，最终 fine-tune 之后，调整为 $\Phi+\Delta\Phi$（基于反向传播和梯度下降）
#         - 每一个下游任务都要学习对应的 $\Delta\Phi$（$|\Delta\Phi|=|\Phi|$）
#     - LoRA 作为一个 parameter efficient 将与（进一步）下游任务相关的 $\Delta\Phi=\Delta\Phi(\Theta)$，进一步编码（encode）为规模更小的参数 $\Theta$
#         - $|\Theta| \ll |\Delta\Phi|=|\Phi|$
#         - LoRA 采用 low-rank representation 来 encode $\Delta\Phi$

# %% [markdown]
# ## 单元 4：展示论文公式图片
#
# 【建议会】代码第一行`# https://arxiv.org/abs/2106.09685`只是Python注释，不发起网络请求。下一行`Image('../imgs/lora.png', width=600)`读取本地PNG，返回Image对象；notebook显示末尾表达式。
#
# 路径为str，width为int展示宽度600，没有模型shape或可训练变量。相对路径基于内核工作目录，配套代码的tutorials上一级imgs存放原图。把notebook复制到别处并不保证图片路径可用。Image的data是字节，不是`[B,C,H,W]`模型输入。
#
# 公式图与论文第4.1节相连：冻结W0，用两个因子表示Delta W，再把两个分支输出相加。图片展示不执行公式。原链接注释完整保留，新增中文注释解释图片用途及路径。

# %%
# 读取LoRA论文公式截图；width=600仅指定展示宽度，公式没有在此执行。
# Image(...) 调用展示类构造实例；第一个位置实参是 str 本地图片路径。
# 原路径 ../imgs/lora.png 由内核当前工作目录解析，不自动以 notebook 文件位置为基准。
# 从本讲代码/tutorials 启动内核，.. 才会指向代码，再到配套 imgs 目录。
# 若图片路径不可用会报 FileNotFoundError；这不是模型加载失败。
# width=600 是 int 关键字参数，控制前端展示宽度，不修改原图像素。
# 返回 IPython.display.Image 对象，data 属性包含嵌入的 PNG 字节。
# 数据流为 str 路径 -> PNG bytes -> Image 实例 -> notebook 展示，无 torch.Tensor。
# notebook 显示单元末尾表达式；普通 Python 脚本不会自动渲染这个对象。
# 保留的图片输出是作者历史记录；本次实际读取与哈希比较另存复现记录。
# 原 arxiv URL 是 Python 注释，不会发起请求或下载论文。
# 图对应论文低秩公式：列向量 W0 x + BAx；本教程后文采用行向量 xW + x(W_AW_B)。
# https://arxiv.org/abs/2106.09685
Image('../imgs/lora.png', width=600)

# %% [markdown]
# ## 单元 5：ft vs. lora 标题
#
# 【建议会】ft在这里是fine-tuning简称，不是Python函数。标题引出完整更新与低秩更新的表示关系，本格没有训练循环。

# %% [markdown]
# ## ft vs. lora

# %% [markdown]
# ## 单元 6：完整微调的函数表示
#
# 【必须会】Markdown的LaTeX表示输入x经参数W控制的函数得到h，微调后改用W+Delta W。f不一定只是线性函数；对后面线性层例子，可以具体写h=xW。
#
# W是已有权重，Delta W是任务更新量，h是输出特征。若X为`[N,k]`、W为`[k,d]`，输出为`[N,d]`。本格没有定义名叫f_W的Python函数，也没有更新Tensor。箭头表示概念转变，不是反向传播API。

# %% [markdown]
#
# $$
# \begin{split}
# &h=f_W(x)\\
# &\Downarrow\\
# &h=f_{W+\Delta W}(x)\\
# \end{split}
# $$

# %% [markdown]
# ## 单元 7：完整微调示意图
#
# 【建议会】`Image('../imgs/regular-finetuning.png', width=600)`的API、类型和相对路径规则与单元4相同，但展示的对象换成完整参数更新图。图从左到右是原模型前向、反向获得完整Delta W、更新后模型前向；每个前向框内部从下往上读输入→权重→h。权重块拥有参数，箭头不拥有参数。
#
# 这里的600只改变展示尺寸。没有读取checkpoint、没有调用optimizer。原执行输出是历史PNG展示，本次验证只确认同一图片能真实读取并产生Image。

# %%
# 展示完整微调的三个阶段示意；只读取PNG，不执行反向传播。
# Image(...) 调用展示类构造实例；第一个位置实参是 str 本地图片路径。
# 原路径 ../imgs/regular-finetuning.png 由内核当前工作目录解析，不自动以 notebook 文件位置为基准。
# 从本讲代码/tutorials 启动内核，.. 才会指向代码，再到配套 imgs 目录。
# 若图片路径不可用会报 FileNotFoundError；这不是模型加载失败。
# width=600 是 int 关键字参数，控制前端展示宽度，不修改原图像素。
# 返回 IPython.display.Image 对象，data 属性包含嵌入的 PNG 字节。
# 数据流为 str 路径 -> PNG bytes -> Image 实例 -> notebook 展示，无 torch.Tensor。
# notebook 显示单元末尾表达式；普通 Python 脚本不会自动渲染这个对象。
# 保留的图片输出是作者历史记录；本次实际读取与哈希比较另存复现记录。
# 图示输入 -> 原权重 -> 输出，再经完整微调更新权重；图中的箭头不是 autograd 调用。
Image('../imgs/regular-finetuning.png', width=600)

# %% [markdown]
# ## 单元 8：把完整更新拆成两个分支
#
# 【必须会】`Image('../imgs/regular-finetuning-alt.png', width=500)`展示等价分解：基分支xW，增量分支xDelta W，在相加处得到同shape输出。两路均为`[N,d]`，相加不是拼接。
#
# 完整Delta W依然是`[k,d]`的大矩阵；“多画一条支路”本身不产生参数节省。LoRA的下一步才把这一块分解成小因子。width=500不意味着模型hidden_size=500，本格返回Image而非矩阵。

# %%
# 展示基权重与完整增量两个分支；width=500不是模型特征维度。
# Image(...) 调用展示类构造实例；第一个位置实参是 str 本地图片路径。
# 原路径 ../imgs/regular-finetuning-alt.png 由内核当前工作目录解析，不自动以 notebook 文件位置为基准。
# 从本讲代码/tutorials 启动内核，.. 才会指向代码，再到配套 imgs 目录。
# 若图片路径不可用会报 FileNotFoundError；这不是模型加载失败。
# width=500 是 int 关键字参数，控制前端展示宽度，不修改原图像素。
# 返回 IPython.display.Image 对象，data 属性包含嵌入的 PNG 字节。
# 数据流为 str 路径 -> PNG bytes -> Image 实例 -> notebook 展示，无 torch.Tensor。
# notebook 显示单元末尾表达式；普通 Python 脚本不会自动渲染这个对象。
# 保留的图片输出是作者历史记录；本次实际读取与哈希比较另存复现记录。
# 数学上 x(W+Delta W)=xW+xDelta W；两路输出均 [N,d]，相加不是 cat 拼接。
# 完整 Delta W 仍为 [k,d]；增加一条画出来的分支本身不节省参数。
Image('../imgs/regular-finetuning-alt.png', width=500)

# %% [markdown]
# ## 单元 9：低秩分解和参数量算例
#
# 【必须会】本格用A/B作维度名，WA/WB作矩阵名，容易与论文的A/B因子混淆。采用本篇统一的k/d后：W为`[k,d]`，WA为`[k,r]`，WB为`[r,d]`，Delta W=WA WB为`[k,d]`。
#
# $$P_{\mathrm{full}}=kd,\qquad P_{\mathrm{LoRA}}=r(k+d),\qquad \rho=\frac{r(k+d)}{kd}.$$
#
# P是参数个数，rho是该层可训练权重的比例，不含bias。k=100、d=500、r=5时，50000变成3000，占6%；不是全部网络总参数或总显存变成6%。因子乘积rank至多r，也可能小于r。768×768、r=8对应12288/589824≈2.0833%。
#
# 这些算式在Markdown中，没有Python执行；新增验证脚本实际计算两个比例。矩阵名字和数值均保留，说明插在原单元之前。

# %% [markdown]
#
# - $W\in\mathbb R^{A\times B}$
#
# $$
# \Delta W=W_AW_B,\\
# W_A\in \mathbb R^{A\times r}, W_B\in \mathbb R^{r\times B}
# $$
#
# - 最终的参数量由 $A\times B $ 降至 $r\times (A+B)$
#     - A=100，B=500，r=5
#     - 5\*(100+500) / (100\*500) == 3000/50000 == 6%
#     - 这就叫 Parameter efficiency

# %% [markdown]
# ## 单元 10：LoRA 双分支图
#
# 【必须会】`Image('../imgs/lora-weights.png', width=600)`显示灰色基权重和紫色两个小因子。沿图从下往上：输入X `[N,k]`分流，左路XW `[N,d]`；右路XWA `[N,r]`再乘WB `[r,d]`得到`[N,d]`；相加为H。
#
# 左侧W冻结，右侧因子可训练，加法和箭头无参数。图未画出alpha/r缩放，实际实现需核对scale。输出标成Embedding h是中间特征，不意味着本格调用`nn.Embedding`。
#
# 这条未合并分支在训练及未合并推理时存在；无dropout干扰的线性情况下，可将scale WA WB合并进W。图片不能证明任意量化模型合并完全无误差。单元只显示图，不实现合并。

# %%
# 展示低秩增量的双分支结构；没有在此构造WA/WB或进行训练。
# Image(...) 调用展示类构造实例；第一个位置实参是 str 本地图片路径。
# 原路径 ../imgs/lora-weights.png 由内核当前工作目录解析，不自动以 notebook 文件位置为基准。
# 从本讲代码/tutorials 启动内核，.. 才会指向代码，再到配套 imgs 目录。
# 若图片路径不可用会报 FileNotFoundError；这不是模型加载失败。
# width=600 是 int 关键字参数，控制前端展示宽度，不修改原图像素。
# 返回 IPython.display.Image 对象，data 属性包含嵌入的 PNG 字节。
# 数据流为 str 路径 -> PNG bytes -> Image 实例 -> notebook 展示，无 torch.Tensor。
# notebook 显示单元末尾表达式；普通 Python 脚本不会自动渲染这个对象。
# 保留的图片输出是作者历史记录；本次实际读取与哈希比较另存复现记录。
# 右支路 X:[N,k] -> XW_A:[N,r] -> XW_AW_B:[N,d]，再与 XW 相加。
# 基 W 冻结，两个小因子可训练；本单元只展示关系，不创建这些参数。
Image('../imgs/lora-weights.png', width=600)

# %% [markdown]
# ## 单元 11：alpha 标题
#
# 【建议会】本格标题写alpha参数，但下一格实际讨论rank。本篇把alpha与r区分：r控制瓶颈宽度，alpha通常用于缩放s=alpha/r。两者都是超参数；alpha未在原单元13赋值，不能从标题推断已有定义。

# %% [markdown]
# ## 伪代码

# %% [markdown]
# ## 单元 12：rank 的容量与代价
#
# 【必须会】原英文的意思是：较小r使小矩阵参数更少、适配容量受限；较大r提供更多可学习自由度，但训练成本及过拟合风险依任务而定。不是r越大指标一定越好，也不是设小r就总能获得同比例加速。
#
# r为int；必须与因子对应维度一致。若k=d=768、r=8，降维后最后一维是8，而非完整矩阵的768。原文没有实验指标、最优rank或任务数据，本篇不虚构消融结果。

# %% [markdown]
# - $r$ 会是一个超参: a trade-off between model complexity, adaptation capacity, and the risk of underfitting or overfitting
#     - A smaller $r$ leads to a simpler low-rank matrix, which results in fewer parameters to learn during adaptation.
#         -  This can lead to faster training and potentially reduced computational requirements.
#     - However, with a smaller $r$, the capacity of the low-rank matrix to capture task-specific information decreases.
#         - This may result in lower adaptation quality, and the model might not perform as well on the new task compared to a higher $r$.

# %% [markdown]
# ## 单元 13：Markdown 中的 LoRA 伪代码
#
# 【必须会】这一格是代码围栏包住的文本，notebook不会执行它。中文注释版给每条有意义语句加说明，但不修缩进、不替换省略号、不填入alpha，从而保持原资料与原AST。单独可运行的[教学实现](../../复现/scripts/verify_lora.py)明确是新编补充。
#
# | 语句/名字 | 类型、shape与作用 | 依赖或状态变化 |
# |---|---|---|
# | input_dim=768 | int，输入特征数k | 配置，不是样本数量 |
# | output_dim=768 | int，输出宽度d | 不等于词表大小 |
# | rank=8 | int，低秩瓶颈r | 控制两个因子尺寸 |
# | W=... | Ellipsis占位对象 | 不能参与矩阵乘法，须另提供真实冻结Tensor |
# | torch.empty(input_dim,rank) | 未初始化Tensor `[768,8]` | 分配内存，不能直接当稳定初值 |
# | nn.Parameter(...) | 默认requires_grad=True的Parameter | 挂入nn.Module属性后才由模块注册；局部变量不自动成为模型成员 |
# | WA、WB | `[768,8]`及`[8,768]` | 两个可训练因子；原代码未冻结W |
# | kaiming_uniform_(WA,a=math.sqrt(5)) | 原地随机初始化WA | a是初始化分布参数，不是LoRA的alpha |
# | zeros_(WB) | 原地设WB为0 | 起始增量为0，保留基模型输出 |
# | def regular_forward_matmul(x,W) | 独立函数，非类方法 | 无self；返回值由正确缩进的return决定 |
# | h=x@W | Tensor `[N,768]` | @为矩阵乘法，非逐元素乘法 |
# | return h | 原文位于函数外 | compile产生SyntaxError，并非合法函数返回 |
# | def lora_forward_matmul(...) | 接收X/W/两个因子 | alpha从外部作用域查找，原文未定义 |
# | WA@WB | `[768,768]` | 先显式构造完整增量，增大临时计算/存储 |
# | x@(WA@WB)*alpha | `[N,768]`，再乘缩放 | 若alpha指论文超参数，通常还应除r；原文语义不确定 |
# | h+=... | 原地累加到h | 不更新W，返回前输出仍为`[N,768]` |
#
# `torch.empty`的dtype/设备由运行环境默认值决定，原文未指定。Kaiming按二维Tensor的第二维解释fan_in：原行向量WA的第二维是r，与PEFT Linear的`[r,k]`权重初始化方向不同，不能宣称数值分布完全相同。
#
# 所需导入torch、nn、math都缺失。两个return缺少函数内缩进，W是占位，alpha未赋值，也没有optimizer、loss、数据及迭代。不能只修一处就宣称完整训练代码已可用。
#
# 新编实现使用`H=XW+s(XWA)WB`，s=alpha/r；两次小乘法避免构造完整增量。原代码表达式的数学结合律在实数下成立，浮点运算只在容差内相等。
#
# 以X=[1,2]、W=I、WA=[[1],[2]]、WB=[[3,4]]、s=1为例，XWA=5，增量=[15,20]，总输出=[16,22]。用真实CPU PyTorch检查前向、权重合并、Linear转置方向和梯度；这不是BLOOM任务训练。
#
# 初始化WB=0时，WA初始梯度为0，WB通常有非零梯度；一步更新WB后WA可获得梯度。基权重被冻结，`.grad`保持None，不等于所有中间激活都无需反传。两因子都为0时两边初始梯度都为0，该失败边界也实际核对。

# %% [markdown]
# ```
# # 【必须会】input_dim 是整数配置，记作输入特征宽度 k=768，不是 batch_size。
# # 本格是 Markdown 文本，不会随 notebook 的 Run Cell 执行。
# # 原文未导入 torch、torch.nn as nn 和 math；此注释版不补写执行语句。
# input_dim = 768  # e.g., the hidden size of the pre-trained model
# # output_dim 是输出特征宽度 d=768，不是词表大小，也不是生成 token 数。
# # 若输入 X 为 [N,k]，线性层输出 H 为 [N,d]；N 可理解为本例的样本数。
# output_dim = 768  # e.g., the output size of the layer
# # rank 是低秩瓶颈 r=8，不是输入矩阵 X 的实际 rank。
# # Delta W = W_A @ W_B 的秩至多为 r；两个因子共有 r(k+d)=12288 个参数。
# # 原大矩阵有 kd=589824 个参数；这里比较可训练因子数，不等于整体显存比例。
# rank = 8  # The rank 'r' for the low-rank adaptation
#
# # ... 是 Python 的 Ellipsis 对象，不是自动读取预训练权重的指令。
# # 真实行向量约定需要 W 为浮点 Tensor [k,d]=[768,768]，与 X 设备及 dtype 相容。
# # 完整 LoRA 实现还应冻结基权重；本赋值既没有真实 Tensor，也没有 requires_grad_(False)。
# # nn.Linear 存储权重为 [out,in]=[d,k]；不能直接把它的 weight 当成本例 W 而忽略转置。
# W = ... # from pretrained network with shape input_dim x output_dim
#
# # torch.empty(input_dim, rank) 分配 [k,r]=[768,8] Tensor；empty 不保证填零。
# # nn.Parameter(...) 包装为默认 requires_grad=True 的可学习叶子参数。
# # 原文没有指定 dtype/device，遵从当前默认；该行的 nn 和 torch 仍需导入。
# # 这里只是独立变量；赋给 nn.Module 的属性才会被模块注册为参数。
# W_A = nn.Parameter(torch.empty(input_dim, rank)) # LoRA weight A
# # 第二个可学习因子形状为 [r,d]=[8,768]，与 W_A 的内侧维度 8 匹配。
# # 两个因子乘积是 [k,d]；W_B 的命名不表示它是 batch 维度 B。
# # empty 分配的初值未定义，后面 zeros_ 初始化完成后才适合前向使用。
# W_B = nn.Parameter(torch.empty(rank, output_dim)) # LoRA weight B
#
# # Initialization of LoRA weights
# # 原地随机初始化 W_A；末尾下划线表示修改传入 Tensor 的内容。
# # a=math.sqrt(5) 是初始化分布参数，不是 LoRA 的 alpha 或 alpha/r。
# # 按本例 [k,r] 布局，二维初始化的 fan_in 是第二维 r。
# # PEFT Linear 的 A 权重通常存为 [r,k]，其 fan_in 是 k；不能宣称初始化分布相同。
# nn.init.kaiming_uniform_(W_A, a=math.sqrt(5))
# # 原地把 W_B 全部置零，使初始 W_A @ W_B=0，初始输出等于基模型输出。
# # 若之后有非零损失梯度，B 通常先获得梯度；因 B=0，A 的第一次梯度为零。
# # 这不表示 A 永远不学习；更新 B 后，A 可以有非零梯度。两因子均为零会让两边起始梯度都为零。
# nn.init.zeros_(W_B)
#
# # def 定义独立函数，不是类方法；没有 self，也没有默认参数。
# # 预期输入 x:[N,k] 与 W:[k,d]，计算基线性层输出 [N,d]。
# # 函数体由缩进划定；下面原 return 未缩进，因此原文件不能直接编译执行。
# def regular_forward_matmul(x, W):
#     # @ 是矩阵乘法，不是逐元素乘法；[N,k] @ [k,d] -> h:[N,d]。
#     # 该表达式创建基输出，不修改 W；是否冻结 W 由 requires_grad/优化器设置决定。
#     h = x @ W
# # 【原文缺陷】此 return 没有缩进到函数体，compile 会报 return outside function。
# # 只有放进函数体的 return 才会向调用者返回 h；注释版保留原行，不静默修复。
# # 原格也没有数据、loss、optimizer 或训练循环；可运行补充见复现/scripts/verify_lora.py。
# return h
#
# # 独立函数接收 x、W、W_A、W_B，没有默认值；alpha 将从外部作用域查找。
# # 预期数据流：基分支 x@W 和低秩增量分支相加，输出 [N,d]。
# # 原文 alpha 未定义、W 为占位、return 缩进不合法；不要把本格当成可运行训练器。
# def lora_forward_matmul(x, W, W_A, W_B):
#     # @ 是矩阵乘法，不是逐元素乘法；[N,k] @ [k,d] -> h:[N,d]。
#     # 该表达式创建基输出，不修改 W；是否冻结 W 由 requires_grad/优化器设置决定。
#     h = x @ W  # regular matrix multiplication
#     # 先 W_A@W_B：[k,r]@[r,d] -> [k,d]，再 x@该增量 -> [N,d]。
#     # 乘 alpha 后与同 shape 的 h 相加；h += 修改输出 h，不是修改基权重 W。
#     # 这里显式构造大增量 [k,d]；教学实现改用 (x@W_A)@W_B，经过 [N,r] 中间特征。
#     # 论文常用缩放 s=alpha/r；原文只乘 alpha，无法判断作者是否已把 r 吸收到 alpha 中。
#     # 实数下结合律给出等价表达式；浮点下只能按误差容差比较，不能保证逐位一致。
#     h += x @ (W_A @ W_B) * alpha # use scaled LoRA weights
# # 【原文缺陷】此 return 没有缩进到函数体，compile 会报 return outside function。
# # 只有放进函数体的 return 才会向调用者返回 h；注释版保留原行，不静默修复。
# # 原格也没有数据、loss、optimizer 或训练循环；可运行补充见复现/scripts/verify_lora.py。
# return h
# ```

# %% [markdown]
# ## 单元 14：BLOOM 模型结构截图
#
# 【必须会】`Image('../imgs/bloom.png')`没有显式width参数，使用Image API默认展示行为。代码读取图片，不执行截图里的`BloomForCausalLM`构造，不实例化bitsandbytes层，也不下载7B权重。
#
# 图片实际列出Embedding(250880,4096)、30个BloomBlock、融合QKV投影4096→12288、Attention输出4096→4096、MLP4096→16384→4096、最后lm_head4096→250880。这些是截图配置，不能说适用于全部BLOOM模型。
#
# 以输入ID `[batch,time]`为例，Embedding后为`[batch,time,4096]`；融合投影后为`[batch,time,12288]`，12288=3×4096。它包括Q/K/V，不是12288个attention head。MLP中间宽16384=4×4096。Linear8bitLt是量化线性层名字，图片本身不验证其dtype、显存或性能。

# %%
# 展示历史BLOOM结构截图；不下载模型；省略width使用默认展示行为。
# Image(...) 调用展示类构造实例；第一个位置实参是 str 本地图片路径。
# 原路径 ../imgs/bloom.png 由内核当前工作目录解析，不自动以 notebook 文件位置为基准。
# 从本讲代码/tutorials 启动内核，.. 才会指向代码，再到配套 imgs 目录。
# 若图片路径不可用会报 FileNotFoundError；这不是模型加载失败。
# 原代码未传 width，使用 Image 默认展示宽度；它不表示模型 hidden_size 未定义。
# 返回 IPython.display.Image 对象，data 属性包含嵌入的 PNG 字节。
# 数据流为 str 路径 -> PNG bytes -> Image 实例 -> notebook 展示，无 torch.Tensor。
# notebook 显示单元末尾表达式；普通 Python 脚本不会自动渲染这个对象。
# 保留的图片输出是作者历史记录；本次实际读取与哈希比较另存复现记录。
# 截图配置：Embedding 后 [batch,time,4096]；融合 QKV 投影得到 [batch,time,12288]。
# 12288=3*4096 是融合输出宽度，不是 attention head 数；MLP 中间宽度16384。
# 截图中的 BloomForCausalLM 和 Linear8bitLt 都没有被本行实例化。
Image('../imgs/bloom.png')

# %% [markdown]
# ## 单元 15：插入 LoRA 后的 BLOOM 截图
#
# 【必须会】`Image('../imgs/bloom_lora.png')`也不传width，仅显示历史打印截图。外层PeftModelForCausalLM、LoraModel表示包装结构；基模型仍是BLOOM，不是另一个预训练模型。
#
# 目标是融合`query_key_value`投影，原4096→12288保留；新增lora_A为4096→16，lora_B为16→12288。这张图的r=16，不是单元13的r=8。PEFT Linear存储权重方向为`[out,in]`，所以A权重`[16,4096]`、B权重`[12288,16]`；这与行向量WA/WB是转置关系。
#
# 输入`[batch,time,4096]`在LoRA分支先经dropout(p=0.05)，再变为`[batch,time,16]`、`[batch,time,12288]`，与基分支相加。ModuleDict用default名字管理adapter；空ParameterDict不表示额外embedding参数正在训练。lm_head的CastOutputToFloat是历史包装，不能凭截图确定所有梯度/参数dtype。
#
# 该QKV层新增参数16×(4096+12288)=262144，原矩阵50331648个权重，比例约0.5208%；仅按图中30层同类目标估计共7864320个因子参数。bias与其他目标未计入，截图没有给alpha、冻结标记或训练统计，不能声称已证明整网可训练比例。

# %%
# 展示LoRA插入融合QKV后的历史截图；图中rank=16，不是伪代码的8。
# Image(...) 调用展示类构造实例；第一个位置实参是 str 本地图片路径。
# 原路径 ../imgs/bloom_lora.png 由内核当前工作目录解析，不自动以 notebook 文件位置为基准。
# 从本讲代码/tutorials 启动内核，.. 才会指向代码，再到配套 imgs 目录。
# 若图片路径不可用会报 FileNotFoundError；这不是模型加载失败。
# 原代码未传 width，使用 Image 默认展示宽度；它不表示模型 hidden_size 未定义。
# 返回 IPython.display.Image 对象，data 属性包含嵌入的 PNG 字节。
# 数据流为 str 路径 -> PNG bytes -> Image 实例 -> notebook 展示，无 torch.Tensor。
# notebook 显示单元末尾表达式；普通 Python 脚本不会自动渲染这个对象。
# 保留的图片输出是作者历史记录；本次实际读取与哈希比较另存复现记录。
# 图中 adapter r=16，A 权重 [16,4096]、B 权重 [12288,16]（nn.Linear 的 [out,in] 方向）。
# 输入 [batch,time,4096] -> 瓶颈 [batch,time,16] -> 增量 [batch,time,12288]。
# 单个融合 QKV 的因子数 16*(4096+12288)=262144；不代表整网总参数比例。
# 图中 rank 与 Markdown13 的 r=8 不同；图片未给 alpha，不能据此推断缩放值。
Image('../imgs/bloom_lora.png')
