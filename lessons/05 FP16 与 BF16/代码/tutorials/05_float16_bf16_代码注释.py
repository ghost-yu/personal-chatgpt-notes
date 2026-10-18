# -*- coding: utf-8 -*-
# 第05讲代码阅读版：无JSON、无历史图片编码。
# 按 # %% 分格。工作目录为代码/tutorials；原CUDA设备格在CPU包中失败。
# 双问号只能在IPython执行，此处保留为注释；执行方法与边界见复现/README.md。

# %% [markdown]
# ## 单元 0（code）：导入图片展示工具
#
# `from IPython.display import Image` 是导入语句：从IPython的display模块取出Image类，绑定当前名字Image。
# Image接收图片路径或图片数据，创建可在notebook中显示的对象。单元5、26会调用这个类展示浮点格式图。
# 先执行本格，后面才能使用Image。运行目录应为本讲`代码/tutorials`，因为相对路径依据进程工作目录解析，而不是自动依据notebook文件位置。

# %% 原单元 0
# 导入IPython的Image类，用来展示浮点格式配图。
# Image名称将在单元5、26使用；先运行导入格。
from IPython.display import Image

# %% [markdown]
# ## 单元 1（markdown）：参考资料与设备背景
#
# 两个链接分别是格式介绍和浮点位交互工具；它们不是本notebook的运行依赖。本次未抓取并替代这两篇参考资料，只核对课程内容和一手文档。
# “bf16: Ampere architecture”是NVIDIA硬件语境中的提示，不表示BF16只存在于Ampere，更不表示只要看到torch.bfloat16就能确认当前GPU有高速原生BF16支持。
# CPU/TPU、ROCm和不同PyTorch版本需要分别看实现。本次实际使用PyTorch 2.5.1+cpu，CUDA API返回False与CPU可创建BF16 tensor并不矛盾。

# %% [markdown] 原单元 1
# - references
#     - https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407
#     - https://evanw.github.io/float-toy/
# - bf16: Ampere architecture

# %% [markdown]
# ## 单元 2（markdown）：进入float16章节
#
# FP16通常指IEEE binary16。本章先解释一个标量的16位存储，再讨论这些标量构成tensor时的dtype；此标题本身没有运算。

# %% [markdown] 原单元 2
# ## float16

# %% [markdown]
# ## 单元 3（markdown）：half precision的含义
#
# half precision表示总存储位数为FP32的一半：16 bit=2 byte。它不保证每个十进制数都有一半的正确数字，也不保证全模型显存恰好减半；优化器状态和部分算子可用FP32。

# %% [markdown] 原单元 3
# - 半精度（half precision）

# %% [markdown]
# ## 单元 4（markdown）：进入位布局
#
# 下一格的图按符号位、指数位、小数位拆分FP16。存储布局是后面解码公式的前提；“表示”是章节标题，没有创建变量。

# %% [markdown] 原单元 4
# ### 表示

# %% [markdown]
# ## 单元 5（code）：显示FP16位布局图
#
# `Image('../imgs/float16.png', width=400)`调用Image构造器。第一个位置参数是文件名str，`..`回到代码目录，进入imgs；width为显示宽度，单位为像素，不是tensor的隐藏维度。
# 构造器读PNG，默认对文件图片嵌入字节；返回IPython图片对象。notebook把最后一个表达式结果交给显示系统，支持的前端显示PNG。图片的缩放不会改变浮点存储精度。
# 从左到右：一个符号位、五个指数位、十个显式小数位；这16位共同表示一个标量，后面的例子会按这些字段还原数值。

# %% 原单元 5
# 从代码/tutorials工作目录解释相对路径，读取代码/imgs/float16.png。
# width=400仅控制显示像素宽度；返回Image对象，notebook显示最后一行结果。
# 图中1个符号位+5个指数位+10个小数位合计16bit；不是shape[400]。
Image('../imgs/float16.png', width=400)

# %% [markdown]
# ## 单元 6（markdown）：分清指数域和实际指数
#
# 符号位s为0/1；5位指数域E是无符号整数0…31；10位小数域F是整数0…1023。这些是解码字段，不是tensor shape。
# E=1…30是正常数，实际指数q=E−15，对应−14…15。15是格式偏置β，不是运行时可训练参数。
# 小数值F/1024=F/2^10；正常数有效数为1+F/1024。E=0和E=31另有规则，不能先统一套正常数公式再修正。

# %% [markdown] 原单元 6
# - 1：sign（符号位），5：exponent bits，10：mantissa bits
#     - exponent：指数部分；
#     - mantissa：小数部分；
# - 指数部分
#     - 共5位，全0/全1都有特殊的用途，
#     - 表示范围 00001 ~ 11110（1 ~ 30)，
#     - 减去偏置（15），指数部分（-14 ~ 15）
# - 小数部分
#     - 共10位，
#     - 表示范围 0000000000 - 1111111111（0 ~ 1023）
#     - 再除以 1024，(0 ~ 1023)/1024

# %% [markdown]
# ## 单元 7（markdown）：进入FP16解码公式
#
# 接下来的公式把存储的三个整数域还原成实数。原格是标题；公式位于单元8，不是Python函数或训练loss。

# %% [markdown] 原单元 7
# ### 计算

# %% [markdown]
# ## 单元 8（markdown）：正常数公式
#
# 只在1≤E≤30使用：
# $$x=(-1)^s\,2^{E-15}(1+F/1024).$$
# s控制正负，E−15控制二进制量级，F/1024给有效数补小数。输入s/E/F为整数标量，输出x为实数标量。
# 拆开算：sign=(−1)^s；q=E−15；fraction=F/1024；significand=1+fraction；x=sign×2^q×significand。
# 例如s=0、E=15、F=1，x=1+1/1024=1.0009765625。此格只有LaTeX；补充验证脚本decode_bits执行等价解码并与真实PyTorch tensor对照。

# %% [markdown] 原单元 8
# $$
# (-1)^{\text{sign}}\cdot 2^{\text{exponent}-15}\cdot (1+\frac{\text{mantissa}}{1024})
# $$

# %% [markdown]
# ## 单元 9（markdown）：零、次正规数、无穷与NaN
#
# E=0时没有隐含前导1：
# $$x=(-1)^s\,2^{-14}(F/1024).$$
# F=0得到带符号零；F>0得到次正规数。E=31且F=0为±Inf；E=31且F≠0为NaN。
# NaN是浮点值类别，不是Python None；NaN不等于自己，验证应使用isnan。−0数值等于+0但符号位不同，可用signbit确认。
# 这里不能用E−15公式推E=0，否则会得到错误的指数−15与错误的前导1。

# %% [markdown] 原单元 9
# - 指数部分全 0
# $$
# (-1)^{\text{sign}}\cdot 2^{-14}\cdot (0+\frac{\text{mantissa}}{1024})
# $$
# - 指数部分全 1
#     - 小数部分全为 0，表示 +inf、-inf
#     - 小数部分不全为 0，表示 NaN

# %% [markdown]
# ## 单元 10（markdown）：极值与精度的阅读顺序
#
# 下面分别讨论最负有限数、最大有限数、最小正正常数、最小正次正规数和不同量级的间距。它们不是同一个“最小值”或统一“精度”。

# %% [markdown] 原单元 10
# ### 最小、最大、与精度

# %% [markdown]
# ## 单元 11（markdown）：校正最小值与精度表述
#
# 最大有限FP16：s=0、E=30、F=1023，2^15×(2−2^−10)=65504；改s=1为−65504，即最负有限数。
# 原文“绝对值最小值”列的是最小正次正规数2^−24≈5.96046448e−8；0的绝对值其实更小。最小正正常数是E=1、F=0，得到2^−14。
# 把2^−24直接叫整个FP16的精度会混淆概念。次正规区间距为2^−24；正常区间[2^q,2^(q+1))的间距为2^(q−10)。1附近是2^−10，1024附近是1。
# 原LaTeX中最后一行用==书写相等关系，不是Python比较语句；原格保留，规范公式在此说明与主解释中给出。

# %% [markdown] 原单元 11
# - 最大
#     - 0 11110 1111111111
#     - $(-1)^02^{30-15}(1+\frac{1023}{1024})$ == 65504
# - 最小
#     - 1 11110 1111111111
#     - $(-1)^12^{30-15}(1+\frac{1023}{1024})$ == -65504
#     - 正值的最小值
#         - 0 00000 0000000001
#         - $2^{-14}(0+1/1024)$ = 5.960464477539063e-08
# - 精度
#     - 0 00000 0000000001
#     - $(-1)^02^{-14}(0+\frac1{1024}) == 2^{-24}$

# %% [markdown]
# ## 单元 12（markdown）：准备计算具体位串
#
# 下一格从16位串拆出E和F再计算数值。二进制串中的小数部分需要按二进制整数解码，不能按十进制的0.101解释。

# %% [markdown] 原单元 12
# ### 计算示例

# %% [markdown]
# ## 单元 13（markdown）：26624的完整解码
#
# 位串`0 11101 1010000000`：s=0；11101₂=16+8+4+1=29；1010000000₂=512+128=640。
# 因此q=29−15=14，F/1024=640/1024=0.625，有效数1.625，2^14×1.625=26624。整体16位无符号编码为0x7680。
# 本格不实际算Python；复现补充用整数位解码和真实torch.float16位重解释两种路径核对26624，避免只检查说明文字。

# %% [markdown] 原单元 13
# - 0 11101 1010000000
# $$
# 2^{(29-15)}(1+\frac{640}{1024}) = 26624
# $$

# %% [markdown]
# ## 单元 14（markdown）：进入BF16章节
#
# BF16也是16位，但与FP16分配不同：它用更长指数域覆盖更大范围，用更短小数域交换精度。相同shape的BF16与FP16 tensor占用字节相同。

# %% [markdown] 原单元 14
# ## bf16

# %% [markdown]
# ## 单元 15（markdown）：brain float与TPU背景
#
# brain float关联Google Brain与TPU历史，不是Python变量brain。TPU格式/算子行为与CUDA设备查询分开讨论。
# 一手TPU文档说明转换使用round-to-nearest-even并把次正规数flush to zero。本讲解码给出理论位模式，CPU实际转换另测，不能把CPU能存下次正规数推广到TPU算术。

# %% [markdown] 原单元 15
# - brain float 16
#     - brain: google brain (TPU)

# %% [markdown]
# ## 单元 16（code）：查询当前CUDA/ROCm环境的BF16支持
#
# `import torch`导入PyTorch模块；`torch.cuda.is_bf16_supported()`是cuda子模块的函数，不是tensor方法，无显式参数，返回bool。
# 本次固定PyTorch2.5.1函数签名为`is_bf16_supported(including_emulation=True)`。当前CPU构建cuda不可用，返回False；这不判断CPU BF16能力，也不创建模型。
# 该版本先看HIP，再看cuda可用性、CUDA版本和GPUcompute capability；允许emulation时还会尝试创建BF16设备tensor。这里按当前记录的PyTorch版本解释接口；不同版本的参数见各自API定义。
# 查询可能在其他环境访问/初始化设备；当前CPU路径提前返回。调用链是notebook表达式→torch.cuda函数→环境检查，结果不是训练精度、吞吐或GPU实测。

# %% 原单元 16
# 导入真实PyTorch；不下载模型、数据或权重。
# is_bf16_supported是torch.cuda的函数，默认查询当前CUDA/ROCm环境，返回bool。
# PyTorch2.5.1有including_emulation=True默认参数；版本差异见单元说明。
# 本次CPU版返回False，不能用这个结果否定CPU可创建BF16 tensor。
import torch
torch.cuda.is_bf16_supported()

# %% [markdown]
# ## 单元 17（code）：dtype对象不是数据
#
# `torch.bfloat16`是torch模块的dtype对象；最后一行让notebook显示其repr。
# 这一格介绍bfloat16数据类型。例如，`torch.tensor([1.001], dtype=torch.bfloat16)`会构造一个使用该类型的张量。
# torch.bfloat16是数据类型对象；CUDA设备可用性由当前PyTorch环境与驱动共同决定。CPU补充会把float64标量列表转成shape[N]的FP16/BF16 tensor，明确区分名称展示与真实转换。

# %% 原单元 17
# 查看torch.bfloat16这个数据类型对象；创建张量时可将它传给dtype参数。
# notebook会显示dtype名称；硬件运行能力须另查，不由名称存在来证明。
torch.bfloat16

# %% [markdown]
# ## 单元 18（code）：嵌套CUDA设备查询
#
# 先执行内层`torch.cuda.current_device()`，无显式输入，返回当前CUDA设备索引int；再把这个int传给外层`torch.cuda.get_device_properties(device)`，返回设备属性对象。
# 对象含name、major/minor、total_memory等属性；其显示文本是repr，不是tensor的shape或实时已占用显存。两个函数读取运行时状态，可能触发CUDA惰性初始化。
# CPU版torch在内层就报`AssertionError: Torch not compiled with CUDA enabled`，外层不会被调用。本次保留原表达式并记录该预期失败；没有用假设备属性伪造成功，也不自动把查询改为CPU。
# 要运行此格，需要兼容的CUDA版PyTorch、驱动和可用设备；可用torch.cuda.is_available()检查当前Python环境是否能使用CUDA。

# %% 原单元 18
# 内层current_device先返回设备索引int，再传给get_device_properties。
# 外层返回属性对象（name、compute capability、total_memory等），不是tensor。
# 此访问可能初始化CUDA；CPU包在内层报错，外层不执行。
# 本次记录实际预期失败，保留原代码；原GPU历史输出不当作本机结果。
torch.cuda.get_device_properties(torch.cuda.current_device())

# %% [markdown]
# ## 单元 19（code）：检查HIP构建字段
#
# `torch.version.hip`是版本模块的属性访问，无括号、不调用函数；其值为HIP构建版本字符串或None。
# 本次CPU包返回None；None表示这个包没有HIP版本信息，不单独证明机器没有AMD硬件或支持NVIDIA BF16。notebook一般不显示结果为None的表达式，所以空白输出不是“没执行”。
# 需与torch.version.cuda、torch.cuda.is_available及构建版本共同判断；它不更新参数、不进行张量运算。

# %% 原单元 19
# 访问torch.version模块的HIP构建版本字段，值为str或None。
# 这是属性，不加括号；当前CPU包为None，notebook通常不显示None。
# 没有HIP版本不单独判定GPU品牌或BF16能力，也不进行张量计算。
torch.version.hip

# %% [markdown]
# ## 单元 20（code）：双问号查看实现源码
#
# `torch.cuda.is_bf16_supported??`是IPython的source inspection语法；单问号偏向帮助，双问号尝试给出可获取的实现源码。
# 它不是两个比较运算符，也不调用is_bf16_supported()。IPython把该行转换为pinfo2 magic，通过inspect取得信息；函数源码能否显示取决于对象/安装方式。
# 普通Python解释器会SyntaxError。notebook注释版保留该行，Python阅读版把它作为注释，并注明应在IPython/Jupyter执行。静态核对使用真实IPython转换后的AST，不能直接把原格送给ast.parse。
# 本次用IPython8.37.0实际执行pinfo2并核对得到PyTorch2.5.1的函数源码，另存源码指纹；查看源码本身不是GPU算子运行。
#
# ```ipython
# torch.cuda.is_bf16_supported??
# ```

# %% 原单元 20
# 双问号是IPython源代码查看语法，通过pinfo2展示可获取的函数实现。
# 不等于调用函数；普通Python会报SyntaxError，阅读版将原指令保留为注释。
# 本次实际核对显示的2.5.1源码，不能据此冒充原作者旧版本实现。
# IPython原指令：torch.cuda.is_bf16_supported??

# %% [markdown]
# ## 单元 21（markdown）：BF16三个字段
#
# BF16的s占1位，E占8位，F占7位，总16位。E可存0…255，正常数E=1…254；β=127，实际指数−126…127。
# F存0…127，正常有效数1+F/128；有效二进制精度含隐含1为8位。不要把7个显式小数位理解为7位十进制准确数。
# 与FP32的指数位相同，只说明正常指数范围相同；FP32有23个显式小数位，表示间距更细。

# %% [markdown] 原单元 21
# - 1: 符号位
# - 8：exponent bits（多了三个指数位）
# - 7：mantissa bits

# %% [markdown]
# ## 单元 22（markdown）：范围不是统一精度
#
# 下一节range求极值。不同指数区间的相邻间距仍会改变；最大值和最小正数不能完整描述误差或训练稳定性。

# %% [markdown] 原单元 22
# ### range

# %% [markdown]
# ## 单元 23（markdown）：BF16正常数公式
#
# 正常数1≤E≤254：
# $$x=(-1)^s\,2^{E-127}(1+F/128).$$
# s/E/F为整数标量，128=2^7，输出实数标量。E=127、F=1、s=0时x=1.0078125，即1以上首个BF16数。
# E=0需使用2^−126×F/128；E=255需区分Inf/NaN。解码次序与FP16相同，偏置和小数位不同；没有额外模型forward/loss。

# %% [markdown] 原单元 23
# $$
# (-1)^{\text{sign}}2^{\text{exponent}-127}(1+\frac{\text{mantissa}}{128})
# $$

# %% [markdown]
# ## 单元 24（markdown）：BF16极值与次正规数
#
# 最大有限数s=0、E=254、F=127：2^127×(2−2^−7)≈3.3895313892515355e38；改符号得到最负有限数。
# 最小正正常数为E=1、F=0，即2^−126≈1.17549435e−38。最小正次正规数为E=0、F=1，即2^−126/128=2^−133≈9.18354962e−41。
# 原文列2^−133时讨论的是理论最小正次正规，不是最小正常数。0仍比任何正数小；实际硬件可能flush subnormal。
# 本次CPU张量的位模式与torch.nextafter(0,1)核对该最小正数，但不证明TPU/CUDA kernel也保留它。

# %% [markdown] 原单元 24
# - 最大值
#     - 0 11111110 1111111
# $$
# 2^{254-127}(1+\frac{127}{128})=3.3895313892515355e+38
# $$
#
# - 最小值
#     - 1 11111110 1111111
# $$
# -2^{254-127}(1+\frac{127}{128})=-3.3895313892515355e+38
# $$
#
#     - 正值的最小值
#         - 0 00000000 0000001
#         - $2^{-126}(0+\frac1{128})=2^{-133}$ = 9.183549615799121e-41

# %% [markdown]
# ## 单元 25（markdown）：比较三种格式
#
# 比较固定shape元素的存储、正常指数范围、小数位/间距三项。FP16和BF16每元素都是2 byte，FP32为4 byte。
# BF16与FP32指数位同为8，小数位分别7和23；FP16指数5、小数10。BF16的范围更大但在1附近比FP16更粗。标题不是实验结果。

# %% [markdown] 原单元 25
# ## float16/float32, bf16

# %% [markdown]
# ## 单元 26（code）：显示BF16/FP32/FP16对比图
#
# `Image('../imgs/bf-float.png')`使用Image类，不传width，按图片自身尺寸由前端显示。路径与单元5同样依据工作目录；返回图片对象，不返回FP16 tensor。
# 图自上而下的三行是BF16、FP32、FP16：左侧S为1位，绿色E为指数，红色M为显式小数。数出8/8/5个E和7/23/10个M即可核对，不把颜色当作计算操作。
# FP32行比16位行长，反映存储bit数。没有模型输入输出或训练/推理分支。本次核对展示PNG的哈希/尺寸和原代码参数；图来自固定上游，不是本次性能测量。

# %% 原单元 26
# 读取BF16/FP32/FP16位布局图，返回可显示的Image对象。
# 按原图尺寸显示配图。
# 上到下BF16/FP32/FP16；S=符号、E=指数、M=显式小数，计数见逐格说明。
Image('../imgs/bf-float.png')
