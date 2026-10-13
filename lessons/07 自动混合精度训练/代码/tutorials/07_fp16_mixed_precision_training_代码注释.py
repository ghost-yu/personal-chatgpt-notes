# -*- coding: utf-8 -*-
# 第07讲纯文本代码阅读版，不含JSON/历史输出。
# 原类含巨大通道，完整脚本会下载MNIST/分配大网络/训练，不宜直接执行。
# 先阅读逐格说明；meta检查、小型CPU替代与实际AMP验证见复现/README.md。

# %% [markdown]
# ## 单元 0（markdown）：总览标题
#
# 【必须会】本篇用CNN分类演示自动混合精度，不训练LLM。先追踪数据shape，再追踪forward/loss/backward/step与AMP策略。此格只有标题。

# %% [markdown] 原单元 0
# ## basics & summary

# %% [markdown]
# ## 单元 1（markdown）：AMP背景与Markdown示例
#
# 【必须会】Conv2d链接是卷积尺寸依据；其他链接是精度和benchmark背景，原文件没锁这些浮动网页的版本。本次依据固定notebook及PyTorch实现，不自动执行外部benchmark。
# “提升batch_size”需要限定模型、设备、算子和显存条件，不等于准确率提升。下面片段只有scaler构造，不是一套完整训练程序；需要先import torch。
# torch.cuda.amp.GradScaler()创建CUDA梯度缩放状态对象，通常放训练循环外。当前CPU包会警告并禁用它；对象存在不证明启用了FP16或在GPU训练。Markdown原块与注释均核对AST。

# %% [markdown] 原单元 1
# - 回顾下 CNN 结构与输入 shape 的适配
#     - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# - `torch.cuda.amp` 怎么用的；
#     - fp16: [loss scaling](https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407)
#         - https://github.com/mli/transformers-benchmarks/blob/main/transformers.ipynb
#     - 极大地提升 batch_size
#
# ```
# # basic usages
#
# scaler = torch.cuda.amp.GradScaler()
# ```

# %% [markdown]
# ## 单元 2（markdown）：CNN流程标题
#
# 【必须会】分类流程为图像→卷积/激活/池化→flatten→logits→分类损失→更新。没有tokenizer或adapter，此标题不创建模型。

# %% [markdown] 原单元 2
# ## cnn pipeline

# %% [markdown]
# ## 单元 3（code）：导入各模块
#
# 【建议会】torch管tensor/自动微分；nn管注册参数的层；F是functional模块别名，本篇用ReLU。datasets是torchvision数据集模块；transforms是预处理；optim是优化器模块；DataLoader组批；tqdm显示迭代进度。
# from...import导入对象、as取别名，均不下载MNIST、不实例化大CNN。包版本要兼容；本次真实导入torch2.5.1+cpu/torchvision0.20.1+cpu，不用假模块替代导入。

# %% 原单元 3
# 【必须会/建议会】各导入绑定模块/别名；不构造模型、不下载数据。
# 导入torch；后续才创建tensor和调用设备API。
import torch
# 导入functional模块并取别名F；本篇F.relu是无参数激活。
import torch.nn.functional as F
# 数据集模块别名datasets，本篇使用MNIST。
import torchvision.datasets as datasets
# 预处理模块；ToTensor把灰度图转浮点[1,28,28]。
import torchvision.transforms as transforms
# 优化器模块；本篇用Adam。
from torch import optim
# 注册层和损失的模块；用于CNN继承和层构造。
from torch import nn
# DataLoader负责组批；不改变网络结构。
from torch.utils.data import DataLoader
# 进度包装函数，不计算loss或梯度。
from tqdm import tqdm

# %% [markdown]
# ## 单元 4（code）：选择设备但未移动对象
#
# 【必须会】Python条件表达式先查cuda.is_available，选择字符串cuda或cpu，再构造torch.device对象。最后一行显示设备。
# 此格不搬模型/数据，搬运在.to(device)。CPU备用路径不等于CUDA autocast真正启用；本次device=cpu。

# %% 原单元 4
# 【必须会/建议会】创建设备对象，当前CPU；没有移动tensor。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %% [markdown]
# ## 单元 5（markdown）：模型标题
#
# 【必须会】下一格只是定义CNN类，构造层在以后调用CNN(...)时执行。标题没有资源分配。

# %% [markdown] 原单元 5
# ### Model

# %% [markdown]
# ## 单元 6（code）：原CNN定义、参数与shape
#
# 【必须会】class CNN(nn.Module)继承模型基类；__init__(self,in_channels=1,num_classes=10)是初始化方法，self为实例。super(CNN,self).__init__()初始化父类，给self.conv1等赋nn层对象会注册参数。
# Conv2d默认bias=True、groups1、dilation1。原conv1将1→5120通道，conv2将5120→10240；k3/s1/p1保持高宽。MaxPool2d(k2/s2)无训练参数，重复调用同一个pool对象两次，各使边长减半。fc1固定输入10240×7×7，不会自动适配任意图像。
# forward(self,x)接收浮点[N,C,H,W]，model(x)经Module机制调用它；默认MNIST为[N,1,28,28]→[N,5120,28,28]→[N,5120,14,14]→[N,10240,14,14]→[N,10240,7,7]→[N,501760]→[N,10]。
# F.relu保持shape；reshape(x.shape[0],−1)保留batch、推导展平特征数；fc1返回未归一化logits，不在这里softmax。return输出tensor，前向不直接调用optimizer更新权重。
# 原通道巨大，不能在代码注释里悄悄改小。本次原类/forward仅在meta设备核对形状和参数，无真实权重数据；另建8/16通道教学模型用于数值训练，四个常量改动单独审计。

# %% 原单元 6
# 【必须会/建议会】原大CNN类保持5120/10240通道，只有元数据验证，不能盲目实例化。
# Simple CNN
# 继承nn.Module，__init__注册层，forward定义数据路径。
class CNN(nn.Module):
    # 实例初始化，默认1通道/10类；self不是外部数据。
    def __init__(self, in_channels=1, num_classes=10):
        # 初始化父类Module，让参数/子模块注册机制可用。
        super(CNN, self).__init__()
        # 构建第一层Conv2d，原输出通道5120。
        self.conv1 = nn.Conv2d(
            # 该层输入通道数，需与数据或前一层匹配。
            in_channels=in_channels,
            # 输出特征通道数，不是类别数；会显著影响参数/激活规模。
            out_channels=5120,
            # 卷积核边长3，参数shape含3×3。
            kernel_size=3,
            # 卷积步长1，结合padding保持高宽。
            stride=1,
            # 每边补1，3×3卷积后高宽不变。
            padding=1,
        )
        # /2, downsampling
        # 无参数2×2/步长2最大池化，同对象调用两次。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二层输入必须对齐conv1的5120，输出10240。
        self.conv2 = nn.Conv2d(
            # 该层输入通道数，需与数据或前一层匹配。
            in_channels=5120,
            # 输出特征通道数，不是类别数；会显著影响参数/激活规模。
            out_channels=10240,
            # 卷积核边长3，参数shape含3×3。
            kernel_size=3,
            # 卷积步长1，结合padding保持高宽。
            stride=1,
            # 每边补1，3×3卷积后高宽不变。
            padding=1,
        )
        # (channels*w*h)
            # w, h: 取决于初始的 width, height
        # 输入固定10240×7×7，28×28经过两次池化才适配。
        self.fc1 = nn.Linear(10240 * 7 * 7, num_classes)

    # 接收NCHW浮点tensor；model(x)自动进入该方法。
    def forward(self, x):
        # 卷积改变通道，ReLU保持shape；不是softmax。
        x = F.relu(self.conv1(x))
        # /2
        # 高宽各减半，28→14→7。
        x = self.pool(x)
        # 卷积改变通道，ReLU保持shape；不是softmax。
        x = F.relu(self.conv2(x))
        # /2
        # 高宽各减半，28→14→7。
        x = self.pool(x)
        # 4d => 2d, (bs, features)
        # 保留batch维，−1推断展平特征数，不自动改变fc权重。
        x = x.reshape(x.shape[0], -1)
        # 二维特征→logits[N,10]，用于CrossEntropyLoss。
        x = self.fc1(x)
        # 返回logits，不执行优化器更新。
        return x

# %% [markdown]
# ## 单元 7（code）：模型摘要工具
#
# 【建议会】从torchsummary导入summary函数，此格不执行模型。summary通常会构造虚拟输入并用hook运行前向、列参数量，不能理解为纯静态打印；本次只核对原导入。

# %% 原单元 7
# 【必须会/建议会】导入summary函数；实际摘要会进行虚拟forward。
from torchsummary import summary

# %% [markdown]
# ## 单元 8（code）：RGB224与固定fc不匹配
#
# 【必须会】CNN(in_channels=3)只改第一层输入通道；仍是5120/10240大通道与7×7的fc。summary的input_size=(3,224,224)不含batch维，batch_size32主要用于报告估算；torchsummary1.5.1实际虚拟前向使用2样本，不保证实际运行32样本。
# 224经两次池化到56，flatten为[N,32112640]，fc权重[10,501760]不匹配。原保存summary显示8/64通道、56×56fc，是另一状态，不能当作现在顺序重跑成功。
# 本次不在CPU创建巨大真实模型或执行本格原调用，用meta输入核对失败位置。小型28×28模型可真实运行summary，但与原RGB224调用分开。

# %% 原单元 8
# 【必须会/建议会】原RGB224调用与固定7×7fc不匹配，历史摘要来自不同类状态。
# 实例化原大CNN再移动设备，可能耗大量内存；本次整体格不执行。
model = CNN(in_channels=3)
# 工具会虚拟forward；RGB224不匹配当前fc，历史输出不能代替重跑。
summary(model, input_size=(3, 224, 224), batch_size=32, device='cpu')

# %% [markdown]
# ## 单元 9（markdown）：进入训练数据和循环
#
# 【必须会】接下来定义MNIST设置、数据加载，再分别重建FP32和AMP训练。反复重定义同名变量/函数会替换当前绑定，不会自动延续上一次模型权重。

# %% [markdown] 原单元 9
# ### training pipeline

# %% [markdown]
# ## 单元 10（code）：初始超参数
#
# 【必须会】in_channels1配灰度，num_classes10配数字标签；learning_rate=3e−4是float，batch_size32、num_epochs3为int。Karpathy’s constant是经验昵称，不保证普遍最优。
# 只赋变量，不训练或分配batch。14/17格随后分别把batch改128/256并重建对象；本次直接执行本格赋值核对，再在教学上下文显式用N8、batch4、epoch1。

# %% 原单元 10
# 【必须会/建议会】本格batch32；后面128/256覆盖它，不代表连续使用同一loader。
# 对齐 mnist
in_channels = 1
num_classes = 10

# 学习率3e−4是经验设置；不是GradScaler的scale。
learning_rate = 3e-4 # karpathy's constant
# 每批样本数，32/128/256分别是不同重建阶段。
batch_size = 32
# 遍历数据3轮，batch不同意味着更新次数也不同。
num_epochs = 3

# %% [markdown]
# ## 单元 11（code）：MNIST与DataLoader
#
# 【必须会】MNIST(root='dataset/',train=True/False,transform=ToTensor(),download=True)返回训练/测试Dataset，可能在root路径下载文件。ToTensor是可调用预处理对象：uint8灰度图→float32[1,28,28]，通常像素除255。
# DataLoader(dataset=...,batch_size=...,shuffle=True)返回组批迭代器，图像[N,1,28,28]，标签int64[N]；末批N可能较小。两组都shuffle不改变准确率定义，但测试通常不需要洗牌。
# FashionMNIST整段仍被注释，不执行；本次不下载MNIST、不偷偷换Dataset类。独立补充用8个可复现合成样本和真实TensorDataset/DataLoader，原完整数据加载格未执行。

# %% 原单元 11
# 【必须会/建议会】原MNIST可下载数据；FashionMNIST仍注释，教学合成数据另建。
# 训练分割Dataset，download=True可能写数据到root目录。
train_dataset = datasets.MNIST(
    # ToTensor生成浮点灰度tensor；train选择数据分割。
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
# 测试分割Dataset，预处理同训练，但不用于更新。
test_dataset = datasets.MNIST(
    # ToTensor生成浮点灰度tensor；train选择数据分割。
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
# 按batch_size组批并洗牌，返回迭代器。
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 原测试也shuffle，准确率仍按所有样本计数。
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# train_dataset = datasets.FashionMNIST(
#     root="dataset/", train=True, transform=transforms.ToTensor(), download=True
# )
# test_dataset = datasets.FashionMNIST(
#     root="dataset/", train=False, transform=transforms.ToTensor(), download=True
# )
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# %% [markdown]
# ## 单元 12（code）：取一批并解包
#
# 【必须会】iter(train_loader)构造迭代器，next取首批，二元解包绑定batch_x与batch_y；print读取shape。
# 图像float32[N,1,28,28]，标签int64[N]。重新iter通常从新一轮迭代开始，可影响shuffle/RNG状态，不是固定的“下一批”。本次在合成loader上真实执行原语句，记录[4,1,28,28]与[4]。

# %% 原单元 12
# 【必须会/建议会】iter/next取批次并解包，图像NCHW、标签一维整数。
# 取一个批次并二元解包；重建iter不是“固定下一批”。
batch_x, batch_y = next(iter(train_loader))
print(batch_x.shape, batch_y.shape)

# %% [markdown]
# ## 单元 13（markdown）：FP32分支标题
#
# 【必须会】标题本身不强制dtype。下一格新建默认FP32模型、优化器和数据，训练函数中没有autocast，因此作为普通精度基线。

# %% [markdown] 原单元 13
# #### float 32

# %% [markdown]
# ## 单元 14（code）：普通训练与evalute
#
# 【必须会】本格重新设batch128、epoch3，重复构造数据/loader/CNN，再.to(device)。criterion=CrossEntropyLoss()需要float logits[N,10]和int64标签[N]，返回标量loss，不先softmax；optimizer=Adam(model.parameters(),lr=...)持有参数引用与逐步状态。
# train()为普通函数，读取全局model/loader/device/criterion/optimizer/num_epochs，不显式返回。双循环遍历epoch与batch_idx；tqdm只显示进度，不训练模型。batch_idx是序号而非图像标签。
# batch_x/y.to(device)返回在该设备的tensor；model→logits→criterion→loss。zero_grad清历史梯度（不改权重），backward沿图累计梯度，step才更新权重/Adam moment。zero_grad放forward后、backward前仍有效；原scaler代码是注释。
# evalute是原函数拼写，输入model/test_loader，返回0维准确率tensor。eval改变模块模式，no_grad只关闭梯度记录；logits.max(1)返回最大值与索引，_忽略值，preds[N]与标签比较再sum，size(0)累加样本数。最后无条件model.train()，不是恢复调用前模式。
# 原完整格含大CNN构造、下载与末尾train()，本次不执行整体。提取原train/evalute定义不改函数体，在明确小模型/合成loader/epoch1上下文执行，实际更新参数与评估；不冒充MNIST准确率。

# %% 原单元 14
# 【必须会/建议会】完整原格会下载/构建大模型并train；仅函数体在小型教学上下文验证。
# 对齐 mnist
in_channels = 1
num_classes = 10

# 学习率3e−4是经验设置；不是GradScaler的scale。
learning_rate = 3e-4 # karpathy's constant
# 每批样本数，32/128/256分别是不同重建阶段。
batch_size = 128
# 遍历数据3轮，batch不同意味着更新次数也不同。
num_epochs = 3

# 训练分割Dataset，download=True可能写数据到root目录。
train_dataset = datasets.MNIST(
    # ToTensor生成浮点灰度tensor；train选择数据分割。
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
# 测试分割Dataset，预处理同训练，但不用于更新。
test_dataset = datasets.MNIST(
    # ToTensor生成浮点灰度tensor；train选择数据分割。
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
# 按batch_size组批并洗牌，返回迭代器。
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 原测试也shuffle，准确率仍按所有样本计数。
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# 实例化原大CNN再移动设备，可能耗大量内存；本次整体格不执行。
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
# 交叉熵接收未softmax的logits[N,10]和int64标签[N]。
criterion = nn.CrossEntropyLoss()
# Adam保存参数引用及moment状态，更新发生于step。
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# before training
# scaler = torch.cuda.amp.GradScaler()

# 普通函数，读取全局训练对象并修改权重/优化器/scaler，默认返回None。
def train():
    # tqdm仅包装epoch循环并显示进度。
    for epoch in tqdm(range(num_epochs)):
        # enumerate给批次加序号，内部解包图像/标签。
        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
            # 移动图像到device，shape不变。
            batch_x = batch_x.to(device)
            # 移动标签，仍为整数[N]。
            batch_y = batch_y.to(device)

            # forward
            # 调用Module.forward，得到未归一化分类分数。
            logits = model(batch_x)
            # 得到标量平均CE，backward需要这个计算图。
            loss = criterion(logits, batch_y)

            # backward
            # 清上批累积梯度，不更新参数；forward后/backward前仍有效。
            optimizer.zero_grad()
            # 沿图累计参数梯度，不直接更新权重。
            loss.backward()
#             scaler.scale(loss).backward()

            # gradient descent
            # FP32分支用梯度更新参数与Adam状态。
            optimizer.step()
#             scaler.step(optimizer)
#             scaler.update()

# 保留原拼写，返回0维准确率tensor，最后无条件切train。
def evalute(model, test_loader):
    # Python整数起步，后面加tensor变成tensor。
    total_correct = 0
    # 样本数分母，不是批次数。
    total_samples = 0
    # 设置模块评估模式，不等于关闭autograd。
    model.eval()

    # 上下文关闭梯度记录，不更新权重。
    with torch.no_grad():
        # 逐批评估，原这里没有autocast。
        for batch_x, batch_y in test_loader:
            # 移动图像到device，shape不变。
            batch_x = batch_x.to(device)
            # 移动标签，仍为整数[N]。
            batch_y = batch_y.to(device)

            # 调用Module.forward，得到未归一化分类分数。
            logits = model(batch_x)
            # max(1)沿类别维返回值/索引，_忽略值，preds[N]。
            _, preds = logits.max(1)
            # 相等比较得到bool[N]，sum累计正确个数。
            total_correct += (preds == batch_y).sum()
            # size(0)读取实际批样本数，末批可小于设定batch。
            total_samples += batch_y.size(0)
    # 无条件恢复train，不保留函数进入前的模式。
    model.train()
    # 正确数/样本数，返回准确率比例，而非百分数。
    return total_correct/total_samples

# 执行当前定义训练函数；原整体格会真实训练大CNN。
train()

# %% [markdown]
# ## 单元 15（code）：FP32评估输出
#
# 【建议会】先调用evalute(model,train_loader)，再调用测试loader，乘100后f字符串:.2f保留百分数两位小数。此格只评估与print，不更新参数。
# 每次evalute最后都将模型设train；历史本格输出为空，不能编造老师的准确率。补充在合成样本上执行原打印，只证明统计流程，不作为MNIST质量结果。

# %% 原单元 15
# 【必须会/建议会】调用评估并打印百分数；不更新权重，历史输出为空。
# 调用evalute后乘100，:.2f把百分数显示两位小数。
print(f"Accuracy on training set: {evalute(model, train_loader)*100:.2f}")
# 调用evalute后乘100，:.2f把百分数显示两位小数。
print(f"Accuracy on test set: {evalute(model, test_loader)*100:.2f}")

# %% [markdown]
# ## 单元 16（markdown）：进入自动混合精度
#
# 【必须会】AMP按算子选dtype，模型主参数仍可FP32；不是永久model.half()。FP16动态范围较窄，梯度缩放处理部分数值问题，不自动提高分类准确率。

# %% [markdown] 原单元 16
# #### 混合精度训练

# %% [markdown]
# ## 单元 17（code）：AMP训练的顺序与状态
#
# 【必须会】本格把batch改256，重新创建数据、CNN、criterion、Adam及GradScaler，覆盖之前同名train/evalute。不是在上一FP32模型上接着训练。
# scaler在循环外创建，保存scale状态。with torch.cuda.amp.autocast()是上下文管理器：只包forward/loss，进入时选择算子精度，退出恢复；没有把backward包在里面。
# scale(loss)返回与计算图连接的放大标量，backward得到放大梯度；scaler.step(optimizer)还原尺度并检查Inf/NaN，必要时不调用更新；update调整scale。不要再额外optimizer.step()，原该行仍是注释。
# 公式L̃=aL、g̃=a∇L，恢复g=g̃/a；a不是学习率。极小梯度1e−8乘1024变约1.024e−5，更可能被FP16保留，但放大过度也可Inf，需要动态回退。
# evalute与FP32分支相同，没有autocast，不能因AMP训练batch能放下就假定同batch FP32评估也能放下。CPU上原CUDA autocast/scaler会警告并禁用；这种跑通不是AMP开启。
# 本次另外明确改autocast为torch.amp.autocast('cpu',dtype=torch.float16)，scaler用CPU控制器，在同小模型上核对真实FP16输出/FP32loss与梯度、有效更新及非有限梯度跳步。不是3090显存实验。

# %% 原单元 17
# 【必须会/建议会】完整原AMP格重新建模型；CPU原CUDA AMP自动禁用，补充开启CPU FP16另记。
in_channels = 1
num_classes = 10

# 学习率3e−4是经验设置；不是GradScaler的scale。
learning_rate = 3e-4 # karpathy's constant
# 每批样本数，32/128/256分别是不同重建阶段。
batch_size = 256
# 遍历数据3轮，batch不同意味着更新次数也不同。
num_epochs = 3

# 训练分割Dataset，download=True可能写数据到root目录。
train_dataset = datasets.MNIST(
    # ToTensor生成浮点灰度tensor；train选择数据分割。
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
# 测试分割Dataset，预处理同训练，但不用于更新。
test_dataset = datasets.MNIST(
    # ToTensor生成浮点灰度tensor；train选择数据分割。
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
# 按batch_size组批并洗牌，返回迭代器。
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 原测试也shuffle，准确率仍按所有样本计数。
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 实例化原大CNN再移动设备，可能耗大量内存；本次整体格不执行。
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
# 交叉熵接收未softmax的logits[N,10]和int64标签[N]。
criterion = nn.CrossEntropyLoss()
# Adam保存参数引用及moment状态，更新发生于step。
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# before training
# CUDA GradScaler创建于循环外；CPU包禁用，补充CPU控制器另列。
scaler = torch.cuda.amp.GradScaler()

# 普通函数，读取全局训练对象并修改权重/优化器/scaler，默认返回None。
def train():
    # tqdm仅包装epoch循环并显示进度。
    for epoch in tqdm(range(num_epochs)):
        # enumerate给批次加序号，内部解包图像/标签。
        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
            # 移动图像到device，shape不变。
            batch_x = batch_x.to(device)
            # 移动标签，仍为整数[N]。
            batch_y = batch_y.to(device)

            # forward
            # 只在forward/loss区域自动选择算子dtype，退出恢复策略。
            with torch.cuda.amp.autocast():
                # 调用Module.forward，得到未归一化分类分数。
                logits = model(batch_x)
                # 得到标量平均CE，backward需要这个计算图。
                loss = criterion(logits, batch_y)

            # backward
            # 清上批累积梯度，不更新参数；forward后/backward前仍有效。
            optimizer.zero_grad()
            # loss scaling
            # 放大loss后反传；梯度仍需step阶段恢复尺度。
            scaler.scale(loss).backward()

            # gradient descent
#             optimizer.step()
            # 还原梯度/检查非有限值，必要时跳过真正optimizer.step。
            scaler.step(optimizer)
            # 根据检测结果增大/回退scale，保留到下一批。
            scaler.update()

# 保留原拼写，返回0维准确率tensor，最后无条件切train。
def evalute(model, test_loader):
    # Python整数起步，后面加tensor变成tensor。
    total_correct = 0
    # 样本数分母，不是批次数。
    total_samples = 0
    # 设置模块评估模式，不等于关闭autograd。
    model.eval()

    # 上下文关闭梯度记录，不更新权重。
    with torch.no_grad():
        # 逐批评估，原这里没有autocast。
        for batch_x, batch_y in test_loader:
            # 移动图像到device，shape不变。
            batch_x = batch_x.to(device)
            # 移动标签，仍为整数[N]。
            batch_y = batch_y.to(device)

            # 调用Module.forward，得到未归一化分类分数。
            logits = model(batch_x)
            # max(1)沿类别维返回值/索引，_忽略值，preds[N]。
            _, preds = logits.max(1)
            # 相等比较得到bool[N]，sum累计正确个数。
            total_correct += (preds == batch_y).sum()
            # size(0)读取实际批样本数，末批可小于设定batch。
            total_samples += batch_y.size(0)
    # 无条件恢复train，不保留函数进入前的模式。
    model.train()
    # 正确数/样本数，返回准确率比例，而非百分数。
    return total_correct/total_samples

# 执行当前定义训练函数；原整体格会真实训练大CNN。
train()

# %% [markdown]
# ## 单元 18（code）：AMP训练后的评估
#
# 【建议会】打印表达式与单元15相同，但当前模型/loader绑定来自17。evalute内部没有autocast；本次在小型合成数据上下文验证原打印、no_grad与模式变化。
# 原保存输出为空，没有实验准确率数值。评估返回tensor可格式化，百分比乘100不是新的指标。

# %% 原单元 18
# 【必须会/建议会】调用17定义的evalute；原评估没有autocast。
# 调用evalute后乘100，:.2f把百分数显示两位小数。
print(f"Accuracy on training set: {evalute(model, train_loader)*100:.2f}")
# 调用evalute后乘100，:.2f把百分数显示两位小数。
print(f"Accuracy on test set: {evalute(model, test_loader)*100:.2f}")
