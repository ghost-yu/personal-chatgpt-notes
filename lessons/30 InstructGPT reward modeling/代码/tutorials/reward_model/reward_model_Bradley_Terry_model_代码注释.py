# 原notebook Python中文注释阅读版：reward_model_Bradley_Terry_model
# 所有原代码保留；绘图使用matplotlib，补充BFGS需要真实SciPy。

# %% 原单元 0 · markdown
# 参考资料包括 Bradley–Terry 概念、OpenAI InstructGPT 介绍和论文；主要对应 InstructGPT v1 的§3.5与附录C.2。
# - references
#     - https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
#     - InstructGPT:
#         - https://openai.com/index/instruction-following/
#         - https://arxiv.org/abs/2203.02155

# %% 原单元 1 · code
# 从 IPython.display 导入 Image 类，供 notebook 展示图片。
# Image 根据图片路径或 URL 创建可显示的图像对象。
from IPython.display import Image


# %% 原单元 2 · code
# torch 提供张量，F 提供 logsigmoid 等函数，NumPy 计算数组，pyplot 绘制概率与损失曲线；nn 用于构建神经网络模块。
# 导入后续计算使用的模块或函数。
import torch
# 导入后续计算使用的模块或函数。
from torch import nn
# 导入后续计算使用的模块或函数。
import torch.nn.functional as F
# 导入后续计算使用的模块或函数。
import numpy as np
# 导入后续计算使用的模块或函数。
import matplotlib.pyplot as plt


# %% 原单元 3 · markdown
# InstructGPT 这一节介绍奖励模型在三阶段训练路线中的位置。
# ### instructGPT

# %% 原单元 4 · code
# Image(url=...) 创建远程图片的显示对象，width=500 指定展示宽度。正文内嵌了论文三阶段流程图，可结合阅读。
# 构造HTML图对象；url只是来源或相对路径。
Image(url='https://miro.medium.com/v2/resize:fit:1100/format:webp/1*R8D4ixMVWxSOyvvIoc_T-w.png', width=500)


# %% 原单元 5 · markdown
# 论文公式1以同题 chosen/rejected 的分差 delta 计算每对损失 -log sigmoid(delta)。binom(K,2) 是 K 个候选的无序比较对上限；实现时先计算每对损失，再按约定对比较和问题取均值。
# $$
# \text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \left( \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right) \right]
# $$

# %% 原单元 6 · code
# 展示 OpenAI 的训练流程图，width=600 控制宽度。顺序为 SFT→偏好排序与RM训练→PPO；三阶段都从已有的预训练语言能力出发。正文已内嵌论文 Figure 2。
# 构造HTML图对象；url只是来源或相对路径。
Image(url='https://images.ctfassets.net/kftzwdyauwt9/690sZjaYwbes9IMEgfyJ7O/f536896b496b13749518f53d4caaa713/Methods_Diagram_dark_mode.jpg?w=2048&q=80&fm=webp', width=600)


# %% 原单元 7 · markdown
# D>C>A=B 有4个候选、6个无序对；A/B平局。按论文去掉平局后，保留 C>A、C>B、D>A、D>B、D>C 五个严格偏好对。
# - D>C>A=B

# %% 原单元 8 · code
# quality-labels.png 展示单个回答的1–7分质量评分及属性标注。url 使用相对路径，工作目录应位于 notebook 所在文件夹；正文也直接展示了这张图。
# 构造HTML图对象；url只是来源或相对路径。
Image(url='./imgs/quality-labels.png', width=500)


# %% 原单元 9 · markdown
# 绝对评分依赖标注者的分数尺度，成对比较直接记录同题两个回答的相对选择。这里用 siamese 作类比：同一个打分器共享参数，分别处理 (x,yw) 与 (x,yl)，得到两个分数。
# - 单个的标注（监督学习，直接回归 rating），rating 未必是全局唯一的，
#     - labeler A 认为的 3分在 labeler B 的 4分或者2分，即 rating 的 scale 未必对所有的 labelers 都是适用的；
#     - 李克特量表（Likert scale）
# - reward model
#     - 双输入，siamese network；

# %% 原单元 10 · markdown
# 这一节从偏好概率推导损失，并解释分数与概率的转换。
# #### 损失函数的理解

# %% 原单元 11 · markdown
# 概率建模标题；先建立人类偏好的Bernoulli变量，再推导似然。
# #### 概率建模

# %% 原单元 12 · markdown
# 设 logit(p)=rw-rl=delta，先求赔率，再解出 p=sigmoid(delta)。观察到 chosen 胜出时，损失为 -log(p)。delta 可以为负，表示当前模型把 chosen 排在了后面。
# $$
# \begin{split}
# \delta=r_w-r_l&=\log {odds}=\log \frac{p}{1-p}\\
# &\Downarrow\\
# p&=\frac{1}{1+\exp(-\delta)}=\sigma(\delta)\\
# &\Downarrow\\
# &-\log p=-\log\sigma(\delta)
# \end{split}
# $$

# %% 原单元 13 · markdown
# p 表示给定问题与两回答时，人类选择 chosen 的模型概率。二元交叉熵一般包含 y 和 1-y 两项；按 chosen/rejected 定向存储后 y=1，保留 -log(p) 这一项。
# - log odds（对数赔率，赔率的对数）：$r_w-r_l=\log odds$
#     - $p=p(y_w\gt y_l)$
#     - 奖励差，表示人类更可能偏好响应1（$r_w$，$r_\text{chosen}$）而非响应2（$r_\ell$，$r_\text{rejected}$）的对数赔率。
# - They use a cross-entropy loss, with the comparisons as labels—the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler
#     - 我们已经假定人类标注者更偏好响应1，设定真实标签 $y=1$；（如果人类更偏好响应2，设定 $y=0$）
#     - 则交叉熵损失为 $-[y\log P+(1-y)\log (1-P)] = -\log P$
# - 多数时候我们看到 log probability 的时候基本就是在计算交叉熵；

# %% 原单元 14 · code
# np.linspace 取0.01到0.99之间的100个概率点；np.log 计算自然对数，得到 log_odds=log(p/(1-p))。figure 创建画布，plot 画出概率与 log odds 的关系，后续函数设置轴标签、标题和网格。
# 创建包含两端点的均匀float64数组，长度为num参数。
p = np.linspace(0.01, 0.99, 100)
# 自然对数赔率；p/(1-p)是odds，结果不是概率。
log_odds = np.log(p / (1 - p))  # Calculate log odds

# Plotting the log odds curve
# 创建新图；figsize单位是英寸。
plt.figure(figsize=(4, 3))
# 绘制横轴与纵轴同长度的数组；label用于图例。
plt.plot(p, log_odds)
# 设置横轴含义，不参与数值计算。
plt.xlabel('Probability (p)')
# 设置纵轴含义；区分loss与梯度的数值。
plt.ylabel('Log Odds')
# 设置图标题，不是模型训练名称。
plt.title('Log Odds Curve')
# 显示辅助网格。
plt.grid(True)


# %% 原单元 15 · markdown
# 多个候选标题；同prompt候选间比较高度相关，组合数不等于独立样本信息量。
# ### multiple responses

# %% 原单元 16 · markdown
# K=9最多36对、64个prompt最多2304对，K=4最多6对。论文K范围4–9、ties丢弃。把每prompt的全部比较放入同一batch元素可复用每个候选的一次前向；不应据此断言“多造pair一定抗过拟合”或跨prompt大小不同也均匀加权。
# $$
# k=9
# $$
# - 可以很大的膨胀数据集的规模
#     - batch size 64，k = 9
#     - $\leq 64 \cdot \binom{9}{2}=2304$
# - 这里取K=9，是考虑到人工标注的时候，很大一部分时间是花在读懂这个prompt。而读懂prompt和一两个答案之后，其它答案就理解的很快了。所以排序9个比排序4个所花的时间并不是增加了一倍。而且 $C_9^2=36,C_4^2=6$，等于额外开销不到一倍，而标注的数据多了6倍。
# - 另外之前的工作不仅是 K=4 ，而且在标注的时候只标注最好的一个 ，这样就是4选1，这样只需要最大化最优答案的分数就行，所以容易过拟合。而这里采用的是
#  $C_k^2$ 个排序对，使得整个问题变得复杂一点，缓解了过拟合

# %% 原单元 17 · code
# math.comb(9,2)=36，表示9个候选的无序比较对数；64个问题最多形成64×36=2304个比较。
# 导入后续计算使用的模块或函数。
import math
# 无序组合数C(9,2)=36，64个prompt至多2304对。
64 * math.comb(9, 2)


# %% 原单元 18 · markdown
# Bradley–Terry标题；补充notebook给出正强度pi与log强度theta的极大似然估计。
# ### Bradley-Terry model

# %% 原单元 19 · code
# 创建 Bradley–Terry 示意图的显示对象，宽度为400。概率关系与推导见正文第2节。
# 构造HTML图对象；url只是来源或相对路径。
Image(url='https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0605ace1-cc21-400a-941f-154a1faa103c_1600x1427.png', width=400)


# %% 原单元 20 · markdown
# 损失对delta的导数标题；需要分清对分差、对两个分数、对网络参数的导数。
# #### gradient of loss

# %% 原单元 21 · markdown
# ell=softplus(-delta)，ell_prime=sigmoid(delta)-1，ell_second=sigmoid(delta)*(1-sigmoid(delta))。原“loss is proportional to log odds”不准确：delta才是log odds，loss是其非线性函数。
# $$
# \begin{split}
# &\ell=-\log\sigma(\delta)\\
# &\ell'=\sigma(\delta)-1
# \end{split}
# $$
#
# - loss is proportional to the log odds that the first resp is greater than the second;

# %% 原单元 22 · code
# 相等分数时 delta=0。第一项使用自然对数，结果 ln2≈0.693147；第二项用 log10，结果约0.301030。把后者乘以 ln10，即可统一到自然对数单位。
# `r1=r2` 时的 loss
# PyTorch稳定计算log(sigmoid)；第二项log10不是同底数。
-F.logsigmoid(torch.tensor(0.)), -np.log10(1/(1+np.exp(-0)))


# %% 原单元 23 · code
# 在[-10,10]上取1000个 float64 分差，计算 sigmoid、负对数损失和导数 sigma-1。蓝线展示非负的损失，红色虚线展示(-1,0)内的导数。极端分差的稳定计算写法见正文 logsigmoid 例子。
# 导入后续计算使用的模块或函数。
import numpy as np
# 导入后续计算使用的模块或函数。
import matplotlib.pyplot as plt

# Define the delta range
# 创建包含两端点的均匀float64数组，长度为num参数。
delta = np.linspace(-10, 10, 1000)

# Define the function -log(sigma(delta)) and its gradient
# sigmoid(delta)是偏好概率；逐元素exp，大负delta可能溢出。
sigma = 1 / (1 + np.exp(-delta))
# sigmoid(delta)是偏好概率；逐元素exp，大负delta可能溢出。
neg_log_sigma = -np.log(sigma)
# sigmoid(delta)是偏好概率；逐元素exp，大负delta可能溢出。
grad_neg_log_sigma = sigma-1  # derivative of -log(sigma(delta)) with respect to delta

# Plotting
# 创建新图；figsize单位是英寸。
plt.figure(figsize=(5, 3))
# 绘制横轴与纵轴同长度的数组；label用于图例。
plt.plot(delta, neg_log_sigma, label='$-\log(\sigma(\delta))$', color='blue')
# 绘制横轴与纵轴同长度的数组；label用于图例。
plt.plot(delta, grad_neg_log_sigma, label="Gradient of $-\log(\sigma(\delta))$", color='red', linestyle='--')

# 在指定y处绘水平参考线。
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
# 在指定x处绘垂直参考线。
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')

# 设置图标题，不是模型训练名称。
plt.title('Plot of $-log(\sigma(\delta))$ and its Gradient')
# 设置横轴含义，不参与数值计算。
plt.xlabel('delta')
# 设置纵轴含义；区分loss与梯度的数值。
plt.ylabel('Function Value / Gradient')
# 显示前面plot的label图例。
plt.legend()
# 显示辅助网格。
plt.grid(True)
# 展示真实matplotlib图；本次Agg后端另保存同一figure。
plt.show()


# %% 原单元 24 · markdown
# delta=3 时 loss≈0.048587，梯度≈-0.047426。随着 delta 继续增大，损失与梯度幅度平滑减小，loss 在 delta→+inf 时趋于0。
# - $\delta=r_1-r_2\geq 3$ 以后 loss 接近于0，且基本没有梯度了，；

# %% 原单元 25 · code
# 依次计算delta=1/3/-6的自然对数loss，约.313262/.048587/6.002476；负差惩罚大，对delta梯度趋-1而非无界增大。
-np.log(1/(1+np.exp(-1))), -np.log(1/(1+np.exp(-3))), -np.log(1/(1+np.exp(-(-6))))


# %% 原单元 26 · code
# delta=-6的sigmoid约.002473，表示模型当前认为chosen胜出的概率很低，不改变人工chosen标签；它推动训练修正错误排序。
1/(1+np.exp(-(-6)))
