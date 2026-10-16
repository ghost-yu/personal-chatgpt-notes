# 原notebook Python中文注释阅读版：reward_model_Bradley_Terry_model
# 所有原代码保留；绘图使用matplotlib，补充BFGS需要真实SciPy。

# %% 原单元 0 · markdown
# 原参考入口：Bradley–Terry概念、OpenAI InstructGPT介绍和论文。外链是来源，不是指令；本次以固定InstructGPT v1的§3.5/附录C.2为准。
# - references
#     - https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
#     - InstructGPT:
#         - https://openai.com/index/instruction-following/
#         - https://arxiv.org/abs/2203.02155

# %% 原单元 1 · code
# 从IPython.display导入Image类，供notebook展示图片；导入不下载权重。
# 从IPython.display导入Image类，供notebook展示图片；导入不下载权重。
# 导入实际Image展示对象；不验证远程图片内容。
from IPython.display import Image


# %% 原单元 2 · code
# torch/nn/F提供Tensor与稳定logsigmoid，NumPy计算数组，pyplot画函数。这里nn虽导入但原notebook没有神经网络模型/训练；不要编造原模型。
# torch/nn/F提供Tensor与稳定logsigmoid，NumPy计算数组，pyplot画函数。这里nn虽导入但原notebook没有神经网络模型/训练；不要编造原模型。
# 导入本段实际用到的模块或函数，不在此行下载模型。
import torch
# 导入本段实际用到的模块或函数，不在此行下载模型。
from torch import nn
# 导入本段实际用到的模块或函数，不在此行下载模型。
import torch.nn.functional as F
# 导入本段实际用到的模块或函数，不在此行下载模型。
import numpy as np
# 导入本段实际用到的模块或函数，不在此行下载模型。
import matplotlib.pyplot as plt


# %% 原单元 3 · markdown
# InstructGPT标题；本文分析其reward modeling，不将2022年论文当作ChatGPT全部工程实现公开。
# ### instructGPT

# %% 原单元 4 · code
# 远程图Image(url=...)只构造HTML对象。本次未验证远程图内容，不能根据其URL宣称看过图片或把它当实验结果。
# 远程图Image(url=...)只构造HTML对象。本次未验证远程图内容，不能根据其URL宣称看过图片或把它当实验结果。
# 构造HTML图对象；url只是来源或相对路径。
Image(url='https://miro.medium.com/v2/resize:fit:1100/format:webp/1*R8D4ixMVWxSOyvvIoc_T-w.png', width=500)


# %% 原单元 5 · markdown
# 论文Eq.(1)：delta是同prompt下chosen与rejected的标量reward之差；每对损失为-log sigmoid(delta)。binom(K,2)表示K个候选的无序比较对上限；实现要明确sum与mean边界，已经对pair求mean后不能再无意除一次组合数。
# $$
# \text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \left( \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right) \right]
# $$

# %% 原单元 6 · code
# 原OpenAI流程图远程URL保留；只验证Image对象。流程用论文Figure 2的文字说明核对：SFT→偏好排序/训练RM→PPO。预训练基础模型在三个阶段之前，不能把第1阶段直接等同于从头预训练。
# 原OpenAI流程图远程URL保留；只验证Image对象。流程用论文Figure 2的文字说明核对：SFT→偏好排序/训练RM→PPO。预训练基础模型在三个阶段之前，不能把第1阶段直接等同于从头预训练。
# 构造HTML图对象；url只是来源或相对路径。
Image(url='https://images.ctfassets.net/kftzwdyauwt9/690sZjaYwbes9IMEgfyJ7O/f536896b496b13749518f53d4caaa713/Methods_Diagram_dark_mode.jpg?w=2048&q=80&fm=webp', width=600)


# %% 原单元 7 · markdown
# D>C>A=B有4个候选：总无序对6，其中A/B平局。按论文丢弃ties，只留下5个严格偏好对；不把平局随意设成chosen标签1。
# - D>C>A=B

# %% 原单元 8 · code
# 本地quality-labels.png使用url参数构造HTML相对链接；cwd需为本notebook目录。原图实际是单输出质量1–7分和若干属性的标注界面，不是四候选排序界面；原Git字节保留并已视觉查看。
# 本地quality-labels.png使用url参数构造HTML相对链接；cwd需为本notebook目录。原图实际是单输出质量1–7分和若干属性的标注界面，不是四候选排序界面；原Git字节保留并已视觉查看。
# 构造HTML图对象；url只是来源或相对路径。
Image(url='./imgs/quality-labels.png', width=500)


# %% 原单元 9 · markdown
# 单个绝对rating可能受标注人打分尺度影响，pairwise更关注相对选择，但不保证消除所有偏好分歧。siamese类比指同一个参数共享的打分器分别处理(x,yw)/(x,yl)，不等于两个独立reward网络，也不一定是cross-encoder一次输入两答案。
# - 单个的标注（监督学习，直接回归 rating），rating 未必是全局唯一的，
#     - labeler A 认为的 3分在 labeler B 的 4分或者2分，即 rating 的 scale 未必对所有的 labelers 都是适用的；
#     - 李克特量表（Likert scale）
# - reward model
#     - 双输入，siamese network；

# %% 原单元 10 · markdown
# 损失理解标题；后续从概率假设推导，不把分数本身当概率。
# #### 损失函数的理解

# %% 原单元 11 · markdown
# 概率建模标题；先建立人类偏好的Bernoulli变量，再推导似然。
# #### 概率建模

# %% 原单元 12 · markdown
# 假定logit(p)=rw-rl=delta。先指数化得odds，再解p=sigmoid(delta)，最后对观察到chosen胜出的标签1取负log。不是“模型必然已给chosen更大分”的假设；delta可以为负。
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
# p是给定prompt和两响应时人类偏好chosen的模型概率，不是确定标量rw>rl这个事件的随机概率。BCE一般形式含y和1-y；按chosen/rejected定向后y=1。自然对数概率可以用于似然/交叉熵，但看到log并不总是交叉熵。
# - log odds（对数赔率，赔率的对数）：$r_w-r_l=\log odds$
#     - $p=p(y_w\gt y_l)$
#     - 奖励差，表示人类更可能偏好响应1（$r_w$，$r_\text{chosen}$）而非响应2（$r_\ell$，$r_\text{rejected}$）的对数赔率。
# - They use a cross-entropy loss, with the comparisons as labels—the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler
#     - 我们已经假定人类标注者更偏好响应1，设定真实标签 $y=1$；（如果人类更偏好响应2，设定 $y=0$）
#     - 则交叉熵损失为 $-[y\log P+(1-y)\log (1-P)] = -\log P$
# - 多数时候我们看到 log probability 的时候基本就是在计算交叉熵；

# %% 原单元 14 · code
# np.linspace取100个概率点[.01,.99]，shape[100] float64；np.log为自然对数，log_odds=log(p/(1-p))。plt.figure创建画布、plot绘二维曲线、xlabel/ylabel/title/grid只设置显示。不是训练曲线，没有需要学习的参数。
# np.linspace取100个概率点[.01,.99]，shape[100] float64；np.log为自然对数，log_odds=log(p/(1-p))。plt.figure创建画布、plot绘二维曲线、xlabel/ylabel/title/grid只设置显示。不是训练曲线，没有需要学习的参数。
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
# math.comb(9,2)=36是整数无序组合数，乘64返回int2304；不是64×9条独立prompt，也不包括反向重复对。
# math.comb(9,2)=36是整数无序组合数，乘64返回int2304；不是64×9条独立prompt，也不包括反向重复对。
# 导入本段实际用到的模块或函数，不在此行下载模型。
import math
# 无序组合数C(9,2)=36，64个prompt至多2304对。
64 * math.comb(9, 2)


# %% 原单元 18 · markdown
# Bradley–Terry标题；补充notebook给出正强度pi与log强度theta的极大似然估计。
# ### Bradley-Terry model

# %% 原单元 19 · code
# 远程Bradley–Terry示意图只构造HTML，不验证内容；概率公式由原Markdown与实际损失实现独立核对。
# 远程Bradley–Terry示意图只构造HTML，不验证内容；概率公式由原Markdown与实际损失实现独立核对。
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
# 相等分数delta=0：-F.logsigmoid是自然对数loss ln2=.693147；第二项np.log10给.301030。二者对数底不同，不是两个算法给同一单位的loss。该格原样保留，说明中纠正。
# 相等分数delta=0：-F.logsigmoid是自然对数loss ln2=.693147；第二项np.log10给.301030。二者对数底不同，不是两个算法给同一单位的loss。该格原样保留，说明中纠正。
# `r1=r2` 时的 loss
# PyTorch稳定计算log(sigmoid)；第二项log10不是同底数。
-F.logsigmoid(torch.tensor(0.)), -np.log10(1/(1+np.exp(-0)))


# %% 原单元 23 · code
# delta为1000个float64点[-10,10]；用sigmoid、负log和sigma-1得到两条解析函数曲线。蓝线loss非负，红线导数在(-1,0)，不是两次训练。np.exp/先sigmoid再log在极端输入会溢出，补充用稳定softplus/logsigmoid核对。
# delta为1000个float64点[-10,10]；用sigmoid、负log和sigma-1得到两条解析函数曲线。蓝线loss非负，红线导数在(-1,0)，不是两次训练。np.exp/先sigmoid再log在极端输入会溢出，补充用稳定softplus/logsigmoid核对。
# 导入本段实际用到的模块或函数，不在此行下载模型。
import numpy as np
# 导入本段实际用到的模块或函数，不在此行下载模型。
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
# delta=3时loss=.048587、梯度约-.047426，仍非0；不是硬阈值，也不是达3后无法继续优化。有限实数上loss始终随delta减小，极限delta→+inf时趋0；有限精度数值饱和另说。
# - $\delta=r_1-r_2\geq 3$ 以后 loss 接近于0，且基本没有梯度了，；

# %% 原单元 25 · code
# 依次计算delta=1/3/-6的自然对数loss，约.313262/.048587/6.002476；负差惩罚大，对delta梯度趋-1而非无界增大。
# 依次计算delta=1/3/-6的自然对数loss，约.313262/.048587/6.002476；负差惩罚大，对delta梯度趋-1而非无界增大。
-np.log(1/(1+np.exp(-1))), -np.log(1/(1+np.exp(-3))), -np.log(1/(1+np.exp(-(-6))))


# %% 原单元 26 · code
# delta=-6的sigmoid约.002473，表示模型当前认为chosen胜出的概率很低，不改变人工chosen标签；它推动训练修正错误排序。
# delta=-6的sigmoid约.002473，表示模型当前认为chosen胜出的概率很低，不改变人工chosen标签；它推动训练修正错误排序。
1/(1+np.exp(-(-6)))
