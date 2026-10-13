# -*- coding: utf-8 -*-
"""全部单元说明 + 原语句前插入中文注释；不修改代码、元数据或历史输出。"""
import copy
import json
from pathlib import Path

# 每格说明明确该格目的、变量或API行为及与前后格的联系。
NOTES = {
0: ('导入显示、数组与张量库', '【必须会】`from IPython.display import Image` 导入图片显示类；`import numpy as np` 给NumPy模块起别名；`torch`提供Tensor算术；`from torch import nn`导入神经网络命名空间，但本数值示例没有用它定义网络。导入执行模块初始化，不会加载OPT权重。后续先用torch演示单向量，再用NumPy演示矩阵。'),
1: ('基础量化入口', '【必须会】basics即基础。先弄清缩放、舍入、重建，再理解矩阵量化；当前Markdown不执行Python。'),
2: ('浮点位布局配图', '【建议会】`Image`是IPython显示对象，filename为相对当前工作目录的`../imgs/floats-rep.png`；`width=500`是展示宽度，不把算法输入变成500维。图片返回对象由Jupyter显示；脚本运行时最后一个表达式不会自动显示。应从`代码/tutorials`作为工作目录运行，三张原图已放到相邻`代码/imgs`。'),
3: ('论文、库与异常特征', '【必须会】LLM.int8论文组合vector-wise quantization（向量级量化）和mixed-precision decomposition（混合精度分解）。bitsandbytes是相关开源实现；QLoRA是相关作者后续工作，不能把本讲等同于QLoRA。原英文emergent features指随模型规模涌现、幅度很大且系统出现的特征；少量这样的特征可能对注意力和预测重要。论文、源码的固定版本与实验边界见主解释；本格链接不是安装命令。'),
4: ('单向量演示标题', '【必须会】从一个向量的最大幅度开始，观察一套尺度如何影响每个元素；不是网络训练。'),
5: ('量化与信息损失配图', '【建议会】读取`../imgs/quant-freeze.png`，宽度400像素；展示量化示意，不执行压缩内核，不证明推理准确率。配图与第9–12格算术对照。'),
6: ('缩放与逆缩放公式', '【必须会】设x为浮点元素，x_max为向量absmax，factor=127/x_max。第一式把x缩放成y，第二式逆缩放；原公式没有写round，因此仅缩放再逆缩放可代数抵消。实际有损量化是q=round(127*x/x_max)，再重建x_hat=q*x_max/127。x_max=0时要单独处理，原格没有保护；127对应对称范围[-127,127]，不是int8全部256种编码。'),
7: ('生成随机向量', '【必须会】`torch.randn(5)`函数默认生成CPU浮点Tensor，shape=[5]、标准正态分布；本环境默认float32。未设seed时每次值不同。末行`a`由notebook展示；赋值将变量绑定到新Tensor，下一格求尺度。'),
8: ('求整个向量absmax', '【必须会】`torch.abs(a)`逐元素取绝对值；结果的`.max()`是Tensor方法，不传dim时归约所有元素，返回0维Tensor[]。`a_max`是幅度，不是最大带符号值；不改变原a。它会广播到第9格的所有5个元素。'),
9: ('把幅度缩到127范围', '【必须会】`a/a_max`逐元素除以共享尺度，非零a_max时落在[-1,1]；乘127得到`scaled`，shape仍[5]、float32。标量[]广播到向量，不需要复制成5项；本格没有舍入和int8存储转换。'),
10: ('舍入数值但保留浮点容器', '【必须会】`torch.round(scaled)`按最近整数舍入，中点采用ties-to-even（例如0.5→0，1.5→2）；返回与输入相同shape/dtype的Tensor。`int8val`只是变量名，仍为float32[5]。真正每元素1字节需要`.to(torch.int8)`，此处保留原程序不修正。'),
11: ('逆缩放重建', '【必须会】Tensor方法`.float()`返回/转换到float32，不会撤销round的损失；本例输入已经float32。乘a_max/127恢复尺度，`fp32val`是近似重建向量[5]。零尺度保护仍未加入。'),
12: ('平均绝对重建误差', '【必须会】先计算`fp32val-a`，abs消除符号，mean归约5项返回float32标量[]；这是单向量MAE，不是语言模型loss或准确率。随机值改变会导致误差变化。'),
13: ('合并单向量流程并打印', '【必须会】重新randn覆盖a和后续变量，不能与第7格同一个样本混用。流程仍是absmax→缩放→round→反量化→MAE；三个print返回None并输出原向量、重建向量和误差。int8val没有int8转换，fp32val只沿用输入float32。'),
14: ('异常值为何影响普通值', '【必须会】outliers即异常幅度值。共享一个最大尺度时，一个大幅度元素会增大所有普通元素的量化步长；不能仅删除它，因为它可能承担重要模型信息。'),
15: ('量化误差的直觉', '【必须会】原句指出异常值会令量化误差变大。精确说，这是共享尺度下普通值分辨率下降的风险，并非任意两次随机采样必定误差单调上升；第16、17格重新采样，不构成严格同输入对照。补充验证固定其余元素、只改变末项。'),
16: ('原位设置末项异常值', '【必须会】`a[-1]`是最后一个元素，负索引从尾部数；赋值`=-5`修改现有Tensor的内容，不增加维度。重新randn得到float32[5]，这里仅展示它，还没有重建。其余随机项理论上也可能超过5，不能一概断言absmax必然等于5。'),
17: ('含异常值的完整量化演示', '【必须会】再次创建a并把末项改成-5；max尺度同时作用于全部元素。`int7val`是原拼写，实际同样round到127范围，既不是7-bit dtype也不是7-bit算法。返回重建及MAE；本次用固定seed重跑，与原保存值分别记录。'),
18: ('LLM.int8算法部分', '【必须会】接下来将单向量尺度推广到矩阵的每行/每列，再把异常特征交给浮点支路；并非整个LLM都成为整数。'),
19: ('向量级量化标题', '【必须会】vector-wise意味着每个输入行/权重列各有尺度，比整个矩阵共用一套尺度更适合内积。'),
20: ('每个向量自己的normalization constant', '【必须会】输入行向量和paper权重列向量分别保存最大幅度。若用absmax，则量化放大倍数127/absmax和反量化尺度absmax/127互为倒数；不要把不同文献里同名constant当成同一方向。'),
21: ('矩阵乘法的维度契约', '【必须会】A[b,h]与B[h,o]沿共享h求和，得到C[b,o]。b为输入行/token数，h为输入特征，o为输出特征。每个C[i,j]是A第i行和B第j列的内积。PyTorch Linear保存weight[o,h]，paper方向需要转置。'),
22: ('Markdown中的完整量化伪代码', '【必须会】本格不是可直接执行的Python：`A16.absmax(1)`并非标准Tensor方法；原`B:[b,o]`维度应为[h,o]。正确尺度为a=abs(A).max(axis=1,keepdims=True)[b,1]、d=abs(B).max(axis=0,keepdims=True)[1,o]；零尺度须保护。先round并存成int8，转int32累加Q_A@Q_B，再乘a*d/127²得到浮点[b,o]。原absmax_A_vec@absmax_B_vec仅在明确二维列/行形状时才是外积；两个1D向量@会做内积或维度失败。原文字缺少round、类型转换和零尺度处理，中文注释说明纠正但不改原单元。公式对应论文式(7)，参数/返回shape见主解释第3节。'),
23: ('矩阵例子入口', '【必须会】接下来固定3行、5特征、2输出，能手算两条支路并与完整矩阵乘法对照；不涉及OPT权重。'),
24: ('论文两支路示意图', '【必须会】原图`../imgs/llm-int8.png`：输入与权重按同一个输入特征维拆分，普通支路量化/整数乘法/反量化，异常支路浮点乘法，两者同shape后相加。图中的dtype是算法设计，后面的NumPy变量名不保证实际达到图中dtype。'),
25: ('构建固定输入与paper方向权重', '【必须会】`np.asarray`函数将嵌套整数列表转为二维ndarray：X[3,5]、W[5,2]。未传dtype，平台默认整数，本次Windows NumPy1.26.4实际int32；不预设float16。X列1/3有大幅度45/12/37与-17/-63/-83。方括号是列表，逗号分隔行，最后一个尾逗号不增加额外参数。'),
26: ('完整矩阵乘法基准', '【必须会】ndarray方法`X.dot(W)`在二维时做矩阵乘法，输出[3,2]；不是逐元素`*`。本例整数基准[[38,34],[-167,124],[-174,168]]，值小没有本次整数溢出。下一步拆分仍必须相加恢复这个输出空间。'),
27: ('按特征维拆普通与异常部分', '【必须会】冒号保留该轴全部；索引列表做高级索引，返回选取的新数组。X取列，W取行：reg[0,2,4]给X_reg[3,3]/W_reg[3,2]；out[1,3]给X_out[3,2]/W_out[2,2]。两组互斥且合并覆盖5特征，避免丢失/重复求和。原程序手工选列；补充按max(abs(X),axis=0)>6自动识别。'),
28: ('普通支路标题', '【必须会】small values是相对异常特征而言；本格8bit只是设计目标，后续还要检查round与astype才能判断真实存储。'),
29: ('观察普通输入', '【必须会】`X_reg`是[3,3]整数数组，行依次为[2,-1,-1]、[0,3,2]、[-1,-1,0]。裸变量表达式只展示当前对象，不重新计算或复制变量。'),
30: ('先观察列absmax', '【建议会】`np.abs`逐元素绝对值；`np.max(...,axis=0)`沿行归约，返回每列最大幅度[2,3,2]，shape=[3]。这是探索，不是第33格输入行尺度；axis=0与axis=1不能互换。'),
31: ('观察普通权重', '【必须会】W_reg[3,2]依次[-1,0]、[0,-2]、[-1,2]；仍是paper方向输入特征行、输出列。它与X_reg的3个共享普通特征一一对应。'),
32: ('准备两套向量尺度', '【必须会】输入每行absmax，权重每列absmax；接下来用新增轴表达可广播的列/行向量。'),
33: ('归约并插入长度1轴', '【必须会】Cx=np.max(abs(X_reg),axis=1)得到[2,3,1]，`[:,np.newaxis]`变[3,1]。Cw沿axis=0得到[1,2]，`[np.newaxis,:]`变[1,2]。np.newaxis即None，只增加长度1维，不增添数值；广播后Cx*Cw为[3,2]输出尺度。'),
34: ('检查权重尺度', '【必须会】Cw=[[1,2]]、shape[1,2]，每个输出列一个尺度。仅展示Cw，不会执行量化。'),
35: ('量化阶段标题', '【必须会】标题quantize需要与下一格实际语句区分；原下一格只缩放，没有完整整数编码。'),
36: ('原程序仅缩放，无舍入或转换', '【必须会】`127/Cx`、`127/Cw`为浮点除法，分别广播到X_reg[3,3]/W_reg[3,2]。实际X_i8/W_i8为float64，不是int8；例如-1/2*127=-63.5被保留，不变成-64。变量名不能证明位宽。整数补充算例另列，不在注释版偷偷补round/astype。'),
37: ('整数乘法的设计目标', '【必须会】标题int8 matmul描述论文普通支路；第38格实际仍使用float64 dot，没有整数CUDA内核。'),
38: ('缩放后的浮点矩阵乘法', '【必须会】X_i8[3,3].dot(W_i8[3,2])得到Out_i32[3,2]，实际float64。名称i32不等于累加器int32。补充先round→int8存储，再转int32@，防止直接int8累加溢出。'),
39: ('反量化阶段标题', '【必须会】dequant即恢复原数值尺度。已丢掉的舍入细节无法恢复；原程序没有舍入，所以原算例近乎精确恢复并不奇怪。'),
40: ('输入尺度shape', '【必须会】ndarray的`.shape`是属性而非函数调用，返回Python tuple(3,1)。长度1列维用于输出[3,2]广播。'),
41: ('权重尺度shape', '【必须会】Cw.shape返回tuple(1,2)。与Cx(3,1)结合形成(3,2)而非长度2内积。'),
42: ('Kronecker积在本形状下等于尺度外积', '【建议会】`np.kron(Cx,Cw)`函数执行Kronecker积，当前列向量[3,1]和行向量[1,2]恰好给[[2,4],[3,6],[1,2]]，shape[3,2]。更直接是广播`Cx*Cw`；一般任意形状kron不是逐元素乘法。'),
43: ('恢复普通贡献', '【必须会】Out_i32逐元素乘输出尺度/127²，得到Out_f16_1[3,2]；原输入float64，结果仍float64，不是FP16。缩放/逆缩放抵消，本例结果接近X_reg@W_reg。没有量化舍入，所以不能由此得出8bit无误差。'),
44: ('异常支路的设计目标', '【必须会】标题16-bit表示论文高精度支路；第45格原整数ndarray没有FP16转换，实际仍需检查dtype。'),
45: ('异常贡献的矩阵乘法', '【必须会】X_out[3,2].dot(W_out[2,2])输出[3,2]，本环境实际int32；原Out_f16_2名字不改变dtype。异常支路保留大幅度特征的贡献，不把它们裁剪或丢掉。补充可把它们显式转FP16以验证真实半精度算术，但不能声称运行bitsandbytes。'),
46: ('两支路相加', '【必须会】两个[3,2]数组逐元素加，float64普通支路和整数异常支路组合得到float64输出。原未舍入示例接近完整X.dot(W)，是一项代数拆分演示。本次整数量化补充的误差另记，不能覆盖原历史输出。'),
47: ('Hugging Face集成入口', '【必须会】前面讲算法，后面通过预训练OPT模型展示集成入口；原notebook没有训练loss/optimizer/dataset，也没有论文C4评测。'),
48: ('三个安装提示都是注释', '【建议会】三行以#开头，Python不执行pip，也不锁版本。bitsandbytes负责相关低精度层，transformers负责模型/tokenizer，accelerate负责设备分配等。真正安装需在终端执行并匹配平台；本次仅已有CPU Transformers，不安装旧CUDA库来冒充重现。'),
49: ('导入Auto加载接口', '【必须会】AutoTokenizer和AutoModelForCausalLM是transformers类，from_pretrained是类加载入口：根据checkpoint的配置选择具体tokenizer/因果语言模型。导入本身不下载6.7B权重；返回对象、网络下载和设备迁移发生在下一几格。'),
50: ('修改进程代理环境', '【建议会】os是标准库，os.environ是当前进程环境的可变映射；两条赋值会影响后续HTTP/HTTPS请求。127.0.0.1:7890必须是本机正在运行的代理，不能假定每人都有。原语句保留，本次跳过此格，不修改真实代理环境。'),
51: ('prompt到token ID', '【必须会】MAX_NEW_TOKENS只是int变量128，ckpt是远程模型ID字符串，sample是英文prompt。AutoTokenizer.from_pretrained默认可联网下载/缓存文件；tokenizer(...)是对象可调用接口，return_tensors=pt返回BatchEncoding，`.input_ids`为int64[1,S]，通常默认CPU。只取input_ids丢掉其他字段，原格没显式保留attention_mask。原OPT未下载；教学补充独立使用本地小词表，不伪造此格执行成功。'),
52: ('加载并量化OPT底座', '【必须会】from_pretrained的ckpt指定预训练模型；device_map=auto要求加载机制根据资源分配层；load_in_8bit=True触发历史bitsandbytes集成与线性层替换。返回model并涉及权重、量化状态、设备内存；不是把全部参数/激活/cache改成int8，也不是数据并行训练。6.7B权重未下载，此格未执行。CPU小OPT和参数遍历仅验证共享生成接口，不替代量化加载。'),
53: ('生成总长度与decode', '【必须会】model.generate读取input_ids[1,S]，返回generated_ids[1,S+T]；原传max_length=128限制总长度，尽管变量名叫MAX_NEW_TOKENS。max_new_tokens才限制新增数量。decode取第一个batch完整序列，包括prompt，skip_special_tokens=True仅移除注册的特殊token。设备需与模型输入层匹配。教学本地小OPT实际执行这两行，但设置总长8并明确不量化。'),
54: ('枚举参数设备与存储类型', '【必须会】nn.Module方法named_parameters()默认recurse=True，迭代(name,Parameter)二元组；enumerate再加整数序号i。para[0]为名字，para[1]为Tensor参数，.device/.dtype是属性。f字符串把表达式插入文字，\\t是制表符，print返回None。这里观察参数，不更新参数；尺度/布局缓存未必在named_parameters里，不能据此推算全部显存或证明int8内核执行。'),
55: ('空代码格', '【可以暂时不掌握】原source为空，保留编号、元数据和历史输出字段。空格没有模型逻辑，不凭空添加训练/评价内容；注释版只加入说明性注释，AST仍为空。'),
}
BNB_NOTES = {
0: ('量化配置标题', '【必须会】config是配置入口；该辅助notebook只有3格，不是bitsandbytes底层实现。'),
1: ('导入BitsAndBytesConfig', '【必须会】transformers中的配置类保存4/8-bit加载选项；导入不会构建配置、替换模型或下载权重。构造器默认load_in_8bit=False、load_in_4bit=False，llm_int8_threshold=6.0。主notebook沿用旧布尔参数，参考新写法见主解释。本次真实导入并构造8bit配置，检查相互排斥的ValueError，不等于运行量化内核。'),
2: ('4-bit与8-bit互斥', '【必须会】原Markdown提示load_in_8bit/load_in_4bit二选一为true。更准确是两者不能同时True，可以都False（不开这两种量化）；配置校验不是模型能力证明。实际BitsAndBytesConfig(load_in_8bit=True,load_in_4bit=True)应抛ValueError；开启4bit的配置还可能查询bitsandbytes安装版本。本次不安装4bit后端。'),
}

# 键为原单元，值逐条对应原有效代码行之前的新注释。保持原注释和空白不变。
COMMENTS = {
0: ['IPython图片展示类；这里只导入，不读图。', '数组模块别名np，后面用二维ndarray演示。', '张量库，先做CPU单向量运算。', '神经网络命名空间；本数值段未定义nn.Module。'],
2: ['从当前工作目录读取原图；width控制展示像素，不影响量化算法。'],
5: ['读取原量化示意图，返回可供Jupyter展示的Image对象。'],
7: ['标准正态5项，默认CPU float32[5]；每次运行重新采样。', '裸表达式在Jupyter显示，不改变数据。'],
8: ['逐元素绝对值再全局归约，标量absmax[]供共享缩放。', '展示尺度；max不是最大带符号值。'],
9: ['标量广播到5项，缩放到[-127,127]；仍float32。', '展示缩放向量；尚未舍入。'],
10: ['round舍入数值、保留float32容器；变量名int8val不等于int8存储。', '查看舍入后的浮点Tensor[5]。'],
11: ['逆缩放到原幅度，float()不能恢复舍入损失，输出float32[5]。'],
12: ['5项绝对误差的均值，输出标量[]，不是模型准确率。'],
13: ['重新采样，覆盖旧a。', '整个新向量共用标量absmax。', '浮点缩放，shape[5]不变。', '最近整数舍入，dtype仍float32。', '逆缩放重建；没有真正int8编码。', '打印新原向量。', '打印近似重建向量。', '计算并打印平均绝对误差。'],
16: ['新随机float32[5]。', '负索引-1是最后一项，赋值原位修改为-5。', '展示含大幅度末项的向量。'],
17: ['再采样，和第16格不是同一随机向量。', '原位改末项；共享尺度可能增大量化步长。', 'absmax由所有元素决定，随机其他项也可能更大。', '仍是浮点缩放，不改变dtype。', 'int7val是原拼写，round并不代表7bit。', '重建float32向量。', '原向量输出。', '重建输出。', '单次MAE；不能与不同随机输入当成严格误差对照。'],
24: ['原图展示两条支路；读图不执行CUDA内核。'],
25: ['整数嵌套列表→X[3,5]；未指定float16，实际dtype由NumPy平台决定。', '第二输入行，共5特征。', '第三输入行；尾逗号不新增矩阵轴。', 'paper方向权重W[5,2]，共享特征维5。', '特征1的两项权重。', '特征2的两项权重。', '特征3的两项权重。', '特征4的两项权重，结束矩阵。'],
26: ['二维dot是矩阵乘法，整数基准[3,2]。'],
27: ['X取普通特征列0/2/4→[3,3]；高级索引产生选取数组。', 'W取相同特征行0/2/4→[3,2]。', 'X取异常特征列1/3→[3,2]。', 'W取相同特征行1/3→[2,2]，两支路必须覆盖且互斥。'],
29: ['普通输入[3,3]，仅查看。'],
30: ['axis=0归约行，输出每列absmax[3]；不是后面输入每行尺度。'],
31: ['普通权重[3,2]，与3个普通特征对齐。'],
33: ['axis=1→每行absmax[3]，加新轴变Cx[3,1]以便广播。', 'axis=0→每列absmax[2]，加新轴变Cw[1,2]。'],
34: ['权重两列尺度[[1,2]]。'],
36: ['127/Cx浮点广播；没有round/astype，X_i8实际float64[3,3]。', '127/Cw浮点广播；W_i8实际float64[3,2]，名称不改变dtype。'],
38: ['原dot输入float64，Out_i32也是float64[3,2]，没有int32累加器。'],
40: ['shape是ndarray属性，返回(3,1)而非Tensor。'],
41: ['返回权重尺度形状(1,2)。'],
42: ['当前列/行向量kron等于输出尺度外积，shape[3,2]；一般kron不是*。'],
43: ['缩放抵消，恢复普通贡献[3,2]；未显式转FP16，实际float64。', '展示恢复结果；近乎精确因为原程序没有量化舍入。'],
45: ['异常特征矩阵乘法[3,2]；原整数输入仍整数输出，不是真FP16。'],
46: ['同shape两支路相加，不重叠特征；float64+整数→float64。'],
48: [],
49: ['Auto类根据checkpoint配置选择具体实现，导入不下载OPT。'],
50: ['标准库环境接口。', '修改当前进程HTTP代理；读者必须确有此服务，本次跳过。', '同样修改HTTPS代理，非算法参数。'],
51: ['变量名说new，但第53格实际传max_length，限制总长度。', '远程OPT6.7B模型标识；加载可能下载大权重/tokenizer。', '一条英文输入字符串，不是训练dataset。', '类加载入口下载/缓存tokenizer文件，本次原格未执行。', '返回BatchEncoding，取int64[1,S]词表索引，默认CPU，不能改成int8。', '仅查看输入ID。'],
52: ['原6.7B量化加载会下载权重并分配内存，本次未执行。', 'auto让加载器分配模块设备；8bit替换可量化线性层，非所有张量整数化。'],
53: ['生成结果包括prompt，max_length限制总token数，不是最多新增128。', '取第一个样本解码，跳过特殊token但不会自动移除prompt。'],
54: ['遍历(名字,Parameter)，enumerate另给i；只观察，不更新参数。', 'f字符串插值设备/dtype；尺度缓存不一定在参数迭代器内。'],
55: [],
}

def note(i, c, notes):
    title, detail = notes[i]
    return '## 单元 {}（{}）：{}\n\n{}\n'.format(i, c['cell_type'], title, detail)

def annotate(i, source, comments, notes):
    additions = iter(comments[i])
    result = ['# ' + notes[i][0] + '：详见本格中文说明，原执行语句保持不变。\n']
    for line in source.splitlines(True):
        if line.strip() and not line.lstrip().startswith('#'):
            comment = next(additions)
            indent = line[:len(line) - len(line.lstrip())]
            result.append(indent + '# ' + comment + '\n')
        result.append(line)
    assert next(additions, None) is None
    return ''.join(result)

def build(lesson, filename, notes, comments, label, prefix):
    directory = lesson / '代码/tutorials'
    original = json.loads((directory / filename).read_text(encoding='utf-8'))
    assert set(notes) == set(range(len(original['cells'])))
    intro = '# 第08讲 ' + label + '\n\n全部原单元逐格说明；原执行语句、元数据及历史输出保留。原保存输出不是本次结果。原OPT6.7B/CUDA量化未运行；CPU数值和小模型生成补充见复现/README.md。\n'
    for changed, suffix in [(False, '原代码_带说明.ipynb'), (True, '代码注释.ipynb')]:
        nb = copy.deepcopy(original)
        nb['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'id': prefix + '-intro', 'source': intro.splitlines(True)}]
        for i, c in enumerate(original['cells']):
            nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'id': prefix + '-note-' + str(i), 'source': note(i, c, notes).splitlines(True)})
            keep = copy.deepcopy(c)
            if changed and c['cell_type'] == 'code':
                keep['source'] = annotate(i, ''.join(c['source']), comments, notes).splitlines(True)
            nb['cells'].append(keep)
        # notebook禁用Git换行转换；派生文件显式写UTF8/LF，原下载文件保持原字节。
        (directory / (('bitsandbytes_' if prefix == 'bnb08' else '') + suffix)).write_bytes((json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode('utf-8'))
    py = ['# -*- coding: utf-8 -*-', '# 第08讲 ' + label + ' Python阅读版；原执行语句未修改。', '# 按# %%分格阅读；最后OPT6.7B段可能下载权重，原代理格会修改环境。', '# 实际CPU验证范围和跳过单元见复现/README.md。', '']
    doc = '\n# ' + label + '：全部单元\n\n'
    for i, c in enumerate(original['cells']):
        detail = note(i, c, notes)
        source = ''.join(c['source'])
        doc += detail + '\n'
        py += ['# %% [markdown]'] + ['# ' + s for s in detail.splitlines()] + ['']
        if c['cell_type'] == 'code':
            doc += '原代码：\n\n```python\n' + source + '\n```\n\n'
            py += ['# %% 原单元 {}'.format(i)] + annotate(i, source, comments, notes).splitlines() + ['']
        else:
            doc += '原Markdown（保留原符号/伪代码，可与上面中文说明对照）：\n\n' + source + '\n\n'
            py += ['# %% [markdown] 原单元 {}'.format(i)] + ['# ' + s for s in source.splitlines()] + ['']
    reading = directory / (filename.removesuffix('.ipynb') + '_代码注释.py')
    reading.write_text('\n'.join(s.rstrip() for s in py).rstrip() + '\n', encoding='utf-8')
    return doc

def main():
    lesson = Path(__file__).resolve().parents[2]
    title, detail = NOTES[2]
    NOTES[2] = (title, detail + ' 图中蓝色sign为符号位；FP32为1+8指数+23显式尾数位，FP16为1+5+10，BF16为1+8+7。TF32图表示计算精度/范围，不是可以直接创建的19bit PyTorch存储dtype；详见第05讲。')
    title, detail = NOTES[5]
    NOTES[5] = (title, detail + ' 从左到右：8项向量→absmax5.4→放大倍数127/5.4≈23.5185→整数编码→除相同倍数重建。图把23.5185显示成23.5、重建保留一位小数，不能据下排相同文字断言无损；例如编码28实际重建28*5.4/127≈1.19055，不等于1.2。')
    doc = '# 第08讲代码单元解释\n\n主notebook全部56格（含空格55），辅助bitsandbytes全部3格。编号分别从0开始；Markdown22伪代码也完整说明。\n\n[主解释](解释.md) · [带注释notebook](代码/tutorials/代码注释.ipynb) · [实际验证](复现/README.md)\n\n'
    doc += build(lesson, 'LLM.int8.ipynb', NOTES, COMMENTS, 'LLM.int8', 'llmint808')
    doc += build(lesson, 'bitsandbytes.ipynb', BNB_NOTES, {1: ['导入配置类，本格未构造配置或调用量化加载。']}, 'bitsandbytes配置', 'bnb08')
    (lesson / '代码单元解释.md').write_text('\n'.join(s.rstrip() for s in doc.splitlines()).rstrip() + '\n', encoding='utf-8')
    print('All56+3 notes; 2 originals -> 4 annotated/explained notebooks and 2 Python reading files.')

if __name__ == '__main__':
    main()
