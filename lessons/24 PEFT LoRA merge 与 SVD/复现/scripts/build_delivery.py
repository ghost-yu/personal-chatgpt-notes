# -*- coding: utf-8 -*-
"""Publish three unchanged originals and comment-only derivatives."""
import argparse
import ast
import base64
import copy
import csv
import hashlib
import io
import json
import subprocess
import tokenize
from pathlib import Path
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "7fbc29824171fd07dc2c30fe922e61ebc018ad1d"
FILES = {
    "lora_inject": ("inject", """
参考低层注入接口。原链接未固定版本；本次固定本机PEFT0.12.0发行文件及SHA，不声称讲师用同一版本。
导入inspect检查继承/源码，torch和nn构造张量/模块，inject_adapter_in_model直接改模型内部结构；导入本身不下载模型。
DummyModel继承nn.Module。__init__注册embedding、linear、lm_head；super调用父初始化；forward定义调用流程。输入int64[B,L]→Embedding float32[B,L,1000]→Linear[B,L,1000]→lm_head[B,L,100]。没有激活、softmax或loss，不是训练完的语言模型。
LoraConfig是配置对象。r64、alpha16令s=alpha/r=.25；dropout .1只在训练分支随机丢输入；bias none不额外训练偏置；仅名字linear匹配，不注入embedding和lm_head。A[64,1000]、B[1000,64]，新增128000参数。
展示配置，不发生第二次注入。默认普通LoRA且非RSLoRA，不能套alpha/sqrt(r)。
先展示未注入的模型，作为后续对象身份/输出对照。
实例化三个随机层。nn.Module的注册让parameters/state_dict能递归查找子模块；此时没有预训练权重。
type返回注入前的torch.nn.Linear类，不执行前向。
LongTensor是int64词ID；[1,8]→[1,8,100]。0–7均在100词表内，最后100维是未归一化logits；本格只看shape。
进入注入分支。原embedding和lm_head继续使用，linear变成LoRA包装层。
inject_adapter_in_model(config,model)在原模型上替换目标层，返回同一个DummyModel，不产生PeftModel外层。旧linear被保存在base_layer中，基础参数冻结，只有A/B可训练。新分支初始B0，所以初始输出不变。
对比类型：embedding仍是Embedding，linear为PEFT的Linear。类名字相同不代表实现相同，应看完整模块路径。
isinstance检查继承关系；LoRA包装仍然是nn.Module，可以参与注册、调用和autograd。
inspect.getmro返回方法解析顺序元组，解释多继承查找次序，不是模型执行顺序或层数。
forward??是IPython源码查询。核心为base_layer(x)+B(A(dropout(x)))*s；普通Python不能直接解析双问号，阅读版采用真实IPython转换。本次捕获pinfo2源码文本，不启动外部分页器。
进入保存。只取adapter状态不等于完整模型保存；基础模型、配置和词表仍需另保留。
get_peft_model_state_dict是函数，按模型的peft_config筛选适配器权重。
返回字典含linear.lora_A.weight[64,1000]和linear.lora_B.weight[1000,64]，默认adapter名在导出键中去掉。没有embedding/base_linear/lm_head完整权重，低层API也不自动提供PeftModel.save_pretrained。
"""),
    "lora_svd_basics": ("svd", """
原文链接Torch SVD及漂移的PEFT main行号。本次以torch2.5.1和PEFT0.12.0本地源码作可执行证据。
Image只负责展示，不是矩阵分解库。
远程LoRA图：原URL和HTML引用保留；未下载图片，未验证远端图像内容。新增解释提供可直接读的流程图。
导入torch并固定seed0，影响后续随机矩阵。本次原版/注释版各自从该格重新设seed。
d,n分别是输出/输入维度100；low_rank20是构造与截断使用的目标秩，不是batch或token数。
两个随机float32矩阵[100,20]@[20,100]得到W[100,100]，数学秩至多20。随机实数下通常20，数值秩还受dtype和容差影响。
matrix_rank使用奇异值容差判断数值秩。本次真实得到20，不把历史结果当新结果。
进入SVD，不做模型训练。
第二个远程图片展示，width400只影响HTML布局。
低秩矩阵重建：保留20个主要奇异值应在浮点误差内恢复W。
torch.linalg.svd(W,full_matrices=True)返回U[100,100]、S[100]、Vh[100,100]，Vh已是V的共轭转置。float32 CPU实际执行。
只展示三个shape，方阵容易掩盖矩形full/reduced SVD尺寸差异，解释中另外推导。
diag把奇异值向量变对角矩阵；U@diag(S)@Vh重建W；torch.dist默认Frobenius距离，数值误差不是严格0。
切U前20列/S前20项/Vh前20行，再把S乘进U。变量U此后代表U20Σ20，不再是正交矩阵；重建用U@Vh，不能再乘一次S。
比较截断重建误差。对构造的秩20矩阵应很小；不意味着任意100×100矩阵都能由秩20无损表示。
换成普通随机满秩矩阵，展示压缩误差。
randn_like保持W的shape/device/dtype，重新生成独立随机W2，数学上几乎必然满秩。
数值秩本次100；不是把low_rank变量改成100。
对满秩W2做完整SVD，重新覆盖U/S/Vh三个变量。
完整100项重建W2应只有浮点误差，不能用此误差评价语言模型。
只保留20项后有明显误差。最优秩20近似仍会丢弃剩余80项；其Frobenius误差平方等于被丢弃奇异值平方和。
"""),
    "lora_merge": ("main", """
本讲主入口及官方合并文档。合并博客讲不同方法，不能把本格链接理解成所有方法已经运行或验证。
设置GPU可见性及本机7890代理，会改变当前进程环境。本次跳过此格，CPU离线验证不依赖这些设置。
整格为安装命令注释，不执行pip。本次明确锁定PEFT0.12.0，不运行upgrade改变环境。
PeftConfig读adapter配置；PeftModel管理多个adapter；AutoModelForCausalLM和AutoTokenizer负责基础模型和分词；load_dataset/random在本notebook未实际用于训练。
先掌握形状与缩放，再读多适配器合并。
重要Markdown源码也详注。Linear.weight按[out,in]存储；输入行向量右乘weight.T。普通LoRA s=alpha/r，RSLoRA为alpha/sqrt(r)。公式必须明确ΔW是否已经含s，避免重复乘；本次PEFT0.12.0的SVD支路确实重复缩放，不能直接当理想公式。
远程Medium图，只保留URL和历史HTML引用。width400不是矩阵维数，本次没有验证远程图片内容。
课程提到Linear/Embedding/Conv2d。本次实际验证前两种，未执行卷积LoRA；并非声称所有PEFT版本只支持这三种。
Markdown中的regex实际对model.modules这个绑定方法的字符串做匹配，没有调用modules()；repr可含整个模型而匹配出adapter名、base_layer等。结果不适合自动挑选目标层，补充用named_modules取得真实模块路径。
开始加载三个已存在的adapter；加载不等于训练或默认同时激活三个。
原adapter为smangrul/tinyllama_lora_norobots，device cuda仅赋字符串。PeftConfig.from_pretrained下载配置，不加载1.1B基础权重。本次把地址和设备显式改为本地随机教学fixture/CPU。
显示base_model_name_or_path、r8、alpha16等配置；历史基础模型TinyLlama1.1B不是本次小模型。
原模型4bit/device_map auto加载需要相应量化后端及权重，本次不执行。CPU变体从本地随机Llama小checkpoint加载FP32，并加载自建真实SentencePiece词表，不冒充原模型。
tokenizer长度包含新增token；原历史32005，本次教学词表另记录实际数量。
基础模型config.vocab_size与分词器可能不同；原历史32000，差5需调整embedding和输出头。
resize_token_embeddings扩展词表，保留已有行并初始化新增行；须在加载含词表相关权重的adapter前完成。相同长度只是必要条件，ID语义和基础权重也须匹配。
两个词表键集合做差，历史新增pad、im_end及三种角色起始token。只比较字符串不保证ID排列一致；本次用相同五个新增字符串构造明确教学分词器。
PeftModel.from_pretrained把本地或远程adapter挂到已有基础模型，名字norobots用于后续选择。返回PeftModel包装，区别于低层inject返回原类。
load_adapter分别挂载sql/adcopy；原SQL只作用七个投影层，另外两个还含embed_tokens/lm_head。共同基础模型、词表、目标层形状必须兼容，加载成功不自动证明语义兼容。
展示配置字典，各键对应命名adapter，并不是三个独立基础模型。
keys视图列出已注册adapter名字，不说明当前激活哪个。
遍历dict的key/value，打印r、alpha、alpha/r、fan_in_fan_out。三者原均r8/alpha16/s2，布尔False不妨碍Embedding在get_delta_weight内部转置。
显示整个PEFT结构，可看Linear与Embedding采用不同A/B容器、共享同一base_layer。
创建第四个名为merge的adapter，不在此把它折叠进基础权重。
方法分cat、SVD矩阵分解、直接组合A/B等支路。TIES/DARE只作概念和入口说明，本讲实际原调用是svd，没有做论文质量比较。
adapters和weights按位置配对，权重2/.3/.7不自动归一化；density .2在纯svd支路不使用。已有merge先删再建。该API返回None并修改模型，默认rank取最大8、alpha也设8→新s1；没有自动set_adapter('merge')。
逐adapter打印target_modules，SQL没有embedding/lm_head，因此在这些层只合并另外两个。不能把缺失层当作该adapter的随机权重。
再次展示结构，新增merge的A/B可见，但可见不代表激活。
仅注释的查看表达式，不执行；也计入代码覆盖。
取得embedding层scaling字典，原和本次均norobots2/adcopy2/merge1，没有sql。
原历史为bitsandbytes Linear4bit；本次CPU是普通PEFT Linear，不把类型差异隐去或伪装量化。
手写SVD演示聚焦Embedding层，不是手动合并所有层。
读取Embedding层，base权重[V,H]，LoRA因子A[r,V]/B[H,r]。
四个变量是已有Parameter的引用，不是clone；后续只读乘法未改这些原因子。norobots/adcopy两组，SQL不在该层。
原A[8,32005]、B[2048,8]；本次小模型H16、词表约百，形状和真实测量在报告中给出。
weights全局被覆盖成两项；valid_weights已经含s=2。之后delta_lora_i又含s，累计重复缩放。三来源原权重在此前API调用时已使用，不被此格追溯修改。
transpose为本notebook普通函数。False直接返回同对象；True返回.T视图，若是Parameter则再包成新Parameter，可改变autograd叶对象。本次乘积是Tensor，走最后分支。Embedding转置与配置fan_in_fan_out=False不是矛盾。
BA[H,V]转置成[V,H]并乘scaling2，得到已缩放ΔW_i。原main源码行号漂移；本次完整固定源码读本可直接对照。
展示第一个已缩放Embedding增量，不是新基础模型参数。
展示第二个已缩放增量；两者均与Embedding.weight同shape。
确认[V,H]；原[32005,2048]，CPU教学采用实际小shape。
valid_weights又乘过s，使结果为sum(w_i*s_i²*B_i*A_i).T。特别注意PEFT0.12.0的get_delta_weight与SVD helper也有这条重复缩放路径；因此本格可匹配旧库但偏离理想sum(w_i*ΔW_i)。
再转置到[H,V]做SVD，保留8项并把Σ吸收进U。原full_matrices=True的Vh[32005,32005]单FP32张量约3.82GiB，未在原大尺寸运行；本次小fixture保留True真实执行。
Vh[8,V]就是新A形状，UΣ[H,8]就是新B形状，顺序不能反过来。
单独看Vh不能证明整块ΔW一致，奇异向量有符号/简并旋转不唯一性。
读取库创建的merge因子引用；没有把手写U/Vh赋回模型。
核对库新A/B形状；应比较乘积及实际输出，而不是要求单因子元素位级相同。
只显示A，未对B或完整ΔW做断言。新增验证比较重建矩阵、尾部奇异值误差以及缩放约定。
进入生成；原文没有切换merge，需先检查active_adapters。
指令任务，只是一个prompt示例，不是标准评测集。
apply_chat_template用角色/content拼接模板，tokenize=False返回str，add_generation_prompt追加assistant起始。本次使用明确自建ChatML教学模板，不宣称来自原Hub资产。
tokenizer生成int64[1,T]及mask；原.to(cuda)和256续写本次显式改CPU/4。采样top_p .95/temp .2/重复惩罚1.2；decode包括prompt。仍是norobots激活，随机输出不作为任务能力证据。
广告任务标题，不会自动激活adcopy或merge。
system/user消息描述商品；模板返回一条字符串。原文字原样保留，不把商品内容当成执行指令。
同样显式CPU/4续写变体，保留原采样参数；实际生成不是广告效果或合并优劣评测。
SQL任务标题；任务切换不等于adapter切换。
三引号多行字符串保持原内容，列名/问题作为模型输入，print只是展示；不是执行数据库SQL。
原64续写改4、cuda改cpu。tokenizer('</s>').input_ids[-1]取EOS，必须检查tokenizer语义。默认do_sample=False为greedy，未显式set_adapter，所以不能据这条历史回答证明merge有效。
空代码单元保留原顺序、输出、元数据并解释，执行为空操作。
"""),
}
FILES = {k: (v[0], v[1].strip().splitlines()) for k, v in FILES.items()}

RULES = [
    ("def inject_adapter_in_model", "低层入口：配置、原模型、adapter名；在原模型中注入，返回原模型。"),
    ("def _check_add_weighted_adapter", "检查目标集合/来源rank，决定最终算法、结果rank和目标层集合。"),
    ("def add_weighted_adapter", "LoraModel方法：创建新适配器参数，返回None，不自动切换激活项。"),
    ("def _svd_generalized", "逐目标层合成矩阵再分解；本版本valid_weights与get_delta_weight均含s。"),
    ("def _generalized_task", "直接分别组合A/B；linear会引入不同来源的交叉乘积。"),
    ("def get_delta_weight", "层方法返回与基础weight同shape的增量，已经包含scaling。"),
    ("def merge_and_unload", "把选定或当前活跃增量加到基础W并移除adapter包装，区别于创建merge。"),
    ("if adapter_name in list", "已有同名adapter则直接返回，避免静默以为新权重已重算。"),
    ("combination_type, new_rank", "校验来源存在和rank要求；单来源会转linear。"),
    ("self.peft_config[adapter_name] = replace", "复制配置，结果rank和alpha一起设为new_rank，使普通新缩放为1。"),
    ("self.inject_adapter", "为所有目标层分配新A/B；没有在这里调用set_adapter。"),
    ("_freeze_adapter", "冻结新建adapter，不等于把它激活。"),
    ("if isinstance(target, LoraLayer)", "只在真实LoRA层操作，不遍历任意Python对象。"),
    ("target_lora_A.data =", "写入新适配器A；源adapter不在此行被改写。"),
    ("target_lora_B.data =", "写入新适配器B；.data是这里官方源码的绕过autograd赋值方式。"),
    ("valid_weights.append(weight *", "此固定版本先乘源scaling；下方get_delta_weight已含s，导致再次缩放。"),
    ("valid_weights.append(math.sqrt", "linear对A/B各用sqrt(w*s)，需w*s非负，并产生交叉项。"),
    ("delta_weight = [target.get_delta_weight", "每个返回增量已经乘过源s，Embedding还已经转到[V,H]。"),
    ("delta_weight = task_arithmetic", "逐元素加权求和；纯svd不使用density。"),
    ("delta_weight = delta_weight.T", "Embedding转回[H,V]或转置存储的Linear对齐B@A方向。"),
    ("output_tensor = transpose", "get_delta_weight在这里乘源scaling；这是双重缩放审计的关键一半。"),
    ("result = result + lora_B", "先A降维、B升维、乘s，再加基础输出；只对活跃且存在的adapter执行。"),
    ("result = self.base_layer", "计算冻结基础支路；已folded时不再重复加LoRA分支。"),
    ("for active_adapter in", "只遍历当前激活项，注册多个adapter不表示全部相加。"),
    ("loras_A.append", "cat把w*s吸收到A，避免产生不同来源的交叉项。"),
    ("loras_B.append", "B按列拼接，配合A按行拼接精确表示加权增量。"),
    ("return Vh, U", "返回A=Vh前r行和B=U前r列乘Σ，顺序与函数赋值一致。"),
    ("class DummyModel", "定义nn.Module子类；类是结构模板，实例才持有实际参数。"),
    ("def __init__", "构造方法，实例创建时注册子层。"),
    ("super().__init__", "先初始化父nn.Module内部注册结构。"),
    ("self.embedding =", "100词映射到1000维；权重[100,1000]。"),
    ("self.linear =", "目标层权重[1000,1000]、偏置[1000]。"),
    ("self.lm_head =", "1000维投影为100个未归一化词logit。"),
    ("def forward", "实例被model(input_ids)调用时执行此方法。"),
    ("x = self.embedding", "int64[B,L]查表得到float32[B,L,1000]。"),
    ("x = self.linear", "在末维做线性变换，保留batch/序列轴。"),
    ("x = self.lm_head", "末维1000变100，得到[B,L,100]。"),
    ("return x", "返回最后张量，不额外softmax或计算loss。"),
    ("from ", "从指定包导入本格需要的真实API；导入不等于加载远程权重。"),
    ("import ", "导入模块命名空间供后续使用。"),
    ("lora_alpha=", "普通LoRA缩放分子alpha，与r相除得到支路系数。"),
    ("lora_dropout=", "训练时随机丢输入比例；eval时禁用，不能与density混淆。"),
    ("r=", "LoRA中间秩，决定A/B中间轴和新增参数量。"),
    ("bias=", "none表示不额外训练基础层bias。"),
    ("target_modules=", "指定注入目标名字，不能只按打印文本猜目标路径。"),
    ("lora_config =", "创建配置，本行尚未替换模型内部层。"),
    ("model = DummyModel", "实例化三个随机基础层，不加载预训练权重。"),
    ("model = inject", "在原对象内替换匹配层，返回同一模型实例。"),
    ("dummy_inputs =", "构造有效范围内的int64词ID张量[1,8]。"),
    ("model(dummy_inputs)", "触发Module.__call__→forward，只查看结果shape。"),
    ("type(", "读取实际Python类，注意torch和PEFT的Linear是不同类。"),
    ("isinstance(", "测试对象是否属于该类或其子类。"),
    ("inspect.getmro", "读取类的方法解析顺序，不是神经网络计算顺序。"),
    ("model.linear.forward??", "IPython pinfo2检查源码；普通Python需转换。"),
    ("get_peft_model_state_dict", "筛选adapter状态，不包含完整基础模型参数。"),
    ("os.environ", "修改当前进程环境变量；CPU离线验证跳过代理/GPU设置。"),
    ("Image(", "返回远程图片HTML引用，width仅为显示宽度。"),
    ("peft_model_id =", "adapter仓库地址，区别于基础模型地址。"),
    ("device =", "记录设备字符串；本行没有移动模型。"),
    ("config =", "读取adapter元数据，取得基础模型路径等配置。"),
    ("model = AutoModel", "加载基础因果语言模型；原4bit路径未在CPU执行。"),
    ("tokenizer =", "加载分词器和特殊token规则，不加载神经权重。"),
    ("len(tokenizer)", "分词器含新增token的总词数，应匹配模型可用行数。"),
    ("model.config.vocab_size", "读取当前模型词表容量，不保证等于分词器长度。"),
    ("model.resize", "扩展输入embedding与输出头，使新增token有对应参数。"),
    ("AutoTokenizer.from", "分别读两个词表的键并取集合差；不验证ID顺序。"),
    ("model = PeftModel", "包装基础模型并加载第一个命名adapter。"),
    ("_ = model.load_adapter", "注册另一组adapter参数；下划线接收未使用返回值。"),
    ("model.peft_config", "读取各adapter配置或keys视图，不表示激活状态。"),
    ("for k, v in", "items同时遍历adapter名和配置对象。"),
    ("print(k, v.r", "显示rank/alpha/普通缩放/存储转置配置。"),
    ("adapters =", "列表顺序必须与weights位置对应。"),
    ("weights =", "人为合并系数，不自动归一化；后续同名变量会被覆盖。"),
    ("adapter_name =", "将生成的新适配器命名为merge。"),
    ("density =", "供剪枝等支路使用；本次纯svd不读取它。"),
    ("combination_type =", "选择具体合并算法，本次是svd。"),
    ("if adapter_name in", "重复执行前先检查是否已有同名结果。"),
    ("model.delete_adapter", "删除旧merge配置/因子；可能影响active状态，不能当纯读取。"),
    ("model.add_weighted_adapter", "生成新A/B，不自动激活，不折叠到基础W。"),
    ("for adapter in", "按来源列表检查每个adapter的目标集合。"),
    ("print(adapter", "SQL目标缺少embedding/lm_head，逐层应过滤有效来源。"),
    ("model.model.get_submodule", "按点分层路径读取子层和scaling或类。"),
    ("lora_a_", "引用Embedding的A参数[r,V]；不是复制。"),
    ("lora_b_", "引用Embedding的B参数[H,r]；不是复制。"),
    ("scaling =", "本例两组源adapter缩放均为16/8=2。"),
    ("valid_weights =", "这里已经把合并系数乘s；与后面已缩放delta叠加会重复。"),
    ("def transpose", "普通函数显式控制二维转置；参数是Tensor或Parameter。"),
    ("if not fan_in_fan_out", "False时保留原对象和shape。"),
    ("return weight", "返回原对象、转置视图或新Parameter，详见本格说明。"),
    ("if isinstance(weight", "Parameter是可注册参数；重新包装会改变autograd叶对象。"),
    ("delta_lora_1 =", "B@A后转置并乘源s，得到已缩放Embedding增量[V,H]。"),
    ("delta_lora_2 =", "同上，为第二个源adapter的已缩放增量。"),
    ("delta_lora = valid", "再乘valid_weights里的s，得到双重缩放的矩阵。"),
    ("delta_lora = delta_lora.T", "转回[H,V]以分解成B[H,r]和A[r,V]。"),
    ("U, S, Vh =", "返回奇异向量/奇异值；full=True在大矩形词表上很占内存。"),
    ("U = U[:,", "保留左奇异向量前r列；切片通常是view。"),
    ("S = S[:", "保留前r个最大奇异值。"),
    ("U = U @", "把奇异值吸收进左因子；此后U不再正交。"),
    ("Vh = Vh[:", "保留右奇异向量前r行，对应新A。"),
    ("messages =", "构造角色/content字典列表，尚未分词。"),
    ("text = tokenizer.apply", "套模板返回str，追加assistant生成提示。"),
    ("text =", "保存多行SQL提示字符串，不执行数据库命令。"),
    ("inputs = tokenizer", "字符串转int64词ID和attention_mask，通常[1,T]。"),
    ("inputs = {", "字典推导式逐张量移到指定设备，原来是cuda。"),
    ("outputs = model.generate", "自回归生成；总输出长度含prompt，检查当前激活adapter。"),
    ("print(tokenizer.decode", "把整条输出ID解码，包含prompt且可能含特殊token。"),
    ("torch.random.manual_seed", "固定后续torch随机数序列，不固定所有硬件算法。"),
    ("d, n =", "定义矩阵输出/输入维度，不是batch或序列长度。"),
    ("low_rank =", "构造低秩矩阵及截断SVD所用目标秩。"),
    ("W =", "两个窄矩阵相乘，所得矩阵秩至多low_rank。"),
    ("W2 =", "构造相同shape/dtype的独立随机满秩对照矩阵。"),
    ("torch.linalg.matrix_rank", "按奇异值容差计算数值秩，不是精确符号代数秩。"),
    ("torch.dist", "默认计算Frobenius距离；浮点重建不要求严格零。"),
]

def annotate(src, note=""):
    # Never insert comments inside multiline strings (SQL or official docstrings).
    protected = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            protected.update(range(tok.start[0] + 1, tok.end[0] + 1))
    out = ["# " + note + "\n"] if note else []
    for n, line in enumerate(src.splitlines(keepends=True), 1):
        if n not in protected and line.strip() and not line.lstrip().startswith("#"):
            for prefix, text in RULES:
                if line.strip().startswith(prefix):
                    if prefix in ("lora_a_", "lora_b_") and "=" not in line:
                        continue
                    out.append(line[:len(line)-len(line.lstrip())] + "# " + text + "\n")
                    break
        out.append(line)
    return "".join(out)

def source_text(src, node):
    first = min([node.lineno] + [n.lineno for n in getattr(node, "decorator_list", [])])
    return "".join(src.splitlines(keepends=True)[first-1:node.end_lineno])

def format_reading(src):
    # Clean formatting in the text reading only; never alter multiline string contents.
    protected = set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.STRING and t.end[0] > t.start[0]:
            protected.update(range(t.start[0], t.end[0]+1))
    result = "".join(line if n in protected else line.rstrip()+"\n"
                     for n,line in enumerate(src.splitlines(keepends=True),1))
    return result.rstrip()+"\n"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--peft-root", required=True)
    p.add_argument("--sources", required=True)
    args = p.parse_args()
    target = ROOT / "代码/tutorials/peft"
    target.mkdir(parents=True, exist_ok=True)
    sources = Path(args.sources)
    sources.mkdir(parents=True, exist_ok=True)
    installed = Path(args.peft_root)
    record = installed.parent / "peft-0.12.0.dist-info/RECORD"
    records = {row[0]: row[1] for row in csv.reader(io.StringIO(record.read_text(encoding="utf-8")))}
    shas = {}
    rels = ["mapping.py", "tuners/lora/model.py", "tuners/lora/layer.py", "utils/save_and_load.py"]
    for rel in rels:
        raw = (installed / rel).read_bytes()
        encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        assert records["peft/" + rel] == "sha256=" + encoded, rel
        shas[rel] = hashlib.sha256(raw).hexdigest()
        out = sources / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
    manifest = {"repository": "https://github.com/chunhuizhang/personal_chatgpt", "commit": COMMIT,
                "files": [], "peft_version": "0.12.0", "peft_installed_sources_sha256": shas,
                "peft_integrity": "Matches installed official distribution RECORD; tag download failed with TLS EOF; lecturer exact version unknown",
                "subtitle": {"file": "435af888-0796-42aa-bbe5-64d38f6d91c3.txt", "sha256": hashlib.sha256(Path("E:/435af888-0796-42aa-bbe5-64d38f6d91c3.txt").read_bytes()).hexdigest()}}
    md = ["# 第24讲全部单元说明\n\n三份notebook共98格；原编号各自从0开始。普通代码、Markdown源码片段、仅注释与空格均覆盖。\n"]
    transformer = TransformerManager()
    for stem, (prefix, notes) in FILES.items():
        path = "tutorials/peft/" + stem + ".ipynb"
        raw = subprocess.check_output(["git", "show", COMMIT + ":" + path], cwd=args.upstream)
        (target / (stem + ".ipynb")).write_bytes(raw)
        original = json.loads(raw)
        assert len(notes) == len(original["cells"]), (stem, len(notes), len(original["cells"]))
        manifest["files"].append({"upstream_path": path, "published_path": "代码/" + path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        intro = {"cell_type": "markdown", "metadata": {}, "source": ["# 第24讲：" + stem + "\n\n原单元、历史输出及元数据保留；新增说明与代码注释不冒充原作者文字。教学CPU变体见复现记录。\n"]}
        explained = copy.deepcopy(original)
        annotated = copy.deepcopy(original)
        explained["cells"] = [copy.deepcopy(intro)]
        annotated["cells"] = [copy.deepcopy(intro)]
        reading = "# -*- coding: utf-8 -*-\n# 原课程代码中文注释阅读版；不是直接运行原GPU/远程权重的承诺。\n"
        md.append("## " + stem + "\n")
        for i, (cell, note) in enumerate(zip(original["cells"], notes)):
            note = "【必须会】" + note if cell["cell_type"] == "code" else "【建议会】" + note
            text = "### 原单元 " + str(i) + "（" + cell["cell_type"] + "）\n\n" + note + "\n"
            added = {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
            explained["cells"] += [copy.deepcopy(added), copy.deepcopy(cell)]
            a = copy.deepcopy(cell)
            src = "".join(cell["source"])
            if cell["cell_type"] == "code":
                if "??" in src:
                    body = "# " + note + "\n" + src
                    py = "# " + note + "\n# 原IPython双问号语句，以下是其Python转换。\n" + transformer.transform_cell(src)
                else:
                    body = annotate(src, note)
                    py = body
                a["source"] = body.splitlines(keepends=True)
                reading += "\n# %% 原代码单元 " + str(i) + "\n" + py + "\n"
            elif stem == "lora_merge" and i in [5, 8]:
                # Retain every original Markdown line; annotated snippets are an explicit addition.
                snippets = []
                import re, textwrap
                for block in re.findall(chr(96)*3 + r"[^\n]*\n(.*?)" + chr(96)*3, src, re.S):
                    snippets.append(annotate(textwrap.dedent(block), note))
                extra = "\n\n新增源码注释（上方原片段保留）：\n\n" + "\n\n".join(chr(96)*3+"python\n"+s+chr(96)*3 for s in snippets)
                a["source"] = (src + extra).splitlines(keepends=True)
                for s in snippets:
                    reading += "\n# Markdown单元 " + str(i) + " 源码片段：\n" + "\n".join("# " + l for l in s.splitlines()) + "\n"
                text += extra + "\n"
            annotated["cells"] += [copy.deepcopy(added), a]
            md.append(text)
        for suffix, nb in [("原代码_带说明.ipynb", explained), ("代码注释.ipynb", annotated)]:
            name = suffix if prefix == "main" else prefix + "_" + suffix
            (target / name).write_bytes((json.dumps(nb, ensure_ascii=False, indent=1)+"\n").encode())
        ast.parse(reading)
        (target / (stem + "_代码注释.py")).write_bytes(format_reading(reading).encode())
    (ROOT / "来源清单.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2)+"\n").encode())
    (ROOT / "代码单元解释.md").write_bytes(("\n".join(md).rstrip()+"\n").encode())
    # Exact complete methods, grouped in their original owners; all docstrings preserved.
    core = "# -*- coding: utf-8 -*-\n"
    core += "".join((installed/"mapping.py").read_text(encoding="utf-8").splitlines(keepends=True)[:13])
    core += "\n# PEFT0.12.0完整核心对象摘录，只添加说明；保留原库的重复缩放行为。\n# 这是源码阅读文件，缺少原模块完整imports与其他方法，不作为独立可执行模块。\n\n"
    selections = {
        "mapping.py": {"": ["inject_adapter_in_model"]},
        "tuners/lora/model.py": {"LoraModel": ["_check_add_weighted_adapter", "add_weighted_adapter", "_svd_generalized_task_arithmetic_weighted_adapter", "_generalized_task_arithmetic_weighted_adapter", "merge_and_unload"]},
        "tuners/lora/layer.py": {"Linear": ["get_delta_weight", "forward"], "Embedding": ["get_delta_weight", "forward"]},
    }
    for rel, groups in selections.items():
        s = (installed/rel).read_text(encoding="utf-8")
        tree = ast.parse(s)
        for owner, names in groups.items():
            nodes = next(n for n in tree.body if getattr(n, "name", None) == owner).body if owner else tree.body
            if owner:
                core += "class " + owner + ":\n    # 以下方法完整保留在原owner名下；其余方法依赖原模块。\n"
            for name in names:
                node = next(n for n in nodes if getattr(n, "name", None) == name)
                core += ("    " if owner else "") + "# 来源：" + rel + "；完整方法 " + name + "\n"
                core += annotate(source_text(s, node)) + "\n\n"
    ast.parse(core)
    (target/"PEFT合并核心源码_代码注释.py").write_bytes(core.rstrip().encode()+b"\n")
    Path(sources/"字幕原文.txt").write_bytes(Path("E:/435af888-0796-42aa-bbe5-64d38f6d91c3.txt").read_bytes())
    print("Built three originals / six derivatives / three Python readings / complete PEFT method reading; all98 cells.")

if __name__ == "__main__":
    main()
