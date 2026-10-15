# -*- coding: utf-8 -*-
# 原课程代码中文注释阅读版；不是直接运行原GPU/远程权重的承诺。

# %% 原代码单元 1
# 【必须会】设置GPU可见性及本机7890代理，会改变当前进程环境。本次跳过此格，CPU离线验证不依赖这些设置。
# 导入模块命名空间供后续使用。
import os
# 修改当前进程环境变量；CPU离线验证跳过代理/GPU设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 修改当前进程环境变量；CPU离线验证跳过代理/GPU设置。
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# 修改当前进程环境变量；CPU离线验证跳过代理/GPU设置。
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# %% 原代码单元 2
# 【必须会】整格为安装命令注释，不执行pip。本次明确锁定PEFT0.12.0，不运行upgrade改变环境。
# !pip install --upgrade peft

# %% 原代码单元 3
# 【必须会】PeftConfig读adapter配置；PeftModel管理多个adapter；AutoModelForCausalLM和AutoTokenizer负责基础模型和分词；load_dataset/random在本notebook未实际用于训练。
# 从指定包导入本格需要的真实API；导入不等于加载远程权重。
from peft import PeftModel, PeftConfig
# 从指定包导入本格需要的真实API；导入不等于加载远程权重。
from transformers import AutoModelForCausalLM, AutoTokenizer
# 从指定包导入本格需要的真实API；导入不等于加载远程权重。
from datasets import load_dataset
# 导入模块命名空间供后续使用。
import torch
# 导入模块命名空间供后续使用。
import random
# 从指定包导入本格需要的真实API；导入不等于加载远程权重。
from IPython.display import Image

# Markdown单元 5 源码片段：
# # 【建议会】重要Markdown源码也详注。Linear.weight按[out,in]存储；输入行向量右乘weight.T。普通LoRA s=alpha/r，RSLoRA为alpha/sqrt(r)。公式必须明确ΔW是否已经含s，避免重复乘；本次PEFT0.12.0的SVD支路确实重复缩放，不能直接当理想公式。
# # torch.nn.Linear, fan_in_fan_out: False
# # y = xA^T + b
# self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

# Markdown单元 5 源码片段：
# # 【建议会】重要Markdown源码也详注。Linear.weight按[out,in]存储；输入行向量右乘weight.T。普通LoRA s=alpha/r，RSLoRA为alpha/sqrt(r)。公式必须明确ΔW是否已经含s，避免重复乘；本次PEFT0.12.0的SVD支路确实重复缩放，不能直接当理想公式。
# # lora.layer
# if use_rslora:
#     self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
# else:
#     self.scaling[adapter_name] = lora_alpha / r

# %% 原代码单元 6
# 【必须会】远程Medium图，只保留URL和历史HTML引用。width400不是矩阵维数，本次没有验证远程图片内容。
# 返回远程图片HTML引用，width仅为显示宽度。
Image(url='https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EnUd1eXLvXCxRZj9NW2BeA.png', width=400)

# Markdown单元 8 源码片段：
# # 【建议会】Markdown中的regex实际对model.modules这个绑定方法的字符串做匹配，没有调用modules()；repr可含整个模型而匹配出adapter名、base_layer等。结果不适合自动挑选目标层，补充用named_modules取得真实模块路径。
# # 导入模块命名空间供后续使用。
# import re
# pattern = r'\((\w+)\): Linear'
# # (norobots): Linear
# # (sql): Linear
# # (adcopy): Linear
# # (merge): Linear
# linear_layers = re.findall(pattern, str(model.modules))
# target_modules = list(set(linear_layers))
# target_modules

# %% 原代码单元 10
# 【必须会】原adapter为smangrul/tinyllama_lora_norobots，device cuda仅赋字符串。PeftConfig.from_pretrained下载配置，不加载1.1B基础权重。本次把地址和设备显式改为本地随机教学fixture/CPU。
# adapter仓库地址，区别于基础模型地址。
peft_model_id = "smangrul/tinyllama_lora_norobots"
# 记录设备字符串；本行没有移动模型。
device = "cuda"
# 读取adapter元数据，取得基础模型路径等配置。
config = PeftConfig.from_pretrained(peft_model_id)

# %% 原代码单元 11
# 【必须会】显示base_model_name_or_path、r8、alpha16等配置；历史基础模型TinyLlama1.1B不是本次小模型。
config

# %% 原代码单元 12
# 【必须会】原模型4bit/device_map auto加载需要相应量化后端及权重，本次不执行。CPU变体从本地随机Llama小checkpoint加载FP32，并加载自建真实SentencePiece词表，不冒充原模型。
# 加载基础因果语言模型；原4bit路径未在CPU执行。
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, device_map="auto")
# 加载分词器和特殊token规则，不加载神经权重。
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# %% 原代码单元 13
# 【必须会】tokenizer长度包含新增token；原历史32005，本次教学词表另记录实际数量。
# 分词器含新增token的总词数，应匹配模型可用行数。
len(tokenizer)

# %% 原代码单元 14
# 【必须会】基础模型config.vocab_size与分词器可能不同；原历史32000，差5需调整embedding和输出头。
# 读取当前模型词表容量，不保证等于分词器长度。
model.config.vocab_size

# %% 原代码单元 15
# 【必须会】resize_token_embeddings扩展词表，保留已有行并初始化新增行；须在加载含词表相关权重的adapter前完成。相同长度只是必要条件，ID语义和基础权重也须匹配。
# 扩展输入embedding与输出头，使新增token有对应参数。
model.resize_token_embeddings(len(tokenizer))

# %% 原代码单元 16
# 【必须会】两个词表键集合做差，历史新增pad、im_end及三种角色起始token。只比较字符串不保证ID排列一致；本次用相同五个新增字符串构造明确教学分词器。
# 分别读两个词表的键并取集合差；不验证ID顺序。
AutoTokenizer.from_pretrained("smangrul/tinyllama_lora_norobots").vocab.keys() - AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T").vocab.keys()

# %% 原代码单元 17
# 【必须会】PeftModel.from_pretrained把本地或远程adapter挂到已有基础模型，名字norobots用于后续选择。返回PeftModel包装，区别于低层inject返回原类。
# 包装基础模型并加载第一个命名adapter。
model = PeftModel.from_pretrained(model, peft_model_id, adapter_name="norobots")

# %% 原代码单元 18
# 【必须会】load_adapter分别挂载sql/adcopy；原SQL只作用七个投影层，另外两个还含embed_tokens/lm_head。共同基础模型、词表、目标层形状必须兼容，加载成功不自动证明语义兼容。
# 注册另一组adapter参数；下划线接收未使用返回值。
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
# 注册另一组adapter参数；下划线接收未使用返回值。
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")

# %% 原代码单元 19
# 【必须会】展示配置字典，各键对应命名adapter，并不是三个独立基础模型。
# 读取各adapter配置或keys视图，不表示激活状态。
model.peft_config

# %% 原代码单元 20
# 【必须会】keys视图列出已注册adapter名字，不说明当前激活哪个。
# 读取各adapter配置或keys视图，不表示激活状态。
model.peft_config.keys()

# %% 原代码单元 21
# 【必须会】遍历dict的key/value，打印r、alpha、alpha/r、fan_in_fan_out。三者原均r8/alpha16/s2，布尔False不妨碍Embedding在get_delta_weight内部转置。
# items同时遍历adapter名和配置对象。
for k, v in model.peft_config.items():
    # 显示rank/alpha/普通缩放/存储转置配置。
    print(k, v.r, v.lora_alpha, v.lora_alpha/v.r, v.fan_in_fan_out)

# %% 原代码单元 22
# 【必须会】显示整个PEFT结构，可看Linear与Embedding采用不同A/B容器、共享同一base_layer。
model

# %% 原代码单元 25
# 【必须会】adapters和weights按位置配对，权重2/.3/.7不自动归一化；density .2在纯svd支路不使用。已有merge先删再建。该API返回None并修改模型，默认rank取最大8、alpha也设8→新s1；没有自动set_adapter('merge')。
# 列表顺序必须与weights位置对应。
adapters = ["norobots", "adcopy", "sql"]
# 人为合并系数，不自动归一化；后续同名变量会被覆盖。
weights = [2.0, 0.3, 0.7]
# 将生成的新适配器命名为merge。
adapter_name = "merge"
# 供剪枝等支路使用；本次纯svd不读取它。
density = 0.2
# combination_type = "ties"
# 选择具体合并算法，本次是svd。
combination_type = "svd"
# 重复执行前先检查是否已有同名结果。
if adapter_name in model.peft_config:
    # 删除旧merge配置/因子；可能影响active状态，不能当纯读取。
    model.delete_adapter(adapter_name)
# 生成新A/B，不自动激活，不折叠到基础W。
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type=combination_type, density=density)

# %% 原代码单元 26
# 【必须会】逐adapter打印target_modules，SQL没有embedding/lm_head，因此在这些层只合并另外两个。不能把缺失层当作该adapter的随机权重。
# 按来源列表检查每个adapter的目标集合。
for adapter in adapters:
    # SQL目标缺少embedding/lm_head，逐层应过滤有效来源。
    print(adapter, model.peft_config[adapter].target_modules)

# %% 原代码单元 27
# 【必须会】再次展示结构，新增merge的A/B可见，但可见不代表激活。
model

# %% 原代码单元 28
# 【必须会】仅注释的查看表达式，不执行；也计入代码覆盖。
# model.base_model.model.model.layers[0].mlp

# %% 原代码单元 29
# 【必须会】取得embedding层scaling字典，原和本次均norobots2/adcopy2/merge1，没有sql。
# 按点分层路径读取子层和scaling或类。
model.model.get_submodule('model.embed_tokens').scaling

# %% 原代码单元 30
# 【必须会】原历史为bitsandbytes Linear4bit；本次CPU是普通PEFT Linear，不把类型差异隐去或伪装量化。
# 读取实际Python类，注意torch和PEFT的Linear是不同类。
type(model.model.get_submodule('model.layers.0.self_attn.q_proj'))

# %% 原代码单元 32
# 【必须会】读取Embedding层，base权重[V,H]，LoRA因子A[r,V]/B[H,r]。
model.base_model.model.model.embed_tokens

# %% 原代码单元 33
# 【必须会】四个变量是已有Parameter的引用，不是clone；后续只读乘法未改这些原因子。norobots/adcopy两组，SQL不在该层。
# 引用Embedding的A参数[r,V]；不是复制。
lora_a_1 = model.base_model.model.model.embed_tokens.lora_embedding_A['norobots']
# 引用Embedding的A参数[r,V]；不是复制。
lora_a_2 = model.base_model.model.model.embed_tokens.lora_embedding_A['adcopy']
# 引用Embedding的B参数[H,r]；不是复制。
lora_b_1 = model.base_model.model.model.embed_tokens.lora_embedding_B['norobots']
# 引用Embedding的B参数[H,r]；不是复制。
lora_b_2 = model.base_model.model.model.embed_tokens.lora_embedding_B['adcopy']

# %% 原代码单元 34
# 【必须会】原A[8,32005]、B[2048,8]；本次小模型H16、词表约百，形状和真实测量在报告中给出。
lora_a_1.shape, lora_b_1.shape, lora_a_2.shape, lora_b_2.shape

# %% 原代码单元 35
# 【必须会】weights全局被覆盖成两项；valid_weights已经含s=2。之后delta_lora_i又含s，累计重复缩放。三来源原权重在此前API调用时已使用，不被此格追溯修改。
# 本例两组源adapter缩放均为16/8=2。
scaling = [2, 2]
# 人为合并系数，不自动归一化；后续同名变量会被覆盖。
weights = [2, 0.3]
# 这里已经把合并系数乘s；与后面已缩放delta叠加会重复。
valid_weights = [2*2, 2*0.3]

# %% 原代码单元 36
# 【必须会】transpose为本notebook普通函数。False直接返回同对象；True返回.T视图，若是Parameter则再包成新Parameter，可改变autograd叶对象。本次乘积是Tensor，走最后分支。Embedding转置与配置fan_in_fan_out=False不是矛盾。
# 普通函数显式控制二维转置；参数是Tensor或Parameter。
def transpose(weight, fan_in_fan_out):
    # False时保留原对象和shape。
    if not fan_in_fan_out:
        # 返回原对象、转置视图或新Parameter，详见本格说明。
        return weight

    # Parameter是可注册参数；重新包装会改变autograd叶对象。
    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    # 返回原对象、转置视图或新Parameter，详见本格说明。
    return weight.T

# %% 原代码单元 37
# 【必须会】BA[H,V]转置成[V,H]并乘scaling2，得到已缩放ΔW_i。原main源码行号漂移；本次完整固定源码读本可直接对照。
# https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L450-L482
# B@A后转置并乘源s，得到已缩放Embedding增量[V,H]。
delta_lora_1 = transpose(lora_b_1 @ lora_a_1, True) * scaling[0]
# 同上，为第二个源adapter的已缩放增量。
delta_lora_2 = transpose(lora_b_2 @ lora_a_2, True) * scaling[1]

# %% 原代码单元 38
# 【必须会】展示第一个已缩放Embedding增量，不是新基础模型参数。
delta_lora_1

# %% 原代码单元 39
# 【必须会】展示第二个已缩放增量；两者均与Embedding.weight同shape。
delta_lora_2

# %% 原代码单元 40
# 【必须会】确认[V,H]；原[32005,2048]，CPU教学采用实际小shape。
delta_lora_1.shape, delta_lora_2.shape

# %% 原代码单元 41
# 【必须会】valid_weights又乘过s，使结果为sum(w_i*s_i²*B_i*A_i).T。特别注意PEFT0.12.0的get_delta_weight与SVD helper也有这条重复缩放路径；因此本格可匹配旧库但偏离理想sum(w_i*ΔW_i)。
# 再乘valid_weights里的s，得到双重缩放的矩阵。
delta_lora = valid_weights[0] * delta_lora_1 + valid_weights[1] * delta_lora_2
delta_lora

# %% 原代码单元 42
# 【必须会】再转置到[H,V]做SVD，保留8项并把Σ吸收进U。原full_matrices=True的Vh[32005,32005]单FP32张量约3.82GiB，未在原大尺寸运行；本次小fixture保留True真实执行。
# 转回[H,V]以分解成B[H,r]和A[r,V]。
delta_lora = delta_lora.T
# 返回奇异向量/奇异值；full=True在大矩形词表上很占内存。
U, S, Vh = torch.linalg.svd(delta_lora.float(), full_matrices=True)
# 保留左奇异向量前r列；切片通常是view。
U = U[:, :8]
# 保留前r个最大奇异值。
S = S[:8]
# 把奇异值吸收进左因子；此后U不再正交。
U = U @ torch.diag(S)
# 保留右奇异向量前r行，对应新A。
Vh = Vh[:8, :]

# %% 原代码单元 43
# 【必须会】Vh[8,V]就是新A形状，UΣ[H,8]就是新B形状，顺序不能反过来。
Vh.shape, U.shape

# %% 原代码单元 44
# 【必须会】单独看Vh不能证明整块ΔW一致，奇异向量有符号/简并旋转不唯一性。
Vh

# %% 原代码单元 45
# 【必须会】读取库创建的merge因子引用；没有把手写U/Vh赋回模型。
# 引用Embedding的A参数[r,V]；不是复制。
lora_a_new = model.base_model.model.model.embed_tokens.lora_embedding_A['merge']
# 引用Embedding的B参数[H,r]；不是复制。
lora_b_new = model.base_model.model.model.embed_tokens.lora_embedding_B['merge']

# %% 原代码单元 46
# 【必须会】核对库新A/B形状；应比较乘积及实际输出，而不是要求单因子元素位级相同。
lora_a_new.shape, lora_b_new.shape

# %% 原代码单元 47
# 【必须会】只显示A，未对B或完整ΔW做断言。新增验证比较重建矩阵、尾部奇异值误差以及缩放约定。
lora_a_new

# %% 原代码单元 50
# 【必须会】apply_chat_template用角色/content拼接模板，tokenize=False返回str，add_generation_prompt追加assistant起始。本次使用明确自建ChatML教学模板，不宣称来自原Hub资产。
# 构造角色/content字典列表，尚未分词。
messages = [
    {"role": "user", "content": "Write an essay about Generative AI."},
]
# 套模板返回str，追加assistant生成提示。
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
text

# %% 原代码单元 51
# 【必须会】tokenizer生成int64[1,T]及mask；原.to(cuda)和256续写本次显式改CPU/4。采样top_p .95/temp .2/重复惩罚1.2；decode包括prompt。仍是norobots激活，随机输出不作为任务能力证据。
# 字符串转int64词ID和attention_mask，通常[1,T]。
inputs = tokenizer(text, return_tensors="pt")
# 字典推导式逐张量移到指定设备，原来是cuda。
inputs = {k: v.to("cuda") for k, v in inputs.items()}
# 自回归生成；总输出长度含prompt，检查当前激活adapter。
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
# 把整条输出ID解码，包含prompt且可能含特殊token。
print(tokenizer.decode(outputs[0]))

# %% 原代码单元 53
# 【必须会】system/user消息描述商品；模板返回一条字符串。原文字原样保留，不把商品内容当成执行指令。
# 构造角色/content字典列表，尚未分词。
messages = [
    {"role": "system", "content": "Create a text ad given the following product and description."},
    {"role": "user", "content": "Product: Sony PS5 PlayStation Console\nDescription: The PS5 console unleashes new gaming possibilities that you never anticipated."},
]
# 套模板返回str，追加assistant生成提示。
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
text

# %% 原代码单元 54
# 【必须会】同样显式CPU/4续写变体，保留原采样参数；实际生成不是广告效果或合并优劣评测。
# 字符串转int64词ID和attention_mask，通常[1,T]。
inputs = tokenizer(text, return_tensors="pt")
# 字典推导式逐张量移到指定设备，原来是cuda。
inputs = {k: v.to("cuda") for k, v in inputs.items()}
# 自回归生成；总输出长度含prompt，检查当前激活adapter。
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
# 把整条输出ID解码，包含prompt且可能含特殊token。
print(tokenizer.decode(outputs[0]))

# %% 原代码单元 56
# 【必须会】三引号多行字符串保持原内容，列名/问题作为模型输入，print只是展示；不是执行数据库SQL。
# 保存多行SQL提示字符串，不执行数据库命令。
text = """Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query:"""
print(text)

# %% 原代码单元 57
# 【必须会】原64续写改4、cuda改cpu。tokenizer('</s>').input_ids[-1]取EOS，必须检查tokenizer语义。默认do_sample=False为greedy，未显式set_adapter，所以不能据这条历史回答证明merge有效。
# 字符串转int64词ID和attention_mask，通常[1,T]。
inputs = tokenizer(text, return_tensors="pt")
# 字典推导式逐张量移到指定设备，原来是cuda。
inputs = {k: v.to("cuda") for k, v in inputs.items()}
# 自回归生成；总输出长度含prompt，检查当前激活adapter。
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tokenizer("</s>").input_ids[-1])
# 把整条输出ID解码，包含prompt且可能含特殊token。
print(tokenizer.decode(outputs[0]))

# %% 原代码单元 58
# 【必须会】空代码单元保留原顺序、输出、元数据并解释，执行为空操作。
