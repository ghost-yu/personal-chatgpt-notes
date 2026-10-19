# 第04讲 PEFT / LoRA 源码注释阅读版
# 来源：chunhuizhang/personal_chatgpt，提交 7fbc29824171fd07dc2c30fe922e61ebc018ad1d。
# 无JSON或历史输出，以 # %% 分格；建议在配套notebook逐格核对环境后执行。
# 六段Markdown源码节选保留为注释，包含原缺陷，不作为独立可运行类/脚本。
# 固定官方源码的离线CPU注入验证见复现/scripts/verify_injection.py。

# %% [markdown]
# # 第04讲中文代码说明
#
# 原 notebook 单独保存；阅读版保留原单元顺序，便于对照。
# 注释解释原语句及 Markdown 源码节选，配套运行环境与结果见复现说明。

# %% [markdown]
# ## 单元 0：为什么读源码
#
# 这一段介绍读源码的用途：沿函数调用理解自动注入、默认目标和参数冻结。后续重点追踪 get_peft_model 怎样改变模型结构。

# %% [markdown]
# - 源码面前，了无秘密
#     - 学习成熟的框架，提升代码水平
#     - 后续直接改源码

# %% [markdown]
# ## 单元 1：type 标题
#
# Markdown标题引出三种类型。它不调用type()，也不创建Python类型变量。

# %% [markdown]
# ## type

# %% [markdown]
# ## 单元 2：任务映射与适配方法映射
#
# 两个围栏均为字典源码节选，Markdown不会执行。键选择实现类；值为**类对象**，不是已创建的模型实例。冒号是键值对应，逗号分隔项，大括号构造dict；下划线/大写名字表示常量命名习惯，不保证不可修改。
#
# 第一张映射：SEQ_CLS对应序列分类包装；SEQ_2_SEQ_LM对应编码器-解码器语言模型包装；CAUSAL_LM对应因果LM包装；TOKEN_CLS对应逐token分类包装。虽然常量叫MODEL_TYPE，它在这里按task_type选择任务外层，不是按bloom/llama架构选择。
#
# 第二张映射：PeftType.LORA→LoraModel、PROMPT_TUNING→PromptEmbedding、P_TUNING→PromptEncoder、PREFIX_TUNING→PrefixEncoder、ADALORA→AdaLoraModel、ADAPTION_PROMPT→AdaptionPromptModel。PeftType为Enum，`.LORA`为枚举成员；各方法的实现和参数形态不同，不能把prompt embedding也称为LoRA矩阵。
#
# 两段依赖已导入各类名和PeftType，原节选未给这些import。在隔离字典上下文补入固定官方类，真实执行两段并与v0.3.0映射逐项比较；不是实例化全部适配方法。真正get_peft_model先按任务创建包装，再在非prompt学习路径选择LoraModel，详见主文。

# %% [markdown]
#
# - **task_type/model_type**:
#
#
#     - `LoraConfig(task_type='CAUSAL_LM')`
#
#     ```
#     # 这是任务类型到包装类的dict，依赖这些类名已导入；Markdown本身不执行。
#     # 字符串键对应task_type，值是类对象而不是构造好的模型；这里并非bloom/llama架构映射。
#     # 按任务类型选外层包装，值是类对象而不是模型实例。
#     MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
#         # 序列级分类任务包装。
#         "SEQ_CLS": PeftModelForSequenceClassification,
#         # 编码器-解码器语言模型任务包装。
#         "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
#         # 因果语言模型任务包装，BLOOM使用此分支。
#         "CAUSAL_LM": PeftModelForCausalLM,
#         # 逐token分类任务包装。
#         "TOKEN_CLS": PeftModelForTokenClassification,
#     }
#     ```
#
# - **peft_type**
#
#     ```
#     # 这是适配方法到实现类的dict，键是PeftType枚举成员，值仍是类对象。
#     # LORA、PROMPT_TUNING和PREFIX_TUNING管理不同形式的参数，不能都叫低秩权重矩阵。
#     # 按适配方法枚举选实现类；需要各类名/PeftType预先导入。
#     PEFT_TYPE_TO_MODEL_MAPPING = {
#         # 低秩更新的注入与adapter管理。
#         PeftType.LORA: LoraModel,
#         # 可训练软prompt embedding，不是LoRA矩阵。
#         PeftType.PROMPT_TUNING: PromptEmbedding,
#         # prompt编码器实现。
#         PeftType.P_TUNING: PromptEncoder,
#         # prefix编码器实现，与低秩权重支路不同。
#         PeftType.PREFIX_TUNING: PrefixEncoder,
#         # 自适应低秩分配方法。
#         PeftType.ADALORA: AdaLoraModel,
#         # 另一种提示适配方法，非本例LORA路径。
#         PeftType.ADAPTION_PROMPT: AdaptionPromptModel,
#     }
#     ```

# %% [markdown]
# ## 单元 3：PeftModel 标题
#
# 这是介绍PeftModel的Markdown标题，反引号让类名按代码样式显示。

# %% [markdown]
# ## `PeftModel`

# %% [markdown]
# ## 单元 4：包装对象的初始化属性
#
# 这段依赖方法内部self、传入的model和adapter_name。self.base_model=model保存对象引用；self.config=self.base_model.config复用配置引用；active_adapter保存当前adapter名字，通常为'default'。字符串'default'是adapter标识，不是任务CAUSAL_LM。
#
# 这三条赋值不复制基权重，也没有完成LoRA注入。v0.3.0完整PeftModel.__init__随后把base_model进一步包装为LoraModel。只看这个节选不能以为最后base_model永远是裸BLOOM。
#
# 普通顶层会因self未定义失败，故原围栏不是可独立运行文件。新增验证在明确隔离的属性上下文检查引用关系，同时实际调用官方构造器检查最终外层→LoraModel→原BLOOM结构，二者的证据范围分开记录。这里关注的是Python对象之间的引用关系。

# %% [markdown]
# ```
# # 这是方法内部的初始化节选，需要已有self、model和adapter_name。
# # self是正在构造的包装实例；三条赋值保存引用与名字，无Tensor计算或权重复制。
# # 完整PeftModel构造器随后会继续包装base_model；节选没有展示整个注入过程。
# # 保存传入模型引用；is可核对身份，没有clone/deepcopy或权重分配。
# self.base_model = model
# # 访问基模型配置并保存同一对象引用；改配置可能影响双方，不是加载新checkpoint。
# self.config = self.base_model.config
# # adapter_name="default"
# # 保存当前adapter名字str，默认通常为default；与CAUSAL_LM任务类型不同。
# self.active_adapter = adapter_name
# ```

# %% [markdown]
# ## 单元 5：配置与注入模块标题
#
# Markdown对比LoraConfig与LoraModel：前者存超参数，后者管理注入和adapter；配置对象保存选项，模型对象按配置管理各层。

# %% [markdown]
# ## LoraConfig 与 LoraModel

# %% [markdown]
# ## 单元 6：LoraConfig 标题
#
# 引出数据类源码节选。标题不定义类，不运行dataclass装饰器。

# %% [markdown]
# ### LoraConfig

# %% [markdown]
# ## 单元 7：不完整的 dataclass 源码
#
# @dataclass为装饰器，给类生成__init__等方法；LoraConfig继承PeftConfig；冒号为类型注解，field保存默认值/metadata。metadata中的help是说明，不是算法或校验代码。类型注解不会自动验证传入参数。
#
# | 源字段 | 原节选含义 | 限制 |
# |---|---|---|
# | r:int=field(default=8,...) | 默认瓶颈8，int配置 | 不是后面真正配置的r=16 |
# | target_modules:Optional[Union[List[str],str]] | 列表/正则字符串/None的类型说明 | 原节选没有默认值赋值；Optional不自动等于default=None |
# | lora_alpha:int=field(default=None,...) | 原节选默认None | 使用前应给有效数值或看完整版本处理，不能把None当1 |
# | lora_dropout:float=field(default=None,...) | 原节选默认None | 不等于概率0 |
#
# 独立运行还缺dataclass、field、typing及PeftConfig导入。即使补这些真实依赖，本节选也在类装饰阶段报`TypeError: non-default argument 'target_modules' follows default argument`：有默认的r后面又出现未设默认的target_modules。
#
# 固定官方v0.3.0完整定义实际给target_modules加了field(default=None,...)，还有fan_in_fan_out、bias、modules_to_save、init_lora_weights和__post_init__等。后者设置peft_type=LORA。节选缺少这些内容，不能把本格当成完整类或注入结果。中文注释版只标明缺陷，保留原字句/AST，不悄悄补默认值。

# %% [markdown]
# ```
# # 【原文缺陷】本数据类节选缺少导入，以及target_modules的默认field。
# # 即使补齐真实导入/父类，默认r后出现无默认target_modules仍会触发TypeError。
# # 这里只增加注释，不补字段；真正运行的9格导入包中完整LoraConfig，与此节选不同。
# # 装饰器在类定义结束后生成 __init__ 等方法，帮助配置类组织字段。
# @dataclass
# # 继承PeftConfig的配置类；类型注解不自动进行运行时参数类型校验。
# class LoraConfig(PeftConfig):
#     # field(default=8,metadata=...)给整数r设置默认8；help只提供字段说明，不执行算法。
#     r: int = field(default=8, metadata={"help": "Lora attention dimension"})
#     # Optional[Union[List[str],str]]表示列表/字符串/None的类型范围，不自动生成default=None。
#     # 本节选漏了默认field；在已有默认r后放无默认字段，dataclass装饰会报字段顺序TypeError。
#     target_modules: Optional[Union[List[str], str]]
#     # 节选默认None，不能不核对完整配置就直接拿来做缩放。
#     lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
#     # 节选默认None，不等于概率0；help metadata只提供说明。
#     lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
#
# ```

# %% [markdown]
# ## 单元 8：默认目标字典与模块扫描
#
# 本格两段源码分别是架构映射与扫描逻辑。开头“default:query_key_value”只适用于BLOOM等对应架构，不是全部模型的统一默认。
#
# 目标字典的值为list[str]：t5/mt5用q/v；bart、opt、gptj、gpt_neo、llama用q_proj/v_proj；gpt2用c_attn；bloom、gpt_neox、chatglm用query_key_value；blip-2同时列q/v/q_proj/v_proj；bert、roberta、xlm-roberta、electra、layoutlm用query/value；deberta-v2用query_proj/value_proj；deberta用in_proj。中文注释逐项解释其为各架构的模块名，不能在BLOOM上换成LLaMA名字。
#
# 扫描中的self是LoraModel，self.model是原模型。named_modules迭代路径与模块对象，列表推导式保存key路径快照；`_`忽略模块值。先快照后替换避免遍历模块树时直接改变遍历集合。它不同于named_parameters的参数名字与Parameter。
#
# target_modules为str时，re.fullmatch返回Match或None，匹配**完整路径**；为列表时，生成器逐项检查key.endswith(target_key)，any将布尔结果归并。因此裸正则'query_key_value'不匹配带transformer前缀的完整路径，而列表['query_key_value']能匹配后缀；正则'.*query_key_value'可匹配。
#
# 原扫描节选依赖self/lora_config/re，而且只得到target_module_found，不包含后续真正替换。匹配失败应结合完整库的ValueError，不是这段else自己抛错。本次用真实BLOOM模块树执行原扫描片段，比较列表、完整正则、裸正则及错误目标的结果；未知/找不到目标的真实get_peft_model路径另行验证。
#
# 以H=4096融合QKV为例，原weight为[12288,4096]，r=16时A为[16,4096]、B为[12288,16]，输入[B,S,4096]经过小支路变为[B,S,12288]。本格字典/扫描本身不做这些矩阵乘法；真正前向在库Linear或Linear8bitLt内。

# %% [markdown]
# - `target_modules`
#     - The names of the modules to apply Lora to.
#     - default: `query_key_value`
#         - `'transformer.h.0.self_attention.query_key_value'`
#     - 都意味着一个大矩阵 $W\in \mathbb R^{A\times B}$；
#
#
#
# ```
# # 这是模型架构到默认目标后缀列表的dict，不是task_type或peft_type映射。
# # BLOOM用融合query_key_value；LLaMA用分离q_proj/v_proj，不能在不同架构间照搬名字。
# # 按架构选择默认模块名；query_key_value不是所有模型的默认。
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
#     # 此架构默认目标为T5的q/v投影；不是统一目标。
#     "t5": ["q", "v"],
#     # 此架构默认目标为mT5的q/v投影；不是统一目标。
#     "mt5": ["q", "v"],
#     # 此架构默认目标为BART的分离q_proj/v_proj；不是统一目标。
#     "bart": ["q_proj", "v_proj"],
#     # 此架构默认目标为GPT-2融合c_attn投影（Transformers Conv1D）；不是统一目标。
#     "gpt2": ["c_attn"],
#     # 此架构默认目标为BLOOM融合query_key_value；不是统一目标。
#     "bloom": ["query_key_value"],
#     # 此架构默认目标为BLIP-2不同子模块的q/v和q_proj/v_proj；不是统一目标。
#     "blip-2": ["q", "v", "q_proj", "v_proj"],
#     # 此架构默认目标为OPT分离q_proj/v_proj；不是统一目标。
#     "opt": ["q_proj", "v_proj"],
#     # 此架构默认目标为GPT-J分离q_proj/v_proj；不是统一目标。
#     "gptj": ["q_proj", "v_proj"],
#     # 此架构默认目标为GPT-NeoX融合query_key_value；不是统一目标。
#     "gpt_neox": ["query_key_value"],
#     # 此架构默认目标为GPT-Neo分离q_proj/v_proj；不是统一目标。
#     "gpt_neo": ["q_proj", "v_proj"],
#     # 此架构默认目标为BERT的query/value；不是统一目标。
#     "bert": ["query", "value"],
#     # 此架构默认目标为RoBERTa的query/value；不是统一目标。
#     "roberta": ["query", "value"],
#     # 此架构默认目标为XLM-R的query/value；不是统一目标。
#     "xlm-roberta": ["query", "value"],
#     # 此架构默认目标为ELECTRA的query/value；不是统一目标。
#     "electra": ["query", "value"],
#     # 此架构默认目标为DeBERTa-v2的query_proj/value_proj；不是统一目标。
#     "deberta-v2": ["query_proj", "value_proj"],
#     # 此架构默认目标为DeBERTa融合in_proj；不是统一目标。
#     "deberta": ["in_proj"],
#     # 此架构默认目标为LayoutLM的query/value；不是统一目标。
#     "layoutlm": ["query", "value"],
#     # 此架构默认目标为LLaMA分离q_proj/v_proj；不是统一目标。
#     "llama": ["q_proj", "v_proj"],
#     # 此架构默认目标为ChatGLM融合query_key_value；不是统一目标。
#     "chatglm": ["query_key_value"],
# }
# ```
#
# ```
# # 这是LoraModel内部的扫描节选，依赖self.model、lora_config和re模块。
# # 只计算是否匹配目标，不创建LoRA参数或执行模块替换；真正替换在_find_and_replace后续。
# # key是完整模块路径str；判断结果可能是Match/None或bool，没有输入Tensor和shape变换。
# # named_modules()迭代(完整路径str,模块对象)，列表推导式只保留key；_表示忽略对象。
# # 保存路径快照，便于后续改模块树；named_modules与named_parameters的内容不同。
# key_list = [key for key, _ in self.model.named_modules()]
# # 依次遍历完整路径；本节选不替换模块，后续官方实现才按匹配结果执行替换。
# for key in key_list:
#     # isinstance(...,str)区分完整路径正则模式与目标名列表；不是区分模型类别。
#     if isinstance(lora_config.target_modules, str):
#         # re.fullmatch(pattern,key)要求正则覆盖整个路径，返回Match或None。
#         # 裸query_key_value无法匹配带前缀路径，.*query_key_value可匹配；不是子串search。
#         target_module_found = re.fullmatch(lora_config.target_modules, key)
#     # 列表目标通过后缀检查，与完整正则语义不同。
#     else:
#         # key.endswith(target_key)逐个检查后缀；生成器供any短路判断，最终返回bool。
#         # 列表[query_key_value]可匹配完整路径末尾；它与裸正则字符串的语义不同。
#         target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
# ```
#

# %% [markdown]
# ## 单元 9：真正构造配置对象
#
# from peft 导入当前安装版本的 LoraConfig。这里 r=16、alpha=32、dropout=0.05、bias=none、task=CAUSAL_LM，修正分支缩放为 alpha/r=2。
#
# target_modules那行仍是注释，q_proj/v_proj不生效，默认由架构bloom映射融合QKV。本格默认构造可训练adapter配置，未进入forward、冻结或替换。metadata帮助手段不影响此处数值。

# %%
# 使用包中完整LoraConfig构造实例，非Markdown节选；r16、alpha32、scale2。
# from peft import LoraConfig 导入包中的完整配置类，并未执行Markdown7的节选。
# LoraConfig(...) 创建配置对象config，无输入token、logits或参数梯度。
# 原实例r=16、alpha=32，标准缩放为2；dropout和bias策略由后面注入层使用。
# target_modules那行是注释，没有选择q_proj/v_proj；BLOOM默认选择融合query_key_value。
from peft import LoraConfig
config = LoraConfig(
    # 瓶颈维度16，对应两个小因子的公共维度。
    r=16, #low rank
    # 缩放分子32，标准alpha/r=2。
    lora_alpha=32, #alpha scaling， scale lora weights/outputs
    # target_modules=["q_proj", "v_proj"], #if you know the
    # adapter支路训练dropout概率0.05。
    lora_dropout=0.05,
    # 不额外训练偏置，但保留旧bias参与前向。
    bias="none",
    # 按CAUSAL_LM选择任务包装；target_modules注释不生效。
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

# %% [markdown]
# ## 单元 10：model 到 loramodel 标题
#
# Markdown标题表示接下来加载及包装流程。箭头不是Python运算，不复制权重。

# %% [markdown]
# ### model => loramodel

# %% [markdown]
# ## 单元 11：导入 Auto 接口
#
# 导入AutoTokenizer、AutoConfig、AutoModelForCausalLM三个类；实际后续只用最后一个加载。前两者在本notebook未被调用，不能据导入宣称词表/config另下载成功。导入不构造7B模型。

# %%
# 导入三个Auto类；本文件实际仅调用AutoModelForCausalLM加载。
# from transformers import ... 把三个Auto类绑定到当前命名空间。
# Auto类按配置选择具体实现；本notebook后面只调用AutoModelForCausalLM。
# 导入另外两类不等于已加载tokenizer或另一次AutoConfig下载。
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# %% [markdown]
# ## 单元 12：7B 量化加载
#
# AutoModelForCausalLM.from_pretrained为类方法，模型ID为bigscience/bloom-7b1，load_in_8bit=True请求旧式bitsandbytes量化，device_map='auto'自动设备分派。返回BLOOM因果LM对象，并可能联网/缓存/分配设备内存；原代码没有model revision。
#
# load_in_8bit 接入量化加载，device_map=auto 分配模型模块。加载后的目标层类型决定后面采用的适配路径，如 Linear8bitLt；普通 CPU Linear 路径在配套小 BLOOM 例子中展开。

# %%
# 按仓库 ID 加载7B模型，并设置量化与设备分配；运行条件见复现说明。
# from_pretrained 是类方法，接收str模型ID与加载选项，返回因果语言模型模块。
# 可能联网读取配置和checkpoint、写缓存、分配设备内存；原文未固定revision。
# 原BLOOM-7b1隐藏宽度H=4096，融合QKV权重为[12288,4096]（nn.Linear的[out,in]方向）。
# load_in_8bit是历史量化接口；后续可能进入Linear8bitLt，而CPU普通Linear验证不能覆盖该路径。
# 尚未传入int64 token IDs [B,S]；这行模型加载不是forward或generate。
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    # 旧式bitsandbytes量化加载，不保证全部参数/激活都是int8。
    load_in_8bit=True,
    # auto自动设备分派，不等于数据并行world_size。
    device_map='auto',
)

# %% [markdown]
# ## 单元 13：查看原模型
#
# 末尾 model 让 notebook 展示当前 BLOOM 的模型树。接下来可对照注入 LoRA 后的结构，找到新增包装和目标层；普通脚本使用 print(model) 显示。

# %%
# 展示基础模型的模块层次，方便与注入LoRA后的结构对照。
# 末尾表达式返回当前裸模型对象，notebook显示repr；普通Python不自动显示。
# 通过模块名定位后续要替换的线性层。
model

# %% [markdown]
# ## 单元 14：导入注入函数
#
# get_peft_model是包级普通函数，非model实例方法；输入为模型模块与PeftConfig，返回PEFT包装。导入本身不改变模型，也不定义另一套随机权重。

# %%
# 导入包级普通函数get_peft_model；导入尚不修改模型。
# get_peft_model 是包级独立函数，不是 model.get_peft_model 实例方法。
# 导入只绑定函数名字；真正调用在15格，输入是模型模块与PEFT配置。
from peft import get_peft_model

# %% [markdown]
# ## 单元 15：真正执行 LoRA 注入
#
# 调用get_peft_model(model,config)，赋回model名字。原BLOOM内的目标子层被改造，外层为PeftModelForCausalLM，中间base_model为LoraModel，其model指向原BLOOM；包装不是三个完整网络，也不是深拷贝。
#
# 官方v0.3.0的_find_and_replace找到目标，_replace_module用setattr改变父属性，并把旧weight/bias对象引用交给新层。mark_only_lora_as_trainable将非lora_参数冻结。bias='none'不训练原偏置，但前向仍使用它。
#
# 配套 PEFT v0.3.0 CPU 算例沿实际调用观察目标层类型变化、旧 weight/bias 的对象身份及可训练参数集合，结果见复现记录。

# %%
# 包装并替换内部目标子层；复用旧基权重对象，不是深拷贝。
# get_peft_model(model,config) 接收已有模型对象和LoraConfig，返回PEFT任务包装。
# 赋回model重新绑定名字；原BLOOM对象保留，在内部被LoraModel引用，没有深拷贝整网。
# task_type选择PeftModelForCausalLM，peft_type选择LoraModel，model_type=bloom选择默认目标名。
# 官方_find_and_replace扫描模块树，_replace_module通过setattr替换父模块的子层属性。
# 新层复用原weight/bias Parameter，增加A/B小因子；bias=none仍保留旧偏置参与前向。
# 冻结基参数而训练lora_参数；这不是关闭整条计算图或已经执行优化器step。
# 原H=4096时A:[16,4096]、B:[12288,16]，输入[B,S,4096]的增量输出[B,S,12288]。
# 配套算例使用 PEFT v0.3.0 的 CPU Linear 路径，逐步展示注入过程。
model = get_peft_model(model, config)

# %% [markdown]
# ## 单元 16：查看注入后模型
#
# 表达式model显示包装和lora_A/lora_B等模块。沿repr中的层名，可以找到基线性层及新加入的lora_A、lora_B；用requires_grad查看对应权重是否参与训练。
#
# 固定源码CPU验证将两层H16融合QKV替换成真实适配Linear，A[16,16]、B[48,16]，scale2；新增2048个可训练参数。这是小模型尺寸，与原7B的4096/12288/30层不同，原尺寸对应的参数推导见主文。

# %%
# 展示注入后的模块结构，可在输出中找到lora_A和lora_B。
# 外层为PeftModelForCausalLM，中间base_model为LoraModel，其model指向原BLOOM。
# 这些是包装层次，不是三份完整模型；结构repr无法证明requires_grad或权重已训练。
model

# %% [markdown]
# ## 单元 17：空代码格
#
# 原source为空，outputs也为空。保留空格及原元数据，说明版不删它；中文注释版仅加入“原为空，没有运算”的注释，AST仍是空Module。不能把原空格算作模型运行或补写训练代码。

# %%
# 原单元为空，没有运算；保留原元数据和输出，仅加入此说明注释。
# 原source为空；注释版仅加注释，AST仍为空Module，不新增训练代码或模型结果。
