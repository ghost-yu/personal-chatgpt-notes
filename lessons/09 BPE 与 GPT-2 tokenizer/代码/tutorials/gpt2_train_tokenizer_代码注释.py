# -*- coding: utf-8 -*-
# 第09讲Python阅读版：原语句说明及历史错误边界见逐格内容。
# 主要用于阅读；远程语料、代理、Hub上传未在本次验证执行。

# %% [markdown]
# ## 单元 0（markdown）：训练tokenizer入口
#
# 从代码文本学习新的BPE词表与merges，不训练Transformer。

# %% [markdown] 原单元 0
# ## tokenizer

# %% [markdown]
# ## 单元 1（code）：导入加载、语料与进度接口
#
# AutoTokenizer是transformers的类；load_dataset是datasets函数；tqdm是可迭代对象的进度包装器，不负责数据本身或训练算法。导入不读语料。

# %% 原单元 1
# 导入加载、语料与进度接口；原执行语句保留，详细参数/shape/边界见本格说明。
# 根据checkpoint选择具体tokenizer类，导入不加载语言模型权重。
from transformers import AutoTokenizer
# 真实datasets接口；导入与下载/读取语料分开。
from datasets import load_dataset
# 包装可迭代对象显示进度，不统计BPE pair。
from tqdm import tqdm

# %% [markdown]
# ## 单元 2（code）：原本地codeparrot路径
#
# load_dataset(./codeparrot/,split=train,streaming=True)相对当前工作目录；返回流式IterableDataset时按需读。原仓库没有本地语料，不能仅凭streaming生成文件。原格保留但未运行；独立补充用临时JSONL与真实datasets json读取，不假冒课程语料。

# %% 原单元 2
# 原本地codeparrot路径；原执行语句保留，详细参数/shape/边界见本格说明。
# 原语料入口依赖路径/远程数据；本次未以原配置运行。
dataset = load_dataset('./codeparrot/', split='train', streaming=True)

# %% [markdown]
# ## 单元 3（code）：建立有消费位置的迭代器
#
# iter(dataset)返回迭代器，next每调用一次消费一条dict；iter_dataset跨后续函数共用，没有自动复位。本次在明确小语料IterableDataset上下文执行原句。

# %% 原单元 3
# 建立有消费位置的迭代器；原执行语句保留，详细参数/shape/边界见本格说明。
# 建立有消费位置的迭代器；两次训练共用它。
iter_dataset = iter(dataset)

# %% [markdown]
# ## 单元 4（code）：旧GPT2 tokenizer作为处理模板
#
# 加载原GPT2 fast tokenizer；后面重建词表不会保留原ID语义。本地路径补充仅改字符串/local_files_only，固定revision且无权重。

# %% 原单元 4
# 旧GPT2 tokenizer作为处理模板；原执行语句保留，详细参数/shape/边界见本格说明。
# 原远程GPT2路径；验证补充明确只改为固定本地资产。
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# %% [markdown]
# ## 单元 5（markdown）：构建alphabet与训练批标题
#
# 先准备256个基础符号，再提供字符串批给trainer。

# %% [markdown] 原单元 5
# ### build

# %% [markdown]
# ## 单元 6（code）：字节正反映射与基础词表
#
# 无参bytes_to_unicode返回256项dict；(v,k)反转items，list(keys)收集映射字符，base_vocab为list[str]长度256，不是完整50257项词表。initial_alphabet用于覆盖任意UTF8字节。

# %% 原单元 6
# 字节正反映射与基础词表；原执行语句保留，详细参数/shape/边界见本格说明。
# 导入GPT2的256-byte可逆显示映射函数。
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
# 256项byte→唯一显示字符字典，不是token ID词表。
bytes_to_unicode_map = bytes_to_unicode()
# 生成器交换(key,value)，得到显示字符→byte反向字典。
unicode_to_bytes_map = dict((v, k) for k, v in bytes_to_unicode_map.items())
# list保留映射键顺序，基础alphabet含256显示字符。
base_vocab = list(unicode_to_bytes_map.keys())

# %% [markdown]
# ## 单元 7（code）：length与batch_iterator
#
# length=100000设全局读取上限；def函数默认batch_size=1000，读取全局iter_dataset/length。yield使调用返回惰性生成器，每批为list[str]；两个_是普通循环变量。range(0,length,batch_size)按批数计tqdm，内部next在不满批时会耗尽并触发RuntimeError。不能整除时仍读满末批，可能超读；本次执行原函数体并用小上下文实测，不消费10万条。

# %% 原单元 7
# length与batch_iterator；原执行语句保留，详细参数/shape/边界见本格说明。
# 原训练条数上限，不是vocab大小，第二轮不自动重置迭代器。
length = 100000
# 默认每批1000，读取全局length和iter_dataset。
def batch_iterator(batch_size=1000):
#     for _ in tqdm(range(0, len(dataset), batch_size)):
    # 外层按批数计进度；不能整除时原函数仍请求满批。
    for _ in tqdm(range(0, length, batch_size)):
        # 逐next读取content字符串，耗尽会在生成器中触发RuntimeError。
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]

# %% [markdown]
# ## 单元 8（code）：训练新BPE tokenizer
#
# fast对象train_new_from_iterator接收batch_iterator()、vocab_size=12500和initial_alphabet=base_vocab。返回新对象new_tokenizer，不修改旧tokenizer；清空旧vocab/merges并保留处理/特殊token配置。目标含alphabet/特殊token，语料不足可能达不到12500；原格未原配置运行，补充仅改vocab320并调用真实原生成器（默认批大小仍1000）。

# %% 原单元 8
# 训练新BPE tokenizer；原执行语句保留，详细参数/shape/边界见本格说明。
# 比较耗时
# 真实训练返回新对象；旧GPT2的vocab/merges不就地更新。
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(),
                                                  # 目标词表大小含alphabet/特殊token，语料不足可小于目标。
                                                  vocab_size=12500,
                                                  # 确保byte覆盖，不是继承旧50257项vocab。
                                                  initial_alphabet=base_vocab)

# %% [markdown]
# ## 单元 9（code）：展示新对象
#
# 读取new_tokenizer repr，显示的是训练后的词表/配置，不涉及模型logits。

# %% 原单元 9
# 展示新对象；原执行语句保留，详细参数/shape/边界见本格说明。
# 展示当前新对象，不进行第二次训练。
new_tokenizer

# %% [markdown]
# ## 单元 10（markdown）：新词表检查标题
#
# 检查实际大小、特殊token和合并规则，不仅看目标vocab_size。

# %% [markdown] 原单元 10
# ### new tokenizer

# %% [markdown]
# ## 单元 11（code）：按整数ID升序排序
#
# vocab.items是(token,ID)，sorted key=lambda x:x[1]升序、reverse=False；返回新list tokens，不改变tokenizer词表。

# %% 原单元 11
# 按整数ID升序排序；原执行语句保留，详细参数/shape/边界见本格说明。
# items为(token,ID)；lambda指定排序键，返回新列表。
tokens = sorted(new_tokenizer.vocab.items(), key=lambda x: x[1], reverse=False)

# %% [markdown]
# ## 单元 12（code）：观察alphabet前10项
#
# 切片[:10]不含右边界，读取映射字典插入顺序的前10显示符号，非最常见的10个token。

# %% 原单元 12
# 观察alphabet前10项；原执行语句保留，详细参数/shape/边界见本格说明。
# Python切片左闭右开，映射顺序不等于byte键或词频排名。
base_vocab[:10]

# %% [markdown]
# ## 单元 13（code）：第257–279项显示转换
#
# tokens[257:280]最多23个二元组，for t,_解包忽略ID，转换单token为显示文本并返回list[(原token,显示文本)]。257只是原代码切分假设，应查实际基础/特殊token排布；单token可能不是完整UTF8字符。

# %% 原单元 13
# 第257–279项显示转换；原执行语句保留，详细参数/shape/边界见本格说明。
# 257:280
# 解包token/ID，再查看单token展示文本。
[(t, new_tokenizer.convert_tokens_to_string([t])) for t, _ in tokens[257:280]]

# %% [markdown]
# ## 单元 14（code）：词表尾15项
#
# [-15:]从倒数第15开始，按前面ID升序列表取尾部；不是按频率排序。转换只是可视化，不撤销词表训练。

# %% 原单元 14
# 词表尾15项；原执行语句保留，详细参数/shape/边界见本格说明。
# last 15
# 解包token/ID，再查看单token展示文本。
[(t, new_tokenizer.convert_tokens_to_string([t])) for t, _ in tokens[-15:]]

# %% [markdown]
# ## 单元 15（markdown）：Python关键字检查标题
#
# 检查裸字符串是否词表项，只是覆盖统计，不是代码生成质量。

# %% [markdown] 原单元 15
# ### python keyword

# %% [markdown]
# ## 单元 16（code）：导入标准库keyword
#
# keyword.kwlist是当前Python版本硬关键字列表。软关键字另有softkwlist，不能硬编码某版数量或混淆两类。

# %% 原单元 16
# 导入标准库keyword；原执行语句保留，详细参数/shape/边界见本格说明。
# python 标准库
# 标准库关键字列表由当前Python版本决定。
import keyword

# %% [markdown]
# ## 单元 17（code）：关键字数量
#
# len(keyword.kwlist)返回int；本次Python3.12有35项，此数量依版本可能变，不是目标词表大小。

# %% 原单元 17
# 关键字数量；原执行语句保留，详细参数/shape/边界见本格说明。
# 计数硬关键字，非新词表项数。
len(keyword.kwlist)

# %% [markdown]
# ## 单元 18（code）：检查裸关键字成员关系
#
# for遍历每个kw字符串，kw not in new_tokenizer.vocab检查字典键，不存在时print。带空格的Ġreturn和裸return是不同token；缺失不意味着模型不能用多项表示，也不证明生成能力。

# %% 原单元 18
# 检查裸关键字成员关系；原执行语句保留，详细参数/shape/边界见本格说明。
# 逐个检查裸关键字，而非代码上下文中token长度。
for kw in keyword.kwlist:
    # 字典键成员测试；缺失并不等于无法多token表示。
    if kw not in new_tokenizer.vocab:
        # 输出当前数值/字符串，不更新词表；返回None。
        print(f'`{kw}` not in the new tokenizer')

# %% [markdown]
# ## 单元 19（markdown）：第二轮训练标题
#
# retrain这里是再次从旧tokenizer模板训练新的词表，不是继续更新第8格的新merges。

# %% [markdown] 原单元 19
# ### retrain

# %% [markdown]
# ## 单元 20（code）：第二轮重定义函数并继续消费
#
# length=100000*2是200000；同名def覆盖旧函数，依旧共享iter_dataset，没有iter(dataset)复位。train_new_from_iterator在旧tokenizer上调用且vocab32768=2^15，同时改变样本范围/规模与预算，不能归因单一变量。本次AST审计的小语料补充仅改length和vocab，仍真实执行原生成器默认1000，证明继续消费。

# %% 原单元 20
# 第二轮重定义函数并继续消费；原执行语句保留，详细参数/shape/边界见本格说明。
# 原训练条数上限，不是vocab大小，第二轮不自动重置迭代器。
length = 100000*2
# 默认每批1000，读取全局length和iter_dataset。
def batch_iterator(batch_size=1000):
    # 外层按批数计进度；不能整除时原函数仍请求满批。
    for _ in tqdm(range(0, length, batch_size)):
        # 逐next读取content字符串，耗尽会在生成器中触发RuntimeError。
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]
# 真实训练返回新对象；旧GPT2的vocab/merges不就地更新。
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(),
                                                  # 目标词表大小含alphabet/特殊token，语料不足可小于目标。
                                                  vocab_size=32768,
                                                  # 确保byte覆盖，不是继承旧50257项vocab。
                                                  initial_alphabet=base_vocab)

# %% [markdown]
# ## 单元 21（code）：再次检查关键字
#
# 对第二轮新对象重复裸词表成员检查；与第一轮相同API，打印差异不等于下游指标提升。

# %% 原单元 21
# 再次检查关键字；原执行语句保留，详细参数/shape/边界见本格说明。
# 逐个检查裸关键字，而非代码上下文中token长度。
for kw in keyword.kwlist:
    # 字典键成员测试；缺失并不等于无法多token表示。
    if kw not in new_tokenizer.vocab:
        # 输出当前数值/字符串，不更新词表；返回None。
        print(f'`{kw}` not in the new tokenizer')

# %% [markdown]
# ## 单元 22（markdown）：Hub发布标题
#
# 保存本地与推送远程是两种操作；本讲验证采用临时目录save_pretrained/重载。

# %% [markdown] 原单元 22
# ### to the hub

# %% [markdown]
# ## 单元 23（code）：创建仓库提示为注释
#
# 两条#行不会导入huggingface_hub或create_repo。实际创建需账号/权限，原notebook并未在此格执行。用户授权的是公开学习GitHub仓库，不是讲师Hugging Face组织。

# %% 原单元 23
# 创建仓库提示为注释；原执行语句保留，详细参数/shape/边界见本格说明。
# from huggingface_hub import create_repo
# create_repo('codeparrot')

# %% [markdown]
# ## 单元 24（code）：原代理环境写入
#
# os.environ映射赋值HTTP/HTTPS代理到127.0.0.1:7890，会改变本进程后续请求。必须有该服务才有效；本次明确跳过，不修改真实代理。

# %% 原单元 24
# 原代理环境写入；原执行语句保留，详细参数/shape/边界见本格说明。
# 标准库环境接口；下面原代理写入本次跳过。
import os
# 设置当前进程代理；读者不能假定本机有7890服务。
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# 设置当前进程代理；读者不能假定本机有7890服务。
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# %% [markdown]
# ## 单元 25（code）：原讲师组织Hub上传
#
# ckpt/org为字符串，new_tokenizer.push_to_hub会向远程写文件，organization为历史API用法。本次不执行、不模拟上传返回值；源码保留，验证补充只做本地save_pretrained/AutoTokenizer重载。

# %% 原单元 25
# 原讲师组织Hub上传；原执行语句保留，详细参数/shape/边界见本格说明。
# 原远程仓库名称，非本地保存目录。
ckpt = 'codeparrot'
# 原讲师组织名；本次不向此组织写入。
org = 'lanchunhui'
# 外部上传副作用，原语句保留但明确不执行。
new_tokenizer.push_to_hub(ckpt, organization=org)

# %% [markdown]
# ## 单元 26（code）：空代码格
#
# 原空格保留，注释版AST空。

# %% 原单元 26
# 空代码格；原执行语句保留，详细参数/shape/边界见本格说明。
