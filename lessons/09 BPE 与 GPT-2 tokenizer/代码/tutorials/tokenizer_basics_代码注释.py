# -*- coding: utf-8 -*-
# 第09讲Python阅读版：原语句说明及历史错误边界见逐格内容。
# 主要用于阅读；远程语料、代理、Hub上传未在本次验证执行。

# %% [markdown]
# ## 单元 0（code）：图片显示接口
#
# 从IPython.display导入Image类；导入不读图，单元17才使用。返回显示对象并由Jupyter渲染，不负责tokenizer训练。

# %% 原单元 0
# 图片显示接口；原执行语句保留，详细参数/shape/边界见本格说明。
# 导入图片显示类，实际读图在Image调用。
from IPython.display import Image

# %% [markdown]
# ## 单元 1（markdown）：基础概念入口
#
# 先区分字符、UTF8 bytes、token字符串和整数ID，再讨论词表训练。Markdown不执行代码。

# %% [markdown] 原单元 1
# ## basics

# %% [markdown]
# ## 单元 2（markdown）：常见词与padding设置代码块
#
# 常见词反复切分可能增加token数量，但词表预算/频率/算法都影响边界，不能保证每个常见词单token。Markdown代码块依次设置对象属性：padding_side先left后right，最终为right；pad_token=eos_token只是把已有EOS复用为填充标记，不新增embedding。attention_mask与loss屏蔽仍须正确处理。原“Fix weird overflow”记录了作者对特定运行问题的处理。本块需要预先存在tokenizer，不是完整可运行程序；已加逐行注释而保留原赋值。
#
# 原Markdown中的设置代码（未作为Python执行格）：
#
# - 常见词最好不要再切分
#     - 否则会显著地增加 token 的长度
# - 设置
#
# ```
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
# ```

# %% [markdown] 原单元 2
# - 常见词最好不要再切分
#     - 否则会显著地增加 token 的长度
# - 设置
#
# ```
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
# ```

# %% [markdown]
# ## 单元 3（code）：AutoTokenizer类
#
# transformers.AutoTokenizer根据checkpoint配置选择实现；from_pretrained负责文件加载，use_fast默认倾向可用的fast类。导入不加载权重。fast的backend_tokenizer/tokens()接口后续会使用，slow对象不一定支持。

# %% 原单元 3
# AutoTokenizer类；原执行语句保留，详细参数/shape/边界见本格说明。
# 根据checkpoint选择具体tokenizer类，导入不加载语言模型权重。
from transformers import AutoTokenizer

# %% [markdown]
# ## 单元 4（code）：封装逐ID解码函数
#
# def定义普通函数，tokenizer参数为可调用对象、text为str。调用tokenizer(text,add_special_tokens=False)返回BatchEncoding，取input_ids为list[int]；列表推导式对每个ID调用decode返回list[str]。byte-level token可能只含部分UTF8字符，单ID解码可能是�；整体decode与convert_ids_to_tokens用途不同。

# %% 原单元 4
# 封装逐ID解码函数；原执行语句保留，详细参数/shape/边界见本格说明。
# 普通函数接收tokenizer对象和文本，返回逐ID解码的list[str]。
def tokenize_str(tokenizer, text):
    # 编码后取list[int]；add_special_tokens=False避免自动添加模型标记。
    input_ids = tokenizer(text, add_special_tokens=False)['input_ids']
    # 逐ID可能含不完整UTF8字符，不能替代整体decode。
    return [tokenizer.decode(token_id) for token_id in input_ids]

# %% [markdown]
# ## 单元 5（code）：加载T5 tokenizer作比较
#
# AutoTokenizer.from_pretrained(t5-base)返回T5 tokenizer，可能联网/缓存，算法不是GPT2 byte-level BPE。本次独立补充只把路径字符串换成固定本地T5资产并设local_files_only；未下载T5模型权重。原调用和历史输出保留。

# %% 原单元 5
# 加载T5 tokenizer作比较；原执行语句保留，详细参数/shape/边界见本格说明。
# 加载 T5 分词器，按路径从本地缓存或 Hub 读取分词资产。配套例子使用固定本地资产。
tokenizer_t5 = AutoTokenizer.from_pretrained('t5-base')

# %% [markdown]
# ## 单元 6（code）：普通词的T5切分
#
# 先print(tokenize_str(...hello))，再展示sex的切分。返回是list[str]，print返回None；函数行为沿用单元4。切得更碎不能单独证明训练语料过滤。示例仅用于分词观察，不是模型语义输出。

# %% 原单元 6
# 普通词的T5切分；原执行语句保留，详细参数/shape/边界见本格说明。
# 输出当前数值/字符串，不更新词表；返回None。
print(tokenize_str(tokenizer_t5, 'hello'))
# t5 tokenizer for 'sex'
# 调用单ID解码函数，返回分段观察值。
tokenize_str(tokenizer_t5, 'sex')

# %% [markdown]
# ## 单元 7（markdown）：Python代码分词标题
#
# Python换行、缩进、标点都可能有语义，要整体往返而非先清空白。

# %% [markdown] 原单元 7
# ## tokenize python code

# %% [markdown]
# ## 单元 8（markdown）：BPE基础符号与合并
#
# 原文说明从256个byte值开始，反复合并当前高频相邻pair。143859是原笔记提及的Unicode数量，没有在本次确认其统计版本，也不是GPT2词表大小。BPE学习统计/merges，不训练神经网络权重；新pair可含已经合并的片段。

# %% [markdown] 原单元 8
# - 256 ~ 143,859
#     - BPE: from 256 byte values;
#     - iteratively merging the most frequently co-occurring pair of tokens in the vocabulary

# %% [markdown]
# ## 单元 9（code）：多行原始字符串样本
#
# r三引号定义一个str，保留反斜杠、换行和缩进；里面def/print是字符串内容，不是在此格定义/调用say_hello。末尾print仅打印这个字符串。四空格空行也保留，不能在阅读版自动strip。Python阅读版用相邻字符串字面量表示同一个值，AST字符串常量核对相等。
#
# ```python
# python_code = r'''def say_hello():
#     print('Hello, World!')
#
# # print hello
# say_hello()
# '''
# print(python_code)
# ```

# %% 原单元 9
# 多行原始字符串样本；原执行语句保留，详细参数/shape/边界见本格说明。
# 原三引号str改为相邻字面量阅读排版；值/空白/换行与AST不变。
# 多行原始字符串保留代码文本、换行和缩进，不执行其内部Python。
python_code = (
    'def say_hello():\n'
    "    print('Hello, World!')\n"
    '    \n'
    '# print hello\n'
    'say_hello()\n'
)
# 输出当前数值/字符串，不更新词表；返回None。
print(python_code)

# %% [markdown]
# ## 单元 10（code）：加载GPT2并查看IDs和tokens
#
# 加载 fast tokenizer 后，对 python_code 编码，input_ids 返回整数列表，tokens() 返回对应的 token 字符串列表。示例再次调用编码以观察返回对象；配套运行的加载路径见复现说明。

# %% 原单元 10
# 加载GPT2并查看IDs和tokens；原执行语句保留，详细参数/shape/边界见本格说明。
# 原远程GPT2路径；验证补充明确只改为固定本地资产。
tokenizer = AutoTokenizer.from_pretrained('gpt2')
# 输出当前数值/字符串，不更新词表；返回None。
print(tokenizer(python_code)['input_ids'])
# 输出当前数值/字符串，不更新词表；返回None。
print(tokenizer(python_code).tokens())

# %% [markdown]
# ## 单元 11（code）：检查normalizer
#
# backend_tokenizer是Rust后端对象，normalizer为属性，可是None。GPT2没有BERT式先小写/去重音步骤；None不表示没有预切分/BPE。此格不改变状态。

# %% 原单元 11
# 检查normalizer；原执行语句保留，详细参数/shape/边界见本格说明。
# none
# 读取后端normalizer属性；GPT2这里为None。
tokenizer.backend_tokenizer.normalizer

# %% [markdown]
# ## 单元 12（code）：预切分与offset
#
# pre_tokenize_str 接收字符串，返回 [(片段,(start,end))]。这一步先划分可交给子词算法处理的片段，offset 标记其在原文中的位置；下一步再按 BPE 合并规则处理。

# %% 原单元 12
# 预切分与offset；原执行语句保留，详细参数/shape/边界见本格说明。
# pretokenization
# offset tracking
# 真实预切分返回片段及offset，不等于最终BPE token。
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code)

# %% [markdown]
# ## 单元 13（code）：一个byte的整数值
#
# ord接受单字符str或长度1的bytes；b\xe2是一个byte，结果226。它可能是欧元UTF8的首byte，不是欧元Unicode码点8364。

# %% 原单元 13
# 一个byte的整数值；原执行语句保留，详细参数/shape/边界见本格说明。
# 单byte到整数；不等于完整Unicode字符码点。
ord(b'\xe2')

# %% [markdown]
# ## 单元 14（code）：Unicode与UTF8逐字节
#
# u前缀在Python3与普通str一样；a和€是字符。a.encode为一个byte，ord可读97；€.encode返回三个bytes[226,130,172]，原注释4 bytes不准确。迭代bytes时i已是int，chr后ord又回到同一整数，等价list(e.encode)。两条f字符串展示对象，不修改字符内容。

# %% 原单元 14
# Unicode与UTF8逐字节；原执行语句保留，详细参数/shape/边界见本格说明。
# Unicode character composed of 1-4 bytes

# 两个Unicode字符串；欧元UTF8有3个bytes。
a, e = u'a', u'€'

# 1 bytes
# 长度1的bytes可传ord，a的值为97。
byte = ord(a.encode('utf-8'))
# 输出当前数值/字符串，不更新词表；返回None。
print(f"{a}, {a.encode('utf-8')}, {byte}")

# 4 bytes
# byte = ord(e.encode('utf-8'))
# ord 接受的是一个char
# 迭代bytes已有int，chr再ord等价取原byte值。
byte = [ord(chr(i)) for i in e.encode('utf-8')]
# 输出当前数值/字符串，不更新词表；返回None。
print(f"{e}, {e.encode('utf-8')}, {byte}")

# %% [markdown]
# ## 单元 15（markdown）：训练tokenizer标题
#
# 这一节开始阅读文本预处理、子词学习和编码的完整流水线。

# %% [markdown] 原单元 15
# ## training a tokenizer

# %% [markdown]
# ## 单元 16（markdown）：处理流水线与三种子词算法
#
# normalization规范文本，pretokenization先划片段，tokenizer model应用子词规则/词表ID，postprocessing补模型标记。BPE逐步添加合并，WordPiece使用不同训练规则，Unigram可迭代删减候选；它们不是同一种算法。原拼写postprocesssing指后处理。

# %% [markdown] 原单元 16
# - 不涉及权重或者反向传播
# - tokenizer 的 processing pipeline
#     - normalization
#     - pretokenization
#     - tokenizer model
#     - postprocesssing
# - subword tokenization algorithms (subword: tokens are part of words)
#     - BPE: byte pair encoding
#         - 迭代式地添加策略，直到一个 target vocabulary size
#     - word piece
#     - unigram
#         - 迭代式地删除策略，直到一个 target vocabulary size

# %% [markdown]
# ## 单元 17（code）：token_steps原图
#
# Image 读取 ../imgs/token_steps.png，width=600 指定展示宽度。图中沿文字→规范化→预切分→子词→编号/特殊标记展开；小写、去重音及 CLS/SEP 展示了类似 BERT 的处理配置，具体步骤随 tokenizer 设置变化。

# %% 原单元 17
# token_steps原图；原执行语句保留，详细参数/shape/边界见本格说明。
# 相对当前cwd读取配图，width只是展示像素。
Image('../imgs/token_steps.png', width=600)

# %% [markdown]
# ## 单元 18（markdown）：BERT示例入口
#
# 下方是WordPiece示意，不属于GPT2 BPE完整实现；不要把不同模型的特殊标记混为一谈。

# %% [markdown] 原单元 18
# ### BERT

# %% [markdown]
# ## 单元 19（markdown）：官方搭建tokenizer示例链接
#
# 原 Colab 链接指向 Hugging Face 课程 notebook，可对照完整的分词处理与训练流水线。

# %% [markdown] 原单元 19
# - https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/building_tokenizer.ipynb

# %% [markdown]
# ## 单元 20（code）：导入数据加载函数
#
# 导入 datasets.load_dataset。后续调用时，path/name/split/streaming 决定读取的数据与返回的 Dataset 或 IterableDataset。

# %% 原单元 20
# 导入数据加载函数；原执行语句保留，详细参数/shape/边界见本格说明。
# 真实datasets接口；导入与下载/读取语料分开。
from datasets import load_dataset

# %% [markdown]
# ## 单元 21（code）：Wikitext与文本批生成器
#
# load_dataset(wikitext,name=wikitext-2-raw-v1,split=train)涉及远程语料。本次原数据调用未执行；另在小型真实Dataset上下文执行原get_training_corpus定义。len(dataset)为总行数，range步长1000，dataset[i:i+1000]切片返回dict[list]，取text为list[str]；yield按需交出一批，末批可不足1000。这与train notebook中的逐next流式函数不同。

# %% 原单元 21
# Wikitext与文本批生成器；原执行语句保留，详细参数/shape/边界见本格说明。
# name: subset
# 原语料入口依赖路径/远程数据；本次未以原配置运行。
dataset = load_dataset('wikitext', name='wikitext-2-raw-v1', split='train')
# 生成器返回Dataset的text批，切片末批可不足1000。
def get_training_corpus(dataset):
    # 按1000行步长遍历，len用于非流式Dataset。
    for i in range(0, len(dataset), 1000):
        # Dataset切片是列字典，取text返回list[str]。
        yield dataset[i:i+1000]['text']

# %% [markdown]
# ## 单元 22（code）：保留原悬空逗号语法错误
#
# 原第一行的末尾逗号导致普通 Python 编译报 SyntaxError，第二行随之无法执行。去掉末尾逗号后，Tokenizer(models.WordPiece(unk_token=[UNK])) 可创建 WordPiece 模型对象；完整训练还需配置预处理器、trainer 和保存步骤。阅读版把原错误格列作注释，修正示例见配套说明。

# %% 原单元 22
# 保留原悬空逗号语法错误；原执行语句保留，详细参数/shape/边界见本格说明。
# 原单元22无法解析，以下整格作为原文注释保留；独立修正版见复现。
# from tokenizers import Tokenizer, models,
# tokenizer = Tokenizer(models.WordPiece(unk_token='[UNK]'))

# %% [markdown]
# ## 单元 23（markdown）：GPT2映射入口
#
# 下面恢复GPT2接口；上面的BERT/WordPiece片段与它分开。

# %% [markdown] 原单元 23
# ### gpt2

# %% [markdown]
# ## 单元 24（markdown）：byte到显示字符表
#
# 表中byte13→č、32→Ġ、10→Ċ；这些字符是映射符号，不是原文真的输入它们。非断行空格U+00A0实际UTF8是194/160两bytes，不能把单个160当完整字符。byte映射256种，而完整GPT2词表50257项。

# %% [markdown] 原单元 24
# -
#
# | Description                                        | Character   | Bytes     | Mapped bytes |
# |----------------------------------------------------|-------------|-----------|--------------|
# | Regular characters                                 | `a` and `?` | 97 and 63 | `a` and `?`  |
# | A nonprintable control character (carriage return) | `U+000D`    | 13        | `č`          |
# | A space                                            | ` `         | 32        | **`Ġ`**      |
# | A nonbreakable space                               | `\xa0`      | 160       | `ł`          |
# | A newline character                                | `\n`        | 10        | **`Ċ`**      |

# %% [markdown]
# ## 单元 25（code）：导入bytes_to_unicode
#
# 函数所属transformers.models.gpt2.tokenization_gpt2，返回dict[int,str]，从256种byte值到256个唯一可打印显示字符；不是Unicode语义翻译。

# %% 原单元 25
# 导入bytes_to_unicode；原执行语句保留，详细参数/shape/边界见本格说明。
# 导入GPT2的256-byte可逆显示映射函数。
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# %% [markdown]
# ## 单元 26（code）：建立正向byte映射
#
# 调用无参函数并绑定字典bytes_to_unicode_map，末行展示全表。映射保持可逆；此格不是将字符投到embedding。

# %% 原单元 26
# 建立正向byte映射；原执行语句保留，详细参数/shape/边界见本格说明。
# 256项byte→唯一显示字符字典，不是token ID词表。
bytes_to_unicode_map = bytes_to_unicode()
# 展示映射对象，不更改字典。
bytes_to_unicode_map

# %% [markdown]
# ## 单元 27（code）：构造反向映射
#
# .items() 遍历 (byte整数,显示字符)，生成器把 (v,k) 交换，再由 dict 收集成 dict[str,int]，用于将显示字符映射回原始 byte。

# %% 原单元 27
# 构造反向映射；原执行语句保留，详细参数/shape/边界见本格说明。
# string to id
# 生成器交换(key,value)，得到显示字符→byte反向字典。
unicode_to_bytes_map = dict((v, k) for k, v in bytes_to_unicode_map.items())
#
# 展示反向byte映射对象。
unicode_to_bytes_map

# %% [markdown]
# ## 单元 28（code）：映射长度
#
# len(dict)返回Python整数256，统计字典项，不等于文本字符数或GPT2完整vocab_size。

# %% 原单元 28
# 映射长度；原执行语句保留，详细参数/shape/边界见本格说明。
# 返回基础映射项数256。
len(bytes_to_unicode_map)

# %% [markdown]
# ## 单元 29（code）：基础alphabet的顺序
#
# keys()给显示字符，list收集256项并保留插入顺序；base_vocab[0]/[-1]取首尾。列表下标不是原byte值；base_vocab[10]不可当byte10查询。print输出显示符号。

# %% 原单元 29
# 基础alphabet的顺序；原执行语句保留，详细参数/shape/边界见本格说明。
# list保留映射键顺序，基础alphabet含256显示字符。
base_vocab = list(unicode_to_bytes_map.keys())
# 输出当前数值/字符串，不更新词表；返回None。
print(base_vocab[0], base_vocab[-1])

# %% [markdown]
# ## 单元 30（code）：查询换行byte的显示符号
#
# 用字典键10读取Ċ；与原文换行字符串\n/Unicode码点分别区分。

# %% 原单元 30
# 查询换行byte的显示符号；原执行语句保留，详细参数/shape/边界见本格说明。
# 输出当前数值/字符串，不更新词表；返回None。
print(bytes_to_unicode_map[10])

# %% [markdown]
# ## 单元 31（code）：再次加载并预切分
#
# 重新绑定tokenizer为GPT2，调用pre_tokenize_str(python_code)。原格不会学习新merges；只读已有规则。本次加载路径补充固定本地资产，保留其他原语句；读取与单元12结果核对一致。

# %% 原单元 31
# 再次加载并预切分；原执行语句保留，详细参数/shape/边界见本格说明。
# 原远程GPT2路径；验证补充明确只改为固定本地资产。
tokenizer = AutoTokenizer.from_pretrained('gpt2')
# 真实预切分返回片段及offset，不等于最终BPE token。
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code)

# %% [markdown]
# ## 单元 32（markdown）：空白保留规则
#
# 原英文：连续空格保留且可能作为一个预片段，一个词前的空格可附着到后面的词。ĊĠĠĠ表示换行和三个空格，不是真实拉丁字符输入。预切分片段不等于最终token，不能保证四空格缩进仅占一项。

# %% [markdown] 原单元 32
# - Spaces, and in particular consecutive spaces, are conserved (for instance, the three spaces in 'ĊĠĠĠ').
# - Consecutive spaces are considered as a single word.
# - Each space preceding a word is attached to and considered a part of the subse‐quent word (e.g., in 'Ġsay').

# %% [markdown]
# ## 单元 33（code）：展示tokenizer对象
#
# 裸变量表达式展示类型、词表规模等repr，不生成文本、不改变词表。

# %% 原单元 33
# 展示tokenizer对象；原执行语句保留，详细参数/shape/边界见本格说明。
# 展示已有对象，不生成语言模型回答。
tokenizer

# %% [markdown]
# ## 单元 34（markdown）：词表训练标题
#
# 本notebook后续主要看现有词表；完整train_new_from_iterator在另一个train notebook。

# %% [markdown] 原单元 34
# ### train a tokenizer

# %% [markdown]
# ## 单元 35（markdown）：endoftext特殊token
#
# GPT2词表末尾特殊token<|endoftext|>，原资产ID50256。重新训练后的ID可能改变；EOS、PAD、BOS的使用要看配置，不能类比BERT CLS/SEP。

# %% [markdown] 原单元 35
# - `<|endoftext|>`, is the special token used to specify the end of a text sequence and was added after the BPE vocabulary was built

# %% [markdown]
# ## 单元 36（code）：按token表示长度降序
#
# tokenizer.vocab是dict[str,int]，items给二元组；sorted返回新list，lambda x:len(x[0])按表示字符串长度，reverse=True降序。[:10]左闭右开；推导式逐token转展示文本。长度不是语料词频，也不是token原文byte数；可能含不完整UTF8片段。 原sort没有同长度的第二排序键，词表字典遍历次序可不同；前10项若跨并列边界，两次列表可能不同。比较两次前10项时，先看长度与转换结果，再检查并列项的排列。

# %% 原单元 36
# 按token表示长度降序；原执行语句保留，详细参数/shape/边界见本格说明。
# items为(token,ID)；lambda指定排序键，返回新列表。
tokens = sorted(tokenizer.vocab.items(), key=lambda x: len(x[0]), reverse=True)
# 只转换所取token的表示，不改变词表或模型。
[tokenizer.convert_tokens_to_string([token]) for token, _ in tokens[:10]]

# %% [markdown]
# ## 单元 37（code）：按ID降序
#
# lambda x:x[1]按ID，reverse=True从大到小；前12项不自动等于最常见词。convert_tokens_to_string接收list[str]，拼接/反byte映射，不训练模型。

# %% 原单元 37
# 按ID降序；原执行语句保留，详细参数/shape/边界见本格说明。
# items为(token,ID)；lambda指定排序键，返回新列表。
tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1], reverse=True)
# 只转换所取token的表示，不改变词表或模型。
[tokenizer.convert_tokens_to_string([token]) for token, _ in tokens[:12]]

# %% [markdown]
# ## 单元 38（code）：空代码格
#
# 原source空，保留编号与元数据；注释版只加入说明性注释，AST仍为空。

# %% 原单元 38
# 空代码格；原执行语句保留，详细参数/shape/边界见本格说明。
