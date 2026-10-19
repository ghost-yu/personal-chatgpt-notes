# -*- coding: utf-8 -*-
# 原课程Python注释阅读版；完整词表扫描会捕获并打印Rust洞位panic，请先读复现说明。

# %% 原代码单元 2
# tiktoken提供真实字节BPE，langdetect是统计语言识别，regex支持Unicode属性和更丰富正则。import不加载GPT-4o权重，不调用API。
# 字节BPE库；后续get_encoding加载词表，不加载语言模型。
import tiktoken
# 短文本语言分类工具，固定seed仍不保证判断正确。
import langdetect
# Unicode正则库，不是Python内置re。
import regex as re

# %% 原代码单元 3
# get_encoding按编码名读取注册构造器，首次可能下载词表并校验SHA，返回Encoding对象，后续缓存复用。本次预置真实完整词表缓存，原调用不改且禁止联网验证。
# https://platform.openai.com/tokenizer
# 绑定cl100k_base Encoding实例，后续按此词表解释ID。
T1 = tiktoken.get_encoding('cl100k_base')
# 绑定o200k_base实例；两套ID不可跨模型随意混用。
T2 = tiktoken.get_encoding('o200k_base')

# %% 原代码单元 4
# cl100k 将这句英文编码为5个 token ID，各 ID 指向词表中的相应条目。
# 确定性字符串→list[int]，不是模型forward。
T1.encode('i love chatgpt')

# %% 原代码单元 5
# o200k同句仍是5个ID但数值不同，直接反例说明词表更大不保证每句token严格更少。
# 使用新词表分词，len不保证对所有文本都更短。
T2.encode('i love chatgpt')

# %% 原代码单元 6
# n_vocab=max_token_value+1，分别100277与200019，不等于所有中间ID都有效。cl普通100256+特殊5共100261，o普通199998+特殊2共200000，其余是洞位。
# 读取max ID+1，含无效洞位，不能当有效ID数量。
T1.n_vocab, T2.n_vocab

# %% 原代码单元 9
# cl对指定中文完整句编码后len=34，len数列表元素，不是字数、字节数或张量长度。
# 数cl编码的token个数，不数字符或UTF8字节。
len(T1.encode('你好，我的名字是GPT-4o。我是一种新型的语言模型，很高兴见到你!'))

# %% 原代码单元 10
# o对同一句len=24，减少10个即29.41%，旧/新=1.4167；不同文本不保证同比例。
# 数o编码个数，同一句34→24。
len(T2.encode('你好，我的名字是GPT-4o。我是一种新型的语言模型，很高兴见到你!'))

# %% 原代码单元 11
# with 保证结束后关闭文件，encoding 指定 UTF-8。readlines 先读入全部文本，循环再逐行编码累加。./xyj.txt 是配套原文；Python 文本模式会规范化读取到的换行。
# 旧编码计数清零。
token_ids_len1 = 0
# 新编码计数清零。
token_ids_len2 = 0
# 上下文管理器负责关闭文件；原readlines一次加载全部行。
with open('./xyj.txt', 'r', encoding='utf-8') as f:
    # 按Python文本换行规范化后的行编码，不对正文strip。
    for line in f.readlines():
        # 累加这一行的cl token数，保留行末换行。
        token_ids_len1 += len(T1.encode(line))
        # 累加同一行的o token数，确保比较输入一致。
        token_ids_len2 += len(T2.encode(line))

# %% 原代码单元 12
# 两个整数及比率：cl990508/o703303，旧/新1.4083659532，新/旧0.7100427256，减少约29%。这是逐行全文统计，不是任意中文或整篇一次编码的定理。
# 展示两个计数与旧/新、新/旧比率，降低比例为1−新/旧。
token_ids_len1, token_ids_len2, token_ids_len1/token_ids_len2, token_ids_len2/token_ids_len1

# %% 原代码单元 15
# 原词组在o200k编码为单个177431。它属于普通mergeable token，不是special_tokens里的保留控制符；原课特殊中文tokens是口语描述。
# 使用新词表分词，len不保证对所有文本都更短。
T2.encode("给主人留下些什么吧")

# %% 原代码单元 16
# 同词组在cl100k拆成12个ID；比较必须连同完整解码检查，单token可能只是UTF8字符的一部分。
# 确定性字符串→list[int]，不是模型forward。
T1.encode("给主人留下些什么吧")

# %% 原代码单元 17
# 扫描o200k所有range(n_vocab)位置，包括19个无效洞位；decode默认替换坏UTF8，len数Python字符。bare except吞掉Rust PanicException甚至中断；按长度降序稳定排序后langdetect筛zh-cn，实际只打印20条，不是注释写的100。语言检测对短串不可靠且有随机性，本次外置seed26；保留原算法/历史panic输出。
# 建立ID→单token解码后Python字符数的字典。
length_dict = {}

# 上界max ID+1包含洞位；原代码会触发Rust panic后被bare except吞掉。
for i in range(T2.n_vocab):
    try:
        # decode默认errors=replace，部分UTF8字节可能显示替换字符。
        length_dict[i] = len(T2.decode([i]))
    # 裸捕获还会吞KeyboardInterrupt等；保留原码，安全版本独立提供。
    except:
        # 忽略当前错误，不会修复洞位或记录原因。
        pass

# Sort by length
# sorted按负长度升序即按长度降序，同长度保留原ID顺序。
length_dict = dict(sorted(length_dict.items(), key=lambda item: -item[1]))
# print(length_dict)
# Print the top 100 chinese wordsÅ
# 计已打印项目，不是已扫描项目数。
tot = 0
# 迭代排序字典的key，即token ID，不是(id,length)元组。
for item in length_dict:
    try:
        # 返回zh-cn只是启发式标签，短串和标点均可能误判。
        if langdetect.detect(T2.decode([item])) == "zh-cn":
            # 打印原ID与单token解码文本，文本可含换行。
            print(item, T2.decode([item]))
            # 仅命中并打印后加1，跳过的条目不计。
            tot += 1
    # 裸捕获还会吞KeyboardInterrupt等；保留原码，安全版本独立提供。
    except:
        # 忽略当前错误，不会修复洞位或记录原因。
        pass
    # 原程序实际只取20条，与top100注释不一致。
    if tot == 20:
        # 退出当前for，不退出整个notebook。
        break

# %% 原代码单元 18
# 同流程扫描cl100k，16个无效洞位。标点/换行也可能被langdetect判zh-cn，不能把输出理解成真实中文词频榜或训练语料分布。实际执行完整100277位置。
# 建立ID→单token解码后Python字符数的字典。
length_dict = {}

# 上界max ID+1包含洞位；原代码会触发Rust panic后被bare except吞掉。
for i in range(T1.n_vocab):
    try:
        # decode默认errors=replace，部分UTF8字节可能显示替换字符。
        length_dict[i] = len(T1.decode([i]))
    # 裸捕获还会吞KeyboardInterrupt等；保留原码，安全版本独立提供。
    except:
        # 忽略当前错误，不会修复洞位或记录原因。
        pass

# Sort by length
# sorted按负长度升序即按长度降序，同长度保留原ID顺序。
length_dict = dict(sorted(length_dict.items(), key=lambda item: -item[1]))
# print(length_dict)
# Print the top 100 chinese words
# 计已打印项目，不是已扫描项目数。
tot = 0
# 迭代排序字典的key，即token ID，不是(id,length)元组。
for item in length_dict:
    try:
        # 返回zh-cn只是启发式标签，短串和标点均可能误判。
        if langdetect.detect(T1.decode([item])) == "zh-cn":
            # 打印原ID与单token解码文本，文本可含换行。
            print(item, T1.decode([item]))
            # 仅命中并打印后加1，跳过的条目不计。
            tot += 1
    # 裸捕获还会吞KeyboardInterrupt等；保留原码，安全版本独立提供。
    except:
        # 忽略当前错误，不会修复洞位或记录原因。
        pass
    # 原程序实际只取20条，与top100注释不一致。
    if tot == 20:
        # 退出当前for，不退出整个notebook。
        break

# %% 原代码单元 19
# 再次扫描完整o200k，以正则剔除全空白/非词字符/特定符号串；不是中文检测。英文、数字、代码片段仍可通过。tot==20退出，只取20条；bare except问题不改原版，在独立诊断提供安全写法。
# 建立ID→单token解码后Python字符数的字典。
length_dict = {}

# 上界max ID+1包含洞位；原代码会触发Rust panic后被bare except吞掉。
for i in range(T2.n_vocab):
    try:
        # decode默认errors=replace，部分UTF8字节可能显示替换字符。
        length_dict[i] = len(T2.decode([i]))
    # 裸捕获还会吞KeyboardInterrupt等；保留原码，安全版本独立提供。
    except:
        # 忽略当前错误，不会修复洞位或记录原因。
        pass

# Sort by length
# sorted按负长度升序即按长度降序，同长度保留原ID顺序。
length_dict = dict(sorted(length_dict.items(), key=lambda item: -item[1]))
# print(length_dict)
# Print the top 100 chinese words
# 计已打印项目，不是已扫描项目数。
tot = 0
# 锚定整串；剔除只由符号/空白等组成的文本，并非筛中文。
pattern = r'^[\s\W_=+\\-]*$'
# 迭代排序字典的key，即token ID，不是(id,length)元组。
for item in length_dict:
    try:
        # print(T2.decode([item]), re.match(pattern, T2.decode([item])))
        # 保留未被全符号模式匹配的字符串，英文/数字也可通过。
        if not re.match(pattern, T2.decode([item])):
            # 打印原ID与单token解码文本，文本可含换行。
            print(item, T2.decode([item]))
            # 仅命中并打印后加1，跳过的条目不计。
            tot += 1
    # 裸捕获还会吞KeyboardInterrupt等；保留原码，安全版本独立提供。
    except:
        # 忽略当前错误，不会修复洞位或记录原因。
        pass
    # 原程序实际只取20条，与top100注释不一致。
    if tot == 20:
        # 退出当前for，不退出整个notebook。
        break

# %% 原代码单元 20
# 原空代码格，保留；只新增说明注释，不补造训练/调用结果。
