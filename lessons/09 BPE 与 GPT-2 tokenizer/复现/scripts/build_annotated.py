# -*- coding: utf-8 -*-
"""72格具体说明、仅增加代码注释；原语法错误也保留。"""
import ast
import copy
import io
import json
import re
import tokenize
from pathlib import Path
from IPython.core.inputtransformer2 import TransformerManager

BASICS = {
0: ('图片显示接口', '【建议会】从IPython.display导入Image类；导入不读图，单元17才使用。返回显示对象并由Jupyter渲染，不负责tokenizer训练。'),
1: ('基础概念入口', '【必须会】先区分字符、UTF8 bytes、token字符串和整数ID，再讨论词表训练。Markdown不执行代码。'),
2: ('常见词与padding设置代码块', '【必须会】常见词反复切分可能增加token数量，但词表预算/频率/算法都影响边界，不能保证每个常见词单token。Markdown代码块依次设置对象属性：padding_side先left后right，最终为right；pad_token=eos_token只是把已有EOS复用为填充标记，不新增embedding。attention_mask与loss屏蔽仍须正确处理。原“Fix weird overflow”是经验注释，不是所有FP16任务的保证。本块需要预先存在tokenizer，不是完整可运行程序；已加逐行注释而保留原赋值。'),
3: ('AutoTokenizer类', '【必须会】transformers.AutoTokenizer根据checkpoint配置选择实现；from_pretrained负责文件加载，use_fast默认倾向可用的fast类。导入不加载权重。fast的backend_tokenizer/tokens()接口后续会使用，slow对象不一定支持。'),
4: ('封装逐ID解码函数', '【必须会】def定义普通函数，tokenizer参数为可调用对象、text为str。调用tokenizer(text,add_special_tokens=False)返回BatchEncoding，取input_ids为list[int]；列表推导式对每个ID调用decode返回list[str]。没有模型参数更新。byte-level token可能只含部分UTF8字符，单ID解码可能是�；整体decode与convert_ids_to_tokens用途不同。'),
5: ('加载T5 tokenizer作比较', '【建议会】AutoTokenizer.from_pretrained(t5-base)返回T5 tokenizer，可能联网/缓存，算法不是GPT2 byte-level BPE。本次独立补充只把路径字符串换成固定本地T5资产并设local_files_only；未下载T5模型权重。原调用和历史输出保留。'),
6: ('普通词的T5切分', '【建议会】先print(tokenize_str(...hello))，再展示sex的切分。返回是list[str]，print返回None；函数行为沿用单元4。切得更碎不能单独证明训练语料过滤。示例仅用于分词观察，不是模型语义输出。'),
7: ('Python代码分词标题', '【必须会】Python换行、缩进、标点都可能有语义，要整体往返而非先清空白。'),
8: ('BPE基础符号与合并', '【必须会】原文说明从256个byte值开始，反复合并当前高频相邻pair。143859是原笔记提及的Unicode数量，没有在本次确认其统计版本，也不是GPT2词表大小。BPE学习统计/merges，不训练神经网络权重；新pair可含已经合并的片段。'),
9: ('多行原始字符串样本', '【必须会】r三引号定义一个str，保留反斜杠、换行和缩进；里面def/print是字符串内容，不是在此格定义/调用say_hello。末尾print仅打印这个字符串。四空格空行也保留，不能在阅读版自动strip。Python阅读版用相邻字符串字面量表示同一个值，AST字符串常量核对相等。'),
10: ('加载GPT2并查看IDs和tokens', '【必须会】加载返回fast tokenizer；对python_code调用对象，input_ids是list[int]，tokens()是BatchEncoding方法，返回byte映射后的token字符串列表。两次编码是重复调用，没有训练。原远程路径未在验证时直接调用；本地固定资产补充仅改加载路径/local_files_only，并逐项审计；后两行真实执行。'),
11: ('检查normalizer', '【建议会】backend_tokenizer是Rust后端对象，normalizer为属性，可是None。GPT2没有BERT式先小写/去重音步骤；None不表示没有预切分/BPE。此格不改变状态。'),
12: ('预切分与offset', '【必须会】pre_tokenizer.pre_tokenize_str接收str，返回list[(片段str,(start,end))]。这是预切分，不是最终BPE tokens；offset对应原文本位置，映射字符长度可能不同。直接读取方法没有重新训练。'),
13: ('一个byte的整数值', '【必须会】ord接受单字符str或长度1的bytes；b\\xe2是一个byte，结果226。它可能是欧元UTF8的首byte，不是欧元Unicode码点8364。'),
14: ('Unicode与UTF8逐字节', '【必须会】u前缀在Python3与普通str一样；a和€是字符。a.encode为一个byte，ord可读97；€.encode返回三个bytes[226,130,172]，原注释4 bytes不准确。迭代bytes时i已是int，chr后ord又回到同一整数，等价list(e.encode)。两条f字符串展示对象，不修改字符内容。'),
15: ('训练tokenizer标题', '【必须会】开始读处理流水线与学习规则；没有forward/backward、optimizer或梯度。'),
16: ('处理流水线与三种子词算法', '【必须会】normalization规范文本，pretokenization先划片段，tokenizer model应用子词规则/词表ID，postprocessing补模型标记。BPE逐步添加合并，WordPiece使用不同训练规则，Unigram可迭代删减候选；它们不是同一种算法。原拼写postprocesssing指后处理。'),
17: ('token_steps原图', '【必须会】Image读取相对路径../imgs/token_steps.png，width=600是展示像素，返回显示对象。图中小写/去重音及CLS/SEP是类似BERT的示例，不是GPT2固定行为。文字→规范化→预切分→子词→ID/特殊标记；没有embedding或训练loss。配图已下载到代码/imgs，建议从代码/tutorials打开notebook。'),
18: ('BERT示例入口', '【建议会】下方是WordPiece示意，不属于GPT2 BPE完整实现；不要把不同模型的特殊标记混为一谈。'),
19: ('官方搭建tokenizer示例链接', '【建议会】原Colab指向Hugging Face课程notebook，可用于对照流水线；链接不是本次运行证据，不据此认为当前课程格已完整实现。'),
20: ('导入数据加载函数', '【建议会】datasets.load_dataset是模块函数，配置路径/name/split/streaming决定返回Dataset或IterableDataset等。导入本身不下载Wikitext。'),
21: ('Wikitext与文本批生成器', '【必须会】load_dataset(wikitext,name=wikitext-2-raw-v1,split=train)涉及远程语料。本次原数据调用未执行；另在小型真实Dataset上下文执行原get_training_corpus定义。len(dataset)为总行数，range步长1000，dataset[i:i+1000]切片返回dict[list]，取text为list[str]；yield按需交出一批，末批可不足1000。这与train notebook中的逐next流式函数不同。'),
22: ('保留原悬空逗号语法错误', '【必须会】第一行from tokenizers import Tokenizer, models,末尾逗号且无括号，普通Python编译即SyntaxError，第二行不会执行。注释版保留这个错误，静态报告不宣称此格AST可解析。去掉最后一个逗号后可构造Tokenizer(models.WordPiece(unk_token=[UNK]))，仅有模型对象，缺少normalizer、pretokenizer、trainer和保存，仍不是完整BPE训练。修正版只作独立补充。Python阅读版将整格列作注释，避免掩盖原错误或悄悄修复。'),
23: ('GPT2映射入口', '【必须会】下面恢复GPT2接口；上面的BERT/WordPiece片段与它分开。'),
24: ('byte到显示字符表', '【必须会】表中byte13→č、32→Ġ、10→Ċ；这些字符是映射符号，不是原文真的输入它们。非断行空格U+00A0实际UTF8是194/160两bytes，不能把单个160当完整字符。byte映射256种，而完整GPT2词表50257项。'),
25: ('导入bytes_to_unicode', '【必须会】函数所属transformers.models.gpt2.tokenization_gpt2，返回dict[int,str]，从256种byte值到256个唯一可打印显示字符；不是Unicode语义翻译。'),
26: ('建立正向byte映射', '【必须会】调用无参函数并绑定字典bytes_to_unicode_map，末行展示全表。映射保持可逆；此格不是将字符投到embedding。'),
27: ('构造反向映射', '【必须会】.items()迭代(byte,显示字符)，生成器表达式(v,k)交换，dict收集为dict[str,int]。原注释string to id指byte整数，不是GPT2训练词表的token ID；对象单独展示不改变状态。'),
28: ('映射长度', '【必须会】len(dict)返回Python整数256，统计字典项，不等于文本字符数或GPT2完整vocab_size。'),
29: ('基础alphabet的顺序', '【必须会】keys()给显示字符，list收集256项并保留插入顺序；base_vocab[0]/[-1]取首尾。列表下标不是原byte值；base_vocab[10]不可当byte10查询。print输出显示符号。'),
30: ('查询换行byte的显示符号', '【必须会】用字典键10读取Ċ；与原文换行字符串\\n/Unicode码点分别区分。'),
31: ('再次加载并预切分', '【必须会】重新绑定tokenizer为GPT2，调用pre_tokenize_str(python_code)。原格不会学习新merges；只读已有规则。本次加载路径补充固定本地资产，保留其他原语句；读取与单元12结果核对一致。'),
32: ('空白保留规则', '【必须会】原英文：连续空格保留且可能作为一个预片段，一个词前的空格可附着到后面的词。ĊĠĠĠ表示换行和三个空格，不是真实拉丁字符输入。预切分片段不等于最终token，不能保证四空格缩进仅占一项。'),
33: ('展示tokenizer对象', '【建议会】裸变量表达式展示类型、词表规模等repr，不生成文本、不改变词表。'),
34: ('词表训练标题', '【必须会】本notebook后续主要看现有词表；完整train_new_from_iterator在另一个train notebook。'),
35: ('endoftext特殊token', '【必须会】GPT2词表末尾特殊token<|endoftext|>，原资产ID50256。重新训练后的ID可能改变；EOS、PAD、BOS的使用要看配置，不能类比BERT CLS/SEP。'),
36: ('按token表示长度降序', '【建议会】tokenizer.vocab是dict[str,int]，items给二元组；sorted返回新list，lambda x:len(x[0])按表示字符串长度，reverse=True降序。[:10]左闭右开；推导式逐token转展示文本。长度不是语料词频，也不是token原文byte数；可能含不完整UTF8片段。'),
37: ('按ID降序', '【建议会】lambda x:x[1]按ID，reverse=True从大到小；前12项不自动等于最常见词。convert_tokens_to_string接收list[str]，拼接/反byte映射，不训练模型。'),
38: ('空代码格', '【可以暂时不掌握】原source空，保留编号与元数据；注释版只加入说明性注释，AST仍为空。'),
}
TRAIN = {
0: ('训练tokenizer入口', '【必须会】从代码文本学习新的BPE词表与merges，不训练Transformer。'),
1: ('导入加载、语料与进度接口', '【必须会】AutoTokenizer是transformers的类；load_dataset是datasets函数；tqdm是可迭代对象的进度包装器，不负责数据本身或训练算法。导入不读语料。'),
2: ('原本地codeparrot路径', '【必须会】load_dataset(./codeparrot/,split=train,streaming=True)相对当前工作目录；返回流式IterableDataset时按需读。原仓库没有本地语料，不能仅凭streaming生成文件。原格保留但未运行；独立补充用临时JSONL与真实datasets json读取，不假冒课程语料。'),
3: ('建立有消费位置的迭代器', '【必须会】iter(dataset)返回迭代器，next每调用一次消费一条dict；iter_dataset跨后续函数共用，没有自动复位。本次在明确小语料IterableDataset上下文执行原句。'),
4: ('旧GPT2 tokenizer作为处理模板', '【必须会】加载原GPT2 fast tokenizer；后面重建词表不会保留原ID语义。本地路径补充仅改字符串/local_files_only，固定revision且无权重。'),
5: ('构建alphabet与训练批标题', '【必须会】先准备256个基础符号，再提供字符串批给trainer。'),
6: ('字节正反映射与基础词表', '【必须会】无参bytes_to_unicode返回256项dict；(v,k)反转items，list(keys)收集映射字符，base_vocab为list[str]长度256，不是完整50257项词表。initial_alphabet用于覆盖任意UTF8字节。'),
7: ('length与batch_iterator', '【必须会】length=100000设全局读取上限；def函数默认batch_size=1000，读取全局iter_dataset/length。yield使调用返回惰性生成器，每批为list[str]；两个_是普通循环变量。range(0,length,batch_size)按批数计tqdm，内部next在不满批时会耗尽并触发RuntimeError。不能整除时仍读满末批，可能超读；本次执行原函数体并用小上下文实测，不消费10万条。'),
8: ('训练新BPE tokenizer', '【必须会】fast对象train_new_from_iterator接收batch_iterator()、vocab_size=12500和initial_alphabet=base_vocab。返回新对象new_tokenizer，不修改旧tokenizer；清空旧vocab/merges并保留处理/特殊token配置。目标含alphabet/特殊token，语料不足可能达不到12500；原格未原配置运行，补充仅改vocab320并调用真实原生成器（默认批大小仍1000）。'),
9: ('展示新对象', '【建议会】读取new_tokenizer repr，显示的是训练后的词表/配置，不涉及模型logits。'),
10: ('新词表检查标题', '【必须会】检查实际大小、特殊token和合并规则，不仅看目标vocab_size。'),
11: ('按整数ID升序排序', '【必须会】vocab.items是(token,ID)，sorted key=lambda x:x[1]升序、reverse=False；返回新list tokens，不改变tokenizer词表。'),
12: ('观察alphabet前10项', '【建议会】切片[:10]不含右边界，读取映射字典插入顺序的前10显示符号，非最常见的10个token。'),
13: ('第257–279项显示转换', '【建议会】tokens[257:280]最多23个二元组，for t,_解包忽略ID，转换单token为显示文本并返回list[(原token,显示文本)]。257只是原代码切分假设，应查实际基础/特殊token排布；单token可能不是完整UTF8字符。'),
14: ('词表尾15项', '【建议会】[-15:]从倒数第15开始，按前面ID升序列表取尾部；不是按频率排序。转换只是可视化，不撤销词表训练。'),
15: ('Python关键字检查标题', '【必须会】检查裸字符串是否词表项，只是覆盖统计，不是代码生成质量。'),
16: ('导入标准库keyword', '【建议会】keyword.kwlist是当前Python版本硬关键字列表。软关键字另有softkwlist，不能硬编码某版数量或混淆两类。'),
17: ('关键字数量', '【建议会】len(keyword.kwlist)返回int；本次Python3.12有35项，此数量依版本可能变，不是目标词表大小。'),
18: ('检查裸关键字成员关系', '【必须会】for遍历每个kw字符串，kw not in new_tokenizer.vocab检查字典键，不存在时print。带空格的Ġreturn和裸return是不同token；缺失不意味着模型不能用多项表示，也不证明生成能力。'),
19: ('第二轮训练标题', '【必须会】retrain这里是再次从旧tokenizer模板训练新的词表，不是继续更新第8格的新merges。'),
20: ('第二轮重定义函数并继续消费', '【必须会】length=100000*2是200000；同名def覆盖旧函数，依旧共享iter_dataset，没有iter(dataset)复位。train_new_from_iterator在旧tokenizer上调用且vocab32768=2^15，同时改变样本范围/规模与预算，不能归因单一变量。本次AST审计的小语料补充仅改length和vocab，仍真实执行原生成器默认1000，证明继续消费。'),
21: ('再次检查关键字', '【建议会】对第二轮新对象重复裸词表成员检查；与第一轮相同API，打印差异不等于下游指标提升。'),
22: ('Hub发布标题', '【建议会】保存本地与推送远程是两种操作；本讲验证采用临时目录save_pretrained/重载。'),
23: ('创建仓库提示为注释', '【建议会】两条#行不会导入huggingface_hub或create_repo。实际创建需账号/权限，原notebook并未在此格执行。用户授权的是公开学习GitHub仓库，不是讲师Hugging Face组织。'),
24: ('原代理环境写入', '【建议会】os.environ映射赋值HTTP/HTTPS代理到127.0.0.1:7890，会改变本进程后续请求。必须有该服务才有效；本次明确跳过，不修改真实代理。'),
25: ('原讲师组织Hub上传', '【建议会】ckpt/org为字符串，new_tokenizer.push_to_hub会向远程写文件，organization为历史API用法。本次不执行、不模拟上传返回值；源码保留，验证补充只做本地save_pretrained/AutoTokenizer重载。'),
26: ('空代码格', '【可以暂时不掌握】原空格保留，注释版AST空。'),
}
AUX = {
0: ('数据辅助片段标题', '【建议会】仅探查本地路径和缓存，不是BPE算法或完整训练。'),
1: ('Linux shell cd', '【建议会】!是IPython shell语法，非普通Python；/media/whaow为讲师Linux路径。单次子shell cd通常不改变内核后续cwd，本次Windows跳过此命令。注释版保留命令，阅读版转换为等价get_ipython().system调用，需IPython环境。'),
2: ('数据加载与构建器接口', '【建议会】load_dataset读数据，DownloadConfig控制下载，load_dataset_builder构造加载器并探查缓存。导入不加载课程codeparrot语料。'),
3: ('全部注释的代理示例', '【建议会】原行全以#开头，不修改任何环境。HTTP/HTTPS/all_proxy分别对应代理配置，CURL_CA_BUNDLE空串是原示意，不能当成读者通用网络修复。'),
4: ('shell工作目录查询', '【建议会】!pwd要求shell支持，Windows通常没有该命令；与内核Python cwd不是同一个概念。仅进行IPython语法转换AST核对，不执行Linux命令。'),
5: ('探查原本地dataset缓存', '【建议会】load_dataset_builder(./dataset/codeparrot/).cache_dir取得构建器缓存路径字符串，不等于已经加载完整dataset。原DownloadConfig与load_dataset行都是注释，路径和缓存历史属于讲师。本次原路径缺失故跳过，独立临时JSONL/json builder实际检查cache_dir属性。'),
}

# 首次语法/API的含义在逐格说明中展开；原有效语句前再插入就近注释。
LINE_COMMENTS = [
('from IPython.display', '导入图片显示类，实际读图在Image调用。'),
('from transformers.models', '导入GPT2的256-byte可逆显示映射函数。'),
('from transformers import', '根据checkpoint选择具体tokenizer类，导入不加载语言模型权重。'),
('from datasets import', '真实datasets接口；导入与下载/读取语料分开。'),
('from tqdm import', '包装可迭代对象显示进度，不统计BPE pair。'),
('from tokenizers import', '原句末尾悬空逗号导致SyntaxError；保留错误，修正版独立说明。'),
('def tokenize_str', '普通函数接收tokenizer对象和文本，返回逐ID解码的list[str]。'),
('input_ids =', '编码后取list[int]；add_special_tokens=False避免自动添加模型标记。'),
('return [tokenizer.decode', '逐ID可能含不完整UTF8字符，不能替代整体decode。'),
('tokenizer_t5 =', 'T5分词器加载可能联网；补充固定本地资产，不下载模型权重。'),
('tokenizer = AutoTokenizer', '原远程GPT2路径；验证补充明确只改为固定本地资产。'),
('tokenizer = Tokenizer', '仅创建WordPiece模型对象，还缺trainer/处理流水线。'),
('python_code =', '多行原始字符串保留代码文本、换行和缩进，不执行其内部Python。'),
('Image(', '相对当前cwd读取配图，width只是展示像素。'),
('tokenizer.backend_tokenizer.normalizer', '读取后端normalizer属性；GPT2这里为None。'),
('tokenizer.backend_tokenizer.pre_tokenizer', '真实预切分返回片段及offset，不等于最终BPE token。'),
('ord(', '单byte到整数；不等于完整Unicode字符码点。'),
('a, e =', '两个Unicode字符串；欧元UTF8有3个bytes。'),
('byte = ord', '长度1的bytes可传ord，a的值为97。'),
('byte = [', '迭代bytes已有int，chr再ord等价取原byte值。'),
('dataset = load_dataset', '原语料入口依赖路径/远程数据；本次未以原配置运行。'),
('def get_training_corpus', '生成器返回Dataset的text批，切片末批可不足1000。'),
('for i in range', '按1000行步长遍历，len用于非流式Dataset。'),
('yield dataset', 'Dataset切片是列字典，取text返回list[str]。'),
('bytes_to_unicode_map =', '256项byte→唯一显示字符字典，不是token ID词表。'),
('unicode_to_bytes_map =', '生成器交换(key,value)，得到显示字符→byte反向字典。'),
('len(bytes_to_unicode_map)', '返回基础映射项数256。'),
('base_vocab =', 'list保留映射键顺序，基础alphabet含256显示字符。'),
('tokens = sorted', 'items为(token,ID)；lambda指定排序键，返回新列表。'),
('[tokenizer.convert', '只转换所取token的表示，不改变词表或模型。'),
('[(t, new_tokenizer', '解包token/ID，再查看单token展示文本。'),
('iter_dataset =', '建立有消费位置的迭代器；两次训练共用它。'),
('length =', '原训练条数上限，不是vocab大小，第二轮不自动重置迭代器。'),
('def batch_iterator', '默认每批1000，读取全局length和iter_dataset。'),
('for _ in tqdm', '外层按批数计进度；不能整除时原函数仍请求满批。'),
('yield [next', '逐next读取content字符串，耗尽会在生成器中触发RuntimeError。'),
('new_tokenizer =', '真实训练返回新对象；旧GPT2的vocab/merges不就地更新。'),
('vocab_size=', '目标词表大小含alphabet/特殊token，语料不足可小于目标。'),
('initial_alphabet=', '确保byte覆盖，不是继承旧50257项vocab。'),
('import keyword', '标准库关键字列表由当前Python版本决定。'),
('len(keyword.kwlist)', '计数硬关键字，非新词表项数。'),
('for kw in', '逐个检查裸关键字，而非代码上下文中token长度。'),
('if kw not in', '字典键成员测试；缺失并不等于无法多token表示。'),
('import os', '标准库环境接口；下面原代理写入本次跳过。'),
('os.environ[', '设置当前进程代理；读者不能假定本机有7890服务。'),
('ckpt =', '原远程仓库名称，非本地保存目录。'),
('org =', '原讲师组织名；本次不向此组织写入。'),
('new_tokenizer.push_to_hub', '外部上传副作用，原语句保留但明确不执行。'),
('load_dataset_builder(', '仅构建加载器并取cache_dir，不代表数据全量加载。'),
('!cd', 'IPython shell命令；Linux路径，Windows验证跳过。'),
('!pwd', 'IPython shell查询；本次仅转换语法，不执行Linux命令。'),
('print(', '输出当前数值/字符串，不更新词表；返回None。'),
('base_vocab[:', 'Python切片左闭右开，映射顺序不等于byte键或词频排名。'),
('tokenize_str(', '调用单ID解码函数，返回分段观察值。'),
('new_tokenizer', '展示当前新对象，不进行第二次训练。'),
('tokenizer', '展示已有对象，不生成语言模型回答。'),
('bytes_to_unicode_map', '展示映射对象，不更改字典。'),
('unicode_to_bytes_map', '展示反向byte映射对象。'),
]
FENCE = re.compile(r'(```[^\n]*\n)(.*?)(```)', re.S)

def note(i, cell, notes):
    title, body = notes[i]
    return '## 单元 {}（{}）：{}\n\n{}\n'.format(i, cell['cell_type'], title, body)

def annotate(source, title):
    # 多行字符串内部是用户代码文本，不向其内部插入注释。
    inside = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(source).readline):
            if t.type == tokenize.STRING and t.end[0] > t.start[0]:
                inside.update(range(t.start[0] + 1, t.end[0] + 1))
    except tokenize.TokenError:
        pass
    lines = ['# ' + title + '；原执行语句保留，详细参数/shape/边界见本格说明。\n']
    for number, line in enumerate(source.splitlines(True), 1):
        stripped = line.strip()
        if number not in inside and stripped and not stripped.startswith('#'):
            for start, meaning in LINE_COMMENTS:
                if stripped.startswith(start):
                    lines.append(line[:len(line) - len(line.lstrip())] + '# ' + meaning + '\n')
                    break
        lines.append(line)
    return ''.join(lines)

def reading_code(filename, i, source):
    if filename == 'tokenizer_basics.ipynb' and i == 22:
        return '# 原单元22无法解析，以下整格作为原文注释保留；独立修正版见复现。\n' + '\n'.join('# ' + line for line in source.splitlines()) + '\n'
    if filename == 'tokenizer_basics.ipynb' and i == 9:
        tree = ast.parse(source)
        text = tree.body[0].value.value
        literal = 'python_code = (\n' + ''.join('    ' + repr(line) + '\n' for line in text.splitlines(True)) + ')\nprint(python_code)\n'
        assert ast.dump(ast.parse(literal), include_attributes=False) == ast.dump(tree, include_attributes=False)
        return '# 原三引号str改为相邻字面量阅读排版；值/空白/换行与AST不变。\n' + literal
    if source.lstrip().startswith('!'):
        return '# 原IPython命令：' + source.strip() + '\n' + TransformerManager().transform_cell(source)
    return source

def build(lesson, filename, notes, prefix):
    directory = lesson / '代码/tutorials'
    original = json.loads((directory / filename).read_text(encoding='utf-8'))
    assert set(notes) == set(range(len(original['cells'])))
    for changed, suffix in [(False, '原代码_带说明.ipynb'), (True, '代码注释.ipynb')]:
        nb = copy.deepcopy(original)
        intro = '# 第09讲 ' + filename + '\n\n全部原格说明。注释版仅增加注释，原语句、元数据、历史输出及原错误保留。新执行结果另见复现；原远程语料/代理/上传没有运行。\n'
        nb['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'id': prefix + '-intro', 'source': intro.splitlines(True)}]
        for i, cell in enumerate(original['cells']):
            nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'id': prefix + '-note-' + str(i), 'source': note(i, cell, notes).splitlines(True)})
            keep = copy.deepcopy(cell)
            source = ''.join(cell['source'])
            if changed and cell['cell_type'] == 'code':
                keep['source'] = annotate(source, notes[i][0]).splitlines(True)
            elif changed and filename == 'tokenizer_basics.ipynb' and i == 2:
                def add(m):
                    return m[1] + '# 需要已有tokenizer；三个属性赋值有状态副作用，最终padding_side=right。\n' + annotate(m[2], 'padding/EOS设置') + m[3]
                keep['source'] = FENCE.sub(add, source).splitlines(True)
            nb['cells'].append(keep)
        output = suffix if filename == 'tokenizer_basics.ipynb' else filename.removesuffix('.ipynb') + '_' + suffix
        (directory / output).write_bytes((json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode('utf-8'))
    doc = '\n# ' + filename + '：全部单元\n\n'
    py = ['# -*- coding: utf-8 -*-', '# 第09讲Python阅读版：原语句说明及历史错误边界见逐格内容。', '# 主要用于阅读；远程语料、代理、Hub上传未在本次验证执行。', '']
    for i, cell in enumerate(original['cells']):
        detail = note(i, cell, notes)
        source = ''.join(cell['source'])
        doc += detail + '\n'
        py += ['# %% [markdown]'] + ['# ' + line for line in detail.splitlines()] + ['']
        if cell['cell_type'] == 'code':
            doc += '原代码：\n\n```python\n' + source + '\n```\n\n'
            py += ['# %% 原单元 {}'.format(i)] + annotate(reading_code(filename, i, source), notes[i][0]).splitlines() + ['']
        else:
            py += ['# %% [markdown] 原单元 {}'.format(i)] + ['# ' + line for line in source.splitlines()] + ['']
            if filename == 'tokenizer_basics.ipynb' and i == 2:
                doc += '原Markdown中的设置代码（未作为Python执行格）：\n\n' + source + '\n\n'
    (directory / (filename.removesuffix('.ipynb') + '_代码注释.py')).write_text('\n'.join(line.rstrip() for line in py).rstrip() + '\n', encoding='utf-8')
    return doc

def main():
    lesson = Path(__file__).resolve().parents[2]
    title, detail = BASICS[36]
    BASICS[36] = (title, detail + ' 原sort没有同长度的第二排序键，词表字典遍历次序可不同；前10项若跨并列边界，两次列表可能不同。本次保留两次输出并核对前10项长度/转换契约，不能把不同列表冒充完全相同。')
    doc = '# 第09讲代码单元解释\n\n全部72格：basics39、train27、辅助gpt2 6；原编号分别从0开始。原22语法错误与两条IPython命令保留，重要Markdown2设置块也详注。\n\n[主解释](解释.md) · [基础代码注释](代码/tutorials/代码注释.ipynb) · [训练代码注释](代码/tutorials/gpt2_train_tokenizer_代码注释.ipynb) · [验证记录](复现/README.md)\n\n'
    for filename, notes, prefix in [('tokenizer_basics.ipynb', BASICS, 'basics09'), ('gpt2_train_tokenizer.ipynb', TRAIN, 'train09'), ('gpt2.ipynb', AUX, 'aux09')]:
        doc += build(lesson, filename, notes, prefix)
    (lesson / '代码单元解释.md').write_text('\n'.join(line.rstrip() for line in doc.splitlines()).rstrip() + '\n', encoding='utf-8')
    print('All72 cells explained; 6 derived notebooks, Markdown2 annotations and 3 Python reading files built.')

if __name__ == '__main__':
    main()
