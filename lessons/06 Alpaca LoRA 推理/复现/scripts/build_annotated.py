# -*- coding: utf-8 -*-
"""逐格说明全部28单元；向18代码格只增加注释，保留安装/代理/交互/空单元。"""
import copy
import json
from pathlib import Path

NOTES = {
0:('两种源码安装方式', '''【建议会】两条`!pip install -q git+...`分别用HTTPS和SSH从Git安装Transformers；`!`是IPython调用shell，`pip install`安装，`-q`减少输出，`git+`声明源码来源。
它们是替代方式，通常选一条，不需要顺序安装两次。SSH依赖已有身份/配置；原命令没固定SHA或tag，不能据此重建老师4.30.0.dev0。
本次不执行此格。注释版保留原指令，Python阅读版把它们变成注释，重做验证的固定环境见复现文档。此格会修改环境，不是模型推理代码。'''),
1:('导入Transformers模块', '''【建议会】import把模块绑定为transformers；不会在此下载LLaMA或加载7B权重。后面版本属性和类导入才引用它；与模型推理是不同步骤。'''),
2:('查看实际安装版本', '''【建议会】`transformers.__version__`是模块属性str，notebook显示最后一个表达式结果。作者历史输出是4.30.0.dev0；本次教学环境为4.44.2。
版本名称没有包含开发版Git SHA，不能精确锁老师实现。此行不设置版本，也不更改模型。'''),
3:('基础概念标题', '''【必须会】本节串起基模型、Alpaca指令数据、LoRA参数增量与推理。标题本身不创建模型或调用函数。'''),
4:('LLaMA到Alpaca LoRA', '''【必须会】基模型提供预训练语言能力；Alpaca提供指令训练语境；LoRA是微调参数的一种方式，本notebook只读取已有adapter。
“llama→alpaca”是方法背景，不意味着先下载一个完整Stanford Alpaca模型再叠adapter。标准prompt是文字输入格式，不是网络结构；本篇没有训练loss或optimizer。'''),
5:('进入模型与分词器加载', '''【必须会】模型把token ID变成logits；tokenizer把字符串变成ID/attention mask。两者分别加载，不是同一个对象。
本格是章节标题；真实下载发生在10、14格。'''),
6:('外部生成示例链接', '''【建议会】此链接指向Alpaca-LoRA的generate.py，作为背景。它引用浮动main分支，原notebook未给出同版SHA；本次不把该链接当作固定实现证据。
实际解释和AST核对以本仓库固定提交的28格源码为准；读取链接不是自动授权运行外部脚本。'''),
7:('三个类各自负责什么', '''【必须会】LlamaTokenizer是分词器类；LlamaForCausalLM是有词表输出头的LLaMA类；GenerationConfig是生成配置类。from...import把三个类名绑定当前作用域，不在此创建对象。
后续from_pretrained是类方法，可能读远程/本地文件；配置构造器只是创建参数对象。本次会真正导入这些类，不把导入成功当成7B加载成功。'''),
8:('原代理配置保持注释', '''【建议会】原三行均为注释：没import os，也没写HTTP_PROXY或HTTPS_PROXY。127.0.0.1:7890是示例代理地址，不是模型端口。
如果自行去掉#，会改变进程网络环境；本次不启用代理，也不执行附件里的口述命令。空操作也纳入覆盖审计。'''),
9:('进入基模型加载', '''【必须会】先建立基模型，再装adapter。此标题没有变量变化；下一格的model尚未变成PEFT包装。'''),
10:('加载原7B基模型', '''【必须会】`LlamaForCausalLM.from_pretrained(...)`输入历史Hub标识str及关键字参数，返回LlamaForCausalLM实例，绑定model；不传训练数据。
load_in_8bit=True使用旧式bitsandbytes量化加载，可能保留embedding/norm等浮点参数；device_map='auto'交给Accelerate自动分派，不等于两张卡各放一半或数据并行。
预期input_ids为int64[B,S]，模型内部embedding为浮点[B,S,H]，LM logits为[B,S,V]。历史7B常见H4096、V32000；本次未实测原权重。
本格可能下载大文件并分配内存。本次不执行它，教学补充用本地随机小型LLaMA FP32模型，绝不把它写成原8bit7B复现。'''),
11:('展示基模型结构', '''【建议会】单独写model，notebook显示repr：可以看embedding、层、attention投影和LM head。它不调用forward/generate，不复制权重，不训练。
在本次教学上下文里展示的是随机小型LLaMA，不能把其shape当成历史7B的参数。'''),
12:('逐个参数检查device和dtype', '''【必须会】named_parameters()是model的方法，返回(name,Parameter)迭代器；enumerate给每项加从0起的序号i。
para是二元tuple：para[0]为名称str，para[1]为Parameter/tensor。原打印只显示device和dtype，名称打印行仍被注释。
i是枚举序号，不是层号。device可为cpu/cuda:N；dtype是元素类型，不是shape。遍历与print不搬设备、不改dtype、不更新参数。
历史多卡分布是当时机器输出，本次FP32 CPU小模型会显示cpu/torch.float32。读取实际保存输出不等于当前再次测得原分派。'''),
13:('进入tokenizer加载', '''【必须会】模型处理整数ID，tokenizer建立文本与ID的规则。必须与基模型词表兼容；模型结构相同并不能保证两个tokenizer可互换。'''),
14:('加载分词器文件', '''【必须会】`LlamaTokenizer.from_pretrained(model_id)`读取SentencePiece词表、特殊token和tokenizer配置，返回分词器实例tokenizer。
同一个Hub名字不是重新下载另一份7B权重。本次不执行远程加载；教学补充用真实SentencePiece训练的小词表，再通过LlamaTokenizer本地加载，不冒充原decapoda词表。'''),
15:('展示tokenizer配置', '''【建议会】最后表达式tokenizer显示词表规模、特殊token等repr；此格没有编码某条指令。
vocab_size通常是基础词表规模，len(tokenizer)还可能包括新增token；embedding ID范围要匹配。实际小词表在复现记录单独列出。'''),
16:('进入LoRA装载', '''【必须会】从已建基模型进入已有adapter推理。这里不是第03讲创建并训练新adapter的步骤。标题不改变model。'''),
17:('Alpaca-LoRA项目背景', '''【建议会】项目链接是方法背景和来源归属，原notebook没有固定其main版本。本篇只核对本notebook与固定官方PEFT源码，不照链接自动运行项目或下载权重。'''),
18:('导入PEFT包装类', '''【必须会】PeftModel是PEFT包装入口类，import不会注入LoRA。实际读取配置/参数在下一格的类方法from_pretrained。
本次验证直接导入官方PEFT v0.3.0固定提交7120c9a，而不是只根据安装包名称猜版本。'''),
19:('读取已有adapter', '''【必须会】`PeftModel.from_pretrained(model, adapter_id)`第一参数是现有基模型对象，第二参数是adapter目录/Hub标识。返回任务对应的PEFT对象，再绑定model。
在固定v0.3.0，adapter_name默认'default'，is_trainable默认False：读取配置、根据任务选包装、注入目标层、加载adapter权重并进入推理状态；不是新训练步骤。
LoRA目标层复用基权重，并增A/B低秩支路。对4096→4096、rank r的投影，A[r,4096]、B[4096,r]，x[B,S,4096]→[B,S,r]→[B,S,4096]。
本次不调用原远程adapter标识；补充明确加载自编非零adapter的本地文件，核对保存/重载后前向一致、default键及冻结，不把随机adapter当成Alpaca训练成果。'''),
20:('展示装载后模型', '''【建议会】model现在是PEFT包装，repr可看LoRA层和基模型。不产生回答、不证明已训练、更不是将A/B合并到基础权重。
推理需要后面的prompt→tokenizer→generate。'''),
21:('区分保存目标与库默认目标', '''【必须会】从peft导入mapping模块、从peft.utils导入other模块，都是内部调试接口，升级版本时路径或导出可能变化。
model.config.model_type是架构名str；model.peft_config是adapter名字到配置的字典，['default']取当前配置对象，.target_modules读取保存的目标列表。
历史输出q_proj/k_proj/v_proj/o_proj四种目标来自adapter配置；other字典里的llama默认q_proj/v_proj只用于未指定目标时的推断，两者不矛盾。
最后一行是属性/字典展示，不是再次注入。mapping虽被导入，这个单元没有调用它。实际小型adapter也显式存四种目标，并核对默认字典仍为两种。'''),
22:('进入指令示例', '''【必须会】接下来的模板、生成配置和函数只是推理接口；此标题没有产生训练样本或做标准评估。'''),
23:('Stanford Alpaca背景', '''【建议会】链接给出指令训练项目背景；原notebook的模板是本次逐字符核对的实现。没有额外下载Stanford数据集，没有声明已重现原项目评估。'''),
24:('两个prompt分支与字符串语法', '''【必须会】generate_prompt是本notebook定义的普通函数；instruction预期str，input默认None，返回完整str，不改模型状态。
if input是真值判断：None/''走无Input模板，非空字符串走带Input模板。input参数在函数局部遮蔽内置input，外部单元26仍调用Python内置输入。
f三引号保留换行，大括号插入instruction/input。英文引导句分别要求完成指令，或结合补充上下文完成指令；Instruction/Input/Response标签与换行都是tokenizer会接收的输入。
return立即结束该分支；Response冒号后没有示例答案，让模型续写。本次执行原函数两个分支、空字符串分支，逐字核对模板；没有把注释塞进f字符串里改变prompt。'''),
25:('配置、完整数据流及输出风险', '''【必须会】GenerationConfig对象保存temperature1.5、top_p0.8、num_beams4；原代码没设置do_sample，所以默认False、num_return_sequences默认1，实际为beam search，不能仅凭top_p说已随机采样。
inference是普通函数，instruction/input与上一格相同；读取全局model、tokenizer、generation_config，主要打印Response，隐式返回None。函数定义本身不生成回答。
prompt(str)→tokenizer(...,return_tensors='pt')得到BatchEncoding；input_ids为int64[1,S]，attention_mask也为[1,S]。原代码只取ID再.cuda()，没有传mask；单条无padding输入不等于批量padding场景安全。
cuda()是tensor方法，默认当前CUDA设备；CPU包在调用函数时于此失败。分派模型的入口设备应核对，auto不是保证。
generate接收配置和ID，max_new_tokens256只限制新增G，0<G≤256；结构化generation_output.sequences为int64[R,S+G]，原默认R1。beam内部处理4条候选，output_scores保留逐步候选分数，可能增加内存；后文没有使用scores。
for s按第0维取一条序列，s为[S+G]。decode默认不跳过特殊token且包含输入前缀；split('### Response:')[1]按字符串找第二段，缺标记会IndexError，输入含同标记可能提错位置；strip仅去首尾空白。print输出而无return值。
本次原函数定义/配置原样执行，并记录调用的真实CUDA失败；CPU补充仅把.cuda()改成.to('cpu')，其余表达式保留，真实生成、解码和打印另记。随机小模型输出不用于能力评估。'''),
26:('交互输入会等待人', '''【建议会】Python先调用内置input('Instruction: ')，向终端要字符串，再把返回值作为inference第一个位置参数。不是给函数局部input参数重新赋值。
本格会等待键盘，可能随后生成256token，本次不自动执行交互格。核心函数用明确测试字符串调用；静态审计仍保留并核对本格AST。'''),
27:('空代码单元', '''【可以暂时不掌握】原格为空，没有额外算法、输出或变量变化。说明版保留空格，注释版仅添加解释注释，AST仍为空；不能漏掉它来凑“全部代码”覆盖。'''),
}

CODE_NOTES = {
0:['【建议会】原两条安装指令是HTTPS/SSH替代方式，未固定版本；本次不执行。','!是IPython shell语法，阅读版保留为注释；-q只减少安装输出。'],
1:['【建议会】导入transformers模块供版本属性和后续类使用；不下载模型。'],
2:['【建议会】访问版本字符串，不设置版本；原历史4.30.0.dev0与本次4.44.2分开记录。'],
7:['【必须会】分别导入分词器类、带LM输出头的模型类、生成配置类；此行只导入。'],
8:['【建议会】原代理示例全部是注释，不导入os、不设置环境；本次保持未启用。'],
10:['【必须会】类方法加载历史基模型标识；此格可能下载7B权重，本次未执行。','load_in_8bit是旧量化加载；device_map=auto自动分派，不保证两卡均分。','返回基模型对象model，尚未读Alpaca adapter；历史shape不作本次实测。'],
11:['【建议会】展示模型repr，不forward、不训练；教学上下文模型明确另设。'],
12:['【必须会】named_parameters返回(name,Parameter)；enumerate给每项加序号。','para[1]是参数tensor；device/dtype是设备/元素类型，不是shape或层号。','原名称打印行仍被注释；只读检查和print，不搬设备、不改权重。'],
14:['【必须会】读取分词器文件并返回tokenizer，不再加载一份7B网络。','模型与tokenizer词表及特殊ID必须匹配；本次不运行原远程加载。'],
15:['【建议会】展示tokenizer配置，不在此编码指令；词表对象不是input_ids。'],
18:['【必须会】导入PEFT包装类；实际注入/读取参数发生在from_pretrained。'],
19:['【必须会】现有基模型+已训练adapter标识→PEFT对象，再绑定model。','读取已保存配置与A/B权重，不是get_peft_model创建新adapter，也不训练。','固定v0.3.0默认adapter_name=default、is_trainable=False；本次远程格不执行。'],
20:['【建议会】展示PEFT结构；不生成文本，不合并权重，也不证明训练成功。'],
21:['【必须会】读取已加载adapter目标与库默认字典，二者含义不同。','default是adapter字典键；q/k/v/o为历史保存目标，默认llama仅q/v。','内部模块导入路径随版本变化；本次直接核对固定PEFT0.3.0。'],
24:['【必须会】普通函数：instruction为指令str，input可选；返回完整prompt str。','f三引号内部文本保持原样；注释不能插入字符串内部，否则会改变模型输入。','None和空字符串走无Input分支；Response结尾留空供续写。'],
25:['【必须会】构造配置后定义inference；定义函数本身不执行推理。','原do_sample默认False、返回数默认1：4beam不等于4个答案，top_p不自动启用采样。','完整shape流与CUDA失败、未传mask、保留scores和split风险见本格说明。'],
26:['【建议会】内置input会等键盘；返回str传给inference，本次不自动执行交互格。'],
27:['【可以暂时不掌握】原空代码格；只加注释，无可执行表达式，保留原元数据/输出。'],
}

LINE_NOTES = {
    'load_in_8bit=True,':'旧式bitsandbytes量化加载路线；不是AMP上下文。',
    'device_map="auto",':'自动安排子模块设备，不等于两卡固定均分。',
    'for i, para in enumerate':'i是枚举序号；para是(name,Parameter)二元tuple。',
    'print(f':'para[1]取参数，.device/.dtype只读查看；\t是输出制表符。',
    'from peft import mapping':'导入内部模块供调试；下面实际没有调用mapping函数。',
    'from peft.utils import other':'固定版本的内部工具模块，默认目标字典在其中。',
    "print('model_type'":'读取基模型架构字符串，例如llama；不是任务名。',
    "print(model.peft_config":'从default配置读已保存目标，优先于默认推断。',
    'other.TRANSFORMERS':'最后表达式展示默认字典，不再次注入层。',
    'if input:':'真值判断，None/空字符串为False。',
    'return f':'返回多行f字符串；其内部原文本和换行保持不变。',
    'else:':'没有非空input的模板分支，只含Instruction/Response。',
    'temperature=1.5,':'仅在相应采样路径调分布；原配置没有do_sample=True。',
    'top_p=0.8,':'核采样候选阈值；不自动启动采样，原注释意图与实际策略需区分。',
    'num_beams=4,':'内部保留4条束；num_return_sequences仍默认1。',
    'def inference':'读取全局模型/分词器/配置，打印结果，隐式返回None。',
    'prompt = generate_prompt':'指令/可选上下文→完整模板str。',
    'inputs = tokenizer':'调用分词器，return_tensors=pt得到PyTorch BatchEncoding。',
    'input_ids = inputs':'取int64[1,S]并移动到当前CUDA；原代码未把attention_mask传入generate。',
    'generation_output = model.generate':'调用模型/PEFT的生成方法，包含多步解码和缓存，不是一次forward。',
    'input_ids=input_ids,':'输入token ID，不是embedding或原始文字。',
    'generation_config=generation_config,':'传入上面配置，原默认beam search。',
    'return_dict_in_generate=True,':'返回结构化生成对象，通过.sequences取序列。',
    'output_scores=True,':'保留逐步分数，后文未用；beam分数通常为[4,V]。',
    'max_new_tokens=256':'只限制新增token上限，输出总长S+G。',
    'for s in generation_output.sequences:':'按返回条数遍历，s为一维int64[S+G]。',
    'output = tokenizer.decode(s)':'ID→字符串，默认不跳过特殊token，包含prompt前缀。',
    'print("Response:"':'split取第一个Response标记后的片段；标记缺失/重复有风险，strip去首尾空白。',
}


def note(index,cell):
    title, text = NOTES[index]
    return '## 单元 {}（{}）：{}\n\n{}\n'.format(index,cell['cell_type'],title,text)


def annotate(index,source):
    prefix = '\n'.join('# '+s for s in CODE_NOTES[index])+'\n'
    lines=[]
    in_string=False
    for line in source.splitlines(True):
        stripped=line.strip()
        # 先判断当前是否处在原多行字符串内部；这些行绝不插入注释。
        if not in_string and stripped and not stripped.startswith('#'):
            for start,text in LINE_NOTES.items():
                if stripped.startswith(start):
                    indent=line[:len(line)-len(line.lstrip())]
                    lines.append(indent+'# '+text+'\n')
                    break
        lines.append(line)
        if line.count('"""') % 2:
            in_string=not in_string
    assert not in_string
    return prefix+''.join(lines)


def main():
    lesson=Path(__file__).resolve().parents[2]
    directory=lesson/'代码/tutorials'
    nb=json.loads((directory/'06_alpaca_llama_7b_lora.ipynb').read_text(encoding='utf-8'))
    assert set(NOTES)==set(range(28))
    assert set(CODE_NOTES)=={i for i,c in enumerate(nb['cells']) if c['cell_type']=='code'}
    intro='# 第06讲：原代码与中文说明\n\n原编号从0开始，原输出为老师历史记录。说明版不改原单元，注释版仅加注释；远程7B加载、源码安装与交互格本次未执行。CPU教学验证的改动/模型/词表单独说明，不能作为Alpaca回答质量证据。\n'
    for is_annotated,name in ((False,'原代码_带说明.ipynb'),(True,'代码注释.ipynb')):
        result=copy.deepcopy(nb)
        result['cells']=[{'cell_type':'markdown','metadata':{},'id':'lesson06-overview','source':intro.splitlines(True)}]
        for index,cell in enumerate(nb['cells']):
            result['cells'].append({'cell_type':'markdown','metadata':{},'id':'lesson06-note-'+str(index),'source':note(index,cell).splitlines(True)})
            keep=copy.deepcopy(cell)
            if is_annotated and cell['cell_type']=='code':
                keep['source']=annotate(index,''.join(cell['source'])).splitlines(True)
            result['cells'].append(keep)
        (directory/name).write_bytes((json.dumps(result,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    doc='# 第06讲 Alpaca LoRA推理代码单元解释\n\n全部28格覆盖：18个代码格、10个Markdown格；安装、代理注释、交互与空格都保留原0起始编号。\n\n[主解释](解释.md) · [代码注释notebook](代码/tutorials/代码注释.ipynb) · [验证记录](复现/README.md)\n\n'
    py=['# -*- coding: utf-8 -*-','# 第06讲纯文本代码阅读版，无JSON和历史输出。',
        '# 源码安装指令保留为注释；7B加载/.cuda()/交互输入仍是原代码，勿直接整文件运行。',
        '# 使用notebook逐格核对环境；本次CPU教学替代及验证边界见复现/README.md。','']
    for index,cell in enumerate(nb['cells']):
        detail=note(index,cell)
        doc+=detail+'\n'
        py+=['# %% [markdown]']+['# '+line for line in detail.splitlines()]+['']
        original=''.join(cell['source'])
        if cell['cell_type']=='code':
            doc+='原代码：\n\n```'+('ipython' if index==0 else 'python')+'\n'+original+'\n```\n\n'
            py+=['# %% 原单元 {}'.format(index)]
            for line in annotate(index,original).splitlines():
                py.append('# IPython原指令：'+line if line.startswith('!') else line)
        else:
            py+=['# %% [markdown] 原单元 {}'.format(index)]+['# '+line for line in original.splitlines()]
        py+=['']
    (lesson/'代码单元解释.md').write_text(doc.rstrip()+'\n',encoding='utf-8')
    # 多行f字符串内部保持原行，不能对所有行统一rstrip以修改prompt空白。
    (directory/'06_alpaca_llama_7b_lora_代码注释.py').write_text('\n'.join('#' if line=='# ' else line for line in py).rstrip()+'\n',encoding='utf-8')
    print('All 28 cell notes and 18 code comments generated; two 57-cell notebooks and Python reading version.')


if __name__=='__main__':
    main()
