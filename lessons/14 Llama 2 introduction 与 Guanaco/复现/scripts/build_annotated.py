# -*- coding: utf-8 -*-
"""23格/16代码格与Markdown15全部shell命令详注；原语句/历史输出保留。"""
import copy
import json
import re
from pathlib import Path

NOTES={
0:('基础概念入口','【必须会】区分base/chat、权重格式与推理/训练两条路径；下方生成来自chat权重，不是后面Guanaco微调后的效果。'),
1:('发布入口与上下文','【必须会】7B/13B/70B规模×原始/HF格式×base/chat，共12入口；hf是格式/生态，不是能力等级。4K指输入与生成的总token预算。官方blog/模型页为来源链接，不自动执行或获取权限；7B/13B不能混称70B的GQA设置。'),
2:('文本生成章节','【必须会】字符串→分词ID→chat模型generate→decode。pipeline默认返回完整prompt加生成后续，普通字符串不自动变成INST模板。'),
3:('导入分词、模型、量化配置与Torch','【必须会】AutoTokenizer/AutoModelForCausalLM为工厂类，transformers和torch为模块。BitsAndBytesConfig导入不自动量化；本notebook推理没有用它加载4bit。导入不会下载7B。'),
4:('保存chat模型Hub标识','【必须会】字符串指向7b-chat-hf；训练Markdown15使用7b-hf基础模型，是两个checkpoint。本次原格保留执行，CPU上下文另把model_name指向同一随机小模型的本地目录。'),
5:('作者的本地代理','【建议会】HTTP_PROXY/HTTPS_PROXY写进程网络配置，7890应有真实代理；不是GPU配置。本次跳过，不修改用户代理环境。'),
6:('tokenizer与生成pipeline加载','【必须会】from_pretrained返回分词对象；pipeline工厂返回TextGenerationPipeline，内部还会自己加载tokenizer，原代码未传外部tokenizer。torch_dtype FP16是浮点加载，device_map auto按容量分配模块，不是4bit或DDP。本次CPU变体只把FP16→FP32、auto→None，并在上下文提供本地随机小模型路径；真实对象与原7B分开。'),
7:('显示外部分词对象','【建议会】表达式显示词表和special tokens，不执行模型。外部tokenizer与pipeline.tokenizer是不同实例，改外部pad不自动同步内部。'),
8:('显示pipeline模型','【必须会】访问真实model属性，repr展示模块树而不forward。原历史LLaMA hidden4096、32层、vocab32000；小模型结构另记录。'),
9:('统计参数与requires_grad','【必须会】named_parameters给(name,Parameter)，numel数元素；if requires_grad只计允许求导参数；+=累加。1.强制浮点比例。原6,738,415,616和比值1不等于正在训练；pipeline推理没有optimizer更新。4bit打包权重的存储numel另有语义，不能把7B误称3.5B。本次真实小模型统计，另用meta结构核对7B逻辑计数。'),
10:('英文随机生成三候选','【必须会】调用pipeline实例的__call__。do_sample随机采样；top_k每步候选10；num_return_sequences3是同prompt三个候选，不是三轮聊天。eos为停止ID；max_length200包括输入而不是新增200。字符串普通推荐问句，没有INST。CPU小模型整格保留这些生成参数，随机输出不当聊天能力。'),
11:('打印英文候选','【必须会】sequences为list[dict]，for逐个取seq，[generated_text]取字符串，+两个换行排版。默认full_text包含prompt；print不运行新模型。原历史样例保留，CPU随机输出另记录。'),
12:('中文随机生成和打印','【必须会】Unicode中文prompt同样先分词，字符数不等token数；三个候选、top_k10、总长200与英文格相同。异常续写不能仅凭样例归因于中文词表损坏或微调失败。原格真实CPU执行，结果只核对流程及注释等价。'),
13:('微调章节入口','【必须会】切换到base权重与Guanaco监督数据；不能把前面的chat生成当作后续微调评估。'),
14:('TRL训练器入口','【建议会】SFTTrainer管理数据准备/训练，不自动只监督Assistant；需读脚本和collator默认值。'),
15:('安装、clone与完整微调命令','【必须会】三条shell逻辑命令：pip install trl安装库；git clone取示例；python运行示例。原未锁tag，Bash反斜杠续行不能直接照搬PowerShell；注释不插入续行链中，避免破坏语义。原flags模型base7b、Guanaco数据、4bit、PEFT、micro16、累积2，有效单卡batch32。固定参考TRL0.5示例默认seq512、epoch3、r64/alpha16缩放.25、packingFalse；作者改稿未找到。原4090约5小时与字幕3090矛盾，是估计非实测。本次解析原命令并在本地明确CPU参数下执行真实固定示例，不执行原pip/clone或7B量化。9846*3/(16*2)实际923.0625，按每epoch616micro/308更新才得到924。'),
16:('Guanaco数据章节入口','【必须会】原timdettmers/openassistant-guanaco，不是别的同名1k/INST重排子集；text是多轮对话字符串。'),
17:('导入数据加载函数','【建议会】模块函数load_dataset，导入不下载数据。本次原格完整执行，加载用固定五行本地样本变体另记录。'),
18:('从Hub取训练分区','【必须会】数据集标识与split=train，返回Dataset。原历史9846行；本次不重新取全量，CPU变体改为json/local文件读取已有固定revision五行。样本文件来源和hash另核对，不把五行当全体。'),
19:('查看数据集','【建议会】独立表达式显示列和行数，不自动训练。原9846与教学5行是不同范围。'),
20:('打印首行完整text','【必须会】索引0取字典，text取多轮字符串。原首行Human→Assistant→Human，结尾请求没有回答；不能认为每一行都是一问一答。'),
21:('按###拆字符串','【必须会】str.split返回list[str]，每项没有###，首项可能空；for/print只是显示。内容内出现###也会被误拆，不是可靠messages解析器或loss mask。真实分词后的监督边界要另核对。'),
22:('粗算优化步数','【必须会】前两行#只是解释；/产生float923.0625，不自动向上取整。需先ceil(9846/16)=616micro/epoch，再累积2得308更新×3=924；最后有效窗口只有22样本，非32。本次原表达式保留执行；未启动原924更新。'),
}

COMMENTS={
'from transformers import AutoTokenizer, AutoModelForCausalLM':'导入按配置选分词/因果LM实现的工厂类。',
'import transformers':'导入模块，后续pipeline是模块函数。',
'from transformers import BitsAndBytesConfig':'量化配置类，本notebook推理加载没有使用它。',
'import torch':'导入Tensor/自动求导模块，不启动训练。',
'model_name = "meta-llama/Llama-2-7b-chat-hf"':'chat推理权重标识字符串，与后续base训练模型不同。',
'import os':'导入进程环境操作模块。',
"os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'":'当前进程HTTP代理，需本地真实服务；本次跳过该格。',
"os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'":'HTTPS代理设置，不是GPU或模型参数。',
'tokenizer = AutoTokenizer.from_pretrained(model_name)':'类方法加载外部分词对象，不加载7B模型权重。',
'pipeline = transformers.pipeline(':'工厂函数构造TextGenerationPipeline，内部持有model/tokenizer。',
'"text-generation",':'任务名选择生成pipeline，不自动构造聊天模板。',
'model=model_name,':'按标识加载模型；原没传外部tokenizer，所以内部另加载分词对象。',
'torch_dtype=torch.float16,':'原FP16浮点权重加载，不是4bit量化；CPU变体另改FP32。',
'device_map="auto"':'按容量分配模块，不自动DDP；CPU变体另改None。',
'tokenizer':'显示外部分词对象，可能与pipeline内部实例状态不同。',
'pipeline.model':'访问生成pipeline持有的模型，不执行forward。',
'trainable_params = 0':'初始化允许求导参数的元素计数。',
'all_param = 0':'初始化全部参数元素计数。',
'for name, param in pipeline.model.named_parameters():':'遍历参数名/Parameter；不会更新权重。',
'all_param += param.numel()':'numel计元素并累加，量化打包容器需另辨认逻辑shape。',
'if param.requires_grad:':'判断是否允许求导，不表示已有loss或optimizer。',
'trainable_params += param.numel()':'只将允许求导参数加入分子。',
'print(trainable_params, all_param, trainable_params*1./all_param)':'打印元素计数及float比例；推理中比值1也不执行训练。',
'sequences = pipeline(':'调用真实pipeline对象，返回候选list[dict]。',
'\'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\\n\',':'普通英文prompt，没有[INST]模板；输入长度计入200token预算。',
"'我比较喜欢《肖申克的救赎》、《星际穿越》这样的电影，你有其他电影推荐吗？\\n',":'中文字符串先分词，字符长度不同于token长度。',
'do_sample=True,':'每一步按概率随机抽样，需固定seed才能受控比较。',
'top_k=10,':'只从该步分数最高的10个token中采样，不是输出10条回答。',
'num_return_sequences=3,':'同一prompt三份候选，不是三轮历史对话。',
'eos_token_id=tokenizer.eos_token_id,':'外部分词对象提供EOS停止ID，原Llama2为2。',
'max_length=200,':'输入加输出总长上限200，而不是新增200。',
'for seq in sequences:':'逐个取候选字典用于显示。',
"print(seq['generated_text'] + '\\n\\n')":'取默认完整文本并加换行排版，prompt重复不必是模型再次生成。',
"print(seq['generated_text'])":'显示中文候选；CPU随机模型结果不是语言质量评价。',
'from datasets import load_dataset':'导入HF数据加载模块函数，不在import时下载。',
"dataset = load_dataset('timdettmers/openassistant-guanaco', split='train')":'原Hub训练分区，历史9846行；本次五行本地变体另记。',
'dataset':'显示Dataset结构，不训练/分词所有样本。',
"print(dataset[0]['text'])":'取首行多轮text并打印，末尾Human不一定已有回答。',
"for part in dataset[0]['text'].split('###'):":'字符串分隔后遍历，非tokenizer/messages解析或监督mask。',
'print(part)':'显示分隔片段，可能包含空首项。',
'9846*3/(16*2*1)':'普通除法得923.0625；实际924更新需按每epoch批次取整推导。',
}


def annotate(src,i):
    result=['# '+NOTES[i][0]+'；原语句保留，实际CPU范围另记。\n']
    for line in src.splitlines(True):
        key=line.strip()
        if key and not key.startswith('#') and key!=')':
            assert key in COMMENTS,key
            result.append(line[:len(line)-len(line.lstrip())]+'# '+COMMENTS[key]+'\n')
        result.append(line)
    return ''.join(result)


def shell_note(src):
    result=[]
    for line in src.splitlines(True):
        key=line.strip()
        if key=='pip install trl': result.append('# 安装TRL库，原命令未锁版本；本次使用独立固定环境。\n')
        elif key.startswith('git clone '): result.append('# 取示例仓库，原未固定tag；不要假定与已安装库同版本。\n')
        elif key.startswith('python trl/'):
            result += ['# Python解释器 + 示例路径；下方反斜杠是Bash续行，PowerShell语法不同。\n',
                       '# --model_name：base 7b-hf，不是前面的chat checkpoint。\n',
                       '# --dataset_name：原始Guanaco多轮text字段。\n',
                       '# --load_in_4bit：量化基础权重；--use_peft：训练LoRA。\n',
                       '# --batch_size16与--gradient_accumulation_steps2：单卡有效batch32。\n',
                       '# 没写seq_length/epochs/r/alpha：参考v0.5.0默认512/3/64/16。\n']
        result.append(line)
    return ''.join(result)


def note(i,c): return '## 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],*NOTES[i])


def main():
    root=Path(__file__).resolve().parents[2];folder=root/'代码/tutorials'
    raw=json.loads((folder/'llama2_introduction.ipynb').read_bytes())
    assert len(raw['cells'])==23 and set(NOTES)==set(range(23))
    for name,changed in [('原代码_带说明.ipynb',False),('代码注释.ipynb',True)]:
        nb=copy.deepcopy(raw);intro='# 第14讲原代码说明\n\n全部23格/16代码格及Markdown15三条shell命令。原历史输出保留，本次小模型并非Llama2聊天能力/7B量化训练。\n'
        nb['cells']=[dict(cell_type='markdown',metadata={},id='lesson14-intro',source=intro.splitlines(True))]
        for i,c in enumerate(raw['cells']):
            nb['cells'].append(dict(cell_type='markdown',metadata={},id='lesson14-note-'+str(i),source=note(i,c).splitlines(True)))
            keep=copy.deepcopy(c);src=''.join(c['source'])
            if changed and c['cell_type']=='code':keep['source']=annotate(src,i).splitlines(True)
            elif changed and i==15:keep['source']=shell_note(src).splitlines(True)
            nb['cells'].append(keep)
        (folder/name).write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    doc='# 第14讲代码单元解释\n\n全部23格/16代码格与Markdown15三条shell命令；编号从0开始。\n\n[主解释](解释.md) · [注释notebook](代码/tutorials/代码注释.ipynb) · [真实验证](复现/README.md)\n\n'
    py=['# -*- coding: utf-8 -*-','# 第14讲Python阅读版，保留原代理/远程7B/FP16设置。','# CPU验证及外部训练示例运行改动见复现README。','']
    for i,c in enumerate(raw['cells']):
        src=''.join(c['source']);desc=note(i,c);doc+=desc+'\n'
        py+=['# %% [markdown]']+['# '+x for x in desc.splitlines()]+['']
        if c['cell_type']=='code':
            doc+='原代码：\n\n```python\n'+src+'\n```\n\n'
            py+=['# %% 原单元 '+str(i)]+annotate(src,i).splitlines()+['']
        elif i==15:
            shown=shell_note(src);doc+='原Markdown命令与历史估计（命令前加入注释，未执行原shell命令）：\n\n'+shown+'\n'
            py+=['# %% [markdown] 原单元15 shell命令']+['# '+x for x in shown.splitlines()]+['']
        else:py+=['# %% [markdown] 原单元 '+str(i)]+['# '+x for x in src.splitlines()]+['']
    (root/'代码单元解释.md').write_bytes(('\n'.join(x.rstrip() for x in doc.splitlines()).rstrip()+'\n').encode('utf-8'))
    (folder/'llama2_introduction_代码注释.py').write_bytes(('\n'.join(x.rstrip() for x in py).rstrip()+'\n').encode('utf-8'))
    print('23 notes/16 code and Markdown15 shell comments, two47-cell notebooks and reading Python built.')


if __name__=='__main__': main()
