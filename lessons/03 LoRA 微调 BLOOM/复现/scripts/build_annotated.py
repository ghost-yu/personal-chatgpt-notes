"""47格逐格说明与37格中文代码注释；仅转换IPython语法，不执行原命令。"""
import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager

COMMENTS = {
 0:'IPython的!将命令交给shell安装依赖；Git默认分支未锁提交，本次核对不执行安装。',
 4:'导入Tensor/神经网络、量化加载、Auto接口及LoRA配置/注入；导入不等于加载7B。',
 5:'加载watermark扩展，要求包已安装；不是普通Python语法。',
 6:'打印当前导入模块版本；保留的历史报告不能当作本次环境。',
 7:'使用函数方式打印指定包的版本；返回字符串，不做模型计算。',
 8:'加载BLOOM因果模型和匹配分词器；可能下载权重并分配设备内存。',
 9:'读取结构配置；注释中的model.config不执行；这里不再次加载全部权重。',
 10:'notebook显示原模型repr，没有forward或训练；普通脚本不自动显示。',
 11:'返回输入Embedding模块，未对token ID查表，也未解冻权重。',
 12:'展示分词器对象；没有输入文本，所以尚未分词。',
 14:'只观察首个参数dtype，不能据此判断全模型精度。',
 15:'冻结所有现有权重，将一维参数转FP32；shape不变，不代表跳过反向链。',
 16:'开启激活重算并使embedding输出激活可求梯度；不解冻embedding权重。',
 17:'包装lm_head：先执行原head，再把[B,S,V]输出转FP32。',
 19:'定义独立统计函数；只读requires_grad/numel并打印，默认返回None。',
 20:'创建LoRA配置；r是瓶颈宽度，alpha/r=2是增量缩放；尚未执行注入。',
 21:'注入并包装目标模块，赋回model；随后统计可训练参数，不是深拷贝。',
 22:'显示注入后的模型结构，不能用repr代替训练结果或冻结状态核对。',
 25:'加载原名言数据，可能联网/写缓存；原ID未锁revision；本次不下载该数据。',
 26:'显示DatasetDict的split结构；还不是Tensor训练batch。',
 27:'按名称取得train Dataset，返回整个split，不是单行。',
 28:'转换为pandas表格以查看数据，可能将数据物化到内存。',
 29:'先取quote列，再切前四条完整文本；不是前四个token。',
 30:'取前四个作者字符串；作者没有参与后面的训练文本拼接。',
 31:'批取前四行，返回各字段对应列表的dict，不是list[dict]。',
 32:'将第一条tags列表转成Python展示字符串，非JSON或类别ID。',
 33:'拼接名言、固定分隔符和标签字符串；prediction是文本列名，不是模型预测。',
 34:'查看新增prediction列的前五条；要求merge/map已成功。',
 35:'按从0开始的索引取第五条单行dict；与切片[:4]返回形态不同。',
 36:'试分词前四条文本，默认返回列表形式BatchEncoding，未写回dataset。',
 38:'batched map传入字符串列表；添加token列；没有在此padding或truncation。',
 39:'展示分词后字段；原注释未执行；不意味着已经得到GPU Tensor。',
 41:'导入训练调度、训练参数和CLM批次构造接口；导入不会启动训练。',
 42:'构造Trainer后执行训练；会改变adapter和训练状态，原7B训练本次未运行。',
 44:'将第一条prompt分词为Tensor，生成最多50个新token，再decode；autocast不搬设备。',
 45:'更长的第二条prompt，路径与前格相同；不是独立评估集，原CUDA生成未运行。',
 46:'查看Trainer持有的collator；没有传samples，所以不真正构造batch。',
}
LINE_NOTES = {
 0:{'!pip install -q bitsandbytes':'安装量化、数据、训练分派与loralib依赖；后续LoRA实际走PEFT。',
    '!pip install -q git+https://github.com/huggingface/transformers':'从会变化的main安装Transformers。',
    '!pip install -q git+https://github.com/huggingface/peft':'从默认分支安装PEFT，缺少历史SHA。'},
 8:{'load_in_8bit=':'旧版bitsandbytes量化加载；不保证全部参数或激活都是int8。',
    'device_map=':'自动设备分派，不是数据并行副本数。',
    'tokenizer =':'单独加载词表与分词规则，模型权重不在tokenizer内。'},
 15:{'for i, param':'遍历Parameter并给出编号；i不参与计算。',
     'param.requires_grad':'关闭基权重的梯度累积，LoRA会在后面另加可训练参数。',
     'if param.ndim':'选择所有一维参数，不只LayerNorm。',
     'param.data =':'按原历史写法改底层存储dtype，shape保持；绕过一般autograd跟踪。'},
 16:{'model.gradient_checkpointing':'减少保存的激活，反向时重算，不是写磁盘checkpoint。',
     'model.enable_input':'开启embedding输出激活梯度，不改变其weight.requires_grad=False。'},
 17:{'class CastOutput':'继承Sequential，沿用其构造函数，原head成为子模块。',
     'def forward':'self是当前实例，x为[B,S,H]输入激活。',
     'return super':'调用父类运行head，输出[B,S,V]，只转换dtype。',
     'model.lm_head =':'替换模块属性，改变模块树；不是创建另一个词表。'},
 19:{'trainable_params =':'初始化可训练参数计数。','all_param =':'初始化全部参数计数。',
     'for _, param':'遍历名字和参数；下划线表示忽略名字。',
     'all_param +=':'numel为元素数，不是量化显存字节数。',
     'if param.requires_grad':'只累加可训练的Parameter。',
     'trainable_params +=':'累计当前可训练Parameter元素数。',
     'print(':'打印计数和比例；全空模型会除零，原函数未处理该边界。'},
 20:{'r=':'瓶颈维度16，对应因子的共同小维度。','lora_alpha=':'缩放分子32，标准scale=32/16=2。',
     'lora_dropout=':'训练时adapter分支输入dropout概率0.05。','bias=':'none表示不额外训练bias。',
     'task_type=':'因果语言模型任务；target_modules注释未生效，BLOOM默认映射融合QKV。'},
 33:{'def merge':'独立函数接收单行dict，没有默认参数。','row[':'给传入dict原地添加prediction字段；author不参与拼接。',
     'return row':'返回同一dict，供Dataset.map接收。','dataset[':'map返回新Dataset，再赋回train；其他split未在此处理。'},
 42:{'model=model':'使用已经注入adapter的model对象。','train_dataset=':'使用train split，未提供eval_dataset。',
     'per_device_train_batch_size=':'每设备每微批4条，不是有效batch总数。',
     'gradient_accumulation_steps=':'累计4个微批再更新，单设备完整周期有效batch=16。',
     'warmup_steps=':'前100个优化step预热。','max_steps=':'上限200次优化更新，不是200个样本。',
     'learning_rate=':'优化器学习率2e-4，与LoRA scale不同。','fp16=':'混合精度训练，非所有参数强制fp16。',
     'logging_steps=':'每次优化更新记录日志。','output_dir=':'训练文件写入outputs目录。',
     'data_collator=':'mlm=False：动态padding、创建labels，padding目标设-100。',
     'model.config.use_cache':'关训练生成缓存以配合checkpointing，不关闭autograd。',
     'trainer.train':'进入真实训练调度、前向、loss、反向、梯度累积与优化器更新。'},
 44:{'batch =':'return_tensors=pt产生[1,S]整数Tensor，原代码默认在CPU。',
     'with torch.cuda':'CUDA AMP上下文；不会把batch搬到CUDA或显式设置eval。',
     'output_tokens =':'**batch展开关键字输入，输出含prompt及最多50个新token。',
     'print(':'取第一行decode并跳过特殊token；通常仍含输入前缀。'},
 45:{'batch =':'第二条更长prompt使S变化；相同tokenizer和同一model。',
     'with torch.cuda':'仍是CUDA autocast，不自动搬运输入。',
     'output_tokens =':'生成上限仍为50个新增token，未组成独立验证集。',
     'print(':'decode结果为str，两个单例不构成准确率测量。'},
}

# 在原语句前解释输入/输出与调用链；原语句、缩进和历史输出保持不变。
DETAIL_NOTES = {
 0:['! 是 IPython 交给 shell 执行的前缀；-q 只减少日志，不改变安装版本。',
    '这三行会联网并修改环境；从 main/默认分支安装不等于2023年的历史dev版本。',
    '原命令未安装 watermark；loralib 被安装，但后续 adapter 实际由 PEFT 构造。'],
 4:['import torch 绑定模块；import torch.nn as nn 为神经网络模块指定别名。',
    'AutoTokenizer 加载分词规则；AutoConfig 加载结构配置；AutoModelForCausalLM 构造因果LM。',
    'LoraConfig 是配置类，get_peft_model 是注入函数；导入这两者尚未添加 adapter。',
    'bnb 是 bitsandbytes 的模块别名；本文件未直接调用 bnb，量化由加载路径触发。'],
 5:['%load_ext 是 IPython magic，加载已安装扩展到当前内核；不会安装缺失的 watermark。',
    '标准 Python 文件不能直接解析百分号语法；Python阅读版把此原指令记为注释。'],
 6:['--iversions 报告当前内核导入模块的版本；保存的旧输出不是当前运行环境。',
    '没有模型输入/输出或参数更新；版本字符串中的 dev0 仍不足以确定 Git 提交。'],
 7:['watermark 此时是从同名包导入的独立函数；packages 接收逗号分隔的 str。',
    '返回环境报告字符串，print 显示它；没有训练/评估调用，报告也未列 bitsandbytes。'],
 8:['【必须会】from_pretrained 是类方法；模型ID为 str，可触发配置/权重下载与设备内存分配。',
    '返回 model: nn.Module；原 BLOOM-7b1 的隐藏宽度 H=4096、词表 V=250880。',
    '尚无输入 token；加载模型不会自动得到 logits 或生成回答。',
    'tokenizer 是独立分词器对象；其ID及特殊token必须与预训练模型相容。',
    '原文未固定 model/tokenizer revision；本次CPU教学使用随机小模型，不执行本格7B加载。'],
 9:['AutoConfig.from_pretrained 返回模型结构配置对象，读取的是配置而不是全部权重。',
    '没有赋给变量，notebook会展示最后表达式；# model.config 原注释不执行。',
    '配置加载仍可能联网；不是纯粹访问已经存在的 model.config。'],
 10:['model 是前面创建的模块对象；单独写对象名字会让 notebook 显示 repr。',
     '没有输入 [B,S] token IDs，因此没有 forward、loss 或 logits；普通脚本不会自动显示。'],
 11:['get_input_embeddings() 是 model 的方法，无显式实参，返回输入词嵌入模块。',
     '调用返回模块而不是嵌入结果；真正查表时输入 int64 IDs [B,S]，输出浮点 [B,S,H]。',
     '原注释中的属性路径未执行；本调用也不改变 embedding.weight.requires_grad。'],
 12:['tokenizer 是分词器对象；本格只显示它的表示，没有文本实参。',
     '真正分词在36/38/44/45，调用 tokenizer(text) 才会创建 BatchEncoding。'],
 14:['parameters() 返回参数迭代器；list(...) 将其物化，索引[0]取得第一个 Parameter。',
     '.dtype 是类型属性；这不是全模型dtype统计，也不证明量化权重、激活和adapter精度一致。'],
 15:['param 是 Parameter；requires_grad=False 阻止其梯度累积，后面才加入可训练LoRA因子。',
     '冻结权重不等于关闭整条计算图，adapter学习仍需要梯度穿过模型中间运算。',
     'ndim 是维度数，一维参数也可能是 bias；这里不是只选择LayerNorm。',
     '.data 的历史写法直接改底层存储；to(float32) 保留shape，并不是优化器step。'],
 16:['两个调用均修改模型行为：激活checkpointing保存较少中间值，反向时重新计算。',
     'enable_input_require_grads 对embedding输出设置梯度hook；权重仍然被冻结。',
     '例如IDs [B,S] -> embedding激活 [B,S,H]；激活可求梯度和权重可训练是两件事。'],
 17:['class 定义子类；nn.Sequential 管理按顺序调用的子模块，包装器沿用父类构造函数。',
     'forward 是模型方法，self是当前实例；调用模块时PyTorch经__call__调度forward。',
     'super() 委托父类执行已有head，不是再次构造模型；[B,S,H] -> logits [B,S,V]。',
     '.to(torch.float32) 只转换输出dtype，没有把head权重整体转FP32。'],
 19:['model 是唯一实参，没有默认值；函数只统计，不修改参数或返回模型。',
     'named_parameters() 给出 (名字, Parameter)；numel() 统计元素数，不能当显存字节。',
     '函数定义后不会立即运行；21格调用它，print之后隐式返回None。'],
 20:['config 是 LoraConfig 实例，没有token数据或训练张量；后面get_peft_model才使用配置。',
     '融合QKV输入 [B,S,H]；标准LoRA分支经 A:[r,H]、B:[3H,r] 后输出 [B,S,3H]。',
     '图与库采用nn.Linear权重[out,in]布局，不能混用上一讲行向量因子的存储方向。',
     'target_modules 行是注释；核对版本中BLOOM默认映射为 query_key_value，不是 q_proj/v_proj。'],
 21:['返回外层PEFT模型包装，并在传入基模型的目标层加入adapter；不会深拷贝全部权重。',
     '原大权重保留并冻结，新增A/B参与训练；此处配置r=16，标准缩放alpha/r=32/16=2。',
     '以原H=4096为例：A权重[16,4096]、B权重[12288,16]；单层因子共262144元素。'],
 22:['注入后仍需用requires_grad和真实前向核对；结构repr不能代替这些检查。',
     '同一个模型以后接收token输入、经BLOOM计算logits，adapter只是选定投影上的增量。'],
 25:['load_dataset 是独立函数；模型与数据加载相互独立，不会因为load_dataset而训练模型。',
     '原ID是 str，返回DatasetDict，train是一个split；未固定revision，样本快照可能变化。',
     '本次教学用明确新编的5条本地数据，未下载原Abirate数据。'],
 26:['dataset 是 DatasetDict，按split名映射到Dataset；本格只显示结构。',
     '此时字段如 quote/author/tags 是数据记录，不是GPU Tensor。'],
 27:['字典键train选择一个split，返回整个Dataset；还没有选样本或字段。',
     '不是随机划分、不是取batch，也没有把全部数据转换为Tensor。'],
 28:['to_pandas 是Dataset的方法，无显式参数；返回 pandas.DataFrame。',
     '这是浏览数据的表格，可能物化到内存；后面的Trainer仍接收Dataset，不接收这个表格。'],
 29:['先按列名quote取得字符串列，再[:4]按行取前四条，结果为list[str]。',
     '切片上界4不包含索引4；字符串内容保留，不是切前四个token。'],
 30:['author列的前四个字符串仅用于查看；后面merge不把author拼入训练文本。',
     '知道作者是谁与模型训练目标不是同一个字段，不能据此宣称训练作者分类。'],
 31:['Dataset切片[:4]返回列名 -> 值列表的dict，而不是四个单行dict组成的list。',
     '例如 quote:list[str]、tags:list[list[str]]；后面batched map也使用列列表字典。'],
 32:['tags[0] 是第一条标签列表；str(list) 包含方括号和引号，成为文本续写目标的一部分。',
     '它不是JSON序列化，也不是每个标签对应一个固定分类类别ID。'],
 33:['row:dict 来自Dataset的一条记录；quote:str，tags:list[str]，函数没有默认参数。',
     '输入名言和标签经字符串拼接形成prediction:str；prediction不是model的推理结果。',
     '返回同一个被添加字段的row；外层Dataset.map返回新Dataset，赋回train split。'],
 34:['prediction列为list[str]；前五条用于核对merge的格式、分隔符和标签列表文本。',
     '不会调用generate，也没有评估预测与真实标签是否相同。'],
 35:['[4] 是第5条单行记录，返回字段 -> 单个值的dict；不同于[:4]的列列表字典。',
     '依赖数据至少有5行；新编CPU教学数据也安排5行，以实际覆盖这个索引。'],
 36:['tokenizer.__call__ 输入list[str]，返回BatchEncoding映射；默认input_ids等仍为Python列表。',
     '未指定return_tensors、padding、truncation；各行长度可能不同，也不主动截断超长输入。',
     '返回值未赋回Dataset；这里只试分词，38才正式写入token列。'],
 38:['lambda定义匿名函数，samples为一批记录的列列表字典；函数返回分词后的字段映射。',
     'batched=True 是map处理模式，不是Trainer微批大小4；DatasetDict会对每个split执行map。',
     '添加input_ids:list[int]、attention_mask:list[int]等字段；原文本列默认保留。',
     '若额外split没有prediction，本匿名函数会KeyError；原文只给train做了merge。'],
 39:['注释中的字段名只帮助阅读；实际表达式dataset显示token列和split结构。',
     '本格不构造 [B,S] Tensor；转换和补齐发生在后面collator。'],
 41:['Trainer是训练控制器，TrainingArguments是参数对象；collator是样本到batch的转换器。',
     'CLM的token位移与交叉熵由BLOOM内部实现，不是在这条import里计算。'],
 42:['【必须会】样本列表 -> collator -> int64 input_ids/attention_mask/labels [B,Smax]。',
     'mlm=False：复制input_ids作为labels，padding目标设-100，让交叉熵忽略补齐位置。',
     'BLOOM输出logits [B,S,V]；logits[:,:-1,:]与labels[:,1:]配对，预测下一token。',
     '原目标对整段非padding文本计算loss，没有只保留回答标签段的mask。',
     '单设备完整累积周期的有效batch=4*4=16；多卡数量不能由device_map=auto直接推断。',
     '原200步指optimizer更新数；Trainer负责梯度累积、反向与优化器step。',
     '本格构造的trainer会持有model/dataset/collator，train会改adapter及训练状态并可能写文件。',
     'CPU验证另设随机小模型、fp16=False、微批2/累积2/2步；未执行原7B的FP16 200步训练。'],
 44:['batch:BatchEncoding，以pt输出int64 input_ids/attention_mask [1,S]，默认仍在CPU。',
     'with 是上下文管理；CUDA autocast只控制部分浮点计算精度，不负责输入设备搬运。',
     '**batch 展开关键字实参；generate输出整数token序列 [1,S+G]，G最多50。',
     'decode的输入是一行token IDs，返回str；skip_special_tokens跳过特殊ID，通常仍含prompt。',
     '原文没有显式eval()/inference_mode；此处注释不添加执行逻辑，原CUDA生成未复现。'],
 45:['第二条prompt更长，所以输入长度S与第一条不同；仍使用同一model与tokenizer。',
     '输出上限是新增50个token，不是总长度50；更长prompt不会自动产生独立验证集。',
     'autocast、**batch、decode与上一格一致；这两条生成不能证明名言标签任务准确率。'],
 46:['trainer.data_collator 是可调用批次构造对象；本格仅取属性并显示repr，返回该对象。',
     '需要另调用collator(samples)才会padding并建立labels；查看属性不能证明训练已成功。'],
}


def source(cell):
    value=cell.get('source','')
    return ''.join(value) if isinstance(value,list) else value


def ast_signature(text):
    # 转换!/%语法只是字符串变换，绝不eval/exec转换出来的安装命令。
    converted=TransformerManager().transform_cell(text)
    return ast.dump(ast.parse(converted))


def annotate(text,index):
    lines=['# '+COMMENTS[index]+'\n']
    lines.extend('# '+row+'\n' for row in DETAIL_NOTES[index])
    for line in text.splitlines(keepends=True):
        for prefix,comment in LINE_NOTES.get(index,{}).items():
            if line.strip().startswith(prefix):
                indent=line[:len(line)-len(line.lstrip())]
                lines.append(indent+'# '+comment+'\n')
                break
        lines.append(line)
    return ''.join(lines)


def ordinary_python(text):
    # 原文件仅有独立行的shell/magic；只在阅读版注释掉，notebook原指令不变。
    return '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith(('!', '%')))


def export_readable(notebook, target):
    parts=['# 第03讲 PEFT / BLOOM 代码注释阅读版\n'
           '# 来源：chunhuizhang/personal_chatgpt，提交 7fbc29824171fd07dc2c30fe922e61ebc018ad1d。\n'
           '# 无JSON或历史输出；# %% 分格。建议在配套notebook中按单元检查依赖后执行。\n'
           '# IPython的!/%指令记为注释，普通Python语句保留；不是独立CPU训练脚本。\n'
           '# 离线CPU教学调用链见复现/scripts/verify_pipeline.py。\n\n']
    for cell in notebook['cells']:
        text=source(cell)
        if cell['cell_type']=='code':
            lines=[]
            for line in text.splitlines():
                if line.lstrip().startswith(('!', '%')):
                    line='# IPython原指令（在notebook中执行）：'+line
                lines.append(line.rstrip())
            parts.append('# %%\n'+'\n'.join(lines).rstrip()+'\n\n')
        else:
            parts.append('# %% [markdown]\n'+'\n'.join(('# '+line).rstrip() for line in text.splitlines())+'\n\n')
    Path(target).write_text(''.join(parts).rstrip()+'\n',encoding='utf-8')


def notebook_note(text):
    return re.sub(r'\]\(([^)]+)\)',lambda m: ']('+(m.group(1) if '://' in m.group(1)
                  or m.group(1).startswith('#') else '../../'+m.group(1))+')',text)


def generate(notebook,explanation,output_dir):
    raw=Path(notebook).read_bytes()
    original=json.loads(raw.decode('utf-8-sig'))
    cells=original['cells']
    assert len(cells)==47
    code=[i for i,c in enumerate(cells) if c['cell_type']=='code']
    assert set(code)==set(COMMENTS)==set(DETAIL_NOTES) and len(code)==37
    text=Path(explanation).read_text(encoding='utf-8')
    matches=list(re.finditer(r'^## 单元 (\d+)：.*$',text,re.M))
    notes={}
    for pos,m in enumerate(matches):
        end=matches[pos+1].start() if pos+1<len(matches) else text.index('## 覆盖、学习优先级')
        notes[int(m.group(1))]=text[m.start():end].strip()
    assert set(notes)==set(range(47))
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    report={'original_cells':47,'explained_cells':sorted(notes),'code_cells':code,
            'commented_code_cells':sorted(COMMENTS),'source_sha256':hashlib.sha256(raw).hexdigest(),
            'ipython_syntax_transform_only':True,'original_commands_executed':False,'artifacts':{}}
    for filename,commented in [('原代码_带说明.ipynb',False),('代码注释.ipynb',True)]:
        result=copy.deepcopy(original)
        result['cells']=[{'cell_type':'markdown','metadata':{},'id':'reading-header',
                          'source':'# 第03讲中文代码说明\n\n新增注释只解释原行为；原输出是历史记录。\n'
                                   '原文件另存。安装/模型下载/原7B训练没有因本注释文件生成而执行。CPU补充验证见复现README。'}]
        for i,cell in enumerate(cells):
            result['cells'].append({'cell_type':'markdown','metadata':{},'id':'explain-%02d'%i,'source':notebook_note(notes[i])})
            current=copy.deepcopy(cell)
            if commented and i in COMMENTS:
                current['source']=annotate(source(cell),i)
                assert ast_signature(source(cell))==ast_signature(source(current)),i
            else:
                assert current==cell
            result['cells'].append(current)
        nbformat.validate(nbformat.from_dict(result))
        target=output_dir/filename
        target.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        report['artifacts'][filename]={'cells':len(result['cells']),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        if commented:
            export_readable(result, output_dir/'03_lora_practice_代码注释.py')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--notebook',required=True)
    parser.add_argument('--explanation',default=str(Path(__file__).resolve().parents[2]/'代码单元解释.md'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--record',required=True)
    args=parser.parse_args()
    report=generate(args.notebook,args.explanation,args.output_dir)
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('generated: 47 explained, 37 commented, schema and transformed AST passed')
