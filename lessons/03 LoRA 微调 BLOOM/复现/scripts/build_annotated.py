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


def source(cell):
    value=cell.get('source','')
    return ''.join(value) if isinstance(value,list) else value


def ast_signature(text):
    # 转换!/%语法只是字符串变换，绝不eval/exec转换出来的安装命令。
    converted=TransformerManager().transform_cell(text)
    return ast.dump(ast.parse(converted))


def annotate(text,index):
    lines=['# '+COMMENTS[index]+'\n']
    for line in text.splitlines(keepends=True):
        for prefix,comment in LINE_NOTES.get(index,{}).items():
            if line.strip().startswith(prefix):
                indent=line[:len(line)-len(line.lstrip())]
                lines.append(indent+'# '+comment+'\n')
                break
        lines.append(line)
    return ''.join(lines)


def generate(notebook,explanation,output_dir):
    raw=Path(notebook).read_bytes()
    original=json.loads(raw.decode('utf-8-sig'))
    cells=original['cells']
    assert len(cells)==47
    code=[i for i,c in enumerate(cells) if c['cell_type']=='code']
    assert set(code)==set(COMMENTS) and len(code)==37
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
            result['cells'].append({'cell_type':'markdown','metadata':{},'id':'explain-%02d'%i,'source':notes[i]})
            current=copy.deepcopy(cell)
            if commented and i in COMMENTS:
                current['source']=annotate(source(cell),i)
                assert ast_signature(source(cell))==ast_signature(source(current)),i
            else:
                assert current==cell
            result['cells'].append(current)
        nbformat.validate(nbformat.from_dict(result))
        target=output_dir/filename
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        report['artifacts'][filename]={'cells':len(result['cells']),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
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
