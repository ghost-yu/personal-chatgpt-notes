"""解释18格，并对8格Python和6段Markdown源码仅加中文注释。"""
import argparse
import ast
import copy
import hashlib
import json
import re
import textwrap
from pathlib import Path
import nbformat

CODE_NOTES={
    9:'使用包中完整LoraConfig构造实例，非Markdown节选；r16、alpha32、scale2。',
    11:'导入三个Auto类；本文件实际仅调用AutoModelForCausalLM加载。',
    12:'原7B量化加载可能下载权重/占用设备；本次未运行，不是CPU教学模型构造。',
    13:'展示裸模型repr；没有forward、复制或注入。',
    14:'导入包级普通函数get_peft_model；导入尚不修改模型。',
    15:'包装并替换内部目标子层；复用旧基权重对象，不是深拷贝。',
    16:'展示注入后结构；不能证明已经训练或保存adapter。',
    17:'原单元为空，没有运算；保留原元数据和输出，仅加入此说明注释。',
}
PREFIX_NOTES={
    'r=':'瓶颈维度16，对应两个小因子的公共维度。',
    'lora_alpha=':'缩放分子32，标准alpha/r=2。',
    'lora_dropout=':'adapter支路训练dropout概率0.05。',
    'bias=':'不额外训练偏置，但保留旧bias参与前向。',
    'task_type=':'按CAUSAL_LM选择任务包装；target_modules注释不生效。',
    'load_in_8bit=':'旧式bitsandbytes量化加载，不保证全部参数/激活都是int8。',
    'device_map=':'auto自动设备分派，不等于数据并行world_size。',
    'MODEL_TYPE_TO_PEFT_MODEL_MAPPING':'按任务类型选外层包装，值是类对象而不是模型实例。',
    'PEFT_TYPE_TO_MODEL_MAPPING':'按适配方法枚举选实现类；需要各类名/PeftType预先导入。',
    '"SEQ_CLS"':'序列级分类任务包装。',
    '"SEQ_2_SEQ_LM"':'编码器-解码器语言模型任务包装。',
    '"CAUSAL_LM"':'因果语言模型任务包装，BLOOM使用此分支。',
    '"TOKEN_CLS"':'逐token分类任务包装。',
    'PeftType.LORA':'低秩更新的注入与adapter管理。',
    'PeftType.PROMPT_TUNING':'可训练软prompt embedding，不是LoRA矩阵。',
    'PeftType.P_TUNING':'prompt编码器实现。',
    'PeftType.PREFIX_TUNING':'prefix编码器实现，与低秩权重支路不同。',
    'PeftType.ADALORA':'自适应低秩分配方法。',
    'PeftType.ADAPTION_PROMPT':'另一种提示适配方法，非本例LORA路径。',
    'self.base_model =':'保存原模型引用；完整构造后还会进一步包装为LoraModel。',
    'self.config =':'复用原配置引用，不复制权重。',
    'self.active_adapter =':'记录adapter名字，通常default，不是CAUSAL_LM任务名。',
    '@dataclass':'装饰器生成初始化方法；这段仅是缺字段的节选。',
    'class LoraConfig':'继承PeftConfig；要执行还需要导入dataclass/field/typing/父类。',
    'r: int':'节选默认r=8，与下面实际实例r=16不同。',
    'target_modules:':'Optional只说明类型；这里漏掉默认field，装饰时会报默认字段顺序错误。',
    'lora_alpha:':'节选默认None，不能不核对完整配置就直接拿来做缩放。',
    'lora_dropout:':'节选默认None，不等于概率0；help metadata只提供说明。',
    'TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING':'按架构选择默认模块名；query_key_value不是所有模型的默认。',
    'key_list =':'named_modules提供路径和模块；下划线忽略对象，先保存路径快照。',
    'for key in':'遍历已有完整模块路径；此节选仅匹配，不执行替换。',
    'if isinstance':'字符串目标按完整路径正则匹配；列表目标进入else。',
    'target_module_found = re.fullmatch':'返回Match或None；裸query_key_value不会匹配带前缀路径。',
    'else:':'列表目标通过后缀检查，与完整正则语义不同。',
    'target_module_found = any':'生成器逐个检查后缀；any得到布尔值。',
}
MODEL_DESCRIPTIONS={
    't5':'T5的q/v投影','mt5':'mT5的q/v投影','bart':'BART的分离q_proj/v_proj',
    'gpt2':'GPT-2融合c_attn投影（Transformers Conv1D）','bloom':'BLOOM融合query_key_value',
    'blip-2':'BLIP-2不同子模块的q/v和q_proj/v_proj',
    'opt':'OPT分离q_proj/v_proj','gptj':'GPT-J分离q_proj/v_proj',
    'gpt_neox':'GPT-NeoX融合query_key_value','gpt_neo':'GPT-Neo分离q_proj/v_proj',
    'bert':'BERT的query/value','roberta':'RoBERTa的query/value',
    'xlm-roberta':'XLM-R的query/value','electra':'ELECTRA的query/value',
    'deberta-v2':'DeBERTa-v2的query_proj/value_proj','deberta':'DeBERTa融合in_proj',
    'layoutlm':'LayoutLM的query/value','llama':'LLaMA分离q_proj/v_proj',
    'chatglm':'ChatGLM融合query_key_value',
}
FENCE=re.compile(r'(```[^\n]*\n)(.*?)(```)',re.S)

CODE_DETAILS={
    9:['【必须会】from peft import LoraConfig 导入包中的完整配置类，并未执行Markdown7的节选。',
       'LoraConfig(...) 创建配置对象config，无输入token、logits或参数梯度。',
       '原实例r=16、alpha=32，标准缩放为2；dropout和bias策略由后面注入层使用。',
       'target_modules那行是注释，没有选择q_proj/v_proj；BLOOM默认选择融合query_key_value。'],
    11:['from transformers import ... 把三个Auto类绑定到当前命名空间。',
        'Auto类按配置选择具体实现；本notebook后面只调用AutoModelForCausalLM。',
        '导入另外两类不等于已加载tokenizer或另一次AutoConfig下载。'],
    12:['from_pretrained 是类方法，接收str模型ID与加载选项，返回因果语言模型模块。',
        '可能联网读取配置和checkpoint、写缓存、分配设备内存；原文未固定revision。',
        '原BLOOM-7b1隐藏宽度H=4096，融合QKV权重为[12288,4096]（nn.Linear的[out,in]方向）。',
        'load_in_8bit是历史量化接口；后续可能进入Linear8bitLt，而CPU普通Linear验证不能覆盖该路径。',
        '尚未传入int64 token IDs [B,S]；这行模型加载不是forward或generate。'],
    13:['末尾表达式返回当前裸模型对象，notebook显示repr；普通Python不自动显示。',
        '没有输入Tensor，所以不计算[B,S,V] logits，也不复制/冻结/替换权重。'],
    14:['get_peft_model 是包级独立函数，不是 model.get_peft_model 实例方法。',
        '导入只绑定函数名字；真正调用在15格，输入是模型模块与PEFT配置。'],
    15:['【必须会】get_peft_model(model,config) 接收已有模型对象和LoraConfig，返回PEFT任务包装。',
        '赋回model重新绑定名字；原BLOOM对象保留，在内部被LoraModel引用，没有深拷贝整网。',
        'task_type选择PeftModelForCausalLM，peft_type选择LoraModel，model_type=bloom选择默认目标名。',
        '官方_find_and_replace扫描模块树，_replace_module通过setattr替换父模块的子层属性。',
        '新层复用原weight/bias Parameter，增加A/B小因子；bias=none仍保留旧偏置参与前向。',
        '冻结基参数而训练lora_参数；这不是关闭整条计算图或已经执行优化器step。',
        '原H=4096时A:[16,4096]、B:[12288,16]，输入[B,S,4096]的增量输出[B,S,12288]。',
        '本次实际验证为官方v0.3.0、小型CPU普通Linear路径；没有运行本格对应的原7B量化注入。'],
    16:['外层为PeftModelForCausalLM，中间base_model为LoraModel，其model指向原BLOOM。',
        '这些是包装层次，不是三份完整模型；结构repr无法证明requires_grad或权重已训练。',
        '本格没有loss/backward/optimizer，也没有保存adapter。'],
    17:['原source为空；注释版仅加注释，AST仍为空Module，不新增训练代码或模型结果。'],
}
MARKDOWN_DETAILS={
    2:[['【必须会】这是任务类型到包装类的dict，依赖这些类名已导入；Markdown本身不执行。',
        '字符串键对应task_type，值是类对象而不是构造好的模型；这里并非bloom/llama架构映射。'],
       ['这是适配方法到实现类的dict，键是PeftType枚举成员，值仍是类对象。',
        'LORA、PROMPT_TUNING和PREFIX_TUNING管理不同形式的参数，不能都叫低秩权重矩阵。']],
    4:[['【必须会】这是方法内部的初始化节选，需要已有self、model和adapter_name。',
        'self是正在构造的包装实例；三条赋值保存引用与名字，无Tensor计算或权重复制。',
        '完整PeftModel构造器随后会继续包装base_model；节选没有展示整个注入过程。']],
    7:[['【原文缺陷】本数据类节选缺少导入，以及target_modules的默认field。',
        '即使补齐真实导入/父类，默认r后出现无默认target_modules仍会触发TypeError。',
        '这里只增加注释，不补字段；真正运行的9格导入包中完整LoraConfig，与此节选不同。']],
    8:[['【必须会】这是模型架构到默认目标后缀列表的dict，不是task_type或peft_type映射。',
        'BLOOM用融合query_key_value；LLaMA用分离q_proj/v_proj，不能在不同架构间照搬名字。'],
       ['这是LoraModel内部的扫描节选，依赖self.model、lora_config和re模块。',
        '只计算是否匹配目标，不创建LoRA参数或执行模块替换；真正替换在_find_and_replace后续。',
        'key是完整模块路径str；判断结果可能是Match/None或bool，没有输入Tensor和shape变换。']],
}
PREFIX_NOTES.update({
    'self.base_model =':'保存传入模型引用；is可核对身份，没有clone/deepcopy或权重分配。',
    'self.config =':'访问基模型配置并保存同一对象引用；改配置可能影响双方，不是加载新checkpoint。',
    'self.active_adapter =':'保存当前adapter名字str，默认通常为default；与CAUSAL_LM任务类型不同。',
    '@dataclass':'装饰器在类定义结束后生成__init__等方法；不是模型forward或参数训练。',
    'class LoraConfig':'继承PeftConfig的配置类；类型注解不自动进行运行时参数类型校验。',
    'r: int':'field(default=8,metadata=...)给整数r设置默认8；help只提供字段说明，不执行算法。',
    'target_modules:':'Optional[Union[List[str],str]]表示列表/字符串/None的类型范围，不自动生成default=None。\n'
                       '本节选漏了默认field；在已有默认r后放无默认字段，dataclass装饰会报字段顺序TypeError。',
    'key_list =':'named_modules()迭代(完整路径str,模块对象)，列表推导式只保留key；_表示忽略对象。\n'
                 '保存路径快照，便于后续改模块树；named_modules与named_parameters的内容不同。',
    'for key in':'依次遍历完整路径；本节选不替换模块，后续官方实现才按匹配结果执行替换。',
    'if isinstance':'isinstance(...,str)区分完整路径正则模式与目标名列表；不是区分模型类别。',
    'target_module_found = re.fullmatch':'re.fullmatch(pattern,key)要求正则覆盖整个路径，返回Match或None。\n'
                                        '裸query_key_value无法匹配带前缀路径，.*query_key_value可匹配；不是子串search。',
    'target_module_found = any':'key.endswith(target_key)逐个检查后缀；生成器供any短路判断，最终返回bool。\n'
                               '列表[query_key_value]可匹配完整路径末尾；它与裸正则字符串的语义不同。',
})


def source(cell):
    value=cell.get('source','')
    return ''.join(value) if isinstance(value,list) else value


def blocks(text):
    return [textwrap.dedent(m.group(2)).strip() for m in FENCE.finditer(text)]


def annotate_lines(text):
    out=[]
    for line in text.splitlines(keepends=True):
        stripped=line.strip();note=None
        for prefix,description in PREFIX_NOTES.items():
            if stripped.startswith(prefix):note=description;break
        if note is None:
            for model,description in MODEL_DESCRIPTIONS.items():
                if stripped.startswith('"'+model+'":'):note='此架构默认目标为'+description+'；不是统一目标。';break
        if note:
            indent=line[:len(line)-len(line.lstrip())]
            out.extend(indent+'# '+row+'\n' for row in note.splitlines())
        out.append(line)
    return ''.join(out)


def annotate_block(text, notes):
    lines=[line for line in text.splitlines() if line.strip()]
    indent=' '*min(len(line)-len(line.lstrip()) for line in lines)
    return ''.join(indent+'# '+note+'\n' for note in notes)+annotate_lines(text)


def notebook_note(text):
    return re.sub(r'\]\(([^)]+)\)',lambda m: ']('+(m.group(1) if '://' in m.group(1)
                  or m.group(1).startswith('#') else '../../'+m.group(1))+')',text)


def export_readable(notebook,target):
    parts=['# 第04讲 PEFT / LoRA 源码注释阅读版\n'
           '# 来源：chunhuizhang/personal_chatgpt，提交 7fbc29824171fd07dc2c30fe922e61ebc018ad1d。\n'
           '# 无JSON或历史输出，以 # %% 分格；建议在配套notebook逐格核对环境后执行。\n'
           '# 六段Markdown源码节选保留为注释，包含原缺陷，不作为独立可运行类/脚本。\n'
           '# 固定官方源码的离线CPU注入验证见复现/scripts/verify_injection.py。\n\n']
    for cell in notebook['cells']:
        text=source(cell)
        if cell['cell_type']=='code':
            parts.append('# %%\n'+ '\n'.join(line.rstrip() for line in text.splitlines())+'\n\n')
        else:
            parts.append('# %% [markdown]\n'+'\n'.join(('# '+line).rstrip() for line in text.splitlines())+'\n\n')
    Path(target).write_text(''.join(parts).rstrip()+'\n',encoding='utf-8')


def generate(notebook,explanation,output_dir):
    raw=Path(notebook).read_bytes();original=json.loads(raw.decode('utf-8-sig'))
    cells=original['cells'];assert len(cells)==18
    code=[i for i,c in enumerate(cells) if c['cell_type']=='code']
    assert set(code)==set(CODE_NOTES)==set(CODE_DETAILS)
    markdown={i:len(blocks(source(c))) for i,c in enumerate(cells) if c['cell_type']=='markdown' and blocks(source(c))}
    assert markdown=={2:2,4:1,7:1,8:2}
    text=Path(explanation).read_text(encoding='utf-8')
    headings=list(re.finditer(r'^## 单元 (\d+)：.*$',text,re.M));notes={}
    for pos,m in enumerate(headings):
        end=headings[pos+1].start() if pos+1<len(headings) else text.index('## 覆盖、学习优先级')
        notes[int(m.group(1))]=text[m.start():end].strip()
    assert set(notes)==set(range(18))
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    report={'source_sha256':hashlib.sha256(raw).hexdigest(),'original_cells':18,
        'explained_cells':sorted(notes),'code_cells':code,'commented_code_cells':sorted(CODE_NOTES),
        'markdown_blocks_per_cell':markdown,'commented_markdown_blocks':6,'original_empty_code_cells':[17],'artifacts':{}}
    for filename,commented in [('原代码_带说明.ipynb',False),('代码注释.ipynb',True)]:
        result=copy.deepcopy(original)
        result['cells']=[{'cell_type':'markdown','metadata':{},'id':'reading-header',
            'source':'# 第04讲中文代码说明\n\n原版另存；原输出为历史记录，不冒充本次运行。\n'
                     '注释仅解释原行为，保留Markdown源码节选缺陷；原7B量化加载未运行。'}]
        for i,cell in enumerate(cells):
            result['cells'].append({'cell_type':'markdown','metadata':{},'id':'explain-%02d'%i,'source':notebook_note(notes[i])})
            current=copy.deepcopy(cell)
            if commented and i in CODE_NOTES:
                current['source']='# '+CODE_NOTES[i]+'\n'+''.join('# '+row+'\n' for row in CODE_DETAILS[i])+annotate_lines(source(cell))
                assert ast.dump(ast.parse(source(current)))==ast.dump(ast.parse(source(cell)))
            elif commented and i in markdown:
                block_notes=iter(MARKDOWN_DETAILS[i])
                current['source']=FENCE.sub(lambda m:m.group(1)+annotate_block(m.group(2),next(block_notes))+m.group(3),source(cell))
                before,after=blocks(source(cell)),blocks(source(current))
                assert len(before)==len(after)
                for a,b in zip(before,after):assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))
            else:assert current==cell
            result['cells'].append(current)
        nbformat.validate(nbformat.from_dict(result))
        target=output_dir/filename
        target.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        report['artifacts'][filename]={'cells':len(result['cells']),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        if commented:
            export_readable(result,output_dir/'04_PEFT_lora_代码注释.py')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--notebook',required=True)
    parser.add_argument('--explanation',default=str(Path(__file__).resolve().parents[2]/'代码单元解释.md'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--record',required=True)
    args=parser.parse_args();report=generate(args.notebook,args.explanation,args.output_dir)
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('18 cells explained, 8 code cells + 6 Markdown blocks commented; schema/AST passed')
