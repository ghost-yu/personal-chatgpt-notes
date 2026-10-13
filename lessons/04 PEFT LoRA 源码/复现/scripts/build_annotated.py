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
            out.append(indent+'# '+note+'\n')
        out.append(line)
    return ''.join(out)


def generate(notebook,explanation,output_dir):
    raw=Path(notebook).read_bytes();original=json.loads(raw.decode('utf-8-sig'))
    cells=original['cells'];assert len(cells)==18
    code=[i for i,c in enumerate(cells) if c['cell_type']=='code']
    assert set(code)==set(CODE_NOTES)
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
            result['cells'].append({'cell_type':'markdown','metadata':{},'id':'explain-%02d'%i,'source':notes[i]})
            current=copy.deepcopy(cell)
            if commented and i in CODE_NOTES:
                current['source']='# '+CODE_NOTES[i]+'\n'+annotate_lines(source(cell))
                assert ast.dump(ast.parse(source(current)))==ast.dump(ast.parse(source(cell)))
            elif commented and i in markdown:
                current['source']=FENCE.sub(lambda m:m.group(1)+annotate_lines(m.group(2))+m.group(3),source(cell))
                before,after=blocks(source(cell)),blocks(source(current))
                assert len(before)==len(after)
                for a,b in zip(before,after):assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))
            else:assert current==cell
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
    args=parser.parse_args();report=generate(args.notebook,args.explanation,args.output_dir)
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('18 cells explained, 8 code cells + 6 Markdown blocks commented; schema/AST passed')
