# -*- coding: utf-8 -*-
"""主12格与补充16格全部说明，Python和Markdown片段只增加注释。"""
import copy
import json
from pathlib import Path

MAIN={
0:('导入图片展示类','【建议会】IPython.display.Image是展示对象，不是模型；后面从相对路径读取PNG。'),
1:('基础章节','【必须会】先分清原始分片、模型参数与推理入口。'),
2:('仓库、模型规格与head_dim','【必须会】main对应llama2是当时描述，本次固定llama_v2 commit。7B dim4096/n_heads32，每头128；multiple_of256用于MLP维度取整。Markdown中的self.head_dim整数除法是源码片段，需要args/self上下文；下载链接是资料，不执行下载或申请。'),
3:('展示早期权重目录','【建议会】Image读取../imgs/llama_weights.png，width400是展示宽度，不改模型。原输出为历史图片，真实重做从本讲代码/tutorials工作目录执行。'),
4:('展示Llama2权重目录','【必须会】base/chat与7B/13B/70B、pth分片和params.json属于原始格式；图片总大小不等单模型显存。width300只控制展示。'),
5:('两个completion入口','【必须会】text/chat共享Llama.generate，差别在权重和输入组织，不是两个独立解码算法。'),
6:('text入口','【必须会】base续写普通字符串，BOS=True/EOS=False；echo默认False，与HF pipeline完整文本默认不同。'),
7:('7B与13B文本启动命令','【必须会】两条torchrun逻辑命令；nproc1/2对应MP分片，不是batch。ckpt_dir是原始pth目录，tokenizer_path是独立词表。max_seq_len128/max_batch_size4是buffer/cache上限；示例main另默认max_gen_len64。反斜杠是Bash续行，PowerShell应换语法；原权重和NCCL路径本次未执行。'),
8:('chat入口','【必须会】messages先经角色检查、system合并和INST模板，再进入相同生成器。'),
9:('7B与13B聊天启动命令','【必须会】两个chat checkpoint目录，nproc1/2对应原始分片。max_seq_len512、max_batch_size8；官方例子六个dialog，batch4会超限。B是对话条数，不是消息数或GPU数；不把CPU随机验证当GPU命令执行。'),
10:('辅助工具章节','【可以暂时不掌握】终端工作流与模型算法分开。'),
11:('终端资料链接','【可以暂时不掌握】外部终端使用文章是辅助链接，本讲没有复制其操作或执行命令。'),
}
AUX={
0:('接口对照提纲','【必须会】补充notebook直接展示tokenizer ID、接口类型与特殊标签。'),
1:('同一模型包装与共同generate','【必须会】Markdown构造片段：Tokenizer返回词表对象，Transformer接受ModelArgs，load_state_dict加载权重，Llama保存两对象引用。strict=False不保证shape错误可忽略；text/chat共享generate，权重与prompt组织不同。片段缺少checkpoint等上下文，不能当完整加载脚本执行。'),
2:('真实SentencePiece词表加载','【必须会】SentencePieceProcessor是类，model_file指当前目录tokenizer.model；返回sp_model实例，不加载LLM权重。本次ASCII临时目录复制同一词表后，原格整格执行。'),
3:('BOS ID','【必须会】实例方法bos_id()返回整数1，句子开始符；原生encode默认不自动添加它。'),
4:('EOS ID','【必须会】eos_id()返回整数2，结束符；它不等于pad。'),
5:('pad ID','【必须会】pad_id()返回-1，表示此词表没有合法pad项；不能把-1当Embedding的有效词表ID。原生成buffer以它作未填标记。'),
6:('接口参数章节','【必须会】先看输入/返回类型，再追包装与generate。'),
7:('text/chat方法签名','【必须会】self是Llama实例；prompts是list[str]，dialogs是list[list[Message]]。温度默认.6/top_p.9；max_gen_len None在方法中变max_seq_len-1；logprobs False。text另有echo False，chat无echo参数。两个def只有签名而没有body，原Markdown是介绍片段，保留不补成假实现，不能独立编译执行。'),
8:('chat参数结构','【必须会】一条dialog含可变多条消息，batch是dialog条数。'),
9:('消息类型与保留标签','【必须会】Message继承TypedDict描述role/content字典；Dialog=List[Message]只是类型别名，不执行模型。B_INST/E_INST为模板字符串，B_SYS/E_SYS含换行；SPECIAL_TAGS做字面子串检查。UNSAFE_ERROR是包装层固定错误，不是模型学会拒答。role需先定义，源码类型说明与运行校验分开。'),
10:('展示system分隔符换行','【建议会】print打印前一个换行、结束标签、后两个换行及print自己的终止换行；不是新的system消息。'),
11:('特殊标签分词章节','【必须会】标签是可分词字符串，不能因尖括号或方括号就当成一个special token。'),
12:('编码INST开始标签','【必须会】sp_model.encode返回list[int]，默认整数ID，不默认BOS/EOS；[INST]由多个普通词表片段表示。'),
13:('编码普通单词','【必须会】what为字符串，分词结果是ID列表，不是词向量。'),
14:('编码INST结束标签','【必须会】[/INST]与开始标签的分词长度不必相同，不能机械设固定标签宽度。'),
15:('编码完整instruction字符串','【必须会】只分词模板字符串；没有调用LLM，也没有BOS/EOS自动追加。要生成需官方Tokenizer.encode的明确边界与completion包装。'),
}
COMMENTS={
'from IPython.display import Image':'导入IPython图片展示类，不加载模型。',
"Image('../imgs/llama_weights.png', width=400)":'相对本讲代码/tutorials读取图片，400是显示宽度。',
"Image('../imgs/llama2-weights.png', width=300)":'展示原始Llama2权重目录，300不改变文件或权重。',
'from sentencepiece import SentencePieceProcessor':'导入真实词表处理类，不能代替语言模型。',
"sp_model = SentencePieceProcessor(model_file='./tokenizer.model')":'当前工作目录需对应词表；返回真实SentencePieceProcessor。',
'sp_model.bos_id()':'返回开始符整数1；不是生成句子。',
'sp_model.eos_id()':'返回结束符整数2，与pad不同。',
'sp_model.pad_id()':'返回-1，词表未定义合法pad ID。',
'print("\\n<</SYS>>\\n\\n")':'字符串含模板换行，print再追加一个换行。',
"sp_model.encode('[INST]')":'返回普通片段ID列表，未自动添加BOS/EOS。',
"sp_model.encode('what')":'普通词字符串编码为list[int]，不是embedding。',
"sp_model.encode('[/INST]')":'结束标签也可能是多个token。',
"sp_model.encode('[INST] what is the recipe of mayonnaise? [/INST]')":'只编码整个instruction字符串，不调用generate或模型。',
}

def annotate_code(text):
    result=[]
    for line in text.splitlines(True):
        key=line.strip()
        if key and not key.startswith('#'):
            assert key in COMMENTS,key
            result.append(line[:len(line)-len(line.lstrip())]+'# '+COMMENTS[key]+'\n')
        result.append(line)
    return ''.join(result)

def annotate_md(text,index,aux):
    result=[];inside=False;introduced=False
    for line in text.splitlines(True):
        if line.lstrip().startswith('```'):
            inside=not inside;introduced=False
        elif inside and line.strip() and not line.lstrip().startswith('#'):
            if line.lstrip().startswith('torchrun '):
                result += ['# torchrun启动模型并行进程；nproc与原始checkpoint分片一致，不是batch。\n',
                           '# ckpt_dir指定base或chat权重；tokenizer_path为单独词表。\n',
                           '# max_seq_len/max_batch_size限制序列与batch，反斜杠是Bash续行。\n']
            elif aux and index==7:
                if line.strip().startswith('def '):result.append('# 方法签名只有声明，没有函数体；参数/返回类型见前置单元说明。\n')
                elif line.strip() not in (') -> List[CompletionPrediction]:',') -> List[ChatPrediction]:'):
                    result.append(line[:len(line)-len(line.lstrip())]+'# 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。\n')
            elif not introduced:
                result.append('# 源码介绍片段；所需对象/类型上下文和每行用途见前置说明。\n');introduced=True
        result.append(line)
    return ''.join(result)

def main():
    root=Path(__file__).resolve().parents[2];folder=root/'代码/tutorials';doc='# 第15讲代码单元解释\n\n主12格与接口补充16格，编号在各原notebook内从0开始。\n\n[主解释](解释.md) · [阅读入口](README.md)\n\n'
    for filename,notes,aux in [('llama2_src_01.ipynb',MAIN,False),('llama2_src_text_chat.ipynb',AUX,True)]:
        raw=json.loads((folder/filename).read_bytes());assert len(raw['cells'])==len(notes)
        base=filename[:-6];py=['# -*- coding: utf-8 -*-','# 第15讲 '+base+' Python阅读版；原语句保留。','']
        doc+='## '+filename+'\n\n'
        for derived,changed in [('原代码_带说明.ipynb',False),('代码注释.ipynb',True)]:
            out=copy.deepcopy(raw);out['cells']=[dict(cell_type='markdown',metadata={},id=base+'-intro',source=['# 第15讲说明与注释\n','原单元/历史输出保留，模型教学运行另记录。\n'])]
            for i,c in enumerate(raw['cells']):
                title,detail=notes[i];note='### 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],title,detail)
                out['cells'].append(dict(cell_type='markdown',metadata={},id=base+'-note-'+str(i),source=note.splitlines(True)))
                saved=copy.deepcopy(c);s=''.join(c['source'])
                if changed:s=annotate_code(s) if c['cell_type']=='code' else annotate_md(s,i,aux)
                saved['source']=s.splitlines(True);out['cells'].append(saved)
            name=derived if not aux else base+'_'+derived
            (folder/name).write_bytes((json.dumps(out,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
        for i,c in enumerate(raw['cells']):
            title,detail=notes[i];s=''.join(c['source']);doc+='### 单元 {}（{}）：{}\n\n{}\n\n'.format(i,c['cell_type'],title,detail)
            py+=['# %% '+str(i)]+['# '+x for x in (title+'\n'+detail).splitlines()]
            if c['cell_type']=='code':doc+='```python\n'+s+'\n```\n\n';py+=annotate_code(s).splitlines()
            else:
                shown=annotate_md(s,i,aux);doc+=shown+'\n\n';py+=['# '+x for x in shown.splitlines()]
            py+=['']
        (folder/(base+'_代码注释.py')).write_bytes(('\n'.join(x.rstrip() for x in py).rstrip()+'\n').encode('utf-8'))
    (root/'代码单元解释.md').write_bytes(('\n'.join(x.rstrip() for x in doc.splitlines()).rstrip()+'\n').encode('utf-8'))
    print('28 cells / 12 code cells explained; four derived notebooks and two reading scripts built.')

if __name__=='__main__':main()
