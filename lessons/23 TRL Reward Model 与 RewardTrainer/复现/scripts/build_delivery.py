# -*- coding: utf-8 -*-
"""Preserve all course cells and add Chinese comments, never change original statements."""
import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager

ROOT=Path(__file__).resolve().parents[2]
COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
TRL_COMMIT='e90e8d91d2265e484f229c45a5eb8982f94a2936'
SHAS={'reward_trainer.py':'f496262efc35b3e50f9d899531a4a8cb08bcd1a57dd0669ebf5da945ff4b7787',
 'trainer_utils.py':'efc0bda563040376282642ef2bb42bb228fe6ffbaad001b3072f70282f3e5ca0',
 'reward_config.py':'f566e90ec99ea24331d931b6d59123abd0046bd5e7b6232f93fb5110ba26f02b'}
NOTES={
0:'【建议会】设置当前Python进程的HTTP/HTTPS代理为本机7890，不是安装代理服务。只有机器上确有代理才有效。本次不执行此格，以免把后续离线验证指向不存在的服务。',
1:'【必须会】AutoTokenizer按模型配置选择分词器；load_dataset读取偏好数据；torch负责Tensor/autograd；tqdm.notebook是进度条；Image用于远程HTML图片展示。导入不等于下载模型，原tqdm在后续没有显式调用。',
2:'【建议会】RM是Reward Model，奖励模型。接下来的任务是学会给完整对话打分，不是训练模型逐字生成回复。',
3:'【必须会】r_theta(x,y)是输入prompt和回答后的一个实数score/logit，不是现成的绝对质量概率。num_labels=1时模型输出[B,1]；有监督标签时分类模型可能走MSE回归，但RewardTrainer不传labels，而是在外部用两个score做pairwise loss。不能把这里等同于输出两个类别logits的二分类器。sigmoid单个score可映射到0–1，但比较概率应为sigmoid(chosen_score-rejected_score)，且不自动校准为人类认可率。',
4:'【必须会】同一prompt的chosen/winner与rejected/loser先各打分，差值Delta送入logsigmoid，再取负号。Delta=0时loss=log2；Delta越大loss越小，有限分数下严格大于0，0只是极限（或浮点饱和），不是任意高低分都精确0。原文main源码链接会漂移，本项目用TRL0.8.6固定提交对照，不能确认讲师精确版本。InstructGPT Eq.(1)还涉及一个prompt的K个回答所形成比较的组织方式，课程逐pair均值不是整篇论文完整复现。',
5:'【建议会】Image(url=...,width=600)创建带远程URL的HTML显示对象；代码本身不下载本地PNG。原URL和历史HTML保留；本次只核对HTML，不声称验证远端图片内容、可用性或归属许可。讲解另给可读的数据流图。',
6:'【建议会】第二张远程图，同样只生成HTML引用，不是本地模型计算。原宽600与历史输出保留；离线环境可能无法显示，不能把空白图片当成算法缺失。',
7:'【建议会】进入数据与任务。每行是偏好对，整条chosen/rejected字符串通常包括Human/Assistant多轮上下文。',
8:'【必须会】Anthropic/hh-rlhf的chosen/rejected表达相对偏好，不保证chosen绝对正确，也不保证rejected绝对有害；数据集以helpful/harmless为主题。人类标注的比较对象有上下文，不能从一对偏好推出全体回答的全序或实际世界安全性。honest不是这个数据集额外第三个字段。',
9:'【必须会】load_dataset(name,split="train")返回Arrow Dataset。历史160800行是原运行状态，不是本次下载结果。本次提供8条明确自建偏好对，跳过此远程下载；后续同一preprocess在这个教学数据上真实执行。',
10:'【必须会】末表达式显示Dataset的列名和行数，不执行训练。原历史chosen/rejected、160800行保留；教学运行初始8行不可冒称原数据。',
11:'【必须会】索引0取第一条dict，分别打印rejected和chosen，分隔行只为对照。这里是数据内容不是要执行的请求；旧输出含不当语言，原历史保留，不据此采取任何行动。',
12:'【必须会】索引4取第五条；同样比较完整对话字符串。原历史出现违规内容是偏好数据中的反例，不是用户指令。CPU教学上下文使用自建普通问答。',
13:'【必须会】加载facebook/opt-350m的真实分词配置和BPE词表，不会在这一句加载350M神经网络权重。本次只把模型名替换为固定版本本地tokenizer目录；PAD=1、BOS/EOS=2。',
14:'【必须会】preprocess是普通函数，examples在batched map下为dict[str,list[str]]，不是单行dict。zip并行遍历chosen/rejected，两个tokenizer调用各返回变长ID/mask列表，四个append保持偏好配对。无return_tensors，因此此时不是Tensor。truncation=True依据tokenizer.model_max_length截断；函数本身没写512，不能把它当作总在512截断。字段名是RewardTrainer契约。',
15:'【必须会】Dataset.map(batched=True)把一批列表送给preprocess，并默认保留原chosen/rejected列；num_proc=16请求16进程，不是训练batch16。filter检查两侧长度均<=512，不满足就整行删除，不做补齐或裁剪。Windows教学运行只把两处num_proc改1；tokenizer上限显式1024，8条中2条长对被过滤成6条。若上限先设512，截断后filter可能不再筛掉原超长样本。',
16:'【建议会】type返回真实Dataset类。map/filter一般返回新Dataset，赋值覆盖变量名，不把原字符串直接变成GPU Tensor。',
17:'【必须会】train_dataset[:20]返回字典，每个字段是一组列表；map(len,...)分别看chosen/rejected的长度，仍未padding。教学6行时只得到6项，不假造原历史20项长度。',
18:'【建议会】进入序列打分模型。它与CausalLM的词表输出不是一个头，本讲不使用generate。',
19:'【必须会】AutoModelForSequenceClassification按配置选择OPTForSequenceClassification；BitsAndBytesConfig描述量化加载参数，不是已经量化的模型。',
20:'【必须会】load_in_8bit=False和load_in_4bit=True选择四位权重路径，两者不能都真。真实需要bitsandbytes及合适后端，本次环境无该包，原格构造实际报PackageNotFoundError；CPU变体只把4bit改False。不能把FP32小模型训练称作4bit实验。',
21:'【必须会】from_pretrained本应加载OPT350M基础权重并新建num_labels=1的score头；device_map={"":0}指定整模型设备0，不等于CPU；trust_remote_code=True允许仓库自定义实现，本地标准OPT不需这一能力。use_cache=False关闭训练中的past缓存，不关闭attention。教学变体仅把加载调用换成从显式小配置构造真实随机OPT；此处未下载原权重。',
22:'【建议会】整格只有注释，原本用于查看每个参数dtype/device，不会执行任何打印。仍保留和说明，不能因无输出删掉。',
23:'【建议会】model.forward??是IPython源码检查语法，不是Python运算符。notebook保留原语法；纯文本阅读版用IPython等价pinfo2调用，并解释CLI分页hook仅改变展示方式。验证真实获取OPT forward源码，不把inspect结果当训练。',
24:'【必须会】显示模型结构。原历史是350M的4bit投影，词表embedding50272/维512、hidden1024；教学小模型结构不同。不要从原显示文本断言本机运行了该量化模型。',
25:'【必须会】遍历named_parameters，用requires_grad选择可训练参数，numel统计元素数。这是在注入LoRA之前，结果不是LoRA参数量，也不是显存字节数。量化参数的存储表示更不能仅按numel乘FP32大小推算。',
26:'【建议会】RewardTrainer继承transformers.Trainer；沿用训练调度，重写pairwise compute_loss与预测步骤。该训练器训练奖励模型，不在此执行PPO更新策略。',
27:'【必须会】TrainingArguments指定训练调度；LoraConfig配置适配器；RewardTrainer连接偏好数据和打分模型。安装库版本影响参数签名，本次明确用TRL0.8.6/Transformers4.44.2/PEFT0.12.0，非讲师精确环境承诺。',
28:'【必须会】r16是低秩，alpha16令缩放alpha/r=1；bias="none"不额外训练bias；SEQ_CLS启用序列分类包装；modules_to_save=["score"]使打分头可训练并随adapter保存。OPT默认目标q_proj/v_proj，不是所有线性层。B初始0令LoRA初始增量0；不能因此把score头也丢掉。',
29:'【必须会】原1000 optimizer steps、每设备4对、累积1、lr1.41e-5/AdamW，每50步保存和记录，tensorboard日志，remove_unused_columns=False保留偏好字段；一对需要chosen/rejected两次模型前向，不是单条文本batch4。RewardTrainer自动注入PEFT并建立默认collator；max_length512传给padding不自动执行截断。CPU教学变体改两步/batch2/lr.001/日志与保存路径等调度，明确记录全部替换。',
30:'【必须会】此时遍历trainer.model才是PEFT包装后的参数；q/v的LoRA A/B与score的可训练副本应在列表里。本次PEFT0.12.0中base及score.original_module冻结，小OPT有1024个LoRA参数+16个score参数=1040。原历史却同时列出score.original_module和modules_to_save两份512权重，24层q/v的1572864+两份score1024=1573888，不能把旧输出改写成只有一个score头或当成本次参数量。',
31:'【必须会】train做真实优化更新；save_pretrained保存PEFT adapter及modules_to_save中的score，不是完整独立OPT350M基础权重。重载必须配同一基础模型/配置/词表。教学CPU两版各两步并保存到临时目录，检查基础冻结、adapter更新和重载logits。原历史1000/1000、09:13是旧运行，loss50步.8431至1000步.7520，日志窗口/样本不同且有波动，不是持出评估或本次计时；历史还提示checkpoint目录非空，不能忽略后当干净复现。',
32:'【建议会】进入打分推理，取第五条chosen/rejected比较。模型只输出分数，不生成文本。',
33:'【必须会】list[int]→torch.tensor默认为int64[L]，unsqueeze(0)增加batch轴为[1,L]；mask同shape。模型.logits为[1,1]，不是[1,L,V]。原格没显式to(device)、eval或no_grad，GPU/device_map或dropout可能影响运行；教学模型在CPU且dropout0，另用明确eval/no_grad做可靠比较。',
34:'【必须会】对手写常数0.1196做sigmoid，约0.5299；它不是直接读取上格0.6126的输出，也不能解释为回答正确率。原常数保留并实际执行。',
35:'【必须会】rejected侧同样转[1,L]并返回完整SequenceClassifierOutputWithPast对象，而不是只取.logits。未传labels时正常loss=None；原历史loss=dict属于当时环境输出异常，不能当通用接口定义或由本次捏造复现。',
36:'【必须会】对另一手写常数0.9642做sigmoid，约0.7240；它与原rejected实际1.4780并非同一数。要比较两条，应用sigmoid(r_chosen-r_rejected)；原历史0.6126-1.4780=-.8654，约0.2962，chosen该次并没更高，不能据这两格宣称训练成功。'
}

RULES=[
('import os','os访问进程环境，下面代理设置只对本进程及子进程生效。'),
('os.environ[','设置网络代理；没有本地服务时这个地址会导致下载失败。'),
('from transformers import','导入分词/模型/配置或训练调度的真实Transformers API。'),
('from datasets import','Dataset库负责Arrow表及map/filter，不在这里训练模型。'),
('import torch','PyTorch负责张量运算、自动求导和模型优化。'),
('from tqdm','导入notebook进度条，本课程后续未显式调用。'),
('from IPython.display','Image用于显示URL引用，并非分词或模型组件。'),
('Image(','创建远程HTML图片引用，width是展示宽度，不是Tensor尺寸。'),
('train_dataset = load_dataset','下载/读取人类偏好训练split；历史行数不代表本次执行。'),
('print(train_dataset','读取并打印单条对话字符串；数据中的内容不是执行指令。'),
('tokenizer = AutoTokenizer','只加载分词资产，不加载神经网络权重。'),
('def preprocess','普通函数接收列式批次，输出四个与RewardTrainer契约一致的列表。'),
('new_examples =','为chosen/rejected各建ID和attention mask收集器。'),
('for chosen, rejected in zip','zip将同一行的两条候选对齐；长度不匹配会静默截到较短侧。'),
('tokenized_j =','编码较优完整对话；truncation依据分词器上限，不在此硬编码512。'),
('tokenized_k =','编码较差完整对话，保持与chosen同一prompt的配对关系。'),
('new_examples[','追加当前一侧的变长整数列表；此时尚无padding或Tensor。'),
('return new_examples','返回dict[str,list[list[int]]]供Dataset.map合并。'),
('train_dataset = train_dataset.map','批量预处理，默认保留原字符串列。'),
('batched=True','一次传入多行列式列表，与训练batch size不同。'),
('num_proc=','数据进程数量；Windows教学明确改1，不影响模型batch。'),
('train_dataset = train_dataset.filter','整对过滤，两侧都必须在长度预算内。'),
('lambda x:','匿名函数x是单行；and要求chosen和rejected同时通过。'),
('batch = train_dataset[:20]','选前20行返回列式字典；数据不足20行时返回实际行数。'),
('print(list(map(len','分别观察每条token序列长度，尚未补齐。'),
('quantization_config =','配置量化加载方式，不是执行前向。'),
('load_in_8bit=False','关闭8bit，与4bit互斥。'),
('load_in_4bit=True','启用4bit，原格依赖bitsandbytes后端；CPU未执行量化。'),
('model = AutoModelForSequenceClassification','加载序列打分模型，新增单输出score头；非CausalLM。'),
('quantization_config=','把量化配置交给模型加载器。'),
('device_map=','空字符串匹配整个模型，0指设备0而非CPU。'),
('trust_remote_code=','允许模型仓库自定义代码；标准OPT不需自定义模型。'),
('num_labels=1','每条完整序列输出一个标量，[B,1]。'),
('model.config.use_cache','训练关闭past KV缓存，仍正常计算注意力。'),
('total_params = 0','初始化可训练参数元素计数，不是显存统计。'),
('for name, para in','遍历名字与Parameter；LoRA注入前后对象不同。'),
('if para.requires_grad','只统计允许训练的参数，不代表本步梯度已非零。'),
('total_params +=','numel返回参数张量的元素数。'),
('from peft import','导入LoRA适配器配置。'),
('from trl import','RewardTrainer在Trainer调度上实现偏好对损失。'),
('peft_config =','构建配置，真正注入由RewardTrainer初始化完成。'),
('r=16','LoRA秩16，决定A/B中间维度。'),
('lora_alpha=16','增量缩放alpha/r=1，不是优化器学习率。'),
('bias="none"','不额外开放基础bias训练。'),
('task_type="SEQ_CLS"','采用序列分类PEFT包装，不是因果语言模型任务。'),
('modules_to_save=','score头也要训练/保存，否则重载可能丢失评分映射。'),
('training_args =','训练调度配置，默认device由框架决定。'),
('output_dir=','checkpoint和日志的基础目录。'),
('max_steps=','优化器更新步数，不是样本条数或epoch。'),
('per_device_train_batch_size=','每设备每微批的偏好对数；两侧各需一次前向。'),
('gradient_accumulation_steps=','多少微批梯度合成一次optimizer更新。'),
('learning_rate=','AdamW更新步长，与LoRA alpha含义不同。'),
('optim=','选择PyTorch AdamW优化器。'),
('save_steps=','按optimizer步数保存checkpoint。'),
('logging_steps=','按optimizer步数记录日志。'),
('report_to=','日志接收器；tensorboard需相应依赖。'),
('remove_unused_columns=False','保留chosen/rejected四字段，不能按普通model.forward签名删掉。'),
('trainer = RewardTrainer','连接model/data/tokenizer，默认collator补齐两侧，并注入PEFT。'),
('peft_config=','让RewardTrainer调用get_peft_model，冻结基础并开放adapter/score。'),
('max_length=512','传给默认padding collator，不等于在这里执行truncation。'),
('trainer.train()','实际执行前向→成对loss→反向→AdamW调度。'),
('trainer.model.save_pretrained','保存adapter与score副本，基础模型仍需另行匹配。'),
('trainer.model(input_ids=','整数序列加batch轴后打分；原格未显式移动设备或关闭梯度。'),
('attention_mask=torch.tensor','mask从[L]加成[1,L]，1为有效token，0为补齐。'),
('torch.sigmoid(torch.tensor','只对手写常数做sigmoid，不能当上格模型的偏好概率。'),
('def __init__','初始化真实Trainer子类，连接PEFT、collator、父类训练调度。'),
('if type(args) == TrainingArguments','兼容旧TrainingArguments入口并发弃用提示。'),
('if not is_peft_available()','缺PEFT且要求LoRA时明确报错。'),
('elif is_peft_available()','已安装PEFT时按配置包装模型。'),
('if not isinstance(model, PeftModel)','避免给已有PeftModel再注入一层。'),
('if getattr(model,','量化模型才走kbit训练准备；FP32小模型不走该分支。'),
('model = prepare_model_for_kbit_training','准备量化训练/冻结/必要dtype，非本次CPU量化验证。'),
('model = get_peft_model','真正按LoRA配置修改模型并管理可训练参数。'),
('if compute_metrics is None','默认评估使用成对选择准确率，不是绝对分数阈值。'),
('if data_collator is None','没提供自定义collator则创建偏好对专用补齐器。'),
('if tokenizer is None','默认collator需要分词器的PAD规则。'),
('data_collator = RewardDataCollator','chosen/rejected分别动态padding。'),
('if args.remove_unused_columns','普通Trainer列删除机制不适用于四个偏好字段。'),
('super().__init__','调用父Trainer初始化，继承优化器/loader/保存等调度。'),
('def compute_loss','RewardTrainer重写成对损失；不使用模型内labels回归loss。'),
('rewards_chosen = model(','共享评分模型对chosen侧做前向，返回[B,1]原始score。'),
('rewards_rejected = model(','同一个模型对rejected侧前向，共享全部参数。'),
('return_dict=True','要求可按logits键读取结构化模型返回。'),
('if "margin" in inputs','可选margin支路；此固定版collator给[B]，需留意与[B,1]的广播。'),
('loss = -nn.functional.logsigmoid','先做两侧score差，稳定logsigmoid后对批次取平均。'),
('if return_outputs','True时同时返回loss和两侧原score，不是词表logits。'),
('def prediction_step','评估路径包装输入、关闭梯度，输出每对两候选的概率。'),
('inputs = self._prepare_inputs','父Trainer负责把Tensor放到目标device。'),
('with torch.no_grad()','评估不构建反向图，但仍需模型eval控制dropout。'),
('loss, logits_dict = self.compute_loss','复用pairwise loss并取得两侧score。'),
('logits = torch.stack','[2,B,1]→均值[2,B]→沿候选softmax→转置[B,2]。'),
('labels = torch.zeros','每对chosen在第0列，因此标签全0。'),
('def __call__','collator对象可当函数调用，输入若干单行dict。'),
('features_chosen =','chosen侧单独收集input_ids和mask。'),
('features_rejected =','rejected侧单独收集，两侧padding长度可以不同。'),
('has_margin =','仅看第一行是否有margin；混合缺字段不安全。'),
('batch_chosen = self.tokenizer.pad','按chosen当前批最长长度补齐。'),
('batch_rejected = self.tokenizer.pad','按rejected当前批最长长度补齐。'),
('margin = torch.tensor','此版生成[B]，与[B,1]相减会广播成[B,B]，本讲单独验证。'),
('def compute_accuracy','评估[B,2]的argmax是否等于全0标签，平分时可能误导。'),
('predictions = np.argmax','选择分数较高的候选；完全相等时argmax偏向第一列。'),
('accuracy = np.array','得到本批候选选择准确率，不代表人类总体认可率。')]

def annotate(src,header=''):
    result=[('# '+header+'\n') if header else ''];inside=False
    for line in src.splitlines(keepends=True):
        if '"""' in line:
            if line.count('"""')%2:inside=not inside
            result.append(line);continue
        if not inside:
            for prefix,text in RULES:
                if line.strip().startswith(prefix):
                    result.append(line[:len(line)-len(line.lstrip())]+'# '+text+'\n');break
        result.append(line)
    return ''.join(result)

def extract(src,node):
    start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    return ''.join(src.splitlines(keepends=True)[start-1:node.end_lineno])+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--sources',required=True);args=p.parse_args();sources=Path(args.sources)
    target=ROOT/'代码/tutorials/reward_model';target.mkdir(parents=True,exist_ok=True)
    path='tutorials/reward_model/trl_reward_trainer.ipynb'
    raw=subprocess.check_output(['git','show',COMMIT+':'+path],cwd=args.upstream)
    (target/'trl_reward_trainer.ipynb').write_bytes(raw)
    for name,h in SHAS.items():assert hashlib.sha256((sources/name).read_bytes()).hexdigest()==h
    manifest={'repository':'https://github.com/chunhuizhang/personal_chatgpt','commit':COMMIT,'files':[{'upstream_path':path,'published_path':'代码/'+path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}],
        'reference_trl_version':'0.8.6','reference_trl_commit':TRL_COMMIT,'official_source_sha256':SHAS,
        'tokenizer':{'repository':'facebook/opt-350m','commit':'08ab08cc4b72ff5593870b5d527cf4230323703c','local_files_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((sources/'opt-tokenizer').iterdir()) if f.is_file()}},
        'historical_environment':'exact lecturer version not established; selected compatible fixed reference explicitly'}
    (ROOT/'来源清单.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    original=json.loads(raw);assert len(original['cells'])==37
    intro={'cell_type':'markdown','metadata':{},'source':['# 第23讲：TRL Reward Model与RewardTrainer\n\n原单元从0编号，原代码与历史输出保留。CPU小模型变体独立记录，不是原4bit OPT350M训练。\n']}
    explained=copy.deepcopy(original);annotated=copy.deepcopy(original)
    explained['cells']=[copy.deepcopy(intro)];annotated['cells']=[copy.deepcopy(intro)]
    md=['# 第23讲全部单元说明\n\n37格，29代码、8Markdown。原IPython检查、仅注释格、远程图引用、历史异常均保留。\n']
    reading='# -*- coding: utf-8 -*-\n# 原notebook全部29代码格的中文注释阅读版；运行前阅读复现范围，不自动执行网络/GPU/1000步训练。\n'
    transform=TransformerManager()
    for i,c in enumerate(original['cells']):
        n={'cell_type':'markdown','metadata':{},'source':(f'## 原单元 {i}（{c["cell_type"]}）\n\n'+NOTES[i]+'\n').splitlines(keepends=True)}
        explained['cells'] += [copy.deepcopy(n),copy.deepcopy(c)];a=copy.deepcopy(c)
        if c['cell_type']=='code':
            src=''.join(c['source']);body=annotate(src,NOTES[i]);a['source']=body.splitlines(keepends=True)
            if i==23:body='# 原IPython语法：model.forward??\n# pinfo2为同一源码检查的可解析Python形式，需IPython运行上下文。\n'+transform.transform_cell(src)
            reading+=f'\n# %% 原代码单元 {i}\n'+body+'\n'
        annotated['cells'] += [copy.deepcopy(n),a];md.append(''.join(n['source']))
    for name,nb in [('原代码_带说明.ipynb',explained),('代码注释.ipynb',annotated)]:
        (target/name).write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    (ROOT/'代码单元解释.md').write_bytes(('\n'.join(md).rstrip()+'\n').encode('utf-8'))
    (target/'trl_reward_trainer_代码注释.py').write_bytes(('\n'.join(line.rstrip() for line in reading.splitlines()).rstrip()+'\n').encode('utf-8'))
    license_header=''.join((sources/'reward_trainer.py').read_text(encoding='utf-8').splitlines(keepends=True)[:13])
    core='# -*- coding: utf-8 -*-\n'+license_header+'\n# 固定TRL0.8.6完整RewardTrainer类、默认collator和compute_accuracy，只加中文注释。\n# 精选对象依赖原模块的imports，不是可独立导入的完整模块；原算法包括margin广播边界均保留。\n\n'
    for fname,names in [('reward_trainer.py',['RewardTrainer']),('trainer_utils.py',['RewardDataCollatorWithPadding','compute_accuracy'])]:
        s=(sources/fname).read_text(encoding='utf-8');tree=ast.parse(s)
        for name in names:
            node=next(n for n in tree.body if getattr(n,'name',None)==name)
            core+=annotate(extract(s,node))+'\n'
    (target/'RewardTrainer核心源码_代码注释.py').write_bytes((core.rstrip()+'\n').encode('utf-8'))
    print('Built37 original cells, two75-cell derivatives, 29-cell Python and complete Trainer/collator source reading.')

if __name__=='__main__':main()
