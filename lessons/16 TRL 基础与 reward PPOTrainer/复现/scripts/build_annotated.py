# -*- coding: utf-8 -*-
"""42格逐项说明；原语句、输出及Markdown概览保留，只增加注释。"""
import copy
import json
from pathlib import Path
NOTES={
0:('图片展示导入','【建议会】IPython.display.Image类读取图片，不执行训练。'),
1:('TRL基础章节','【必须会】SFT、RM训练、PPO是不同阶段，不能把Trainer名字当完成RLHF证据。'),
2:('来源链接','【建议会】原链接指main会变化，本讲参考固定TRL0.4.7；作者版本未锁定。'),
3:('三阶段示意图','【必须会】SFT目标文本、RewardTrainer偏好对、PPO策略优化；固定reward1.0不等于训练RM。图来自上游，历史输出保留。'),
4:('value head章节','【必须会】critic的value与外部reward model不同。'),
5:('value输出接口','【必须会】拼写foward指forward。包装返回(logits,loss,value)；hidden[B,T,H]经Linear(H,1)→[B,T,1]再squeeze→[B,T]，不是从词表概率生成reward。'),
6:('PPO step入口章节','【必须会】step消费queries、纯responses、每样本scalar score，内部KL/GAE/loss/update。'),
7:('偏好训练与策略图','【必须会】打分训练与PPO是两阶段；本例没有实例化RM，不以图证明完成摘要任务。'),
8:('PPO算法图','【建议会】rollout保存old logprob，训练多轮clip policy/value；reference是KL锚点，不是old policy。width600仅展示。'),
9:('SFT示例章节','【必须会】这里的Llama2/Guanaco命令属于SFT，不是后面的GPT2 PPO。'),
10:('SFT安装与启动命令','【必须会】三条shell逻辑命令：pip安装、clone源码、python执行示例。原未锁版本；model base7B、Guanaco、4bit/PEFT、micro4/K2单卡有效batch8。Bash反斜杠不能照搬PowerShell；本次不运行此GPU命令。'),
11:('SFT默认优化器与scheduler','【必须会】AdamW/linear/warmup0属于前面SFT路径；本PPO0.4.7默认Adam，不能挪用。'),
12:('SFT历史截图','【建议会】图片是原历史材料，不是新训练日志。'),
13:('PPO quick start章节','【必须会】接下来GPT2/value/reference/固定score路径。'),
14:('rollout evaluation optimization图','【必须会】LM生成、外部score、active/ref与value优化分工；KL是惩罚偏离，value不是RM。width500为展示。'),
15:('三个模型名的含义','【必须会】active可更新，reference约束偏离，RM提供样本分数；本例RM缺失，另有value head。'),
16:('导入框架、随机模块和TRL对象','【必须会】torch/numpy/random为模块，GPT2Tokenizer为类，PPOConfig为配置类，PPOTrainer管理真实训练。仅import不下载模型。'),
17:('设置随机种子函数','【建议会】def定义函数，调用42设置Torch/CUDA/NumPy/Python状态；CPU环境CUDA设种不代表运行GPU。cudnn.deterministic约束相关kernel；PPOConfig默认seed0会在Trainer初始化覆盖种子。'),
18:('active/ref包装和分词器加载','【必须会】两次from_pretrained加载相同基础来源，但新增value head随机，不能说两个包装全状态相等。pad=eos50256不扩词表。CPU变体只把三处gpt2标识改本地随机小GPT2/真实词表目录。'),
19:('查看tokenizer','【建议会】显示对象，不能据repr证明已训练语言模型。'),
20:('查看基础LM','【必须会】model.pretrained_model为GPT2LMHeadModel，输出词表logits与hidden；没有value head。'),
21:('查看value head','【必须会】Dropout+Linear(H,1)，GPT2 H768时769参数，小模型H16时17参数；Flatten repr不代表forward调用它。'),
22:('PPO配置与Trainer初始化','【必须会】字典**展开batch_size1，保留mini1/ppo_epochs4/seed0/lr1e-5等参考默认。位置参数按0.4.7签名；无dataset需自行送样本。active optimizer默认Adam，显式ref只作reference。'),
23:('查看active设备','【建议会】device属性返回设备，不移动模型。本次实际cpu，不当GPU证据。'),
24:('query字符串到tensor','【必须会】末尾空格保留，encode return_tensors pt→int64[1,7]，to移动到active设备。固定GPT2 IDs1212/3329/314/1816/284/262/220。'),
25:('生成kwargs字典','【必须会】min_length-1/top_k0.0是历史设置，兼容性按固定版本实测；top_p1、采样、pad=eos、新增20。字典创建不生成文本。'),
26:('真实rollout包含prompt','【必须会】列表推导按batch行给list[tensor[7]]；**kwargs展开，return_prompt=True返回query+续写，通常[27]。这是后面重复prompt根源，CPU运行保留原参数。'),
27:('decode完整返回','【必须会】response_tensor[0]取第一样本完整ID，decode返回含query的字符串；变量名response不证明它是纯续写。'),
28:('固定score与真实PPO更新','【必须会】reward为list[0维float tensor]，值1.0，没有RM。step把query和response拼接，原7+27=34；第二份query被优化。一次step默认4个ppo_epochs，不是只一次optimizer更新。修正版另测试，不改原格。'),
29:('查看训练统计','【建议会】dict含KL、policy/value等及时间；不凭固定score/单次stats宣称偏好能力改善。历史未保存可读stats。'),
30:('内部路线章节','【必须会】下面是缩略源码概览，不是完整PPO实现。'),
31:('PPO概览片段','【必须会】Markdown缩略拼接不同层级调用，缩进不构成可执行完整程序；按原文保留。batched_forward active/ref→compute_rewards→四轮minibatch→loss/更新；省略GAE与上下文，不能当完整算法或把kl loss当一个独立CE。'),
32:('重复prompt长度核对','【必须会】torch.cat沿唯一轴拼[7]+[27]→[34]，三次print展示；标准Trainer内部以batch输入。正确纯response20时总27。'),
33:('decode重复文本','【必须会】拼接IDs解码可见query两遍，证明不只是打印重复，而是输入被重复。'),
34:('基础模型前向章节','【必须会】不带labels，不是SFT loss。'),
35:('实际基础LM forward','【必须会】一维IDs[34]输入，output_hidden_states True返回各层hidden和词表logits；此原打印不带显式batch轴。本次随机H16/V50257模型，不能照抄历史H768。'),
36:('取末层hidden与logits','【必须会】hidden_states[-1]最后层，[34,H]；logits[34,V]。原注释的loss访问没有执行；未传labels通常loss None。'),
37:('value计算章节','【必须会】从hidden映射标量，不从词表logits或RM打分。'),
38:('再次查看value模块','【建议会】同一个v_head引用，不新建critic。'),
39:('无梯度value forward','【必须会】with上下文禁止记录梯度，不自动关Dropout；Linear[34,H]→[34,1]，squeeze(-1)→[34]。历史H768说明保留，CPU实际H16另记录。'),
40:('查看每位置value','【必须会】tensor[34]可正可负，不是34个完整回答reward或概率。'),
41:('空代码格','【建议会】原空格保留，有注释说明但不编造计算。'),
}
COMMENTS={
'from IPython.display import Image':'导入图片展示类，不启动模型。',
'import torch':'导入Tensor、自动求导和随机状态模块。','import numpy as np':'NumPy模块别名np。','import random':'Python随机模块。',
'from transformers import GPT2Tokenizer':'GPT2 slow tokenizer类，加载词表/merges。','from transformers import AutoModelForCausalLM':'因果LM工厂类，本例实际使用TRL包装。',
'from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer':'value包装、PPO配置与训练器，固定0.4.7。',
'def setup_seed(seed):':'定义函数，调用时修改几类随机状态。','torch.manual_seed(seed)':'设置Torch随机状态。','torch.cuda.manual_seed_all(seed)':'设置CUDA随机种子，不等于CUDA可用或运行GPU。','np.random.seed(seed)':'设置NumPy随机状态。','random.seed(seed)':'设置Python随机状态。','torch.backends.cudnn.deterministic = True':'约束相关cuDNN算法；不保证所有环境全确定性。','setup_seed(42)':'调用函数；Trainer之后可能按config.seed0重新设种。',
'model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")':'active基础LM+新增随机value head，CPU变体本地小模型。','model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")':'独立reference包装；新增head不保证与active相等。','tokenizer = GPT2Tokenizer.from_pretrained("gpt2")':'载入slow GPT2分词器；CPU变体使用相同词表导出的本地文件。','tokenizer.pad_token = tokenizer.eos_token':'pad复用EOS50256，词表大小不变。',
'tokenizer':'查看分词器对象。','model.pretrained_model':'查看基础LM，不含value head。','model.v_head':'查看critic的标量输出头。','ppo_config = {"batch_size": 1}':'覆盖rollout batch为1，其余仍默认。','config = PPOConfig(**ppo_config)':'**展开字典为关键字参数，默认seed0/ppo_epochs4。','ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)':'真实初始化Trainer、optimizer和reference路径。','model.pretrained_model.device':'设备属性，CPU小模型返回cpu。',
'query_txt = "This morning I went to the "':'原query末尾空格参与分词。','query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)':'编码int64[1,7]并移动到active设备。','query_tensor':'查看实际ID tensor。',
'generation_kwargs = {':'构造生成关键字参数字典。','"min_length": -1,':'历史设置，兼容性按固定4.30环境实测。','"top_k": 0.0,':'原意关闭top-k，是否接受浮点0按固定版本验证。','"top_p": 1.0,':'不额外截取nucleus概率质量。','"do_sample": True,':'按随机概率采样，不是argmax。','"pad_token_id": tokenizer.eos_token_id,':'padding采用EOS ID。','"max_new_tokens": 20,':'最多新增20个token，不含query。',
'response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=True, **generation_kwargs)':'真实rollout列表；True返回query+续写，后面不能原样当纯response。','response_txt = tokenizer.decode(response_tensor[0])':'第一返回样本ID解码为完整字符串。','response_txt':'显示包含query的结果，不代表微调质量。',
'reward = [torch.tensor(1.0, device=model.pretrained_model.device)]':'每样本0维float score，没有reward model forward。','train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)':'原错误边界保留：7+27拼34，默认四epoch真实更新。','train_stats':'统计dict，时间字段不可当数值等价依据。',
'input_ids = torch.cat([query_tensor[0], response_tensor[0]])':'沿一维拼接query与已含query的返回，重复输入。','print(query_tensor[0].shape)':'打印query长度7。','print(response_tensor[0].shape)':'打印完整返回长度，通常27。','print(input_ids.shape)':'打印拼接总长，通常34。','tokenizer.decode(input_ids)':'解码可见query重复。',
'base_model_output = model.pretrained_model(input_ids, output_hidden_states=True)':'真实基础LM forward，提供hidden但没有labels。','last_hidden_state = base_model_output.hidden_states[-1]':'取最后一层hidden，[T,H]。','print(last_hidden_state.shape)':'显示T/H，CPU小模型H16与历史H768分开。','lm_logits = base_model_output.logits':'取词表分数[T,50257]。','print(lm_logits.shape)':'打印词表维度，不代表value。','with torch.no_grad():':'上下文关闭梯度记录，不自动关闭dropout。','value = model.v_head(last_hidden_state).squeeze(-1)':'Linear(H,1)后只去掉最后单维，[T]。','print(value.shape)':'打印每位置value shape。','value':'查看标量tensor序列，不是RM评分。',
}

def code(text,i):
    out=['# '+NOTES[i][0]+'；原语句/历史输出保留。\n']
    for line in text.splitlines(True):
        key=line.strip()
        if key and not key.startswith('#') and key!='}':
            detail='读取原配图；宽度仅展示，原相对路径需要正确工作目录。' if key.startswith('Image(') else COMMENTS[key]
            out.append(line[:len(line)-len(line.lstrip())]+'# '+detail+'\n')
        out.append(line)
    return ''.join(out)

def markdown(text,i):
    out=[];inside=False;first=True
    for line in text.splitlines(True):
        if line.lstrip().startswith('```'):inside=not inside
        elif inside:
            if i==10 and (line.startswith('pip ') or line.startswith('git clone ') or line.startswith('python ')):
                out.append('# '+('安装库；原未锁版本。' if line.startswith('pip ') else '取示例源码；需与库版本匹配。' if line.startswith('git ') else 'SFT示例base7B/Guanaco/4bit/PEFT/micro4/K2；非本PPO路径，Bash续行保留。')+'\n')
            elif i==31 and line.strip() and not line.lstrip().startswith('#'):
                if first:out.append('# 这是缩略伪代码，不完整可执行；上下文、GAE和缩进层级被省略。\n');first=False
                out.append(line[:len(line)-len(line.lstrip())]+'# active/ref前向、reward shaping与minibatch/loss调用；具体变量见本格说明和主解释。\n')
        out.append(line)
    return ''.join(out)

def main():
    root=Path(__file__).resolve().parents[2];folder=root/'代码/tutorials/trl_hf';raw=json.loads((folder/'trl_basics.ipynb').read_bytes())
    assert len(raw['cells'])==42 and set(NOTES)==set(range(42))
    for name,changed in [('原代码_带说明.ipynb',False),('代码注释.ipynb',True)]:
        n=copy.deepcopy(raw);n['cells']=[dict(cell_type='markdown',metadata={},id='lesson16-intro',source=['# 第16讲说明与注释\n','原样例含重复query；修正另验证，不改原语句。\n'])]
        for i,c in enumerate(raw['cells']):
            title,detail=NOTES[i];s='## 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],title,detail)
            n['cells'].append(dict(cell_type='markdown',metadata={},id='lesson16-note-'+str(i),source=s.splitlines(True)))
            saved=copy.deepcopy(c);s=''.join(c['source'])
            if changed:s=code(s,i) if c['cell_type']=='code' else markdown(s,i)
            saved['source']=s.splitlines(True);n['cells'].append(saved)
        (folder/name).write_bytes((json.dumps(n,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    doc='# 第16讲代码单元解释\n\n全部42格/28代码格（含空41）、Markdown10命令与31概览片段。\n\n[主解释](解释.md) · [阅读入口](README.md)\n\n';py=['# -*- coding: utf-8 -*-','# 第16讲原PPO代码阅读版，远程GPT2与重复query保留，修正验证另记。','']
    for i,c in enumerate(raw['cells']):
        title,detail=NOTES[i];s=''.join(c['source']);note='## 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],title,detail)
        doc+=note+'\n';py+=['# %% '+str(i)]+['# '+x for x in note.splitlines()]
        if c['cell_type']=='code':doc+='```python\n'+s+'\n```\n\n';py+=code(s,i).splitlines()
        else:doc+=markdown(s,i)+'\n\n';py+=['# '+x for x in markdown(s,i).splitlines()]
        py+=['']
    (root/'代码单元解释.md').write_bytes(('\n'.join(x.rstrip() for x in doc.splitlines()).rstrip()+'\n').encode('utf-8'))
    (folder/'trl_basics_代码注释.py').write_bytes(('\n'.join(x.rstrip() for x in py).rstrip()+'\n').encode('utf-8'))
    print('42 notes/28 code including empty, two85-cell notebooks and reading Python built.')
if __name__=='__main__':main()
