"""保留固定上游单元，只插入中文注释及说明；不执行 notebook。"""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parents[2]
BACK = Path('G:/zzz/Personal ChatGPT 系列/27 TRL RLHF PPOTrainer 与 OpenRLHF/资料')
FOLDER = ROOT / '代码/tutorials/trl_hf'
NOTES = [
    '设置本进程代理与 NCCL 环境变量，再导入图片展示类。代理127.0.0.1:7890不是通用网络配置；两条NCCL开关只在相应后端生效，不能让CPU版PyTorch获得CUDA。复现跳过此格，使用离线文件。',
    '导入PyTorch、GPT2Tokenizer和旧版TRL三个核心类。AutoModelForCausalLMWithValueHead包装语言模型，并从最后一层hidden states另接标量价值头；PPOConfig保存超参，PPOTrainer组织生成和优化。本讲锁定TRL0.8.6，不将新版API套入旧代码。',
    '只有注释，不执行shell。!export是Unix/IPython风格，Windows不能直接照搬；上一格使用os.environ才实际修改进程环境。',
    '整体结构标题。先分清Actor、Reference、Critic和外部Reward Model，再读PPO更新。',
    '原链接指向可变main；本交付将TRL锁到0.8.6、OpenRLHF锁到2024-05历史提交。PPO-penalty和PPO-clip是原论文两种surrogate。本讲实际实现使用clip优化，另外在逐token reward中惩罚偏离固定reference，不能把这两个KL约束混为一谈。',
    '展示原OpenRLHF框架图。url参数只保存相对URL，构造Image不等于浏览器已加载图片；本交付提供真实图片。Actor生成response，Actor/Reference计算logprob，Critic计算value，Reward Model计算整句score，组装experience后更新Actor/Critic。',
    '模型对比标题。ref是固定参考模型；old是当前rollout的行为策略快照。',
    '分别加载两份GPT2包装器。不是变量别名；两份语言模型从同一预训练权重起步，两个随机value head不保证相同。ref的value输出不会参与GAE。CPU验证仅替换两个加载路径，使用同一随机小GPT2 checkpoint，不能当原GPT2预训练效果。',
    '价值头作用于[B,L,H]，不是把词表概率[B,L,V]压到1。Linear(H,1)后squeeze(-1)得[B,L]。纠正原文：model_ref对应固定pi_ref/SFT；old_logprobs才对应每轮更新前的pi_old。自适应KL控制器截断的是相对误差，不是把KL值直接限制到±0.2。',
    '展示自适应beta公式。beta乘以1+clip(KL/target-1,-0.2,0.2)*n_steps/horizon。n_steps是经验条数；在正常步长下KL偏高会提高beta。它没有把参数硬投影回某个KL球。',
    '只有注释# model，Notebook不会展示模型，也不改变状态。',
    '读取语言模型参数所在设备；这是属性访问，不是搬运模型。无GPU的实测为cpu。',
    '从GPT2加载真实BPE tokenizer；最初pad_token为None，EOS为50256的特殊字符串。把pad指向EOS复用现有ID，不增加词表；attention_mask仍须区分真实EOS与padding。',
    'encode将字符串变成ID列表，decode做逆映射。特殊EOS字符串在该tokenizer里编码成单个50256，不等于整句只允许该ID。',
    'vocab_size返回50257，不是输入长度、hidden size或模型参数量。',
    '字典**ppo_config解包为关键字参数。mini_batch_size=1/batch_size=1表示每次收一条经验、每个mini batch一条；ppo_epochs仍取旧库默认4，所以一次step不只一次Adam更新。PPOTrainer通过Accelerate准备模型/优化器/参考模型。',
    '只有注释：提示真正的设备处理发生在PPOTrainer.__init__，此格不执行prepare。',
    'print两个设备。CPU验证两者均cpu；设备相同不代表两者参数都会被优化。应实际核对optimizer参数集合与ref状态。',
    'query_txt为prompt字符串。encode(return_tensors="pt")得int64[1,Q]，to只转设备；真实GPT2 BPE此句Q=7。尾部空格也参与分词。',
    '生成配置：min_length=-1不要求正最短长度；top_k=0关闭top-k筛选，top_p=1关闭nucleus截断，do_sample=True进行随机采样；max_new_tokens=20是新增token上限，不是固定生成20。EOS可提前结束。',
    '进入生成阶段：使用active model，不使用ref，也不更新参数。',
    'list(query_tensor)沿batch维拆成一个int64[Q]张量的列表，不是把token转换成Python整数列表。',
    'generate接收list[Tensor[Q]]。return_prompt=False明确只返回新增response；与第16讲return_prompt=True不同，此处再拼query不会重复prompt。decode接受第一条响应ID，产出str。',
    'Notebook展示response_txt。随机小模型输出只用于核对接口，不能评价语言能力；历史输出属于上游运行。',
    '整格均为注释。直接model.generate通常会返回prompt+response，不能把该注释示例当作本格实际执行或等价的纯response接口。',
    '外部人工指定分数1.0，scores为长度B的list，元素是float32零维tensor。这里只模拟Reward Model接口，没有训练或调用真实偏好评分模型；所有回答常数1不提供质量区分。',
    '进入PPO step：准备输入、缓存旧策略logprob/value、固定ref前向、KL形状奖励、GAE、多个epoch的clip/value loss和反向更新。',
    '关键词queries/responses/scores明确三种输入。外层list长度B=1，内层query/response是各自一维ID。step返回包含loss、KL、ratio和耗时等统计的dict，同时实际修改active模型/optimizer及KL控制器状态。',
    'Markdown保存的是不完整step摘录，不能当独立函数运行。实际完整PPOTrainer见源码阅读版。令L=Q+T，logits[B,L,V]；log_softmax+gather和shift得[B,L-1]，不是词表维分布。response mask只保留T个动作；KL默认采样log比可负。外部score只加到最后有效token。GAE先用未白化优势算returns，再白化优势用于policy loss。',
    '展示原TRL PPO loss图。两支相加：policy clip loss与vf_coef乘value loss；旧0.8.6代码记录entropy，但没有默认把它作为第三个entropy bonus加进loss。',
    'ratio=exp(new_logprob-old_logprob)是当前策略与rollout旧策略对同一个已采样动作的概率比。这里分母不是固定reference；使用log差避免先算极小概率再相除，但exp仍可能溢出，不能宣称绝对数值安全。',
    'for逐一迭代统计字典的键并打印；keys()返回可迭代视图。某些字段只是诊断指标，不都参与autograd或loss。',
    '空代码单元：没有输入输出，不运行训练。保留原位置，避免单元编号错位。',
]

LINE_HINTS = {
    "os.environ": '修改当前Python进程及其子进程环境；不是修改系统全局设置。',
    'from_pretrained': '读取模型/分词器资产；默认远程名称可能访问网络，验证只使用明确本地路径。',
    'PPOConfig(': '构造旧API配置对象；未指定字段仍保留该版本默认值。',
    'PPOTrainer(': '绑定active模型、固定参考模型、tokenizer及优化器，建立训练状态。',
    'return_prompt=False': '返回纯新增token，后续step再拼prompt时不会重复。',
    'torch.tensor(1.0': '零维浮点分数，没有从生成文本得到的计算图；不是序列标签。',
    'train_stats =': '真正执行PPO训练，返回统计字典且更新参数。',
    'tokenizer.pad_token =': '复用EOS作为padding，不新增embedding行。',
    'query_tensor =': '先得到int64[1,Q]，再放到语言模型设备。',
    'list(query_tensor)': '沿第0维拆batch；列表里的每个元素仍是一维tensor。',
    'Image(': '构造展示对象；相对路径以notebook位置为基准。',
    'for key in': '迭代字典键；这里只打印已有统计，不追加一次训练。',
    'print(': '输出用于人工检查，不改变模型权重。',
}

CORE_HINTS = {
    'input_ids = [torch.cat': '每条经验拼接query和纯response，再由collator补齐batch；不把query算作response动作。',
    'input_data.pop("labels"': '去掉SFT标签；此处优化的是PPO目标，不直接用语言模型自带CE loss。',
    'model.eval()': '关闭dropout；eval不等于no_grad，是否记录图由外层上下文决定。',
    'logits, _, values = model': '包装器输出logits[B,L,V]和value[B,L]；中间LM loss在这里不用。',
    'logprobs = logprobs_from_logits': '位置t预测token t+1；log_softmax后按真实采样token gather成[B,L-1]。',
    'start = len(query_batch': 'shift后第一个response动作位于Q-1；注意不是Q。',
    'start += attention_mask': '左padding会把有效序列起点右移，mask边界也必须同步移动。',
    'end = start +': '有效response区间是[start,end)，右端不包含。',
    'masks[j, :start]': '清除prompt动作位置；训练不会把prompt当生成动作优化。',
    'masks[j, end:]': '清除response后padding；pad与EOS复用ID仍依靠位置mask区分。',
    'last_non_masked_index': '寻找最后有效response动作；全零mask会索引失败，须在调用前保证非空。',
    'reward[last_non_masked_index]': '整句外部score只在终止动作注入一次，不是每个token重复加一遍。',
    'non_score_reward =': '逐token负beta乘KL样本；默认样本log比可能负，因此该奖励也可能正。',
    'return logprob - ref_logprob': '单次采样log概率比；对当前策略采样求期望才是KL，不保证逐项非负。',
    'F.kl_div': 'full分支输入全词表log分布，计算词表求和的真正条件KL。',
    'values = values * mask': '把无效位置value归零后向后递推；不是删除序列维。',
    'rewards = rewards * mask': '无效位置reward归零；所有masked统计仍须使用相同mask。',
    'for t in reversed': 'GAE从右向左递推，用未来TD残差改善早期动作的信用分配。',
    'nextvalues =': '序列最后位置采用终止bootstrap=0；不能直接套到被截断但未终止的任意环境。',
    'delta =': 'TD残差r_t+gamma*V_{t+1}-V_t；三项同shape[B]。',
    'lastgaelam = delta': '累计delta_t+gamma*lambda*A_{t+1}；lambda控制偏差/方差折衷。',
    'returns = advantages + values': 'value回归目标用未白化GAE加旧value，之后才白化policy优势。',
    'advantages = masked_whiten': '只在有效动作上估计均值/方差，再标准化；原始mask外值不一定为零。',
    'advantages = advantages.detach': '优势作为固定target，不让policy反传穿过旧value/GAE计算图。',
    'ratio = torch.exp': '分母是rollout旧策略old_logprobs；不使用固定ref_logprobs。',
    'pg_losses =': '负号把最大化surrogate变成最小化loss；正优势提高动作概率。',
    'pg_losses2 =': 'clip概率比到[1-eps,1+eps]；结合max形成悲观surrogate。',
    'pg_loss = masked_mean': '对有效response token平均，prompt/padding不贡献目标。',
    'vpredclipped =': '把新value限制在旧value附近作为另一条回归支路。',
    'vf_losses1 =': '当前value对固定returns的平方误差。',
    'vf_losses2 =': '裁剪value对同一returns的平方误差；不能换成另一套target。',
    'vf_loss =': '两种误差取更大再乘1/2，不是取更小或普通单支MSE。',
    'self.config.vf_coef': 'critic项加权；源码loss返回的value项已经乘系数，调用处不要再乘第二次。',
    'avg_ratio >': '极端平均ratio触发跳过，将本mini batch损失归零；不是KL硬约束。',
    'entropy = masked_mean': '这里记录熵统计；默认loss表达式没有熵奖励项。',
    'self.accelerator.backward': '实际autograd反传，Accelerate处理混合精度/梯度累积等环境。',
    'self.optimizer.step()': '更新optimizer拥有的active参数；ref不应出现在其参数组。',
    'self.optimizer.zero_grad': '清理梯度，累积细节交由Accelerate包装器管理。',
    'with torch.no_grad()': 'rollout旧策略数据不记录梯度；后续mini batch重新前向才用于反传。',
    'old_logprobs': 'rollout阶段保存的行为策略log概率，在多个PPO epoch内保持固定。',
    'ref_logprobs': '固定reference/SFT策略的log概率，仅用于偏离惩罚，不作PPO ratio分母。',
    'kl_ctl.update': '用本次统计KL更新下一次reward使用的beta，不直接修改Actor参数。',
    'proportional_error =': '截断KL相对target的误差到±0.2，不是截断KL本身。',
    'self.value *= mult': '原地更新控制器beta；不同于optimizer学习一个可微参数。',
    'last_hidden_state =': '取最后一层hidden states[B,L,H]，不是词表logits[B,L,V]。',
    'value = self.v_head': 'Linear(H,1)经squeeze(-1)变为[B,L]；仅移除最后的单例轴。',
    'self.summary = nn.Linear': '价值头每个位置共享同一组H+1个参数。',
    'logp = F.log_softmax': '沿词表轴归一化；log_softmax比先softmax再log更稳定。',
    'logpy = torch.gather': 'labels[B,L]扩成[B,L,1]选出各位置实际token的log概率。',
    'bessel_correction =': '无偏方差乘N/(N-1)；有效token仅1时除零是本旧版真实边界。',
}

def write(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix != '.ipynb':
        protected=set()
        if p.suffix=='.py':
            for token in tokenize.generate_tokens(io.StringIO(s).readline):
                if token.type==tokenize.STRING and token.end[0]>token.start[0]:
                    protected.update(range(token.start[0],token.end[0]+1))
        s=''.join(line if i in protected else line.rstrip()+'\n' for i,line in enumerate(s.splitlines(True),1)).rstrip()+'\n'
    p.write_text(s, encoding='utf-8', newline='\n')

def annotate(source, hints):
    """只在多行字符串外插入注释；任何原有字符串/语句均保留。"""
    protected = set()
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type == tokenize.STRING and t.end[0] > t.start[0]:
            protected.update(range(t.start[0], t.end[0]+1))
    out = []
    for i, line in enumerate(source.splitlines(True), 1):
        stripped = line.strip()
        if i not in protected and stripped and not stripped.startswith('#'):
            for key, hint in hints.items():
                if key in line:
                    indent = line[:len(line)-len(line.lstrip())]
                    out.append(indent+'# 中文说明：'+hint+'\n')
                    break
        out.append(line)
    return ''.join(out)

def main():
    raw = json.loads((FOLDER/'trl_ppotrainer_helloworld.ipynb').read_bytes())
    assert len(raw['cells']) == len(NOTES) == 33
    derivatives = [copy.deepcopy(raw), copy.deepcopy(raw)]
    for n in derivatives:
        n['cells'] = [{'cell_type':'markdown','metadata':{},'source':['# 第27讲：PPOTrainer 代码说明\n','原始输出是上游历史记录；本次验证见复现目录。单元从0编号。\n']}]
    reading = ['# 第27讲 Python 阅读版：来自固定上游notebook，新增内容均为注释。\n# 实际可执行验证请使用复现/scripts/verify_delivery.py；本文件保留远程加载及环境设置。\n']
    unitdoc = ['# 第27讲全部单元说明\n\n原文件33格，单元从0编号。Markdown摘录不是完整函数，历史输出不等于本次运行。\n']
    for i, c in enumerate(raw['cells']):
        text = ''.join(c['source'])
        explanation = f'## 单元 {i}（{c["cell_type"]}）\n\n{NOTES[i]}\n\n'
        unitdoc.append(explanation + '~~~'+('python' if c['cell_type']=='code' else 'markdown')+'\n'+text+'\n~~~\n\n')
        for n, add in zip(derivatives,[False,True]):
            n['cells'].append({'cell_type':'markdown','metadata':{},'source':explanation.splitlines(True)})
            cell = copy.deepcopy(c)
            if add and c['cell_type']=='code':
                source = '# 本格说明：'+NOTES[i]+'\n'+annotate(text, LINE_HINTS)
                assert ast.dump(ast.parse(text)) == ast.dump(ast.parse(source))
                cell['source'] = source.splitlines(True)
            n['cells'].append(cell)
        reading.append('\n# ===== 原单元 '+str(i)+' =====\n# '+NOTES[i]+'\n')
        reading.append(annotate(text,LINE_HINTS)+'\n' if c['cell_type']=='code' else ''.join('# '+l+'\n' for l in text.splitlines()))
    for name,n in zip(['原代码_带说明.ipynb','代码注释.ipynb'],derivatives):
        write(FOLDER/name,json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    write(FOLDER/'trl_ppotrainer_helloworld_代码注释.py',''.join(reading))
    write(ROOT/'代码单元解释.md',''.join(unitdoc))
    selected = {
        'trl/trainer/ppo_trainer.py': ['PPOTrainer'],
        'trl/trainer/utils.py': ['AdaptiveKLController','FixedKLController'],
        'trl/models/modeling_value_head.py': ['ValueHead','AutoModelForCausalLMWithValueHead'],
        'trl/core.py': ['logprobs_from_logits','masked_mean','masked_var','masked_whiten','clip_by_value'],
    }
    pieces = ['# TRL 0.8.6 完整核心类/函数阅读版（中文注释）\n# 来源Hugging Face TRL，Apache-2.0，许可证见LICENSE-trl。\n# 仅源码阅读：省略原模块imports，不能单独import作为替代库。\n# 各对象完整保留，PPOTrainer所有方法均在；验证使用安装的真实TRL而不是本阅读副本。\n']
    manifest = {'upstream_commit':'7fbc29824171fd07dc2c30fe922e61ebc018ad1d','notebook':'tutorials/trl_hf/trl_ppotrainer_helloworld.ipynb','files':{},'core_objects':selected,'trl_version':'0.8.6','openrlhf_commit':'fbc1eab137c18651b28e1fa8d9b01c592e130b41','openrlhf_commit_selection':'2024-06-01以前的最后一次提交；用于历史框架对照，非断言讲师版本','subtitle':'E:/cfebaaee-f078-4e1d-9bbe-374686097bf3.txt（仅本地保存）'}
    for f in [FOLDER/'trl_ppotrainer_helloworld.ipynb',*(ROOT/'代码/imgs').glob('*.png')]:
        manifest['files'][f.relative_to(ROOT).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    for rel, names in selected.items():
        p=BACK/'trl-0.8.6'/rel;source=p.read_text('utf-8');manifest.setdefault('trl_source_sha256',{})[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        header=[]
        for line in source.splitlines(True):
            if line.startswith('#') or not line.strip():header.append(line)
            else:break
        pieces.append('\n# 原文件版权与许可声明：'+rel+'\n'+''.join(header))
        for node in ast.parse(source).body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
                chunk=''.join(source.splitlines(True)[min([node.lineno]+[x.lineno for x in node.decorator_list])-1:node.end_lineno])
                decorated=annotate(chunk, CORE_HINTS)
                assert ast.dump(ast.parse(chunk))==ast.dump(ast.parse(decorated))
                pieces.append('\n# ===== '+rel+' :: '+node.name+' =====\n'+decorated+'\n')
    write(FOLDER/'PPOTrainer核心源码_代码注释.py',''.join(pieces))
    (FOLDER/'LICENSE-trl').write_bytes((BACK/'trl-LICENSE').read_bytes())
    pieces=['# OpenRLHF 历史提交 fbc1eab137c18651b28e1fa8d9b01c592e130b41 源码阅读\n# Apache-2.0，见LICENSE-OpenRLHF。省略imports，仅作阅读，不启动Ray/DeepSpeed。\n']
    selected_open = {'openrlhf-models-utils.py':['compute_approx_kl','compute_reward','log_probs_from_logits','masked_mean'], 'openrlhf-loss.py':['PolicyLoss','ValueLoss'], 'openrlhf-experience_maker.py':['NaiveExperienceMaker.make_experience','NaiveExperienceMaker.get_advantages_and_returns']}
    for filename,names in selected_open.items():
        p=BACK/filename;text=p.read_text('utf-8');manifest.setdefault('openrlhf_source_sha256',{})[filename]=hashlib.sha256(p.read_bytes()).hexdigest()
        for name in names:
            node=ast.parse(text)
            for part in name.split('.'):
                node=next(n for n in node.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==part)
            import textwrap
            chunk=''.join(text.splitlines(True)[min([node.lineno]+[x.lineno for x in node.decorator_list])-1:node.end_lineno])
            # 方法提升到模块层时仅去掉语法缩进；不能改变docstring内部的空格。
            protected=set()
            for token in tokenize.generate_tokens(io.StringIO(chunk).readline):
                if token.type==tokenize.STRING and token.end[0]>token.start[0]:
                    protected.update(range(token.start[0]+1,token.end[0]+1))
            indent=len(chunk.splitlines()[0])-len(chunk.splitlines()[0].lstrip())
            chunk=''.join(line if i in protected else line[indent:] for i,line in enumerate(chunk.splitlines(True),1))
            hints={**CORE_HINTS,'action_log_probs =':'Actor对本轮rollout动作的旧log概率；之后放入Experience，更新阶段作为固定分母。','base_action_log_probs =':'固定initial_model前向；它对应TRL ref，不是每个mini batch更新的old快照。','r = self.reward_model':'独立Reward Model输出每条序列标量；与逐token critic value区分。','r = r.clamp':'该历史OpenRLHF将整句分数截断到[-10,10]，与TRL此处默认行为不同。','log_ratio =':'单样本log比，乘action_mask后返回；仍可能为负。','device="cuda"':'原make_experience硬编码CUDA，CPU机器不执行此完整方法。','eos_indices =':'反向查最后有效动作；全零mask没有合法终止动作，必须在外层保证。','return Experience(':'把本轮数据存入结构体供若干PPO epoch使用，不意味着无限复用陈旧经验。'}
            pieces.append('\n# ===== '+filename+' :: '+name+' =====\n'+annotate(chunk,hints)+'\n')
    write(FOLDER/'OpenRLHF核心源码_代码注释.py',''.join(pieces))
    (FOLDER/'LICENSE-OpenRLHF').write_bytes((BACK/'openrlhf-LICENSE').read_bytes())
    manifest['openrlhf_core_objects']=selected_open
    write(ROOT/'来源清单.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Built 33-cell explanations, 2 derivatives (67 cells each), 2 Python readings and full core objects.')

if __name__ == '__main__':
    main()
