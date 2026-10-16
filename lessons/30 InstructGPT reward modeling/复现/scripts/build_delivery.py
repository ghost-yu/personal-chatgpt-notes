"""固定两原notebook和原图；只加注释的派生版与可读Python。"""
import copy
import hashlib
import importlib.metadata as metadata
import inspect
import io
import json
from pathlib import Path
import subprocess
import tokenize

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=Path('G:/zzz/Llama 入门教程/代码/personal_chatgpt')
COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
CODE=ROOT/'代码/tutorials/reward_model'
MAIN='reward_model_Bradley_Terry_model'
SUP='bradley_terry_modeling'
NOTES={MAIN:[
 '原参考入口：Bradley–Terry概念、OpenAI InstructGPT介绍和论文。外链是来源，不是指令；本次以固定InstructGPT v1的§3.5/附录C.2为准。',
 '从IPython.display导入Image类，供notebook展示图片；导入不下载权重。',
 'torch/nn/F提供Tensor与稳定logsigmoid，NumPy计算数组，pyplot画函数。这里nn虽导入但原notebook没有神经网络模型/训练；不要编造原模型。',
 'InstructGPT标题；本文分析其reward modeling，不将2022年论文当作ChatGPT全部工程实现公开。',
 '远程图Image(url=...)只构造HTML对象。本次未验证远程图内容，不能根据其URL宣称看过图片或把它当实验结果。',
 '论文Eq.(1)：delta是同prompt下chosen与rejected的标量reward之差；每对损失为-log sigmoid(delta)。binom(K,2)表示K个候选的无序比较对上限；实现要明确sum与mean边界，已经对pair求mean后不能再无意除一次组合数。',
 '原OpenAI流程图远程URL保留；只验证Image对象。流程用论文Figure 2的文字说明核对：SFT→偏好排序/训练RM→PPO。预训练基础模型在三个阶段之前，不能把第1阶段直接等同于从头预训练。',
 'D>C>A=B有4个候选：总无序对6，其中A/B平局。按论文丢弃ties，只留下5个严格偏好对；不把平局随意设成chosen标签1。',
 '本地quality-labels.png使用url参数构造HTML相对链接；cwd需为本notebook目录。原图实际是单输出质量1–7分和若干属性的标注界面，不是四候选排序界面；原Git字节保留并已视觉查看。',
 '单个绝对rating可能受标注人打分尺度影响，pairwise更关注相对选择，但不保证消除所有偏好分歧。siamese类比指同一个参数共享的打分器分别处理(x,yw)/(x,yl)，不等于两个独立reward网络，也不一定是cross-encoder一次输入两答案。',
 '损失理解标题；后续从概率假设推导，不把分数本身当概率。',
 '概率建模标题；先建立人类偏好的Bernoulli变量，再推导似然。',
 '假定logit(p)=rw-rl=delta。先指数化得odds，再解p=sigmoid(delta)，最后对观察到chosen胜出的标签1取负log。不是“模型必然已给chosen更大分”的假设；delta可以为负。',
 'p是给定prompt和两响应时人类偏好chosen的模型概率，不是确定标量rw>rl这个事件的随机概率。BCE一般形式含y和1-y；按chosen/rejected定向后y=1。自然对数概率可以用于似然/交叉熵，但看到log并不总是交叉熵。',
 'np.linspace取100个概率点[.01,.99]，shape[100] float64；np.log为自然对数，log_odds=log(p/(1-p))。plt.figure创建画布、plot绘二维曲线、xlabel/ylabel/title/grid只设置显示。不是训练曲线，没有需要学习的参数。',
 '多个候选标题；同prompt候选间比较高度相关，组合数不等于独立样本信息量。',
 'K=9最多36对、64个prompt最多2304对，K=4最多6对。论文K范围4–9、ties丢弃。把每prompt的全部比较放入同一batch元素可复用每个候选的一次前向；不应据此断言“多造pair一定抗过拟合”或跨prompt大小不同也均匀加权。',
 'math.comb(9,2)=36是整数无序组合数，乘64返回int2304；不是64×9条独立prompt，也不包括反向重复对。',
 'Bradley–Terry标题；补充notebook给出正强度pi与log强度theta的极大似然估计。',
 '远程Bradley–Terry示意图只构造HTML，不验证内容；概率公式由原Markdown与实际损失实现独立核对。',
 '损失对delta的导数标题；需要分清对分差、对两个分数、对网络参数的导数。',
 'ell=softplus(-delta)，ell_prime=sigmoid(delta)-1，ell_second=sigmoid(delta)*(1-sigmoid(delta))。原“loss is proportional to log odds”不准确：delta才是log odds，loss是其非线性函数。',
 '相等分数delta=0：-F.logsigmoid是自然对数loss ln2=.693147；第二项np.log10给.301030。二者对数底不同，不是两个算法给同一单位的loss。该格原样保留，说明中纠正。',
 'delta为1000个float64点[-10,10]；用sigmoid、负log和sigma-1得到两条解析函数曲线。蓝线loss非负，红线导数在(-1,0)，不是两次训练。np.exp/先sigmoid再log在极端输入会溢出，补充用稳定softplus/logsigmoid核对。',
 'delta=3时loss=.048587、梯度约-.047426，仍非0；不是硬阈值，也不是达3后无法继续优化。有限实数上loss始终随delta减小，极限delta→+inf时趋0；有限精度数值饱和另说。',
 '依次计算delta=1/3/-6的自然对数loss，约.313262/.048587/6.002476；负差惩罚大，对delta梯度趋-1而非无界增大。',
 'delta=-6的sigmoid约.002473，表示模型当前认为chosen胜出的概率很低，不改变人工chosen标签；它推动训练修正错误排序。',
],SUP:[
 '导入NumPy与真实SciPy optimize.minimize。主CPU环境原无SciPy，本次固定1.13.1仅安装到本讲本地runtime-libs，未替换共享NumPy/Torch。',
 '参考链接保留；chatgpt.com会话链接未读取，不能当已核实论文或指令。补充是四队/五品牌BT估计，与神经reward模型的参数化方式不同。',
 'w[i,j]为i胜j次数，n=w+w.T对称。固定theta_A=0消除整体平移自由度，优化其余3个log强度。每对i<j只算一次；BFGS数值梯度最小化负对数似然。外层最多10次重复minimize，打印的2次是外层调用数，不是BFGS内部迭代。原循环在收敛分支break前不写回result.x，本次记录两者微小差异。',
 '五品牌成对比较表，单元行列是胜出次数；每对总数100，共10个无序对/1000次比较。表中归一化强度不能当作任意一场比较的胜率。',
 '正强度pi_i>0，Pij=pi_i/(pi_i+pi_j)；令theta_i=log pi_i便等于sigmoid(theta_i-theta_j)。似然按无序对求和，若对有序i,j全求和就需防止重复计数。',
 'data的四元组是(i,j,n_ij,n_ji)。固定最后品牌log_pi4=0，优化前4个。max-shift log-sum-exp避免直接exp大数；函数返回负LL标量，没有更新参数。Python解包/for循环逐对累积，brand_indices字典在此例未用于核心计算。',
 'x0四维零向量；minimize(method=BFGS)真实优化，无解析jac时用数值差分。result.success必须检查；exp后归一化强度只是一种定标，不是成对胜率矩阵。这里第二次定义的neg_log_likelihood覆盖了前面同名函数，分开阅读两个任务。',
 '原空白代码格保留；没有隐藏的训练/评估逻辑。',
]}

def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8',newline='\n')

def reading_whitespace(s):
    # 阅读排版去行尾空格；多行字符串内部逐字保留，原notebook不经过此函数。
    protected=set()
    for t in tokenize.generate_tokens(io.StringIO(s).readline):
        if t.type==tokenize.STRING and t.start[0]!=t.end[0]:
            protected.update(range(t.start[0],t.end[0]+1))
    return '\n'.join(line if i in protected else line.rstrip()
                     for i,line in enumerate(s.splitlines(),1))+'\n'
def note(line):
    rules=[
      ('from IPython','导入实际Image展示对象；不验证远程图片内容。'),
      ('import ','导入本段实际用到的模块或函数，不在此行下载模型。'),
      ('Image(','构造HTML图对象；url只是来源或相对路径。'),
      ('np.linspace','创建包含两端点的均匀float64数组，长度为num参数。'),
      ('log_odds =','自然对数赔率；p/(1-p)是odds，结果不是概率。'),
      ('plt.figure','创建新图；figsize单位是英寸。'),
      ('plt.plot','绘制横轴与纵轴同长度的数组；label用于图例。'),
      ('plt.xlabel','设置横轴含义，不参与数值计算。'),
      ('plt.ylabel','设置纵轴含义；区分loss与梯度的数值。'),
      ('plt.title','设置图标题，不是模型训练名称。'),
      ('plt.grid','显示辅助网格。'),
      ('plt.legend','显示前面plot的label图例。'),
      ('plt.axhline','在指定y处绘水平参考线。'),
      ('plt.axvline','在指定x处绘垂直参考线。'),
      ('plt.show','展示真实matplotlib图；本次Agg后端另保存同一figure。'),
      ('sigma =','sigmoid(delta)是偏好概率；逐元素exp，大负delta可能溢出。'),
      ('neg_log_sigma =','逐元素负自然对数；极端概率0会产生inf。'),
      ('grad_neg_log_sigma','对delta的解析导数，sigmoid(delta)-1；非训练日志。'),
      ('F.logsigmoid','PyTorch稳定计算log(sigmoid)；第二项log10不是同底数。'),
      ('math.comb','无序组合数C(9,2)=36，64个prompt至多2304对。'),
      ('theta_full =','把被固定的基准log强度0拼回完整向量。'),
      ('exp_theta =','log强度指数化为正强度；此原实现大输入可能overflow。'),
      ('for i in range','遍历对象下标i；后面j从i+1起避免重复一对。'),
      ('for j in range','只枚举无序对i<j。'),
      ('if n[i, j]','没有交手数据的对象对不贡献似然。'),
      ('w[i, j] *','i获胜次数乘i的log强度。'),
      ('w[j, i] *','j获胜次数乘j的log强度。'),
      ('n[i, j] *','减去总次数乘log(exp(theta_i)+exp(theta_j))。'),
      ('return -L','取负号供最小化器使用；最大似然变为最小负似然。'),
      ('neg_log_likelihood(','定义目标或将目标函数交给真实SciPy；不是直接求解析解。'),
      ('theta_opt = np.zeros','三维自由log强度初始化为0，基准A另固定。'),
      ('for iteration','外层重复调用完整BFGS，不能当作BFGS内部迭代次数。'),
      ('result = minimize','真实BFGS优化，未传jac时使用数值差分。'),
      ('if not result.success','检查优化器是否成功，不静默接受失败结果。'),
      ('raise Exception','失败时中断此格并保留消息。'),
      ('np.all(np.abs','比较本次与上次自由参数逐项差，阈值tolerance。'),
      ('theta_opt = result.x','记录本次最优自由参数；break分支在此行之前。'),
      ('p_normalized =','强度除总和只做定标，不能代替两者比较的概率。'),
      ('p = np.exp','正强度exp(theta)，以A=1为基准。'),
      ('log_pis_full =','固定最后品牌log强度0，拼成长度5的完整数组。'),
      ('for i, j, n_ij, n_ji','四元组解包：两个品牌下标与两方向观测次数。'),
      ('max_log_pi_pj =','两数取max供稳定log-sum-exp移位。'),
      ('log_sum_ij =','先减max再exp，最后加回max，避免正指数overflow。'),
      ('LL +=','累加当前比较对的binomial log likelihood（省略与参数无关常数）。'),
      ('x0 =','四个自由log强度的起点；不是五个品牌强度本身。'),
      ('if result.success','只有成功时才报告估计参数。'),
      ('log_pis_est =','把固定为0的最后一维补回。'),
      ('pis_est = np.exp','把log强度还原为正强度。'),
      ('pis_est_normalized =','把正强度归一化；成对概率仍需pi_i/(pi_i+pi_j)。'),
      ('for idx, brand','enumerate同时给位置和品牌名。'),
    ]
    for pat,msg in rules:
        if pat in line:return msg
    return None

def annotate(src,intro):
    protected=set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type==tokenize.STRING and t.start[0]!=t.end[0]:protected.update(range(t.start[0],t.end[0]+1))
    out=['# '+intro]
    for i,line in enumerate(src.splitlines(),1):
        msg=note(line) if i not in protected and not line.lstrip().startswith('#') else None
        if msg:out.append(line[:len(line)-len(line.lstrip())]+'# '+msg)
        out.append(line.rstrip() if line.lstrip().startswith('#') else line)
    return '\n'.join(out)+'\n'

def build():
    CODE.mkdir(parents=True,exist_ok=True);manifest={'upstream_commit':COMMIT,'files':{}}
    docs=['# 第30讲：主与补充notebook全部单元\n\n单元各自从0编号。原文/历史输出保留，新增说明区分错误与本次证据。\n']
    for name in [MAIN,SUP]:
        rel='tutorials/reward_model/'+name+'.ipynb';raw=subprocess.check_output(['git','-C',str(UPSTREAM),'show',COMMIT+':'+rel]);(CODE/(name+'.ipynb')).write_bytes(raw)
        manifest['files']['代码/'+rel]=hashlib.sha256(raw).hexdigest();nb=json.loads(raw);assert len(nb['cells'])==len(NOTES[name])
        reading=['# 原notebook Python中文注释阅读版：'+name,'# 所有原代码保留；绘图使用matplotlib，补充BFGS需要真实SciPy。']
        for mode in ['annotated','explained']:
            d=copy.deepcopy(nb);cells=[{'cell_type':'markdown','id':name[:20]+'-head','metadata':{},'source':['# 第30讲：'+name+'\n原历史输出不是本次运行结果；新增说明纠正原文不严谨处。']}]
            for i,c in enumerate(nb['cells']):
                cells.append({'cell_type':'markdown','id':name[:20]+f'-note-{i}','metadata':{},'source':[f'### 原单元 {i} · {c["cell_type"]}\n\n{NOTES[name][i]}\n']});n=copy.deepcopy(c)
                if mode=='annotated' and c['cell_type']=='code':n['source']=annotate(''.join(c['source']),NOTES[name][i]).splitlines(keepends=True)
                cells.append(n)
            d['cells']=cells
            output=('代码注释' if mode=='annotated' else '原代码_带说明')+'.ipynb'
            if name==SUP:output=SUP+'_'+output
            write(CODE/output,json.dumps(d,ensure_ascii=False,indent=1)+'\n')
        docs.append('\n## '+name+'\n')
        for i,c in enumerate(nb['cells']):
            src=''.join(c['source']);msg=NOTES[name][i]
            docs.append(f'\n### 原单元 {i} · {c["cell_type"]}\n\n{msg}\n\n~~~'+('python' if c['cell_type']=='code' else 'markdown')+'\n'+src+'\n~~~\n')
            reading.extend(['',f'# %% 原单元 {i} · {c["cell_type"]}','# '+msg])
            if c['cell_type']=='code':reading.append(annotate(src,msg))
            else:reading.extend(('# '+line).rstrip() for line in src.splitlines())
        write(CODE/(name+'_代码注释.py'),reading_whitespace('\n'.join(reading).rstrip()+'\n'))
    rel='tutorials/reward_model/imgs/quality-labels.png';raw=subprocess.check_output(['git','-C',str(UPSTREAM),'show',COMMIT+':'+rel]);(CODE/'imgs').mkdir(exist_ok=True);(CODE/'imgs/quality-labels.png').write_bytes(raw);manifest['files']['代码/'+rel]=hashlib.sha256(raw).hexdigest()
    doc='\n'.join(line.rstrip() for line in ''.join(docs).splitlines())+'\n';write(ROOT/'代码单元解释.md',doc)
    from transformers.models.llama import modeling_llama as hf
    source=inspect.getsource(hf.LlamaForSequenceClassification)
    core=['# Transformers4.44.2 LlamaForSequenceClassification完整类，只增中文注释。',
          '# Copyright HuggingFace Inc. Apache-2.0，见同目录LICENSE.transformers。',
          '# 这是本次随机教学RM使用的相关实现，不是OpenAI InstructGPT官方工程。',
          'from transformers.models.llama import modeling_llama as _official',
          'globals().update({name:value for name,value in vars(_official).items() if not name.startswith("__")})','']
    hints=[('self.model =','完整LlamaModel骨干；不是简单线性假模型。'),('self.score =','把每个token最后隐藏向量H映射到num_labels，本例为一个标量。'),
      ('transformer_outputs =','真实骨干前向，含Embedding/Attention/RoPE/MLP/Norm。'),('hidden_states =','取骨干最后隐状态[B,T,H]。'),
      ('logits = self.score','逐token线性打分[B,T,1]，还未选序列末位置。'),('sequence_lengths =','定位最后非pad位置；当前测试使用合法右padding。'),
      ('pooled_logits =','按batch下标和每条末位置取标量，shape[B,1]。'),('if labels is not None','此类的自带labels路径可能做回归/分类；教学pairwise不传labels，在外部计算。'),
      ('loss_fct = MSELoss','num_labels=1的默认labels路径是MSE，不是BT偏好loss，不能误用。'),
      ('return SequenceClassifierOutputWithPast','返回真实输出对象，logits为序列分数，并可带hidden/cache等。')]
    # 通过token确定多行字符串区域，避免改变官方docstring。
    protected=set()
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type==tokenize.STRING and t.start[0]!=t.end[0]:protected.update(range(t.start[0],t.end[0]+1))
    for i,line in enumerate(source.splitlines(),1):
        if i not in protected:
            for pat,msg in hints:
                if pat in line:core.append(line[:len(line)-len(line.lstrip())]+'# '+msg);break
        core.append(line)
    write(CODE/'Llama奖励打分源码_代码注释.py','\n'.join(core).rstrip()+'\n')
    dist=metadata.distribution('transformers');entry=next(x for x in dist.files if str(x).endswith('.dist-info/LICENSE'));(CODE/'LICENSE.transformers').write_bytes(Path(dist.locate_file(entry)).read_bytes())
    manifest.update(transformers='4.44.2',official_class='LlamaForSequenceClassification',official_class_sha256=hashlib.sha256(source.encode()).hexdigest(),
      paper='https://arxiv.org/pdf/2203.02155v1',paper_scope='§3.5 Eq1/Figure2/AppendixC.2，web PDF文本已读；本地下载TLS失败，未声明原论文.pdf已保存',
      subtitle='E:/8adc33be-d8dd-428f-a921-70a0b0849a21.txt，仅本地',
      remote_images='主4/6/19只构造URL展示对象，远程图内容未验证',
      official_training_repository='未找到本讲原6B RM训练checkpoint及完整训练工程；HF小模型和SciPy都是明确教学/相关实现')
    write(ROOT/'来源清单.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Built main27+supp8 cells, 12+5 code cells, four derivatives55/17 and two original reading files')

if __name__=='__main__':build()
