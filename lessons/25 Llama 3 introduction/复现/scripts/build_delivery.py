# -*- coding: utf-8 -*-
"""Build fixed original notebooks and strictly comment-only code derivatives."""
import argparse
import ast
import copy
import hashlib
import io
import json
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "7fbc29824171fd07dc2c30fe922e61ebc018ad1d"
META_COMMIT = "cf3ea595be1edfb019d21b14d8b215a39161c860"
FILES = {
    "llama3_0419": ("", """
datetime是类，now()返回当前本地datetime对象，不参与模型运算；Image稍后展示原图。历史时间2024年4月不是本次运行时间。
本格改进程http/https代理到7890；不属于模型算法。离线CPU验证明确跳过，避免把无服务端口写进环境。
两行都以#开头，不会执行pip；-U会升级依赖，正式复现使用版本记录而不是盲目运行安装注释。
导入Transformers适配库与PyTorch张量库；import本身不下载权重，不产生GPU模型。
原课列Meta权重/推理实现及HF适配的入口。region问句只是原作者记录，不是访问规则；遵守发布方实际申请条件。本讲核对2024年4月首发Llama3，而不是后续Llama3.1。
Image读取本地PNG，width400只控制显示宽度。相对路径../../imgs从代码/tutorials/llama解析到代码/imgs，不是从仓库根解析。
blobfile是读取tiktoken词表依赖；这是注释，不安装。验证在单独runtime-libs目录准备tiktoken0.7.0/blobfile2.1.1。
进入基本量级，后面的显存估计只是权重存储下界。
8.03e9个参数乘2bytes再除1024³，实际单位GiB，字符串写GB不准确；不含KV缓存、激活、加载临时副本或训练优化器。
heads是Q头数，不等于KV头数。2-7B是32Q/32KV，3-8B是32Q/8KV；head_dim=4096/32=128，不能只看此表认定架构相同。
默认CUDA类型片段只用于Meta GPU构建。CPU机器即使进入else仍会选择CUDA HalfTensor，实际失败。set_default_tensor_type改变全局默认类型且已弃用；不要把它当作CPU回退。
进入性能说明，区分模型卡中的历史测评与本地实际验证。
最强/略低是历史简述，必须落实任务和口径。固定官方卡MMLU5shot：3-8B66.6，2-7B45.7，2-70B69.7；不能据此推导所有任务排名或2026年现状。
进入模型结构：加载、展示模块树、逐参数计数。
AutoModelForCausalLM是自动选择因果语言模型实现的工厂；from_pretrained还会解析配置及加载checkpoint，和仅初始化配置不同。
原格下载2-7B并自动放置FP16权重。CPU教学明确替换为相同完整配置在meta设备构造，只验证真实模块shape/参数量，不伪称加载预训练权重。
交互环境显示模型repr；只展示模块树，不执行forward。Q/K/V/O均4096→4096，MLP宽11008，词表32000。
计算MLP相对隐藏宽度11008/4096=2.6875；SwiGLU三条线性层，不是普通两层4D ReLU。
named_parameters返回名字/Parameter迭代器；numel返回元素数而不是字节。逐个求和6738415616，meta张量也能精确计数。
原格下载3-8B FP16权重。教学同样只替换加载语句为完整配置的meta模型；词表128256/KV8/MLP14336/rope500000。
交互显示3-8B模块：Q4096→4096、K/V4096→1024、O4096→4096；GQA缩窄KV投影但保留32个Q头。
总计8030261248个参数，包括未共享的输入embedding与输出lm_head；两个独立矩阵各128256×4096。
精确参数量乘2bytes=14.957527160644531GiB权重；BF16和FP16都是2bytes，不代表推理总显存或训练显存。
进入tokenizer，字符串转整数ID的步骤发生在embedding之前。
两种都是BPE但文件格式/预切分/词表不同。列表推导逐条字符串encode，bos/eos为必须显式传入的仅关键字参数。128256已含256个特殊词，不要再加一次。
SentencePieceProcessor从原499723bytes模型读取32000词表；vocab_size是方法。ID语义只属于该词表，不能直接把2的ID喂给3。
从同目录原tokenizer.py导入Meta Tokenizer类，以原tiktoken rank文件构造实例。构造函数添加256特殊token；本次真实运行，不替换成伪分词器。
n_words是整数属性，不是函数；真实128256=128000普通mergeable ranks+256特殊token。
仅注释，不会展示字典；新增说明记录BOS128000/EOS128001/头128006与128007/EOT128009。pad_id=-1为哨兵，不可直接喂embedding。
进入输入嵌入，词ID查表得到连续向量。
完整embedding权重[V,D]=[128256,4096]。ParallelEmbedding沿D切，VocabParallelEmbedding沿V切；完整首发Meta Transformer采用后者。本次只验证单rank，不把大词表或内存描述当作无条件性能结论。
进入Meta官方demo，源代码与HF接口是两条构建路径。
launch.json允许//注释，属于JSONC。module运行torch.distributed.launch而非被注释program；1是进程数，512是运行缓存长度上限，6是最大batch容量而非必有六条输入。原CLI/NCCL/checkpoint未运行。
进入Transformers pipeline路径，自动串起分词、模型生成与解码。
text-generation为任务；model为基础模型ID；model_kwargs只传给模型加载，BF16是权重dtype；device_map auto需相应加载支持。教学用真实随机CPU小Llama与原词表转换的Fast tokenizer，并显式限制4个新增token，不比较语言质量。
进入Instruct示例，训练过的指令模型和对话格式需要配合；Base同样能生成文本但不因此获得对话对齐。
messages是list[dict]，apply_chat_template返回带角色头的str；add_generation_prompt补assistant开头。terminators含EOS和EOT，任一可结束；max_new_tokens是新增预算，temperature/top_p控制采样。输出默认含prompt，用字符len(prompt)切掉前缀；本次模型随机且预算明确改4，不冒称海盗回答复现。
打印完整序列化提示；这不是生成回答。逐消息以EOT结尾而不是EOS，最后assistant头后有两个换行，BOS仅出现一次。
"""),
    "llama3_supp_data_config": ("supp_", """
补充数据与配置，和主notebook配套；不把第29讲7B/8B逐层对比notebook混入本讲。
15T+预训练token、10M+人工标注与不含Meta用户数据均为2024官方model card声明；没有下载训练语料或独立审计。token不是样本或字数，公开可获取不等于全部无授权条件。
进入配置：同一个概念在Meta/HF有不同字段名。
dim→hidden_size4096，n_layers→num_hidden_layers32，n_heads→num_attention_heads32，n_kv_heads→num_key_value_heads8，vocab_size128256；multiple_of1024与ffn_dim_multiplier1.3共同决定中间宽14336；norm_eps1e-5，rope_theta500000。此JSON没写上下文长度，不能据此猜8k；8k来自官方卡。
进入SwiGLU中间宽度推导，4D只是构造参数的初值。
FeedForward构造接受dim=D与初始hidden_dim=4D；实际内部先乘2/3再乘1.3并向1024倍数取整。self.feed_forward注册模块；args必须先由ModelArgs构造，片段并非独立可执行程序。
hidden_dim先是4×4096=16384，尚未是权重真实宽度；赋值改变Python变量，不分配权重。
完整原FeedForward类：继承nn.Module，super初始化注册表。w1/w3是列并行[D→M]，w2行并行[M→D]；SwiGLU为w2(silu(w1(x))*w3(x))，*为逐元素门控。init_method=lambda x:x不初始化数值，本次小实验显式初始化，真实权重通常随后从checkpoint加载。
依次int(32768/3)=10922，int(10922×1.3)=14198，向1024整数倍上取整到14336。//为整除；必须按原顺序截断，不能先把公式随意约分。
14336/4096=3.5是最终MLP宽度倍率；三矩阵参数3DM=176160768每层，不是两层4D参数。
原空代码单元，保留；新增注释只说明它不执行任何模型工作。
"""),
}
FILES = {k: (p, s.strip().splitlines()) for k, (p, s) in FILES.items()}

RULES = [
    ("from ", "从模块导入指定对象；后续调用名由此绑定。"),
    ("import ", "导入依赖；不等于下载或构造模型。"),
    ("datetime.now", "返回当前本地时间，验证只检查类型，不要求与历史相同。"),
    ("os.environ", "修改当前进程网络代理环境；离线验证跳过本格。"),
    ("Image(", "返回显示对象，读取相对路径PNG，宽度仅控制展示。"),
    ("'8.03B", "参数数×每参数2bytes÷1024³；输出标签应理解为GiB。"),
    ("model_id =", "保存Hub仓库ID字符串；赋值本身不下载。"),
    ("llama2_7b =", "原逻辑加载7B真实权重，CPU验证明确改为完整meta配置。"),
    ("llama3_8b =", "原逻辑加载8B真实权重，CPU验证明确改为完整meta配置。"),
    ("total_params = 0", "计数器清零，避免将上个模型计数累加。"),
    ("for name, para", "逐个解包参数名和Parameter对象，非逐层forward。"),
    ("total_params +=", "numel只读取shape算元素数，无需物理权重数据。"),
    ("8030261248", "FP16/BF16两字节权重存储估计，不含缓存/激活。"),
    ("sp_model =", "加载真实SentencePiece文件，文件格式不同于tiktoken。"),
    ("sp_model.vocab_size", "调用方法得到词条数32000。"),
    ("llama3_tokenizer =", "读取原tiktoken词表并附加256特殊ID。"),
    ("llama3_tokenizer.n_words", "读取属性128256，已包括特殊token。"),
    ("pipeline =", "构造生成流水线，返回可调用TextGenerationPipeline对象。"),
    ('"text-generation"', "任务名决定分词→generate→decode这条路径。"),
    ("model=", "原为Hub模型ID；教学变体显式传真实CPU随机小模型。"),
    ("model_kwargs=", "该字典在原调用用于模型加载，非采样参数。"),
    ("device_map=", "让加载器安排设备，auto不保证一定使用GPU。"),
    ("device=", "原指定cuda；CPU教学替换构造，不谎称GPU运行。"),
    ("pipeline(", "调用流水线得到含generated_text的列表，默认包括输入前缀。"),
    ("messages =", "按顺序保存角色与文本；尚未变成ID。"),
    ('{"role":', "role控制模板角色头，content为消息正文。"),
    ("prompt =", "套用Jinja对话模板返回str，再由流水线分词。"),
    ("tokenize=False", "返回文本而非ID列表，便于检查特殊标记与头部。"),
    ("add_generation_prompt=True", "末尾追加assistant开头，让模型接着生成正文。"),
    ("terminators =", "列表里两种结束ID是或关系，不要求连续生成两者。"),
    ("pipeline.tokenizer.eos", "读取结束全文ID128001。"),
    ("pipeline.tokenizer.convert", "把EOT标记映射成128009，结束一个对话轮次。"),
    ("outputs =", "生成结果列表；随机教学变体只生成至多4个新token。"),
    ("max_new_tokens=", "原预算256个新增token，不是输入加输出总长。"),
    ("eos_token_id=", "覆盖结束条件为EOS/EOT两个ID。"),
    ("do_sample=", "True按分布采样，False才是此示例常见greedy路径。"),
    ("temperature=", "正温度缩放logits，0.6使分布相对更集中。"),
    ("top_p=", "保留累计概率达到阈值的候选后再归一化采样。"),
    ("print(outputs", "默认输出含prompt，用字符长度切片仅保留续写字符串。"),
    ("print(prompt", "显示格式化输入，不是模型新生成文本。"),
    ("hidden_dim = 4", "初始4D=16384，仅Python标量。"),
    ("hidden_dim = int(2", "先乘2/3并向零截断到10922。"),
    ("hidden_dim = int(1.3", "再乘1.3并截断到14198。"),
    ("hidden_dim = 1024", "以ceil整除形式向上对齐1024，得到14336。"),
    ("class Tokenizer", "官方分词器类，负责字符串与ID之间的可逆映射。"),
    ("class ChatFormat", "官方对话模板对象，复用Tokenizer组装角色头和EOT。"),
    ("class Message", "TypedDict只作类型提示，不自动进行运行时角色合法性检查。"),
    ("class ModelArgs", "dataclass声明配置字段，实际checkpoint JSON覆盖默认值。"),
    ("dim:", "模型隐藏宽度D，输入嵌入/残差都保留此维度。"),
    ("n_layers:", "TransformerBlock数量L，不包含最终norm和输出头。"),
    ("n_heads:", "查询Q头数H，单头宽度为dim//n_heads。"),
    ("n_kv_heads:", "KV头数G，None时Attention采用与Q相同的头数。"),
    ("vocab_size:", "词表总行数，默认-1需被实际词表大小覆盖。"),
    ("multiple_of:", "SwiGLU中间宽上取整倍数；8B JSON实际覆盖为1024。"),
    ("ffn_dim_multiplier:", "SwiGLU宽度额外倍率，8B实际为1.3，None则不乘。"),
    ("norm_eps:", "RMSNorm分母稳定项1e-5，不是可训练参数。"),
    ("rope_theta:", "位置旋转频率底数500000，不是上下文长度。"),
    ("max_batch_size:", "运行缓存batch容量，默认32，可被CLI覆盖。"),
    ("max_seq_len:", "运行缓存序列容量默认2048，不是8k训练上下文声明。"),
    ("if torch.cuda.is_bf16_supported", "检查CUDA BF16能力，不是检查能否退回CPU。"),
    ("torch.set_default_tensor_type", "原代码设置全局CUDA默认类型；else也依赖CUDA，CPU会失败。"),
    ("class FeedForward", "完整官方SwiGLU层，仅加注释，依赖真实FairScale初始化。"),
    ("def __init__", "实例初始化；self是该对象，不是输入张量。"),
    ("super().__init__", "初始化nn.Module注册机制，随后赋值的子层才可被遍历。"),
    ("hidden_dim = int(2 *", "先将4D缩到约8D/3并截断。"),
    ("if ffn_dim_multiplier", "可选倍率不为None时才应用。"),
    ("hidden_dim = int(ffn", "乘配置倍率1.3后再次截断。"),
    ("hidden_dim = multiple", "将中间宽上取整为multiple_of的整数倍。"),
    ("self.w1 =", "门控分支权重[M,D]；world1输出[B,T,M]。"),
    ("self.w2 =", "输出投影权重[D,M]；world1恢复[B,T,D]。"),
    ("self.w3 =", "内容分支权重[M,D]，独立于w1。"),
    ("def forward", "nn.Module.__call__会调用forward；x为浮点[B,T,D]。"),
    ("return self.w2", "silu门控逐元素乘内容，最后投影回D，无bias。"),
    ("num_reserved_special_tokens =", "256是特殊token总数，不是普通词表大小。"),
    ("pat_str =", "正则先切分文本，再对各段做字节BPE；保留原表达式。"),
    ("mergeable_ranks =", "从原文件加载bytes→rank的128000项字典。"),
    ("num_base_tokens =", "特殊ID从普通词表末端128000起排。"),
    ("special_tokens =", "列表顺序确定特殊ID，不能任意排序或少加reserved项。"),
    ("self.special_tokens =", "建立特殊字符串到唯一整数ID的映射。"),
    ("self.model =", "真正的tiktoken.Encoding，底层实现不是本脚本伪造。"),
    ("self.n_words:", "总词表128256，普通与特殊ID合计。"),
    ("self.bos_id:", "文首标记128000，和换行、角色标记不同。"),
    ("self.eos_id:", "全文终止标记128001。"),
    ("self.pad_id:", "-1是padding哨兵，正式forward前须被有效ID替换。"),
    ("self.stop_tokens =", "全文EOS和轮次EOT均可终止生成。"),
    ("def encode(", "bos/eos为无默认的仅关键字参数，返回list[int]。"),
    ("assert type(s)", "本版本要求精确str类型；类型注解不代替断言。"),
    ("TIKTOKEN_MAX_ENCODE_CHARS =", "分批处理超长字符串，避免旧tiktoken大输入异常。"),
    ("MAX_NO_WHITESPACES_CHARS =", "限制连续空白/非空白段长度，不等于模型上下文长度。"),
    ("substrs =", "嵌套生成器按400k字符和连续段限制延迟产出字符串。"),
    ("t: List", "收集输出ID；可变list仅属于本次调用。"),
    ("t.extend(", "追加各段编码，或追加消息部分，不创建batch张量。"),
    ("if bos:", "可选在最前插入一次BOS。"),
    ("t.insert", "列表首插128000，不是文本字符串拼接。"),
    ("if eos:", "可选追加全文EOS。"),
    ("t.append", "追加结束标记ID。"),
    ("return t", "返回普通Python列表，后续才转int64张量。"),
    ("def decode(", "接受ID序列，返回Unicode字符串。"),
    ("return self.model.decode", "cast只提供类型提示；实际交给tiktoken解码。"),
    ("@staticmethod", "静态辅助函数无self，不读取实例状态。"),
    ("def _split_", "yield把函数变为字符串片段生成器。"),
    ("current_slice_len =", "记录当前连续空白或非空白段字符数。"),
    ("current_slice_is_space =", "空串单独处理，避免访问s[0]越界。"),
    ("slice_start =", "保存下个输出片段起点索引。"),
    ("for i in range(len(s))", "按字符索引扫描，不按字节或token索引。"),
    ("is_now_space =", "Unicode isspace判定当前字符是否空白。"),
    ("if current_slice_is_space ^", "^为布尔异或，空白属性变化时重置段计数。"),
    ("yield ", "惰性返回片段，调用方继续迭代后再从此处恢复。"),
    ("if current_slice_len >", "只在连续同类字符超限后切分。"),
    ("self.tokenizer =", "保存传入分词器引用，多个消息共享同一ID映射。"),
    ("def encode_header", "角色头：start_header、角色ID、end_header、两个换行。"),
    ("tokens = []", "从空list开始组装当前头部或对话。"),
    ("tokens.append", "按协议顺序追加特殊ID。"),
    ("tokens.extend", "把普通文本编码或已有列表展开追加。"),
    ("return tokens", "返回组装完成的一维整数列表。"),
    ("def encode_message", "头部+strip后的正文+EOT；不自动插入BOS。"),
    ("tokens = self.encode_header", "复用角色头方法，减少格式分歧。"),
    ("def encode_dialog_prompt", "整段对话仅首部BOS，每条消息EOT，末尾assistant头。"),
    ("for message in dialog", "保持system/user/assistant输入顺序，不自动校验交替。"),
]

def annotate(src, note=""):
    protected = set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.STRING and t.end[0] > t.start[0]:
            protected.update(range(t.start[0]+1, t.end[0]+1))
    out = ["# "+note+"\n"] if note else []
    for i, line in enumerate(src.splitlines(keepends=True), 1):
        if i not in protected and line.strip() and not line.lstrip().startswith("#"):
            for start, text in RULES:
                if line.strip().startswith(start):
                    out.append(line[:len(line)-len(line.lstrip())]+"# "+text+"\n")
                    break
        out.append(line)
    return "".join(out)

def reading_format(src):
    protected = set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.STRING and t.end[0] > t.start[0]:
            protected.update(range(t.start[0], t.end[0]+1))
    return "".join(l if i in protected else l.rstrip()+"\n"
                   for i,l in enumerate(src.splitlines(keepends=True),1)).rstrip()+"\n"

def annotate_jsonc(src):
    fields={
        "configurations":"调试配置数组，可包含多种启动方式。",
        "name":"调试器显示名，不决定实际运行哪个程序。",
        "type":"使用debugpy的Python调试支持。",
        "request":"launch启动进程，不是attach到已有进程。",
        "module":"运行Python模块torch.distributed.launch；不是被注释的program。",
        "console":"选择集成终端，和模型运算无关。",
        "justMyCode":"false允许跟进库源码。",
        "args":"顺序传给启动器的字符串参数数组。",
        "--nproc_per_node":"每节点一个worker进程，不等于模型batch。",
        "example_text_completion.py":"目标脚本，需在Meta官方仓库工作目录运行。",
        "--ckpt_dir":"Meta分片权重目录，不是HF模型ID。",
        "--tokenizer_path":"原tiktoken模型文件路径。",
        "--max_seq_len":"本次缓存长度上限512，不是训练上下文声明。",
        "--max_batch_size":"最大batch容量6，不保证实际有六个prompt。",
        "dim":"隐藏宽4096，HF对应hidden_size。",
        "n_layers":"32个TransformerBlock。",
        "n_heads":"32个Q头。",
        "n_kv_heads":"8个KV头，每组供4个Q头使用。",
        "vocab_size":"128256，已经含256特殊token。",
        "multiple_of":"中间宽向1024倍数上取整。",
        "ffn_dim_multiplier":"原宽度缩到2/3后再乘1.3。",
        "norm_eps":"RMSNorm稳定项1e-5。",
        "rope_theta":"RoPE频率底数500000。",
    }
    out=[]
    for line in src.splitlines(keepends=True):
        m=re.match(r'\s*"([^"]+)"',line)
        if m and m[1] in fields:
            out.append(line[:len(line)-len(line.lstrip())]+"// "+fields[m[1]]+"\n")
        out.append(line)
    return "".join(out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--meta-sources", required=True)
    a=p.parse_args()
    target=ROOT/"代码/tutorials/llama"
    target.mkdir(parents=True, exist_ok=True)
    manifest={"repository":"https://github.com/chunhuizhang/personal_chatgpt",
              "commit":COMMIT, "meta_repository":"https://github.com/meta-llama/llama3",
              "meta_commit":META_COMMIT, "files":[], "official_sources_sha256":{},
              "subtitle":"E盘根目录字幕中未找到本讲独立文本；7B/8B对比字幕属于第29讲。"}
    def publish(rel):
        data=subprocess.check_output(["git","-C",a.upstream,"show",COMMIT+":"+rel])
        dest=ROOT/"代码"/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        manifest["files"].append(dict(upstream_path=rel,published_path="代码/"+rel,
                                     bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        return data
    for rel in ["imgs/meta-llama-model.png","tutorials/llama/llama2_tokenizer.model",
                "tutorials/llama/llama3_tokenizer.model","tutorials/llama/tokenizer.py"]:
        publish(rel)
    meta=Path(a.meta_sources)
    for rel in ["llama/model.py","llama/tokenizer.py","llama/generation.py","MODEL_CARD.md","LICENSE","README.md"]:
        manifest["official_sources_sha256"][rel]=hashlib.sha256((meta/rel).read_bytes()).hexdigest()
    assert (target/"tokenizer.py").read_bytes()==(meta/"llama/tokenizer.py").read_bytes()
    (target/"LICENSE").write_bytes((meta/"LICENSE").read_bytes())
    (target/"Notice").write_bytes("Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\n".encode())
    (target/"tokenizer_代码注释.py").write_bytes(reading_format(annotate((target/"tokenizer.py").read_text(encoding="utf-8"))).encode())
    md=["# 第25讲全部单元说明\n\n主38格、补充11格，共49格；原编号各自从0开始。所有代码、Markdown、仅注释及空格均覆盖。\n"]
    for stem,(prefix,notes) in FILES.items():
        raw=publish("tutorials/llama/"+stem+".ipynb")
        nb=json.loads(raw)
        assert len(nb["cells"])==len(notes),(stem,len(notes))
        intro={"cell_type":"markdown","metadata":{},"source":["# 第25讲："+stem+"\n\nBuilt with Meta Llama 3\n\n新增中文说明与代码注释；上游单元、元数据及历史输出保留。历史输出不是本次大模型运行结果。\n"]}
        explained=copy.deepcopy(nb); annotated=copy.deepcopy(nb)
        explained["cells"]=[copy.deepcopy(intro)]
        annotated["cells"]=[copy.deepcopy(intro)]
        reading="# -*- coding: utf-8 -*-\n# 原课程代码阅读版；原GPU/大模型加载保留，CPU变体另见复现脚本。\n"
        md.append("## "+stem+"\n")
        for i,(c,note) in enumerate(zip(nb["cells"],notes)):
            note=("【必须会】" if c["cell_type"]=="code" else "【建议会】")+note
            text="### 原单元 "+str(i)+"（"+c["cell_type"]+"）\n\n"+note+"\n"
            added={"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
            explained["cells"] += [copy.deepcopy(added),copy.deepcopy(c)]
            cell=copy.deepcopy(c); src="".join(c["source"])
            if c["cell_type"]=="code":
                code=annotate(src,note)
                cell["source"]=code.splitlines(keepends=True)
                reading+="\n# %% 原代码单元 "+str(i)+"\n"+code+"\n"
            elif ((stem=="llama3_supp_data_config" and i in (3,5,7)) or
                  (stem=="llama3_0419" and i in (10,32))):
                blocks=re.findall(chr(96)*3+r"[^\n]*\n(.*?)"+chr(96)*3,src,re.S)
                jsonc=(stem=="llama3_supp_data_config" and i==3) or (stem=="llama3_0419" and i==32)
                extra="\n\n新增源码注释（保留上方原片段）：\n\n"+"\n\n".join(
                    chr(96)*3+("jsonc\n"+annotate_jsonc(b) if jsonc else "python\n"+annotate(b,note))+chr(96)*3 for b in blocks)
                cell["source"]=(src+extra).splitlines(keepends=True)
                text+=extra+"\n"
            annotated["cells"] += [copy.deepcopy(added),cell]
            md.append(text)
        for name,obj in [("原代码_带说明.ipynb",explained),("代码注释.ipynb",annotated)]:
            (target/(prefix+name)).write_bytes((json.dumps(obj,ensure_ascii=False,indent=1)+"\n").encode())
        ast.parse(reading)
        (target/(stem+"_代码注释.py")).write_bytes(reading_format(reading).encode())
    src=(meta/"llama/model.py").read_text(encoding="utf-8")
    core="".join(src.splitlines(keepends=True)[:2])+"\n# 完整官方配置/FeedForward类摘录，仅注释；依赖原模块的imports和FairScale初始化。\n"
    for name in ("ModelArgs","FeedForward"):
        node=next(n for n in ast.parse(src).body if getattr(n,"name","")==name)
        start=min([node.lineno]+[d.lineno for d in node.decorator_list])
        core+="\n"+annotate("".join(src.splitlines(keepends=True)[start-1:node.end_lineno]))+"\n"
    (target/"Llama3核心源码_代码注释.py").write_bytes(reading_format(core).encode())
    # Markdown rendering copy may trim line-end spaces; original notebook text stays untouched.
    rendered="\n".join(line.rstrip() for line in "\n".join(md).splitlines()).rstrip()+"\n"
    (ROOT/"代码单元解释.md").write_bytes(rendered.encode())
    (ROOT/"来源清单.json").write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode())
    print("Built: 2 originals, 4 derived notebooks, 4 Python readings, 49 cells and original assets.")

if __name__=="__main__":
    main()
