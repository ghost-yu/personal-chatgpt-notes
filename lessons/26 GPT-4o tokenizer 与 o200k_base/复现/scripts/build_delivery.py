# -*- coding: utf-8 -*-
"""Preserve originals and generate comment-only readings, with pinned real vocab assets."""
import argparse
import ast
import base64
import copy
import csv
import hashlib
import io
import json
import subprocess
import tokenize
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
COMMIT="7fbc29824171fd07dc2c30fe922e61ebc018ad1d"
VOCABS={
    "cl100k_base":("9b5ad71b2ce5302211f9c61530b329a4922fc6a4","223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    "o200k_base":("fb374d419588a4632f3f557e76b4b70aebbca790","446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"),
}
FILES={
    "tokenizer_gpt4o":("", """
原课微信讨论链接保留，当前访问失败，未核实文章正文；不把二手议论当成官方训练细节。
比较两个编码名字，不代表把已有模型的词表直接热替换。模型权重与ID语义必须匹配。
tiktoken提供真实字节BPE，langdetect是统计语言识别，regex支持Unicode属性和更丰富正则。import不加载GPT-4o权重，不调用API。
get_encoding按编码名读取注册构造器，首次可能下载词表并校验SHA，返回Encoding对象，后续缓存复用。本次预置真实完整词表缓存，原调用不改且禁止联网验证。
cl100k把此英文句子转5个ID。ID是词表行号，不是Unicode码点；没有embedding/模型前向。
o200k同句仍是5个ID但数值不同，直接反例说明词表更大不保证每句token严格更少。
n_vocab=max_token_value+1，分别100277与200019，不等于所有中间ID都有效。cl普通100256+特殊5共100261，o普通199998+特殊2共200000，其余是洞位。
原课讨论压缩token_ids，是相同原文使用更少token，不是把每个整数ID本身压成更少bits。
原文更大词表必更少和价格更低需加条件：两词表/预切分不同，非集合嵌套；总账单还涉及模型单价、输入/输出/缓存等。这里只测文本长度，不引用当前价格。
cl对指定中文完整句编码后len=34，len数列表元素，不是字数、字节数或张量长度。
o对同一句len=24，减少10个即29.41%，旧/新=1.4167；不同文本不保证同比例。
with确保文件关闭，encoding指定UTF8；readlines一次读入全部文本再逐行编码累加，不是流式训练。相对路径./xyj.txt配套原文件完整发布，保留原字节/原换行；Python文本模式会规范化换行。
两个整数及比率：cl990508/o703303，旧/新1.4083659532，新/旧0.7100427256，减少约29%。这是逐行全文统计，不是任意中文或整篇一次编码的定理。
进入中文词表观察：看token对应文本，不能直接测embedding训练充分程度或GPT回答能力。
原文关于语料过滤导致异常回答的因果解释是推测，notebook未提供GPT-4o训练日志、语料、权重或受控实验。分享链接历史回答不能由本次分词实验重现；gist是第三方观察脚本，不是官方训练说明。
原词组在o200k编码为单个177431。它属于普通mergeable token，不是special_tokens里的保留控制符；原课特殊中文tokens是口语描述。
同词组在cl100k拆成12个ID；比较必须连同完整解码检查，单token可能只是UTF8字符的一部分。
扫描o200k所有range(n_vocab)位置，包括19个无效洞位；decode默认替换坏UTF8，len数Python字符。bare except吞掉Rust PanicException甚至中断；按长度降序稳定排序后langdetect筛zh-cn，实际只打印20条，不是注释写的100。语言检测对短串不可靠且有随机性，本次外置seed26；保留原算法/历史panic输出。
同流程扫描cl100k，16个无效洞位。标点/换行也可能被langdetect判zh-cn，不能把输出理解成真实中文词频榜或训练语料分布。实际执行完整100277位置。
再次扫描完整o200k，以正则剔除全空白/非词字符/特定符号串；不是中文检测。英文、数字、代码片段仍可通过。tot==20退出，只取20条；bare except问题不改原版，在独立诊断提供安全写法。
原空代码格，保留；只新增说明注释，不补造训练/调用结果。
"""),
    "tokenizer_supply":("supp_", """
配套接口补充，解释数字与空格边界；不是额外一讲，也没有tokenizer训练。
同一cl100k编码对象，本次使用完整真实缓存；get_encoding重复调用会复用注册表对象。
数字分块并不等于十进制位独立编码，词表ID也无数学上的数值连续关系。
1234+4567按正则最多三位数字一段，再BPE，得到[4513,19,10,10961,22]。加号本身也是token；不是执行算术。
遍历原5个ID并逐个decode/print，展示123、4、+、456、7。for逐次赋给t_id，打印五行而不是一个总和。
前导空格可参与token文本，所以world与空格world不一定同ID。
无空格world编码14957，不能直接推出所有单词同样行为。
带前导空格world编码1917，空格属于此token的字节内容，不是先统一strip。
hello world编码[15339,1917]，第二个token已经含空格，不需再插一个空格ID。
重复编码world仍14957；确定性来自固定词表/正则/输入，不依赖模型采样。
解码[1917,14957]得到空格worldworld。decode按token字节拼接，不按词义自动加空格。
原空代码格，保留；本讲没有fine-tune模型。
"""),
}
FILES={k:(p,s.strip().splitlines())for k,(p,s)in FILES.items()}
RULES=[
    ("import tiktoken","字节BPE库；后续get_encoding加载词表，不加载语言模型。"),
    ("import langdetect","短文本语言分类工具，固定seed仍不保证判断正确。"),
    ("import regex","Unicode正则库，不是Python内置re。"),
    ("T1 =","绑定cl100k_base Encoding实例，后续按此词表解释ID。"),
    ("T2 =","绑定o200k_base实例；两套ID不可跨模型随意混用。"),
    ("T =","补充实验仍用cl100k，缓存命中返回同对象。"),
    ("T1.encode","确定性字符串→list[int]，不是模型forward。"),
    ("T2.encode","使用新词表分词，len不保证对所有文本都更短。"),
    ("T1.n_vocab","读取max ID+1，含无效洞位，不能当有效ID数量。"),
    ("len(T1","数cl编码的token个数，不数字符或UTF8字节。"),
    ("len(T2","数o编码个数，同一句34→24。"),
    ("token_ids_len1 =","旧编码计数清零。"),
    ("token_ids_len2 =","新编码计数清零。"),
    ("with open","上下文管理器负责关闭文件；原readlines一次加载全部行。"),
    ("for line in","按Python文本换行规范化后的行编码，不对正文strip。"),
    ("token_ids_len1 +=","累加这一行的cl token数，保留行末换行。"),
    ("token_ids_len2 +=","累加同一行的o token数，确保比较输入一致。"),
    ("token_ids_len1,","展示两个计数与旧/新、新/旧比率，降低比例为1−新/旧。"),
    ("length_dict = {}","建立ID→单token解码后Python字符数的字典。"),
    ("for i in range","上界max ID+1包含洞位；原代码会触发Rust panic后被bare except吞掉。"),
    ("length_dict[i] =","decode默认errors=replace，部分UTF8字节可能显示替换字符。"),
    ("except:","裸捕获还会吞KeyboardInterrupt等；保留原码，安全版本独立提供。"),
    ("pass","忽略当前错误，不会修复洞位或记录原因。"),
    ("length_dict = dict","sorted按负长度升序即按长度降序，同长度保留原ID顺序。"),
    ("tot =","计已打印项目，不是已扫描项目数。"),
    ("for item in length_dict","迭代排序字典的key，即token ID，不是(id,length)元组。"),
    ("if langdetect.detect","返回zh-cn只是启发式标签，短串和标点均可能误判。"),
    ("print(item","打印原ID与单token解码文本，文本可含换行。"),
    ("tot +=","仅命中并打印后加1，跳过的条目不计。"),
    ("if tot ==","原程序实际只取20条，与top100注释不一致。"),
    ("break","退出当前for，不退出整个notebook。"),
    ("pattern =","锚定整串；剔除只由符号/空白等组成的文本，并非筛中文。"),
    ("if not re.match","保留未被全符号模式匹配的字符串，英文/数字也可通过。"),
    ("T.encode","补充：确定性编码，空格和数字分块影响ID。"),
    ("for t_id","遍历原显式ID列表，对每个ID单独解码。"),
    ("print(T.decode","显示token字节恢复的文本；本例数字/加号均完整UTF8。"),
    ("T.decode","按token字节拼接解码，不按词自动补空格。"),
    ("def get_encoding","注册表按名字查找并缓存Encoding，首次构造才读资源。"),
    ("if encoding_name in ENCODINGS","缓存命中直接返回对象，不重新下载。"),
    ("with _lock","可重入锁保护并发注册/构造。"),
    ("constructor =","取编码名对应的构造函数，不是模型类。"),
    ("enc = Encoding","把构造器的dict作为关键字参数展开到Encoding。"),
    ("ENCODINGS[encoding_name]","缓存本次构造的实例。"),
    ("return enc","返回Encoding对象，不是编码后的ID。"),
    ("def cl100k_base","定义旧词表构造器：真实rank、正则与5个特殊ID。"),
    ("def o200k_base","定义新词表构造器：199998普通rank与2个特殊ID。"),
    ("mergeable_ranks =","读取真实bytes→rank字典，下载数据需符合expected_hash。"),
    ("special_tokens =","这里是真正控制符ID映射，不是异常中文普通token。"),
    ("pat_str =","预切分正则，和词表共同决定最终token数。"),
    ("class Encoding","Encoding持有正则/词表与Rust CoreBPE，不持有神经网络权重。"),
    ("def __init__","构造对象；星号后的参数必须以关键字提供。"),
    ("self.max_token_value =","在普通/特殊token的ID里取最大值，不计算有效项数。"),
    ("if explicit_n_vocab","只有显式指定时才检查词表项数与max ID的一致性。"),
    ("self._core_bpe =","真正的Rust CoreBPE实现，Python这里只封装参数与错误处理。"),
    ("def encode(","实例方法返回list[int]；默认禁止直接编码特殊控制符文本。"),
    ("if allowed_special","all先展开为该Encoding真正的特殊token集合。"),
    ("if disallowed_special","默认all排除已允许集合，其余匹配时抛ValueError。"),
    ("if match :=","海象运算符保存正则匹配对象并判断是否存在。"),
    ("return self._core_bpe.encode","交给真实Rust BPE；不是argmax或模型采样。"),
    ("except UnicodeEncodeError","此处只处理Unicode输入错误，不是捕获无效ID的Rust panic。"),
    ("def decode(","把ID列表先解为bytes再UTF8解码，默认替换无效片段。"),
    ("return self._core_bpe.decode_bytes","先拼全部token bytes再decode，不能逐token损失后再拼。"),
    ("def decode_single_token_bytes","安全取得单ID原始字节；无效ID抛KeyError。"),
    ("return self._core_bpe.decode_single_token_bytes","不做UTF8转换，因此不会因半个字符而自动出现替换字符。"),
    ("@property","以属性语法访问n_vocab，不能写成n_vocab()。"),
    ("def n_vocab","返回max_token_value+1，洞位也占数值区间。"),
    ("return self.max_token_value","cl100277/o200019，并非有效词条总数。"),
    ("def bpe_encode","官方教育版纯Python BPE，展示同一rank优先合并过程。"),
    ("parts = [bytes","初始每个UTF8字节单独一段，不是每个Unicode字符一段。"),
    ("while True:","每轮选一个可合并相邻对，直到不存在候选。"),
    ("for i, pair","按位置遍历相邻片段，不能跨预切分边界。"),
    ("rank = mergeable_ranks.get","查拼接字节的rank；无词表项返回None。"),
    ("if rank is not None","选最小rank；编码时不是重新统计当前文本频率。"),
    ("if min_rank is None","没有合法合并时结束，不要求最终只剩一个token。"),
    ("parts = parts[:min_idx]","只合并选中的邻接片段，其余顺序保留。"),
    ("tokens = [mergeable","最终每片对应rank/ID，返回普通列表。"),
]

def sha(b):return hashlib.sha256(b).hexdigest()

def annotate(src,note=""):
    protected=set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type==tokenize.STRING and t.end[0]>t.start[0]:
            protected.update(range(t.start[0]+1,t.end[0]+1))
    out=["# "+note+"\n"] if note else []
    for i,l in enumerate(src.splitlines(keepends=True),1):
        if i not in protected and l.strip() and not l.lstrip().startswith("#"):
            for start,meaning in RULES:
                if l.strip().startswith(start):
                    out.append(l[:len(l)-len(l.lstrip())]+"# "+meaning+"\n");break
        out.append(l)
    return "".join(out)

def clean_reading(src):
    protected=set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type==tokenize.STRING and t.end[0]>t.start[0]:
            protected.update(range(t.start[0],t.end[0]+1))
    return "".join(l if i in protected else l.rstrip()+"\n" for i,l in enumerate(src.splitlines(keepends=True),1)).rstrip()+"\n"

def segment(src,node):
    start=min([node.lineno]+[d.lineno for d in getattr(node,"decorator_list",[])])
    return "".join(src.splitlines(keepends=True)[start-1:node.end_lineno])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream",required=True)
    p.add_argument("--tiktoken-root",required=True,help="parent of installed tiktoken and dist-info")
    p.add_argument("--local",required=True)
    args=p.parse_args();local=Path(args.local);lib=Path(args.tiktoken_root)
    target=ROOT/"代码/tutorials/tokenizer";target.mkdir(parents=True,exist_ok=True)
    manifest={"repository":"https://github.com/chunhuizhang/personal_chatgpt","commit":COMMIT,"files":[],
        "tiktoken_release":"0.7.0","tiktoken_tag_url":"https://github.com/openai/tiktoken/tree/0.7.0",
        "subtitle":"E盘根目录字幕中未找到本讲独立文本；232e8b13字幕属于第09讲，未冒用。",
        "vocabularies":{},"installed_sources_sha256":{}}
    def publish(rel):
        raw=subprocess.check_output(["git","-C",args.upstream,"show",COMMIT+":"+rel])
        dest=ROOT/"代码"/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        manifest["files"].append(dict(upstream_path=rel,published_path="代码/"+rel,bytes=len(raw),sha256=sha(raw)))
        return raw
    publish("tutorials/tokenizer/xyj.txt")
    for name,(key,h) in VOCABS.items():
        raw=(local/"tiktoken-cache"/key).read_bytes();assert sha(raw)==h
        dest=target/"词表"/(name+".tiktoken");dest.parent.mkdir(exist_ok=True);dest.write_bytes(raw)
        manifest["vocabularies"][name]=dict(published_path="代码/tutorials/tokenizer/词表/"+name+".tiktoken",
            sha256=h,bytes=len(raw),official_url="https://openaipublic.blob.core.windows.net/encodings/"+name+".tiktoken",
            transfer_url="https://raw.githubusercontent.com/zurawiki/tiktoken-rs/main/tiktoken-rs/assets/"+name+".tiktoken",
            transfer_note="Official endpoint TLS handshake failed; mirror bytes exactly match official expected_hash in tiktoken0.7.0")
    record=lib/"tiktoken-0.7.0.dist-info/RECORD"
    entries={r[0]:r[1]for r in csv.reader(io.StringIO(record.read_text(encoding="utf-8")))}
    source_dir=local/"tiktoken-source";source_dir.mkdir(exist_ok=True)
    for rel in ["tiktoken/core.py","tiktoken/registry.py","tiktoken/load.py","tiktoken/_educational.py","tiktoken_ext/openai_public.py","tiktoken-0.7.0.dist-info/LICENSE"]:
        data=(lib/rel).read_bytes()
        expected="sha256="+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        assert entries[rel]==expected,rel
        dest=source_dir/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        manifest["installed_sources_sha256"][rel]=sha(data)
    installed_public=(lib/"tiktoken_ext/openai_public.py").read_text(encoding="utf-8")
    assert installed_public==(local/"openai_public_remote.py").read_text(encoding="utf-8")
    manifest["remote_release_constructor_text_matches_installed"]=True
    (target/"LICENSE-tiktoken").write_bytes((lib/"tiktoken-0.7.0.dist-info/LICENSE").read_bytes())
    md=["# 第26讲全部单元说明\n\n主21格，配套接口补充12格，共33格/25代码。保留原代码/历史输出；空格、原链接和未经证实的解释均标明。\n"]
    for stem,(prefix,notes)in FILES.items():
        raw=publish("tutorials/tokenizer/"+stem+".ipynb");original=json.loads(raw)
        assert len(original["cells"])==len(notes),(stem,len(notes))
        intro={"cell_type":"markdown","metadata":{},"source":["# 第26讲："+stem+"\n\n新增中文说明与注释。原单元/历史输出保持；langdetect外置seed和实际全词表执行见复现记录。历史输出不是本次API实验。\n"]}
        explained=copy.deepcopy(original);annotated=copy.deepcopy(original)
        explained["cells"]=[copy.deepcopy(intro)];annotated["cells"]=[copy.deepcopy(intro)]
        reading="# -*- coding: utf-8 -*-\n# 原课程Python注释阅读版；完整词表扫描会捕获并打印Rust洞位panic，请先读复现说明。\n"
        md.append("## "+stem+"\n")
        for i,(c,note)in enumerate(zip(original["cells"],notes)):
            text="### 原单元 "+str(i)+"（"+c["cell_type"]+"）\n\n【"+("必须会" if c["cell_type"]=="code" else "建议会")+"】"+note+"\n"
            added={"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
            explained["cells"] += [copy.deepcopy(added),copy.deepcopy(c)]
            cell=copy.deepcopy(c)
            if c["cell_type"]=="code":
                src=annotate("".join(c["source"]),note);cell["source"]=src.splitlines(keepends=True)
                reading+="\n# %% 原代码单元 "+str(i)+"\n"+src+"\n"
            annotated["cells"] += [copy.deepcopy(added),cell];md.append(text)
        for name,obj in [("原代码_带说明.ipynb",explained),("代码注释.ipynb",annotated)]:
            (target/(prefix+name)).write_bytes((json.dumps(obj,ensure_ascii=False,indent=1)+"\n").encode())
        ast.parse(reading);(target/(stem+"_代码注释.py")).write_bytes(clean_reading(reading).encode())
    core="# -*- coding: utf-8 -*-\n# OpenAI tiktoken0.7.0完整核心对象摘录，仅添加中文注释。MIT，见同目录LICENSE-tiktoken。\n# 源码阅读文件，依赖原模块的imports/全局对象，不作为独立可执行模块。\n\n"
    selections={
        "tiktoken/registry.py":{"":["get_encoding"]},
        "tiktoken_ext/openai_public.py":{"":["cl100k_base","o200k_base"]},
        "tiktoken/core.py":{"Encoding":["__init__","encode","decode","decode_single_token_bytes","n_vocab"]},
        "tiktoken/_educational.py":{"":["bpe_encode"]},
    }
    for rel,groups in selections.items():
        src=(lib/rel).read_text(encoding="utf-8");tree=ast.parse(src)
        for owner,names in groups.items():
            nodes=next(n for n in tree.body if getattr(n,"name","")==owner).body if owner else tree.body
            if owner:core+="class "+owner+":\n"
            for name in names:
                node=next(n for n in nodes if getattr(n,"name","")==name)
                core+=("    " if owner else "")+"# "+rel+"：完整"+name+"\n"+annotate(segment(src,node))+"\n\n"
    ast.parse(core);(target/"tiktoken核心源码_代码注释.py").write_bytes(clean_reading(core).encode())
    (ROOT/"代码单元解释.md").write_bytes(("\n".join(md).rstrip()+"\n").encode())
    (ROOT/"来源清单.json").write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode())
    print("Built: 2 originals, 4 derivatives, 3 Python readings, 33 cells, full corpus and verified vocabularies.")

if __name__=="__main__":main()
