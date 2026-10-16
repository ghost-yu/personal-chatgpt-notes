# -*- coding: utf-8 -*-
"""独立补充：安全词表扫描、UTF8/特殊ID/正则边界和真实BPE验证。

这里不是原notebook的替换；原扫描仍由verify_delivery完整执行。
依赖tiktoken0.7.0及真实完整cl/o词表；不调用语言模型或付费API。
"""
import heapq
import regex
import tiktoken
from tiktoken._educational import bpe_encode

def safe_vocabulary_scan(enc):
    """输入Encoding；返回统计和最长20个含Han字符的普通token，不做语言推断。"""
    holes=[];invalid_utf8=0;valid=0
    # 0.7.0私有字段：显式锁版本；特殊控制ID不混入普通词表排名。
    special_ids=set(enc._special_tokens.values())
    top=[]
    for token_id in range(enc.n_vocab):
        try:
            raw=enc.decode_single_token_bytes(token_id)
        except KeyError:
            # 精确捕获不存在ID；不吞KeyboardInterrupt/任意未知错误。
            holes.append(token_id)
            continue
        valid+=1
        try:
            text=raw.decode("utf-8",errors="strict")
        except UnicodeDecodeError:
            # 部分字节token合法，但单独看不是完整Unicode字符。
            invalid_utf8+=1
            continue
        if token_id not in special_ids and regex.search(r"\p{Script=Han}",text):
            # 正负键保证长度降序、同长度ID升序；只保留20项，避免保存整张解码表。
            row=(len(text),-token_id,token_id,text,raw.hex())
            if len(top)<20:heapq.heappush(top,row)
            else:heapq.heappushpop(top,row)
    return dict(name=enc.name,n_vocab=enc.n_vocab,valid_ids=valid,holes=holes,
        ordinary_count=len(enc._mergeable_ranks),special_tokens=enc._special_tokens,
        single_token_invalid_utf8=invalid_utf8,
        longest_20_containing_han=[dict(characters=r[0],id=r[2],text=r[3],bytes_hex=r[4])
                                   for r in sorted(top,reverse=True)])

def run():
    cl=tiktoken.get_encoding("cl100k_base");o=tiktoken.get_encoding("o200k_base")
    scans=[safe_vocabulary_scan(e) for e in (cl,o)]
    assert [s["valid_ids"] for s in scans]==[100261,200000]
    assert [len(s["holes"])for s in scans]==[16,19]
    assert scans[0]["holes"]==[100256]+list(range(100261,100276))
    assert scans[1]["holes"]==[199998]+list(range(200000,200018))
    assert [s["ordinary_count"]for s in scans]==[100256,199998]
    samples=["i love chatgpt","你好，世界！","hello world"," world","1234+4567",
             "café e\u0301","🧑🏽‍💻","\t\n  ","antidisestablishmentarianism","HTTPServer_v2"]
    comparisons=[]
    for text in samples:
        row={"text":text}
        for enc in (cl,o):
            ids=enc.encode(text);raw=[enc.decode_single_token_bytes(i)for i in ids]
            assert b"".join(raw)==text.encode("utf-8")
            assert enc.decode(ids)==text
            pieces=regex.findall(enc._pat_str,text)
            assert "".join(pieces)==text
            # 官方纯Python教育版与Rust生产实现逐ID比较，限定短输入。
            educational=[i for part in pieces for i in bpe_encode(enc._mergeable_ranks,
                         part.encode("utf-8"),visualise=None)]
            assert ids==educational
            row[enc.name]=dict(ids=ids,count=len(ids),pretokenized=pieces,
                              token_bytes_hex=[b.hex()for b in raw])
        comparisons.append(row)
    assert comparisons[0]["cl100k_base"]["count"]==comparisons[0]["o200k_base"]["count"]==5
    # 两套词表并非简单的集合扩张：寻找普通ASCII子词的真实反例。
    old_only=set(cl._mergeable_ranks)-set(o._mergeable_ranks)
    examples=[]
    for raw in sorted(old_only,key=lambda b:(len(b),b)):
        try:s=raw.decode("ascii")
        except UnicodeDecodeError:continue
        if regex.fullmatch(r"[A-Za-z]{3,24}",s):
            left=cl.encode(s);right=o.encode(s)
            if len(left)<len(right):
                examples.append(dict(text=s,cl=left,o=right))
                if len(examples)==3:break
    assert examples
    # 所有token的byte序列保真，逐token损失性Unicode显示再拼接却可能破坏文字。
    text="🧑🏽‍💻";utf8={}
    for enc in (cl,o):
        ids=enc.encode(text)
        lossy="".join(enc.decode([i])for i in ids)
        assert lossy!=text
        assert enc.decode(ids)==text
        utf8[enc.name]={"ids":ids,"per_token_lossy":lossy,"whole_decode":enc.decode(ids)}
    # 真正的special token策略：默认拒绝、显式允许、按普通文本处理。
    special={}
    for enc in (cl,o):
        marker="<|endoftext|>"
        try:enc.encode(marker)
        except ValueError:rejected=True
        else:rejected=False
        assert rejected
        allowed=enc.encode(marker,allowed_special="all")
        ordinary=enc.encode(marker,disallowed_special=())
        assert allowed==[enc._special_tokens[marker]] and ordinary!=allowed
        special[enc.name]=dict(default_rejected=True,allowed=allowed,ordinary=ordinary)
    assert 177431 not in o._special_tokens.values()
    pattern=r'^[\s\W_=+\\-]*$'
    regex_results={s:bool(regex.match(pattern,s))for s in ["abc","123","中文","___","  \n","+-=\\","🙂"]}
    assert regex_results=={"abc":False,"123":False,"中文":False,"___":True,"  \n":True,"+-=\\":True,"🙂":True}
    # 数字的预切分及空格本身是数据。
    assert cl.encode("1234+4567")==[4513,19,10,10961,22]
    assert cl.encode("world")==[14957] and cl.encode(" world")==[1917]
    assert cl.decode([1917,14957])==" worldworld"
    # 小词表展示merge rank决定优先级，而不是编码时重新统计词频。
    toy={bytes([i]):i for i in range(256)}
    toy.update({b"ab":256,b"bc":257})
    toy1=bpe_encode(toy,b"abc",visualise=None)
    assert toy1==[256,99]
    toy[b"ab"]=257;toy[b"bc"]=256
    toy2=bpe_encode(toy,b"abc",visualise=None)
    assert toy2==[97,256]
    return dict(scans=scans,samples=comparisons,old_tokens_absent_from_new=len(old_only),
        new_tokens_absent_from_old=len(set(o._mergeable_ranks)-set(cl._mergeable_ranks)),
        larger_vocab_can_give_more_tokens=examples,utf8_boundaries=utf8,special_policy=special,
        regex_full_symbol_matches=regex_results,toy_merge_ranks={"ab_first":toy1,"bc_first":toy2},
        known_anomalous_id_is_ordinary=True,model_or_api_run=False)
