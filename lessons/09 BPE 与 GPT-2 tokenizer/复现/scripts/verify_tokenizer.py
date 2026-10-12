# -*- coding: utf-8 -*-
"""Small real GPT-2 fast-tokenizer retraining; no model weights or Hub writes."""
import argparse
import itertools
import json
import keyword
import os
import platform
import tempfile
import urllib.request
from pathlib import Path
from collections import Counter

REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
CORPUS = ["def say_hello():\n    print('Hello, World!')\n", "async def task():\n    await task()\n", "if value is None:\n    return False\n", "# 中文注释 € 😀\nx = 1\n"] * 4

def safe_batches(data, limit, batch_size):
    it = iter(data)
    for start in range(0, limit, batch_size):
        batch = list(itertools.islice(it, min(batch_size, limit-start)))
        if not batch:
            return
        yield [row['content'] for row in batch]

def original_batches(it, length, size):
    for _ in range(0,length,size):
        yield [next(it)['content'] for _ in range(size)]

def main():
    # Tokenizer-only process: do not import the local incomplete torch namespace.
    os.environ['USE_TORCH']='0'
    os.environ['USE_TF']='0'
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path)
    parser.add_argument('--download-base',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    base=args.base or root/'资料/gpt2-tokenizer'
    if args.download_base:
        base.mkdir(parents=True,exist_ok=True)
        for name in ['tokenizer.json','tokenizer_config.json','config.json']:
            data=urllib.request.urlopen('https://huggingface.co/openai-community/gpt2/resolve/'+REVISION+'/'+name,timeout=30).read()
            (base/name).write_bytes(data)
    old=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    assert old.is_fast and len(old)==50257
    original_json=old.backend_tokenizer.to_str()
    mapping=bytes_to_unicode()
    inverse={v:k for k,v in mapping.items()}
    assert len(mapping)==len(inverse)==256
    assert mapping[32]=='Ġ' and mapping[10]=='Ċ' and mapping[13]=='č'
    sample='中文 € 😀\n    print(1)\t'
    byte_roundtrip=bytes(inverse[mapping[b]] for b in sample.encode('utf-8')).decode('utf-8')
    assert byte_roundtrip==sample
    def iterator():
        for i in range(0,len(CORPUS),3):yield CORPUS[i:i+3]
    new=old.train_new_from_iterator(iterator(),vocab_size=320,initial_alphabet=list(inverse),show_progress=False)
    assert old.backend_tokenizer.to_str()==original_json
    enc=new(sample,add_special_tokens=False)
    assert new.decode(enc['input_ids'],clean_up_tokenization_spaces=False)==sample
    with tempfile.TemporaryDirectory() as tmp:
        new.save_pretrained(tmp)
        loaded=AutoTokenizer.from_pretrained(tmp,local_files_only=True)
        assert loaded(sample,add_special_tokens=False)['input_ids']==enc['input_ids']
    data=[{'content':str(i)} for i in range(7)]
    assert list(safe_batches(data,5,2))==[['0','1'],['2','3'],['4']]
    assert list(safe_batches(data,12,3))==[['0','1','2'],['3','4','5'],['6']]
    it=iter(data)
    first=list(original_batches(it,4,2))
    second=list(original_batches(it,2,2))
    assert first==[['0','1'],['2','3']] and second==[['4','5']]
    exhaustion=None
    try:list(original_batches(iter(data),8,3))
    except RuntimeError as e:exhaustion=type(e).__name__+': '+str(e)
    assert exhaustion=='RuntimeError: generator raised StopIteration'
    words={'aa':4,'ab':3,'aba':2}
    counts=Counter()
    for word,freq in words.items():
        for pair in zip(word,word[1:]):counts[pair]+=freq
    assert counts[('a','b')]==5 and counts[('a','a')]==4
    info=json.loads(new.backend_tokenizer.to_str())
    result={'status':'passed','scope':'real fast tokenizer training on 16 local snippets; no LM training',
      'environment':{'python':platform.python_version(),'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},
      'gpt2_revision':REVISION,'original_vocab_size':len(old),'new_vocab_size':len(new),'training_snippets':len(CORPUS),
      'byte_mapping':{'space':mapping[32],'newline':mapping[10],'carriage_return':mapping[13],'euro_bytes':list('€'.encode('utf-8'))},
      'sample':sample,'new_ids':enc['input_ids'],'new_tokens':new.convert_ids_to_tokens(enc['input_ids']),
      'full_decode':new.decode(enc['input_ids'],clean_up_tokenization_spaces=False),'roundtrip_reload':True,'old_tokenizer_unchanged':True,
      'new_special_token_ids':new.all_special_ids,'new_merges_count':len(info['model']['merges']),
      'original_euro_individual_decode':[old.decode([i],clean_up_tokenization_spaces=False) for i in old.encode('€')],
      'original_emoji_individual_decode':[old.decode([i],clean_up_tokenization_spaces=False) for i in old.encode('😀')],
      'python_keyword_count':len(keyword.kwlist),'new_keyword_presence':{k:k in new.get_vocab() for k in ['def','return','await','nonlocal']},
      'stream_first_run':first,'stream_second_run':second,'original_exhaustion':exhaustion,'safe_partial_batch':list(safe_batches(data,5,2)),
      'weighted_pair_counts':{' '.join(k):v for k,v in counts.items()}}
    out=root/'复现/outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'教学验证.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
