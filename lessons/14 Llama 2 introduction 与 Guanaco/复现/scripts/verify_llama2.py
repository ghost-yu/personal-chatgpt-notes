# -*- coding: utf-8 -*-
"""Real SentencePiece encoding plus arithmetic/static checks; no LLM weights."""
import argparse
import ast
import hashlib
import json
import math
import pathlib
import re
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True, type=pathlib.Path)
    parser.add_argument('--samples', type=pathlib.Path)
    parser.add_argument('--script', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('outputs/教学验证.json'))
    args = parser.parse_args()
    # Read with pathlib to avoid native Windows non-ASCII filename handling.
    sp = spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    assert sp.vocab_size() == 32000 and sp.bos_id() == 1 and sp.eos_id() == 2
    texts = ['Recommend a film about space.', '请推荐一部关于宇宙的电影。']
    counts = []
    for text in texts:
        ids = sp.encode(text, out_type=int)
        assert sp.decode(ids) == text and all(0 <= i < 32000 for i in ids)
        counts.append(len(ids))
    prompt = '[INST] <<SYS>>\nAnswer briefly.\n<</SYS>>\n\nRecommend a film about space. [/INST]'
    chat_ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    assert chat_ids[0] == 1 and chat_ids[-1] != 2
    # Human/Assistant delimiters are text; they are not one special token each.
    marker_counts = {x: len(sp.encode(x, out_type=int)) for x in ['### Human:', '### Assistant:', '[INST]', '[/INST]']}
    h, inter, vocab, layers, rank = 4096, 11008, 32000, 32, 64
    logical_params = 2*vocab*h + layers*(4*h*h + 3*h*inter + 2*h) + h
    adapter_params = layers*2*rank*(h+h)
    assert logical_params == 6738415616 and adapter_params == 33554432
    row_reports = []
    if args.samples:
        rows = json.loads(args.samples.read_text(encoding='utf-8'))
        for row in rows:
            text = row['text']
            ids = [sp.bos_id()] + sp.encode(text, out_type=int)
            roles = re.findall(r'### (Human|Assistant):', text)
            row_reports.append({'token_count_with_bos': len(ids), 'over_512': len(ids)>512,
                                'tokens_removed_by_512_cap': max(0,len(ids)-512),
                                'roles': roles, 'last_role': roles[-1] if roles else None})
    defaults = {}
    if args.script:
        tree = ast.parse(args.script.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScriptArguments')
        for n in cls.body:
            if isinstance(n,ast.AnnAssign) and isinstance(n.value,ast.Call):
                for kw in n.value.keywords:
                    if kw.arg=='default': defaults[n.target.id] = ast.literal_eval(kw.value)
        assert defaults['seq_length']==512 and defaults['peft_lora_r']==64
        assert defaults['peft_lora_alpha']==16 and defaults['num_train_epochs']==3
    micro_batches = math.ceil(9846/16)
    update_steps = 3*(micro_batches//2)
    assert micro_batches==616 and update_steps==924
    report = {'status':'partial','model_training':False,'tokenizer_runtime':spm.__version__,
              'tokenizer_sha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
              'special_ids':{'bos':sp.bos_id(),'eos':sp.eos_id(),'unk':sp.unk_id()},
              'vocab':sp.vocab_size(),'own_text_token_counts':counts,'roundtrip':'passed',
              'chat_prompt_tokens_including_single_bos':len(chat_ids),'delimiter_token_counts':marker_counts,
              'samples':row_reports,'script_defaults':defaults,'logical_parameters':logical_params,
              'default_q_v_lora_parameters':adapter_params,'lora_scale':16/64,
              'micro_batches_per_epoch':micro_batches,'planned_optimizer_steps':update_steps,
              'raw_formula_fractional_steps':9846*3/32,'fp16_weight_gib':logical_params*2/1024**3,
              'context_budget':{'context':4096,'input':3500,'remaining':596}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
