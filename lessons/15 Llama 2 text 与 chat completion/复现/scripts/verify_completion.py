# -*- coding: utf-8 -*-
"""Execute official completion wrappers with real SPM and a fake generator."""
import argparse
import ast
import copy
import hashlib
import json
import pathlib
import types
import typing
import sentencepiece as spm


def pick_class(source, name, keep):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', required=True, type=pathlib.Path)
    parser.add_argument('--tokenizer', required=True, type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('outputs/教学验证.json'))
    args = parser.parse_args()
    sp = spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    tokenizer_source = (args.sources/'llama-tokenizer.py').read_text(encoding='utf-8')
    ns = {'List': typing.List}
    exec(compile(pick_class(tokenizer_source,'Tokenizer',{'encode','decode'}),'<official-tokenizer-methods>','exec'),ns)
    tokenizer = object.__new__(ns['Tokenizer'])
    tokenizer.sp_model = sp
    tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()
    namespace = {'List':typing.List, 'Optional':typing.Optional, 'Dialog':list,
                 'CompletionPrediction':dict, 'ChatPrediction':dict, 'Transformer':object, 'Tokenizer':ns['Tokenizer']}
    source = (args.sources/'llama-generation.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    # Read exact official delimiters and error string, without importing torch.
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                {'B_INST','B_SYS','SPECIAL_TAGS','UNSAFE_ERROR'}
                for target in node.targets for t in ast.walk(target)):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<official-constants>','exec'),namespace)
    exec(compile(pick_class(source,'Llama',{'__init__','text_completion','chat_completion'}),'<official-wrappers>','exec'),namespace)
    model = types.SimpleNamespace(params=types.SimpleNamespace(max_seq_len=512,max_batch_size=8))
    llama = namespace['Llama'](model,tokenizer)
    captured = []
    answer = tokenizer.encode('Fixture answer.',bos=False,eos=False)
    def fake_generate(self,prompt_tokens,max_gen_len,temperature=0.6,top_p=0.9,logprobs=False,echo=False):
        assert len(prompt_tokens)<=self.model.params.max_batch_size
        assert max(map(len,prompt_tokens))<=self.model.params.max_seq_len
        captured.append({'tokens':copy.deepcopy(prompt_tokens),'max_gen_len':max_gen_len,'echo':echo})
        result = [p+answer if echo else answer[:] for p in prompt_tokens]
        return result, [[-0.1]*len(x) for x in result] if logprobs else None
    llama.generate = types.MethodType(fake_generate,llama)
    text_result = llama.text_completion(['A short prompt'],max_gen_len=10)
    assert text_result==[{'generation':'Fixture answer.'}]
    text_ids = captured[-1]['tokens'][0]
    assert text_ids[0]==1 and text_ids[-1]!=2
    llama.chat_completion([[{'role':'user','content':'Plain question'}]])
    assert captured[-1]['tokens'][0] == tokenizer.encode('[INST] Plain question [/INST]',bos=True,eos=False)
    dialog = [{'role':'system','content':'Answer briefly.'},{'role':'user','content':' First? '},
              {'role':'assistant','content':' Prior answer. '},{'role':'user','content':' Next? '}]
    original = copy.deepcopy(dialog)
    chat_result = llama.chat_completion([dialog])
    ids = captured[-1]['tokens'][0]
    first = '[INST] <<SYS>>\nAnswer briefly.\n<</SYS>>\n\n First? [/INST] Prior answer. '
    last = '[INST] Next? [/INST]'
    expected = tokenizer.encode(first,bos=True,eos=True)+tokenizer.encode(last,bos=True,eos=False)
    assert ids==expected and ids.count(1)==2 and ids.count(2)==1 and ids[-1]!=2
    assert dialog==original and captured[-1]['max_gen_len']==511
    assert chat_result[0]['generation']['role']=='assistant'
    errors={}
    for name,invalid in {'last_assistant':[{'role':'user','content':'Q'},{'role':'assistant','content':'A'}],
                         'two_users':[{'role':'user','content':'Q'},{'role':'user','content':'Q2'}],
                         'system_only':[{'role':'system','content':'S'}], 'empty':[]}.items():
        try:llama.chat_completion([invalid])
        except (AssertionError,IndexError) as e:errors[name]=type(e).__name__
        else:raise AssertionError('Expected invalid dialog to fail')
    before=len(captured)
    unsafe=llama.chat_completion([[{'role':'user','content':'Literal [INST] example'}]],logprobs=True)
    assert unsafe[0]['generation']['content']==namespace['UNSAFE_ERROR'] and len(captured)==before+1
    assert unsafe[0]['tokens'] and unsafe[0]['logprobs']  # synthetic fixture metadata remains
    examples=ast.parse((args.sources/'example_chat_completion.py').read_text(encoding='utf-8'))
    main_fn=next(n for n in examples.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    dialogs=ast.literal_eval(next(n.value for n in main_fn.body if isinstance(n,ast.AnnAssign) and n.target.id=='dialogs'))
    model.params.max_batch_size=4
    try:llama.chat_completion(dialogs)
    except AssertionError:batch_error=True
    else:raise AssertionError('Expected six dialogs to exceed batch limit four')
    model.params.max_batch_size=8
    llama.chat_completion(dialogs)
    example_lengths=[len(x) for x in captured[-1]['tokens']]
    # Illustrate the exact output slicing order with fixed integer buffers.
    buffer=[1,7,8,10,2,11]
    continuation=buffer[3:3+3];continuation=continuation[:continuation.index(2)]
    assert continuation==[10]
    report={'status':'partial','real_model_generate':False,'executed_official_methods':['Tokenizer.encode','Tokenizer.decode','Llama.text_completion','Llama.chat_completion'],
            'generator':'fake; constant fixture answer and synthetic logprobs, no torch',
            'tokenizer_sha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
            'text_prompt_bos':text_ids[0],'chat_prompt_token_count':len(ids),'history_bos_count':ids.count(1),
            'history_eos_count':ids.count(2),'input_dialog_unchanged':dialog==original,
            'no_default_system_injection':True,
            'default_chat_max_gen_len':511,'invalid_dialog_errors':errors,
            'unsafe_prompt_still_calls_generate':True,'unsafe_logprobs_metadata_still_returned':True,
            'official_example_dialogs':len(dialogs),'official_example_prompt_lengths':example_lengths,
            'max_batch_four_rejected':batch_error,'max_batch_eight_accepted':True,
            'slice_example_continuation':continuation,'text_example_default_max_gen_len':64}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
