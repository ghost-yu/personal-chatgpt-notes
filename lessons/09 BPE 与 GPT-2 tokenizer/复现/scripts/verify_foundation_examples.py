"""Execute current foundation examples verbatim; local tokenizer assets, CPU only."""
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[2]
notes = root/'解释.md'
source = json.loads((root/'复现/outputs/source_download.json').read_text(encoding='utf-8'))
cache = Path(source['local_cache'])
verified_assets = []
for entry in source['tokenizer_assets_only']:
    if entry['tokenizer'] != 'gpt2':
        continue
    path = cache/'gpt2'/entry['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    verified_assets.append(entry)
foundation = notes.read_text(encoding='utf-8').split('<details>',1)[0]
blocks = re.findall(r'```python\n(.*?)\n```', foundation, re.S)
assert len(blocks) == 3
scope = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    for block in blocks:
        exec(compile(block,str(notes),'exec'),scope)
assert scope['merges'] == [('a','b'),('a','a'),('ab','a')]
assert scope['parts'] == ['aba','ab'] and scope['ids'] == [4,2]
assert scope['splits'] == {'aa':['aa'],'ab':['ab'],'aba':['aba']}
assert scope['word_freq'] == {'aa':4,'ab':3,'aba':2}
assert len(scope['text']) == 2 and list(scope['raw']) == [226,130,172,240,159,152,128]
assert scope['recovered_bytes'].decode('utf-8') == scope['text']
assert len(scope['byte_map']) == len(scope['reverse_map']) == 256
assert all(scope['reverse_map'][scope['byte_map'][i]] == i for i in range(256))
base, new = scope['base'], scope['new']
assert scope['CACHE'].resolve() == (cache/'gpt2').resolve()
assert base.is_fast and new.is_fast and base is not new
assert len(base) == 50257 and len(new) == 300
assert base.get_vocab() == scope['base_vocab_before']
assert base.eos_token_id == 50256 and new.eos_token_id == 0
assert len(scope['emoji_ids']) == 2
assert [base.decode([i]) for i in scope['emoji_ids']] == ['\ufffd','\ufffd']
assert base.decode(scope['emoji_ids']) == chr(0x1f600)
assert [len(batch) for batch in scope['batches'](scope['corpus'])] == [2,2,2,1]
assert new.decode(scope['new_ids'],clean_up_tokenization_spaces=False) == scope['probe']
assert scope['same_ids'] is True
assert not Path(scope['directory']).exists(), 'Temporary output should be cleaned'
for text in scope['corpus'] + ['\t\r\n  x\u00a0y']:
    assert new.decode(new(text,add_special_tokens=False)['input_ids'],
                      clean_up_tokenization_spaces=False) == text
backend = json.loads(new.backend_tokenizer.to_str())
assert backend['model']['type'] == 'BPE'
assert len(backend['model']['merges']) == 43
assert backend['normalizer'] is None
provenance = json.loads((root/'复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
for entry in provenance['images']:
    assert hashlib.sha256((root/entry['image']).read_bytes()).hexdigest() == entry['image_sha256']
original_image = next(e for e in source['originals'] if e['source']=='imgs/token_steps.png')
assert hashlib.sha256((root/original_image['published_path']).read_bytes()).hexdigest() == original_image['sha256']
corpus_hash = hashlib.sha256(json.dumps(scope['corpus'],ensure_ascii=False).encode('utf-8')).hexdigest()
record = {
    'status':'passed','date':'2026-10-17',
    'notes_sha256':hashlib.sha256(notes.read_bytes()).hexdigest(),
    'executed':'Three Python blocks extracted verbatim before historical details in current 解释.md',
    'code_sha256':[hashlib.sha256(b.encode('utf-8')).hexdigest() for b in blocks],
    'command':[sys.executable,str(Path(__file__).resolve())], 'working_directory':str(Path.cwd()),
    'python':platform.python_version(),'transformers':importlib.metadata.version('transformers'),
    'tokenizers':importlib.metadata.version('tokenizers'),'device':'CPU',
    'stdout':stdout.getvalue(),
    'manual_bpe':{'word_frequency':scope['word_freq'],'merges':scope['merges'],
                  'final_splits':scope['splits'],'unseen_word':'abaab','pieces':scope['parts'],'ids':scope['ids']},
    'byte_mapping':{'input':scope['text'],'byte_values':list(scope['raw']),
                    'all_256_entries_bijective':True,'display':scope['shown']},
    'real_tokenizer_training':{'corpus':scope['corpus'],'corpus_sha256':corpus_hash,
        'rows':7,'batch_sizes':[2,2,2,1],'base_vocabulary':len(base),'new_vocabulary':len(new),
        'merges':len(backend['model']['merges']),'base_unchanged':True,
        'base_eos_id':base.eos_token_id,'new_eos_id':new.eos_token_id,
        'emoji_ids':scope['emoji_ids'],'emoji_individual_decode':['\ufffd','\ufffd'],
        'probe':scope['probe'],'new_ids':scope['new_ids'],
        'all_corpus_roundtrip':True,'save_reload_same_ids':True,'temporary_directory_cleaned':True},
    'verified_gpt2_assets':verified_assets,'source_commit':source['commit'],
    'paper_images':provenance['images'],'paper_images_visually_inspected':True,
    'course_image_sha256':original_image['sha256'],'course_image_visually_inspected':True,
    'limits':'Independent character-BPE, UTF-8 map, seven-row real fast-tokenizer training only. No model weights, old 2000-row/runtime/T5 verifier rerun, codeparrot data, original 100k/200k experiment, Hub upload, language/translation quality or performance test.'
}
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(stdout.getvalue())
print('Manual BPE, byte mapping, real local training, roundtrip, reload and image hashes passed.')
