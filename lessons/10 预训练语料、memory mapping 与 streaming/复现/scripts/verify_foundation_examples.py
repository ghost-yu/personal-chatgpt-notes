"""Execute current six-row foundation examples verbatim; no corpus/model download."""
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
foundation = notes.read_text(encoding='utf-8').split('<details>',1)[0]
blocks = re.findall(r'```python\n(.*?)\n```',foundation,re.S)
assert len(blocks) == 3
scope = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    for block in blocks:
        exec(compile(block,str(notes),'exec'),scope)
assert scope['first'] == {'id':0,'text':'cat'}
assert scope['second'] == {'id':1,'text':'中文'}
assert scope['remaining_ids'] == [2,3,4,5]
assert scope['replay_ids'] == [0,1,2,3,4,5]
assert scope['normal_type'] == 'Dataset'
assert scope['storage_type'] == 'MemoryMappedTable'
assert scope['selected_row'] == {'id':3,'text':'hello'}
assert scope['slice_rows'] == {'id':[1,2],'text':['中文','a b']}
assert scope['eager_seen'] == [0,1,2,3,4,5]
assert scope['original_columns'] == ['id','text']
assert scope['mapped_columns'] == ['id','text','chars']
assert scope['before_next'] == [] and scope['lazy_seen'] == [0]
assert scope['first_lazy'] == {'id':0,'text':'cat','chars':3}
assert [row['id'] for row in scope['first_two']] == [0,1]
assert [row['id'] for row in scope['after_two']] == [2,3]
assert scope['mapped_file_type'] == 'MemoryMappedFile'
assert scope['chinese_text'] == '中文'
assert scope['restored_rows'] == scope['rows']
assert not scope['file'].exists() and not scope['arrow_file'].exists()
assert not Path(scope['folder']).exists()
source = json.loads((root/'复现/outputs/source_download.json').read_text(encoding='utf-8'))
for entry in source['files']:
    assert hashlib.sha256((root/entry['published_path']).read_bytes()).hexdigest() == entry['sha256']
provenance = json.loads((root/'复现/outputs/论文截图来源.json').read_text(encoding='utf-8'))
for entry in provenance['images']:
    assert hashlib.sha256((root/entry['image']).read_bytes()).hexdigest() == entry['image_sha256']
record = {
    'status':'passed','date':'2026-10-17',
    'notes_sha256':hashlib.sha256(notes.read_bytes()).hexdigest(),
    'executed':'Three Python blocks extracted verbatim before historical details in current 解释.md',
    'code_sha256':[hashlib.sha256(b.encode('utf-8')).hexdigest() for b in blocks],
    'command':[sys.executable,str(Path(__file__).resolve())], 'working_directory':str(Path.cwd()),
    'python':platform.python_version(),'datasets':importlib.metadata.version('datasets'),
    'pyarrow':importlib.metadata.version('pyarrow'),'device':'CPU',
    'stdout':stdout.getvalue(),'rows':scope['rows'],
    'rows_sha256':hashlib.sha256(json.dumps(scope['rows'],ensure_ascii=False).encode('utf-8')).hexdigest(),
    'jsonl_iteration':{'first':scope['first'],'second':scope['second'],
        'remaining_ids':scope['remaining_ids'],'replay_ids':scope['replay_ids'],
        'jsonl_file_bytes_not_rss':scope['jsonl_bytes']},
    'hf_dataset':{'ordinary_object':scope['normal_type'],'ordinary_storage':scope['storage_type'],
        'streaming_object':'IterableDataset','index3':scope['selected_row'],'slice1_3':scope['slice_rows'],
        'eager_calls_at_return':scope['eager_seen'],'original_columns':scope['original_columns'],
        'mapped_columns':scope['mapped_columns'],'lazy_calls_at_definition':scope['before_next'],
        'lazy_calls_after_one_output':scope['lazy_seen'],'lazy_first':scope['first_lazy'],
        'take2_ids':[0,1],'skip2_take2_ids':[2,3]},
    'arrow_mapping':{'actual_object':scope['mapped_file_type'],'row1_text':scope['chinese_text'],
        'ipc_file_bytes_not_rss':scope['arrow_file_bytes'],'arrow_nbytes_not_rss':scope['table_nbytes'],
        'all_six_rows_restored':True,'temporary_files_cleaned':True},
    'paper_images':provenance['images'],'paper_images_visually_inspected':True,
    'original_image_sha256':{e['source']:e['sha256'] for e in source['files'] if e['source'].endswith('.png')},
    'all_four_course_images_visually_inspected':True,'source_commit':source['commit'],
    'limits':'Six-row independent CPU teaching examples only. Ordinary map output explicitly kept in memory. No old five-row/notebook/GPT-2/Windows two-worker verifier rerun, large-corpus download, C4/Pile/T5 reconstruction, tokenizer/model training, RSS/peak-memory, disk-bandwidth or quality/performance experiment.'
}
(root/'复现/outputs/基础改写验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(stdout.getvalue())
print('JSONL iterator, eager/lazy map, index/slice, take/skip, real mmap and image hashes passed.')
