# -*- coding: utf-8 -*-
"""Real local Arrow/JSONL experiments. No remote corpus or model downloads."""
import json
import platform
import tempfile
from pathlib import Path

def main():
    import datasets
    import pyarrow as pa
    import fsspec
    import numpy
    import pandas
    import huggingface_hub
    from datasets import load_dataset, load_from_disk
    rows=[{'id':i,'text':text,'label':i%2} for i,text in enumerate(['hello world','中文文本','a b c','stream me','last row'])]
    eager_seen=[]
    lazy_seen=[]
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        file=folder/'tiny.jsonl'
        file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        ds=load_dataset('json',data_files={'train':str(file)},split='train',cache_dir=str(folder/'cache'))
        stream=load_dataset('json',data_files={'train':str(file)},split='train',streaming=True)
        assert len(ds)==5 and ds[0]==rows[0]
        assert ds[1:3]=={'id':[1,2],'text':['中文文本','a b c'],'label':[1,0]}
        def eager_fn(row):
            eager_seen.append(row['id'])
            return {'chars':len(row['text'])}
        mapped=ds.map(eager_fn,load_from_cache_file=False)
        assert eager_seen==list(range(5))
        assert 'chars' not in ds.column_names and 'chars' in mapped.column_names
        def lazy_fn(row):
            lazy_seen.append(row['id'])
            return {'chars':len(row['text'])}
        lazy=stream.map(lazy_fn)
        assert lazy_seen==[]
        first=next(iter(lazy))
        assert lazy_seen==[0] and first['chars']==11
        again=list(lazy.take(2))
        assert lazy_seen==[0,0,1]
        assert list(stream.take(2))==rows[:2]
        assert next(iter(stream.skip(3)))==ds[3]
        stream_len_error=None
        try:len(stream)
        except TypeError as e:stream_len_error=type(e).__name__
        assert stream_len_error=='TypeError'
        selected=ds.select([3,1]);assert selected['id']==[3,1]
        shuffled=list(stream.shuffle(seed=42,buffer_size=3))
        assert sorted(r['id'] for r in shuffled)==list(range(5))
        assert shuffled==list(stream.shuffle(seed=42,buffer_size=3))
        def batch_fn(batch):
            assert isinstance(batch['text'],list)
            return {'chars':[len(t) for t in batch['text']]}
        batched=ds.map(batch_fn,batched=True,batch_size=2,load_from_cache_file=False)
        assert batched['chars']==mapped['chars']
        mapped.save_to_disk(str(folder/'saved'))
        disk=load_from_disk(str(folder/'saved'),keep_in_memory=False)
        assert disk.to_dict()==mapped.to_dict()
        assert type(disk.data).__name__=='MemoryMappedTable'
        # Explicit Arrow memory mapping, rather than inferring it from low RSS.
        arrowfile=folder/'table.arrow'
        table=pa.Table.from_pylist(rows)
        with pa.OSFile(str(arrowfile),'wb') as sink:
            with pa.ipc.new_stream(sink,table.schema) as writer:writer.write_table(table)
        with pa.memory_map(str(arrowfile),'r') as source:
            loaded=pa.ipc.open_stream(source).read_all()
            assert loaded.to_pylist()==rows
            arrow_source_type=type(source).__name__
        result={'status':'passed','scope':'5 local JSONL rows; real datasets/Arrow API, no web streaming or large-corpus benchmark',
          'environment':{'python':platform.python_version(),'datasets':datasets.__version__,'pyarrow':pa.__version__,'fsspec':fsspec.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__,'huggingface_hub':huggingface_hub.__version__},
          'dataset_type':type(ds).__name__,'stream_type':type(stream).__name__,'disk_table_type':type(disk.data).__name__,'arrow_source_type':arrow_source_type,
          'rows':len(ds),'dataset_nbytes':ds.data.nbytes,'jsonl_bytes':file.stat().st_size,'arrow_file_bytes':arrowfile.stat().st_size,
          'single_row':ds[0],'slice_rows_1_3':ds[1:3], 'eager_calls_after_map':eager_seen,'lazy_calls_after_definition':0,'lazy_calls_after_first_next':1,'lazy_calls_after_new_take2':lazy_seen,
          'take2':list(stream.take(2)),'skip3_first':next(iter(stream.skip(3))),'stream_len_error':stream_len_error,'mapped_char_lengths':mapped['chars'],
          'shuffled_ids':[r['id'] for r in shuffled],'save_reload_equal':True,'memory_map_verified':True}
    out=Path(__file__).resolve().parents[1]/'outputs';out.mkdir(parents=True,exist_ok=True)
    (out/'教学验证.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
