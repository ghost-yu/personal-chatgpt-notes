# -*- coding: utf-8 -*-
"""第10讲独立教学补充：原notebook没有这些实现。只处理小型本地数据。

路线：JSONL→HF Dataset/IterableDataset→map/tokenizer→批次；另验证Arrow映射。
不下载C4/Pile、不训练模型、不把微型运行耗时或RSS当成大语料性能。
"""
import gc
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
from transformers import AutoTokenizer


class IndexedRows(Dataset):
    """PyTorch map-style：rows为CPU float32[N,5]，索引返回一个样本dict。"""
    def __init__(self, rows):
        # 保存Tensor引用，不复制全部数据；本教学Tensor只有100×5项。
        self.rows = rows

    def __len__(self):
        # len(dataset)调用此方法；第0维是样本数，不是特征数。
        return self.rows.shape[0]

    def __getitem__(self, index):
        # 方括号索引自动调用，返回features[5]与Python整数index。
        return {'features': self.rows[index], 'index': index}


class RangeRows(IterableDataset):
    """PyTorch iterable-style：顺序产生start到stop-1；按真实worker交错分片。"""
    def __init__(self, start, stop):
        self.start, self.stop = start, stop

    def __iter__(self):
        # 在主进程num_workers=0时为None；worker进程里带id/num_workers。
        worker = get_worker_info()
        start = self.start if worker is None else self.start + worker.id
        step = 1 if worker is None else worker.num_workers
        # 两worker分别3/5/7和4/6，覆盖但不重复；yield返回dict而非一次性list。
        for i in range(start, self.stop, step):
            yield {'value': i}


def take_and_close(dataset, count):
    """最多读取count条，返回list[dict]；显式关闭生成器释放Windows文件句柄。"""
    iterator = iter(dataset)
    rows = []
    try:
        for _ in range(count):
            try:
                rows.append(next(iterator))
            except StopIteration:
                break
    finally:
        if hasattr(iterator, 'close'):
            iterator.close()
    return rows


def run_demo(folder, tokenizer_dir):
    """folder为本次临时Path；tokenizer_dir指向固定GPT2资产；返回实测结果dict。"""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    # str字符数、UTF8字节数、token数是三个指标，label只是演示整数列。
    rows = [{'id': i, 'text': text, 'label': i % 2} for i, text in enumerate(
        ['hello world', '中文文本', 'a b c', 'stream me', 'last row'])]
    jsonl = folder / 'tiny.jsonl'
    # ensure_ascii=False写实际UTF8中文；write_bytes明确LF，避免平台换行干扰大小比较。
    jsonl.write_bytes((''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)).encode('utf-8'))
    # 默认streaming=False：准备Arrow缓存后返回常规Dataset，可用len/索引。
    normal = load_dataset('json', data_files={'train': str(jsonl)}, split='train', cache_dir=str(folder / 'cache'))
    # 同一文件的streaming=True返回按需顺序读取的HF IterableDataset。
    stream = load_dataset('json', data_files={'train': str(jsonl)}, split='train', streaming=True)
    assert len(normal) == 5 and normal[0] == rows[0]
    assert normal[1:3] == {'id': [1, 2], 'text': ['中文文本', 'a b c'], 'label': [1, 0]}
    assert normal.select([3, 1])['id'] == [3, 1]
    # 这里只打印5行样本，不对大语料使用list(stream)物化全量。
    assert take_and_close(stream, 5) == rows
    try:
        len(stream)
    except TypeError as error:
        len_error = str(error)
    else:
        raise AssertionError('流式数据本例不提供len')

    eager_seen, lazy_seen = [], []
    def add_chars(row):
        # 单样本dict→新增一列；seen是观察调用时机的教学副作用。
        eager_seen.append(row['id'])
        return {'chars': len(row['text'])}
    mapped_file = str(folder / 'mapped.arrow')
    # 返回新Dataset；强制真正执行转换，不因命中缓存掩盖处理时机。
    mapped = normal.map(add_chars, load_from_cache_file=False, cache_file_name=mapped_file)
    assert eager_seen == [0, 1, 2, 3, 4] and 'chars' not in normal.column_names
    assert mapped['chars'] == [11, 4, 5, 9, 8]
    # 显式复用同一已存在cache_file_name：本例无再次调用；不能泛化函数改动后的缓存正确性。
    cached = normal.map(add_chars, load_from_cache_file=True, cache_file_name=mapped_file)
    assert eager_seen == [0, 1, 2, 3, 4] and cached.to_dict() == mapped.to_dict()

    def add_chars_lazy(row):
        lazy_seen.append(row['id'])
        return {'chars': len(row['text'])}
    lazy = stream.map(add_chars_lazy)
    assert lazy_seen == []  # 仅构建管线，还没有处理样本。
    first = take_and_close(lazy, 1)[0]
    assert first['chars'] == 11 and lazy_seen == [0]
    again = take_and_close(lazy.take(2), 2)
    assert lazy_seen == [0, 0, 1]  # 新迭代器重新走来源和转换，不复用首轮全部结果。

    # batched map输入是dict[str,list]；输出列长度必须一致。
    batch_seen = []
    def batch_chars(batch):
        batch_seen.append(list(batch['id']))
        return {'chars': [len(text) for text in batch['text']]}
    batched = stream.map(batch_chars, batched=True, batch_size=2)
    assert batch_seen == []
    first_batch_result = take_and_close(batched, 1)[0]
    assert first_batch_result['chars'] == 11 and batch_seen == [[0, 1]]
    # 请求一条输出也处理了两条输入，所以map batch_size不等于下游DataLoader batch_size。
    take2 = take_and_close(stream.take(2), 5)
    skip3 = take_and_close(stream.skip(3), 1)[0]
    assert take2 == rows[:2] and skip3 == rows[3]
    shuffle1 = take_and_close(stream.shuffle(seed=42, buffer_size=3), 5)
    shuffle2 = take_and_close(stream.shuffle(seed=42, buffer_size=3), 5)
    assert shuffle1 == shuffle2 and sorted(r['id'] for r in shuffle1) == list(range(5))

    # map().skip()先转换被跳过行；skip().map()跳过后才转换，用计数而非猜测验证。
    map_skip_seen, skip_map_seen = [], []
    def before_skip(row):
        map_skip_seen.append(row['id']); return {'chars': len(row['text'])}
    def after_skip(row):
        skip_map_seen.append(row['id']); return {'chars': len(row['text'])}
    assert take_and_close(stream.map(before_skip).skip(3), 1)[0]['id'] == 3
    assert take_and_close(stream.skip(3).map(after_skip), 1)[0]['id'] == 3
    assert map_skip_seen == [0, 1, 2, 3] and skip_map_seen == [3]

    # Arrow保存/重载与独立mmap读取；不以“内存似乎低”代替映射证据。
    mapped.save_to_disk(str(folder / 'saved'))
    disk = load_from_disk(str(folder / 'saved'), keep_in_memory=False)
    assert type(disk.data).__name__ == 'MemoryMappedTable' and disk.to_dict() == mapped.to_dict()
    arrowfile = folder / 'table.arrow'; table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(arrowfile), 'wb') as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    with pa.memory_map(str(arrowfile), 'r') as memory_file:
        loaded = pa.ipc.open_stream(memory_file).read_all()
        assert loaded.to_pylist() == rows  # 仅5行物化，TB语料不可照搬全量to_pylist。
        memory_type = type(memory_file).__name__

    # 固定本地GPT2 tokenizer：不调用网络，不加载Transformer权重。
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    assert tokenizer.is_fast and len(tokenizer) == 50257
    token_seen = []
    def tokenize_batch(batch):
        token_seen.append(list(batch['id']))
        # truncation=True使用tokenizer的默认模型长度；这里没有padding，输出变长list。
        return tokenizer(batch['text'], truncation=True)
    tokenized = stream.map(tokenize_batch, batched=True, batch_size=2)
    assert token_seen == []
    tok_first = take_and_close(tokenized, 1)[0]
    assert token_seen == [[0, 1]]
    assert tok_first['input_ids'] == tokenizer(rows[0]['text'])['input_ids']
    # 对整个5行再迭代，tokenized实例本身没有保存已处理全部结果。
    token_rows = take_and_close(tokenized, 5)
    assert [tokenizer.decode(r['input_ids'], clean_up_tokenization_spaces=False) for r in token_rows] == [r['text'] for r in rows]
    # GPT2本来没有pad；明确复用EOS并用attention_mask区分实际token与padding。
    tokenizer.pad_token = tokenizer.eos_token
    padded = tokenizer([r['text'] for r in rows], padding=True, truncation=True, return_tensors='pt')
    assert padded['input_ids'].dtype == torch.int64 and padded['input_ids'].shape[0] == 5
    assert padded['attention_mask'].shape == padded['input_ids'].shape
    for i, text in enumerate(r['text'] for r in rows):
        ids = padded['input_ids'][i][padded['attention_mask'][i].bool()].tolist()
        assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == text

    # 真正PyTorch map-style Dataset：100样本、5特征；默认collate组成规则Tensor。
    features = torch.arange(500, dtype=torch.float32).reshape(100, 5)
    indexed = IndexedRows(features)
    batches = list(DataLoader(indexed, batch_size=32, shuffle=False, num_workers=0))
    assert [len(b['index']) for b in batches] == [32, 32, 32, 4]
    assert torch.equal(torch.cat([b['features'] for b in batches]), features)
    assert torch.equal(torch.cat([b['index'] for b in batches]), torch.arange(100))
    # 常规HF Dataset也满足索引协议；字符串列只组为list，不会自动变成token ID。
    hf_batch = next(iter(DataLoader(normal, batch_size=2, num_workers=0)))
    assert hf_batch['id'].dtype == torch.int64 and hf_batch['text'] == [r['text'] for r in rows[:2]]
    range_data = RangeRows(3, 8)
    single = [i for batch in DataLoader(range_data, batch_size=2, num_workers=0) for i in batch['value'].tolist()]
    multiple = [i for batch in DataLoader(range_data, batch_size=2, num_workers=2) for i in batch['value'].tolist()]
    assert single == [3, 4, 5, 6, 7] and sorted(multiple) == single and len(set(multiple)) == 5

    result = {
        'scope': '5行JSONL、100×5特征与range3–7；独立教学补充，非原保存代码',
        'data_types': {'normal': type(normal).__name__, 'stream': type(stream).__name__,
                       'disk_table': type(disk.data).__name__, 'memory_file': memory_type},
        'rows': rows, 'slice1_3': normal[1:3], 'selected_ids': normal.select([3, 1])['id'],
        'sizes': {'jsonl_bytes': jsonl.stat().st_size, 'Arrow_data_nbytes': normal.data.nbytes,
                  'Arrow_IPC_file_bytes': arrowfile.stat().st_size, 'not_RSS_measurements': True},
        'JSONL_sha256': hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        'eager_map_seen': eager_seen, 'explicit_cache_hit_added_calls': 0,
        'lazy_seen_definition': [], 'lazy_seen_after_first_and_new_take2': lazy_seen,
        'batched_map_seen_for_one_output': batch_seen, 'chars': mapped['chars'],
        'map_then_skip_seen': map_skip_seen, 'skip_then_map_seen': skip_map_seen,
        'take2': take2, 'skip3_first': skip3, 'stream_len_error': len_error,
        'shuffle_seed42_buffer3_ids': [r['id'] for r in shuffle1], 'mmap_and_saved_rows_equal': True,
        'tokenizer': {'vocab': len(tokenizer), 'first_output_triggered_input_ids': [0, 1],
                      'calls_after_reiterating': token_seen, 'token_rows': token_rows,
                      'padded_ids_shape': list(padded['input_ids'].shape), 'ids_dtype': str(padded['input_ids'].dtype),
                      'mask': padded['attention_mask'].tolist(), 'all_text_roundtrips': True},
        'PyTorch': {'IndexedRows_samples': len(indexed), 'features_shape': [100, 5],
                    'batches_shapes': [list(b['features'].shape) for b in batches],
                    'features_dtype': str(features.dtype), 'hf_text_batch_type': type(hf_batch['text']).__name__,
                    'range_single_worker': single, 'range_two_workers': multiple,
                    'actual_two_workers_no_duplicates': True},
        'model_weights_downloaded': False, 'LM_training': False, 'large_corpus_benchmark': False,
    }
    # 返回不含Dataset/生成器对象的普通dict，释放映射与闭包引用以便临时目录清理。
    del normal, mapped, cached, disk, loaded, table, stream, lazy, batched, tokenized, hf_batch
    gc.collect()
    return result
