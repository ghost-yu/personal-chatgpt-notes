# 第 10 讲复现说明

状态：**部分成功**。原 notebook 的 6 个代码格与注释版逐格真实执行一致；全部 13 格说明、AST/schema/原历史输出、4 张原图 hash 核对通过。独立教学补充实际运行 JSONL、HF map/streaming、Arrow/mmap、固定 GPT-2 分词与 padding、PyTorch DataLoader（包括 Windows 2 worker）。未复现课程约 20 GB 数据、C4/Pile/T5 全量训练或大数据 RSS/吞吐。

## 重做命令

下面从本讲目录执行，使用 Python 3.12。分词资产放在仓库外，只有配置/词表，没有模型权重；已存在且 SHA 相同则复用。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r 复现/requirements.txt
$taskTokenizerDir = Join-Path $env:LOCALAPPDATA 'personal-chatgpt-notes/gpt2-607a30d'
.venv/Scripts/python.exe 复现/scripts/download_original.py --tokenizer-dir "$taskTokenizerDir"
.venv/Scripts/python.exe 复现/scripts/build_annotated.py
.venv/Scripts/python.exe 复现/scripts/verify_delivery.py --tokenizer-dir "$taskTokenizerDir"
```

- `download_original.py`：下载固定上游原 notebook 和 4 图，核对 SHA。可选 `--tokenizer-dir` 另外准备 GPT-2 固定 revision 的 3 个资产文件，不下载大语料/权重；原内容变化会使校验失败。
- `build_annotated.py`：保留 13 个原单元，生成各 27 格的说明/注释 notebook、Python 阅读版与单元索引。注释版只插入代码注释。
- `verify_delivery.py`：先核对结构、原单元/历史输出与 AST，再分别执行原/注释 6 格并核对图片对象。随后真实调用 `代码/数据读取_教学补充.py`，验证数据接口与下游批次；临时数据随实验退出清理。
- `--tokenizer-dir` 是分词资产目录，验证时检查 3 个固定 hash，并仅本地加载。复用第 09 讲已有固定 GPT-2 资产亦可。

## 本次实际环境与命令

使用已有共享 CPU 环境，没有安装新环境或改动 Python 3.7。实测 Python 3.12.14、torch 2.5.1+cpu、datasets 2.19.2、pyarrow 16.1.0、numpy 1.26.4、pandas 2.2.2、huggingface-hub 0.36.2、fsspec 2024.3.1、transformers 4.44.2、tokenizers 0.19.1、IPython 8.37.0、nbformat 5.10.4、Pillow 12.3.0。

从仓库根目录运行：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/10 预训练语料、memory mapping 与 streaming/复现/scripts/verify_delivery.py' --tokenizer-dir 'G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/验证tokenizer缓存/gpt2'
```

GPT-2 资产 revision：`607a30d783dfa663caf39e06633721c8d4cfcd7e`。实际运行返回 exit 0，原/注释 6 格结果一致，真实 Windows 2 worker 完成且临时目录清理。

## 可核查结果

| 记录 | 范围 |
|---|---|
| [source_download.json](outputs/source_download.json) | 原 notebook/4 图下载 URL、原字节 SHA、必要分词资产 |
| [annotation_verification.json](outputs/annotation_verification.json) | 13 格覆盖、6 代码 AST、3 schema、原输出/元数据、Python 阅读版 |
| [runtime_verification.json](outputs/runtime_verification.json) | 两版各 6 格、实际图片数据、教学补充输出和环境/分词 hash |

补充 5 行文本是 `hello world`、`中文文本`、`a b c`、`stream me`、`last row`。LF JSONL 215 bytes、Arrow data 145 bytes、IPC 文件 648 bytes，均非 RSS；不是格式压缩比 benchmark。map/skip 次序的回调分别 `[0,1,2,3]` 和 `[3]`，显式缓存命中新增 0 次。stream map 分词 batch2，变长 list 另经 padding 得到 int64 `[5,7]`；按 mask decode 全部恢复。100×5 float32 特征得到 32/32/32/4 行四批次；range3–7 的实际两 worker 分片无重复。下游模型、loss、optimizer 和多 GPU 未运行。

## 早期记录

[verify_datasets.py](scripts/verify_datasets.py)、[教学验证.json](outputs/教学验证.json) 与 [requirements-lock.txt](requirements-lock.txt) 保留早期 HF/Arrow-only 实验，当时 hub 为 0.23.4 且 `pip check` 通过；这份旧锁不描述本次 CPU/tokenizer 环境。旧 JSONL 220 bytes 使用 CRLF，最新明确 LF，每行少 1 byte。早期实验没有 PyTorch/tokenizer，最新范围以上述新记录为准。
