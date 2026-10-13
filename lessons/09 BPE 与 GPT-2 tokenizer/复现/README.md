# 第 09 讲复现记录

状态：**部分成功**。三份原notebook/一张原配图、全部72格中文说明、52个代码格逐项核对、六份派生notebook、三份Python阅读版已交付；真实小语料BPE训练和保存重载通过。原课程10/20万代码数据、讲师Hub发布及语言模型训练未运行。

## 本次环境与执行范围

Windows11、Python3.12.14、Transformers4.44.2、tokenizers0.19.1、datasets2.19.2、pyarrow16.1.0、sentencepiece0.2.0、tqdm4.70.1、IPython8.37.0、nbformat5.10.4；requirements锁定这些CPU/tokenizer验证依赖。没有下载模型权重、远程Wikitext或课程codeparrot。仅下载固定GPT-2/T5的配置与tokenizer资产，留在本地缓存，不额外把完整资产快照推入阅读仓库。

| notebook | 原语句执行 | 独立补充与排除 |
|---|---|---|
| tokenizer_basics，26代码格 | 21个完整原格（含空格38）；22实际编译得到预期SyntaxError | 5/10/31只改为本地固定资产路径+local_files_only，逆变换AST核对；21只执行原生成器定义，在真实小Dataset上运行，跳过远程load_dataset |
| gpt2_train_tokenizer，21代码格 | 1、3、6、7、9、11–14、16–18、21、23、26共15格；其中3/7和后续检查使用明确小语料上下文 | 4改本地资产；8仅改vocab12500→320；20只改100000→500（length表达式仍乘2）和32768→336。原2/24/25数据、代理、Hub跳过 |
| gpt2辅助，5代码格 | 2真实导入；3全注释格真实执行且不改环境 | 1/4 Linux shell不执行，仅IPython转换AST核对；5原缺失路径不执行，另用临时JSONL/json builder检查cache_dir |
| basics Markdown2 | 原/注释设置块在独立GPT2对象真实执行，最终padding_side=right、pad=eos | 不拿这一设置证明普遍修复FP16溢出 |

所有原加载API和训练API均使用真实库；没有mock from_pretrained、load_dataset、trainer或上传返回值。加载路径、训练常量替换明确逐项列出并逆变换AST核对，不能称为整份原环境运行。边界小例子的tqdm仅替换为identity静默显示，主训练使用真实tqdm。

## 本次实测

- 三份原notebook与固定上游提交的下载字节SHA256相等；basics39格、train27格、辅助6格共72格说明齐全，Markdown2设置块详注。
- 原+派生共9份notebook schema通过，全部原元数据/历史输出保留。52代码格中51格可解析AST等价（包括两条IPython shell转换），原basics22错误按同一SyntaxError消息保留；不把无法解析的原格写成AST通过。三份阅读版的可解析代码AST核对；错误22整格为注释，单元9多行str用同值相邻字面量排版以保留空白，shell显式转换为get_ipython调用。
- 固定GPT-2原词表50257；真实T5分词、GPT2 IDs/tokens/预切分offset、normalizer=None、排序与特殊token查询执行。欧元UTF8为[226,130,172]，不是原注释4bytes；byte32/10/13分别映射Ġ/Ċ/č，256项可逆。
- 中文、欧元、emoji、tab、换行、CRLF、四空格缩进和不换行空格的整体编码/解码往返成功。emoji单ID解码出现�，整体解码正确；单个byte/token表示不能替代整体Unicode解码。
- basics36按字符串长度排序没有二级键，来自词表字典的并列次序可变，实际两次前10项可不同。保留双方原输出，核对长度序列`[128,66,65,64,64,64,64,64,56,48]`与逐项转换契约；不改原sort，也不宣称这格原列表完全一致。其余受控输出/训练结果一致。
- 临时JSONL固定2000行自建代码，真实load_dataset(json,...,streaming=True)返回IterableDataset。原batch_iterator默认batch1000和函数体不改，第一轮读0–999、目标320得到320词/63merges；第二轮继续读1000–1999、目标336得到336词/79merges。旧GPT2对象与第一轮新对象均未被第二轮就地修改。特殊token新ID为0，不能沿用原50256。
- 两个新tokenizer分别真实save_pretrained、AutoTokenizer本地重载，sample的IDs/完整decode完全一致。没有训练embedding或测试代码生成能力。
- 原生成器length5/batch2请求6条；7行数据、length8/batch3时已交出前两批，剩余1行在构造不满批时被消费后丢失，真实报`generator raised StopIteration`。修正版保留末批、不超出limit。原Wikitext生成器在2051行真实Dataset上返回1000/1000/51三批。
- 原22实际失败；仅删除一个悬空逗号的独立补充可建立WordPiece对象，仍未配置完整trainer。原Linux路径/代理/Hub未执行或伪造成功。

细节和两版执行记录见 [runtime_verification.json](outputs/runtime_verification.json)。前一次验证临时JSONL在Windows清理时遇到流句柄占用，现显式关闭底层生成器并回收exec函数globals引用环，清理通过；排序并列造成的原输出差异按契约验证而非掩盖。

## 重做本次验证

使用独立Python3.12环境，在本讲`复现`目录执行：

```powershell
python -m pip install -r requirements.txt
python scripts/download_original.py --tokenizer-cache 'G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/验证tokenizer缓存'
python scripts/build_annotated.py
python scripts/verify_annotations.py
python scripts/verify_runtime.py --tokenizer-cache 'G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/验证tokenizer缓存'
```

`--tokenizer-cache`可换为自己的绝对目录，但下载和runtime须指向同一位置。下载仅取固定原notebook/PNG及两套tokenizer配置，不取权重；每个资产的revision/大小/SHA记在source_download，runtime重新核对。GPT2 revision=`607a30d783dfa663caf39e06633721c8d4cfcd7e`，T5 revision=`a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`。临时JSONL和保存文件由验证脚本创建并清理；不改真正数据/代理，不发布Hub。

实际运行解释器：`G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe`，工作目录`G:/zzz/personal-chatgpt-notes`，传入上述脚本在第09讲内的路径。runtime临时切换到`代码/tutorials`读取原配图，结束恢复cwd；Jupyter建议也从该目录打开。Python阅读版不是整份原环境可运行承诺，尤其要注意原缺失路径和上传格。

## 证据

- [原文件与固定tokenizer资产记录](outputs/source_download.json)
- [全部单元/AST/错误/schema审计](outputs/annotation_verification.json)
- [实际运行、语料哈希及比较](outputs/runtime_verification.json)
- [早期16片段教学结果](outputs/教学验证.json)：以下旧环境记录保留。

## 早期独立实验（历史记录）

环境：Windows、Python 3.7.6、Transformers 4.30.2、tokenizers 0.13.3。脚本使用 `USE_TORCH=0` 与 `USE_TF=0`，仅影响当前进程。首次失败为 `AttributeError: module 'torch' has no attribute 'Tensor'`，出现在 `decode → to_py_obj → is_torch_tensor`；本地 torch 是缺少 API 的命名空间，tokenizer-only 路径禁用框架检测后通过。没有修改全局安装。

在本讲目录执行，首次下载仅 tokenizer/config 资产：

```powershell
python 复现/scripts/verify_tokenizer.py --download-base
```

已经有资产时：

```powershell
python 复现/scripts/verify_tokenizer.py --base 资料/gpt2-tokenizer
```

固定 GPT-2 revision 为 `607a30d783dfa663caf39e06633721c8d4cfcd7e`。只下载 tokenizer.json、tokenizer_config.json、config.json，没有模型权重。16 条自建片段，目标词表 320，实际 320；63 条 merges。检查 byte 映射、UTF-8 整体往返、旧 tokenizer 未改变、临时目录保存/重载、生成器消费位置/耗尽/不足一批与 weighted pair 计数。

输出为 `outputs/教学验证.json`；具体训练的 ID 排布不作跨平台保证，报告保存本次观察。课程 `./codeparrot/` 数据与 datasets 依赖未就绪，未直接执行原 notebook，也未执行 `push_to_hub` 或讲师代理设置。
