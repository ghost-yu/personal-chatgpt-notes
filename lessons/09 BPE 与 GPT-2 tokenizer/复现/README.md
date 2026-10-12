# 第 09 讲复现记录

状态：部分成功。实际训练小语料 GPT-2 fast tokenizer，未运行原 10/20 万文件训练与语言模型。

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
