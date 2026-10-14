# 第 15 讲复现记录

## 新增完整交付与真实模型验证

**部分成功**：主12格/3代码、补充16格/9代码，全部28格说明与代码/Markdown详注；两原件、两原PNG、四派生版、两Python阅读版。新增运行使用完整固定官方源码、真实FairScale0.4.13单进程Gloo和随机小型Transformer，没有替代generate或模型API。早期fixture检查在本文后半保留，不能当成模型运行。

环境为Windows/Python3.11.0、torch2.0.1+cpu、SentencePiece0.2.0、FairScale0.4.13、nbformat5.10.4、IPython8.37.0；实际复用第14讲隔离环境，将FairScale用 `--no-deps --target` 安装到第15讲本地材料目录，没有改变旧环境的已安装包。完整包列表见[环境快照](outputs/environment_freeze.txt)。

官方参考revision为7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e。verifier要求四个对应源码hash：

| 本地名字 | 官方路径 | SHA256 |
|---|---|---|
| llama-generation.py | llama/generation.py | acc739aeda4aa222305ca5e7ae020df38d0fa841530d966f041c13e065db05c4 |
| llama-model.py | llama/model.py | 56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754 |
| llama-tokenizer.py | llama/tokenizer.py | a675762526cd5f6e9d742a642e33a3560ea825522b8a9efb531d36b7c7a69e6c |
| example_chat_completion.py | example_chat_completion.py | 63fafc6376b66e2312b80791ddbe68f4c81c37fb7878bc07a7c71619fffa2437 |

tokenizer.model仍为课程固定文件，SHA256 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347；需取得已有对应词表。原官方源码保留版权/许可，完整快照和字幕留在本地。

从仓库根目录重做：

```powershell
py -3.11 -m venv "$env:TEMP/lesson15-env"
$lessonPython = "$env:TEMP/lesson15-env/Scripts/python.exe"
& $lessonPython -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.0.1+cpu"
& $lessonPython -m pip install -r "lessons/15 Llama 2 text 与 chat completion/复现/requirements.txt"
$lessonSources = "$env:TEMP/lesson15-sources"
New-Item -ItemType Directory -Path $lessonSources -Force | Out-Null
$lessonRevision = '7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e'
$lessonFiles = @{'llama-generation.py'='llama/generation.py'; 'llama-model.py'='llama/model.py'; 'llama-tokenizer.py'='llama/tokenizer.py'; 'example_chat_completion.py'='example_chat_completion.py'}
foreach ($lessonFile in $lessonFiles.GetEnumerator()) {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/meta-llama/llama/$lessonRevision/$($lessonFile.Value)" -OutFile "$lessonSources/$($lessonFile.Key)"
}
# 指向对应tokenizer.model，本例路径需要替换为自己已有的词表。
$lessonTokenizer = 'path/to/llama2_tokenizer.model'
& $lessonPython "lessons/15 Llama 2 text 与 chat completion/复现/scripts/download_original.py"
& $lessonPython "lessons/15 Llama 2 text 与 chat completion/复现/scripts/build_annotated.py"
& $lessonPython "lessons/15 Llama 2 text 与 chat completion/复现/scripts/verify_delivery.py" --sources $lessonSources --tokenizer $lessonTokenizer
python "工具/check_documents.py"
```

源码取回代码是重做方式，不声称本次重新下载官方文件。本机原notebook/图片由download_original.py的 `--upstream-clone` 读取已下载固定Git blob。实际完整验证命令：

```powershell
& "G:/zzz/Personal ChatGPT 系列/共享运行环境/llama2-sft/Scripts/python.exe" "lessons/15 Llama 2 text 与 chat completion/复现/scripts/verify_delivery.py" --sources "G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料" --tokenizer "G:/zzz/Llama 入门教程/代码/personal_chatgpt/tutorials/llama/llama2_tokenizer.model" --fairscale-path "G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local"
```

[原件来源](outputs/source_download.json)、[注释审计](outputs/annotation_verification.json)、[真实模型报告](outputs/runtime_verification.json)分别记录来源字节、6schema/12代码AST/历史输出/Markdown仅增加注释和四命令语义、实际模型行为。

原模型只改两个cache的.cuda()→.to('cpu')，generate只改三个device='cuda'→'cpu'；模型dim16/1层/2heads/vocab32000/multiple_of8、序列512、batch8，FP32参数1,027,376。没有checkpoint时显式seed42随机初始化矩阵，norm scale置1，不使用官方lambda x:x留下的未初始化权重。没有运行build/NCCL/torchrun或加载权重。

两版原主3格Image展示和补充9格真实SP编码结果一致；标签ID/换行和图像bytes核对。真实text不等长prompt、chat带历史/system、top-p采样、echo和六dialog批量通过；batch4拒绝、8接受。实际role错误保持IndexError/AssertionError；保留标签仍先跑模型后替换输出。受控真实参数强制EOS为最高分，一次forward结束并切除EOS。

真实generate还复核旧版本logprobs问题：三步新增值为-0，而实际logits的greedy log概率约-10.06455/-10.06923/-10.05871。原因是CE先于采样写入，目标仍pad=-1，被忽略；未修正原算法，也不将0当成置信度。原7B/13B、CUDA/FP16/MP2+、真实加载性能及质量benchmark未运行。

## 早期包装层fixture验证（保留原范围）

**部分成功**：从官方固定源码用 AST 提取 Tokenizer.encode/decode、Llama.__init__/text_completion/chat_completion 和常量，直接执行官方方法。真实 SentencePiece 0.2.0 编码；generate 用常量答案 fixture 替代。它只验证包装层，不执行 torch/CUDA/Transformer，也不生成真实答案。

Python 3.12.14，位于本地共享隔离环境；默认 Python 的 torch 缺少 Tensor API。本机8 GiB单卡没有原始7B FP16权重。原实现build使用NCCL与FairScale，不在当前Windows环境尝试启动。没有下载模型或执行签名URL脚本。

工作目录为本目录，实际命令：

```powershell
& "G:/zzz/Personal ChatGPT 系列/共享运行环境/datasets/Scripts/python.exe" scripts/verify_completion.py --sources "G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料" --tokenizer "G:/zzz/Llama 入门教程/代码/personal_chatgpt/tutorials/llama/llama2_tokenizer.model" --output outputs/教学验证.json
```

`--sources` 目录需要 `llama-generation.py`、`llama-tokenizer.py`、`example_chat_completion.py`，取自 Meta 仓库 commit `7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e`，下载后按上述名字保存；`--tokenizer` 是已有模型词表文件。中文文件路径通过pathlib读取bytes传model_proto，避开native文件名问题。

检查内容：BOS/EOS边界、system合并、多轮角色、原dialog未修改、非法角色断言、空/只有system的IndexError、特殊标签错误替换、6个官方dialog与batch4/8的差异。fixture里logprobs是合成-0.1，不能当模型置信度；输出切片为整数例子，未执行真实generate循环。

真正推理需对应权限下的原始 `.pth` 分片、params.json、tokenizer、兼容NCCL/PyTorch/FairScale的设备与环境，MP和分片数相同，再验证真实生成、EOS和缓存。作者精确commit及全部依赖未找到，本次固定源码只是明确的参考版本。
