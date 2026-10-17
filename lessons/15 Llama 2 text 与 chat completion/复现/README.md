# 第 15 讲复现记录

## 2026-10-17：基础讲解与内嵌原图

本节是新一轮正文验证，下方旧的28格完整交付、H16真实模型和早期fixture记录均保留且未重跑。使用 [verify_beginner_note.py](scripts/verify_beginner_note.py) 从当前 `解释.md` 折叠栏前提取四个Python块，按顺序原样执行，不用另一套实现代替正文代码。

环境沿用Python3.11.0、Torch2.0.1+cpu、SentencePiece0.2.0；FairScale0.4.13由现有本地target目录提供。脚本核对固定官方generation/model/tokenizer源码及词表hash，使用真实单进程Gloo/FairScale模型并行MP1。模型两处cache的 `.cuda()` 改CPU、generation三处显式CUDA设备改CPU，AST计数有断言；透明记录器保存参数和真实forward，不提供假答案。没有调用原 `Llama.build` 或更改共享环境。

从仓库根目录运行，材料获取方式及hash见下方旧记录，路径可替换但文件内容必须匹配：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/llama2-sft/Scripts/python.exe' 'lessons/15 Llama 2 text 与 chat completion/复现/scripts/verify_beginner_note.py' --sources 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料' --tokenizer 'G:/zzz/Llama 入门教程/代码/personal_chatgpt/tutorials/llama/llama2_tokenizer.model' --fairscale-path 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local'
```

脚本先将tokenizer复制到ASCII临时路径避开Windows已知路径限制，加载完整官方类，显式seed15初始化随机H8/1层/2头/V32000/FP32模型，所有矩阵normal(0,.02)、归一化权重1，合计512856参数。官方模型原始初始化等待checkpoint填充，不能直接拿未初始化内存生成。模型缓存容量为batch4/序列128，本轮与旧H16/batch8/512实验分开。

四块正文分别验证：字典/列表与真实SP普通3/BOS4/EOS5；官方chat将4消息拼成45token、2BOS/1历史EOS、原调用者未改、真实3步返回；官方text输入长2/4各新3、echo5/7、五次forward起点0/2/3/4/5与长prompt原ID保留；保留标签仍真实运行1次forward再换错误，空/仅system/非法角色/末尾assistant与batch5/长度129六反例均未新增forward。全模型状态前后精确相同，无训练或优化更新。

[本轮结果JSON](outputs/基础讲解验证.json) 记录当前笔记/代码块/脚本/原件/材料hash、版本、真实ID、logits形状、调用轨迹和命令。随机生成内容只验证路径，不展示语言能力。未新增采样、受控EOS或logprobs实验；旧结果在下方保留，不当作本轮重跑。

原图由 [extract_paper_image.py](scripts/extract_paper_image.py) 从Llama 2 v2 PDF第16页直接渲染裁剪Figure9，pypdfium2/Pillow使用独立PDF工具环境。原图及两课程PNG均已实际查看；[截图来源](outputs/论文截图来源.json) 记录PDF版本/页码/裁剪框、PNG和课程图hash。重做：

```powershell
& 'C:/Users/hgfhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'lessons/15 Llama 2 text 与 chat completion/复现/scripts/extract_paper_image.py' --pdf 'G:/zzz/Personal ChatGPT 系列/14 Llama 2 introduction 与 Guanaco/原论文.pdf'
```

先实际查看新PNG，再加 `--confirmed-visual-review` 重做视觉标记；本轮已完成检查。未重绘/改字，图中GAtt对话不是本轮随机输出。正文就地解释其训练与推理接口边界。

本轮状态：**部分成功**。无原7B/13B权重、build/NCCL/torchrun/CUDA/FP16/MP2+、训练/GAtt或质量/性能评估。以下全部是此前完整交付及历史验证范围。

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
