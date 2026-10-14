# 第 17 讲复现记录

状态：部分成功。原notebook全部20代码格真实执行，以及完整未修改官方RMSNorm/FairScale SwiGLU最小前向和反向通过；没有复现原Llama检查点或论文训练/速度。

## 环境与资源

Windows11，Python3.12.14，Torch2.5.1+cpu，NumPy1.26.4，matplotlib3.9.4，nbformat5.10.4，FairScale0.4.13；完整版本见[环境快照](outputs/环境快照.txt)。实际可用CPU Torch，CUDA不可用；物理RAM16849256448bytes，运行后G盘可用27517685760bytes。原课程最大输入1000元素；SwiGLU小验证144参数。4096宽度只用meta张量，不分配大权重。无需训练集、API key或Llama检查点。

官方model.py固定Meta提交7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e，SHA256=56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754，保留本地，遵循原Llama2许可。依赖及源码路径在脚本顶部明确写出，不移动本地材料目录。

## 本次实际命令

工作目录为 `G:/zzz/personal-chatgpt-notes`。下载脚本从本地上游Git仓库读取固定blob，不经过文本重写；builder生成阅读文件；最后一步真正执行Torch/FairScale验证，不只是读取旧报告。

```powershell
$lessonPython = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
& $lessonPython 'lessons/17 RMSNorm 与 SwiGLU/复现/scripts/download_original.py'
& $lessonPython 'lessons/17 RMSNorm 与 SwiGLU/复现/scripts/build_annotated.py'
& $lessonPython 'lessons/17 RMSNorm 与 SwiGLU/复现/scripts/verify_delivery.py'
& 'D:/anaconda/python.exe' '工具/check_documents.py'
```

完整官方源码位于 `G:/zzz/Personal ChatGPT 系列/17 RMSNorm 与 SwiGLU/资料/llama-model.py`；FairScale位于第15讲的 `资料/fairscale-local`，脚本显式加入sys.path。此target安装不改变共享Torch版本。新建环境可使用[requirements](requirements.txt)安装同版本依赖，准备相同官方源码，并调整脚本中本机路径；运行前仍校验源码哈希。不能只安装NumPy就执行新完整验证。

初次用默认Anaconda Python3.7构建新增官方阅读补充时，ast.get_source_segment不存在，构建失败；改用上面的Python3.12.14后成功。该失败没有被当作新版本验证通过的证据。

## 结果与验证范围

[完整交付验证](outputs/完整交付验证.json)保存命令对应的实际环境、全部代码编号、逐格新输出摘要及脚本哈希。原字节35114bytes/SHA a90ee5e811cf42d3a3f3496f0d6449d7def728f497a57525b8bfe6bfd834a61f；旧本地CRLF版本35706bytes/93f2882a…保留为legacy来源，不混淆哈希。

- 全28格说明/20代码格/Markdown3类，两个57格派生版、原版共3schema，comments-only token/AST、阅读版AST与官方完整FeedForward阅读类AST通过。
- 原版与注释版各20代码格，外置Torch/NumPy seed17、Agg后端；实际张量/数组字节摘要、打印、标量及1000点曲线完全一致。只保存新曲线，原notebook历史输出不替换。
- 真实官方RMSNorm与课程类相同输出；FP32输入Jacobian对独立解析式误差8.35e-8、weight梯度8.34e-8。真实FP16/BF16/FP32输入和参数组合均有限；FP32weight导致低精度输入最终输出FP32。float64输入仍在内部float32统计，因此不用strict double gradcheck宣称纯双精度公式验证。
- 零输入eps>0输出0、eps0全部NaN、近零省eps差1.26489、增益2倍、train/eval一致；真实SiLU前向与自动求导通过。
- 真实FairScale0.4.13、Gloo单进程组，未修改完整官方模块。显式normal(std0.2)/seed17初始化所有144参数；[2,3,4]→[2,3,12]→[2,3,4]。NumPy前向差2.43e-17，24项输入梯度中心差分最大6.43e-12，144项全部权重梯度最大1.89e-11。去掉门控确实改变输出约0.380234。finally清理两种进程组。
- meta配置dim4096/multiple256得到hidden11008和135266304参数，未真实分配或执行该大层。

## 早期报告和未运行部分

[旧NumPy脚本](scripts/verify.py)、[旧教学验证](outputs/教学验证.json)保留。早期环境是NumPy float64，官方forward被AST抽取并绑定NumPy fixture，仅核对数学和连接，不构成真实官方类或通信证据。重跑旧记录：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/datasets/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/17 RMSNorm 与 SwiGLU/复现/scripts/verify.py' --official-model 'G:/zzz/Personal ChatGPT 系列/17 RMSNorm 与 SwiGLU/资料/llama-model.py'
```

没有加载原Llama模型、optimizer训练、GPU/NCCL或world_size>1，也没有论文数据集、BLEU/perplexity、模型质量或吞吐评测。CPU低精度算子测试不能证明GPU kernel的精度/性能。本讲SwiGLU源码阅读和实际官方模块测试是教学补充，不伪装成原notebook缺失的实现。
