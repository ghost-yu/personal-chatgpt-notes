# 第19讲复现记录

## 2026-10-17 基础讲解与原论文截图

正文以Q[1,2,2,4]/K[1,2,1,4]贯穿配对、complex view、[1,T,1,P]广播、复数乘法与flatten，包含完整原apply与官方helper。RoFormer v4 p5 Figure1、p7公式34、p8 Figure2原PDF直接截图，三张均视觉检查。公式34截图末对排版按二维旋转规则就地说明；第8节直接解释论文35–37的内容项、Abel上界及论文/课程端点差异。旧讲解完整折叠保留，原件与注释代码不变。

执行命令（仓库根目录）：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' -X utf8 'lessons/19 RoPE apply_rotary_emb/复现/scripts/verify_foundation_examples.py'
```

[新脚本](scripts/verify_foundation_examples.py)从当前正文第一个折叠前提取五块Python代码，真实顺序执行。Python3.12.14/Torch2.5.1+cpu/NumPy1.26.4，完整官方文件导入仍使用第15讲FairScale本地target。原apply与正文AST相同；官方apply/helper去docstring后AST相同，原s函数AST相同。实际导入完整固定model.py且检查SHA，执行三个RoPE函数。

[新报告](outputs/基础讲解验证.json)记录当前笔记/脚本哈希、stdout和数值：

- Q配对[1,2,2,2,2]→complex[1,2,2,2]，频率view[1,2,1,2]与原表共享存储；手工各步、正文完整函数和官方输出精确一致，输入数值不变。位置1/head0约[-1.142640,1.922076,2.959851,4.029799]。独立float64两分量参考最大差5.54e-7。
- T=H=2时，省略频率reshape得到相同输出shape却错用head轴，最大差7.34731436。位置1/head0错误保留原值。
- 两位置的原始/共同后移7点积均7.725471497；正文与官方该分数的全输入梯度精确一致。另用覆盖全部Q/K分量的线性loss核对独立逆旋转梯度，最大差8.94e-8/5.15e-8。绝对位置3/4切片与独立角度对照通过，错误重置到0/1改变Q约15.15504。
- 四种FP16/BF16/FP32/FP64真实输出保留输入dtype、梯度有限；末stride2真实失败，contiguous整理后通过。单独执行原16格定义并调用时缺helper的NameError被捕获，显式提供同提交官方helper后与官方输出相同；原文件未改。
- 课程相位曲线C0=31.5、C255=6.412619851、113个上升区间，与独立cumsum误差5.33e-15；论文求和范围C0=32.5、C255=6.781111551，端点差恒等式误差6.44e-15。另以64对复数内容、256个距离验证引用的Abel恒等式与上界。
- 原6格R函数的30+60/逆30+60合成验证通过；只在原相对目录执行18/19分词两格，读取原499723byte tokenizer得到历史8个ID。

[截图来源](outputs/论文截图来源.json)记录版本、页码、top-left坐标、3倍渲染和PDF/图片哈希。本次运行正文、官方函数、矩阵/分词子集与数学；旧15格整notebook运行未重跑，完整Attention/cache/原权重/GPU/论文任务留待后续。

状态：部分成功。原版/注释版全部15代码格、真实tokenizer、完整固定Meta模块的三个RoPE函数、独立数学与autograd验证通过；没有运行整模型或论文训练。

## 环境与资源

Windows11，Python3.12.14、Torch2.5.1+cpu、NumPy1.26.4、matplotlib3.9.4、sentencepiece0.2.0、nbformat5.10.4、FairScale0.4.13。CUDA不可用、GPU未参与。物理RAM16849256448bytes，运行后G盘空闲27514073088bytes。最大Q仅[2,4,4,8]，频率[32,4]，原曲线256点；不下载权重、训练数据或使用API key。

固定课程提交7fbc29824171fd07dc2c30fe922e61ebc018ad1d；原notebook43654bytes/SHA59fb6124d4c14dfe40c1cb1053a43458f0b745183447195abab18e3609d94f1e，分词器499723bytes/SHA9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347。旧本地CRLF notebook SHA另以legacy字段保留。

官方源码保留本地 `G:/zzz/Personal ChatGPT 系列/19 RoPE apply_rotary_emb/资料/llama-model.py`，Meta提交7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e，SHA56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754。新脚本导入完整未修改文件；无需构造Transformer、加载权重或初始化模型并行通信组。FairScale从第15讲资料/fairscale-local加入sys.path，不修改共享Torch。原Meta片段适用Llama2许可。

## 本次实际命令

工作目录 `G:/zzz/personal-chatgpt-notes`：

```powershell
$lessonPython = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
& $lessonPython 'lessons/19 RoPE apply_rotary_emb/复现/scripts/download_original.py'
& $lessonPython 'lessons/19 RoPE apply_rotary_emb/复现/scripts/build_delivery.py'
& $lessonPython 'lessons/19 RoPE apply_rotary_emb/复现/scripts/verify_delivery.py'
& 'D:/anaconda/python.exe' '工具/check_documents.py'
```

download读取本地上游固定Git blob，校验后按字节保存。build_delivery保留已有15代码格详细解释，补齐全部Markdown，产生两个55格派生版及Python阅读版；三个官方函数的阅读补充省去英文docstring、仅添加中文注释，执行AST不变。verify真正执行原/注释格，使用Agg后端，并切换到代码/tutorials读取原相对分词文件；最后恢复cwd。原notebook没有随机输入；后续小型旋转输入采用外置Torch seed19。

在其他机器，使用Python3.12及[requirements](requirements.txt)，准备固定上游Git和相同Meta源码，再调整脚本明确列出的本地路径；运行前仍检查哈希。不要移动当前来源/共享环境目录。旧[环境快照](outputs/environment-lock.txt)及新[环境快照](outputs/环境快照.txt)分别保留。

## 新结果

[完整交付验证](outputs/完整交付验证.json)含实际版本、全部原格新返回值摘要、核心调用缺口、负例和脚本哈希。

- 全部27格/15代码详注，含纯注释17和空26；两派生版各55格、三schema、comment-only token/AST、原元数据/执行计数/历史输出及Python阅读版通过。三官方函数阅读版执行AST一致，原apply也与官方执行AST一致。
- 原版/注释版各15格真实执行trace精确相同，SentencePiece ID=[1,306,4658,278,6593,310,2834,338]，两种矩阵合成按1e-15绝对容差通过。原曲线[新PNG](outputs/notebook_phase_curve.png)已经视觉检查。
- 两版完成原格后均因reshape_for_broadcast未定义而真实NameError；仅随后显式补入固定官方helper，未改变原单元。Q[2,4,4,8]/K[2,4,2,8]、位置3..6，原/注释/完整官方输出和梯度精确一致；独立sin/cos前向和逆旋转梯度差0。输入数值不变，广播view[1,4,1,4]共享相位存储。
- FP32/FP16/BF16/FP64实际返回原dtype且反向有限；norm误差约4.77e-7/7.85e-4/6.77e-3/4.77e-7。FP64加入1e-10扰动仍因内部.float()丢失，不能声称全double统计或strict double gradcheck。
- last stride2输入真实RuntimeError，contiguous修复通过；跨head轴但满足复数view stride条件的非连续输入也通过。奇数d7与相位[T3]真实失败。T=H=3时省略广播helper静默错轴，当前随机样本差2.90499；旧报告3.83991来自另一随机样本，不混用。
- K长度1被静默广播输出4位置；三维[2,4,8]输入输出[2,4,4,2]，原函数不提供完整四维/等T输入验证。调用方必须保证正确契约。
- 相对点积差7.47e-7、共同位移差4.77e-7、错误重置位置使Q差7.08538。部分和曲线对独立cumsum最大差5.33e-15，C0=31.5、C255=6.41262，113次局部上升；它不是attention概率。
- 原全局d128与s默认128恰好一致；只将全局改64时，s(0)/(64//2)=63，显式s(0,d=64)/32=15.5。改变宽度应同步传参。

初次低精度反向验证用to(dtype).requires_grad_直接复用基础tensor，FP32的to返回同一对象，后续转换出现非叶节点且.grad=None，验证停止。改为每个case独立detach/clone后再转换及启用grad，四种case全部成功。这是验证输入构造的问题，没有把初次失败当模型验证成功。

## 早期报告

[旧脚本](scripts/verify.py)、[教学验证](outputs/教学验证.json)、[曲线数据](outputs/curve_data.json)、[旧43格覆盖](outputs/code_unit_coverage.json)和旧build_annotated.py保留。旧脚本AST提取三个官方函数并绑定真实Torch；它不是新完整文件导入或注释版执行证据。

重做早期验证时使用公开LF原件及相同tokenizer/官方源码：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/19 RoPE apply_rotary_emb/复现/scripts/verify.py' --notebook 'G:/zzz/personal-chatgpt-notes/lessons/19 RoPE apply_rotary_emb/代码/tutorials/llama2_src_03_RoPE_apply_rotary_emb.ipynb' --official-model 'G:/zzz/Personal ChatGPT 系列/19 RoPE apply_rotary_emb/资料/llama-model.py' --tokenizer 'G:/zzz/personal-chatgpt-notes/lessons/19 RoPE apply_rotary_emb/代码/tutorials/tokenizer.model'
```

旧builder只为15代码格插说明，43格报告是当时结果；本次所有27格插说明的版本必须使用build_delivery。新历史和新验证分别保留，不把旧记录标为当前完整交付。

## 未运行范围

没有Llama神经网络检查点、完整Attention/Transformer、KV cache/repeat_kv、CUDA/NCCL或多rank通信、训练/吞吐/长文本质量/RoFormer全实验。本讲tokenizer并非语言模型权重；频率和旋转小输入通过不证明完整生成。当前根目录30个txt中仍未找到独立第19讲字幕。
