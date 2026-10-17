# 第05讲实际验证记录

[本讲入口](../README.md) · [中文讲解](../解释.md) · [详细注释代码](../代码/tutorials/代码注释.ipynb)

**状态：部分成功。** 全部原单元有解释；实际执行原版和注释版的8个代码格，各7格成功、单元18按CPU环境预期失败。真实CPU数值验证成功，不代表CUDA/TPU运算或性能验证。

| 验证 | 实际观察 | 证据 |
|---|---|---|
| 原文件与两张图 | 固定提交下载，三个SHA256均匹配，原字节保留 | [下载记录](outputs/source_download.json) |
| 全部27单元 | 8代码格、19Markdown格，全部有说明；两种派生notebook各55格 | [单元说明](../代码单元解释.md) |
| 注释是否改变原代码 | 所有原单元元数据/历史输出保留，8格经IPython转换后的AST相同；阅读版普通Python AST相同 | [静态核对](outputs/annotation_verification.json) |
| 单元0、5、26 | 真实导入Image并生成两张PNG的展示数据；图字节哈希与原文件一致，width=400/默认 | [执行记录](outputs/runtime_verification.json) |
| 单元16、17、19 | CUDA BF16查询False；dtype为torch.bfloat16；HIP版本None | 同上 |
| 单元18 | 内层current_device报`AssertionError: Torch not compiled with CUDA enabled`，外层不执行 | 同上；记录原表达式实际失败 |
| 单元20双问号 | 真正经IPython pinfo2显示当前函数源码；展示hook改为捕获文本，避免CLI分页等待，不改原代码 | 同上，函数源码指纹与版本 |
| 原版与注释版 | 每个代码格的结果、图片、预期错误相同 | 同上，两份逐格记录 |
| 真实CPU tensor | int16位模式重解释、finfo、nextafter、正负0、Inf/NaN、舍入、溢出、下溢与相同shape存储通过 | 同上；补充自编，不是原notebook训练 |

## 数值结果怎样理解

| 项目 | FP16 | BF16 |
|---|---:|---:|
| 最大有限数 | 65504 | 3.3895313892515355e38 |
| 最小正正常数 | 6.103515625e−5 | 1.1754943508222875e−38 |
| CPU nextafter(0,1)最小正次正规 | 5.960464477539063e−8 | 9.183549615799121e−41 |
| 1以上首个间距 | 0.0009765625 | 0.0078125 |
| 1.001转换结果 | 1.0009765625 | 1.0 |
| 100000转换结果 | Inf | 99840 |
| shape[7]仅数据存储 | 14 byte | 14 byte |

同shape FP32数据占28 byte。`finfo.tiny`是最小正常数，不能用它冒充最小次正规。位串`0 11101 1010000000`通过解码和真实FP16位重解释均得到26624。

数据流为`int16[7] → view(dtype)[7] → float64[7]`，比较理论解码；`float64[2] → float16/bfloat16[2]`比较转换。`view(dtype)`重解释位，`.to(dtype)`转换数值，两者不可混用。FP16/BF16各测七类模式：+0、−0、最小次正规、最小正常、最大有限、+Inf、NaN。`signbit`确认负零；`isnan`确认NaN，不用`==`比较NaN。

补充测试还确认两个中点的最近偶数舍入：FP16得到1与1.001953125；BF16得到1与1.015625。最小正次正规的一半舍入到0。这里是当前CPU转换行为，不能推广成所有硬件/算子的下溢规则；[Google TPU文档](https://docs.cloud.google.com/tpu/docs/bfloat16)说明TPU转换会把BF16次正规数flush to zero。

## 环境与重做命令

实际环境为Windows11、Python3.12.14、PyTorch2.5.1+cpu、IPython8.37.0、nbformat5.10.4。原notebook元数据只有Python3.10.13，没有完整依赖锁定；历史GPU设备输出不作为本次结果。无需权重、数据集、密钥，也不下载模型。

从仓库根目录执行，`python`应指向上述验证环境：

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ipython==8.37.0 nbformat==5.10.4
python "lessons/05 FP16 与 BF16/复现/scripts/download_original.py"
python "lessons/05 FP16 与 BF16/复现/scripts/build_annotated.py"
python "lessons/05 FP16 与 BF16/复现/scripts/verify_annotations.py"
python "lessons/05 FP16 与 BF16/复现/scripts/verify_runtime.py"
python "工具/check_documents.py"
```

第一步指定CPU wheel索引，与本次的预期失败路径一致；不用在已匹配的环境重装。下载脚本固定上游SHA，校验字节，只有这一脚本需要网络。生成脚本按原编号插入说明并只向代码前加注释。静态脚本核对schema、AST、历史输出和源图。执行脚本使用真实IPython和PyTorch；遇到单元18的已知CPU失败后记录并继续，以覆盖后续格。全本notebook直接运行通常会在这一格停止，本文不隐瞒这点。

CLI中源码检查最初触发Windows外部分页器错误与等待；已停止该次验证进程，并改为IPython的文本展示hook，再完整重跑。该hook只决定展示方式，`??`→pinfo2→源码检查依然真实执行，没有模拟GPU或替换函数。

PyTorch当前函数实现对应[官方v2.5.1源码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/__init__.py)，函数签名为`including_emulation=True`，CPU不可用路径提前False。本次源码SHA256为`da412742e5d0acd01830f65c619d61bd573c64062f689f9a96d7adc445e90f62`；不宣称是老师旧版本实现。

## 原文件与旧记录

原notebook SHA256为`96346db1e5fde82fb4f66e74e2ba58a157236275dbbf8033665aef777dd2a712`；图片哈希、URL及字节数见[来源清单](../来源清单.json)。不改变上游源码或把旧输出清空。保留此前的[教学算术记录](outputs/教学验证.json)，本次新的真实执行证据单独记录。

未运行CUDA/ROCm/TPU BF16算术、AMP训练、吞吐、显存或模型效果实验；不知道老师的完整软件版本。没有完整论文模型可复现，主文和单元说明已明确公式对应的代码位置与缺失边界。


## 2026-10-17：基础说明、原图与正文算例

本轮从当前解释.md原样提取两块Python代码执行：标量解码6.5与26624；CPU float64[3]分别转FP16/BF16/FP32并转回float64读出舍入结果。核对1.001、100000和2^-25的舍入/溢出/下溢，nextafter测1以上首个间距，同shape数据大小为6/6/12byte。具体输出见[本轮验证](outputs/基础改写验证.json)，记录中Inf用字符串保存以保持严格JSON格式，正文直接提供可读表格。

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' './lessons/05 FP16 与 BF16/复现/scripts/verify_foundation_examples.py'
& 'C:/Users/hgfhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' './lessons/05 FP16 与 BF16/复现/scripts/extract_paper_images.py' --pdf 'G:/zzz/Personal ChatGPT 前七讲/05 FP16 与 BF16/资料/BFLOAT16_1905.12322v3.pdf'
```

使用既有Python3.12.14、PyTorch2.5.1+cpu环境，不安装新依赖、不下载模型。原runtime_verification的全格执行与位模式验证没有重跑。没有GPU/TPU算术、AMP训练、显存、速度或质量实验。

课程两张PNG直接使用原文件，本轮实际查看并按来源清单核对SHA256。补充论文为A Study of BFLOAT16 for Deep Learning Training，arXiv:1905.12322v3；仅将格式表作为补充依据，未扩展为该论文全部训练实验。原PDF第4页Table1直接渲染裁剪，已实际查看，表头、数据行与原表N/A完整保留；正文区分表中范围、累加精度和跨硬件次正规数限制。来源/页码/坐标/哈希见[截图记录](outputs/论文截图来源.json)。原PDF留本地，公开仓库包含内嵌PNG；其他机器可从记录中的官方URL取得同版本PDF后修改--pdf路径。

原notebook、两种派生notebook、Python注释代码和历史输出均未修改。本轮新增CPU例子是讲解与核对格式规则，不能升级为原课程GPU支持或BFLOAT16论文训练结果复现。
