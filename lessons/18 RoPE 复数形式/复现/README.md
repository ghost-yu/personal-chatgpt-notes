# 第18讲复现记录

状态：部分成功。两份原notebook及对应注释版全部35代码格、完整未修改官方频率函数、原配图两函数转录、数学及自动求导验证通过；没有完整模型或论文实验。

## 环境与材料

Windows11，Python3.12.14、Torch2.5.1+cpu、NumPy1.26.4、matplotlib3.9.4、nbformat5.10.4；完整官方文件导入还使用FairScale0.4.13。CUDA不可用，GPU没有参与。物理RAM16849256448bytes，运行后G盘可用27515985920bytes；最大新增表[4096,64] complex64约2MiB，其他输入为小数组。无训练集、API key、Llama检查点或权重下载需求。

固定课程提交7fbc29824171fd07dc2c30fe922e61ebc018ad1d，主原49974bytes、总览原264849bytes；两个上游哈希和旧本地CRLF哈希分别在来源清单记录。原PNG182819bytes按固定blob保留，两个原notebook的相对图片路径不改。原论文PDF、完整Meta源码与外部依赖保留本地。

Meta model.py路径：`G:/zzz/Personal ChatGPT 系列/18 RoPE 复数形式/资料/llama-model.py`，固定提交7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e，SHA56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754。新脚本加载整个未修改文件，无需实例化Transformer或初始化模型并行组；实际precompute不涉及通信。FairScale从第15讲`资料/fairscale-local`加入sys.path，不改变共享Torch版本。原Meta内容适用Llama2许可。

## 实际命令

工作目录 `G:/zzz/personal-chatgpt-notes`：

```powershell
$lessonPython = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
& $lessonPython 'lessons/18 RoPE 复数形式/复现/scripts/download_original.py'
& $lessonPython 'lessons/18 RoPE 复数形式/复现/scripts/build_annotated.py'
& $lessonPython 'lessons/18 RoPE 复数形式/复现/scripts/verify_delivery.py'
& 'D:/anaconda/python.exe' '工具/check_documents.py'
```

下载脚本读取本地上游Git固定blob并校验哈希，不经过文本换行处理；builder添加中文注释和逐格说明；verify执行真实Tensor运算，先切换到代码/tutorials工作目录以读取原图，最后恢复cwd。Agg绘图无需GUI，代码没有随机输入。

在其他机器可用Python3.12和[requirements](requirements.txt)建立同版本环境，准备相同官方源码及上游固定Git版本后调整脚本明确列出的本机路径；运行前仍校验字节。不要移动当前三个本地材料目录。旧[环境快照](outputs/environment-lock.txt)与新[环境快照](outputs/环境快照.txt)分别保留。

## 新验证结果

[完整交付验证](outputs/完整交付验证.json)包含实际版本、逐格新输出摘要、来源图与脚本哈希。全部58格说明/35代码详注/Markdown21函数、两个原版+四个59格派生版共六schema、comment-only token/AST、原元数据/执行计数/历史输出及两Python阅读版通过。

主原/注释各16代码格，总览原/注释各19格，所有对应trace精确一致。原单位圆单元5已保存[新图](outputs/unit_circle.png)并视觉检查；配图Image真实读取原PNG字节。跨实现路径的outer与broadcast位置表差2.98e-8；传统实数与复数表重建差5.96e-8，用rtol1e-6/atol1e-7，不能要求bit相同。

完整官方precompute与两原notebook频率结果精确相同；默认ModelArgs配置D4096/H32/max_seq_len2048得到[4096,64]复数表。start_pos7/长度3切片正确，重置位置使因子最大差0.701566。complex[10,3]和real视图[10,3,2]各240bytes，view共享存储，flip另分配。

总览原end5错误频率表与正确表差0.185397；旧报告统一end10的0.414712是另一范围。float64二维块矩阵与复数旋转一致；小型q/k的相对点积和共同位移误差1.11e-15，梯度解析式误差2.22e-16，polar角度梯度误差0并通过complex gradcheck。共轭和相位符号错误均实际改变点积。

主26原全局d_model缺陷：实例d4/全局6与正确表差0.315820，修改全局12再次改变约1.27394，删除全局真实NameError。单独修正版只在forward替换一处Name为self.d_model，d2/4/6/8与正确表均差5.96e-8以内，未改公开原/注释单元。奇数d5实数类真实stack错误；原复数类输出[10,6]，官方precompute返回[10,2]，不代表完整奇数维支持。同名重定义不改变旧spe实例，两个类均0参数。

原配图为代码截图，人工对照转录rotate_half和apply_rotary_pos_emb并加详注；具体HF版本未找到。两函数实际用q[2,2,3,6]、k[2,1,3,6]及不同batch位置执行；半区/相邻通道通过排列[0,2,4,1,3,5]及其逆保持一致时，与复数旋转精确一致，直接混用差1.59950。没有执行HF整模型或Meta apply_rotary_emb。

## 验证中遇到的错误与早期记录

补充AST修正版的初稿匹配到了构造参数与forward两处d_model，未满足“一处替换”断言而停止；限定到forward后再验证通过。初稿还要求outer/broadcast两路径bit相等，真实差2.98e-8；改为独立记录误差并使用明确float32容差。原版对注释版的精确一致要求始终保留。

早期[verify.py](scripts/verify.py)、[教学验证](outputs/教学验证.json)与[stdout](outputs/notebook_stdout.txt)保留，当时只执行主notebook并AST提取官方函数，未导入完整Meta或执行总览。若重做旧验证，可使用新公开原notebook：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/18 RoPE 复数形式/复现/scripts/verify.py' --notebook 'G:/zzz/personal-chatgpt-notes/lessons/18 RoPE 复数形式/代码/tutorials/llama2_src_03_RoPE_01_complex_position_encoding.ipynb' --official-model 'G:/zzz/Personal ChatGPT 系列/18 RoPE 复数形式/资料/llama-model.py'
```

新清单sha字段为上游LF版本，legacy字段记录旧CRLF字节；旧脚本的允许集合读取清单，故这条命令使用公开LF原件。不要把旧报告覆盖为新完整验证，也不把历史输出当成本次结果。

## 未运行范围

未加载原Llama检查点、执行整模型/Attention/Meta apply_rotary_emb、GPU/NCCL/多rank、低精度模型输出、RoFormer训练或任何长上下文质量和速度指标。2MiB表可生成不代表两倍上下文能力；小型旋转验证不是整模型推理复现。当前E:/根目录30个txt中仍未找到独立本讲字幕。
