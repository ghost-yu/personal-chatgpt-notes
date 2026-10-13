# 第 08 讲复现记录

状态：**部分成功**。两个原notebook、三张原图与固定提交SHA256一致；全部59格说明/38代码格AST、6份notebook schema、原元数据/历史输出及两份Python阅读版通过核对。真实CPU执行和整数教学补充通过，原OPT6.7B量化加载、bitsandbytes CUDA内核及论文评测未运行。

## 本次环境与资源边界

Windows11，Python3.12.14，PyTorch2.5.1+cpu，NumPy1.26.4，Transformers4.44.2，tokenizers0.19.1，IPython8.37.0，nbformat5.10.4，Pillow12.3.0。`torch.cuda.is_available()`为False。早期机器记录有RTX4060 Laptop/8188MiB，但本次使用的CPU虚拟环境不能因此执行CUDA。未安装bitsandbytes、未下载OPT权重/tokenizer，不修改真实HTTP/HTTPS代理，不进行C4/零样本评测。只运行固定小矩阵与2960参数的小OPT，未建立6.7B模型。

原课程Python3.10.9，未严格锁定torch/transformers/bitsandbytes。主解释的Transformers4.30.0与bitsandbytes0.39.1属于固定历史源码参照；本次CPU实验使用4.44.2，不能把它们写成同一个环境。旧`教学验证.json`与`静态检查.json`作为早期记录保留。

## 精确执行范围

| 材料 | 真实执行 | 明确跳过/替代 |
|---|---|---|
| 原与注释版LLM.int8 | 各32个完整代码格：0、2、5、7–13、16–17、24–27、29–31、33–34、36、38、40–43、45–46、48–49、55 | 原50代理、51远程tokenizer、52原模型加载未执行；48只有安装注释、55为空 |
| 原53生成/54参数枚举 | 两个版本的原语句在独立CPU小OPT教学上下文执行 | MAX_NEW_TOKENS设8，本地词表/随机模型；原51/52未执行，未伪装from_pretrained |
| 原/注释版bitsandbytes | 各执行原代码格1，真实导入BitsAndBytesConfig | 配置构造/互斥是独立补充，不是原格已有的代码 |
| Markdown | 全部中文说明，原内容保留 | 主22是伪代码，没有当成可执行Python；没有修改原维度笔误 |
| 三张图片 | 原Image调用真实读取PNG字节、Pillow验证、SHA核对；人工查看三图 | 不把图片显示当成量化算法运行 |

主notebook37个代码格含空格55；辅助1个代码格，共38格静态核对。执行覆盖不等于全部38格原环境运行成功。两份说明/注释主notebook各113格，辅助各7格：最前导读一格，随后每个原格前加一格说明。原代码、元数据、执行编号及保存输出保留；注释版只添加注释。

## 实测结果与含义

- 原单向量代码：`round`返回float32，真实MAE有限；固定seed20261013，两版每格数值相同。补充验证ties-to-even、全零原写法产生NaN，以及固定普通元素时异常末项100令小值被舍入为0。
- 原矩阵：X/W为本平台int32；X_i8/W_i8/Out_i32/Out_f16_1实际上float64，Out_f16_2为int32。两路相加得到`[[38,34],[-167,124],[-174,168]]`，原例误差0.0，因为未舍入，不能宣称真实8bit无损。
- 整数补充：按阈值6识别异常特征[1,3]、普通[0,2,4]。真实int8编码、int32累加后恢复普通支路，混合输出最大误差0.01574803149605941；不拆异常最大误差1.0940541881083732。只支持这个固定算例的比较。
- 普通支路编码qx/qw共15个元素、15字节；相同元素放float64需120字节。此比较只算编码数组，不包含尺度、临时int32副本、异常数组等，不能作为模型整体8倍压缩测量。
- NumPy与PyTorch实际FP16异常支路都与本例整数基准一致；本例值和结果恰好可表示，不代表一般FP16无舍入。普通支路恢复和合并采用float64，**没有复现整条LLM.int8内核的FP16融合/布局**。
- 全零输入行/权重列编码和输出均为0，有限值检查通过。反例`[127,127]@[127,127].T`用int8累加回绕为2，显式int32累加为32258。
- 小OPT：1层、H16、2heads、FFN32、vocab10、max_position32、2960参数，CPU FP32随机初始化。真实本地WordLevel tokenizer生成int64输入[1,4]；原53用max_length=8得[1,8]，独立max_new_tokens=8得[1,12]，都保留输入前缀。原54实际遍历全部参数打印CPU/float32；生成前后参数哈希相同。这不是量化OPT、预训练输出或生成质量评测。
- 配置补充：BitsAndBytesConfig默认4bit/8bit都False合法，8bit配置threshold为6.0，同时True真实抛ValueError。仅验证配置，不导入/执行bitsandbytes。

每个执行格的Tensor/array实际dtype、shape、数值和stdout，以及环境、两版比较、明确排除项见 [runtime_verification.json](outputs/runtime_verification.json)。它报告的历史与补充范围不能扩大为GPU结果。

## 重做命令

在独立Python3.12虚拟环境安装CPU依赖。第一条指定PyTorch CPU wheel来源，第二条安装其余锁定包（requirements中的torch版本可由已安装2.5.1+cpu满足）：

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

从本讲`复现`目录执行：

```powershell
python scripts/download_original.py
python scripts/build_annotated.py
python scripts/verify_annotations.py
python scripts/verify_runtime.py
python scripts/verify_quantization.py
```

下载脚本仅取固定原notebook与PNG，逐一核对哈希，不下载模型。构建脚本重生成逐格说明、派生notebook和Python阅读版；静态脚本检查原语句只增加注释、AST/schema/历史内容、Markdown及图片；runtime脚本从真实文件读取原/注释代码，切换到正确图片工作目录执行上述范围，无网络模型调用。最后一条是保留的早期标准库教学验证，用Python整数算术交叉核对，不是bitsandbytes实现。

实际使用的解释器：`G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe`；工作目录`G:/zzz/personal-chatgpt-notes`；依次传入上述脚本在本讲内的完整路径。原图片展示需要`代码/tutorials`工作目录，runtime内部临时切换后恢复；Jupyter建议直接从该目录打开notebook。Python阅读版主要用于阅读，整份执行会进入原代理与大模型段。

## 证据与未解决项

- [原文件下载字节与哈希](outputs/source_download.json)
- [全部59格/38代码AST与6schema核对](outputs/annotation_verification.json)
- [真实CPU执行与补充结果](outputs/runtime_verification.json)
- [早期标准库教学结果](outputs/教学验证.json)

原OPT checkpoint revision、讲师完整依赖版本、本讲字幕仍未确定。没有测量GPU加载峰值、INT8内核速度、显存、语言质量或论文C4/零样本指标。后续真实模型验证需要单独确认兼容CUDA环境、权重与测试数据，并重新记录；当前状态保持部分成功。
