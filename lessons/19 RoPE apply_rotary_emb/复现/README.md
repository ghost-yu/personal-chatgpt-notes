# 第19讲复现记录

状态：部分成功。Windows、Python3.12.14、torch2.5.1+cpu、NumPy1.26.4、matplotlib3.9.4、sentencepiece0.2.0，使用独立CPU环境。没有下载模型权重或训练数据，GPU未使用。完整实际版本见 [环境锁定](outputs/environment-lock.txt)。

在本讲复现目录中，以Python3.12安装并执行：

```powershell
python -m pip install -r requirements.txt
python scripts/verify.py
```

不传参数时从固定GitHub地址读主notebook、Meta源码与约500KB tokenizer，按来源清单核对哈希。也可传入本地文件实现无网络运行。

本次精确命令，工作目录`G:/zzz`：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/19 RoPE apply_rotary_emb/复现/scripts/verify.py' --notebook 'G:/zzz/Personal ChatGPT 系列/19 RoPE apply_rotary_emb/llama2_src_03_RoPE_apply_rotary_emb.ipynb' --official-model 'G:/zzz/Personal ChatGPT 系列/19 RoPE apply_rotary_emb/资料/llama-model.py' --tokenizer 'G:/zzz/Personal ChatGPT 系列/19 RoPE apply_rotary_emb/资料/tokenizer.model'
```

三个flag指定原始本地文件；接受固定notebook的本地CRLF与上游LF两个哈希，其余内容不匹配停止。原单元18使用相对模型路径，脚本在临时目录放置原文件、运行原代码、恢复工作目录，避免改写原单元。

先执行全部15个原代码单元，含注释与空单元；此时旋转函数只是定义，实际调用会缺helper而NameError。脚本捕获并记录这个原始缺口，然后AST提取官方三个函数，绑定真实torch。为原函数显式提供官方helper后验证原函数与官方一致。未替换torch API或模型输入为fixture。

seed=19，Q `[2,4,4,8]`，K `[2,4,2,8]`，位置从3开始。官方输出/梯度与独立sin/cos参考一致，相对点积差约7.47e-7，共同位移差约4.77e-7。测试FP32/FP16/BF16/FP64的返回dtype及舍入，stride不合、奇数d、错误相位shape和静默广播错轴的反例。

原数学曲线通过独立累积和复核，C(0)=31.5，C(255)≈6.412620，中间113步上升。真实tokenizer得到历史8个ID。证据见 [教学验证.json](outputs/教学验证.json)、[曲线数据](outputs/curve_data.json)。自己的两幅曲线已视觉检查。

这些测试不运行完整Attention、KV缓存、FairScale、CUDA或大模型生成；不证明速度、长文本能力或RoFormer论文评测复现。代码单元覆盖及本地带说明notebook见上级文档。

## 生成带说明的 ipynb

在复现目录执行以下命令，将产物写到你选择的本地配套目录：

```powershell
python scripts/build_annotated.py --output '原代码_带说明.ipynb'
```

也可传入 `--notebook` 指定固定版本的本地文件。本次输出在本地第19讲目录：27个原单元加15个说明单元和1个阅读声明，共43个单元。原代码和历史输出全部保持原样，说明编号与15个代码单元完全匹配，另已通过 notebook schema 验证。证据见 [覆盖记录](outputs/code_unit_coverage.json)。
