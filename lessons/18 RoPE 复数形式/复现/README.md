# 第 18 讲复现记录

状态：部分成功。主 notebook 代码执行成功，官方频率函数和小型数学验证成功；未运行 Llama 模型与 RoFormer 论文实验。

## 环境与准备

Windows，Python 3.12.14，torch 2.5.1+cpu、NumPy 1.26.4、matplotlib 3.9.4。新建独立 CPU 环境，不修改默认 Anaconda。没有检查点、训练数据、API key 需求；没有下载模型权重，8 GiB RTX 4060 Laptop GPU 未使用。torch CPU wheel 下载约205 MB，张量测试仅为小数组。

安装版本参照 [PyTorch 历史版本官方页面](https://pytorch.org/get-started/previous-versions/)，不是老师历史环境复原。完整实际版本见 [environment-lock.txt](outputs/environment-lock.txt)。

在本讲复现目录内使用 Python 3.12：

```powershell
python -m venv .venv
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
& '.venv/Scripts/python.exe' scripts/verify.py
```

默认脚本从固定 GitHub raw URL 读取主 notebook 和 Meta model.py，在执行前按来源清单核对 SHA256；不将下载内容写进公开输出。可以传入本地文件实现无网络运行。它顺序执行主 notebook 的代码单元，使用 Agg 后端绘图；只提取官方 `precompute_freqs_cis` 函数，未导入整个 FairScale 模型文件。

本次精确命令，工作目录 `G:/zzz`：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/18 RoPE 复数形式/复现/scripts/verify.py' --notebook 'G:/zzz/Personal ChatGPT 系列/18 RoPE 复数形式/llama2_src_03_RoPE_01_complex_position_encoding.ipynb' --official-model 'G:/zzz/Personal ChatGPT 系列/18 RoPE 复数形式/资料/llama-model.py'
```

`--notebook` 和 `--official-model` 指定固定版本的本地源码；传入不匹配文件会因 hash 检查停止。数学测试用固定输入，无需随机 seed。

默认网络路径也已实际运行。首次核对发现本地 checkout 为 CRLF、上游原文件为 LF，因此原始字节哈希不同；确认转换行尾后内容完全一致，来源清单分别记录两种哈希，脚本只接受这两份固定字节版本。这次失败发生在执行源码前的哈希检查，修复后网络路径成功。

## 实际结果与边界

- 主 notebook 的所有16个代码单元执行，包含空单元；实数类参数数目为0。
- `[10,6]` sin/cos 表与 complex64 `[10,3]` 表恢复结果最大差 5.96e-8。
- 两种存储各240字节，不支持“复数表示节省一半内存”的结论。
- 缺少 `/dim` 的错误频率最大差约0.414712；奇数维真实 class 的 stack 报错。
- 二维矩阵与复数旋转一致；六维长度误差约4.44e-16，点积/共同位移在1e-12容差内一致。
- 真实 `torch.polar` 的角度 autograd 与解析梯度一致。

记录见 [教学验证.json](outputs/教学验证.json)、[stdout](outputs/notebook_stdout.txt)。自己的示意图保存在上级图片目录，已经视觉检查。未运行 Attention、apply_rotary_emb 完整模型调用、FairScale、CUDA、FP16/BF16 或论文训练；后续第19讲继续核对具体旋转应用。
