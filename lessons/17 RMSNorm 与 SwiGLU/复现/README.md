# 第 17 讲复现记录

状态：部分成功。Windows、Python 3.12.14、NumPy 1.26.4；CPU float64 小型验证。默认 Anaconda 的 torch 不具备 Tensor 接口，未运行 PyTorch/FairScale，也未加载 Llama 权重。RTX 4060 Laptop 8 GiB 不参与此次计算。

通用命令（在本讲复现目录内运行）：

```powershell
python -m pip install -r requirements.txt
python scripts/verify.py
```

本次精确命令，工作目录 `G:/zzz`：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/datasets/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/17 RMSNorm 与 SwiGLU/复现/scripts/verify.py' --official-model 'G:/zzz/Personal ChatGPT 系列/17 RMSNorm 与 SwiGLU/资料/llama-model.py'
```

`--official-model` 可选。提供时按来源清单 SHA256 核对源码，只用 AST 提取 `FeedForward.forward`；三层线性运算和 `F.silu` 绑定 NumPy fixture。省略时仍能验证数学，但不会执行官方方法。没有执行官方类初始化、并行通信、PyTorch kernel 或模型训练。

验证 shape、LayerNorm 均值、RMSNorm 零输入、长度恒定的条件与反例、正比例/平移性质、RMSNorm 和完整 SwiGLU 对输入的解析梯度与中心差分、官方 forward 运算连接、4096→11008 FFN 宽度与参数量。结果见 [教学验证.json](outputs/教学验证.json)。原 notebook 的随机数未设置 seed，本次固定 NumPy seed=17，不要求向量逐个数值与历史输出相同。

升级为 PyTorch 最小 forward 需要可用 torch、FairScale 和适当初始化；全模型还需匹配的配置及权重。当前不验证论文速度、收敛、BLEU 或 perplexity。
