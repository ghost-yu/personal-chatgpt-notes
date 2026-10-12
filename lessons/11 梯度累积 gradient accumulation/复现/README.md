# 第 11 讲复现记录

状态：部分成功。实际运行精确有理数/中心差分教学验证，没有执行 PyTorch autograd、Trainer 或 BERT 训练。

在本讲目录执行：

```powershell
python 复现/scripts/verify_accumulation.py
```

本次 Python 3.7.6；仅标准库。输出 outputs/教学验证.json，检查等大小完整窗口的梯度等价、每 micro step 的不同结果、末尾缩放错误、样本数加权与更新边界，以及历史 GPU 数字的算术。梯度为手推的线性模型平方损失导数，并以中心差分独立核对；梯度缓冲为数值模拟，不是 torch 的 `.grad`。

现有 torch 缺少 tensor API 与版本信息；GPU 8188 MiB。原 demo 需 BERT-large、可用 torch/datasets/transformers/accelerate/NVML、足够设备资源，本次没有执行加载。原 proxy、下载与训练命令未运行；历史 Markdown 结果不代表本次结果。

官方源码参照 Transformers v4.30.0、Accelerate v0.20.1，仅用于静态追踪。若另行真实复现，要固定模型 revision、同一数据/初始化/seed、设备方式与测量峰值，不只比较训练结束后的 NVML 总量。
