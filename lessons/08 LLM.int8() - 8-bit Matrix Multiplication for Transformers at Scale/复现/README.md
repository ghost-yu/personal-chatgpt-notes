# 第 08 讲复现记录

状态：部分成功。静态阅读与纯 Python 教学算术通过；没有运行 OPT 或 bitsandbytes CUDA kernel。

在本目录执行：

```powershell
python scripts/verify_quantization.py
```

Python 3.7.6，Windows，RTX 4060 Laptop GPU 8188 MiB。现有 torch 只有命名空间，没有 tensor API；因此没有完整模型 import/forward。脚本不需要第三方依赖，输出 `outputs/教学验证.json`。它验证舍入、尺度恢复、特征拆分与零尺度保护；不测试硬件性能、生成指标或半精度舍入。

原示例未舍入时误差接近零；修正后混合分解最大绝对误差 0.01574803149605941，不分解的向量级量化为 1.0940541881083732。所有数值仅适用于给定 3×5 例子。
