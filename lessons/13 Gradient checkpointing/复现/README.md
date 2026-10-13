# 第 13 讲复现记录

状态：**部分成功**。手写 reverse differentiation 验证同一 16 层标量 sigmoid 网络在全保存与分段重算时 output、loss、权重梯度和输入梯度一致，并用有限差分独立检查梯度。验证随机 mask 重放的必要性，核对历史实验统计和旧源码。没有运行 PyTorch checkpoint、BERT 或 GPU benchmark。

工作目录为本目录：

```powershell
python scripts/verify_checkpointing.py --output outputs/教学验证.json
```

`--output` 是报告文件。可选 `--sources` 对官方源码快照做 AST 解析和 BERT 分支检查。本次命令为：

```powershell
python scripts/verify_checkpointing.py --sources "G:/zzz/Personal ChatGPT 系列/13 Gradient checkpointing/资料" --output outputs/教学验证.json
```

环境：Windows、Python 3.7.6。默认 torch 安装为不完整 namespace，缺少 Tensor/tensor API；GPU RTX 4060 Laptop 8 GiB。原无 checkpoint 历史占用约 12.6 GiB，不能直接作为本机可运行配置。没有下载 BERT 权重，没有补装大型 GPU 框架。

本次保存的 activation slot 是教学程序统计的逻辑激活索引，不是 RSS、CUDA allocated/reserved 或 NVML。16 层分为 4 段，前向调用从 16 增至 32，逻辑 activation slot 从 17 降至 8；参数、梯度、Python 对象开销不在 slot 统计中。

真实显存复现应在兼容框架和足够显存的设备上，分别以全新进程跑相同初始模型/数据的四组配置：B4/K1/checkpoint-off、B1/K4/off、B4/K1/on、B1/K4/on。明确 `evaluation_strategy='no'` 或提供 eval_dataset；固定随机种子，预热，CUDA synchronize，并用 reset_peak_memory_stats/max_memory_allocated 与 max_memory_reserved 测量训练峰值。不要把本脚本当作这些实验已经执行的证据。
