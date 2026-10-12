# 第 10 讲复现说明

状态：部分成功。真实运行本地 JSONL 的 Dataset/IterableDataset、map、索引、take/skip、shuffle、保存重载与 Arrow memory mapping，未使用 C4/Pile 大语料、远程 streaming、模型或 PyTorch DataLoader。

建议使用独立 Python 3.12 环境，在本讲目录执行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r 复现/requirements.txt
.venv/Scripts/python.exe 复现/scripts/verify_datasets.py
```

本次实际使用工作区另建的隔离环境：Python 3.12.14、datasets 2.19.2、pyarrow 16.1.0、numpy 1.26.4、pandas 2.2.2、huggingface-hub 0.23.4、fsspec 2024.3.1。`pip check` 通过，未改动已有 Python 3.7 环境。锁定依赖可避免旧 datasets 与最新 hub/Arrow 的接口组合问题，完整实际依赖另见 requirements-lock.txt。

脚本生成 5 行自建 JSONL，临时准备缓存、Arrow IPC 和保存目录；临时文件随上下文退出清理。持久输出为 outputs/教学验证.json。map 计数验证真实执行时机，MemoryMappedTable/MemoryMappedFile 验证实际映射路径，不只用低 RSS 猜测。

未记录大数据性能或内存压缩率；微小文件进度条速度不是 benchmark。课程原 notebook 只有图片，字幕的约 20 GB 数据与原运行代码缺失；本次脚本是补充演示。
