# 第 11 讲复现记录

状态：**部分成功**。全部 28 格说明、15 代码 AST、2 段 Markdown 循环 AST、3 schema、原输出/元数据与下载字节核对通过。原/注释各 11 格完整源码执行一致；真实 autograd/SGD 原循环及窗口修正、小随机 BERT Trainer baseline/累积训练通过。未下载 BERT-large 或复现原双卡 GPU 显存/速度。

## 最新重做命令

从本讲目录执行，Python3.12。NVML 验证需要实际 NVIDIA 驱动可用；CPU torch 无需 CUDA runtime，但原 NVML 函数仍调用驱动。没有 NVIDIA/NVML 的环境可先运行早期标准库算术脚本；那不覆盖真实 torch/Trainer/NVML 检查。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r 复现/requirements.txt
.venv/Scripts/python.exe 复现/scripts/download_original.py
.venv/Scripts/python.exe 复现/scripts/build_annotated.py
.venv/Scripts/python.exe 复现/scripts/verify_delivery.py
```

`download_original` 下载固定上游 notebook/原图并核对 hash；`build_annotated` 生成两份 57 格派生 notebook、Python 阅读版和单元说明；`verify_delivery` 核对原语句、元数据/历史输出、AST/schema，再运行真实 CPU 验证。无需模型 checkpoint，临时训练产物在结束清理。`代码/窗口归一化_教学补充.py` 为可阅读的独立修正示例。

## 实际环境与运行命令

使用已有共享 CPU 环境，新增 `nvidia-ml-py==12.560.30`（导入名 pynvml），安装包约 40 kB。实际 Python3.12.14、torch2.5.1+cpu、transformers4.44.2、accelerate0.34.2、datasets2.19.2、numpy1.26.4、IPython8.37.0、nbformat5.10.4、Pillow12.3.0。原 Python3.7 环境未改动；它的旧 torch 不可用记录不描述本次环境。

仓库根目录实际执行：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' -m pip install nvidia-ml-py==12.560.30
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/11 梯度累积 gradient accumulation/复现/scripts/download_original.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/11 梯度累积 gradient accumulation/复现/scripts/build_annotated.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/11 梯度累积 gradient accumulation/复现/scripts/verify_delivery.py'
```

退出 0；输出全 28 格/15 AST/2 Markdown AST 检查通过、真实原/注释各 11 格和 4 次 tiny BERT Trainer 运行通过。此处的 4 次为原/注释×baseline/累积，每次 2 更新。

## 实际范围与证据

| 文件 | 内容 |
|---|---|
| [source_download.json](outputs/source_download.json) | 固定原 notebook/图 URL 与 SHA256 |
| [annotation_verification.json](outputs/annotation_verification.json) | 单元集合、15 代码与两循环 AST、3 schema、原输出/元数据、阅读版 |
| [runtime_verification.json](outputs/runtime_verification.json) | 11 格真实执行、真实 NVML、autograd/SGD、BERT 回调/参数/shape 与环境 |

原/注释实际执行编号：0、2、8、9、10、12、13、15、16、20、21。原格15的 1024×512 随机ID实际运行；18代理未执行，19原远程BERT/CUDA加载未执行。23使用受控 CPU 模型/8×16数据执行整个原训练格；26在独立教学变体中仅替换 num_seqs1024→8、seq_len512→16、模型加载表达式→随机 CPU BertConfig，原 Trainer/NVML 语句保留。

小 BERT：词表30000、hidden16、1层、2heads、intermediate32、max_position512、分类2、dropout0，共490786参数。np seed1234/torch初始化seed42，Trainer默认seed42。相同初始参数和随机ID，baseline batch4/K1 vs accumulation batch1/K4；观察输入 int64 `[4,16]` 或 `[1,16]`、logits float32 `[4,2]` 或 `[1,2]`、loss标量。实际 2 或 8 次forward、均 2 次AdamW/线性scheduler更新；原/注释模型状态逐项完全相同，两设置最大参数差约9.3132e-10。关闭dropout是受控等价条件，不支持真实原配置逐位等价或语义准确率结论。

两段原 Markdown 循环以 Linear/SGD/MSE 上下文真实执行。完整窗口梯度[15,5]，更新参数[-.5,-.5]；短窗口原[.75,-.15]、按实际样本数修正[.5,-.3]；不等大小2+1批梯度约[70,8.6667]，更新边界4/8/10。原/注释循环输出一致；第三次同图backward的真实已释放图错误记录在 JSON，新图/清零后正常。

实际 NVML见1张卡，CPU torch可见CUDA为0，原函数因循环0次打印0MB。记录真实设备used字节用于区分设备查询与错误推断；它不是训练进程峰值。代理环境未设置，无权重下载、CUDA/双卡/AMP/DDP训练或原显存benchmark。

## 早期算术验证保留

[verify_accumulation.py](scripts/verify_accumulation.py)与[教学验证.json](outputs/教学验证.json)保留原 Python3.7.6 标准库 Fraction/中心差分实验。它的梯度缓冲为数值模拟，不是 torch.grad；现在另有上述真实验证，不覆盖或回填旧记录。

```powershell
python 复现/scripts/verify_accumulation.py
```

历史 Transformers v4.30.0/Accelerate v0.20.1 仍用于静态参照，本次真实运行是4.44.2/0.34.2。CPU BERT为可整除完整窗口；框架尾窗口、AMP/DDP和GPU benchmark未验证。完整原复现还需锁定 checkpoint revision、数据/初始化/seed、设备方式与峰值测量。
