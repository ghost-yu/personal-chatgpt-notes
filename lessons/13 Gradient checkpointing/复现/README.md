# 第 13 讲复现记录

## 2026-10-17 基础讲解与原论文截图改写

四幅图直接从已存Chen等论文arXiv:1604.06174v2原PDF渲染裁切：第3页Figure1、第6页Figure3、第5页Algorithm1和公式1，均逐张视觉检查、就地中文解释，未重绘。[截图来源](outputs/论文截图来源.json)记录PDF/图片/脚本哈希、坐标、页码、实际命令和检查状态。完整PDF留在原本地材料目录。

[本轮脚本](scripts/verify_foundation_examples.py)从当前解释.md折叠历史之前提取四段Python原样执行，用Python3.11.0、Torch2.0.1+cpu、Transformers4.30.0；[本轮结果](outputs/基础改写验证.json)另存，不覆盖旧报告。无模型下载、无需NVML或GPU。

- float64标量x=.5/w2：输出1、loss.5、参数梯度1、输入梯度4，权重仍2；ordinary只调用一次，reentrant检查点第一次no_grad/第二次有图，梯度相同。
- 4块Linear(2,2)+sigmoid/24参数/输入[1,2]、segments2：各块调用[1,1,1,1]→[2,2,1,1]，末段普通执行；输出、输入与全部参数梯度精确相同。未执行更新或测保存字节。
- 8元素Dropout：初次输出均[0,0,6,0,0,0,14,0]；恢复RNG重算一致，不恢复变为[2,0,0,0,10,12,14,16]，参数梯度最大差16。
- 随机2层H8/V32/两头BERT二分类，共1706参数；输入int64[2,4]/labels[2]、logits FP32[2,2]，dropout0。两条路径loss均0.6932495832443237，每层调用1→2；所有参数梯度存在，backward结束前权重不变，各一次SGD lr.1后确有更新且两边全梯度/全状态精确相同。直接forward/backward/SGD，不是重跑旧Trainer四组。

从正式仓库根目录执行：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy/Scripts/python.exe' 'lessons/13 Gradient checkpointing/复现/scripts/verify_foundation_examples.py'
```

用已有原PDF重做截图可运行`scripts/extract_paper_images.py --pdf <原v2.pdf>`，需pypdfium2/Pillow。本次使用桌面内置PDF Python；首次生成后先查看四张PNG，确认后才使用`--confirmed-visual-review`写检查标记，程序不自动判断视觉质量。

本轮诊断快照/日志会额外占CPU存储，不作内存基准。没有重跑历史四组H16 BERT Trainer、保存引用字节统计、冻结embedding或eval失败负例，也未执行原notebook训练、BERT-large/CUDA/NVML、AMP/DDP、non-reentrant迁移、论文CNN/LSTM、显存速度或语言/分类质量实验。

## 历史完整交付记录（本轮保留，未重跑）

状态：**部分成功**。公开原notebook、两种47格派生版、纯文本阅读版、全部23格说明与真实Torch教学补充；新运行核对真实重算、梯度、RNG及两版四组CPU BERT Trainer。原GPU/大模型/论文完整实验未复现。

## 证据与环境

- [文件审计](outputs/annotation_verification.json)：23原格、12代码AST、Markdown21开关类AST、Python阅读版、3schema、原语句/元数据/历史输出。
- [真实运行](outputs/runtime_verification.json)：实际命令、软件/源码/脚本哈希、四组训练trace与参数比较、sequential保存引用统计、RNG/输入梯度/评估负例。
- [来源记录](outputs/source_download.json)：从已下载上游Git clone读取固定提交原blob，12418 bytes，SHA256 `a2f49289c867c32bf2ac3795c2068586d0840420eccb98ebcc28bd2e67193711`。来源清单旧a41f…为CRLF工作副本，不与公开原字节混用。

本次环境为Windows、Python3.11.0、torch2.0.1+cpu、Transformers4.30.0、Accelerate0.20.3、Datasets2.13.1、NumPy1.26.4、nvidia-ml-py12.560.30，导入与pip check通过。复用独立 `G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy` 环境；其完整已安装依赖列表在 [环境锁](outputs/environment-lock.txt)，还包含第12讲使用的TRL/PEFT。作者精确原环境未找到，不能将参考版本称作作者依赖锁。

CPU Torch不能运行CUDA，但机器驱动/NVML可查询1张RTX4060 Laptop卡。原NVML查询真实执行并记录物理整卡占用，不当作本次CPU训练显存。此验证脚本要求NVML能访问物理卡0；没有驱动/卡的机器会在原查询格失败，需要明确记录并调整查询范围，不能用假API返回0。

## 重做方法

从本讲目录执行，使用Python3.11，先装CPU Torch再装依赖：

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r 复现/requirements.txt
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe 复现/scripts/build_annotated.py
.venv/Scripts/python.exe 复现/scripts/verify_delivery.py
```

`build_annotated.py`重新生成全部单元说明、注释/说明版及Python阅读版。`verify_delivery.py`先审计文件，再导入真实安装包，模型随机初始化，不下载BERT权重；训练目录和checkpoint由TemporaryDirectory隔离并自动清理，输出只保存验证记录。

原notebook已经在仓库，无需重下。要重新取得原字节可运行：

```powershell
.venv/Scripts/python.exe 复现/scripts/download_original.py
```

脚本默认读取固定raw URL并核对SHA256；已有clone时可给 `--upstream-clone`，读取原Git blob。本次用clone方式，没有宣称进行了新的HTTP下载。

本次实际命令在仓库根目录执行：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy/Scripts/python.exe' 'lessons/13 Gradient checkpointing/复现/scripts/download_original.py' --upstream-clone 'G:/zzz/Llama 入门教程/代码/personal_chatgpt'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy/Scripts/python.exe' 'lessons/13 Gradient checkpointing/复现/scripts/build_annotated.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy/Scripts/python.exe' 'lessons/13 Gradient checkpointing/复现/scripts/verify_delivery.py'
```

## 原格执行与CPU改动

两版完整执行6、7、8、9、13、14、16、17、19、22，共10可行代码格；14与16在对应原组保持整格AST。原6先实际创建512×512随机ID/512标签，再用只改seq_len,dataset_size为16/8的独立变体构造教学数据。原4设备环境设置跳过，原11BERT-large/CUDA不执行；11的CPU变体只把from_pretrained改随机BertConfig构造、cuda改cpu。

原13共享字典完整执行后，教学上下文显式更新output_dir为临时路径、evaluation_strategy=no、max_steps=2、save_strategy=no。另两对照组只给14加checkpoint=True，或把16的checkpoint改False。各组都重新初始化同样模型并核对所有初始状态与样本顺序相同，不串接训练上一组模型。Markdown21原/注释类在隔离环境真实操作BertEncoder开关，默认False、True设置及非encoder不处理均通过；它不是原notebook执行格。

小BERT词表30000、hidden16、2layer/2heads、intermediate32、max_position512、分类2、dropout0，共493010参数；numpy seed1234、模型/Trainer seed42。各组均2 optimizer步骤，使用旧默认Transformers AdamW（经AcceleratedOptimizer包装）。B4/K1/off与on各2次模型forward、每层调用2/4；B1/K4/off与on各8次forward、每层调用8/16；checkpoint首forward每层grad_enabled=False，backward重算True。

固定B/K切checkpoint的训练后参数精确相同；B4K1与B1K4参数最大差1.862645149230957e-9。所有组平均train_loss0.6900777816772461，仅为随机分类流程统计。两版对应四组参数/forward trace/层调用trace精确相同，不比较运行时间或可变NVML读数。没有声称原BERT-large学习质量或GPU显存变化。

## 真实重算与负例

独立float64顺序网络4个8维Linear+sigmoid块，输入[2,8]，segments2：普通每层一次，checkpoint前两层各两次、末两层各一次，总层调用4→6；输出、参数梯度和输入梯度一致，最大梯度差0。初次forward保存引用事件12→7，非参数去重storage640→512bytes；仅观测初次forward保存到的input/activation引用，不是实时内存、RSS、CUDA峰值或理论slot统计。

真实dropout重算默认恢复RNG时输出/梯度一致；关闭preserve_rng_state后初始forward输出仍相同，但重算mask不同、参数梯度最大差0.710811140920421。旧reentrant输入requires_grad=False时实际产生无梯度告警、backward失败；给输入requires_grad或使用此小函数的non-reentrant均与普通参数梯度相同。

冻结小BERT embedding再启用旧reentrant时，分类头梯度存在且loss.backward完成，但encoder32个参数张量无grad；enable_input_require_grads hook后32个encoder参数张量恢复梯度，embedding仍冻结/grad=None。

额外真实Trainer以eval_steps1/max_steps1且不传eval_dataset运行，首次计划评估抛出 `Trainer: evaluation requires an eval_dataset.`。原历史短训练可能没触发，不能证明配置完整。

## 早期标准库验证保留

[verify_checkpointing.py](scripts/verify_checkpointing.py) 与 [教学验证.json](outputs/教学验证.json) 保留Python3.7.6的手写标量导数/有限差分、逻辑激活槽与源码检查。它没有执行Torch，本次没有回填或覆盖旧记录。

```powershell
python 复现/scripts/verify_checkpointing.py --sources 'G:/zzz/Personal ChatGPT 系列/13 Gradient checkpointing/资料' --output 复现/outputs/教学验证.json
```

此命令仅重做旧验证，本次没有再执行。旧16层/4段手写算法将每段都重算，forward16→32、逻辑slot17→8；与新Torch sequential末段普通执行不同。slot不是RSS或GPU字节。

尚未运行原BERT-large、CUDA/AMP/DDP、四组GPU峰值/吞吐实验或论文CNN/LSTM结果；还需足够设备、固定原checkpoint/数据/seed、独立进程与同步/预热，在可比设置中测peak allocated/reserved。现代checkpoint迁移未实测。以上未运行项不能由CPU小模型证据代替。
