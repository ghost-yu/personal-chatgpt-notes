# 第29讲复现记录

## 2026-10-18 正文基础算例

本轮直接提取主讲解折叠前的六块 Python 代码，在 CPU 上执行。使用 Python 3.12.14、PyTorch 2.5.1+cpu、Transformers 4.44.2，单线程。

- 查表与输出 head：两个独立的 4×2 权重，隐藏状态 `[1,2,2]`、logits `[1,2,4]`，数值逐项核对。
- 原第11格完整参数函数 AST 一致；完整尺寸 meta 模型与独立模块统计/公式账一致，总数 6,738,415,616 和 8,030,261,248，净增 1,291,845,632。
- 实际 `repeat_kv` 的两个查询共享 K/V，输出约 17.3106 与 12.6894；完整 `LlamaMLP` 标量例结果 4.386351，三个参数。
- 两个随机两层小模型共 7,280/8,784 参数。logits `[1,4,32]`/`[1,4,64]`，第一层缓存 `[1,4,4,4]`/`[1,1,4,4]`。完整前缀与分段缓存的最后位置误差都为 2.24e-8。
- 实际 greedy generate 与每步完整前缀重算得到相同编号；模型参数保持原值，未创建参数梯度。
- 复用旧脚本的保留性检查和完整核心函数数值检查：7 个 HF 对象 AST/源码一致、包 RECORD 匹配；三种 K/V 头数的独立逐头前向、输入/四投影梯度、SDPA/eager，以及完整 MLP 前向/梯度/gradcheck 均通过。最大梯度差 8.89e-16。

结果见 [基础讲解验证.json](outputs/基础讲解验证.json)。固定 4.44.2 库输出了 tuple cache 与内部 RoPE 的 API 迁移提示，数值检查通过。

~~~powershell
$py = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
& $py ./复现/scripts/verify_foundation_examples.py
~~~

三张截图分别来自 Llama 2 v2 第5页架构段落、GQA v3 第2页 Figure2、GLU Variants v1 第2页公式组6及宽度说明。原PDF位于本地 `G:/zzz/Personal ChatGPT 系列/29 Llama2 7B vs Llama3 8B/资料/`；裁剪坐标、PDF/PNG SHA256 见 [论文截图来源.json](outputs/论文截图来源.json)。三张截图均逐张视觉检查。首发 Llama 3 模型卡沿用固定 Meta 提交 cf3ea595be1edfb019d21b14d8b215a39161c860 的本地快照。

本轮运行范围为架构计数、教学前向、缓存/生成与完整模块梯度核对。原远程 7B/8B checkpoint、GPU 开销和模型回答质量没有在本轮运行；此前三步训练和本地加载结果保留在下面的历史记录。

## 此前交付记录

# 第29讲复现记录

状态：**部分成功**。原件/详注交付、完整尺寸meta参数账、实际随机小Llama的加载、前向、梯度、三步训练和缓存对照成功。没有加载原7B/8B权重，没有测双4090、原语言质量或论文全实验。

## 环境与资源

Windows 11 build26200，Python3.12.14；Torch2.5.1+cpu、Transformers4.44.2、Accelerate0.34.2、nbformat5.10.4、IPython8.37.0。CPU单线程，`torch.cuda.is_available() == False`。物理RAM约15.69GiB，运行末尾的可用内存和工作区磁盘空余字节数保留在报告中；它们是时间点读数，不是运行峰值。完整7B/8B只创建meta结构，不分配对应权重数值。临时小checkpoint在系统temp目录生成并自动清除，没有公开权重、token/API或下载数据集。

详细版本、资源原始字节数和SHA256见 [delivery_verification.json](outputs/delivery_verification.json)。讲师软件版本未知，本记录不冒称历史环境完全还原。

## 重做命令

在本讲目录运行，示例PowerShell：

~~~powershell
$py = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
& $py ./复现/scripts/build_delivery.py
if ($LASTEXITCODE -ne 0) { throw '生成失败' }
& $py ./复现/scripts/verify_delivery.py
if ($LASTEXITCODE -ne 0) { throw '验证失败' }
~~~

`$py`指定已有实际解释器。`build_delivery.py`从固定本地上游Git blob重新生成原件/派生件并提取安装的官方源码；它不执行远程模型加载。验证脚本会比较原件、运行明确meta/小模型路线，并写JSON记录。两脚本根路径均以自身位置定位，只有上游Git路径固定为本地来源位置；搬机器需先取得对应固定仓库并调整`UPSTREAM`和验证中的本地Git路径。

另建环境时使用Python3.12、安装匹配的CPU PyTorch wheel，再安装 [requirements.txt](requirements.txt)。例如在新虚拟环境中`python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r 复现/requirements.txt`；该安装命令本次没有运行，现有环境没有改动。不能因为环境安装成功就宣称原模型复现成功。

## 保留性与源码检查

- 24个原单元，15代码格（包括只含注释的6格、空23格），原Git blob字节不变。
- 两派生notebook各49格；三份schema通过。所有原单元元数据、历史输出保留，新增单元有独立id。
- 原代码与注释代码AST、非注释token一致；普通Python阅读版AST与原全部代码拼接一致。
- Markdown9设备预算代码、21投影与MLP代码同样检查AST/token。Markdown2内联加载语句解释在新增说明中，未谎称执行原Meta权重加载。
- 7个完整HF对象：`rotate_half`、`apply_rotary_pos_emb`、`repeat_kv`、`LlamaRMSNorm`、`LlamaMLP`、`LlamaAttention`、`LlamaSdpaAttention`。独立注释模块真实导入，AST与安装对象一致，modeling_llama.py符合安装包RECORD；完整源SHA写入清单。未声称已从远端tag下载验证。

## 两条执行路线，各自能证明什么

| 路线 | 明确改动 | 实际结果 |
|---|---|---|
| 完整尺寸meta | 仅5/7格改用from_config，在meta设备中创建模型 | 每版13格原语句、2格明确变体；结构/计数输出一致，精确6738415616与8030261248；没有权重数值forward |
| 真正本地CPU小模型 | 仅4格的两个Hub标识改成本地随机checkpoint目录；5/7保留原BF16与auto加载 | 每版14格原语句、1格路径变体；所有参数实际CPU BF16，auto映射`{'': 'cpu'}`，真实forward有限 |

原0格的代理环境赋值在验证子进程中实际执行，结束后恢复；不修改用户系统代理。模型加载只指向本地临时目录，`Image(url=...)`只构造展示对象，未验证外站图片内容。原6格没有实际打印设备，23格没有语句，仍按其真实性质保留。原/注释版捕获stdout、末尾表达式与Image URL后逐项一致。

完整meta分类账：输入表+394264576、输出head+394264576、attention−805306368、MLP+1308622848、norm不变，总差1291845632。该结果证明架构元素数，不证明大模型checkpoint能加载或生成质量。

## 真正训练、重载与缓存

小模型2层、D32、query heads4，旧式KV4/F86/V64，新式KV1/F112/V128。R分别1和4，与本讲共享比例对应；其他数值不冒称原模型缩放定律。两模型分别28960和34976个参数。初始化seed292/293，数据是自建`int64[2,6]`，没有自然语言任务含义。

BF16本地加载与一次有限值forward之后，**补充训练**转FP32。固定dropout0，AdamW lr0.001，各3次更新；每个参数都有有限梯度，两版训练后状态逐tensor精确一致。`model.train()`启用训练行为，`model.eval()`切为评估；它们不自动冻结参数。`with torch.no_grad()`是暂时关闭构图的上下文，用于这里的评估/生成，和eval用途不同。

| 实际结果 | 旧式小模型 | 新式小模型 |
|---|---:|---:|
| 三步loss | 4.15300 → 4.02813 → 3.92751 | 4.84234 → 4.71180 → 4.59872 |
| 变化的state entries | 21 | 21 |
| 最后K cache shape | [2,4,6,8] | [2,1,6,8] |
| 三次增量对完整前缀最大logit差 | 7.45e-8 | 1.49e-7 |

真实`LlamaForCausalLM`的labels路径与独立shift交叉熵一致；保存真实checkpoint后重新`AutoModelForCausalLM.from_pretrained`，state与logits精确一致。greedy生成3个新ID实际执行，返回ID列表只是随机小模型输出，不能解读为语言能力。三步下降仅说明更新路径有效。

## 完整核心模块的独立检查

H4/d_h4/D16，KV头G4/2/1；真实完整Attention分别1024/768/640个参数。使用显式RoPE cos/sin和因果mask，逐query头循环选择`g=h//R`的独立参考不调用原repeat_kv。

- 前向最大差均0；输入和四投影权重梯度最大差不超过4.44e-16。
- 相同权重的完整SDPA对eager前向最大差约3.05e-8；未把前向比较扩大为所有内核梯度或CUDA性能验证。
- 完整MLP参数1152，独立SiLU公式前向最大差1.11e-16、全部输入/权重梯度差8.88e-16；对真实模块用`torch.func.functional_call`进行double gradcheck通过。
- RMSNorm与RoPE函数源码完整保留；完整小模型确实调用。未把第17/18讲验证结果冒称本次又重复全部测试。

## 实际失败、修正与未运行

实际反例：空参数统计ZeroDivisionError；lm_head不匹配时strict=False仍RuntimeError；越界ID为IndexError；D15/H4构造ValueError；H4/G3实际前向head维RuntimeError。具体消息保留在报告。

生成初次对新增说明格未给id，nbformat给出MissingIDFieldWarning；补加仅新增单元的id后重跑通过，原单元未变。首轮诊断后为新式小模型明确设置不同RMSNorm eps与RoPE theta，再完成最终重跑；损失值使用最后报告。原/注释前向、训练和梯度检查没有用stub/fake替代实际PyTorch模块。

官方HF config.json请求因TLS EOF失败；网络浏览部分GitHub/HF页面也不可达，来源边界如实记录。当前torch环境没有pypdf，论文段落核对改用已提供的文档运行时读取已有本地PDF，未安装新依赖。历史缓存/位置接口发出弃用warning，保留实际调用，不宣称适用于最新Transformers。

未运行：原远程7B/8B权重/原双4090分配、完整训练/数据集、论文能力分数、内存峰值或速度测量、原Markdown2 Meta checkpoint加载、原Markdown9缺上下文摘录。原图只有URL展示对象，没有远程内容结论。

## 讲解改版前的运行说明

## 10. 小模型中的加载、前向与缓存

1. 原24格、15代码格全部说明；原6格只有注释，23格为空，均明确保留。原notebook固定Git字节，两个派生版各49格，历史输出和元数据不改。代码AST及非注释token一致，Markdown9/21围栏源码也核对，阅读.py可解析。
2. 完整meta路线：原5/7格改为`from_config`并在`torch.device('meta')`下构造，其他代码格照原语句执行，原/注释版结果一致。没有冒称原BF16远程加载成功。
3. 实际本地小模型路线：仅原4格把两个Hub字符串改成本地随机checkpoint路径；5/7格保留原`from_pretrained(..., torch_dtype=torch.bfloat16, device_map='auto')`。实际CPU BF16加载、前向有限值、参数统计与模型展示通过。没有伪造AutoModel、Embedding或Linear。
4. 训练补充明确转FP32：2层、D32、query heads4、KV heads4/1、F86/112、词表64/128，自建2×6整数序列，AdamW lr=0.001各3步。验证shift loss、每个参数梯度、状态变化、两版状态精确一致、保存重载和逐token缓存对照。它是形状类似的小模型，不是真正的7B/8B能力缩影。
5. 完整HF核心对象7个可导入，AST与安装源一致。eager Attention和独立逐head参考比较前向及所有输入/权重梯度；SDPA比较前向。完整MLP与独立SiLU公式比较，并用`torch.func.functional_call`对实际类进行double gradcheck。

另有真实失败反例：隐藏维不能整除query头时构造报错；H=4、G=3虽可创建投影，但实际前向的head轴不匹配；越界token报IndexError；`strict=False`仍因lm_head形状不同报错。从这些报错可以看到，配置整除、head匹配和参数shape分别在哪一步被检查。

细节及实际数值以 [复现记录](README.md) 和 [验证输出](outputs/delivery_verification.json) 为准。
