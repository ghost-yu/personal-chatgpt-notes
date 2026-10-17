# 第07讲实际验证记录

[本讲入口](../README.md) · [中文讲解](../解释.md) · [详细代码注释](../代码/tutorials/代码注释.ipynb)

**状态：部分成功。** 全部19格说明、12格代码和Markdown1源码片段注释完成；原内容/schema/AST核对通过。原大CNN只有meta形状/参数验证；小型合成数据训练真正执行了PyTorch前向、反传、Adam和评估，并分别验证原CUDA AMP的CPU退回与补充CPU FP16 AMP。未下载MNIST、分配原大模型真实权重或进行3090显存实验。

## 原代码与补充的区别

| 代码范围 | 实际执行 | 条件/改动 |
|---|---|---|
| 单元3、4、6、7、10 | 全格原语句 | 真实torch/torchvision/torchsummary导入、device=cpu、原类定义、原超参数赋值 |
| 单元6原CNN构造/forward | meta设备，形状与numel真实API | 没有大权重/激活的数据存储，不是数值forward |
| 单元8原RGB224问题 | 原类+meta输入核对 | 不执行原summary整格，不分配大模型 |
| 单元11 | 原MNIST加载未执行 | 独立用真实ToTensor/TensorDataset/DataLoader构造8个合成样本，不替换MNIST类或伪装数据 |
| 单元12 | 全格原取批次/print | loader明确来自合成数据，batch4 |
| 单元14原train/evalute | 提取函数定义，函数体不改，真实调用 | 教学8/16通道CNN、batch4、epoch1、Adam lr3e−4 |
| 单元17原train/evalute | 同样提取原函数体并真实调用 | CPU包禁用原CUDA autocast/scaler，结果是FP32退回 |
| 单元17的CPU AMP补充 | 原train体只改一个autocast表达式，并审计反向AST还原 | torch.amp.autocast('cpu',dtype=float16)，CPU GradScaler另建 |
| 单元15、18 | 全格原打印 | 当前教学model/loader上下文；准确率不是MNIST结果 |
| Markdown1片段 | 原代码真正创建CUDA scaler | 当前CPU包警告并禁用，记录is_enabled=False |

完整14/17格包含下载、构造巨大模型和末尾train()，**没有整格运行**。函数定义提取与实际调用单独记录，不把它混成原配置训练成功。tqdm只在教学上下文关闭显示，模型/优化器/数据API均真实执行，没有假tensor或假GradScaler。

原notebook和注释版分别独立运行同一验证；除文件指纹外，所有形状、数值、更新、警告和状态结果相同。证据见[下载记录](outputs/source_download.json)、[注释核对](outputs/annotation_verification.json)、[真实训练记录](outputs/training_verification.json)。旧的[静态/算术教学记录](outputs/教学验证.json)仍保留。

## 原大模型meta检查

原CNN默认MNIST的numel为476,938,250，FP32仅参数约1.77673344GiB。输入meta[2,1,28,28]经原forward得到meta logits[2,10]。此处shape是真的框架计算，元素数值并不存在；没有声称大模型完成数值推理。

原RGB224路径flatten为[2,32112640]，fc要求501760，真实meta错误为：

```text
a and b must have same reduction dim, but got [2, 32112640] X [501760, 10].
```

原单元8历史摘要显示8/64通道和56×56fc，与保存类不一致。小型教学summary另用1×28×28；torchsummary1.5.1的报告batch参数32，实际虚拟前向batch为2，hook已核对。不能把报告参数当成工具实际一次跑了32样本。

## 小型真实训练结果

小模型只改原类中的四个通道常量：两处5120→8、两处10240→16。反向替换后完整类AST与原类相等，forward函数不改。小模型参数9098，数据为8张自编uint8灰度图，经真实ToTensor转float32[8,1,28,28]，标签int64[8]，batch4、不shuffle、epoch1。

| 运行分支 | logits dtype/shape | loss dtype | 控制器状态 | 真正Adam更新 |
|---|---|---|---|---:|
| 原FP32函数 | FP32[4,10] | FP32标量 | 无scaler | 2 |
| 原CUDA AMP函数在CPU | FP32[4,10] | FP32标量 | CUDA scaler disabled、scale1 | 2 |
| 明确CPU FP16补充 | FP16[4,10] | FP32标量 | CPU scaler enabled、128→256 | 2 |

三条路径的主参数和梯度均保持FP32。FP32与原CUDA AMP退回分支的更新后权重完全相等；这证明CPU上的原CUDA AMP没生效。CPU FP16补充的两批loss约2.3334141、2.3120840，FP32约2.3334293、2.3120751；这里仅观察数值/流程，不声称训练质量改进。

原evalute实际核对no_grad、准确率计数和模式变化：即使入函数前model已经eval，函数结束仍切成train。合成样本准确率为0，只用来核对统计定义，不作为MNIST评估；没有“生成成功的准确率”被补进历史输出。

CPU补充只将函数体中的`torch.cuda.amp.autocast()`换成`torch.amp.autocast('cpu',dtype=torch.float16)`。控制器在函数外明确创建为`torch.amp.GradScaler('cpu',init_scale=128,growth_interval=2)`；数据/模型/超参数替代也全部列明。CPU FP16实验支持此环境的行为，不代表GPU Tensor Core速度、显存或通用CPU最佳配置。

## GradScaler不是普通乘法演示

在两次有限更新后，CPU scale从128到256。下一次真实backward后人为把一个参数梯度写成Inf，scaler.step不更新任何权重，Adam step计数也未变；scaler.update使scale256→128。这个反例核对的是实际非有限检测/跳步/回退，不是靠假optimizer模拟。

独立真实autograd小例：参数p=1，loss=2p，scale128；backward得到梯度256，unscale_后为2，SGD lr0.1后p≈0.8。10类均匀logits的CrossEntropyLoss约2.302585，和−ln0.1一致。均为教学补充，不是原大网络结果。

## 环境与重做命令

实际Windows11、Python3.12.14、torch2.5.1+cpu、torchvision0.20.1+cpu、torchsummary1.5.1、numpy1.26.4、tqdm4.70.1、IPython8.37.0、nbformat5.10.4。原Python元数据为3.10.9，未锁完整依赖。CPU运行不需要GPU、权重、账号或MNIST网络下载。

从仓库根目录执行，python使用上述环境：

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "lessons/07 自动混合精度训练/复现/requirements.txt"
python "lessons/07 自动混合精度训练/复现/scripts/download_original.py"
python "lessons/07 自动混合精度训练/复现/scripts/build_annotated.py"
python "lessons/07 自动混合精度训练/复现/scripts/verify_annotations.py"
python "lessons/07 自动混合精度训练/复现/scripts/verify_training.py"
python "工具/check_documents.py"
```

安装和原notebook首次下载需要网络；训练验证只用合成数据。已匹配的环境不必重装。原notebook SHA256为`c007e4adba74d83cb577dc55c7df24697ff88c69b820e6b19996a2bfc4c852c7`，见[来源清单](../来源清单.json)。原代码及输出字节不清理，阅读版只清行尾空白并核对AST。

原CUDA AMP API在当前版有弃用/无CUDA警告，按实际记录保存；迁移只在补充函数发生，不修改公开原代码。固定参考实现为[PyTorch2.5.1 GradScaler](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/grad_scaler.py)。

未进行原MNIST训练、精确最大batch搜索、RTX3090峰值显存、性能/准确率对比；原summary不匹配问题保留并解释。129或257是否能跑、其他设备是否同样翻倍，均没有本次证据。


## 2026-10-17：基础说明、论文原图与正文算例

本轮从当前解释.md原样提取三块Python代码，使用既有Python3.12.14、PyTorch2.5.1+cpu实际执行；不下载MNIST或权重、不分配原大CNN。三块分别隔离数值转换、真实GradScaler、autocast前向，未假装成原网络整套训练。

| 本轮检查 | 实际结果 |
|---|---|
| 1e-8直接转FP16 | 0 |
| 放大1024→FP16→FP32→除1024 | 约1.00117177e-8，保留非零但仍有舍入 |
| 真实CPU scaler、p²、SGD lr0.1 | 放大梯度256，还原为2，p从1变约0.8 |
| growth_interval=1教学设置 | 有限更新后scale128→256 |
| 下一轮人为注入Inf梯度 | 不更新参数，scale256→128 |
| CPU FP16 autocast，2×2 Linear | 参数/输入FP32，logits FP16[2,2]，交叉熵FP32标量 |

完整输出、正文代码指纹与环境见[本轮验证](outputs/基础改写验证.json)。单参数例子明确不包含autocast；另一个线性层例子仅核对前向/loss的dtype，没有优化更新；原代码怎样将两部分合入一个循环已在正文对照。

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' './lessons/07 自动混合精度训练/复现/scripts/verify_foundation_examples.py'
& 'C:/Users/hgfhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' './lessons/07 自动混合精度训练/复现/scripts/extract_paper_images.py' --pdf 'G:/zzz/Personal ChatGPT 前七讲/07 自动混合精度训练/资料/Mixed_Precision_Training_1710.03740v3.pdf'
```

补充方法论文为Mixed Precision Training，arXiv:1710.03740v3；Figure1在PDF第3页，直接渲染裁剪，未重画。已实际查看截图，前向/两种反向、主权重转换、FP32更新箭头及图号完整。来源/坐标/哈希见[截图记录](outputs/论文截图来源.json)。PDF留本地，公开PNG嵌入正文；其他机器重做时从记录中的官方URL取得同版本PDF再替换--pdf路径。正文区分论文概念图与PyTorch AMP实现，没有把它当成本notebook论文实验复现。

原notebook、原注释文件和历史输出保持不变。上文meta形状、8/16通道合成CNN两步训练及原版/注释版比较仍是历史记录，本轮未重跑。没有CUDA/GPU AMP、MNIST原3epoch、RTX3090显存上限、速度或准确率新结果。
