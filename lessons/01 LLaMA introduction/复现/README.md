# 第 01 讲注释与执行核对

状态：**部分成功**。原notebook、图片、两种说明/注释notebook与Python阅读版均已放入仓库。9格全部说明；2个Python单元和6条Markdown启动命令加上详细中文注释。三个notebook的两格Python均实际执行成功；未加载LLaMA权重或复现论文实验。

`scripts/download_original.py` 从固定提交下载原文件和图片。`scripts/build_annotated.py` 生成两份19格notebook，并导出不含JSON和图片编码的[Python阅读版](../代码/tutorials/01_llama_intro_代码注释.py)。说明版保留原单元；注释版给两格Python与六条命令加注释。验证脚本核对schema、原单元/历史输出、Python AST、Markdown命令去注释一致性与阅读版AST，并实际执行两格Python。历史输出保留，不冒充新运行结果。

| 检查 | 结果 |
|---|---|
| 原notebook | HTTP下载448533字节，SHA-256 `fcb9a1004f61825f7169fbfef58e9896c8bdb38d20f2de3a2be21aad23d2ab1b` |
| 来源提交 | `7fbc29824171fd07dc2c30fe922e61ebc018ad1d` |
| 原单元/历史输出 | 说明版全部一致；注释版除新增注释外一致 |
| schema及Python AST | 全部通过；阅读版AST与原两格拼接一致 |
| 六条命令 | 去掉新增Bash注释后与原命令块逐字一致；未执行 |
| 三份notebook实际执行 | 原编号0、4均通过，Image宽度300、PNG333093字节 |
| 模型复现边界 | 未加载权重、未生成文本、未运行论文实验 |

实际环境：Windows，Python3.7.6、IPython7.12.0、nbformat5.0.4。只使用已有展示依赖，无GPU或模型依赖。图片来自配套原仓库的imgs目录。

下面是本次PowerShell执行命令，从公开仓库根目录执行；其他机器需替换Python路径。

```powershell
& 'D:/anaconda/python.exe' './lessons/01 LLaMA introduction/复现/scripts/download_original.py'
& 'D:/anaconda/python.exe' './lessons/01 LLaMA introduction/复现/scripts/build_annotated.py' --notebook './lessons/01 LLaMA introduction/代码/tutorials/01_llama_intro.ipynb' --output-dir './lessons/01 LLaMA introduction/代码/tutorials' --record './lessons/01 LLaMA introduction/复现/outputs/code_unit_coverage.json'
& 'D:/anaconda/python.exe' './lessons/01 LLaMA introduction/复现/scripts/verify_annotations.py' --directory './lessons/01 LLaMA introduction/代码/tutorials' --record './lessons/01 LLaMA introduction/复现/outputs/annotation_verification.json'
```

下载脚本只下载本讲固定来源文件；生成/验证脚本不联网，不启动模型。运行notebook时将内核工作目录设为本讲 `代码/tutorials`，才能读取配套图片。普通Python不会自动显示最后一个Image表达式，Python阅读版用于查看代码和说明。

机器可读记录仅作为补充：[下载记录](outputs/source_download.json)、[覆盖记录](outputs/code_unit_coverage.json)、[执行记录](outputs/annotation_verification.json)。上面的表格已列出结论，无需阅读JSON才能了解结果。所有阅读入口见[本讲目录](../README.md)。

## 2026-10-17：基础讲解与论文原图内嵌

本轮改写讲解，不改原notebook或已有注释代码。原论文v1 PDF已从 `https://arxiv.org/pdf/2302.13971v1` 实际取得，使用pypdfium2将第3页按3倍分辨率渲染，再直接裁取Table 2与Figure 1。两张图均已检查，表头、图例、坐标轴和边缘完整；没有重绘曲线或修改数值。截图离线保存在本讲 `图片/`，出处和页码直接写在笔记图下。

截图维护命令使用内置文档Python（安装有pypdfium2和Pillow），不依赖旧Python3.7环境：

```powershell
& 'C:/Users/hgfhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' './lessons/01 LLaMA introduction/复现/scripts/extract_paper_images.py' --pdf 'G:/zzz/Personal ChatGPT 前七讲/01 LLaMA introduction/资料/LLaMA_2302.13971v1.pdf'
```

其他机器替换解释器和PDF路径。原PDF保存在本地来源目录，公开笔记使用已保存PNG，因此阅读无需再次下载论文。[截图来源记录](outputs/论文截图来源.json)保存PDF和PNG的SHA256、版本、页码、裁剪坐标与脚本哈希。

正文softmax代码已实际执行，三个概率约为0.665241、0.244728、0.090031，总和为1。这个算例只用人为分数演示归一化，不包含模型forward或训练。原7B推理和论文训练依旧未复现。
