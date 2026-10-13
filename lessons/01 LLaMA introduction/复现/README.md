# 第 01 讲注释与执行核对

状态：**部分成功**。原 notebook 的9格全部说明，2/2 Python单元加入中文注释；六条shell启动命令逐条说明但未运行。未加载LLaMA权重或复现论文实验。

`scripts/build_annotated.py` 读取原版和本讲说明文档，生成19格的两份 notebook：说明版保留全部原单元，注释版另外给两格Python插入中文注释。`scripts/verify_annotations.py` 验证schema、历史输出、原单元一致性和注释前后的Python AST，并在真实IPython环境执行两格，比较三份文件的执行结果。原执行计数及输出保留，不冒充新运行结果。

实际环境：Windows，Python3.7.6、IPython7.12.0、nbformat5.0.4。只使用已有展示依赖，无GPU或模型依赖。图片来自配套原仓库的imgs目录。

下面是本次PowerShell执行命令，路径为本机配套资料位置；其他机器需要替换路径。不要把Bash续行反斜杠复制到PowerShell。

```powershell
& 'D:/anaconda/python.exe' './scripts/build_annotated.py' --notebook 'G:/zzz/Personal ChatGPT 前七讲/01 LLaMA introduction/代码/tutorials/01_llama_intro.ipynb' --output-dir 'G:/zzz/Personal ChatGPT 前七讲/01 LLaMA introduction/代码/tutorials' --record './outputs/code_unit_coverage.json'
& 'D:/anaconda/python.exe' './scripts/verify_annotations.py' --directory 'G:/zzz/Personal ChatGPT 前七讲/01 LLaMA introduction/代码/tutorials' --record './outputs/annotation_verification.json'
```

命令工作目录为本文件所在的复现目录。两个脚本要求显式传入原notebook目录，不会自动下载或启动模型。生成记录见[覆盖记录](outputs/code_unit_coverage.json)，真实执行见[验证记录](outputs/annotation_verification.json)。每份执行成功单元为0、4；Image宽度300，PNG333093字节。

原notebook固定入口见[来源清单](../来源清单.json)。本公开仓库发布自编说明和生成/验证脚本，完整第三方代码与图片及生成的完整notebook保留在本地配套目录。
