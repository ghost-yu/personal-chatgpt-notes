# -*- coding: utf-8 -*-
# 第10讲原notebook纯文本阅读版；全部原语句不改，不含JSON历史输出。
# 从本文件所在代码/tutorials目录运行，才能解析../imgs相对路径。
# 原本只有图片；独立数据演示见../数据读取_教学补充.py。

# %% [markdown]
# ## 单元 0（code）：导入图片显示类
#
# `from IPython.display import Image`从模块导入Image类，后面调用构造器得到显示对象；Jupyter渲染对象中的PNG，不执行数据集采集/训练。导入本身不读文件，没有Tensor、model、loss或optimizer。

# %% 原单元 0
# 导入图片显示类；原语句与历史输出保留，新验证另记。
# Image属于IPython显示模块；导入不加载语料或模型。
from IPython.display import Image

# %% [markdown]
# ## 单元 1（markdown）：语料基础入口
#
# 本notebook是语料资料展示：英文、中文、C4、Pile四组图。需要从文本来源、清理、语言、规模和计数口径理解，而不是把图片当作已经load_dataset的数据对象。

# %% [markdown] 原单元 1
# ## basics

# %% [markdown]
# ## 单元 2（markdown）：英文来源标题
#
# English资源与后面的中文资源区分；有网页、百科、代码及混合语料，不能仅按“英文”认为质量/领域都相同。

# %% [markdown] 原单元 2
# ### 英文

# %% [markdown]
# ## 单元 3（code）：展示英文语料截图
#
# `Image`首个字符串指定相对当前工作目录的本地PNG；width=500为展示像素宽度，返回Image对象，不返回HF Dataset。相对路径../imgs/en-dataset.png要求从代码/tutorials打开；原图已放到相邻代码/imgs。图列English CommonCrawl、Wikipedia、C4、GitHub、BLOOM/ROOTS相关数据和RedPajama入口，混合了语料来源、模型名称和历史下载链接。图中LLaMA67%/15%等不能推广为所有模型的混合权重；BLOOM的350B tokens不是参数量。仅保留历史清单，未下载这些语料，也未核验每个旧地址如今仍能使用。详见主解释第2节。

# %% 原单元 3
# 展示英文语料截图；原语句与历史输出保留，新验证另记。
# 首个字符串为本地PNG相对路径；width=500只控制显示像素，不是样本/token数量。
# 构造器返回Image显示对象；Jupyter渲染它，本格不会调用load_dataset或训练。
Image('../imgs/en-dataset.png', width=500)

# %% [markdown]
# ## 单元 4（markdown）：中文来源标题
#
# 中文来源可用于预训练或任务训练，任务类别不能由语言名称直接推出。

# %% [markdown] 原单元 4
# ### 中文

# %% [markdown]
# ## 单元 5（code）：展示中文资源截图
#
# Image 读取 ../imgs/ch-dataset.png，width=500 设置展示尺寸。图中介绍 THUCNews 文本分类数据、清华 NLP 资源、中文 Wikipedia、WuDao、书籍和千言；阅读时分别关注资源入口、数据用途和规模统计口径。

# %% 原单元 5
# 展示中文资源截图；原语句与历史输出保留，新验证另记。
# 首个字符串为本地PNG相对路径；width=500只控制显示像素，不是样本/token数量。
# 构造器返回Image显示对象；Jupyter渲染它，本格不会调用load_dataset或训练。
Image('../imgs/ch-dataset.png', width=500)

# %% [markdown]
# ## 单元 6（markdown）：C4名称
#
# Colossal Clean Crawled Corpus：来自大规模网页抓取并清理的语料。Clean不能理解为每一份内容都可靠/无偏；不同版本和过滤决定分布。此格只有标题，没有清洗代码。

# %% [markdown] 原单元 6
# ## C4 (Colossal Clean Crawled Corpus)

# %% [markdown]
# ## 单元 7（markdown）：T5论文链接与来源关系
#
# 这条链接指向 T5 v3，其中介绍了 C4 的构建。下一张 c4.png 的统计表来自 Documenting Large Webtext Corpora Table 1，两份文献的相关内容在正文分别解释。

# %% [markdown] 原单元 7
# - https://arxiv.org/pdf/1910.10683v3.pdf
#     - 在 T5 这篇论文中首先提出；

# %% [markdown]
# ## 单元 8（code）：展示三种C4版本的统计
#
# 读取../imgs/c4.png，width500。表的行是C4.EN.NOCLEAN/NOBLOCKLIST/EN，列为document数、token数、压缩文件size。document定义是一个URL抓取文本；token用SpaCy English统计，不能直接等同GPT2 BPE；size是压缩JSON，不是RAM/RSS。图中EN为365million文档、156billion tokens、305GB；NOCLEAN2.3TB、NOBLOCKLIST380GB与过滤版本不同。表是规模统计，非模型准确率或本次测速结果。NOCLEAN仍有该文描述的英语识别步骤，未实现完整清洗过程。

# %% 原单元 8
# 展示三种C4版本的统计；原语句与历史输出保留，新验证另记。
# 首个字符串为本地PNG相对路径；width=500只控制显示像素，不是样本/token数量。
# 构造器返回Image显示对象；Jupyter渲染它，本格不会调用load_dataset或训练。
Image('../imgs/c4.png', width=500)

# %% [markdown]
# ## 单元 9（markdown）：Pile混合语料入口
#
# The Pile是一组多来源语料混合，包含网页、论文、代码、论坛等；本格不创建数据集或下载800GB。

# %% [markdown] 原单元 9
# ## The PILE

# %% [markdown]
# ## 单元 10（markdown）：Pile报告与GPT-J示意
#
# 这里列出 Pile 报告及 GPT-J/GPT-J-6B 的关联笔记，帮助理解预训练语料与使用这些语料的模型之间的关系。

# %% [markdown] 原单元 10
# - https://arxiv.org/pdf/2101.00027.pdf
# - GPT-J
#     - GPT-J-6B

# %% [markdown]
# ## 单元 11（code）：展示Pile组成表
#
# Image读取../imgs/pile.png并设置500像素显示宽。表列Component、Raw Size、Weight、Epochs、Effective Size、Mean Document Size；大小用GiB/KiB，不与GB混用。22来源Raw总825.18GiB，重复采样后的Effective1254.20GiB；PubMed Central raw90.27GiB、epochs2、effective180.55GiB、weight14.40%。weight近似按effective/总effective计算，不是raw/825.18。Epochs表示混合重复倍率，非要求磁盘复制两份；显示四舍五入有细小差异。均为报告图中统计，非本次I/O吞吐/RSS测量。

# %% 原单元 11
# 展示Pile组成表；原语句与历史输出保留，新验证另记。
# 首个字符串为本地PNG相对路径；width=500只控制显示像素，不是样本/token数量。
# 构造器返回Image显示对象；Jupyter渲染它，本格不会调用load_dataset或训练。
Image('../imgs/pile.png', width=500)

# %% [markdown]
# ## 单元 12（code）：保留原空代码格
#
# 原source为空；保留编号、元数据和原输出字段。注释版仅加说明性注释，AST仍为空；相关数据处理示例在正文单独展开。

# %% 原单元 12
# 保留原空代码格；原语句与历史输出保留，新验证另记。
# 原空单元没有数据逻辑；字幕缺失演示独立放在数据读取_教学补充.py。
