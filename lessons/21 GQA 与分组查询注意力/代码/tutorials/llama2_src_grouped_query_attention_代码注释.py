# -*- coding: utf-8 -*-
# 原课程只导入Image并展示配图；核心官方代码见GQA核心源码_代码注释.py。

# %% 原代码单元 0
# 导入Image类，供后面的GQA配图展示使用。
from IPython.display import Image

# %% 原代码单元 2
# 展示原图：多个Q heads共享K/V，width只控制显示宽度。
Image('../imgs/gqa.png', width=600)
