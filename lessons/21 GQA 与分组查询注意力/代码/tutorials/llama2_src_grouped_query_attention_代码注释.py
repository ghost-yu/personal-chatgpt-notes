# -*- coding: utf-8 -*-
# 原课程只导入Image并展示配图；核心官方代码见GQA核心源码_代码注释.py。

# %% 原代码单元 0
# 导入展示类，不加载模型或计算GQA。
from IPython.display import Image

# %% 原代码单元 2
# 展示原图：多个Q heads共享K/V，width只控制显示宽度。
Image('../imgs/gqa.png', width=600)
