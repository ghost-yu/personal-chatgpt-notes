# -*- coding: utf-8 -*-
# 第09讲Python阅读版：原语句说明及历史错误边界见逐格内容。
# 主要用于阅读；远程语料、代理、Hub上传未在本次验证执行。

# %% [markdown]
# ## 单元 0（markdown）：数据辅助片段标题
#
# 仅探查本地路径和缓存，不是BPE算法或完整训练。

# %% [markdown] 原单元 0
# ## datasets

# %% [markdown]
# ## 单元 1（code）：Linux shell cd
#
# !是IPython shell语法，非普通Python；/media/whaow为讲师Linux路径。单次子shell cd通常不改变内核后续cwd，本次Windows跳过此命令。注释版保留命令，阅读版转换为等价get_ipython().system调用，需IPython环境。

# %% 原单元 1
# Linux shell cd；原执行语句保留，详细参数/shape/边界见本格说明。
# 原IPython命令：!cd /media/whaow/
get_ipython().system('cd /media/whaow/')

# %% [markdown]
# ## 单元 2（code）：数据加载与构建器接口
#
# load_dataset读数据，DownloadConfig控制下载，load_dataset_builder构造加载器并探查缓存。导入不加载课程codeparrot语料。

# %% 原单元 2
# 数据加载与构建器接口；原执行语句保留，详细参数/shape/边界见本格说明。
# 真实datasets接口；导入与下载/读取语料分开。
from datasets import load_dataset, DownloadConfig, load_dataset_builder

# %% [markdown]
# ## 单元 3（code）：全部注释的代理示例
#
# 原行全以#开头，不修改任何环境。HTTP/HTTPS/all_proxy分别对应代理配置，CURL_CA_BUNDLE空串是原示意，不能当成读者通用网络修复。

# %% 原单元 3
# 全部注释的代理示例；原执行语句保留，详细参数/shape/边界见本格说明。
# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
# os.environ['CURL_CA_BUNDLE'] = ''

# %% [markdown]
# ## 单元 4（code）：shell工作目录查询
#
# !pwd要求shell支持，Windows通常没有该命令；与内核Python cwd不是同一个概念。仅进行IPython语法转换AST核对，不执行Linux命令。

# %% 原单元 4
# shell工作目录查询；原执行语句保留，详细参数/shape/边界见本格说明。
# 原IPython命令：!pwd
get_ipython().system('pwd')

# %% [markdown]
# ## 单元 5（code）：探查原本地dataset缓存
#
# load_dataset_builder(./dataset/codeparrot/).cache_dir取得构建器缓存路径字符串，不等于已经加载完整dataset。原DownloadConfig与load_dataset行都是注释，路径和缓存历史属于讲师。本次原路径缺失故跳过，独立临时JSONL/json builder实际检查cache_dir属性。

# %% 原单元 5
# 探查原本地dataset缓存；原执行语句保留，详细参数/shape/边界见本格说明。
# config = DownloadConfig(delete_extracted=True)
# 仅构建加载器并取cache_dir，不代表数据全量加载。
load_dataset_builder('./dataset/codeparrot/').cache_dir
# load_dataset('./dataset/codeparrot', split='train')
# /media/whaow/.cache/huggingface/datasets/json/codeparrot-ed48b0160d7b840d/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51
