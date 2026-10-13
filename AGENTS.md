# 本项目工作约定

用户要求完成 Personal ChatGPT 所列第 01–30 讲，每次完成一篇。每讲按 paper-code-deep-reading skill 生成中文解释，核对源代码版本、公式与 shape，记录实际验证及未复现边界。

用户已授权将本仓库公开在 GitHub，并要求每次完成一篇都提交。仓库为 https://github.com/ghost-yu/personal-chatgpt-notes ，分支 main。完成新讲后更新 README，运行相关教学验证及 check_documents.py，然后为该讲创建独立 commit 并 push。不要把仅提交本地当作完成公开更新。

第 01–07 讲此前已完成，已作为初始历史提交。第 08 讲新增独立提交。第 09 讲已完成小语料训练；第 10 讲已完成本地 Datasets/Arrow 验证；第 11 讲已完成梯度数学/静态验证；第 12 讲已完成 SFT/packing 数学与源码 fixture 验证；第 13 讲已完成梯度一致性/激活重算/静态验证；第 14 讲已完成真实 tokenizer/样本/配置验证；第 15 讲已完成官方 completion 包装与真实 tokenizer 验证（生成器 fixture）；第 16 讲已完成重复 prompt 边界/真实 tokenizer/PPO 数学验证；第 17 讲已完成 RMSNorm/SwiGLU 数学与梯度、官方 forward fixture 验证；第 18 讲已完成主 notebook 与官方 precompute 的真实 CPU PyTorch 验证；第 19 讲主文、全部15个代码单元说明、本地带说明ipynb与真实CPU PyTorch验证已完成；第20讲以后仍待完成，不能只看状态文件就认定文档或验证已完成。

公开仓库发布教学解释、自己编写的脚本与可公开的验证记录。第三方资料使用来源链接；原始字幕附件、完整第三方源码、图片快照与模型权重保留在本地配套目录。各篇原始 notebook 单元编号从 0 开始。

用户补充明确要求解释每个ipynb的全部代码。必须提供按原始编号的代码单元解释.md、本地原代码_带说明.ipynb，逐个覆盖import、配置、辅助函数、绘图、注释和空单元，解释用途、API、变量/shape、输出及缺口。插入说明后原代码和历史输出必须不变。第19讲已通过15/15覆盖与schema核对。第01–18讲仍待逐篇核验补齐，下一次先回补第01讲，之后完成前18讲再继续20–30讲；每篇仍须独立commit和push。进度见代码覆盖进度.md，不能只依据主文已完成列表判定全部交付完成。
