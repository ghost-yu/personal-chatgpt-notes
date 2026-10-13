# 本项目工作约定

用户要求完成 Personal ChatGPT 所列第 01–30 讲，每次完成一篇。每讲按 paper-code-deep-reading skill 生成中文解释，核对源代码版本、公式与 shape，记录实际验证及未复现边界。

用户已授权将本仓库公开在 GitHub，并要求每次完成一篇都提交。仓库为 https://github.com/ghost-yu/personal-chatgpt-notes ，分支 main。完成新讲后更新 README，运行相关教学验证及 check_documents.py，然后为该讲创建独立 commit 并 push。不要把仅提交本地当作完成公开更新。

第 01–07 讲此前已完成，已作为初始历史提交。第 08 讲新增独立提交。第 09 讲已完成小语料训练；第 10 讲已完成本地 Datasets/Arrow 验证；第 11 讲已完成梯度数学/静态验证；第 12 讲已完成 SFT/packing 数学与源码 fixture 验证；第 13 讲已完成梯度一致性/激活重算/静态验证；第 14 讲已完成真实 tokenizer/样本/配置验证；第 15 讲已完成官方 completion 包装与真实 tokenizer 验证（生成器 fixture）；第 16 讲已完成重复 prompt 边界/真实 tokenizer/PPO 数学验证；第 17 讲已完成 RMSNorm/SwiGLU 数学与梯度、官方 forward fixture 验证；第 18 讲已完成主 notebook 与官方 precompute 的真实 CPU PyTorch 验证；第 19 讲主文、全部15个代码单元说明、本地带说明ipynb与真实CPU PyTorch验证已完成；第20讲以后仍待完成，不能只看状态文件就认定文档或验证已完成。

公开仓库发布教学解释、自己编写的脚本与可公开的验证记录。第三方资料使用来源链接；原始字幕附件、完整第三方源码、图片快照与模型权重保留在本地配套目录。各篇原始 notebook 单元编号从 0 开始。

用户补充要求解释每个ipynb的全部代码，并在代码中加入中文注释。交付代码单元解释.md、本地原代码_带说明.ipynb和代码注释.ipynb。全部import、配置、辅助函数、绘图、注释和空单元都覆盖；重要Markdown命令也解释。说明版原单元和历史输出不变；代码注释版仅加注释，核对AST、schema及实际可行的执行。第01讲全部9格及2/2中文代码注释已验证；第02讲全部16格、7/7代码注释及Markdown13伪代码注释、真实图片展示、CPU LoRA数学梯度/冻结验证已完成；第19讲15/15说明版完成但代码内中文注释待补；第03讲47格说明、37/37代码注释、原单元/历史输出/IPython转换AST/schema及真实CPU PEFT/BLOOM/Dataset/collator/Trainer两步验证已完成；第04讲18格说明、8/8代码注释及6/6 Markdown源码块注释、原单元/历史输出/AST/schema及直接导入固定官方v0.3.0的CPU注入/引用/冻结/前向/梯度/adapter状态核对完成；05–18仍待审计。下一篇第05讲，补齐05–19后继续20–30，每篇独立commit和push。代码覆盖进度.md是覆盖审计入口，不能根据主文列表认定全部交付完成。
