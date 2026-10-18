# 第26讲复现记录

## 2026-10-18：基础讲解与论文截图

正文从字符/字节/ID讲到BPE学习与固定rank、真实编码、文本压缩、原全文计数、UTF-8片段和词表遍历。此前解释逐字保存在末尾折叠栏。六段正文Python直接由验证器提取执行；结果见[基础讲解验证.json](outputs/基础讲解验证.json)，含当前解释与脚本SHA。

| 项目 | 条件与结果 |
|---|---|
| 环境 | Python3.12，完整tiktoken0.7.0与regex2026.9.29；源文件同时与来源清单、本地原源码、发行包RECORD核对 |
| 词表加载 | 两份完整公开词表按官方URL的SHA1缓存键复制到临时缓存；get_encoding沿真实注册、读取、摘要验证与Rust构造路径运行 |
| 字节/BPE | 你好为6bytes；abc的ab优先为[256,99]，bc优先为[97,256]；官方教育函数与真实Rust一致 |
| 原英文/中文 | 英文两编码各5token；指定中文34/24；AIM旧1新2；数字三位分段与前导空格检查 |
| 原全文路径 | 原第11格AST仅open的路径实参不同；原文件SHA保留，4020行、文本模式727731字符；逐行990508/703303，全文989904/702799 |
| 独立全文重计 | 重新按文件迭代器读取、每256行encode_batch，完整累计与正文相同；两种口径相差604/504 |
| 完整词表 | 正文遍历全部200019个o200k候选ID；独立按普通/特殊词表成员集合枚举两套有效ID，排序/筛选逐项相同 |
| 词表统计 | cl/o有效100261/200000、空位16/19、单独非完整UTF-8的有效token773/1562；含Han普通词条840/7534，最长项ID与长度记在JSON |
| 短样例对照 | 十种输入×两套编码，实际正则切段后完整官方Python bpe_encode与Rust逐ID相同；字节拼接及整句解码还原原输入 |
| 原观察与规则 | 原15格字符串直接从AST提取，o普通177431、cl12个token；原符号正则AST相同，abc/123/中文通过not条件，纯符号被排除 |
| 特殊标记/emoji | 默认拒绝、显式允许、作为普通文本三分支；组合emoji逐token显示有替换字符，整体字节还原正确 |

截图来自ACL 2016 BPE原PDF第4页Figure1与Algorithm1，以4倍比例裁剪并逐张查看。两图与中文解释直接嵌入正文；[论文截图来源.json](outputs/论文截图来源.json)保存页码、裁剪坐标与PDF/PNG摘要。本地PDF为 `G:/zzz/Personal ChatGPT 系列/26 GPT-4o tokenizer 与 o200k_base/原论文.pdf`。论文讲字符子词合并，正文另外说明实际tiktoken的字节BPE及预切分。

重做本轮：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/26 GPT-4o tokenizer 与 o200k_base/复现/scripts/verify_foundation_examples.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/工具/check_documents.py'
```

沿用下文LESSON26_LOCAL、LESSON26_TIKTOKEN_LIBS、LESSON26_EXTRA_LIBS环境变量可更换依赖路径。本轮未重跑历史langdetect全量扫描或其裸except/panic路径，原报告保留；当前验证使用完整词表、指定异常类型及独立成员枚举。本讲模型训练原因、GPT-4o回答与付费API不在本轮运行中。

## 此前交付记录

本讲只需要CPU，不加载模型，不调用GPT-4o API，无token费用。原课公开tokenizer操作可验证，原文训练过滤因果解释与历史GPT回答不在已证实范围。

## 环境与来源

Windows11、Python3.12.14、tiktoken0.7.0、langdetect1.0.9，regex/nbformat/requests实际版本在 [验证报告](outputs/delivery_verification.json)。本讲未使用Torch；共享CPU Python仅作解释器。tiktoken及其依赖复用第25讲独立runtime-libs，langdetect新装在本讲runtime-libs，未修改共享环境。

上游固定7fbc29824171fd07dc2c30fe922e61ebc018ad1d。两notebook及2177760bytes原《西游记》按字节发布，含历史输出和原换行。两套完整rank表也随代码发布，验证不依赖网络下载。原词表站点HTTPS握手失败，HTTP返回400；从tiktoken-rs镜像取得文件，全部bytes的SHA256严格等于官方tiktoken0.7.0构造器expected_hash：

| 资源 | bytes | SHA256 |
|---|---:|---|
| cl100k_base.tiktoken | 1681126 | 223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7 |
| o200k_base.tiktoken | 3613922 | 446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d |
| xyj.txt | 2177760 | 52709f91a86a76fc408178638c2b7cbeaa6f00cf892a1dd7ebcd2b79fde44411 |

库的core/registry/load/_educational/openai_public及许可证与已安装0.7.0发行RECORD校验吻合，完整快照保留本地。另用系统curl从官方0.7.0标签获取openai_public.py，文本与安装版相同；文件换行不同，不声称原始字节相同。没有冒用GitHub连接器请求失败的响应作为源码。

未找到独立字幕。原微信正文访问失败；第三方gist与官方2024发布页已经核对，未将gist评论/分享回答当训练证据。

## 实际命令

工作目录G:/zzz/personal-chatgpt-notes：

~~~powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/26 GPT-4o tokenizer 与 o200k_base/复现/scripts/build_delivery.py' --upstream 'G:/zzz/Llama 入门教程/代码/personal_chatgpt' --tiktoken-root 'G:/zzz/Personal ChatGPT 系列/25 Llama 3 introduction/资料/runtime-libs' --local 'G:/zzz/Personal ChatGPT 系列/26 GPT-4o tokenizer 与 o200k_base/资料'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/26 GPT-4o tokenizer 与 o200k_base/复现/scripts/verify_delivery.py'
~~~

builder读取固定Git blob，生成派生文件、来源与完整对象摘录。verifier核对发布字节、schema、AST/token后创建临时cache，把已发布两词表复制成tiktoken所需的URL SHA1文件名；两个独立子进程分别执行原/注释全部代码。requests请求被明确禁止，若缓存不完整会失败，不能偷偷下载或调用API。

原./xyj.txt相对路径以代码/tutorials/tokenizer为cwd。两版各25代码格原样执行，无CPU语句替换或跳过，包括空格。langdetect.DetectorFactory.seed=26是外置重现上下文，不改notebook语句。两份字典扫描摘要、原显示/打印和语料计数逐项比较；原生Rust stderr另捕获计数与摘要，避免把它混成Python打印输出。

全量扫描中的langdetect对大量短串分类会耗时；本次没有把range(n_vocab)缩小，也没把检测结果替换成固定名单。原历史名单的精确随机状态未知，不要求新名单与历史完全相同。

## 换电脑重做

按requirements在独立Python环境安装依赖，再设置：

- LESSON26_UPSTREAM：含上游固定提交的Git仓库，默认现有本地路径。
- LESSON26_TIKTOKEN_LIBS：含tiktoken包及tiktoken-0.7.0.dist-info的父目录。
- LESSON26_EXTRA_LIBS：含langdetect包的父目录，可与上一个变量指向同一专用环境site-packages。
- LESSON26_LOCAL：本地资料目录，包含官方标签openai_public_remote.py和tiktoken-source快照；builder会从发行文件生成后者。

源仓库获取示例：

~~~powershell
git clone https://github.com/chunhuizhang/personal_chatgpt.git C:/work/lesson26-upstream
~~~

从官方标签https://raw.githubusercontent.com/openai/tiktoken/0.7.0/tiktoken_ext/openai_public.py保存openai_public_remote.py到本地资料目录。builder还需资料/tiktoken-cache目录内两份rank文件：文件名分别为官方完整URL的SHA1，即9b5ad71b2ce5302211f9c61530b329a4922fc6a4与fb374d419588a4632f3f557e76b4b70aebbca790。可从仓库发布的两.tiktoken复制并改名；builder仍会核对官方SHA256。其余完整源码保留本地，不上传整套外部仓库。

## 核对边界

主21格/15代码，补充12格/10代码，两种派生版分别43格与25格；总6份notebook schema。所有原metadata、execution_count和历史输出保留，说明版原格完全相同。注释版只增加Python注释，AST和非注释token相同；两Python阅读版AST一致，核心源码9个完整函数/方法AST与token核对。

原扫描会触发19+16+19=54次无效ID原生panic，每版单独记录；bare except使执行继续，不把它写成“没有错误”。PanicException属于BaseException而非Exception，本次隔离探测验证。独立安全扫描用decode_single_token_bytes捕获KeyError、UTF8 strict捕获UnicodeDecodeError，排除真正特殊ID后才生成含Han字的普通token榜，不替代原实验。

最终验证通过：两版各25格原代码，无替换/无跳过；所有显示/打印、三份按长度排序字典的摘要精确相同。每版真实捕获54次洞位panic。逐行990508/703303完全复现历史；727731字符全文一次编码为989904/702799，分别少604/504。安全扫描合法ID100261/200000，洞位16/19，单独严格UTF8解码失败773/1562项；这与ID无效是不同问题。

真实ASCII反例AIM/AMY/APO在cl均1token、o均2token；普通样例中教育版BPE与Rust逐ID相同，整句bytes拼接/解码正确，特殊策略、数字/空格、正则过滤边界及空语言检测错误均通过。完整观察值和三脚本SHA见验证报告。

未复现GPT-4o生成、翻译、embedding充分训练、数据清洗流水线、训练语料分布、tokenizer训练、Transformer训练、当前价格与完整请求计费。下一讲27。
