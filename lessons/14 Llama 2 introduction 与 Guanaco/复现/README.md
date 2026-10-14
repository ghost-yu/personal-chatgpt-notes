# 第 14 讲复现记录

状态：**部分成功**。23格说明、16代码格及Markdown15三命令注释完整交付；原字节、元数据、历史输出、AST和schema通过。新增真实CPU运行覆盖Fast tokenizer、pipeline、完整固定TRL示例两步LoRA及adapter保存重载。随机小LLaMA和固定Guanaco前五行不代表原7B量化训练或论文评估。

## 1. 环境与材料

Windows、Python3.11.0、torch2.0.1+cpu、Transformers4.31.0、TRL0.5.0、PEFT0.4.0、Accelerate0.21.0。直接依赖见[requirements](requirements.txt)，完整版本见[环境快照](outputs/environment_freeze.txt)。CPU torch的CUDA不可用，没有bitsandbytes/xformers。原7B FP16权重约12.55GiB，超过本机单卡8GiB，没有下载对应权重。随机模型和adapter在ASCII临时目录构造，结束后释放。

| 输入 | 固定来源 | SHA256 |
|---|---|---|
| llama2_tokenizer.model | 课程固定提交tutorials/llama/llama2_tokenizer.model | 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347 |
| guanaco-five-rows.json | Guanaco revision 831dabac2283d99420cda0b673d7a2a43849f17a，train前五行 | 0c59149204ecf17b600fa8bfc59a6308e7abc1f985e8ef0efa6ae302360d3561 |
| sft-script-v0.5.0.py | 官方TRL v0.5.0 examples/scripts/sft_trainer.py | e75c36b75daa03445ef9e763847b8e1450b561fbc222f345e1daa65f46687aec |

外部完整源码快照和字幕保留本地。verifier先核对三个hash，不能用最新版替代。五行JSON是UTF8、indent2、CRLF、无末尾换行，保持早期证据；新原notebook为LF Git blob。

## 2. 重做方式

从仓库根目录运行，独立建立Python3.11环境：

```powershell
py -3.11 -m venv "$env:TEMP/lesson14-env"
$lessonPython = "$env:TEMP/lesson14-env/Scripts/python.exe"
& $lessonPython -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.0.1+cpu"
& $lessonPython -m pip install -r "lessons/14 Llama 2 introduction 与 Guanaco/复现/requirements.txt"
& $lessonPython -m pip check
```

三个输入可按上表固定来源获取到仓库外的缓存目录；以下标准库代码是材料取回方式，本次训练使用已有hash匹配的本地材料，不将这段代码当成已经执行的下载记录。网络/hash失败时停止，不放宽检查。

```python
from pathlib import Path
from urllib.request import urlopen
import json, hashlib, os

folder = Path(os.environ['TEMP']) / 'lesson14-assets'
folder.mkdir(exist_ok=True)
commit = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
revision = '831dabac2283d99420cda0b673d7a2a43849f17a'
items = [
    ('llama2_tokenizer.model', 'https://raw.githubusercontent.com/chunhuizhang/personal_chatgpt/' + commit + '/tutorials/llama/llama2_tokenizer.model', '9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'),
    ('sft-script-v0.5.0.py', 'https://raw.githubusercontent.com/huggingface/trl/v0.5.0/examples/scripts/sft_trainer.py', 'e75c36b75daa03445ef9e763847b8e1450b561fbc222f345e1daa65f46687aec'),
]
for name, url, expected in items:
    with urlopen(url, timeout=45) as response:
        data = response.read()
    assert hashlib.sha256(data).hexdigest() == expected, name
    (folder / name).write_bytes(data)
url = 'https://huggingface.co/datasets/timdettmers/openassistant-guanaco/resolve/' + revision + '/openassistant_best_replies_train.jsonl'
with urlopen(url, timeout=45) as response:
    rows = [json.loads(response.readline()) for _ in range(5)]
data = json.dumps(rows, ensure_ascii=False, indent=2).replace('\n', '\r\n').encode('utf-8')
assert hashlib.sha256(data).hexdigest() == '0c59149204ecf17b600fa8bfc59a6308e7abc1f985e8ef0efa6ae302360d3561'
(folder / 'guanaco-five-rows.json').write_bytes(data)
```

下载原notebook、生成派生版、真实验证：

```powershell
& $lessonPython "lessons/14 Llama 2 introduction 与 Guanaco/复现/scripts/download_original.py"
& $lessonPython "lessons/14 Llama 2 introduction 与 Guanaco/复现/scripts/build_annotated.py"
& $lessonPython "lessons/14 Llama 2 introduction 与 Guanaco/复现/scripts/verify_delivery.py" --tokenizer "$env:TEMP/lesson14-assets/llama2_tokenizer.model" --samples "$env:TEMP/lesson14-assets/guanaco-five-rows.json" --sft-script "$env:TEMP/lesson14-assets/sft-script-v0.5.0.py"
python "工具/check_documents.py"
```

本机原notebook用download_original.py的 `--upstream-clone` 读取已下载clone的固定Git blob，未声称新的HTTP下载。实际运行命令：

```powershell
& "G:/zzz/Personal ChatGPT 系列/共享运行环境/llama2-sft/Scripts/python.exe" "lessons/14 Llama 2 introduction 与 Guanaco/复现/scripts/verify_delivery.py" --tokenizer "G:/zzz/Llama 入门教程/代码/personal_chatgpt/tutorials/llama/llama2_tokenizer.model" --samples "G:/zzz/Personal ChatGPT 系列/14 Llama 2 introduction 与 Guanaco/资料/guanaco-five-rows.json" --sft-script "G:/zzz/Personal ChatGPT 系列/14 Llama 2 introduction 与 Guanaco/资料/sft-script-v0.5.0.py"
```

## 3. 结果与失败记录

- [注释审计](outputs/annotation_verification.json)：23格、16代码格、两47格派生版，3schema/AST、原语句/元数据/历史输出、Markdown15三命令语义通过。
- [原件来源](outputs/source_download.json)：19803bytes、固定Git blob，SHA256 b27440095f45a608b79666bf967276621f256ab252e7b156b758e22b73435ab6。旧来源清单的CRLF工作副本hash保留并注明口径。
- [真实运行报告](outputs/runtime_verification.json)：两版各13原格整格执行，6为本地FP32/无device_map变体，18为真实json/data_files五行变体，代理5跳过。两版生成/预处理精确一致；原prompt、max_length200、三候选、top_k10保留。随机模型基础参数1,026,608。
- 完整固定TRL脚本通过runpy执行，覆盖本地路径、batch1、累积2、max_steps2、无鉴权/远程代码、无4/8bit。真实SFTTrainer更新两步，q/v rank64、alpha16、4096可训练参数，B非零、基础权重不变、optimizer仅adapter。没有模拟API或只抽AST。
- Fast长度407/468/775/715/544，截断后407/468/512/512/512；collator前两行[2,468]、61pad。Human标签未屏蔽，独立构造的真实EOS被pad=eos规则屏蔽。32有效token的模型shift与手算loss精确一致。
- adapter四个tensor保存重载后状态精确一致、logits差0。互斥4/8bit实际报错；meta结构核对7B逻辑参数，不分配权重。

首轮验证在训练两步和保存后失败：左padding使截取前32位labels全为-100，CE mean为NaN。修正为有效token区取32位验证shift，保留全ignore负例；没有改变训练数据或padding_side。修正后完整verifier退出0。TRL左padding、旧AdamW/use_auth_token弃用及可选xformers提示作为环境行为记录，不为消除提示安装GPU组件。

## 4. 早期证据与未运行范围

[教学验证.json](outputs/教学验证.json) 和 [verify_llama2.py](scripts/verify_llama2.py) 保留：早期Python3.12.14/SentencePiece0.2.0词表、BOS/EOS、原创文本往返、五行角色/长度、AST默认值和参数/步数推导。它没有运行pipeline或SFTTrainer，新证据不扩大旧报告范围。

未运行原7B/base/chat权重、bitsandbytes/CUDA4-bit、完整9846行、原924更新、五小时耗时、语言质量或论文完整实验。随机输出和两步loss不证明Guanaco微调质量。作者修改稿/精确依赖、原权重revision及全量截断分布未确认；固定参考版本成功不证明最新版CLI兼容。
