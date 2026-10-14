# -*- coding: utf-8 -*-
# 第15讲 llama2_src_text_chat Python阅读版；原语句保留。

# %% 0
# 接口对照提纲
# 【必须会】补充notebook直接展示tokenizer ID、接口类型与特殊标签。
# ## outline

# %% 1
# 同一模型包装与共同generate
# 【必须会】Markdown构造片段：Tokenizer返回词表对象，Transformer接受ModelArgs，load_state_dict加载权重，Llama保存两对象引用。strict=False不保证shape错误可忽略；text/chat共享generate，权重与prompt组织不同。片段缺少checkpoint等上下文，不能当完整加载脚本执行。
# - 两者底层调的都是
#     - `Llama` 类对象的模型结构显然是一致的；
#         ```
# # 源码介绍片段；所需对象/类型上下文和每行用途见前置说明。
#         tokenizer = Tokenizer(model_path=tokenizer_path)
#         model = Transformer(model_args)
#         model.load_state_dict(checkpoint, strict=False)
#         llama = Llama(model, tokenizer)
#         ```
#     - `Llama.generate(prompt_tokens, max_gen_len, ...)`
# - 差异
#     - model weights 不同；
#     - prompt_tokens 的组织不同

# %% 2
# 真实SentencePiece词表加载
# 【必须会】SentencePieceProcessor是类，model_file指当前目录tokenizer.model；返回sp_model实例，不加载LLM权重。本次ASCII临时目录复制同一词表后，原格整格执行。
# tokenizer
# 导入真实词表处理类，不能代替语言模型。
from sentencepiece import SentencePieceProcessor
# 当前工作目录需对应词表；返回真实SentencePieceProcessor。
sp_model = SentencePieceProcessor(model_file='./tokenizer.model')

# %% 3
# BOS ID
# 【必须会】实例方法bos_id()返回整数1，句子开始符；原生encode默认不自动添加它。
# 返回开始符整数1；不是生成句子。
sp_model.bos_id()

# %% 4
# EOS ID
# 【必须会】eos_id()返回整数2，结束符；它不等于pad。
# 返回结束符整数2，与pad不同。
sp_model.eos_id()

# %% 5
# pad ID
# 【必须会】pad_id()返回-1，表示此词表没有合法pad项；不能把-1当Embedding的有效词表ID。原生成buffer以它作未填标记。
# 返回-1，词表未定义合法pad ID。
sp_model.pad_id()

# %% 6
# 接口参数章节
# 【必须会】先看输入/返回类型，再追包装与generate。
# ## 接口层面

# %% 7
# text/chat方法签名
# 【必须会】self是Llama实例；prompts是list[str]，dialogs是list[list[Message]]。温度默认.6/top_p.9；max_gen_len None在方法中变max_seq_len-1；logprobs False。text另有echo False，chat无echo参数。两个def只有签名而没有body，原Markdown是介绍片段，保留不补成假实现，不能独立编译执行。
# ```
# # 方法签名只有声明，没有函数体；参数/返回类型见前置单元说明。
# def text_completion(
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     self,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     prompts: List[str],
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     temperature: float = 0.6,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     top_p: float = 0.9,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     max_gen_len: Optional[int] = None,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     logprobs: bool = False,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     echo: bool = False,
# ) -> List[CompletionPrediction]:
#
#
# # 方法签名只有声明，没有函数体；参数/返回类型见前置单元说明。
# def chat_completion(
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     self,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     dialogs: List[Dialog],
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     temperature: float = 0.6,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     top_p: float = 0.9,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     max_gen_len: Optional[int] = None,
#     # 输入或生成配置字段；默认值保留，完整行为见主解释第5–8节。
#     logprobs: bool = False,
# ) -> List[ChatPrediction]:
# ```

# %% 8
# chat参数结构
# 【必须会】一条dialog含可变多条消息，batch是dialog条数。
# ## chat 接口抽象

# %% 9
# 消息类型与保留标签
# 【必须会】Message继承TypedDict描述role/content字典；Dialog=List[Message]只是类型别名，不执行模型。B_INST/E_INST为模板字符串，B_SYS/E_SYS含换行；SPECIAL_TAGS做字面子串检查。UNSAFE_ERROR是包装层固定错误，不是模型学会拒答。role需先定义，源码类型说明与运行校验分开。
# ```
# # 源码介绍片段；所需对象/类型上下文和每行用途见前置说明。
# class Message(TypedDict):
#     role: Role
#     content: str
#
#
# Dialog = List[Message]
#
# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#
# SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
# UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."
#
# ```
#
# - role
#     - system
#     - user
#     - assistant
#     - "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
# - unsafe requests
#     - dialog content 中包含 `"[INST]", "[/INST]", "<<SYS>>\n", "\n<</SYS>>\n\n"`

# %% 10
# 展示system分隔符换行
# 【建议会】print打印前一个换行、结束标签、后两个换行及print自己的终止换行；不是新的system消息。
# 字符串含模板换行，print再追加一个换行。
print("\n<</SYS>>\n\n")

# %% 11
# 特殊标签分词章节
# 【必须会】标签是可分词字符串，不能因尖括号或方括号就当成一个special token。
# ### special tags

# %% 12
# 编码INST开始标签
# 【必须会】sp_model.encode返回list[int]，默认整数ID，不默认BOS/EOS；[INST]由多个普通词表片段表示。
# 返回普通片段ID列表，未自动添加BOS/EOS。
sp_model.encode('[INST]')

# %% 13
# 编码普通单词
# 【必须会】what为字符串，分词结果是ID列表，不是词向量。
# 普通词字符串编码为list[int]，不是embedding。
sp_model.encode('what')

# %% 14
# 编码INST结束标签
# 【必须会】[/INST]与开始标签的分词长度不必相同，不能机械设固定标签宽度。
# 结束标签也可能是多个token。
sp_model.encode('[/INST]')

# %% 15
# 编码完整instruction字符串
# 【必须会】只分词模板字符串；没有调用LLM，也没有BOS/EOS自动追加。要生成需官方Tokenizer.encode的明确边界与completion包装。
# '[INST] what is the recipe of mayonnaise? [/INST]'
# 只编码整个instruction字符串，不调用generate或模型。
sp_model.encode('[INST] what is the recipe of mayonnaise? [/INST]')
