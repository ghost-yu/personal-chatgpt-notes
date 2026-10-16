# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

# 导入依赖；不等于下载或构造模型。
import os
# 从模块导入指定对象；后续调用名由此绑定。
from logging import getLogger
# 从模块导入指定对象；后续调用名由此绑定。
from pathlib import Path
# 从模块导入指定对象；后续调用名由此绑定。
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

# 导入依赖；不等于下载或构造模型。
import tiktoken
# 从模块导入指定对象；后续调用名由此绑定。
from tiktoken.load import load_tiktoken_bpe


logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


# TypedDict只作类型提示，不自动进行运行时角色合法性检查。
class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


# 官方分词器类，负责字符串与ID之间的可逆映射。
class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    # 256是特殊token总数，不是普通词表大小。
    num_reserved_special_tokens = 256

    # 正则先切分文本，再对各段做字节BPE；保留原表达式。
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    # 实例初始化；self是该对象，不是输入张量。
    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        # 从原文件加载bytes→rank的128000项字典。
        mergeable_ranks = load_tiktoken_bpe(model_path)
        # 特殊ID从普通词表末端128000起排。
        num_base_tokens = len(mergeable_ranks)
        # 列表顺序确定特殊ID，不能任意排序或少加reserved项。
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        # 建立特殊字符串到唯一整数ID的映射。
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        # 真正的tiktoken.Encoding，底层实现不是本脚本伪造。
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        # 总词表128256，普通与特殊ID合计。
        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        # 文首标记128000，和换行、角色标记不同。
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        # 全文终止标记128001。
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        # -1是padding哨兵，正式forward前须被有效ID替换。
        self.pad_id: int = -1
        # 全文EOS和轮次EOT均可终止生成。
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    # bos/eos为无默认的仅关键字参数，返回list[int]。
    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        # 本版本要求精确str类型；类型注解不代替断言。
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        # 分批处理超长字符串，避免旧tiktoken大输入异常。
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        # 限制连续空白/非空白段长度，不等于模型上下文长度。
        MAX_NO_WHITESPACES_CHARS = 25_000

        # 嵌套生成器按400k字符和连续段限制延迟产出字符串。
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        # 收集输出ID；可变list仅属于本次调用。
        t: List[int] = []
        for substr in substrs:
            # 追加各段编码，或追加消息部分，不创建batch张量。
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        # 可选在最前插入一次BOS。
        if bos:
            # 列表首插128000，不是文本字符串拼接。
            t.insert(0, self.bos_id)
        # 可选追加全文EOS。
        if eos:
            # 追加结束标记ID。
            t.append(self.eos_id)
        # 返回普通Python列表，后续才转int64张量。
        return t

    # 接受ID序列，返回Unicode字符串。
    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        # cast只提供类型提示；实际交给tiktoken解码。
        return self.model.decode(cast(List[int], t))

    # 静态辅助函数无self，不读取实例状态。
    @staticmethod
    # yield把函数变为字符串片段生成器。
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        # 记录当前连续空白或非空白段字符数。
        current_slice_len = 0
        # 空串单独处理，避免访问s[0]越界。
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        # 保存下个输出片段起点索引。
        slice_start = 0

        # 按字符索引扫描，不按字节或token索引。
        for i in range(len(s)):
            # Unicode isspace判定当前字符是否空白。
            is_now_space = s[i].isspace()

            # ^为布尔异或，空白属性变化时重置段计数。
            if current_slice_is_space ^ is_now_space:
                # 记录当前连续空白或非空白段字符数。
                current_slice_len = 1
                # 空串单独处理，避免访问s[0]越界。
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                # 只在连续同类字符超限后切分。
                if current_slice_len > max_consecutive_slice_len:
                    # 惰性返回片段，调用方继续迭代后再从此处恢复。
                    yield s[slice_start:i]
                    # 保存下个输出片段起点索引。
                    slice_start = i
                    # 记录当前连续空白或非空白段字符数。
                    current_slice_len = 1
        # 惰性返回片段，调用方继续迭代后再从此处恢复。
        yield s[slice_start:]


# 官方对话模板对象，复用Tokenizer组装角色头和EOT。
class ChatFormat:
    # 实例初始化；self是该对象，不是输入张量。
    def __init__(self, tokenizer: Tokenizer):
        # 保存传入分词器引用，多个消息共享同一ID映射。
        self.tokenizer = tokenizer

    # 角色头：start_header、角色ID、end_header、两个换行。
    def encode_header(self, message: Message) -> List[int]:
        # 从空list开始组装当前头部或对话。
        tokens = []
        # 按协议顺序追加特殊ID。
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        # 把普通文本编码或已有列表展开追加。
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        # 按协议顺序追加特殊ID。
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        # 把普通文本编码或已有列表展开追加。
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        # 返回普通Python列表，后续才转int64张量。
        return tokens

    # 头部+strip后的正文+EOT；不自动插入BOS。
    def encode_message(self, message: Message) -> List[int]:
        # 复用角色头方法，减少格式分歧。
        tokens = self.encode_header(message)
        # 把普通文本编码或已有列表展开追加。
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        # 按协议顺序追加特殊ID。
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        # 返回普通Python列表，后续才转int64张量。
        return tokens

    # 整段对话仅首部BOS，每条消息EOT，末尾assistant头。
    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        # 从空list开始组装当前头部或对话。
        tokens = []
        # 按协议顺序追加特殊ID。
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        # 保持system/user/assistant输入顺序，不自动校验交替。
        for message in dialog:
            # 把普通文本编码或已有列表展开追加。
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        # 把普通文本编码或已有列表展开追加。
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        # 返回普通Python列表，后续才转int64张量。
        return tokens
