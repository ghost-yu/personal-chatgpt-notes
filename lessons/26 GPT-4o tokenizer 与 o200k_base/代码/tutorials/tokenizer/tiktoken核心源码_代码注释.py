# -*- coding: utf-8 -*-
# OpenAI tiktoken0.7.0完整核心对象摘录，仅添加中文注释。MIT，见同目录LICENSE-tiktoken。
# 源码阅读文件，依赖原模块的imports/全局对象，不作为独立可执行模块。

# tiktoken/registry.py：完整get_encoding
# 注册表按名字查找并缓存Encoding，首次构造才读资源。
def get_encoding(encoding_name: str) -> Encoding:
    # 缓存命中直接返回对象，不重新下载。
    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    # 可重入锁保护并发注册/构造。
    with _lock:
        # 缓存命中直接返回对象，不重新下载。
        if encoding_name in ENCODINGS:
            return ENCODINGS[encoding_name]

        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None

        if encoding_name not in ENCODING_CONSTRUCTORS:
            raise ValueError(
                f"Unknown encoding {encoding_name}. Plugins found: {_available_plugin_modules()}"
            )

        # 取编码名对应的构造函数，不是模型类。
        constructor = ENCODING_CONSTRUCTORS[encoding_name]
        # 把构造器的dict作为关键字参数展开到Encoding。
        enc = Encoding(**constructor())
        # 缓存本次构造的实例。
        ENCODINGS[encoding_name] = enc
        # 返回Encoding对象，不是编码后的ID。
        return enc


# tiktoken_ext/openai_public.py：完整cl100k_base
# 定义旧词表构造器：真实rank、正则与5个特殊ID。
def cl100k_base():
    # 读取真实bytes→rank字典，下载数据需符合expected_hash。
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    )
    # 这里是真正控制符ID映射，不是异常中文普通token。
    special_tokens = {
        ENDOFTEXT: 100257,
        FIM_PREFIX: 100258,
        FIM_MIDDLE: 100259,
        FIM_SUFFIX: 100260,
        ENDOFPROMPT: 100276,
    }
    return {
        "name": "cl100k_base",
        "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


# tiktoken_ext/openai_public.py：完整o200k_base
# 定义新词表构造器：199998普通rank与2个特殊ID。
def o200k_base():
    # 读取真实bytes→rank字典，下载数据需符合expected_hash。
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        expected_hash="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    )
    # 这里是真正控制符ID映射，不是异常中文普通token。
    special_tokens = {
        ENDOFTEXT: 199999,
        ENDOFPROMPT: 200018,
    }
    # This regex could be made more efficient
    # 预切分正则，和词表共同决定最终token数。
    pat_str = "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )
    return {
        "name": "o200k_base",
        "pat_str": pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


class Encoding:
    # tiktoken/core.py：完整__init__
    # 构造对象；星号后的参数必须以关键字提供。
    def __init__(
        self,
        name: str,
        *,
        pat_str: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
        explicit_n_vocab: Optional[int] = None,
    ):
        """Creates an Encoding object.

        See openai_public.py for examples of how to construct an Encoding object.

        Args:
            name: The name of the encoding. It should be clear from the name of the encoding
                what behaviour to expect, in particular, encodings with different special tokens
                should have different names.
            pat_str: A regex pattern string that is used to split the input text.
            mergeable_ranks: A dictionary mapping mergeable token bytes to their ranks. The ranks
                must correspond to merge priority.
            special_tokens: A dictionary mapping special token strings to their token values.
            explicit_n_vocab: The number of tokens in the vocabulary. If provided, it is checked
                that the number of mergeable tokens and special tokens is equal to this number.
        """
        self.name = name

        self._pat_str = pat_str
        self._mergeable_ranks = mergeable_ranks
        self._special_tokens = special_tokens

        # 在普通/特殊token的ID里取最大值，不计算有效项数。
        self.max_token_value = max(
            max(mergeable_ranks.values()), max(special_tokens.values(), default=0)
        )
        # 只有显式指定时才检查词表项数与max ID的一致性。
        if explicit_n_vocab:
            assert len(mergeable_ranks) + len(special_tokens) == explicit_n_vocab
            assert self.max_token_value == explicit_n_vocab - 1

        # 真正的Rust CoreBPE实现，Python这里只封装参数与错误处理。
        self._core_bpe = _tiktoken.CoreBPE(mergeable_ranks, special_tokens, pat_str)


    # tiktoken/core.py：完整encode
    # 实例方法返回list[int]；默认禁止直接编码特殊控制符文本。
    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.

        Special tokens are artificial tokens used to unlock capabilities from a model,
        such as fill-in-the-middle. So we want to be careful about accidentally encoding special
        tokens, since they can be used to trick a model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token. This can be controlled on a per-token level using the `allowed_special`
        and `disallowed_special` parameters. In particular:
        - Setting `disallowed_special` to () will prevent this function from raising errors and
          cause all text corresponding to special tokens to be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat all text
          corresponding to special tokens to be encoded as special tokens.

        ```
        >>> enc.encode("hello world")
        [31373, 995]
        >>> enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        [50256]
        >>> enc.encode("<|endoftext|>", allowed_special="all")
        [50256]
        >>> enc.encode("<|endoftext|>")
        # Raises ValueError
        >>> enc.encode("<|endoftext|>", disallowed_special=())
        [27, 91, 437, 1659, 5239, 91, 29]
        ```
        """
        # all先展开为该Encoding真正的特殊token集合。
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        # 默认all排除已允许集合，其余匹配时抛ValueError。
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special
        # 默认all排除已允许集合，其余匹配时抛ValueError。
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            # 海象运算符保存正则匹配对象并判断是否存在。
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())

        # https://github.com/PyO3/pyo3/pull/3632
        if isinstance(allowed_special, frozenset):
            allowed_special = set(allowed_special)

        try:
            # 交给真实Rust BPE；不是argmax或模型采样。
            return self._core_bpe.encode(text, allowed_special)
        # 此处只处理Unicode输入错误，不是捕获无效ID的Rust panic。
        except UnicodeEncodeError:
            # BPE operates on bytes, but the regex operates on unicode. If we pass a str that is
            # invalid UTF-8 to Rust, it will rightfully complain. Here we do a quick and dirty
            # fixup for any surrogate pairs that may have sneaked their way into the text.
            # Technically, this introduces a place where encode + decode doesn't roundtrip a Python
            # string, but given that this is input we want to support, maybe that's okay.
            # Also we use errors="replace" to handle weird things like lone surrogates.
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            # 交给真实Rust BPE；不是argmax或模型采样。
            return self._core_bpe.encode(text, allowed_special)


    # tiktoken/core.py：完整decode
    # 把ID列表先解为bytes再UTF8解码，默认替换无效片段。
    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        """Decodes a list of tokens into a string.

        WARNING: the default behaviour of this function is lossy, since decoded bytes are not
        guaranteed to be valid UTF-8. You can control this behaviour using the `errors` parameter,
        for instance, setting `errors=strict`.

        ```
        >>> enc.decode([31373, 995])
        'hello world'
        ```
        """
        # 先拼全部token bytes再decode，不能逐token损失后再拼。
        return self._core_bpe.decode_bytes(tokens).decode("utf-8", errors=errors)


    # tiktoken/core.py：完整decode_single_token_bytes
    # 安全取得单ID原始字节；无效ID抛KeyError。
    def decode_single_token_bytes(self, token: int) -> bytes:
        """Decodes a token into bytes.

        NOTE: this will decode all special tokens.

        Raises `KeyError` if the token is not in the vocabulary.

        ```
        >>> enc.decode_single_token_bytes(31373)
        b'hello'
        ```
        """
        # 不做UTF8转换，因此不会因半个字符而自动出现替换字符。
        return self._core_bpe.decode_single_token_bytes(token)


    # tiktoken/core.py：完整n_vocab
    # 以属性语法访问n_vocab，不能写成n_vocab()。
    @property
    # 返回max_token_value+1，洞位也占数值区间。
    def n_vocab(self) -> int:
        """For backwards compatibility. Prefer to use `enc.max_token_value + 1`."""
        # cl100277/o200019，并非有效词条总数。
        return self.max_token_value + 1


# tiktoken/_educational.py：完整bpe_encode
# 官方教育版纯Python BPE，展示同一rank优先合并过程。
def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: Optional[str] = "colour"
) -> list[int]:
    # 初始每个UTF8字节单独一段，不是每个Unicode字符一段。
    parts = [bytes([b]) for b in input]
    # 每轮选一个可合并相邻对，直到不存在候选。
    while True:
        # See the intermediate merges play out!
        if visualise:
            if visualise in ["colour", "color"]:
                visualise_tokens(parts)
            elif visualise == "simple":
                print(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        # 按位置遍历相邻片段，不能跨预切分边界。
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            # 查拼接字节的rank；无词表项返回None。
            rank = mergeable_ranks.get(pair[0] + pair[1])
            # 选最小rank；编码时不是重新统计当前文本频率。
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        # 没有合法合并时结束，不要求最终只剩一个token。
        if min_rank is None:
            # 退出当前for，不退出整个notebook。
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        # 只合并选中的邻接片段，其余顺序保留。
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    if visualise:
        print()

    # 最终每片对应rank/ID，返回普通列表。
    tokens = [mergeable_ranks[part] for part in parts]
    return tokens
