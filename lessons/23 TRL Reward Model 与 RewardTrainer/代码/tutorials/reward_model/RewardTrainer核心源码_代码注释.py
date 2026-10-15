# -*- coding: utf-8 -*-
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 固定TRL0.8.6完整RewardTrainer类、默认collator和compute_accuracy，只加中文注释。
# 精选对象依赖原模块的imports，不是可独立导入的完整模块；原算法包括margin广播边界均保留。

class RewardTrainer(Trainer):
    r"""
    The RewardTrainer can be used to train your custom Reward Model. It is a subclass of the
    `transformers.Trainer` class and inherits all of its attributes and methods. It is recommended to use
    an `AutoModelForSequenceClassification` as the reward model. The reward model should be trained on a dataset
    of paired examples, where each example is a tuple of two sequences. The reward model should be trained to
    predict which example in the pair is more relevant to the task at hand.

    The reward trainer expects a very specific format for the dataset. The dataset should contain two 4 entries at least
    if you don't use the default `RewardDataCollatorWithPadding` data collator. The entries should be named
    - `input_ids_chosen`
    - `attention_mask_chosen`
    - `input_ids_rejected`
    - `attention_mask_rejected`

    Optionally, you can also pass a `margin` entry to the dataset. This entry should contain the margin used to modulate the
    loss of the reward model as outlined in https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/.
    If you don't pass a margin, no margin will be used.
    """

    _tag_names = ["trl", "reward-trainer"]

    # 初始化真实Trainer子类，连接PEFT、collator、父类训练调度。
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`RewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`List[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            max_length (`int`, defaults to `None`):
                The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        # 兼容旧TrainingArguments入口并发弃用提示。
        if type(args) == TrainingArguments:
            warnings.warn(
                "Using `transformers.TrainingArguments` for `args` is deprecated and will be removed in a future version. Please use `RewardConfig` instead.",
                FutureWarning,
            )
            if max_length is not None:
                warnings.warn(
                    "The `max_length` argument is deprecated and will be removed in a future version. Please use the `RewardConfig` to set `max_length` instead.",
                    FutureWarning,
                )
        else:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )
            if max_length is not None and args.max_length is None:
                warnings.warn(
                    "The `max_length` argument is deprecated and will be removed in a future version. Please use the `RewardConfig` to set `max_length` instead.",
                    FutureWarning,
                )
        # 缺PEFT且要求LoRA时明确报错。
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        # 已安装PEFT时按配置包装模型。
        elif is_peft_available() and peft_config is not None:
            # 避免给已有PeftModel再注入一层。
            if not isinstance(model, PeftModel):
                # 量化模型才走kbit训练准备；FP32小模型不走该分支。
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`."
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    # 准备量化训练/冻结/必要dtype，非本次CPU量化验证。
                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                # 真正按LoRA配置修改模型并管理可训练参数。
                model = get_peft_model(model, peft_config)

        # 默认评估使用成对选择准确率，不是绝对分数阈值。
        if compute_metrics is None:
            compute_metrics = compute_accuracy

        # 没提供自定义collator则创建偏好对专用补齐器。
        if data_collator is None:
            # 默认collator需要分词器的PAD规则。
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default RewardDataCollatorWithPadding"
                )
            # 兼容旧TrainingArguments入口并发弃用提示。
            if type(args) == TrainingArguments:
                if max_length is None:
                    warnings.warn(
                        "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
                        " It will be set to `512` by default, but you should do it yourself in the future.",
                        UserWarning,
                    )
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    warnings.warn(
                        "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
                        " It will be set to `512` by default, but you should do it yourself in the future.",
                        UserWarning,
                    )
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            # chosen/rejected分别动态padding。
            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            # 普通Trainer列删除机制不适用于四个偏好字段。
            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        # 调用父Trainer初始化，继承优化器/loader/保存等调度。
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    # RewardTrainer重写成对损失；不使用模型内labels回归loss。
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        # 共享评分模型对chosen侧做前向，返回[B,1]原始score。
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            # 要求可按logits键读取结构化模型返回。
            return_dict=True,
        )["logits"]
        # 同一个模型对rejected侧前向，共享全部参数。
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            # 要求可按logits键读取结构化模型返回。
            return_dict=True,
        )["logits"]
        # calculate loss, optionally modulate with margin
        # 可选margin支路；此固定版collator给[B]，需留意与[B,1]的广播。
        if "margin" in inputs:
            # 先做两侧score差，稳定logsigmoid后对批次取平均。
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            # 先做两侧score差，稳定logsigmoid后对批次取平均。
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # True时同时返回loss和两侧原score，不是词表logits。
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    # 评估路径包装输入、关闭梯度，输出每对两候选的概率。
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 父Trainer负责把Tensor放到目标device。
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # 评估不构建反向图，但仍需模型eval控制dropout。
        with torch.no_grad():
            # 复用pairwise loss并取得两侧score。
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum to 1
        # [2,B,1]→均值[2,B]→沿候选softmax→转置[B,2]。
        logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T

        # 每对chosen在第0列，因此标签全0。
        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels


@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    # collator对象可当函数调用，输入若干单行dict。
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # chosen侧单独收集input_ids和mask。
        features_chosen = []
        # rejected侧单独收集，两侧padding长度可以不同。
        features_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        # 仅看第一行是否有margin；混合缺字段不安全。
        has_margin = "margin" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if has_margin:
                margin.append(feature["margin"])
        # 按chosen当前批最长长度补齐。
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # 按rejected当前批最长长度补齐。
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        if has_margin:
            # 此版生成[B]，与[B,1]相减会广播成[B,B]，本讲单独验证。
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch


# 评估[B,2]的argmax是否等于全0标签，平分时可能误导。
def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    if np.array(predictions[:, 0] == predictions[:, 1], dtype=float).sum() > 0:
        warnings.warn(
            f"There are {np.array(predictions[:, 0] == predictions[:, 1]).sum()} out of {len(predictions[:, 0])} instances where the predictions for both options are equal. As a consequence the accuracy can be misleading."
        )
    # 选择分数较高的候选；完全相等时argmax偏向第一列。
    predictions = np.argmax(predictions, axis=1)

    # 得到本批候选选择准确率，不代表人类总体认可率。
    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}
