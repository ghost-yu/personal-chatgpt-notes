# -*- coding: utf-8 -*-
# Copyright 2023-present the HuggingFace Inc. team.
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

# PEFT0.12.0完整核心对象摘录，只添加说明；保留原库的重复缩放行为。
# 这是源码阅读文件，缺少原模块完整imports与其他方法，不作为独立可执行模块。

# 来源：mapping.py；完整方法 inject_adapter_in_model
# 低层入口：配置、原模型、adapter名；在原模型中注入，返回原模型。
def inject_adapter_in_model(
    peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = "default"
) -> torch.nn.Module:
    r"""
    A simple API to create and inject adapter in-place into a model. Currently the API does not support prompt learning
    methods and adaption prompt. Make sure to have the correct `target_names` set in the `peft_config` object. The API
    calls `get_peft_model` under the hood but would be restricted only to non-prompt learning methods.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the Peft model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
    """
    if peft_config.is_prompt_learning or peft_config.is_adaption_prompt:
        raise ValueError("`create_and_replace` does not support prompt learning and adaption prompt yet.")

    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING.keys():
        raise ValueError(
            f"`inject_adapter_in_model` does not support {peft_config.peft_type} yet. Please use `get_peft_model`."
        )

    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]

    # By instantiating a peft model we are injecting randomly initialized LoRA layers into the model's modules.
    peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name)

    return peft_model.model


class LoraModel:
    # 以下方法完整保留在原owner名下；其余方法依赖原模块。
    # 来源：tuners/lora/model.py；完整方法 _check_add_weighted_adapter
    # 检查目标集合/来源rank，决定最终算法、结果rank和目标层集合。
    def _check_add_weighted_adapter(
        self, adapters: list[str], combination_type: str, svd_rank: int | None
    ) -> tuple[str, int, str]:
        """
        Helper function to check if the arguments to add_weighted_adapter are valid and compatible with the underlying
        model.
        """
        # 按来源列表检查每个adapter的目标集合。
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # If more than one of the adapters targets the same module with modules_to_save, raise an error, as these
        # modules cannot be merged. First, find the ModulesToSaveWrapper instances in the model, then check if they
        # have modules for the adapters to be merged.
        modules_to_save_wrappers = [module for module in self.modules() if isinstance(module, ModulesToSaveWrapper)]
        problematic_wrappers = [
            wrapper
            for wrapper in modules_to_save_wrappers
            if sum(adapter in wrapper.modules_to_save for adapter in adapters) > 1
        ]
        if problematic_wrappers:
            raise ValueError(
                "Cannot add weighted adapters if they target the same module with modules_to_save, but found "
                f"{len(problematic_wrappers)} such instance(s)."
            )

        # if there is only one adapter, we can only use linear merging
        # 选择具体合并算法，本次是svd。
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or "
                    "dare_linear."
                )
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type.endswith("svd"):
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] == str:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        elif target_module_types[0] == set:
            new_target_modules = reduce(
                operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        return combination_type, new_rank, new_target_modules


    # 来源：tuners/lora/model.py；完整方法 add_weighted_adapter
    # LoraModel方法：创建新适配器参数，返回None，不自动切换激活项。
    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "svd",
        svd_rank: int | None = None,
        svd_clamp: int | None = None,
        svd_full_matrices: bool = True,
        svd_driver: str | None = None,
        density: float | None = None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """

        # 已有同名adapter则直接返回，避免静默以为新权重已重算。
        if adapter_name in list(self.peft_config.keys()):
            return

        # 校验来源存在和rank要求；单来源会转linear。
        combination_type, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type=combination_type,
            svd_rank=svd_rank,
        )

        # 复制配置，结果rank和alpha一起设为new_rank，使普通新缩放为1。
        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            # LoRA中间秩，决定A/B中间轴和新增参数量。
            r=new_rank,
            # 普通LoRA缩放分子alpha，与r相除得到支路系数。
            lora_alpha=new_rank,
            # 指定注入目标名字，不能只按打印文本猜目标路径。
            target_modules=new_target_modules,
        )
        # 为所有目标层分配新A/B；没有在这里调用set_adapter。
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        # 冻结新建adapter，不等于把它激活。
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            # 只在真实LoRA层操作，不遍历任意Python对象。
            if isinstance(target, LoraLayer):
                # 重复执行前先检查是否已有同名结果。
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                # 写入新适配器A；源adapter不在此行被改写。
                target_lora_A.data = target_lora_A.data * 0.0
                # 写入新适配器B；.data是这里官方源码的绕过autograd赋值方式。
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        # cat把w*s吸收到A，避免产生不同来源的交叉项。
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        # B按列拼接，配合A按行拼接精确表示加权增量。
                        loras_B.append(current_adapter_lora_B.data)

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
                    loras_A = torch.cat(loras_A, dim=0)
                    loras_B = torch.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A.shape[0], :] = loras_A
                    target_lora_B.data[:, : loras_B.shape[1]] = loras_B
                elif combination_type in [
                    "svd",
                    "ties_svd",
                    "dare_linear_svd",
                    "dare_ties_svd",
                    "magnitude_prune_svd",
                ]:
                    target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(
                        combination_type,
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        density,
                        majority_sign_method,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )
                elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
                    target_lora_A.data, target_lora_B.data = self._generalized_task_arithmetic_weighted_adapter(
                        combination_type, adapters, weights, target, density, majority_sign_method
                    )


    # 来源：tuners/lora/model.py；完整方法 _svd_generalized_task_arithmetic_weighted_adapter
    # 逐目标层合成矩阵再分解；本版本valid_weights与get_delta_weight均含s。
    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        valid_adapters = []
        # 这里已经把合并系数乘s；与后面已缩放delta叠加会重复。
        valid_weights = []
        is_embedding = any(adapter in target.lora_embedding_A for adapter in adapters)
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                # 此固定版本先乘源scaling；下方get_delta_weight已含s，导致再次缩放。
                valid_weights.append(weight * target.scaling[adapter])

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
        # 每个返回增量已经乘过源s，Embedding还已经转到[V,H]。
        delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
        # 这里已经把合并系数乘s；与后面已缩放delta叠加会重复。
        valid_weights = torch.tensor(valid_weights).to(delta_weight[0].device)
        if combination_type == "svd":
            # 逐元素加权求和；纯svd不使用density。
            delta_weight = task_arithmetic(delta_weight, valid_weights)
        elif combination_type == "ties_svd":
            delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "dare_linear_svd":
            delta_weight = dare_linear(delta_weight, valid_weights, density)
        elif combination_type == "dare_ties_svd":
            delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "magnitude_prune_svd":
            delta_weight = magnitude_prune(delta_weight, valid_weights, density)
        else:
            raise ValueError(f"Invalid value passed to combination type: {combination_type}")

        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if (hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out) or is_embedding:
            # Embedding转回[H,V]或转置存储的Linear对齐B@A方向。
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        # 返回奇异向量/奇异值；full=True在大矩形词表上很占内存。
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        # 保留左奇异向量前r列；切片通常是view。
        U = U[:, :new_rank]
        # 保留前r个最大奇异值。
        S = S[:new_rank]
        # 把奇异值吸收进左因子；此后U不再正交。
        U = U @ torch.diag(S)
        # 保留右奇异向量前r行，对应新A。
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        # 返回A=Vh前r行和B=U前r列乘Σ，顺序与函数赋值一致。
        return Vh, U


    # 来源：tuners/lora/model.py；完整方法 _generalized_task_arithmetic_weighted_adapter
    # 直接分别组合A/B；linear会引入不同来源的交叉乘积。
    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        # account weights for LoRA A and B layers.
        # 这里已经把合并系数乘s；与后面已缩放delta叠加会重复。
        valid_weights = []
        lora_A_deltas = []
        lora_B_deltas = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A:
                current_adapter_lora_A = target.lora_A[adapter].weight
                current_adapter_lora_B = target.lora_B[adapter].weight
            elif adapter in target.lora_embedding_A:
                current_adapter_lora_A = target.lora_embedding_A[adapter]
                current_adapter_lora_B = target.lora_embedding_B[adapter]
            else:
                continue
            # linear对A/B各用sqrt(w*s)，需w*s非负，并产生交叉项。
            valid_weights.append(math.sqrt(weight * target.scaling[adapter]))
            lora_A_deltas.append(current_adapter_lora_A.data)
            lora_B_deltas.append(current_adapter_lora_B.data)
        # 这里已经把合并系数乘s；与后面已缩放delta叠加会重复。
        valid_weights = torch.tensor(valid_weights).to(lora_A_deltas[0].device)
        lora_deltas = [lora_A_deltas, lora_B_deltas]
        dtype = lora_A_deltas[0].dtype
        for i, task_tensors in enumerate(lora_deltas):
            if combination_type == "linear":
                lora_deltas[i] = task_arithmetic(task_tensors, valid_weights)
            elif combination_type == "ties":
                lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "dare_linear":
                lora_deltas[i] = dare_linear(task_tensors, valid_weights, density)
            elif combination_type == "dare_ties":
                lora_deltas[i] = dare_ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "magnitude_prune":
                lora_deltas[i] = magnitude_prune(task_tensors, valid_weights, density)
            else:
                raise ValueError("Invalid combination type")
        lora_deltas = [delta.to(dtype) for delta in lora_deltas]
        return lora_deltas


    # 来源：tuners/lora/model.py；完整方法 merge_and_unload
    # 把选定或当前活跃增量加到基础W并移除adapter包装，区别于创建merge。
    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )


class Linear:
    # 以下方法完整保留在原owner名下；其余方法依赖原模块。
    # 来源：tuners/lora/layer.py；完整方法 get_delta_weight
    # 层方法返回与基础weight同shape的增量，已经包含scaling。
    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        # 记录设备字符串；本行没有移动模型。
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # get_delta_weight在这里乘源adapter的scaling；继续沿调用链查看后续是否再次缩放。
        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor


    # 来源：tuners/lora/layer.py；完整方法 forward
    # 实例被model(input_ids)调用时执行此方法。
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            # 计算冻结基础支路；已folded时不再重复加LoRA分支。
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            # 计算冻结基础支路；已folded时不再重复加LoRA分支。
            result = self.base_layer(x, *args, **kwargs)
        else:
            # 计算冻结基础支路；已folded时不再重复加LoRA分支。
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            # 只遍历当前激活项，注册多个adapter不表示全部相加。
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                # 本例两组源adapter缩放均为16/8=2。
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # 先A降维、B升维、乘s，再加基础输出；只对活跃且存在的adapter执行。
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result


class Embedding:
    # 以下方法完整保留在原owner名下；其余方法依赖原模块。
    # 来源：tuners/lora/layer.py；完整方法 get_delta_weight
    # 层方法返回与基础weight同shape的增量，已经包含scaling。
    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        # 记录设备字符串；本行没有移动模型。
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # get_delta_weight在这里乘源adapter的scaling；继续沿调用链查看后续是否再次缩放。
        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor


    # 来源：tuners/lora/layer.py；完整方法 forward
    # 实例被model(input_ids)调用时执行此方法。
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            # 计算冻结基础支路；已folded时不再重复加LoRA分支。
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            # 计算冻结基础支路；已folded时不再重复加LoRA分支。
            result = self.base_layer(x, *args, **kwargs)
        else:
            # 计算冻结基础支路；已folded时不再重复加LoRA分支。
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            # 只遍历当前激活项，注册多个adapter不表示全部相加。
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                # 本例两组源adapter缩放均为16/8=2。
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result
