# OpenRLHF 历史提交 fbc1eab137c18651b28e1fa8d9b01c592e130b41 源码阅读
# Apache-2.0，见LICENSE-OpenRLHF。省略imports，仅作阅读，不启动Ray/DeepSpeed。

# ===== openrlhf-models-utils.py :: compute_approx_kl =====
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    # 中文说明：单样本log比，乘action_mask后返回；仍可能为负。
    log_ratio = log_probs - log_probs_base
    return log_ratio * action_mask


# ===== openrlhf-models-utils.py :: compute_reward =====
def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    kl_reward = -kl_coef * kl

    # 中文说明：该历史OpenRLHF将整句分数截断到[-10,10]，与TRL此处默认行为不同。
    r = r.clamp(min=-10, max=10)

    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    # 中文说明：反向查最后有效动作；全零mask没有合法终止动作，必须在外层保证。
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward
    return reward, kl


# ===== openrlhf-models-utils.py :: log_probs_from_logits =====
def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


# ===== openrlhf-models-utils.py :: masked_mean =====
def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
    else:
        return (tensor * mask).sum() / mask.sum()


# ===== openrlhf-loss.py :: PolicyLoss =====
class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss


# ===== openrlhf-loss.py :: ValueLoss =====
class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


# ===== openrlhf-experience_maker.py :: NaiveExperienceMaker.make_experience =====
@torch.no_grad()
def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
    self.actor.eval()
    self.critic.eval()
    # 中文说明：关闭dropout；eval不等于no_grad，是否记录图由外层上下文决定。
    self.initial_model.eval()
    # 中文说明：关闭dropout；eval不等于no_grad，是否记录图由外层上下文决定。
    self.reward_model.eval()
    # generate seq
    # 中文说明：原make_experience硬编码CUDA，CPU机器不执行此完整方法。
    inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
    sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
    num_actions = action_mask.size(1)
    # log probs
    # 中文说明：Actor对本轮rollout动作的旧log概率；之后放入Experience，更新阶段作为固定分母。
    action_log_probs = self.actor(sequences, num_actions, attention_mask)
    # init log probs
    # 中文说明：Actor对本轮rollout动作的旧log概率；之后放入Experience，更新阶段作为固定分母。
    base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
    # values
    value = self.critic(sequences, action_mask, attention_mask)
    # rewards
    # 中文说明：独立Reward Model输出每条序列标量；与逐token critic value区分。
    r = self.reward_model(sequences, attention_mask)
    reward, kl = compute_reward(
        r,
        self.kl_ctl.value,
        action_log_probs,
        base_action_log_probs,
        action_mask=action_mask,
    )
    advantage, returns = self.get_advantages_and_returns(
        value,
        reward,
        action_mask,
        generate_kwargs["gamma"],
        generate_kwargs["lambd"],
    )
    info = {
        "kl": masked_mean(kl, action_mask, dim=-1),
        "reward": r,
        "return": reward.sum(dim=-1),
        "response_length": action_mask.float().sum(dim=-1),
        "total_length": attention_mask.float().sum(dim=-1),
    }
    # reset model state
    self.actor.train()
    self.critic.train()
    # 中文说明：把本轮数据存入结构体供若干PPO epoch使用，不意味着无限复用陈旧经验。
    return Experience(
        sequences,
        action_log_probs,
        value,
        returns,
        advantage,
        attention_mask,
        action_mask,
        info,
    )


# ===== openrlhf-experience_maker.py :: NaiveExperienceMaker.get_advantages_and_returns =====
@torch.no_grad()
def get_advantages_and_returns(
    self,
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    lambd: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)
    # Mask invalid responses
    values = action_mask * values
    rewards = action_mask * rewards
    # 中文说明：GAE从右向左递推，用未来TD残差改善早期动作的信用分配。
    for t in reversed(range(response_length)):
        # 中文说明：序列最后位置采用终止bootstrap=0；不能直接套到被截断但未终止的任意环境。
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        # 中文说明：TD残差r_t+gamma*V_{t+1}-V_t；三项同shape[B]。
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        # 中文说明：累计delta_t+gamma*lambda*A_{t+1}；lambda控制偏差/方差折衷。
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # 中文说明：value回归目标用未白化GAE加旧value，之后才白化policy优势。
    returns = advantages + values
    return advantages.detach(), returns
