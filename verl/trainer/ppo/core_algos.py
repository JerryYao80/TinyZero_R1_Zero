# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

"""
PPO算法的核心函数实现
本文件中实现的函数应该被不同分布式策略的训练器用来实现PPO算法
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    自适应KL控制器
    论文来源：https://arxiv.org/pdf/1909.08593.pdf
    
    用于动态调整KL散度系数，以保持策略更新的稳定性
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        """
        初始化自适应KL控制器
        
        Args:
            init_kl_coef: 初始KL系数
            target_kl: 目标KL散度值
            horizon: 调整周期
        """
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """
        更新KL系数
        
        Args:
            current_kl: 当前KL散度值
            n_steps: 当前步数
        """
        target = self.target
        # 计算比例误差，并限制在[-0.2, 0.2]范围内
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        # 根据误差和步数调整系数
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """
    固定KL控制器
    使用固定的KL系数，不进行动态调整
    """

    def __init__(self, kl_coef):
        """
        初始化固定KL控制器
        
        Args:
            kl_coef: 固定的KL系数
        """
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """
        更新函数（对于固定控制器，不进行任何操作）
        """
        pass


def get_kl_controller(config):
    """
    根据配置获取KL控制器
    
    Args:
        config: 配置对象
    
    Returns:
        KLController: KL控制器实例
    """
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """
    计算广义优势估计（GAE）和回报值
    改编自：https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: 形状为(bs, response_length)的张量，表示每个token的奖励
        values: 形状为(bs, response_length)的张量，表示每个状态的价值估计
        eos_mask: 形状为(bs, response_length)的张量，表示序列结束标记的掩码
        gamma: 折扣因子
        lam: GAE-Lambda参数

    Returns:
        advantages: 形状为(bs, response_length)的张量，表示计算得到的优势值
        returns: 形状为(bs, response_length)的张量，表示计算得到的回报值
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # 从后向前计算GAE
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # 计算时序差分误差
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            # 计算GAE
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # 计算回报值并对优势值进行标准化
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    计算GRPO（Generalized Reward-Proportional Optimization）的优势值
    仅适用于结果监督，即每个响应只有一个标量奖励的情况
    
    Args:
        token_level_rewards: 形状为(bs, response_length)的张量，表示每个token的奖励
        eos_mask: 形状为(bs, response_length)的张量，表示序列结束标记的掩码
        index: 形状为(bs,)的张量，表示每个样本的索引
        epsilon: 用于数值稳定性的小常数
    
    Returns:
        scores: 形状为(bs, response_length)的张量，表示标准化后的分数
        scores: 同上，作为回报值返回
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    # 为每个索引收集分数
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        # 收集每个索引对应的所有分数
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算每个索引的分数统计信息
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 如果只有一个分数，使用默认值
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # 计算均值和标准差
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # 标准化分数
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """
    计算带KL惩罚的奖励
    
    Args:
        token_level_scores: token级别的分数
        old_log_prob: 旧策略的对数概率
        ref_log_prob: 参考策略的对数概率
        kl_ratio: KL散度的权重系数
    
    Returns:
        rewards: 计算得到的奖励
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """
    计算PPO的策略损失
    改编自：https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: 形状为(bs, response_length)的张量，表示旧策略的对数概率
        log_prob: 形状为(bs, response_length)的张量，表示新策略的对数概率
        advantages: 形状为(bs, response_length)的张量，表示优势值
        eos_mask: 形状为(bs, response_length)的张量，表示序列结束标记的掩码
        cliprange: PPO中使用的裁剪范围，参见https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: 标量张量，表示计算得到的策略梯度损失
        pg_clipfrac: 浮点数，表示被裁剪的策略梯度损失的比例
        ppo_kl: PPO算法中的KL散度
    """
    # 计算近似KL散度
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    # 计算PPO的双重损失
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # 取两个损失中的较大值作为最终损失
    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    # 计算被裁剪的比例
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """
    计算分类熵损失
    
    Args:
        logits: 形状为(bs, response_length, vocab_size)的张量，表示模型输出的logits
        eos_mask: 形状为(bs, response_length)的张量，表示序列结束标记的掩码
    
    Returns:
        entropy_loss: 标量张量，表示计算得到的熵损失
    """
    # 计算熵
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """
    计算价值函数损失
    来源：https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds: 形状为(batch_size, response_length)的张量，表示价值头预测的值
        values: 形状为(batch_size, response_length)的张量，表示价值头的旧值
        returns: 形状为(batch_size, response_length)的张量，表示真实回报
        eos_mask: 形状为(bs, response_length)的张量，表示序列结束标记的掩码
        cliprange_value: 价值函数的裁剪范围

    Returns:
        vf_loss: 标量张量，表示价值函数损失
        vf_clipfrac: 浮点数，表示被裁剪的价值函数的比例
    """
    # 计算裁剪后的价值预测
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    # 计算两种损失
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    # 取较大的损失作为最终损失
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    # 计算被裁剪的比例
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """
    计算KL散度惩罚
    来源：https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    
    Args:
        logprob: 当前策略的对数概率
        ref_logprob: 参考策略的对数概率
        kl_penalty: KL惩罚类型
    
    Returns:
        kl: 计算得到的KL散度惩罚
    """
    if kl_penalty == 'kl':
        return logprob - ref_logprob
    elif kl_penalty == 'abs':
        return torch.abs(logprob - ref_logprob)
    elif kl_penalty == 'mse':
        return 0.5 * (logprob - ref_logprob)**2
    elif kl_penalty == 'none':
        return torch.zeros_like(logprob)
    else:
        raise NotImplementedError
