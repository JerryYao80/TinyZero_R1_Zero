# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
PPO训练主程序
注意：我们没有将main与ray_trainer合并，因为ray_trainer被其他main程序使用。
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    """
    根据数据源选择相应的奖励计算函数
    
    Args:
        data_source: 数据源名称
    Returns:
        compute_score: 对应的奖励计算函数
    """
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    else:
        raise NotImplementedError


class RewardManager():
    """
    奖励管理器类
    负责计算和管理强化学习中的奖励值
    """

    def __init__(self, tokenizer, num_examine) -> None:
        """
        初始化奖励管理器
        
        Args:
            tokenizer: 分词器，用于解码模型输出
            num_examine: 需要打印到控制台的批次数量
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        """
        计算奖励值
        
        Args:
            data: 数据协议对象，包含批次数据
        Returns:
            reward_tensor: 计算得到的奖励张量
        """
        # 如果数据中已有奖励分数，直接返回
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # 创建奖励张量，初始化为0
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # 记录已打印的数据源
        already_print_data_sources = {}

        # 遍历每个数据项计算奖励
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # 获取提示词ID
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # 获取响应ID
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # 解码完整序列
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # 获取真实值
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # 选择并计算奖励分数
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            # 打印示例输出
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    """
    主函数入口
    初始化Ray分布式环境
    
    Args:
        config: hydra配置对象
    """
    if not ray.is_initialized():
        # 初始化本地Ray集群
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    """
    主要训练任务
    设置训练环境并启动训练流程
    
    Args:
        config: 训练配置
    """
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # 打印初始配置
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # 从HDFS下载检查点
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # 实例化分词器
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # 根据策略定义工作器类
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        # FSDP（完全分片数据并行）策略
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        # Megatron策略
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # 定义角色与工作器的映射关系
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    # 设置资源池
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # 配置奖励模型
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # 创建奖励管理器实例
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    # 创建资源池管理器
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # 创建并初始化PPO训练器
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
