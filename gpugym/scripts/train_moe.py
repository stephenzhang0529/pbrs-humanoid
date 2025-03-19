"""
训练混合专家(MoE)系统的脚本
该脚本会加载预训练的行走和奔跑模型，并训练门控网络选择最合适的专家
"""

import numpy as np
import os
from datetime import datetime

# 首先导入isaacgym
import isaacgym
from gpugym.envs import *
from gpugym.utils import get_args, task_registry, wandb_helper
from gpugym import LEGGED_GYM_ROOT_DIR

# 然后导入torch相关模块
import torch

# 导入MoE Actor-Critic模型
from gpu_rl.rsl_rl.modules.moe_actor_critic import MoEActorCritic

# 确保MoEActorCritic在全局命名空间中可用
import gpu_rl.rsl_rl.modules.actor_critic
import gpu_rl.rsl_rl.runners.on_policy_runner
# 将MoEActorCritic添加到on_policy_runner的命名空间
gpu_rl.rsl_rl.runners.on_policy_runner.MoEActorCritic = MoEActorCritic
# 添加到actor_critic的命名空间（仅用于安全）
gpu_rl.rsl_rl.modules.actor_critic.MoEActorCritic = MoEActorCritic

from gpu_rl.rsl_rl.algorithms.ppo import PPO

import wandb

def train_moe(args):
    """
    训练混合专家系统
    
    参数:
    - args: 命令行参数
    """
    # 设置任务名称
    args.task = "pbrs:humanoid_moe"
    
    # 创建环境和PPO运行器
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # 设置实验名称
    if args.wandb_name:
        experiment_name = args.wandb_name
    else:
        experiment_name = f'{args.task}_moe'
    
    # 设置日志目录
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    
    # 是否使用Wandb记录
    do_wandb = train_cfg.do_wandb if hasattr(train_cfg, 'do_wandb') else False
    do_wandb = do_wandb and None not in (args.wandb_project, args.wandb_entity)
    
    # 修改PPO算法使用MoEActorCritic
    obs_dim = env.num_obs
    act_dim = env.num_actions
    
    print(f"观察空间维度: {obs_dim}, 动作空间维度: {act_dim}")
    
    # 替换actor_critic为MoEActorCritic
    device = env.device
    policy_cfg = train_cfg.policy
    actor_critic = MoEActorCritic(
        obs_dim,
        obs_dim,
        act_dim,
        actor_hidden_dims=policy_cfg.actor_hidden_dims,
        critic_hidden_dims=policy_cfg.critic_hidden_dims,
        gating_hidden_dims=policy_cfg.gating_net_hidden_dims,
        activation=policy_cfg.activation,
        init_noise_std=policy_cfg.init_noise_std
    ).to(device)
    
    # 使用自定义actor_critic创建PPO
    alg_cfg = train_cfg.algorithm
    
    # 打印调试信息
    print(f"num_learning_epochs类型: {type(alg_cfg.num_learning_epochs)}, 值: {alg_cfg.num_learning_epochs}")
    
    # 确保num_learning_epochs是整数
    num_epochs = int(alg_cfg.num_learning_epochs)
    
    ppo = PPO(
        actor_critic,
        alg_cfg.clip_param,
        num_epochs,
        alg_cfg.num_mini_batches,
        alg_cfg.value_loss_coef,
        alg_cfg.entropy_coef,
        learning_rate=alg_cfg.learning_rate,
        max_grad_norm=alg_cfg.max_grad_norm,
        use_clipped_value_loss=alg_cfg.use_clipped_value_loss,
        device=device
    )
    
    # 替换runner中的算法
    ppo_runner.alg = ppo
    
    # 初始化PPO存储
    ppo_runner.alg.init_storage(env.num_envs, ppo_runner.num_steps_per_env, (env.num_obs,), (env.num_privileged_obs,), (env.num_actions,))
    
    # 配置Wandb
    if do_wandb:
        wandb.config = {}
        
        if hasattr(train_cfg, 'wandb'):
            what_to_log = train_cfg.wandb.what_to_log
            wandb_helper.craft_log_config(env_cfg, train_cfg, wandb.config, what_to_log)
        
        print(f'Received WandB project name: {args.wandb_project}\nReceived WandB entitiy name: {args.wandb_entity}\n')
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   group=args.wandb_group,
                   config=wandb.config,
                   name=experiment_name)
        
        ppo_runner.configure_wandb(wandb)
        ppo_runner.configure_learn(train_cfg.runner.max_iterations, True)
        ppo_runner.learn()
        
        wandb.finish()
    else:
        # 不使用WandB时也需要进行训练
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
if __name__ == '__main__':
    args = get_args()
    train_moe(args) 