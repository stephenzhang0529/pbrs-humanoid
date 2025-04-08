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
from gpu_rl.rsl_rl.modules.actor_critic import ActorCritic

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
    
    # 显式设置为训练模式
    actor_critic.train()
    
    # 加载预训练的行走和奔跑专家模型
    walk_expert = None
    run_expert = None
    
    if hasattr(env_cfg.env, 'experts'):
        # 加载行走专家
        if hasattr(env_cfg.env.experts, 'walk_model_path') and env_cfg.env.experts.walk_model_path:
            walk_model_path = env_cfg.env.experts.walk_model_path
            print(f"加载行走模型: {walk_model_path}")
            walk_expert = ActorCritic(obs_dim, obs_dim, act_dim).to(device)
            try:
                walk_checkpoint = torch.load(walk_model_path, map_location=device)
                walk_expert.load_state_dict(walk_checkpoint['model_state_dict'])
                walk_expert.eval()  # 专家模型设置为评估模式
                print("行走模型加载成功！")
            except Exception as e:
                print(f"加载行走模型失败: {e}")
                walk_expert = None
        
        # 加载奔跑专家
        if hasattr(env_cfg.env.experts, 'run_model_path') and env_cfg.env.experts.run_model_path:
            run_model_path = env_cfg.env.experts.run_model_path
            print(f"加载奔跑模型: {run_model_path}")
            run_expert = ActorCritic(obs_dim, obs_dim, act_dim).to(device)
            try:
                run_checkpoint = torch.load(run_model_path, map_location=device)
                run_expert.load_state_dict(run_checkpoint['model_state_dict'])
                run_expert.eval()  # 专家模型设置为评估模式
                print("奔跑模型加载成功！")
            except Exception as e:
                print(f"加载奔跑模型失败: {e}")
                run_expert = None
    
    # 如果成功加载了两个专家模型，则将它们传递给MoEActorCritic
    if walk_expert is not None and run_expert is not None:
        actor_critic.load_experts(walk_expert, run_expert)
        print("已成功加载两个专家模型到MoEActorCritic")
    else:
        print("警告：未能成功加载专家模型，MoEActorCritic将仅使用自己的actor网络")
    
    # 再次确认我们是否处于训练模式
    actor_critic.train()
    
    # 为确保环境能够正确响应，我们进行更全面的环境验证
    print("===== 开始环境验证测试 =====")
    obs, privileged_obs = env.reset()  # 环境的reset方法返回(obs, privileged_obs)
    
    with torch.no_grad():
        test_action = actor_critic.act(obs)
        print(f"初始动作示例: {test_action[0]}")
        if torch.all(torch.abs(test_action) < 1e-6):
            print("警告：初始动作全为0或接近0，可能表明动作生成有问题")
            
        # 测试环境第一步是否正常
        print("\n测试环境step...")
        next_obs, next_privileged_obs, rewards, dones, infos = env.step(test_action)
        print(f"第一步奖励: {rewards[0].item():.4f}, 是否终止: {dones[0].item()}")
        
        # 强制运行连续30步，确保环境可以稳定运行
        obs, privileged_obs = env.reset()
        print("\n强制环境连续运行30步以验证环境功能...")
        
        for i in range(30):
            action = actor_critic.act(obs)
            
            # 每10步打印一次动作
            if i % 10 == 0:
                print(f"步骤 {i+1} 动作: {action[0]}")
                
            obs, privileged_obs, rewards, dones, infos = env.step(action)
            
            terminated_envs = dones.sum().item()
            print(f"步骤 {i+1}/30, 平均奖励: {rewards.mean().item():.4f}, 终止环境数: {terminated_envs}/{len(dones)}")
            
            # 如果有环境终止，记录一下原因
            if terminated_envs > 0:
                print(f"  有 {terminated_envs} 个环境在步骤 {i+1} 终止")
            
            # 如果所有环境都终止了，重置它们
            if torch.all(dones):
                print(f"所有环境在步骤 {i+1} 终止，重置并继续...")
                obs, privileged_obs = env.reset()
    
    print("===== 环境验证测试完成 =====\n")
    
    # 修改步数和batch大小
    if train_cfg.runner.num_steps_per_env < 24:
        print(f"警告：步数过少，将num_steps_per_env从{train_cfg.runner.num_steps_per_env}增加到24")
        train_cfg.runner.num_steps_per_env = 24
        
    if args.num_envs < 10:
        print(f"警告：环境数量过少，可能会影响训练效果。当前：{args.num_envs}，推荐：4096")
    
    # 使用自定义actor_critic创建PPO
    alg_cfg = train_cfg.algorithm
    
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