"""
混合专家(MoE)系统训练脚本

该脚本用于训练混合专家系统，使机器人能够在各种地形上平稳高效地运动。
它结合了行走和奔跑两个预训练模型，通过非监督学习方法动态调整它们的权重。
"""

import os
import time
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, get_axis_params, torch_rand_float
import torch
import numpy as np
import argparse
import wandb

from gpugym.envs.PBRS.moe import HumanoidMoE, train_moe_system
from gpugym.envs.PBRS.moe_config import HumanoidMoECfg, HumanoidMoECfgPPO


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练混合专家(MoE)系统')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--save_interval', type=int, default=50, help='保存模型的回合间隔')
    parser.add_argument('--eval_interval', type=int, default=10, help='评估模型的回合间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 模型路径参数
    parser.add_argument('--walk_model', type=str, 
                         default='logs/PBRS_HumanoidLocomotion/Mar07_20-21-45_walkmodel/model_10000.pt',
                         help='行走模型路径')
    parser.add_argument('--run_model', type=str,
                         default='logs/PBRS_HumanoidLocomotion/Mar08_07-40-01_runmodel/model_20000.pt',
                         help='奔跑模型路径')
    parser.add_argument('--output_dir', type=str, default='logs/PBRS_HumanoidMoE',
                        help='输出目录，用于保存模型和日志')
    
    # 混合专家系统参数
    parser.add_argument('--terrain_type', type=str, default='heightfield', 
                        choices=['plane', 'heightfield', 'trimesh'],
                        help='地形类型：平面、高度场或三角网格')
    parser.add_argument('--balance_scale', type=float, default=2.0, help='平衡奖励的缩放系数')
    parser.add_argument('--velocity_scale', type=float, default=1.0, help='速度奖励的缩放系数')
    parser.add_argument('--curiosity_scale', type=float, default=0.1, help='好奇心奖励的缩放系数')
    parser.add_argument('--action_smoothing', type=float, default=0.5, help='动作平滑系数(0-1)')
    
    # 训练环境参数
    parser.add_argument('--num_envs', type=int, default=1024, help='并行环境数量')
    parser.add_argument('--headless', action='store_true', help='是否以无头模式运行（不渲染可视化）')
    
    # 日志参数
    parser.add_argument('--wandb', action='store_true', help='是否使用Weights & Biases记录训练过程')
    parser.add_argument('--project_name', type=str, default='HumanoidMoE', help='W&B项目名称')
    parser.add_argument('--run_name', type=str, default=None, help='W&B运行名称')
    
    return parser.parse_args()


def setup_wandb(args, cfg):
    """设置Weights & Biases记录"""
    if args.wandb:
        run_name = args.run_name if args.run_name else f"moe_{time.strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=args.project_name,
            name=run_name,
            config={
                "num_episodes": args.num_episodes,
                "num_envs": args.num_envs,
                "balance_scale": args.balance_scale,
                "velocity_scale": args.velocity_scale,
                "curiosity_scale": args.curiosity_scale,
                "action_smoothing": args.action_smoothing,
                "terrain_type": args.terrain_type,
                "walk_model": args.walk_model,
                "run_model": args.run_model,
            }
        )


def update_config_from_args(cfg, args):
    """从命令行参数更新配置"""
    # 更新模型路径
    cfg.moe.walk_model_path = args.walk_model
    cfg.moe.run_model_path = args.run_model
    
    # 更新训练参数
    cfg.env.num_envs = args.num_envs
    cfg.moe.save_interval = args.save_interval
    cfg.moe.eval_interval = args.eval_interval
    
    # 更新奖励参数
    cfg.moe.balance_reward_scale = args.balance_scale
    cfg.moe.velocity_reward_scale = args.velocity_scale
    cfg.moe.curiosity_reward_scale = args.curiosity_scale
    
    # 更新动作平滑参数
    cfg.moe.action_smoothing = args.action_smoothing
    
    # 更新地形类型
    cfg.terrain.mesh_type = args.terrain_type
    
    return cfg


def create_output_dir(output_dir):
    """创建输出目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"run_{timestamp}")
    
    os.makedirs(output_path, exist_ok=True)
    models_path = os.path.join(output_path, "models")
    os.makedirs(models_path, exist_ok=True)
    
    return output_path, models_path


def evaluate_model(moe_system, num_eval_episodes=10):
    """评估模型性能"""
    total_reward = 0
    average_blend_weights = np.zeros(2)
    walking_percentage = 0
    running_percentage = 0
    
    print("正在评估模型...")
    
    for i in range(num_eval_episodes):
        # 重置环境
        moe_system.reset()
        
        episode_reward = 0
        episode_weights = []
        step = 0
        done = False
        
        while not done and step < moe_system.cfg.env.episode_length_s * moe_system.control_freq:
            # 执行一步
            _, _, rew_buf, reset_buf, extras = moe_system.step(None)
            
            # 记录指标
            episode_reward += rew_buf.mean().item()
            episode_weights.append(extras['blend_weights'].cpu().numpy())
            
            # 检查是否完成
            done = torch.any(reset_buf).item()
            step += 1
        
        # 累计奖励和混合权重
        total_reward += episode_reward
        
        episode_weights = np.concatenate(episode_weights, axis=0)
        avg_weights = episode_weights.mean(axis=0)
        average_blend_weights += avg_weights
        
        # 计算行走和奔跑的比例
        dominant_indices = np.argmax(episode_weights, axis=1)
        walking_percentage += (dominant_indices == 0).mean()
        running_percentage += (dominant_indices == 1).mean()
    
    # 计算平均值
    average_reward = total_reward / num_eval_episodes
    average_blend_weights /= num_eval_episodes
    walking_percentage /= num_eval_episodes
    running_percentage /= num_eval_episodes
    
    eval_results = {
        "average_reward": average_reward,
        "average_walk_weight": average_blend_weights[0],
        "average_run_weight": average_blend_weights[1],
        "walking_percentage": walking_percentage,
        "running_percentage": running_percentage
    }
    
    print(f"评估结果: 平均奖励={average_reward:.2f}, "
          f"行走占比={walking_percentage:.2f}, 奔跑占比={running_percentage:.2f}")
    
    return eval_results


def main():
    """主程序入口"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    output_path, models_path = create_output_dir(args.output_dir)
    print(f"输出目录: {output_path}")
    
    # 加载配置
    cfg = HumanoidMoECfg()
    cfg = update_config_from_args(cfg, args)
    
    # 设置Weights & Biases
    setup_wandb(args, cfg)
    
    # 创建Isaac Gym环境
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = cfg.sim.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True
    
    gym = gymapi.acquire_gym()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模拟
    physics_engine = gymapi.SIM_PHYSX
    sim = gym.create_sim(0, 0, physics_engine, sim_params)
    
    if sim is None:
        print("创建模拟失败")
        return
    
    # 创建MoE系统
    print("正在创建MoE系统...")
    moe_system = HumanoidMoE(cfg, gym,sim, device, args.headless)
    
    # 训练循环
    print(f"开始训练 {args.num_episodes} 个回合...")
    rewards = []
    best_reward = float('-inf')
    
    for episode in range(args.num_episodes):
        # 重置环境
        moe_system.reset()
        
        done = False
        episode_reward = 0
        step = 0
        episode_blend_weights = []
        reward_components = []
        
        while not done and step < cfg.env.episode_length_s * moe_system.control_freq:
            # 执行一步
            _, _, rew_buf, reset_buf, extras = moe_system.step(None)
            
            # 累计奖励
            episode_reward += rew_buf.mean().item()
            
            # 记录混合权重和奖励组件
            episode_blend_weights.append(extras['blend_weights'].cpu().numpy())
            if extras['reward_components']:
                reward_components.append(extras['reward_components'])
            
            # 检查是否完成
            done = torch.any(reset_buf).item()
            step += 1
        
        rewards.append(episode_reward)
        
        # 计算指标
        avg_walk_weight = np.mean([w[:, 0].mean() for w in episode_blend_weights])
        avg_run_weight = np.mean([w[:, 1].mean() for w in episode_blend_weights])
        
        if reward_components:
            avg_balance_reward = np.mean([rc['balance'] for rc in reward_components])
            avg_velocity_reward = np.mean([rc['velocity'] for rc in reward_components])
            avg_curiosity_reward = np.mean([rc['curiosity'] for rc in reward_components])
        else:
            avg_balance_reward = avg_velocity_reward = avg_curiosity_reward = 0
        
        # 输出进度
        if episode % 10 == 0:
            print(f"回合 {episode}/{args.num_episodes}, 奖励: {episode_reward:.2f}, "
                  f"行走权重: {avg_walk_weight:.2f}, 奔跑权重: {avg_run_weight:.2f}")
        
        # 记录到Weights & Biases
        if args.wandb:
            wandb.log({
                "episode": episode,
                "reward": episode_reward,
                "walk_weight": avg_walk_weight,
                "run_weight": avg_run_weight,
                "balance_reward": avg_balance_reward,
                "velocity_reward": avg_velocity_reward,
                "curiosity_reward": avg_curiosity_reward,
                "steps": step
            })
        
        # 定期保存模型
        if episode % args.save_interval == 0 or episode == args.num_episodes - 1:
            model_path = os.path.join(models_path, f"moe_humanoid_episode_{episode}.pt")
            moe_system.save_models(model_path)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_model_path = os.path.join(models_path, "moe_humanoid_best.pt")
                moe_system.save_models(best_model_path)
                print(f"保存最佳模型，奖励: {best_reward:.2f}")
        
        # 定期评估模型
        if episode % args.eval_interval == 0:
            eval_results = evaluate_model(moe_system)
            
            if args.wandb:
                wandb.log({
                    "eval/average_reward": eval_results["average_reward"],
                    "eval/walk_weight": eval_results["average_walk_weight"],
                    "eval/run_weight": eval_results["average_run_weight"],
                    "eval/walking_percentage": eval_results["walking_percentage"],
                    "eval/running_percentage": eval_results["running_percentage"],
                    "episode": episode
                })
    
    # 保存最终模型
    final_model_path = os.path.join(models_path, "moe_humanoid_final.pt")
    moe_system.save_models(final_model_path)
    print(f"训练完成，最终模型已保存至 {final_model_path}")
    
    # 最终评估
    final_eval_results = evaluate_model(moe_system, num_eval_episodes=20)
    print("最终评估结果:")
    for k, v in final_eval_results.items():
        print(f"  {k}: {v:.4f}")
    
    if args.wandb:
        wandb.log({
            "final_eval/average_reward": final_eval_results["average_reward"],
            "final_eval/walk_weight": final_eval_results["average_walk_weight"],
            "final_eval/run_weight": final_eval_results["average_run_weight"],
            "final_eval/walking_percentage": final_eval_results["walking_percentage"],
            "final_eval/running_percentage": final_eval_results["running_percentage"]
        })
        wandb.finish()


if __name__ == "__main__":
    main() 