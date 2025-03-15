"""
混合专家(MoE)系统训练脚本

该脚本用于训练混合专家系统，使机器人能够在各种地形上平稳高效地运动。
它结合了行走和奔跑两个预训练模型，通过非监督学习方法动态调整它们的权重。
"""

import os
import numpy as np

from datetime import datetime
import argparse
import time

import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

# 尝试导入wandb (可选)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("未找到wandb包，将不记录训练过程。如需使用，请安装：pip install wandb")

from gpugym.envs.PBRS.moe import HumanoidMoE
from gpugym.envs.PBRS.moe_config import HumanoidMoECfg, HumanoidMoECfgPPO
from gpugym import LEGGED_GYM_ROOT_DIR


# 添加单例模式的IsaacGym管理器类
class IsaacGymManager:
    """
    IsaacGym环境管理器 - 单例模式
    确保整个应用程序中只有一个IsaacGym实例
    """
    _instance = None
    _gym = None
    _sim = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IsaacGymManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, sim_params, physics_engine=gymapi.SIM_PHYSX, device_id=0, compute_device_id=0):
        """初始化IsaacGym环境"""
        if not self._initialized:
            print("初始化IsaacGym环境...")
            self._gym = gymapi.acquire_gym()
            self._sim = self._gym.create_sim(device_id, compute_device_id, physics_engine, sim_params)
            
            if self._sim is None:
                print("❌ 创建模拟失败")
                raise RuntimeError("无法创建IsaacGym模拟")
                
            self._initialized = True
            print("✅ IsaacGym环境初始化成功")
            
        return self._gym, self._sim
        
    def get_gym(self):
        """获取gym实例"""
        if not self._initialized:
            raise RuntimeError("IsaacGym环境尚未初始化")
        return self._gym
        
    def get_sim(self):
        """获取sim实例"""
        if not self._initialized:
            raise RuntimeError("IsaacGym环境尚未初始化")
        return self._sim
        
    def cleanup(self):
        """清理资源"""
        if self._initialized:
            # IsaacGym没有显式的清理方法，但如果有的话可以在这里调用
            self._initialized = False
            self._gym = None
            self._sim = None
            print("已清理IsaacGym资源")


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练混合专家(MoE)系统')
    
    # 基础训练参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_envs', type=int, default=1024, help='并行环境数量')
    parser.add_argument('--max_iterations', type=int, default=500, help='训练迭代次数')
    parser.add_argument('--headless', action='store_true', help='不显示GUI窗口')
    
    # 模型路径参数
    parser.add_argument('--walk_model', type=str, 
                        default='logs/PBRS_HumanoidLocomotion/Mar07_20-21-45_walkmodel/model_10000.pt',
                        help='行走模型路径')
    parser.add_argument('--run_model', type=str,
                        default='logs/PBRS_HumanoidLocomotion/Mar08_07-40-01_runmodel/model_20000.pt',
                        help='奔跑模型路径')
    
    # 奖励参数
    parser.add_argument('--balance_scale', type=float, default=2.0, help='平衡奖励的缩放系数')
    parser.add_argument('--velocity_scale', type=float, default=1.0, help='速度奖励的缩放系数')
    parser.add_argument('--curiosity_scale', type=float, default=0.1, help='好奇心奖励的缩放系数')
    
    # 地形参数
    parser.add_argument('--terrain_type', type=str, default='heightfield', 
                        choices=['plane', 'heightfield', 'trimesh'],
                        help='地形类型：平面、高度场或三角网格')
    
    # WandB日志参数
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB项目名称')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB实体名称')
    parser.add_argument('--wandb_group', type=str, default=None, help='WandB组名称')
    parser.add_argument('--wandb_name', type=str, default=None, help='WandB运行名称')
    
    return parser.parse_args()


def update_config_from_args(cfg, args):
    """从命令行参数更新配置"""
    # 更新模型路径
    cfg.moe.walk_model_path = args.walk_model
    cfg.moe.run_model_path = args.run_model
    
    # 更新环境参数
    cfg.env.num_envs = args.num_envs
    
    # 更新奖励参数
    cfg.moe.balance_reward_scale = args.balance_scale
    cfg.moe.velocity_reward_scale = args.velocity_scale
    cfg.moe.curiosity_reward_scale = args.curiosity_scale
    
    # 更新地形类型
    cfg.terrain.mesh_type = args.terrain_type
    
    return cfg


def setup_wandb(args, cfg, moe_cfg):
    """设置Weights & Biases日志记录"""
    if not WANDB_AVAILABLE:
        return False
    
    # 检查是否提供了必要的WandB参数
    do_wandb = moe_cfg.do_wandb if hasattr(moe_cfg, 'do_wandb') else False
    do_wandb = do_wandb and None not in (args.wandb_project, args.wandb_entity)
    
    if not do_wandb:
        return False
    
    # 设置运行名称
    if args.wandb_name:
        experiment_name = args.wandb_name
    else:
        experiment_name = f'HumanoidMoE_{datetime.now().strftime("%b%d_%H-%M-%S")}'
    
    # 配置WandB参数
    wandb_config = {
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "balance_scale": args.balance_scale,
        "velocity_scale": args.velocity_scale, 
        "curiosity_scale": args.curiosity_scale,
        "terrain_type": args.terrain_type,
        "walk_model": args.walk_model,
        "run_model": args.run_model,
        "action_smoothing": cfg.moe.action_smoothing,
        "roughness_threshold": cfg.moe.roughness_threshold,
    }
    
    # 初始化WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config=wandb_config,
        name=experiment_name
    )
    
    print(f"WandB初始化成功: {experiment_name}")
    return True


def evaluate_model(moe_system, num_eval_episodes=10):
    """评估模型性能"""
    print("正在评估模型...")
    
    total_reward = 0
    average_blend_weights = np.zeros(2)
    walking_percentage = 0
    running_percentage = 0
    
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


def train(args):
    """
    训练混合专家系统
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载配置
    cfg = HumanoidMoECfg()
    moe_cfg = HumanoidMoECfgPPO()
    cfg = update_config_from_args(cfg, args)
    
    # 创建日志目录
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', moe_cfg.experiment_name)
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(log_root, f"{timestamp}_{moe_cfg.run_name}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"日志目录: {log_dir}")
    
    # 创建模型保存目录
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 设置WandB
    use_wandb = setup_wandb(args, cfg, moe_cfg)
    
    # 设置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = cfg.sim.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True
    
    if hasattr(cfg.sim, 'physx'):
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = cfg.sim.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = cfg.sim.physx.num_velocity_iterations
        sim_params.physx.contact_offset = cfg.sim.physx.contact_offset
        sim_params.physx.rest_offset = cfg.sim.physx.rest_offset
    
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 使用单例模式初始化IsaacGym
    gym_manager = IsaacGymManager()
    try:
        gym, sim = gym_manager.initialize(sim_params, gymapi.SIM_PHYSX, 0, 0)
        print("成功创建IsaacGym环境")
    except Exception as e:
        print(f"创建IsaacGym环境失败: {e}")
        return
    
    # 创建MoE系统，使用共享的sim实例
    print("正在创建MoE系统...")
    physics_engine = gymapi.SIM_PHYSX
    moe_system = HumanoidMoE(cfg, sim_params, physics_engine, device, args.headless, external_sim=sim)
    
    # 训练循环
    print(f"开始训练 {args.max_iterations} 个迭代...")
    rewards = []
    best_reward = float('-inf')
    
    try:
        for iteration in range(args.max_iterations):
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
                if 'reward_components' in extras and extras['reward_components']:
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
            if iteration % 10 == 0 or iteration == args.max_iterations - 1:
                print(f"迭代 {iteration}/{args.max_iterations}, 奖励: {episode_reward:.2f}, "
                      f"行走权重: {avg_walk_weight:.2f}, 奔跑权重: {avg_run_weight:.2f}")
            
            # 记录到WandB
            if use_wandb:
                wandb.log({
                    "iteration": iteration,
                    "reward": episode_reward,
                    "walk_weight": avg_walk_weight,
                    "run_weight": avg_run_weight,
                    "balance_reward": avg_balance_reward,
                    "velocity_reward": avg_velocity_reward,
                    "curiosity_reward": avg_curiosity_reward,
                    "steps": step
                })
            
            # 定期保存模型
            if iteration % cfg.moe.save_interval == 0 or iteration == args.max_iterations - 1:
                model_path = os.path.join(models_dir, f"moe_humanoid_iteration_{iteration}.pt")
                moe_system.save_models(model_path)
                
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_model_path = os.path.join(models_dir, "moe_humanoid_best.pt")
                    moe_system.save_models(best_model_path)
                    print(f"保存最佳模型，奖励: {best_reward:.2f}")
            
            # 定期评估模型
            if iteration % cfg.moe.eval_interval == 0 or iteration == args.max_iterations - 1:
                eval_results = evaluate_model(moe_system)
                
                if use_wandb:
                    wandb.log({
                        "eval/average_reward": eval_results["average_reward"],
                        "eval/walk_weight": eval_results["average_walk_weight"],
                        "eval/run_weight": eval_results["average_run_weight"],
                        "eval/walking_percentage": eval_results["walking_percentage"],
                        "eval/running_percentage": eval_results["running_percentage"],
                        "iteration": iteration
                    })
        
        # 保存最终模型
        final_model_path = os.path.join(models_dir, "moe_humanoid_final.pt")
        moe_system.save_models(final_model_path)
        print(f"训练完成，最终模型已保存至 {final_model_path}")
        
        # 最终评估
        final_eval_results = evaluate_model(moe_system, num_eval_episodes=20)
        print("最终评估结果:")
        for k, v in final_eval_results.items():
            print(f"  {k}: {v:.4f}")
        
        if use_wandb:
            wandb.log({
                "final_eval/average_reward": final_eval_results["average_reward"],
                "final_eval/walk_weight": final_eval_results["average_walk_weight"],
                "final_eval/run_weight": final_eval_results["average_run_weight"],
                "final_eval/walking_percentage": final_eval_results["walking_percentage"],
                "final_eval/running_percentage": final_eval_results["running_percentage"]
            })
            wandb.finish()
        
        return moe_system, rewards, final_eval_results
    
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        if use_wandb:
            wandb.finish()
        raise
    finally:
        # 清理资源
        gym_manager.cleanup()


if __name__ == '__main__':
    args = get_args()
    train(args) 