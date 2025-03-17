"""
混合专家(MoE)系统训练脚本

该脚本用于训练混合专家系统，使机器人能够在各种地形上平稳高效地运动。
它结合了行走和奔跑两个预训练模型，通过非监督学习方法动态调整它们的权重。
"""

import os
import sys
import numpy as np
import argparse
from datetime import datetime
import time
import random

# 确保isaacgym模块先于torch导入
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch

# 然后导入torch
import torch

# 设置导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# 最后导入项目特定模块
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


def parse_arguments():
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
    
    # 输出目录参数
    parser.add_argument('--output_dir', type=str, default='logs/PBRS_HumanoidLocomotion',
                        help='输出目录路径')
    
    # 奖励参数
    parser.add_argument('--balance_scale', type=float, default=2.0, help='平衡奖励的缩放系数')
    parser.add_argument('--velocity_scale', type=float, default=1.0, help='速度奖励的缩放系数')
    parser.add_argument('--curiosity_scale', type=float, default=0.1, help='好奇心奖励的缩放系数')
    
    # 门控网络参数
    parser.add_argument('--gating_hidden_dims', type=str, default='128,128', 
                        help='门控网络隐藏层维度，以逗号分隔')
    parser.add_argument('--terrain_feature_dim', type=int, default=50, 
                        help='地形特征维度')
    
    # 训练参数
    parser.add_argument('--replay_buffer_size', type=int, default=10000, 
                        help='经验回放缓冲区大小')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    
    # 地形参数
    parser.add_argument('--terrain_type', type=str, default='heightfield', 
                        choices=['plane', 'heightfield', 'trimesh'],
                        help='地形类型：平面、高度场或三角网格')
    
    # WandB日志参数
    parser.add_argument('--use_wandb', action='store_true', help='是否使用WandB记录日志')
    parser.add_argument('--project_name', type=str, default='HumanoidMoE', help='W&B项目名称')
    parser.add_argument('--entity_name', type=str, default=None, help='W&B实体名称')
    
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
    
    # 更新门控网络参数
    cfg.moe.gating_hidden_dims = [int(dim) for dim in args.gating_hidden_dims.split(',')]
    cfg.moe.terrain_feature_dim = args.terrain_feature_dim
    
    # 更新训练参数
    cfg.moe.replay_buffer_size = args.replay_buffer_size
    cfg.moe.batch_size = args.batch_size
    cfg.moe.learning_rate = args.learning_rate
    
    # 更新地形类型
    cfg.terrain.mesh_type = args.terrain_type
    
    return cfg


def setup_wandb(args, cfg):
    """设置Weights & Biases日志记录"""
    if not args.use_wandb:
        return None
    
    try:
        import wandb
        
        # 设置运行名称
        run_name = f'HumanoidMoE_{datetime.now().strftime("%b%d_%H-%M-%S")}'
        
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
            "gating_hidden_dims": cfg.moe.gating_hidden_dims,
            "terrain_feature_dim": cfg.moe.terrain_feature_dim,
            "replay_buffer_size": cfg.moe.replay_buffer_size,
            "batch_size": cfg.moe.batch_size,
            "learning_rate": cfg.moe.learning_rate
        }
        
        # 初始化WandB
        wandb.init(
            project=args.project_name,
            entity=args.entity_name,
            config=wandb_config,
            name=run_name
        )
        
        print(f"WandB初始化成功: {run_name}")
        return wandb
    except ImportError:
        print("未找到wandb包，无法记录训练过程。如需使用，请安装：pip install wandb")
        return None


def evaluate_model(moe_system, num_eval_episodes=5):
    """评估模型性能"""
    print(f"评估模型性能 ({num_eval_episodes} 个回合)...")
    
    rewards = []
    walk_weights = []
    run_weights = []
    
    for episode in range(num_eval_episodes):
        moe_system.reset()
        
        done = False
        episode_reward = 0
        episode_walk_weights = []
        episode_run_weights = []
        step = 0
        
        while not done and step < moe_system.cfg.env.episode_length_s * moe_system.control_freq:
            _, _, rew_buf, reset_buf, extras = moe_system.step(None)
            
            # 累计奖励
            episode_reward += rew_buf.mean().item()
            
            # 记录混合权重
            if 'blend_weights' in extras:
                weights = extras['blend_weights'].cpu().numpy()
                episode_walk_weights.append(weights[:, 0])
                episode_run_weights.append(weights[:, 1])
            
            done = torch.any(reset_buf).item()
            step += 1
        
        rewards.append(episode_reward)
        
        # 计算本回合的平均权重
        if episode_walk_weights:
            avg_walk_weight = np.mean(np.concatenate(episode_walk_weights))
            avg_run_weight = np.mean(np.concatenate(episode_run_weights))
            walk_weights.append(avg_walk_weight)
            run_weights.append(avg_run_weight)
            
            print(f"  回合 {episode}: 奖励={episode_reward:.2f}, 行走权重={avg_walk_weight:.2f}, 奔跑权重={avg_run_weight:.2f}")
    
    # 计算整体统计数据
    avg_reward = np.mean(rewards)
    avg_walk_weight = np.mean(walk_weights) if walk_weights else 0.0
    avg_run_weight = np.mean(run_weights) if run_weights else 0.0
    
    # 计算行走/奔跑百分比
    walking_percentage = 0.0
    running_percentage = 0.0
    
    if walk_weights:
        walk_weights_flat = np.concatenate([w.flatten() for w in episode_walk_weights])
        run_weights_flat = np.concatenate([r.flatten() for r in episode_run_weights])
        
        walking_percentage = np.mean(walk_weights_flat > 0.5) * 100
        running_percentage = np.mean(run_weights_flat > 0.5) * 100
    
    results = {
        "average_reward": avg_reward,
        "average_walk_weight": avg_walk_weight,
        "average_run_weight": avg_run_weight,
        "walking_percentage": walking_percentage,
        "running_percentage": running_percentage
    }
    
    print(f"评估结果: 平均奖励={avg_reward:.2f}, 行走={walking_percentage:.1f}%, 奔跑={running_percentage:.1f}%")
    
    return results


def create_gym_environment(cfg):
    """创建IsaacGym环境"""
    # 设置仿真参数
    sim_params = gymapi.SimParams()
    
    # 设置基本参数
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = cfg.sim.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True
    
    # 简化PhysX设置，避免属性错误
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    
    # 调试输出
    print("创建Isaac Gym环境...")
    
    # 创建gym和sim
    gym = gymapi.acquire_gym()
    device_id = 0  # CUDA设备ID
    compute_device_id = 0  # 计算设备ID
    graphics_device_id = 0  # 图形设备ID
    
    sim = gym.create_sim(device_id, graphics_device_id, 
                         gymapi.SIM_PHYSX, sim_params)
    
    if sim is None:
        print("*** 创建sim失败 ***")
        quit()
    
    return gym, sim


def train_moe(args):
    """训练混合专家系统"""
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载配置
    cfg = HumanoidMoECfg()
    ppo_cfg = HumanoidMoECfgPPO()
    
    # 从命令行参数更新配置
    cfg = update_config_from_args(cfg, args)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S_moe')
    output_dir = os.path.join(args.output_dir, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 创建模型保存目录
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存命令行参数
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    # 初始化WandB
    wandb_logger = setup_wandb(args, cfg)
    
    # 创建IsaacGym环境
    gym, sim = create_gym_environment(cfg)
    
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建MoE系统，使用共享的sim实例
    print("创建MoE系统...")
    physics_engine = gymapi.SIM_PHYSX
    headless = args.headless
    
    try:
        # 创建MoE环境
        moe_system = HumanoidMoE(
            cfg=cfg, 
            sim_params=None,  # 已经在sim中设置
            physics_engine=physics_engine, 
            sim_device=device, 
            headless=headless, 
            external_sim=sim
        )
        
        print("MoE系统创建成功！")
    except Exception as e:
        print(f"创建MoE系统时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 训练循环
    print(f"开始训练 {args.max_iterations} 个迭代...")
    start_time = time.time()
    
    rewards = []
    best_reward = float('-inf')
    
    try:
        for iteration in range(args.max_iterations):
            # 重置环境
            moe_system.reset()
            
            done = False
            episode_reward = 0
            step = 0
            
            # 记录混合权重和奖励组件
            episode_blend_weights = []
            reward_components = {'balance': [], 'velocity': [], 'curiosity': []}
            
            while not done and step < cfg.env.episode_length_s * moe_system.control_freq:
                # 执行一步
                _, _, rew_buf, reset_buf, extras = moe_system.step(None)
                
                # 累计奖励
                episode_reward += rew_buf.mean().item()
                
                # 记录混合权重
                blend_weights = extras['blend_weights']
                episode_blend_weights.append(blend_weights.cpu().numpy())
                
                # 记录奖励组件
                if 'reward_components' in extras:
                    for key in reward_components:
                        if key in extras['reward_components']:
                            reward_components[key].append(np.mean(extras['reward_components'][key]))
                
                # 检查是否完成
                done = torch.any(reset_buf).item()
                step += 1
            
            # 计算平均值
            rewards.append(episode_reward)
            
            # 安全处理blend_weights计算
            if episode_blend_weights:
                try:
                    avg_blend_weights = np.mean(np.concatenate(episode_blend_weights, axis=0), axis=0)
                except:
                    # 如果连接失败，尝试平均每个数组
                    avg_blend_weights = np.mean([np.mean(w, axis=0) for w in episode_blend_weights], axis=0)
            else:
                avg_blend_weights = np.array([0.0, 0.0])
                
            avg_reward_components = {k: np.mean(v) if v else 0.0 for k, v in reward_components.items()}
            
            # 输出训练进度
            if iteration % 10 == 0 or iteration == args.max_iterations - 1:
                elapsed = time.time() - start_time
                print(f"迭代 {iteration}/{args.max_iterations}, 奖励: {episode_reward:.2f}, " +
                    f"行走权重: {avg_blend_weights[0]:.2f}, 奔跑权重: {avg_blend_weights[1]:.2f}, " +
                    f"耗时: {elapsed:.2f}秒")
            
            # 记录到WandB
            if wandb_logger:
                log_dict = {
                    "iteration": iteration,
                    "reward": episode_reward,
                    "walk_weight": avg_blend_weights[0],
                    "run_weight": avg_blend_weights[1],
                    "steps": step
                }
                # 添加奖励组件
                for k, v in avg_reward_components.items():
                    log_dict[f"{k}_reward"] = v
                    
                wandb_logger.log(log_dict)
            
            # 保存模型
            save_interval = getattr(cfg.moe, 'save_interval', 50)  # 默认每50次迭代保存一次
            if (iteration % save_interval == 0 and iteration > 0) or iteration == args.max_iterations - 1:
                model_path = os.path.join(models_dir, f"moe_humanoid_iteration_{iteration}.pt")
                moe_system.save_models(model_path)
                
                # 保存最佳模型
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_model_path = os.path.join(models_dir, "moe_humanoid_best.pt")
                    moe_system.save_models(best_model_path)
                    print(f"保存最佳模型，奖励: {best_reward:.2f}")
            
            # 定期评估
            eval_interval = getattr(cfg.moe, 'eval_interval', 100)  # 默认每100次迭代评估一次
            if iteration % eval_interval == 0 and iteration > 0:
                eval_results = evaluate_model(moe_system, num_eval_episodes=3)
                
                if wandb_logger:
                    wandb_logger.log({
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
        
        # 计算训练总时间
        total_time = time.time() - start_time
        print(f"训练完成! 总时间: {total_time:.2f}秒")
        print(f"最终模型已保存至: {final_model_path}")
        
        # 最终评估
        final_eval_results = evaluate_model(moe_system, num_eval_episodes=5)
        print("最终评估结果:")
        for k, v in final_eval_results.items():
            print(f"  {k}: {v:.4f}")
        
        if wandb_logger:
            wandb_logger.log({
                "final_eval/average_reward": final_eval_results["average_reward"],
                "final_eval/walk_weight": final_eval_results["average_walk_weight"],
                "final_eval/run_weight": final_eval_results["average_run_weight"],
                "final_eval/walking_percentage": final_eval_results["walking_percentage"],
                "final_eval/running_percentage": final_eval_results["running_percentage"]
            })
            wandb_logger.finish()
        
        return moe_system, rewards, final_eval_results
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        interrupt_model_path = os.path.join(models_dir, "moe_humanoid_interrupted.pt")
        print(f"保存中断时的模型到: {interrupt_model_path}")
        try:
            moe_system.save_models(interrupt_model_path)
        except:
            print("保存中断模型失败")
    
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试保存当前模型
        try:
            error_model_path = os.path.join(models_dir, "moe_humanoid_error.pt")
            moe_system.save_models(error_model_path)
            print(f"已保存错误时的模型到: {error_model_path}")
        except:
            print("保存错误模型失败")


if __name__ == "__main__":
    args = parse_arguments()
    train_moe(args)