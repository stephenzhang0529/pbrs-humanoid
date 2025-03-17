"""
混合专家(MoE)系统的配置文件
定义了MoE系统的所有配置参数
"""

from gpugym.envs.PBRS.humanoid_config import HumanoidCfg, HumanoidCfgPPO
import torch
from isaacgym import gymapi


class HumanoidMoECfg(HumanoidCfg):
    """混合专家系统配置，继承自HumanoidCfg"""
    
    class env(HumanoidCfg.env):
        # 使用混合专家系统的特定环境参数
        num_envs = 1024
        episode_length_s = 10  # 延长测试时间以观察模型切换效果
    
    class terrain(HumanoidCfg.terrain):
        # 启用地形课程学习
        curriculum = True
        mesh_type = 'heightfield'  # 可以是'plane', 'heightfield'或'trimesh'
        measure_heights = True
    
    class moe:
        # 混合专家系统相关参数
        # 模型路径
        walk_model_path = 'logs/PBRS_HumanoidLocomotion/Mar07_20-21-45_walkmodel/model_10000.pt'
        run_model_path = 'logs/PBRS_HumanoidLocomotion/Mar08_07-40-01_runmodel/model_20000.pt'
        
        # 模型混合参数
        gating_hidden_dims = [128, 128]  # 门控网络隐藏层维度
        terrain_feature_dim = 50   # 地形特征维度
        
        # 非监督学习参数
        use_unsupervised_rewards = True  # 是否使用非监督奖励
        curiosity_reward_scale = 0.1     # 好奇心奖励比例
        balance_reward_scale = 2.0       # 平衡奖励比例
        velocity_reward_scale = 1.0      # 速度奖励比例
        
        # 训练参数
        replay_buffer_size = 10000      # 经验回放缓冲区大小
        batch_size = 64                 # 批量大小
        learning_rate = 3e-4            # 学习率
        gamma = 0.99                    # 折扣因子
        
        # 模型融合参数
        action_smoothing = 0.5          # 动作平滑系数（0-1之间，越大越平滑）
        min_velocity_threshold = 1.0    # 行走/奔跑的速度阈值(m/s)
        roughness_threshold = 0.15      # 粗糙度阈值，用于决定行走/奔跑
        
        # 调试参数
        debug_visualization = False     # 是否显示调试可视化
        save_interval = 50              # 保存模型的间隔（迭代次数）
        eval_interval = 10              # 评估模型的间隔（迭代次数）


class HumanoidMoECfgPPO(HumanoidCfgPPO):
    """PPO配置，继承自HumanoidCfgPPO"""
    
    run_name = "humanoid_moe"
    experiment_name = "PBRS_HumanoidMoE"
    seed = 42  # 固定随机种子以便复现结果
    
    max_iterations = 500  # 训练迭代次数
    
    # WandB日志配置
    do_wandb = True
    
    class wandb:
        class what_to_log:
            # 指定要记录的配置参数
            class algorithm:
                fields = ["learning_rate", "num_learning_epochs", "num_mini_batches"]
            
            class task:
                fields = ["moe.roughness_threshold", "moe.action_smoothing", "moe.min_velocity_threshold"]
            
            # 记录MoE特定指标
            class moe_metrics:
                record_blend_weights = True      # 记录混合权重
                record_reward_components = True  # 记录奖励组成部分
                record_terrain_features = True   # 记录地形特征 