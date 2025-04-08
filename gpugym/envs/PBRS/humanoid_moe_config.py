"""
混合专家(MoE)系统的配置文件
继承自humanoid_config.py，添加MoE特定的配置项
"""

from gpugym.envs.PBRS.humanoid_config import HumanoidCfg
from gpugym.envs.PBRS.humanoid_run_config import HumanoidRunCfg
from gpugym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch

class HumanoidMoECfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        # 使用与行走和奔跑配置一致的num_envs
        num_envs = 4096
        
        # MoE特定的配置
        num_experts = 2  # 专家数量：行走和奔跑
        
        # 混合专家模型的配置
        class experts:
            walk_model_path = "logs/PBRS_HumanoidLocomotion/Mar07_20-21-45_walkmodel/model_10000.pt"
            run_model_path = "logs/PBRS_HumanoidLocomotion/Mar08_07-40-01_runmodel/model_20000.pt"
        
        # 使用复杂地形
        class terrain(HumanoidCfg.terrain):
            mesh_type = 'trimesh'
            curriculum = True      # 是否使用课程学习
            selected = False       # 随机采样或者使用选定的地形
            
            # 地形尺寸、分辨率
            terrain_length = 8.
            terrain_width = 8.
            num_rows = 10
            num_cols = 10
            
            # 随机化高度
            max_init_terrain_level = 5
            
            # 动态地形选择
            terrain_kwargs = {"type": "mixed"}
            
            terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
            
            # 高度测量
            measure_heights = True
            measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # 自定义奖励系数
    class rewards(HumanoidCfg.rewards):
        # 平衡奖励系数
        balance_scale = 1.0
        
        # 速度奖励系数
        speed_scale = 0.5
        
        # 内在好奇心奖励系数
        curiosity_scale = 0.1
        
        # 平滑切换奖励系数
        transition_scale = 0.1
            
    # 配置控制参数
    class control(HumanoidCfg.control):
        # PPO决策频率 (Hz)
        frequency = 60
        
        # 行动边界
        action_scale = 1.0
        exp_avg_decay = 0.1
        
        # PD控制器的参数
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_ppo.pt"
        
        # 关节刚度和阻尼参数
        stiffness = {
            'left_hip_yaw': 30.,
            'left_hip_abad': 30.,
            'left_hip_pitch': 30.,
            'left_knee': 30.,
            'left_ankle': 30.,
            'right_hip_yaw': 30.,
            'right_hip_abad': 30.,
            'right_hip_pitch': 30.,
            'right_knee': 30.,
            'right_ankle': 30.,
        }
        damping = {
            'left_hip_yaw': 5.,
            'left_hip_abad': 5.,
            'left_hip_pitch': 5.,
            'left_knee': 5.,
            'left_ankle': 5.,
            'right_hip_yaw': 5.,
            'right_hip_abad': 5.,
            'right_hip_pitch': 5.,
            'right_knee': 5.,
            'right_ankle': 5.
        }
        
        # 门控网络参数
        class gating_network:
            hidden_dims = [256, 128]  # 隐藏层维度
            
    # 领域随机化参数
    class domain_rand(HumanoidCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        
        # 随机施加推力到机器人上
        push_robots = True
        push_interval_s = 2
        max_push_vel_xy = 1.0
        
    # 观察配置
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            base_z = 1./0.6565  # 添加base_z属性
            
        # 添加缺少的属性
        clip_observations = 100.
        clip_actions = 10.

    # 噪声配置
    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            in_contact = 0.1
            height_measurements = 0.1
            
    # 初始状态配置
    class init_state(HumanoidCfg.init_state):
        reset_mode = 'reset_to_range'
        
        # 腿部init pose, target
        # 使用与humanoid_run_config.py一致的关节配置
        default_joint_angles = {  # 单位为弧度
            'left_hip_yaw': 0.,
            'left_hip_abad': 0.,
            'left_hip_pitch': -0.2,
            'left_knee': 0.25,  # 0.6
            'left_ankle': 0.0,
            'right_hip_yaw': 0.,
            'right_hip_abad': 0.,
            'right_hip_pitch': -0.2,
            'right_knee': 0.25,  # 0.6
            'right_ankle': 0.0,
        }
        
        # 关节位置范围
        dof_pos_range = {
            'left_hip_yaw': [-0.1, 0.1],
            'left_hip_abad': [-0.2, 0.2],
            'left_hip_pitch': [-0.2, 0.2],
            'left_knee': [0.6, 0.7],
            'left_ankle': [-0.3, 0.3],
            'right_hip_yaw': [-0.1, 0.1],
            'right_hip_abad': [-0.2, 0.2],
            'right_hip_pitch': [-0.2, 0.2],
            'right_knee': [0.6, 0.7],
            'right_ankle': [-0.3, 0.3],
        }
        
        # 关节速度范围
        dof_vel_range = {
            'left_hip_yaw': [-0.1, 0.1],
            'left_hip_abad': [-0.1, 0.1],
            'left_hip_pitch': [-0.1, 0.1],
            'left_knee': [-0.1, 0.1],
            'left_ankle': [-0.1, 0.1],
            'right_hip_yaw': [-0.1, 0.1],
            'right_hip_abad': [-0.1, 0.1],
            'right_hip_pitch': [-0.1, 0.1],
            'right_knee': [-0.1, 0.1],
            'right_ankle': [-0.1, 0.1],
        }

    class asset(HumanoidCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}'\
            '/resources/robots/mit_humanoid/mit_humanoid_fixed_arms.urdf'
        keypoints = ["base"]
        end_effectors = ['left_foot', 'right_foot']
        foot_name = 'foot'
        
        # 进一步减少终止接触部位，只在躯干接触时终止
        terminate_after_contacts_on = [
            'base',
            # 完全移除其他所有身体部位
            # 'left_upper_arm',
            # 'right_upper_arm',
            # 'left_lower_arm',
            # 'right_lower_arm',
            # 'left_hand',
            # 'right_hand',
        ]

        # 对应的不惩罚的接触部位
        penalize_contacts_on = []

        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

class HumanoidMoECfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # 添加activation属性
        
        # 门控网络配置
        gating_net_hidden_dims = [256, 128]
        
    # 算法训练配置
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # PPO参数
        entropy_coef = 0.01
        learning_rate = 1e-4
        num_learning_epochs = 5
        gamma = 0.99
        use_clipped_value_loss = True
        clip_param = 0.2
        value_loss_coef = 1.0
        
        # 缺少的参数
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        desired_kl = 0.01
        max_grad_norm = 1.
        # adam优化器参数
        weight_decay = 0
        
    # 运行配置
    class runner:
        policy_class_name = 'MoEActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # 每次迭代的步数
        run_name = 'humanoid_moe'
        experiment_name = 'PBRS_HumanoidMoE'
        
        # 训练迭代次数和保存频率
        max_iterations = 10000
        save_interval = 500
        
        # 加载和恢复配置
        resume = False
        
    # wandb配置
    do_wandb = True
    wandb_project = 'PBRS_HumanoidMoE'
    wandb_group = 'PBRS_HumanoidMoE' 