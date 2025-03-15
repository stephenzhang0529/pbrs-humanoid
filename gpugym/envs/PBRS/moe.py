"""
混合专家(MoE)系统实现文件

该文件实现了一个混合专家系统，结合了行走和奔跑模型，
通过非监督强化学习方法使机器人在不同地形上能够稳定且快速地运动。
"""
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from collections import deque
import random
from gpugym.envs.PBRS.humanoid import Humanoid
from gpugym.envs.PBRS.moe_config import HumanoidMoECfg


class GatingNetwork(nn.Module):
    """
    门控网络，根据地形和机器人状态决定行走和奔跑模型的混合权重
    """
    def __init__(self, terrain_dim, robot_state_dim, hidden_dims=[128, 128]):
        super(GatingNetwork, self).__init__()
        
        # 输入维度是地形特征和机器人状态维度的总和
        input_dim = terrain_dim + robot_state_dim
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ELU())
            prev_dim = dim
        
        # 输出层 - 2个输出分别代表行走和奔跑模型的权重
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, terrain_features, robot_state):
        """
        前向传播
        
        Args:
            terrain_features: 地形特征，形状为 [batch_size, terrain_dim]
            robot_state: 机器人状态，形状为 [batch_size, robot_state_dim]
            
        Returns:
            2D向量表示混合权重，总和为1
        """
        # 连接地形特征和机器人状态
        x = torch.cat([terrain_features, robot_state], dim=1)
        
        # 通过网络前向传播
        logits = self.network(x)
        
        # 使用softmax确保权重和为1
        return F.softmax(logits, dim=1)


class UnsupervisedRewardModule(nn.Module):
    """
    非监督奖励模块，基于好奇心、平衡预测和速度潜力生成内在奖励
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(UnsupervisedRewardModule, self).__init__()
        
        # 前向动力学模型（预测给定当前状态和动作的下一个状态）
        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 平衡预测器（估计保持平衡的概率）
        self.balance_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 速度潜力估计器
        self.velocity_estimator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
    
    def forward(self, state, action, next_state):
        """
        前向传播，计算各种奖励信号
        
        Args:
            state: 当前状态
            action: 当前动作
            next_state: 下一个状态
            
        Returns:
            好奇心奖励，平衡概率，速度潜力
        """
        # 预测下一个状态
        state_action = torch.cat([state, action], dim=1)
        pred_next_state = self.dynamics_model(state_action)
        
        # 计算好奇心奖励（预测误差）
        curiosity_reward = torch.norm(pred_next_state - next_state, dim=1, keepdim=True)
        
        # 计算平衡概率
        balance_prob = self.balance_predictor(state)
        
        # 计算速度潜力
        velocity_potential = self.velocity_estimator(state)
        
        return curiosity_reward, balance_prob, velocity_potential
    
    def calculate_reward(self, state, action, next_state, actual_velocity):
        """
        计算最终的奖励，并更新模型
        
        Args:
            state: 当前状态
            action: 当前动作
            next_state: 下一个状态
            actual_velocity: 实际速度
            
        Returns:
            合并的最终奖励
        """
        self.optimizer.zero_grad()
        
        # 前向传播
        curiosity_reward, balance_prob, velocity_potential = self(state, action, next_state)
        
        # 更新模型
        # 1. 动力学模型损失
        dynamics_loss = F.mse_loss(
            self.dynamics_model(torch.cat([state, action], dim=1)), 
            next_state
        )
        
        # 2. 平衡预测器损失（使用重力投影作为平衡指标）
        # 重力投影在xy平面的平方和越小，平衡性越好
        balance_target = torch.exp(-torch.sum(state[:, 7:9]**2, dim=1, keepdim=True))
        balance_loss = F.mse_loss(self.balance_predictor(state), balance_target)
        
        # 3. 速度估计器损失（回归到实际速度）
        # 提取基础线性速度
        actual_vel_magnitude = torch.norm(state[:, 1:4], dim=1, keepdim=True)
        velocity_loss = F.mse_loss(self.velocity_estimator(state), actual_vel_magnitude)
        
        # 组合损失
        total_loss = dynamics_loss + balance_loss + velocity_loss
        total_loss.backward()
        self.optimizer.step()
        
        # 计算最终奖励
        # 对好奇心进行归一化，避免危险的探索
        normalized_curiosity = torch.tanh(curiosity_reward)
        
        # 组合奖励：平衡很关键，速度是目标，好奇心是探索
        reward = (2.0 * balance_prob) + (1.0 * actual_velocity) + (0.1 * normalized_curiosity)
        
        return reward


class ExpertModels:
    """
    专家模型的包装类，用于加载和使用预训练的行走和奔跑模型
    """
    def __init__(self, walk_model_path, run_model_path, device="cuda"):
        self.device = device
        
        # 加载预训练模型
        self.walk_model = self.load_model(walk_model_path)
        self.run_model = self.load_model(run_model_path)
    
    def load_model(self, model_path):
        """
        加载预训练的PyTorch模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的模型
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        model = torch.load(model_path, map_location=self.device)
        model.eval()  # 设置为评估模式
        return model
    
    def get_walk_action(self, state):
        """
        从行走模型获取动作
        
        Args:
            state: 当前状态
            
        Returns:
            行走模型的动作
        """
        with torch.no_grad():
            action = self.walk_model.act_inference(state)
        return action
    
    def get_run_action(self, state):
        """
        从奔跑模型获取动作
        
        Args:
            state: 当前状态
            
        Returns:
            奔跑模型的动作
        """
        with torch.no_grad():
            action = self.run_model.act_inference(state)
        return action


class ReplayBuffer:
    """
    经验回放缓冲区，用于存储和采样经验
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, terrain_features, action, next_state, next_terrain_features, reward, done):
        """
        添加经验到缓冲区
        """
        self.buffer.append((state, terrain_features, action, next_state, next_terrain_features, reward, done))
    
    def sample(self, batch_size):
        """
        从缓冲区随机采样经验
        
        Args:
            batch_size: 批量大小
            
        Returns:
            批量的经验数据
        """
        samples = random.sample(self.buffer, batch_size)
        states, terrain_features, actions, next_states, next_terrain_features, rewards, dones = zip(*samples)
        
        return (
            torch.cat(states, dim=0),
            torch.cat(terrain_features, dim=0),
            torch.cat(actions, dim=0),
            torch.cat(next_states, dim=0),
            torch.cat(next_terrain_features, dim=0),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        )
    
    def __len__(self):
        """
        返回缓冲区中的经验数量
        """
        return len(self.buffer)


class HumanoidMoE(Humanoid):
    """
    混合专家系统的主类，继承自Humanoid
    """
    def __init__(self, cfg: HumanoidMoECfg, sim_params, physics_engine, sim_device, headless, external_sim=None):
        """
        初始化混合专家系统
        
        Args:
            cfg: 配置对象
            sim_params: 仿真参数
            physics_engine: 物理引擎
            sim_device: 仿真设备
            headless: 是否无头模式
            external_sim: 外部提供的sim实例（如果有）
        """
        # 优先使用外部sim实例
        if external_sim is not None:
            self.sim = external_sim
            self.external_sim = True
        else:
            self.external_sim = False

        # 调用父类初始化（假设父类支持external_sim）
        super().__init__(
            cfg,
            sim_params,
            physics_engine,
            sim_device,
            headless,
            external_sim=external_sim  # 确保父类使用外部sim
        )
        
        self.cfg = cfg
        self.device = sim_device
        
        # 加载专家模型
        self.experts = ExpertModels(
            cfg.moe.walk_model_path,
            cfg.moe.run_model_path,
            device=sim_device
        )
        
        # 初始化门控网络
        self.gating_network = GatingNetwork(
            terrain_dim=cfg.moe.terrain_feature_dim,
            robot_state_dim=cfg.env.num_observations,
            hidden_dims=cfg.moe.gating_hidden_dims
        ).to(sim_device)
        
        # 初始化非监督奖励模块
        self.reward_module = UnsupervisedRewardModule(
            state_dim=cfg.env.num_observations,
            action_dim=cfg.env.num_actions,
            hidden_dim=128
        ).to(sim_device)
        
        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=cfg.moe.replay_buffer_size)
        
        # 初始化优化器
        self.gating_optimizer = optim.Adam(
            self.gating_network.parameters(),
            lr=cfg.moe.learning_rate
        )
        
        # 其他参数
        self.batch_size = cfg.moe.batch_size
        self.gamma = cfg.moe.gamma
        self.terrain_features = torch.zeros(
            (self.num_envs, cfg.moe.terrain_feature_dim),
            device=sim_device
        )
        self.last_actions = torch.zeros(
            (self.num_envs, cfg.env.num_actions),
            device=sim_device
        )
        
        # 记录数据
        self.blend_weights_history = []
        self.reward_components_history = []
        
        # 用于动作平滑
        self.action_smoothing = cfg.moe.action_smoothing
        
        print(f"初始化HumanoidMoE完成，设备：{sim_device}")
    
    def process_terrain(self):
        """
        处理地形数据为特征向量
        
        Returns:
            地形特征向量
        """
        # 如果启用了地形高度测量
        if self.cfg.terrain.measure_heights:
            # 地形高度在self.measured_heights中
            
            # 计算地形的梯度（斜率）- x方向和y方向
            heights = self.measured_heights.view(self.num_envs, -1)
            x_size = len(self.cfg.terrain.measured_points_x)
            y_size = len(self.cfg.terrain.measured_points_y)
            
            # 重塑为2D网格以便计算梯度
            grid = heights.view(self.num_envs, y_size, x_size)
            
            # 计算x和y方向的梯度
            # 在张量的维度1和2上使用有限差分
            # 为简化计算，我们使用相邻点之间的差分
            
            # X方向梯度
            dx = torch.zeros((self.num_envs, y_size, x_size-1), device=self.device)
            for i in range(x_size-1):
                dx[:, :, i] = grid[:, :, i+1] - grid[:, :, i]
            
            # Y方向梯度
            dy = torch.zeros((self.num_envs, y_size-1, x_size), device=self.device)
            for i in range(y_size-1):
                dy[:, i, :] = grid[:, i+1, :] - grid[:, i, :]
            
            # 将梯度展平
            dx_flat = dx.view(self.num_envs, -1)
            dy_flat = dy.view(self.num_envs, -1)
            
            # 计算梯度幅值
            gradient_magnitude = torch.sqrt(dx_flat.pow(2).mean(dim=1, keepdim=True) + 
                                           dy_flat.pow(2).mean(dim=1, keepdim=True))
            
            # 计算高度统计特征
            height_mean = heights.mean(dim=1, keepdim=True)
            height_std = heights.std(dim=1, keepdim=True)
            height_max = heights.max(dim=1, keepdim=True)[0]
            height_min = heights.min(dim=1, keepdim=True)[0]
            height_range = height_max - height_min
            
            # 提取局部高度特征 - 前方、左前方、右前方
            center_idx = heights.shape[1] // 2
            front_idx = center_idx + x_size  # 前方点
            front_left_idx = front_idx - 1   # 左前方点
            front_right_idx = front_idx + 1  # 右前方点
            
            # 确保索引在范围内
            front_idx = min(front_idx, heights.shape[1]-1)
            front_left_idx = max(0, min(front_left_idx, heights.shape[1]-1))
            front_right_idx = min(front_right_idx, heights.shape[1]-1)
            
            height_front = heights[:, front_idx].unsqueeze(1)
            height_front_left = heights[:, front_left_idx].unsqueeze(1)
            height_front_right = heights[:, front_right_idx].unsqueeze(1)
            
            # 计算粗糙度 - 使用高度的局部变化
            # 我们可以使用高度标准差作为粗糙度的指标
            roughness = height_std
            
            # 构建地形特征向量
            features_list = [
                heights,  # 所有高度测量值
                gradient_magnitude,  # 梯度幅值
                height_mean,  # 平均高度
                height_std,  # 高度标准差
                height_range,  # 高度范围
                height_front,  # 前方高度
                height_front_left,  # 左前方高度
                height_front_right,  # 右前方高度
                roughness,  # 粗糙度
            ]
            
            # 连接所有特征
            terrain_features = torch.cat(features_list, dim=1)
            
            # 如果特征维度超过了预定的维度，使用PCA或截断
            if terrain_features.shape[1] > self.cfg.moe.terrain_feature_dim:
                # 简单截断到所需维度
                terrain_features = terrain_features[:, :self.cfg.moe.terrain_feature_dim]
            
            # 如果特征维度小于预定的维度，用零填充
            if terrain_features.shape[1] < self.cfg.moe.terrain_feature_dim:
                padding = torch.zeros(
                    (self.num_envs, self.cfg.moe.terrain_feature_dim - terrain_features.shape[1]),
                    device=self.device
                )
                terrain_features = torch.cat([terrain_features, padding], dim=1)
        else:
            # 如果没有启用地形高度测量，创建一个简单的特征向量
            terrain_features = torch.zeros(
                (self.num_envs, self.cfg.moe.terrain_feature_dim),
                device=self.device
            )
        
        return terrain_features
    
    def get_blend_weights(self, robot_state, terrain_features):
        """
        使用门控网络获取混合权重
        
        Args:
            robot_state: 机器人状态
            terrain_features: 地形特征
            
        Returns:
            混合权重 [行走权重, 奔跑权重]
        """
        with torch.no_grad():
            blend_weights = self.gating_network(terrain_features, robot_state)
        return blend_weights
    
    def step(self, actions):
        """
        覆盖父类的step方法，使用MoE系统选择动作
        
        这个方法不会真正使用传入的动作，而是使用MoE系统生成的动作
        
        Args:
            actions: 原始动作（被忽略）
            
        Returns:
            obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras
        """
        # 处理地形特征
        self.terrain_features = self.process_terrain()
        
        # 获取机器人状态
        robot_state = self.obs_buf
        
        # 使用门控网络获取混合权重
        blend_weights = self.get_blend_weights(robot_state, self.terrain_features)
        
        # 记录混合权重历史
        self.blend_weights_history.append(blend_weights.cpu().numpy())
        
        # 获取专家模型的动作
        walk_actions = self.experts.get_walk_action(robot_state)
        run_actions = self.experts.get_run_action(robot_state)
        
        # 混合动作
        moe_actions = blend_weights[:, 0:1] * walk_actions + blend_weights[:, 1:2] * run_actions
        
        # 应用动作平滑
        if hasattr(self, 'last_actions') and self.action_smoothing > 0:
            smoothed_actions = self.action_smoothing * self.last_actions + (1 - self.action_smoothing) * moe_actions
            self.last_actions = smoothed_actions
        else:
            smoothed_actions = moe_actions
            self.last_actions = moe_actions
        
        # 使用真实的动作执行步骤
        # 保存当前状态以供学习使用
        current_state = robot_state.clone()
        current_terrain = self.terrain_features.clone()
        
        # 调用父类的step方法
        super().step(smoothed_actions)
        
        # 获取下一个状态
        next_state = self.obs_buf
        next_terrain = self.process_terrain()
        
        # 计算非监督奖励
        if self.cfg.moe.use_unsupervised_rewards:
            # 提取实际速度
            actual_velocity = torch.norm(next_state[:, 1:4], dim=1, keepdim=True)
            
            # 计算奖励
            curiosity_reward, balance_prob, velocity_potential = self.reward_module(
                current_state, smoothed_actions, next_state
            )
            
            # 归一化好奇心
            normalized_curiosity = torch.tanh(curiosity_reward)
            
            # 使用缩放因子计算最终奖励
            reward = (
                self.cfg.moe.balance_reward_scale * balance_prob + 
                self.cfg.moe.velocity_reward_scale * actual_velocity + 
                self.cfg.moe.curiosity_reward_scale * normalized_curiosity
            )
            
            # 更新奖励缓冲区
            self.rew_buf = reward.squeeze(-1)
            
            # 记录奖励组件
            reward_components = {
                'balance': (self.cfg.moe.balance_reward_scale * balance_prob).mean().item(),
                'velocity': (self.cfg.moe.velocity_reward_scale * actual_velocity).mean().item(),
                'curiosity': (self.cfg.moe.curiosity_reward_scale * normalized_curiosity).mean().item(),
                'total': reward.mean().item()
            }
            self.reward_components_history.append(reward_components)
        
        # 将经验添加到回放缓冲区
        for i in range(self.num_envs):
            self.replay_buffer.add(
                current_state[i:i+1],
                current_terrain[i:i+1],
                smoothed_actions[i:i+1],
                next_state[i:i+1],
                next_terrain[i:i+1],
                self.rew_buf[i].item(),
                self.reset_buf[i].item()
            )
        
        # 训练门控网络
        self.train_gating_network()
        
        # 创建额外信息字典
        extras = {}
        extras['blend_weights'] = blend_weights
        extras['reward_components'] = reward_components if self.cfg.moe.use_unsupervised_rewards else None
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, extras
    
    def train_gating_network(self):
        """
        训练门控网络以最大化奖励
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从回放缓冲区采样
        states, terrain_features, actions, next_states, next_terrain_features, rewards, dones = self.replay_buffer.sample(self.batch_size)
        
        # 训练门控网络以最大化奖励
        self.gating_optimizer.zero_grad()
        
        # 获取混合权重
        blend_weights = self.gating_network(terrain_features, states)
        
        # 获取专家动作
        walk_actions = self.experts.get_walk_action(states)
        run_actions = self.experts.get_run_action(states)
        
        # 混合动作
        blended_actions = blend_weights[:, 0:1] * walk_actions + blend_weights[:, 1:2] * run_actions
        
        # 计算动作损失（MSE与实际采取的动作）
        action_loss = F.mse_loss(blended_actions, actions)
        
        # 计算奖励损失（目标是最大化奖励）
        reward_loss = -torch.mean(rewards)
        
        # 添加熵正则化以鼓励探索
        entropy = -(blend_weights * torch.log(blend_weights + 1e-10)).sum(dim=1).mean()
        entropy_loss = -0.01 * entropy  # 熵系数
        
        # 组合损失
        total_loss = action_loss + reward_loss + entropy_loss
        
        # 反向传播和优化
        total_loss.backward()
        self.gating_optimizer.step()
        
    def reset_idx(self, env_ids):
        """
        覆盖父类的reset_idx方法，重置MoE系统的状态
        
        Args:
            env_ids: 要重置的环境ID
        """
        # 调用父类的reset_idx
        super().reset_idx(env_ids)
        
        # 重置相关状态
        if len(env_ids) > 0:
            self.last_actions[env_ids] = 0
            
    def save_models(self, path):
        """
        保存训练的模型
        
        Args:
            path: 保存路径
        """
        save_dict = {
            'gating_network': self.gating_network.state_dict(),
            'reward_module': self.reward_module.state_dict(),
            'blend_weights_history': self.blend_weights_history,
            'reward_components_history': self.reward_components_history
        }
        torch.save(save_dict, path)
        print(f"模型已保存到 {path}")
    
    def load_models(self, path):
        """
        加载训练的模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.gating_network.load_state_dict(checkpoint['gating_network'])
        self.reward_module.load_state_dict(checkpoint['reward_module'])
        self.blend_weights_history = checkpoint.get('blend_weights_history', [])
        self.reward_components_history = checkpoint.get('reward_components_history', [])
        print(f"模型已从 {path} 加载")


def train_moe_system(cfg, num_episodes=100, gym=None, sim=None):
    """
    训练MoE系统的辅助函数
    
    Args:
        cfg: 配置对象
        num_episodes: 训练的回合数
        gym: 已创建的gym实例（如果有）
        sim: 已创建的sim实例（如果有）
        
    Returns:
        训练后的MoE系统
    """
    from isaacgym import gymapi
    from isaacgym import gymtorch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 检查是否提供了sim实例
    if sim is None:
        raise RuntimeError("必须提供sim实例，请使用IsaacGymManager创建并传入sim")
    
    # 确保gym实例也已存在
    if gym is None:
        raise RuntimeError("必须提供gym实例，请使用IsaacGymManager创建并传入gym")
    
    # 创建MoE系统
    print("正在创建MoE系统...")
    moe_system = HumanoidMoE(cfg, None, gymapi.SIM_PHYSX, device, False, external_sim=sim)
    
    # 训练循环
    rewards = []
    for episode in range(num_episodes):
        # 重置环境
        moe_system.reset()
        
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < cfg.env.episode_length_s * moe_system.control_freq:
            # 执行一步
            _, _, rew_buf, reset_buf, extras = moe_system.step(None)
            
            # 累计奖励
            episode_reward += rew_buf.mean().item()
            
            # 检查是否完成
            done = torch.any(reset_buf).item()
            step += 1
        
        rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"回合 {episode}, 奖励: {episode_reward}")
        
        if episode % cfg.moe.save_interval == 0:
            moe_system.save_models(f"moe_humanoid_checkpoint_{episode}.pt")
    
    return moe_system, rewards