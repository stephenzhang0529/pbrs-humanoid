"""
混合专家(MoE)系统实现文件
实现了机器人在不同地形上动态选择行走或奔跑模式的混合专家系统
"""

import os

from collections import deque
import random

from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gpugym.utils.math import *
from gpugym.envs.PBRS.humanoid import Humanoid
from gpugym.envs.PBRS.moe_config import HumanoidMoECfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, terrain_features, action, reward, next_state, next_terrain_features, done):
        self.buffer.append((state, terrain_features, action, reward, next_state, next_terrain_features, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, terrain_features, actions, rewards, next_states, next_terrain_features, dones = zip(*samples)
        
        return (torch.stack(states), 
                torch.stack(terrain_features), 
                torch.stack(actions), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.stack(next_states), 
                torch.stack(next_terrain_features),
                torch.tensor(dones, dtype=torch.float32))
    
    def __len__(self):
        return len(self.buffer)


# 加载预训练的专家模型
class ExpertModels:
    def __init__(self, walk_model_path, run_model_path, device='cuda'):
        """
        加载行走和奔跑专家模型
        
        Args:
            walk_model_path: 行走模型路径
            run_model_path: 奔跑模型路径
            device: 设备类型
        """
        self.device = device
        self.walk_model = self._load_model(walk_model_path)
        self.run_model = self._load_model(run_model_path)
        
        print(f"已加载行走模型：{walk_model_path}")
        print(f"已加载奔跑模型：{run_model_path}")
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        actor_critic = checkpoint['model_state_dict']
        return actor_critic
    
    def get_walk_action(self, obs):
        """获取行走模型的动作"""
        with torch.no_grad():
            actions = self._get_action_from_checkpoint(self.walk_model, obs)
        return actions
    
    def get_run_action(self, obs):
        """获取奔跑模型的动作"""
        with torch.no_grad():
            actions = self._get_action_from_checkpoint(self.run_model, obs)
        return actions
    
    def _get_action_from_checkpoint(self, model_state_dict, obs):
        """从checkpoint获取动作
        
        注意：这里假设模型checkpoint中包含actor和critic网络权重，
        并且actor网络可以单独使用来生成动作。
        """
        # 假设actor网络权重可以直接用于推理
        # 实际实现可能需要根据模型结构进行调整
        with torch.no_grad():
            mean = model_state_dict['actor.0.running_mean']
            var = model_state_dict['actor.0.running_var']
            weight = model_state_dict['actor.0.weight']
            bias = model_state_dict['actor.0.bias']
            
            # 标准化输入
            obs_norm = (obs - mean) / torch.sqrt(var + 1e-5)
            x = F.linear(obs_norm, weight, bias)
            x = F.elu(x)
            
            # 第二层
            weight = model_state_dict['actor.2.weight']
            bias = model_state_dict['actor.2.bias']
            x = F.linear(x, weight, bias)
            x = F.elu(x)
            
            # 第三层
            weight = model_state_dict['actor.4.weight']
            bias = model_state_dict['actor.4.bias']
            x = F.linear(x, weight, bias)
            x = F.elu(x)
            
            # 输出层
            weight = model_state_dict['actor.6.weight']
            bias = model_state_dict['actor.6.bias']
            x = F.linear(x, weight, bias)
            
            # 没有激活函数，直接输出动作
            return x


# 门控网络：决定如何融合两个专家模型
class GatingNetwork(nn.Module):
    def __init__(self, terrain_dim, robot_state_dim, hidden_dims=[128, 128]):
        super(GatingNetwork, self).__init__()
        
        # 输入层：地形特征 + 机器人状态
        input_dim = terrain_dim + robot_state_dim
        
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # 输出层：产生两个权重（行走和奔跑）
        layers.append(nn.Linear(prev_dim, 2))
        layers.append(nn.Softmax(dim=1))  # 使用Softmax确保权重和为1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, terrain_features, robot_state):
        """
        前向传播：计算混合权重
        
        Args:
            terrain_features: 地形特征，形状为 [batch_size, terrain_dim]
            robot_state: 机器人状态，形状为 [batch_size, robot_state_dim]
            
        Returns:
            混合权重，形状为 [batch_size, 2]
        """
        # 拼接地形特征和机器人状态
        combined = torch.cat([terrain_features, robot_state], dim=1)
        
        # 计算混合权重
        weights = self.network(combined)
        
        return weights


# 非监督奖励模块
class UnsupervisedRewardModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(UnsupervisedRewardModule, self).__init__()
        
        # 用于预测下一个状态的前向模型
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 用于好奇心奖励的反向模型
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def compute_curiosity_reward(self, state, action, next_state):
        """
        计算好奇心奖励：预测误差越大，奖励越高
        
        Args:
            state: 当前状态 [batch_size, state_dim]
            action: 动作 [batch_size, action_dim]
            next_state: 下一个状态 [batch_size, state_dim]
            
        Returns:
            好奇心奖励 [batch_size]
        """
        # 预测下一个状态
        state_action = torch.cat([state, action], dim=1)
        predicted_next_state = self.forward_model(state_action)
        
        # 计算预测误差（作为好奇心奖励）
        prediction_error = F.mse_loss(predicted_next_state, next_state, reduction='none').mean(dim=1)
        
        return prediction_error
    
    def compute_balance_reward(self, state):
        """
        计算平衡奖励：重力投影越接近垂直，奖励越高
        
        Args:
            state: 当前状态 [batch_size, state_dim]
            
        Returns:
            平衡奖励 [batch_size]
        """
        # 假设state中的索引7-9是投影重力向量
        gravity_projection = state[:, 7:10]
        
        # 计算与垂直方向的偏差（偏差越小，奖励越高）
        ideal_gravity = torch.zeros_like(gravity_projection)
        ideal_gravity[:, 2] = -1.0  # 假设z轴指向上方，理想重力方向是(0, 0, -1)
        
        balance_error = F.mse_loss(gravity_projection, ideal_gravity, reduction='none').sum(dim=1)
        balance_reward = 1.0 / (1.0 + balance_error)  # 转换为奖励（0到1之间）
        
        return balance_reward
    
    def compute_velocity_reward(self, state):
        """
        计算速度奖励：速度越快，奖励越高
        
        Args:
            state: 当前状态 [batch_size, state_dim]
            
        Returns:
            速度奖励 [batch_size]
        """
        # 假设state中的索引1-3是基础线性速度
        lin_vel = state[:, 1:4]
        
        # 计算水平速度（x-y平面）的大小
        velocity_xy = torch.norm(lin_vel[:, :2], dim=1)
        
        # 速度越高，奖励越大（使用tanh函数避免奖励无限增长）
        velocity_reward = torch.tanh(velocity_xy)
        
        return velocity_reward
    
    def compute_reward(self, state, action, next_state):
        """
        综合计算非监督奖励
        
        Args:
            state: 当前状态 [batch_size, state_dim]
            action: 动作 [batch_size, action_dim]
            next_state: 下一个状态 [batch_size, state_dim]
            
        Returns:
            奖励 [batch_size]，奖励组成部分（字典）
        """
        curiosity_reward = self.compute_curiosity_reward(state, action, next_state)
        balance_reward = self.compute_balance_reward(state)
        velocity_reward = self.compute_velocity_reward(state)
        
        # 返回总奖励和各部分奖励（用于调试和分析）
        reward_components = {
            'curiosity': curiosity_reward.detach().cpu().numpy(),
            'balance': balance_reward.detach().cpu().numpy(),
            'velocity': velocity_reward.detach().cpu().numpy()
        }
        
        return curiosity_reward, balance_reward, velocity_reward, reward_components


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
        # 如果提供了外部sim，使用它而不是创建新的
        self.external_sim = external_sim
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, external_sim=external_sim)
        
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
    
    def step(self, actions):
        """
        执行一个仿真步骤
        
        Args:
            actions: 动作（在MoE中不使用，由内部计算）
            
        Returns:
            下一个状态，额外信息
        """
        # 存储当前观察
        current_obs = self.obs_buf.clone()
        
        # 提取地形特征
        self._extract_terrain_features()
        
        # 使用门控网络计算混合权重
        blend_weights = self.gating_network(
            self.terrain_features,
            current_obs
        )
        
        # 从两个专家获取动作
        walk_actions = self.experts.get_walk_action(current_obs)
        run_actions = self.experts.get_run_action(current_obs)
        
        # 混合动作
        blended_actions = blend_weights[:, 0:1] * walk_actions + blend_weights[:, 1:2] * run_actions
        
        # 应用动作平滑（可选）
        if self.action_smoothing > 0:
            blended_actions = self.action_smoothing * self.last_actions + (1 - self.action_smoothing) * blended_actions
            self.last_actions = blended_actions.clone()
        
        # 使用父类的step方法执行动作
        obs, privileged_obs, rewards, dones, infos = super().step(blended_actions)
        
        # 存储经验（用于训练门控网络）
        for i in range(self.num_envs):
            self.replay_buffer.add(
                current_obs[i],
                self.terrain_features[i],
                blended_actions[i],
                rewards[i],
                obs[i],
                self.terrain_features[i],  # 假设地形特征在一步内不变
                dones[i]
            )
        
        # 计算非监督奖励
        curiosity_reward, balance_reward, velocity_reward, reward_components = self.reward_module.compute_reward(
            current_obs,
            blended_actions,
            obs
        )
        
        # 修改奖励
        rewards = (
            self.cfg.moe.balance_reward_scale * balance_reward +
            self.cfg.moe.velocity_reward_scale * velocity_reward +
            self.cfg.moe.curiosity_reward_scale * curiosity_reward
        )
        
        # 训练门控网络
        if len(self.replay_buffer) > self.batch_size:
            self._train_gating_network()
        
        # 记录数据
        self.blend_weights_history.append(blend_weights.detach().cpu().numpy())
        self.reward_components_history.append(reward_components)
        
        # 添加额外信息
        infos['blend_weights'] = blend_weights
        infos['reward_components'] = reward_components
        
        return obs, privileged_obs, rewards, dones, infos
    
    def _extract_terrain_features(self):
        """提取地形特征"""
        # 使用_get_heights获取高度测量
        if self.cfg.terrain.measure_heights:
            heights = self._get_heights()
            
            # 计算地形粗糙度（相邻高度差的标准差）
            heights_reshaped = heights.view(self.num_envs, -1, 1)
            roughness = torch.std(heights_reshaped[:, 1:] - heights_reshaped[:, :-1], dim=1).squeeze()
            
            # 提取主要特征（这里简化为均值、标准差、最大值、最小值等）
            mean_height = torch.mean(heights, dim=1)
            std_height = torch.std(heights, dim=1)
            max_height = torch.max(heights, dim=1)[0]
            min_height = torch.min(heights, dim=1)[0]
            
            # 使用这些统计特征作为地形特征的一部分
            features = [mean_height, std_height, max_height, min_height, roughness]
            
            # 添加一些原始高度样本（选择有代表性的点）
            num_sample_points = min(45, heights.shape[1])
            indices = torch.linspace(0, heights.shape[1]-1, num_sample_points, dtype=torch.long)
            sampled_heights = heights[:, indices]
            
            # 拼接所有特征
            all_features = torch.cat([torch.stack(features, dim=1), sampled_heights], dim=1)
            
            # 确保维度与terrain_feature_dim匹配
            if all_features.shape[1] > self.cfg.moe.terrain_feature_dim:
                all_features = all_features[:, :self.cfg.moe.terrain_feature_dim]
            elif all_features.shape[1] < self.cfg.moe.terrain_feature_dim:
                padding = torch.zeros(self.num_envs, self.cfg.moe.terrain_feature_dim - all_features.shape[1], device=self.device)
                all_features = torch.cat([all_features, padding], dim=1)
            
            self.terrain_features = all_features
        else:
            # 如果不测量高度，使用零向量
            self.terrain_features = torch.zeros((self.num_envs, self.cfg.moe.terrain_feature_dim), device=self.device)
    
    def _train_gating_network(self):
        """训练门控网络"""
        # 从回放缓冲区采样
        states, terrain_features, actions, rewards, next_states, next_terrain_features, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 转移到正确的设备
        states = states.to(self.device)
        terrain_features = terrain_features.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        # 零梯度
        self.gating_optimizer.zero_grad()
        
        # 计算当前状态下的混合权重
        blend_weights = self.gating_network(terrain_features, states)
        
        # 重新计算两个专家的动作
        walk_actions = self.experts.get_walk_action(states)
        run_actions = self.experts.get_run_action(states)
        
        # 计算混合动作
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