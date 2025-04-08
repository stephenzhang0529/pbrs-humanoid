"""
混合专家Actor-Critic模型，包含门控网络和专家选择机制
"""

import numpy as np
from typing import Dict, List
import copy

# 然后导入torch相关模块
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class MoEActorCritic(nn.Module):
    """混合专家Actor-Critic模型类"""
    
    is_recurrent = False
    
    def __init__(self, 
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 gating_hidden_dims=[256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        """
        初始化混合专家Actor-Critic模型
        
        参数:
        - num_actor_obs: actor观察空间维度
        - num_critic_obs: critic观察空间维度
        - num_actions: 动作空间维度
        - actor_hidden_dims: actor网络隐藏层维度
        - critic_hidden_dims: critic网络隐藏层维度
        - gating_hidden_dims: 门控网络隐藏层维度
        - activation: 激活函数
        - init_noise_std: 初始噪声标准差
        """
        if kwargs:
            print("MoEActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MoEActorCritic, self).__init__()
        
        activation_fn = get_activation(activation)
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        
        # 专家模型相关
        self.experts = None
        self.experts_loaded = False
        self.expert_probs = None
        self.selected_expert = None
        self.training_steps = 0
        
        # 存储最近的动作和观察
        self.actions_mean = None
        self.actions_std = None
        self.actions_log_prob = None
        self._entropy = None
        self.critic_values = None
        
        # 门控网络 - 决定使用哪个专家
        layers = []
        prev_dim = num_actor_obs
        
        for dim in gating_hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_fn)
            prev_dim = dim
        
        # 输出是2个专家的概率
        layers.append(nn.Linear(prev_dim, 2))
        
        self.gating_network = nn.Sequential(*layers)
        print(f"Gating Network: {self.gating_network}")
        
        # Actor网络 - 作为门控网络的强化学习反馈
        actor_layers = []
        prev_dim = num_actor_obs
        
        for dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, dim))
            actor_layers.append(activation_fn)
            prev_dim = dim
        
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)
        print(f"Actor Network: {self.actor}")
        
        # Critic网络
        critic_layers = []
        prev_dim = num_critic_obs
        
        for dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, dim))
            critic_layers.append(activation_fn)
            prev_dim = dim
        
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        print(f"Critic Network: {self.critic}")
        
        # 动作噪声参数
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        
        # 存储分布
        self.distribution = None
        
    def reset(self, dones=None):
        """重置状态"""
        pass
    
    def forward(self):
        """前向传播 - 需要在act中实现"""
        raise NotImplementedError
    
    def load_experts(self, walk_expert, run_expert):
        """
        加载专家模型
        
        参数:
        - walk_expert: 行走专家模型
        - run_expert: 奔跑专家模型
        """
        self.experts = [walk_expert, run_expert]
        self.experts_loaded = True
        print("已加载专家模型：行走和奔跑")
    
    @property
    def action_mean(self):
        """返回动作均值"""
        return self.actions_mean
    
    @property
    def action_std(self):
        """返回动作标准差"""
        return self.actions_std
    
    @property
    def entropy(self):
        """返回熵"""
        return self._entropy
    
    def update_distribution(self, observations):
        """更新动作分布"""
        # 确保observations是张量而不是元组或其他类型
        if not isinstance(observations, torch.Tensor):
            print(f"警告：在update_distribution中，observations不是张量，而是{type(observations)}。尝试转换。")
            if isinstance(observations, tuple) and len(observations) > 0:
                observations = observations[0]  # 取第一个元素
            observations = torch.as_tensor(observations, device=self.std.device)
            
        # 获取门控网络输出
        gate_logits = self.gating_network(observations)
        
        # 使用softmax将logits转换为专家权重
        self.expert_probs = F.softmax(gate_logits, dim=-1)
        
        # 存储权重最大的专家（仅用于分析）
        self.selected_expert = torch.argmax(self.expert_probs, dim=-1)
        
        if self.experts_loaded:
            # 无论是否在训练模式，只要专家加载了就使用混合机制
            with torch.no_grad():
                # 获取各专家的动作
                walk_actions_mean = self.experts[0].act_inference(observations)
                run_actions_mean = self.experts[1].act_inference(observations)
                
                # 扩展权重维度以便进行广播
                walk_weights = self.expert_probs[:, 0].unsqueeze(1)
                run_weights = self.expert_probs[:, 1].unsqueeze(1)
                
                # 混合专家动作
                expert_actions_mean = walk_weights * walk_actions_mean + run_weights * run_actions_mean
                
                # 在训练模式下，我们直接使用专家动作并添加一些噪声
                # 这样可以保证初始动作是有效的，同时也允许探索
                self.actions_mean = expert_actions_mean
        else:
            # 如果专家未加载，则使用actor网络
            self.actions_mean = self.actor(observations)
            
        # 增大初始阶段的标准差，促进更多探索
        if hasattr(self, 'training_steps'):
            self.training_steps += 1
            # 随着训练进行，逐渐减小噪声
            decay_factor = max(0.5, min(1.0, 4000 / (self.training_steps + 4000)))
            current_std = self.std * decay_factor
        else:
            self.training_steps = 0
            current_std = self.std
            
        self.actions_std = current_std.expand_as(self.actions_mean)
        
        # 创建分布
        self.distribution = Normal(self.actions_mean, self.actions_std)
    
    def act(self, observations, **kwargs):
        """生成动作"""
        # 确保observations是张量而不是元组或其他类型
        if not isinstance(observations, torch.Tensor):
            print(f"警告：observations不是张量，而是{type(observations)}。尝试转换。")
            if isinstance(observations, tuple) and len(observations) > 0:
                observations = observations[0]  # 取第一个元素
            observations = torch.as_tensor(observations, device=self.std.device)
            
        self.update_distribution(observations)
        
        # 从分布中采样动作
        actions = self.distribution.sample()
        self.actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        
        # 计算熵
        self._entropy = self.distribution.entropy().sum(dim=-1)
        
        # 截断动作到有效范围
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # 检查动作是否全为0或NaN，这可能导致训练问题
        if torch.all(torch.abs(actions) < 1e-6) or torch.isnan(actions).any():
            print(f"警告：生成的动作异常，值为 {actions[0]}")
            
            # 如果专家已加载，则直接使用专家动作作为备选
            if self.experts_loaded:
                # 生成一个随机比例，偏向于使用专家的动作
                expert_idx = torch.randint(0, 2, (observations.shape[0],), device=self.std.device)
                expert_actions = []
                
                for i in range(2):  # 两个专家
                    with torch.no_grad():
                        if i == 0:
                            expert_act = self.experts[0].act_inference(observations)
                        else:
                            expert_act = self.experts[1].act_inference(observations)
                        expert_actions.append(expert_act)
                
                # 为每个样本选择对应专家的动作
                mask = torch.zeros((observations.shape[0], 2), device=self.std.device)
                mask.scatter_(1, expert_idx.unsqueeze(1), 1)
                
                # 混合专家动作 (batch_size, 2, action_dim) * (batch_size, 2, 1) -> (batch_size, action_dim)
                expert_actions = torch.stack(expert_actions, dim=1)  # (batch_size, 2, action_dim)
                mask = mask.unsqueeze(-1)  # (batch_size, 2, 1)
                mixed_actions = (expert_actions * mask).sum(dim=1)  # (batch_size, action_dim)
                
                print(f"使用专家动作替代: {mixed_actions[0]}")
                actions = mixed_actions
        
        # 为了与 PPO 算法兼容，只返回动作张量
        return actions
        
    def get_expert_info(self):
        """
        获取专家信息（用于推理阶段或其他需要专家信息的场景）
        
        返回:
        - dominant_expert: 权重最大的专家索引
        - expert_probs: 各专家的权重
        """
        return self.selected_expert, self.expert_probs
    
    def get_actions_log_prob(self, actions):
        """获取动作的对数概率"""
        log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        # 确保返回形状为[batch_size, 1]
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(1)
        return log_prob
    
    def evaluate(self, critic_observations, **kwargs):
        """评估观察值，返回critic的价值估计"""
        # 确保observations是张量而不是元组或其他类型
        if not isinstance(critic_observations, torch.Tensor):
            if isinstance(critic_observations, tuple) and len(critic_observations) > 0:
                critic_observations = critic_observations[0]  # 取第一个元素
            critic_observations = torch.as_tensor(critic_observations, device=next(self.critic.parameters()).device)
        
        value = self.critic(critic_observations)
        self.critic_values = value
        return value
    
    def act_with_expert_info(self, observations):
        """
        生成动作并返回专家信息（用于推理阶段）
        
        参数:
        - observations: 环境观察
        
        返回:
        - actions: 动作
        - dominant_expert: 权重最大的专家索引  
        - expert_probs: 各专家的权重
        """
        actions = self.act(observations)
        dominant_expert, expert_probs = self.get_expert_info()
        return actions, dominant_expert, expert_probs
    
    def act_inference(self, observations):
        """推理模式下的动作生成"""
        self.update_distribution(observations)
        
        # 使用动作均值（不添加噪声）
        return self.actions_mean
        
    def act_inference_with_expert_info(self, observations):
        """
        推理模式下的动作生成，同时返回专家信息
        
        参数:
        - observations: 环境观察
        
        返回:
        - actions: 动作均值（不添加噪声）
        - dominant_expert: 权重最大的专家索引
        - expert_probs: 各专家的权重
        """
        self.act_inference(observations)
        dominant_expert, expert_probs = self.get_expert_info()
        return self.actions_mean, dominant_expert, expert_probs

def get_activation(act_name):
    """获取激活函数"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None 