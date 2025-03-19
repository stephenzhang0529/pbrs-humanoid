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
        
        # 专家选择结果
        self.selected_expert = None
        self.expert_probs = None
        
    def reset(self, dones=None):
        """重置状态"""
        pass
    
    def forward(self):
        """前向传播 - 需要在act中实现"""
        raise NotImplementedError
    
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
        # 获取门控网络输出
        gate_logits = self.gating_network(observations)
        self.expert_probs = F.softmax(gate_logits, dim=-1)
        
        # 使用actor网络获取动作
        self.actions_mean = self.actor(observations)
        self.actions_std = self.std.expand_as(self.actions_mean)
        
        # 创建分布
        self.distribution = Normal(self.actions_mean, self.actions_std)
        
    def act(self, observations, **kwargs):
        """生成动作"""
        self.update_distribution(observations)
        
        # 专家选择 - 训练时采样，评估时贪婪
        if self.training:
            # 训练时使用采样
            self.selected_expert = torch.multinomial(self.expert_probs, 1).squeeze(-1)
        else:
            # 评估时使用贪婪策略
            self.selected_expert = torch.argmax(self.expert_probs, dim=-1)
        
        # 从分布中采样动作
        actions = self.distribution.sample()
        self.actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        
        # 计算熵
        self._entropy = self.distribution.entropy().sum(dim=-1)
        
        # 截断动作到有效范围
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # 只返回动作，专家选择存储在self.selected_expert中
        return actions
    
    def get_actions_log_prob(self, actions):
        """获取动作的对数概率"""
        log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        # 确保返回形状为[batch_size, 1]
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(1)
        return log_prob
    
    def act_inference(self, observations):
        """推理模式下的动作生成"""
        self.update_distribution(observations)
        
        # 推理时使用确定性策略
        self.selected_expert = torch.argmax(self.expert_probs, dim=-1)
        
        # 使用动作均值（不添加噪声）
        return self.actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        """评估状态值"""
        value = self.critic(critic_observations)
        # 不要squeeze，保持形状为[batch_size, 1]
        self.critic_values = value
        return self.critic_values

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