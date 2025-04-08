"""
MoE模型类，整合门控网络和两个专家模型（行走和奔跑）
"""

import os
import numpy as np

# 然后导入torch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpugym.envs.PBRS.gating_network import GatingNetwork

class MoEModel(nn.Module):
    """
    混合专家模型，整合门控网络和专家模型
    """
    def __init__(self, walk_model_path, run_model_path, obs_dim, action_dim, device='cuda'):
        """
        初始化混合专家模型
        
        参数:
        - walk_model_path: 行走模型路径
        - run_model_path: 奔跑模型路径
        - obs_dim: 观察空间维度
        - action_dim: 动作空间维度
        - device: 运行设备
        """
        super(MoEModel, self).__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 初始化门控网络
        self.gating_network = GatingNetwork(input_dim=obs_dim, device=device)
        
        # 加载专家模型（行走和奔跑）- 只加载actor部分
        self.walk_model = None
        self.run_model = None
        self.walk_model_path = walk_model_path
        self.run_model_path = run_model_path
        
        # 模型将在load_experts方法中加载
        
    def load_experts(self, actor_critic_class):
        """
        加载专家模型
        
        参数:
        - actor_critic_class: ActorCritic类，用于初始化专家模型
        """
        # 加载行走模型
        print(f"加载行走模型: {self.walk_model_path}")
        self.walk_model = actor_critic_class(self.obs_dim, self.obs_dim, self.action_dim)
        walk_checkpoint = torch.load(self.walk_model_path, map_location=self.device)
        self.walk_model.load_state_dict(walk_checkpoint['model_state_dict'])
        self.walk_model.eval()  # 设置为评估模式
        
        # 禁止行走模型参数更新
        for param in self.walk_model.parameters():
            param.requires_grad = False
            
        # 加载奔跑模型
        print(f"加载奔跑模型: {self.run_model_path}")
        self.run_model = actor_critic_class(self.obs_dim, self.obs_dim, self.action_dim)
        run_checkpoint = torch.load(self.run_model_path, map_location=self.device)
        self.run_model.load_state_dict(run_checkpoint['model_state_dict'])
        self.run_model.eval()  # 设置为评估模式
        
        # 禁止奔跑模型参数更新
        for param in self.run_model.parameters():
            param.requires_grad = False
    
    def forward(self, observations):
        """
        前向传播，根据观察计算专家模型权重并融合生成动作
        
        参数:
        - observations: 环境观察
        
        返回:
        - actions: 融合后的动作
        - dominant_expert: 权重最大的专家索引
        - weights: 每个专家的权重
        """
        # 确保模型已加载
        assert self.walk_model is not None and self.run_model is not None, "请先调用load_experts方法加载专家模型"
        
        # 使用门控网络获取混合权重
        weights, dominant_expert = self.gating_network.get_mixture_weights(observations)
        
        # 获取行走模型的动作
        with torch.no_grad():
            self.walk_model.act_inference(observations)
            walk_actions = self.walk_model.action_mean
        
        # 获取奔跑模型的动作
        with torch.no_grad():
            self.run_model.act_inference(observations)
            run_actions = self.run_model.action_mean
        
        # 融合两个专家模型的动作
        batch_size = observations.shape[0]
        actions = torch.zeros((batch_size, self.action_dim), device=self.device)
        
        # 通过权重融合两个专家的动作
        walk_weights = weights[:, 0].unsqueeze(1)  # 将权重扩展为 [batch_size, 1]
        run_weights = weights[:, 1].unsqueeze(1)   # 将权重扩展为 [batch_size, 1]
        
        # 加权组合两个专家的输出
        actions = walk_weights * walk_actions + run_weights * run_actions

        #test
        print("Weights:", weights.cpu().numpy())  # 打印权重
        print("Walk actions:", walk_actions.cpu().numpy())  # 打印行走模型的动作
        print("Run actions:", run_actions.cpu().numpy())  # 打印奔跑模型的动作
        print("Final actions:", actions.cpu().numpy())  # 最终动作
        
        return actions, dominant_expert, weights
        
    def act(self, observations):
        """
        生成动作
        
        参数:
        - observations: 环境观察
        
        返回:
        - actions: 融合后的动作
        """
        with torch.no_grad():
            actions, _, _ = self.forward(observations)
        return actions 