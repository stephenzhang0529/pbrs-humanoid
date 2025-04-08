"""
动态门控网络模块，用于Mixture of Experts系统
负责根据当前状态（地形+机器人状态）计算两个专家模型（行走或奔跑）的组合权重
"""

import numpy as np

# 然后导入torch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    """
    门控网络，用于决定如何融合两个专家模型的输出
    """
    def __init__(self, input_dim, hidden_dims=[256, 128], device='cuda'):
        """
        初始化门控网络
        
        参数:
        - input_dim: 输入维度（观察空间维度）
        - hidden_dims: 隐藏层的维度
        - device: 运行的设备（'cuda'或'cpu'）
        """
        super(GatingNetwork, self).__init__()
        self.device = device
        
        # 构建MLP网络结构
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ELU())
            prev_dim = dim
        
        # 最后一层输出2个值，代表选择walking或running模型的权重
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observations):
        """
        前向传播，计算两个专家的混合权重
        
        参数:
        - observations: 环境的观察

        返回:
        - weights: 两个专家模型的混合权重 [walking权重, running权重]
        """
        gate_logits = self.network(observations)
        
        # 使用softmax将输出转换为权重（和为1的正数）
        weights = F.softmax(gate_logits, dim=-1)
            
        return weights
    
    def get_mixture_weights(self, observations, temperature=1.0):
        """
        根据观察计算专家模型的混合权重
        
        参数:
        - observations: 环境观察
        - temperature: softmax温度参数，值越小权重分布越尖锐，值越大分布越平滑
        
        返回:
        - weights: 专家模型的混合权重 [walking权重, running权重]
        - dominant_expert: 权重最大的专家索引（0表示行走，1表示奔跑）
        """
        gate_logits = self.network(observations)
        
        # 使用带温度的softmax计算权重
        weights = F.softmax(gate_logits / temperature, dim=-1)
        
        # 记录权重最大的专家（仅用于分析）
        dominant_expert = torch.argmax(weights, dim=-1)
        #test
        print("Raw gating weights:", weights.cpu().detach().numpy())

        return weights, dominant_expert
        
    