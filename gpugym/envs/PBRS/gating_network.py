"""
动态门控网络模块，用于Mixture of Experts系统
负责根据当前状态（地形+机器人状态）选择最合适的专家模型（行走或奔跑）
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
        
        # 最后一层输出2个值，代表选择walking或running模型的概率
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observations):
        """
        前向传播，计算选择each专家的概率/权重
        
        参数:
        - observations: 环境的观察

        返回:
        - gate_values: 门控值，表示选择各专家的概率/权重
        """
        gate_logits = self.network(observations)
        
        # Gumbel-softmax trick: 生成离散的one-hot编码
        # 在训练时使用soft版本，在推理时使用hard版本
        if self.training:
            # 训练时使用soft Gumbel-softmax
            gate_values = F.gumbel_softmax(gate_logits, hard=False, tau=1.0)
        else:
            # 推理时使用hard Gumbel-softmax (one-hot)
            gate_values = F.gumbel_softmax(gate_logits, hard=True, tau=1.0)
            
        return gate_values
    
    def select_expert(self, observations, deterministic=False):
        """
        根据观察选择专家模型
        
        参数:
        - observations: 环境观察
        - deterministic: 是否确定性选择（True则选择概率最高的专家）
        
        返回:
        - selected_expert: 选择的专家索引（0表示行走，1表示奔跑）
        - gate_values: 原始门控值
        """
        gate_logits = self.network(observations)
        
        if deterministic:
            # 确定性选择概率最高的专家
            selected_expert = torch.argmax(gate_logits, dim=-1)
            # 将选择结果转换为one-hot编码
            gate_values = F.one_hot(selected_expert, num_classes=2).float()
        else:
            # 根据概率随机选择
            probs = F.softmax(gate_logits, dim=-1)
            selected_expert = torch.multinomial(probs, 1).squeeze(-1)
            # 将选择结果转换为one-hot编码
            gate_values = F.one_hot(selected_expert, num_classes=2).float()
        
        return selected_expert, gate_values 