"""
Environment file for Humanoid Mixture of Experts (MoE) system
This environment allows a humanoid robot to dynamically switch between walking and running models
based on terrain and robot state using a gating network.
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gpugym.utils.math import *
from gpugym.envs import LeggedRobot
import numpy as np
import os

class HumanoidMoE(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, external_sim=None):
        # 保存外部sim引用
        self.external_sim = external_sim
        
        # 调用父类初始化
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 初始化后覆盖sim对象（如果提供了外部sim）
        if self.external_sim is not None:
            self.sim = self.external_sim
            
        # 加载预训练的行走和奔跑模型
        self.walk_model = None
        self.run_model = None
        self.curr_selected_expert = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_selected_expert = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 用于跟踪转换的变量
        self.transition_phase = torch.zeros(self.num_envs, device=self.device)
        self.transition_duration = 10  # 过渡持续的帧数
        
        # 内在好奇心奖励相关参数
        self.curiosity_scale = 0.1  # 内在好奇心奖励的缩放因子
        self.prev_heights = None
        
        # 平衡奖励参数
        self.balance_scale = 1.0
        
        # 速度奖励参数
        self.speed_scale = 0.5

    def _custom_init(self, cfg):
        self.dt_step = self.cfg.sim.dt * self.cfg.control.decimation
        self.pbrs_gamma = 0.99
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.eps = 0.2
        self.phase_freq = 1.
        
        # 初始化地形高度历史记录
        if self.cfg.terrain.measure_heights:
            self.height_points_history = None

    def compute_observations(self):
        """
        扩展观察空间，包括：
        1. 基本的humanoid观察
        2. 地形高度信息
        """
        base_z = self.root_states[:, 2].unsqueeze(1)*self.obs_scales.base_z
        in_contact = torch.gt(
            self.contact_forces[:, self.end_eff_ids, 2], 0).int()
        in_contact = torch.cat(
            (in_contact[:, 0].unsqueeze(1), in_contact[:, 1].unsqueeze(1)),
            dim=1)
        self.commands[:, 0:2] = torch.where(
            torch.norm(self.commands[:, 0:2], dim=-1, keepdim=True) < 0.5,
            0., self.commands[:, 0:2].double()).float()
        self.commands[:, 2:3] = torch.where(
            torch.abs(self.commands[:, 2:3]) < 0.5,
            0., self.commands[:, 2:3].double()).float()
            
        # 获取地形高度测量
        if self.cfg.terrain.measure_heights:
            heights = self._get_heights()
            
            # 保存高度历史以计算变化率
            if self.height_points_history is None:
                self.height_points_history = heights.clone()
            
            # 计算高度变化
            height_changes = heights - self.height_points_history
            self.height_points_history = heights.clone()
            
            # 将高度和高度变化添加到观察中
            heights = heights.clip(-1, 4) * self.obs_scales.height_measurements
            height_changes = height_changes.clip(-1, 1)
            
            # 基本观察
            self.obs_buf = torch.cat((
                base_z,                                 # [1] Base height
                self.base_lin_vel,                      # [3] Base linear velocity
                self.base_ang_vel,                      # [3] Base angular velocity
                self.projected_gravity,                 # [3] Projected gravity
                self.commands[:, 0:3],                  # [3] Velocity commands
                self.smooth_sqr_wave(self.phase),       # [1] Contact schedule
                torch.sin(2*torch.pi*self.phase),       # [1] Phase variable
                torch.cos(2*torch.pi*self.phase),       # [1] Phase variable
                self.dof_pos,                           # [10] Joint states
                self.dof_vel,                           # [10] Joint velocities
                in_contact,                             # [2] Contact states
                heights,                                # [num_height_points] Terrain heights
                height_changes,                         # [num_height_points] Terrain height changes
            ), dim=-1)
        else:
            # 基本观察（无地形高度）
            self.obs_buf = torch.cat((
                base_z,                                 # [1] Base height
                self.base_lin_vel,                      # [3] Base linear velocity
                self.base_ang_vel,                      # [3] Base angular velocity
                self.projected_gravity,                 # [3] Projected gravity
                self.commands[:, 0:3],                  # [3] Velocity commands
                self.smooth_sqr_wave(self.phase),       # [1] Contact schedule
                torch.sin(2*torch.pi*self.phase),       # [1] Phase variable
                torch.cos(2*torch.pi*self.phase),       # [1] Phase variable
                self.dof_pos,                           # [10] Joint states
                self.dof_vel,                           # [10] Joint velocities
                in_contact,                             # [2] Contact states
            ), dim=-1)
            
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                * self.noise_scale_vec

        return self.obs_buf

    def compute_reward(self):
        """计算奖励，包括：
        1. 平衡奖励 - 保持直立
        2. 速度奖励 - 尽量达到最大速度
        3. 内在好奇心奖励 - 鼓励探索新的地形
        """
        # 基础奖励 - 平衡奖励（保持直立）
        orientation_penalty = self._reward_orientation()
        upright_reward = torch.exp(-5.0 * torch.abs(orientation_penalty))
        
        # 速度奖励 - 沿x轴的速度
        velocity = self.base_lin_vel[:, 0]  # x方向的速度
        velocity_reward = torch.tanh(velocity)  # 使用tanh函数，使奖励随速度增加而增加，但有上限
        
        # 平滑动作切换奖励 - 当专家模型切换时提供平滑过渡
        transition_reward = torch.zeros_like(velocity)
        transitioning_envs = (self.prev_selected_expert != self.curr_selected_expert)
        if transitioning_envs.any():
            # 更新过渡阶段
            self.transition_phase[transitioning_envs] = self.transition_duration
            self.prev_selected_expert = self.curr_selected_expert.clone()
        
        # 更新正在过渡的环境的过渡奖励
        envs_in_transition = (self.transition_phase > 0)
        if envs_in_transition.any():
            # 平滑过渡奖励，促进平稳切换
            transition_reward[envs_in_transition] = 0.1 * (1.0 - self.transition_phase[envs_in_transition] / self.transition_duration)
            # 减少过渡阶段计数器
            self.transition_phase[envs_in_transition] -= 1
            
        # 组合所有奖励
        rewards = (
            self.balance_scale * upright_reward + 
            self.speed_scale * velocity_reward +
            transition_reward
        )
        
        # 摔倒惩罚
        termination_penalty = self.reset_buf * -2.0
        rewards += termination_penalty
        
        return rewards

    def check_termination(self):
        """检查是否终止环境，条件为机器人摔倒或其他异常情况"""
        # 通过重力投影判断是否摔倒
        torso_position = self.root_states[:, 0:3]
        torso_rotation = self.root_states[:, 3:7]
        
        # 重力方向在躯干坐标系中的投影
        gravity_direction = quat_rotate_inverse(torso_rotation, self.gravity_vec)
        
        # 判断躯干与垂直方向的夹角是否过大
        tilt = torch.abs(torch.atan2(torch.sqrt(gravity_direction[:, 0] ** 2 + gravity_direction[:, 1] ** 2), torch.abs(gravity_direction[:, 2])))
        
        # 如果倾斜角度大于阈值，则认为机器人摔倒
        max_tilt = 1.0  # 约57度
        self.reset_buf = torch.where(tilt > max_tilt, torch.ones_like(self.reset_buf), self.reset_buf)
        
        # 躯干高度过低，认为摔倒
        torso_height = torso_position[:, 2]
        min_height = 0.4  # 最小高度阈值
        self.reset_buf = torch.where(torso_height < min_height, torch.ones_like(self.reset_buf), self.reset_buf)
        
        return self.reset_buf

    def pre_physics_step(self):
        """在物理模拟步骤前执行，用于处理动作和保存潜力状态"""
        # 保存当前状态的潜力值，用于PBRS计算
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()
        
        # 调用父类的pre_physics_step
        super().pre_physics_step()

    def _reward_orientation(self):
        # 判断躯干与垂直方向的夹角
        # 将重力向量投影到躯干坐标系中
        gravity_direction = quat_rotate_inverse(self.root_states[:, 3:7], self.gravity_vec)
        
        # 计算与理想姿态（垂直方向）的角度
        tilt = torch.atan2(torch.sqrt(gravity_direction[:, 0]**2 + gravity_direction[:, 1]**2), torch.abs(gravity_direction[:, 2]))
        
        return tilt

    def _reward_base_height(self):
        """奖励跟踪指定的基础高度"""
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = (base_height - 0.85)  # 目标高度为0.85m
        error = error.flatten()
        return torch.exp(-torch.square(error)/0.25)  # tracking_sigma = 0.25

    def _reward_joint_regularization(self):
        """奖励关节姿势和对称性"""
        error = 0.
        # Yaw关节在0附近的正则化
        error += self.sqrdexp(self.dof_pos[:, 0])
        error += self.sqrdexp(self.dof_pos[:, 5])
        # Ab/ad关节对称性
        error += self.sqrdexp(self.dof_pos[:, 1] - self.dof_pos[:, 6])
        # Pitch关节对称性
        error += self.sqrdexp(self.dof_pos[:, 2] + self.dof_pos[:, 7])
        return error/4

    def _reward_ankle_regularization(self):
        """脚踝关节在0附近的正则化"""
        error = 0
        error += self.sqrdexp(self.dof_pos[:, 4])
        error += self.sqrdexp(self.dof_pos[:, 9])
        return error

    # 基于潜力的奖励函数
    def _reward_ori_pb(self):
        """基于潜力的姿态奖励"""
        delta_phi = ~self.reset_buf * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt_step

    def _reward_jointReg_pb(self):
        """基于潜力的关节正则化奖励"""
        delta_phi = ~self.reset_buf * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt_step

    def _reward_baseHeight_pb(self):
        """基于潜力的基础高度奖励"""
        delta_phi = ~self.reset_buf * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt_step

    def sqrdexp(self, x):
        """平方指数函数，用于奖励计算"""
        return torch.exp(-torch.square(x))

    def smooth_sqr_wave(self, phase):
        """产生平滑的方波，用于步态相位"""
        phase_shifted = phase + 0.25
        phase_shifted = torch.fmod(phase_shifted, 1.0)
        return self.sqrdexp((phase_shifted - 0.5) / self.eps) 