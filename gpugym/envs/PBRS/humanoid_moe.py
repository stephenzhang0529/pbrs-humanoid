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
        
        # 存储专家权重和主导专家索引
        self.expert_weights = torch.ones((self.num_envs, 2), device=self.device) * 0.5  # 初始权重均等
        self.curr_selected_expert = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 存储权重最大的专家
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
        """
        计算奖励函数
        包括：
        - 存活奖励（正）
        - 前进速度奖励（正）
        - 躯干高度和方向奖励（正）
        - 关节正则化奖励（负）
        """
        # 能量消耗奖励已被忽略
        # torques = self.torques
        # velocity = self.dof_vel
        # energy_cost = torch.sum(torch.abs(torques * velocity), dim=1)
        # power_reward = torch.exp(-0.12 * energy_cost)
        
        # 替换为固定值1.0，相当于没有能量惩罚
        power_reward = torch.ones((self.num_envs,), device=self.device)
        
        # 存活奖励 - 为了鼓励初始探索，这个奖励应该始终有
        alive_reward = torch.ones_like(power_reward) * 0.5  # 增加存活奖励
        
        # 躯干高度奖励
        height_reward = torch.zeros_like(alive_reward)
        base_height = self.root_states[:, 2]
        # 增加高度奖励的平滑性
        height_reward = torch.where(base_height > 0.9, 
                                    1.0 * torch.ones_like(height_reward),
                                    torch.where(base_height > 0.7,
                                               (base_height - 0.7) / 0.2 * torch.ones_like(height_reward),
                                               -3.0 * torch.ones_like(height_reward)))
        
        # 躯干方向奖励
        orientation_penalty = self._reward_orientation()
        upright_reward = torch.exp(-5.0 * torch.abs(orientation_penalty))
        
        # 速度奖励 - 沿x轴的速度，使用更平滑的奖励函数
        velocity = self.base_lin_vel[:, 0]  # x方向的速度
        velocity_reward = torch.tanh(velocity) * 0.8  # 使用tanh函数，使奖励随速度增加而增加，但有上限
        
        # 平滑动作切换奖励 - 当主导专家模型切换时提供平滑过渡
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
            transition_reward[envs_in_transition] = 0.2 * (1.0 - self.transition_phase[envs_in_transition] / self.transition_duration)
            # 减少过渡阶段计数器
            self.transition_phase[envs_in_transition] -= 1
            
        # 合并所有奖励
        total_reward = alive_reward + height_reward + upright_reward + velocity_reward + transition_reward
        
        # 确保奖励不全为0，特别是在训练初期
        if self.episode_length_buf.min() < 10:
            # 在初始阶段给一个小的基础奖励，鼓励探索
            exploration_reward = 0.1 * torch.ones_like(total_reward)
            total_reward += exploration_reward
        
        # 打印平均奖励组件（仅在主环境中）
        if self.common_step_counter % 100 == 0 and self.episode_length_buf.min() > 5:
            # 每100步打印一次奖励组件
            print(f"奖励组件: 存活={alive_reward.mean().item():.2f}, 高度={height_reward.mean().item():.2f}, " 
                  f"方向={upright_reward.mean().item():.2f}, 速度={velocity_reward.mean().item():.2f}, "
                  f"过渡={transition_reward.mean().item():.2f}, 总计={total_reward.mean().item():.2f}")
            self.rew_buf = total_reward
        else:
            self.rew_buf = total_reward
        
        return total_reward

    def check_termination(self):
        """检查是否终止环境，条件为机器人摔倒或其他异常情况"""
        # 完全禁用前30步的终止条件，无论任何情况都不终止环境
        if self.episode_length_buf.min() < 30:  # 增加到30步
            # 每100步打印一次当前episode长度
            if self.common_step_counter % 100 == 0:
                # 打印详细信息，帮助排查问题
                torso_position = self.root_states[:, 0:3]
                torso_height = torso_position[:, 2]
                # 获取基本状态信息
                min_height = torso_height.min().item()
                max_height = torso_height.max().item()
                print(f"⚠️ 禁用终止条件，继续训练。当前episode长度: {self.episode_length_buf.min()}, "
                      f"躯干高度范围: {min_height:.3f}~{max_height:.3f}m")
                
            # 强制返回零，表示不终止
            return torch.zeros_like(self.reset_buf)
            
        # 通过重力投影判断是否摔倒
        torso_position = self.root_states[:, 0:3]
        torso_rotation = self.root_states[:, 3:7]
        
        # 重力方向在躯干坐标系中的投影
        gravity_direction = quat_rotate_inverse(torso_rotation, self.gravity_vec)
        
        # 判断躯干与垂直方向的夹角是否过大
        tilt = torch.abs(torch.atan2(torch.sqrt(gravity_direction[:, 0] ** 2 + gravity_direction[:, 1] ** 2), torch.abs(gravity_direction[:, 2])))
        
        # 清空reset_buf，只有在满足终止条件时才设置为1
        self.reset_buf = torch.zeros_like(self.reset_buf)
        
        # 如果倾斜角度大于阈值，则认为机器人摔倒
        max_tilt = 1.5  # 进一步增加到约86度，大幅提高容忍度
        tilt_term = torch.where(tilt > max_tilt, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        self.reset_buf = self.reset_buf | tilt_term
        
        # 躯干高度过低，认为摔倒
        torso_height = torso_position[:, 2]
        min_height = 0.2  # 进一步降低最小高度阈值到0.2米
        height_term = torch.where(torso_height < min_height, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        self.reset_buf = self.reset_buf | height_term
        
        # 超时重置
        timeout_term = torch.where(self.episode_length_buf >= self.max_episode_length, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        self.reset_buf = self.reset_buf | timeout_term
        
        # 记录终止原因（用于调试）
        if torch.any(self.reset_buf > 0) and self.common_step_counter % 10 == 0:  # 增加记录频率
            tilt_count = tilt_term.sum().item()
            height_count = height_term.sum().item()
            timeout_count = timeout_term.sum().item()
            
            # 记录详细的状态信息，帮助调试
            if tilt_count > 0:
                max_tilt_value = tilt.max().item() * 180/3.14159  # 转换为角度
                print(f"👉 倾斜终止: {tilt_count}个环境, 最大倾斜角度={max_tilt_value:.1f}度")
            
            if height_count > 0:
                min_height_value = torso_height.min().item()
                print(f"👉 高度终止: {height_count}个环境, 最小躯干高度={min_height_value:.3f}m")
                
            if timeout_count > 0:
                print(f"👉 超时终止: {timeout_count}个环境")
                
            # 总结
            total_resets = self.reset_buf.sum().item()
            print(f"✅ 终止统计: 总计={total_resets}, 倾斜过大={tilt_count}, 高度过低={height_count}, 超时={timeout_count}")
        
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

    def update_expert_info(self, dominant_experts, expert_weights):
        """
        更新专家权重和主导专家信息
        
        参数:
        - dominant_experts: 每个环境中权重最大的专家索引
        - expert_weights: 每个环境中各专家的权重
        """
        # 保存上一步的主导专家索引（用于检测转换）
        self.prev_selected_expert = self.curr_selected_expert.clone()
        
        # 更新当前主导专家和权重
        self.curr_selected_expert = dominant_experts
        self.expert_weights = expert_weights 

    def step(self, actions, expert_info=None):
        """
        重写基类的 step 方法，添加专家权重更新功能
        
        参数:
        - actions: 要执行的动作
        - expert_info: 包含专家权重信息的元组 (dominant_experts, expert_weights)，如果提供的话
        """
        # 如果提供了专家信息，更新专家权重和主导专家索引
        if expert_info is not None:
            dominant_experts, expert_weights = expert_info
            self.update_expert_info(dominant_experts, expert_weights)
            
        # 调用父类的 step 方法执行动作
        return super().step(actions) 

    def post_physics_step(self):
        """物理模拟步骤后执行，用于更新观察、计算奖励等"""
        # 在调用父类方法前保存原始reset_buf，以便可以实现我们自己的重置逻辑
        orig_reset_buf = self.reset_buf.clone() if hasattr(self, 'reset_buf') else None
        
        # 调用父类的post_physics_step方法，但我们会覆盖它的终止逻辑
        super().post_physics_step()
        
        # 如果我们处于前30步（保护期），则恢复原始reset_buf（如果有）
        if hasattr(self, 'episode_length_buf') and self.episode_length_buf.min() < 30 and orig_reset_buf is not None:
            self.reset_buf = orig_reset_buf
            
        # 完成后使用我们自己的check_termination方法
        self.reset_buf = self.check_termination()
        