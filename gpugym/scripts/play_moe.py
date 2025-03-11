# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gpugym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from gpugym.envs import *
from gpugym.utils import get_args, export_policy, export_critic, task_registry, Logger

import numpy as np
import torch





# based on the environment created by mixed_terrain_config.py
def play_moe(args):
    task_name = 'pbrs:mixed_terrain'

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)# with tht maximum of 16 envs

    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False  # True
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 1.0
    env_cfg.init_state.reset_ratio = 0.8

    # prepare environment
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Load both walking and running policies
    # Store original path to restore later
    original_model_path = train_cfg.runner.load_run if hasattr(train_cfg.runner, 'load_run') else None

    # Load walking model
    walk_model_path = "Mar07_20-21-45_walkmodel"
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = walk_model_path
    train_cfg.runner.checkpoint = 10000  # Specify the model checkpoint
    ppo_runner_walk, _ = task_registry.make_alg_runner(env=env, name=task_name, args=args, train_cfg=train_cfg)
    policy_walk = ppo_runner_walk.get_inference_policy(device=env.device)
    # Load running model
    run_model_path = "Mar08_07-40-01_runmodel"
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = run_model_path
    train_cfg.runner.checkpoint = 20000
    ppo_runner_run, _ = task_registry.make_alg_runner(env=env, name=task_name, args=args, train_cfg=train_cfg)
    policy_run = ppo_runner_run.get_inference_policy(device=env.device)

    # Restore original path
    if original_model_path:
        train_cfg.runner.load_run = original_model_path

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        # Export walking policy
        os.makedirs(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'policies'), exist_ok=True)
        walk_export_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'policies', 'walking_policy')
        export_policy(ppo_runner_walk.alg.actor_critic, walk_export_path)
        print('Exported walking policy model to: ', walk_export_path)

        # Export running policy
        run_export_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'policies', 'running_policy')
        export_policy(ppo_runner_run.alg.actor_critic, run_export_path)
        print('Exported running policy model to: ', run_export_path)

    # export critic as a jit module (used to run it from C++)
    if EXPORT_CRITIC:
        # Export walking critic
        os.makedirs(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'critics'), exist_ok=True)
        walk_critic_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'critics', 'walking_critic')
        export_critic(ppo_runner_walk.alg.actor_critic, walk_critic_path)
        print('Exported walking critic model to: ', walk_critic_path)

        # Export running critic
        run_critic_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'critics', 'running_critic')
        export_critic(ppo_runner_run.alg.actor_critic, run_critic_path)
        print('Exported running critic model to: ', run_critic_path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    play_log = []
    model_usage_log = [] # Also log which model was used at each step
    env.max_episode_length = 1000. / env.dt

    def calculate_terrain_roughness(
            measured_heights,
            robot_x,
            robot_y,
            normalize=True
    ):
        """
        根据地形配置和测量点高度数据计算机器人所在位置的地形崎岖度

        Args:
            terrain: 地形配置对象，包含以下属性：
                - measured_points_x: 测量点X坐标列表
                - measured_points_y: 测量点Y坐标列表
                - horizontal_scale: 水平缩放比例（米/单位）
                - vertical_scale: 垂直缩放比例（米/单位）
                - measure_heights: 是否启用高度测量
            measured_heights (np.ndarray): 当前测量点的原始高度值数组
            robot_x (float): 机器人当前位置X坐标（世界坐标系）
            robot_y (float): 机器人当前位置Y坐标（世界坐标系）
            normalize (bool): 是否归一化结果到[0,1]

        Returns:
            float: 地形崎岖度（0=平坦，1=极度崎岖）
        """

        # 将仿真单位转换为物理单位
        heights = measured_heights * env_cfg.terrain.vertical_scale

        # 将测量点转换为网格坐标
        try:
            nx = len(env_cfg.terrain.measured_points_x)
            ny = len(env_cfg.terrain.measured_points_y)
            height_grid = heights.reshape(nx, ny)
        except ValueError:
            raise ValueError(f"高度数据长度({len(heights)})与测量点网格尺寸({nx}x{ny})不匹配")

        # 计算高度标准差（全局起伏）
        std_dev = np.std(height_grid)

        # 计算梯度幅度（局部陡峭程度）
        dx = np.gradient(height_grid, axis=0) / env_cfg.terrain.horizontal_scale
        dy = np.gradient(height_grid, axis=1) / env_cfg.terrain.horizontal_scale
        gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
        mean_gradient = np.mean(gradient_magnitude)

        # 综合指标（可根据任务调整权重）
        roughness = 0.6 * std_dev + 0.4 * mean_gradient

        # 归一化处理
        if normalize:
            # 基于垂直缩放范围和典型最大坡度
            max_std = env_cfg.terrain.vertical_scale * 0.3  # 假设最大高度变化为30cm
            max_grad = np.tan(np.deg2rad(60))  # 60度坡度作为极端情况
            roughness = np.clip(roughness / (max_std + max_grad), 0.0, 1.0)

        return float(roughness)

    for i in range(10 * int(env.max_episode_length)):
        # 获取当前机器人的测量点高度（假设obs包含高度数据）Problem
        measured_heights = obs.height_measurements    # 具体实现取决于环境接口

        # 获取机器人当前位置
        robot_x = env.root_states[robot_index, 0].item()
        robot_y = env.root_states[robot_index, 1].item()

        # Get robot position from root states
        robot_pos = env.root_states[robot_index, 0:3].detach().cpu().numpy()

        # Use the new terrain roughness function
        roughness = calculate_terrain_roughness(measured_heights,robot_x,
            robot_y, normalize=True)

        # The threshold can be tuned based on testing
        roughness_threshold = 0.5
        if roughness < roughness_threshold:  # Relatively flat terrain
            current_policy = policy_run
            model_used = "run"
        else:  # Rough terrain
            current_policy = policy_walk
            model_used = "walk"

        # Get actions from the selected policy
        actions = current_policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # Log which model was use
        model_usage_log.append(model_used)

        # Make a training video
        if RECORD_FRAMES:
            if i % 2:
                os.makedirs(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'frames'), exist_ok=True)
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            # Get the observation dimensions from the actual observation
            obs_dim = obs[robot_index, :].cpu().numpy().shape[0]
            actions_dim = actions[robot_index, :].detach().cpu().numpy().shape[0]

            # Adjust logging to handle different robot types
            # For logging we'll use a simplified format that should work with different robots
            play_log.append(
                [i * env.dt]  # Timestep
                + obs[robot_index, :].cpu().numpy().tolist()  # Observations
                + actions[robot_index, :].detach().cpu().numpy().tolist()  # Actions
                + env.root_states[robot_index, :].detach().cpu().numpy().tolist()  # Root states
                + [1.0 if model_used == "run" else 0.0]  # Model identifier
            )
        elif i == stop_state_log:
            # Create directories if they don't exist
            os.makedirs('../analysis/data', exist_ok=True)
            # Save logs
            np.savetxt('../analysis/data/play_log.csv', play_log, delimiter=',')
            with open('../analysis/data/model_usage_log.txt', 'w') as f:
                for step, model in enumerate(model_usage_log):
                    f.write(f"Step {step}: {model}\n")
            print(
                f"Logs saved. Model usage: Walk: {model_usage_log.count('walk')}, Run: {model_usage_log.count('run')}")

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == '__main__':
    EXPORT_POLICY = False
    EXPORT_CRITIC = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play_moe(args)