# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gpugym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from gpugym.envs import *
from gpugym.utils import get_args, export_policy, export_critic, task_registry, Logger

import numpy as np
import torch


def play_moe(args):
    # Fix: Use 'humanoid_pbrs_vel' or your actual task name instead of 'anymal_c_flat'
    # This should match one of the tasks registered in your task_registry
    task_name = 'pbrs:humanoid'  # Update this to your actual task name

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.terrain.num_rows = 10  # 增加地形行数，使得一部分平坦，一部分崎岖
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    # 控制地形类型，前 5 行平坦，后 5 行崎岖
    env_cfg.terrain.terrain_type = "heightfield"  # 添加一个自定义地形类型
    env_cfg.terrain.flat_to_rough_ratio = 0.5  # 前 50% 平坦，后 50% 崎岖

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
    train_cfg.runner.checkpoint = 10000  # 指定具体的模型文件
    ppo_runner_walk, _ = task_registry.make_alg_runner(env=env, name=task_name, args=args, train_cfg=train_cfg)
    policy_walk = ppo_runner_walk.get_inference_policy(device=env.device)

    # Load running model
    run_model_path = "Mar08_07-40-01_runmodel"
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
    # Also log which model was used at each step
    model_usage_log = []
    env.max_episode_length = 1000. / env.dt

    # Function to determine terrain roughness at robot position
    def get_terrain_roughness(robot_position):
        # Assuming env.terrain contains terrain height or roughness data
        # And robot_position is x, y, z in world frame
        x, y = robot_position[0], robot_position[1]

        # Convert world position to terrain grid coordinates
        terrain_size_x = env_cfg.terrain.num_rows
        terrain_size_y = env_cfg.terrain.num_cols

        # Calculate terrain row index (rough terrain is in the back rows)
        # Normalize position to [0, 1] range within terrain bounds
        norm_y = (y + terrain_size_y / 2) / terrain_size_y

        # If robot is in the back half (rougher terrain)
        # This is a simplification - you may need to adjust based on your terrain setup
        if norm_y > 0.5:
            return 1.0  # Rough terrain
        else:
            return 0.0  # Flat terrain

    for i in range(10 * int(env.max_episode_length)):
        # Get robot position from root states
        robot_pos = env.root_states[robot_index, 0:3].detach().cpu().numpy()

        # Determine which policy to use based on terrain roughness
        roughness = get_terrain_roughness(robot_pos)

        # Simple switching: Use running on flat terrain, walking on rough terrain
        # You could also implement a smooth blending between policies
        if roughness < 0.5:  # Flat terrain
            current_policy = policy_run
            model_used = "run"
        else:  # Rough terrain
            current_policy = policy_walk
            model_used = "walk"

        # Get actions from the selected policy
        actions = current_policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # Log which model was used
        model_usage_log.append(model_used)

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
            ### Humanoid PBRS Logging ###
            # [ 1]  Timestep
            # [38]  Agent observations
            # [10]  Agent actions (joint setpoints)
            # [13]  Floating base states in world frame
            # [ 6]  Contact forces for feet
            # [10]  Joint torques
            # [ 1]  Model used (walk/run)
            play_log.append(
                [i * env.dt]
                + obs[robot_index, :].cpu().numpy().tolist()
                + actions[robot_index, :].detach().cpu().numpy().tolist()
                + env.root_states[robot_index, :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[0], :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[1], :].detach().cpu().numpy().tolist()
                + env.torques[robot_index, :].detach().cpu().numpy().tolist()
                + [1.0 if model_used == "run" else 0.0]  # Add model identifier
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