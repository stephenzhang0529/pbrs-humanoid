# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from idlelib.debugobj_r import remote_object_tree_item

from mpl_toolkits.mplot3d.proj3d import rotation_about_vector

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

    z_history=[]
    window_sizes=5# the size of slide window

    def calculate_terrain_roughness(z_history, window_size=15,normalize=True):
        # Use only the most recent samples up to window_size
        recent_z = z_history[-window_size:] if len(z_history) >= window_size else z_history

        if len(recent_z) < 3:  # Need at least a few points to calculate meaningful variance
            return 0.0

        # Calculate standard deviation of z values
        z_std = np.std(recent_z)

        # Calculate the average absolute change between consecutive z values (jitter)
        z_changes = np.abs(np.diff(recent_z))
        mean_change = np.mean(z_changes) if len(z_changes) > 0 else 0

        # Calculate the frequency of direction changes (oscillations)
        direction_changes = 0
        if len(z_changes) > 1:
            for i in range(1, len(z_changes)):
                if (recent_z[i] - recent_z[i - 1]) * (recent_z[i + 1] - recent_z[i]) < 0:
                    direction_changes += 1
            direction_change_rate = direction_changes / (len(recent_z) - 2)
        else:
            direction_change_rate = 0

        # Combine metrics (can be tuned for better performance)
        roughness = 0.5 * z_std + 0.3 * mean_change + 0.2 * direction_change_rate

        # Normalize if requested
        if normalize:
            # Based on typical values observed during testing
            # These thresholds should be tuned based on your specific robot and terrain
            max_expected_std = 0.05  # 5cm standard deviation as maximum expected
            max_expected_change = 0.03  # 3cm average change as maximum expected
            max_expected_direction_change = 0.8  # 80% direction change rate as maximum

            #max_roughness = 0.5 * max_expected_std + 0.3 * max_expected_change + 0.2 * max_expected_direction_change
            max_roughness =0.367*2

            roughness = np.clip(roughness / max_roughness, 0.0, 1.0)

        return float(roughness)



    for i in range(10 * int(env.max_episode_length)):
        # 获取机器人当前位置
        robot_z = env.root_states[robot_index, 2].item()  #0.72

        z_history.append(robot_z)
        roughness=calculate_terrain_roughness(z_history,window_sizes,True)
        # The threshold can be tuned based on testing
        roughness_threshold = 0.4
        print(roughness)
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