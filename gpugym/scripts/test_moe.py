# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
import numpy as np

from gpugym.envs.PBRS.moe import HumanoidMoE
from gpugym.envs.PBRS.moe_config import HumanoidMoECfg, HumanoidMoECfgPPO
from gpugym.utils import get_args, task_registry
import torch

def test_moe():
    """Test the Humanoid MoE model by running a simulation"""
    # Register the MoE task
    if 'humanoid_moe' not in task_registry.task_classes:
        print("Registering humanoid_moe task...")
        task_registry.register('humanoid_moe',
                               HumanoidMoE,
                               HumanoidMoECfg(),
                               HumanoidMoECfgPPO())

    # Parse arguments
    custom_args = [
        "--task", "humanoid_moe",
        "--headless", "False",  # Set to True for headless mode
        "--num_envs", "8",
        "--episode_length", "1000"
    ]

    args = get_args()

    # Create the environment
    env, env_cfg = task_registry.make_env(name="humanoid_moe", args=args)

    # Reset the environment
    obs = env.reset()

    # Track statistics
    episode_length = 0
    max_episode_length = 1000
    policy_switches = np.zeros(env.num_envs)
    last_policies = np.copy(env.active_policies)

    print("Starting MoE test simulation...")
    print("Press Ctrl+C to stop")

    try:
        while episode_length < max_episode_length:
            # Step with zero actions (policies will be selected by MoE)
            actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
            obs, _, _, _ = env.step(actions)

            # Track policy switches
            for i in range(env.num_envs):
                if env.active_policies[i] != last_policies[i]:
                    policy_switches[i] += 1
                    print(f"Env {i}: Switched to {'running' if env.active_policies[i] == 1 else 'walking'} policy")

            last_policies = np.copy(env.active_policies)
            episode_length += 1

            # Print periodic stats
            if episode_length % 100 == 0:
                walk_count = np.sum(env.active_policies == 0)
                run_count = np.sum(env.active_policies == 1)
                print(f"Step {episode_length}/{max_episode_length}")
                print(f"Policies: {walk_count} walking, {run_count} running")
                print(f"Average policy switches: {np.mean(policy_switches):.2f}")

    except KeyboardInterrupt:
        print("Simulation stopped by user")

    # Print final statistics
    print("\nTest completed")
    print(f"Total steps: {episode_length}")

    walk_count = np.sum(env.active_policies == 0)
    run_count = np.sum(env.active_policies == 1)
    print(f"Final policy distribution: {walk_count} walking, {run_count} running")
    print(f"Policy switches per environment: {policy_switches}")
    print(f"Average policy switches: {np.mean(policy_switches):.2f}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    test_moe()