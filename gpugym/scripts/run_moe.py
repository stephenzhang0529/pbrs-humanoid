#!/usr/bin/env python
"""
Script to run the Humanoid Mixture of Experts model.
"""

import argparse
import os
import numpy as np
import isaacgym
import torch
from gpugym.envs.PBRS.moe import HumanoidMoE
from gpugym.envs.PBRS.moe_config import HumanoidMoECfg, HumanoidMoECfgPPO
from gpugym.utils import task_registry


def main():
    """Run the Humanoid MoE model."""
    parser = argparse.ArgumentParser(description='Run Humanoid MoE')
    parser.add_argument('--headless', action='store_true',
                        help='Run without rendering')
    parser.add_argument('--num_envs', type=int, default=64,
                        help='Number of environments to run')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--max_episodes', type=int, default=10,
                        help='Maximum number of episodes to run')
    parser.add_argument('--walk_model', type=str,
                        default='logs/PBRS_HumanoidLocomotion/Mar07_20-21-45_walkmodel/model_10000.pt',
                        help='Path to walking model')
    parser.add_argument('--run_model', type=str,
                        default='logs/PBRS_HumanoidLocomotion/Mar08_07-40-01_runmodel/model_20000.pt',
                        help='Path to running model')
    parser.add_argument('--roughness_threshold', type=float, default=0.1,
                        help='Threshold for switching between walking and running')

    args = parser.parse_args()

    # Register the HumanoidMoE task
    if 'humanoid_moe' not in task_registry.task_classes:
        print("Registering humanoid_moe task...")

        # Create a config with the provided model paths
        cfg = HumanoidMoECfg()
        cfg.moe.expert_paths = {
            "walk": args.walk_model,
            "run": args.run_model
        }
        cfg.moe.roughness_threshold = args.roughness_threshold
        cfg.env.num_envs = args.num_envs

        task_registry.register('humanoid_moe',
                               HumanoidMoE,
                               cfg,
                               HumanoidMoECfgPPO())

    # Setup the environment
    headless = args.headless
    #env_cfg = task_registry.get_task_config('humanoid_moe')
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # Set the random seed
    if args.seed is not None:
        env_cfg.seed = args.seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Create the environment
    env = task_registry.make_env('humanoid_moe', cfg=env_cfg, headless=headless)

    # Run the simulation
    print("Running MoE simulation...")
    obs = env.reset()

    num_episodes = 0
    total_steps = 0

    while num_episodes < args.max_episodes:
        # The MoE will compute its own actions in the step method
        # We just need to pass dummy actions
        dummy_actions = torch.zeros((env.num_envs, env.num_actions),
                                    device=env.device)

        # Step the environment
        obs, rewards, dones, info = env.step(dummy_actions)

        # Log some information
        if total_steps % 100 == 0:
            # Calculate average roughness and expert selection
            avg_roughness = env.roughness_scores.mean().item()
            pct_walking = (env.current_expert == 0).float().mean().item() * 100
            pct_running = (env.current_expert == 1).float().mean().item() * 100

            print(f"Step {total_steps}: "
                  f"Avg Roughness: {avg_roughness:.4f}, "
                  f"Walking: {pct_walking:.1f}%, "
                  f"Running: {pct_running:.1f}%")

        # Check for episode terminations
        if dones.any():
            num_episodes += dones.sum().item()
            print(f"Completed {num_episodes}/{args.max_episodes} episodes")

        total_steps += 1

    print("Simulation complete!")
    env.close()


if __name__ == "__main__":
    main()