"""
Configuration for Humanoid Mixture of Experts (MoE) model.
"""

from gpugym.envs.PBRS.humanoid_config import HumanoidCfg, HumanoidCfgPPO


class HumanoidMoECfg(HumanoidCfg):
    """Configuration for Humanoid MoE environment."""

    class moe:
        # Expert model paths
        expert_paths = {
            "walk": "logs/PBRS_HumanoidLocomotion/Mar07_20-21-45_walkmodel/model_10000.pt",
            "run": "logs/PBRS_HumanoidLocomotion/Mar08_07-40-01_runmodel/model_20000.pt"
        }

        # Terrain roughness parameters
        roughness_window_size = 5  # Size of window to consider for roughness calculation
        roughness_threshold = 0.1  # Threshold for switching between walking and running

        # Transition parameters
        transition_smoothing = 0.8  # Higher values make transitions more gradual (0-1)
        transition_duration = 0.5  # Duration for transitions in seconds

    class env(HumanoidCfg.env):
        # Use fewer environments for testing
        num_envs = 64
        episode_length_s = 20  # Longer episodes to observe behavior

    class terrain(HumanoidCfg.terrain):
        # Enable terrain measurement for roughness calculation
        curriculum = True
        mesh_type = 'trimesh'  # Use trimesh for heightfield terrain
        measure_heights = True

        # Terrain generation parameters
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 10
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal distribution of terrain types

        # Difficulty parameters
        max_difficulty = 5
        min_difficulty = 0

        # Heightfield noise parameters
        horizontal_scale = 0.1
        vertical_scale = 0.005

        # Terrain types
        terrain_types = ["flat", "rolling", "discrete", "stairs", "slope"]

    class asset(HumanoidCfg.asset):
        disable_actions = False  # Enable actions from MoE


class HumanoidMoECfgPPO(HumanoidCfgPPO):
    """PPO configuration for Humanoid MoE."""

    do_wandb = True

    class runner(HumanoidCfgPPO.runner):
        experiment_name = 'MoE_HumanoidLocomotion'
        run_name = 'moe_terrain_select'

        # We don't need many iterations since we're just testing the MoE
        max_iterations = 10

        # Logging
        log_interval = 1
        save_interval = 5

    class wandb:
        # WandB logging configuration
        what_to_log = {
            'policy': ['weights', 'gradient_updates'],
            'environment': ['details', 'rewards'],
            'extras': ['roughness', 'expert_selection']
        }