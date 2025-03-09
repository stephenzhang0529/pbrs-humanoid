"""
Configuration for Humanoid Mixture of Experts (MoE) model.
"""

from gpugym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class HumanoidMoECfg(LeggedRobotCfg):
    class moe:
        roughness_threshold = 0.15  # Threshold for deciding between walking and running


class HumanoidMoECfgPPO(LeggedRobotCfgPPO.runner):
    run_name = "humanoid_moe"
    experiment_name = "PBRS_HumanoidLocomotion"
    seed = -1

    max_iterations = 500  # For demonstration

    # Add WandB logging configuration
    do_wandb = True

    class wandb:
        class what_to_log:
            # Specify which configuration parameters to log
            class algorithm:
                fields = ["learning_rate", "num_learning_epochs", "num_mini_batches"]

            class task:
                fields = ["moe.roughness_threshold"]