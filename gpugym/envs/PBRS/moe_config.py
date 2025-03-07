"""
Configuration file for Humanoid MoE (Mixture of Experts) environment
which selects between walking and running based on terrain roughness
"""

import torch
from gpugym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from gpugym.envs.PBRS.humanoid_config import HumanoidCfg
from gpugym.envs.PBRS.humanoid_run_config import HumanoidRunCfg


class HumanoidMoECfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 39  # Added one extra for terrain roughness
        num_actions = 10
        episode_length_s = 5

    class moe:
        roughness_threshold = 0.05  # Threshold for switching between walk and run
        min_switching_time = 0.5  # Minimum time (seconds) between expert switches

    class terrain(LeggedRobotCfg.terrain):
        curriculum = True
        mesh_type = 'trimesh'
        measure_heights = True

        # Increase variation in terrain types to test both experts
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]  # Flat, slopes, stairs, gaps, rough

        # Generate terrains with varying roughness
        rough_terrain_roughness = 0.1  # Higher values create rougher terrain

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 5.
        heading_command = False
        ang_vel_command = True

        class ranges:
            # Use a wide range of commands to test both walking and running
            lin_vel_x = [0, 4.5]  # min max [m/s]
            lin_vel_y = [-0.75, 0.75]  # min max [m/s]
            ang_vel_yaw = [-2., 2.]  # min max [rad/s]
            heading = [0., 0.]

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'reset_to_range'
        penetration_check = False
        pos = [0., 0., 0.75]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0., 0.],
            [0., 0.],
            [0.72, 0.72],
            [-torch.pi / 10, torch.pi / 10],
            [-torch.pi / 10, torch.pi / 10],
            [-torch.pi / 10, torch.pi / 10]
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5],
            [-.5, .5]
        ]

        default_joint_angles = {
            'left_hip_yaw': 0.,
            'left_hip_abad': 0.,
            'left_hip_pitch': -0.2,
            'left_knee': 0.25,
            'left_ankle': 0.0,
            'right_hip_yaw': 0.,
            'right_hip_abad': 0.,
            'right_hip_pitch': -0.2,
            'right_knee': 0.25,
            'right_ankle': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        stiffness = {
            'left_hip_yaw': 30.,
            'left_hip_abad': 30.,
            'left_hip_pitch': 30.,
            'left_knee': 30.,
            'left_ankle': 30.,
            'right_hip_yaw': 30.,
            'right_hip_abad': 30.,
            'right_hip_pitch': 30.,
            'right_knee': 30.,
            'right_ankle': 30.,
        }
        damping = {
            'left_hip_yaw': 5.,
            'left_hip_abad': 5.,
            'left_hip_pitch': 5.,
            'left_knee': 5.,
            'left_ankle': 5.,
            'right_hip_yaw': 5.,
            'right_hip_abad': 5.,
            'right_hip_pitch': 5.,
            'right_knee': 5.,
            'right_ankle': 5.
        }

        action_scale = 1.0
        exp_avg_decay = None
        decimation = 10

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}' \
               '/resources/robots/mit_humanoid/mit_humanoid_fixed_arms.urdf'
        keypoints = ["base"]
        end_effectors = ['left_foot', 'right_foot']
        foot_name = 'foot'
        terminate_after_contacts_on = [
            'base',
            'left_upper_leg',
            'left_lower_leg',
            'right_upper_leg',
            'right_lower_leg',
            'left_upper_arm',
            'right_upper_arm',
            'left_lower_arm',
            'right_lower_arm',
            'left_hand',
            'right_hand',
        ]

        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

    class rewards(LeggedRobotCfg.rewards):
        # Base height target between walk and run
        base_height_target = 0.66  # Midway between walk (0.7) and run (0.62)
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8

        # negative total rewards clipped at zero (avoids early termination)
        only_positive_rewards = False
        tracking_sigma = 0.5

        class scales(LeggedRobotCfg.rewards.scales):
            # * "True" rewards * #
            action_rate = -1.e-3
            action_rate2 = -1.e-4
            tracking_lin_vel = 10.
            tracking_ang_vel = 5.
            torques = -1e-4
            dof_pos_limits = -10
            torque_limits = -1e-2
            termination = -100

            # * Expert selection reward * #
            expert_selection = 0.5  # Reward for selecting appropriate expert

            # * Shaping rewards * #
            orientation = 5.0
            base_height = 2.0
            joint_regularization = 1.0

            # * PBRS rewards * #
            ori_pb = 1.0
            baseHeight_pb = 1.0
            jointReg_pb = 1.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            base_z = 1. / 0.6565

        clip_observations = 100.
        clip_actions = 10.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            base_z = 0.05
            dof_pos = 0.005
            dof_vel = 0.01
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            in_contact = 0.1
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]

        class physx:
            max_depenetration_velocity = 10.0


class HumanoidMoECfgPPO(LeggedRobotCfgPPO):
    do_wandb = True
    seed = -1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # algorithm training hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1.e-5
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 24
        max_iterations = 1000
        run_name = 'ICRA2023'
        experiment_name = 'HumanoidMoE'
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False

    class policy(LeggedRobotCfgPPO.policy):
        # Larger network to handle the more complex task
        actor_hidden_dims = [512, 256, 256]
        critic_hidden_dims = [512, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = 'elu'

    # Add WandB visualization configuration
    class wandb:
        what_to_log = ['command', 'reward', 'extras', 'terrain']  # Log terrain data