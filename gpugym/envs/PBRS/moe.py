"""
Mixture of Experts (MoE) implementation for humanoid robot that selects
between walking and running models based on terrain roughness.
"""


import numpy as np
import os
from gpugym.utils.helpers import get_load_path
from gpugym.envs import LeggedRobot
import torch


class HumanoidMoE(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Load the pre-trained models
        self.run_model_path = "logs/PBRS_HumanoidLocomotion/Mar08_07-40-01_runmodel/model_20000.pt"
        self.walk_model_path = "logs/PBRS_HumanoidLocomotion/Mar07_20-21-45_walkmodel/model_10000.pt"

        # Load models
        self.run_policy = self._load_policy(self.run_model_path)
        self.walk_policy = self._load_policy(self.walk_model_path)

        # Terrain roughness threshold for decision making
        self.roughness_threshold = cfg.moe.roughness_threshold

        print(f"HumanoidMoE initialized with models:")
        print(f"  - Run model: {self.run_model_path}")
        print(f"  - Walk model: {self.walk_model_path}")
        print(f"  - Roughness threshold: {self.roughness_threshold}")

        # Track which policy is active for each environment
        self.active_policies = np.zeros(self.num_envs, dtype=np.int32)  # 0 for walk, 1 for run

    def _load_policy(self, model_path):
        """Load a policy from a checkpoint file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        policy = checkpoint['model'].to(self.device)
        policy.eval()  # Set to evaluation mode
        return policy

    def compute_terrain_roughness(self, env_ids):
        """Compute terrain roughness for the given environments"""
        roughness = torch.zeros(len(env_ids), device=self.device)

        # Sample points around the robot to measure terrain height variation
        sample_radius = 2.0  # meters around the robot
        num_samples = 16  # number of sample points

        for i, env_id in enumerate(env_ids):
            # Get robot position
            robot_pos = self.root_states[env_id, :3].cpu().numpy()

            # Sample points in a circle around the robot
            theta = np.linspace(0, 2 * np.pi, num_samples)
            sample_x = robot_pos[0] + sample_radius * np.cos(theta)
            sample_y = robot_pos[1] + sample_radius * np.sin(theta)

            # Get terrain heights at sample points
            sample_heights = []
            for x, y in zip(sample_x, sample_y):
                # Convert world coordinates to terrain grid coordinates
                terrain_x = (x - self.cfg.terrain.x_init) / self.cfg.terrain.horizontal_scale
                terrain_y = (y - self.cfg.terrain.y_init) / self.cfg.terrain.horizontal_scale

                if 0 <= terrain_x < self.terrain.height_field_raw.shape[0] and 0 <= terrain_y < \
                        self.terrain.height_field_raw.shape[1]:
                    h = self.terrain.get_height(x, y)
                    sample_heights.append(h)

            if sample_heights:
                # Calculate roughness as the standard deviation of heights
                roughness[i] = torch.tensor(np.std(sample_heights), device=self.device)

        return roughness

    def select_policy(self, env_ids):
        """Select which policy to use based on terrain roughness"""
        roughness = self.compute_terrain_roughness(env_ids)

        # Update active policies
        for i, env_id in enumerate(env_ids):
            if roughness[i] > self.roughness_threshold:
                # Use walking policy for rough terrain
                self.active_policies[env_id] = 0
            else:
                # Use running policy for smooth terrain
                self.active_policies[env_id] = 1

        walking_envs = env_ids[self.active_policies[env_ids] == 0]
        running_envs = env_ids[self.active_policies[env_ids] == 1]

        return walking_envs, running_envs

    def step(self, actions):
        """Override step method to use MoE selection"""
        # Get all environment IDs
        env_ids = torch.arange(self.num_envs, device=self.device)

        # Select policies based on terrain
        walking_envs, running_envs = self.select_policy(env_ids)

        # Process observations using appropriate model
        with torch.no_grad():
            if len(walking_envs) > 0:
                # Get observations for walking environments
                walk_obs = self.get_observations(walking_envs)
                # Get actions from walking policy
                actions[walking_envs] = self.walk_policy.act_inference(walk_obs)

            if len(running_envs) > 0:
                # Get observations for running environments
                run_obs = self.get_observations(running_envs)
                # Get actions from running policy
                actions[running_envs] = self.run_policy.act_inference(run_obs)

        # Execute actions in the environment
        return super().step(actions)

    def reset(self):
        """Reset the environment and MoE state"""
        # Reset active policy tracking
        self.active_policies = np.zeros(self.num_envs, dtype=np.int32)
        return super().reset()