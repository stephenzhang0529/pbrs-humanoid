"""
Mixture of Experts (MoE) implementation for humanoid robot that selects
between walking and running models based on terrain roughness.
"""

import torch
import numpy as np
import os
from gpugym.envs.PBRS.humanoid import Humanoid
from gpugym.envs.PBRS.humanoid_config import HumanoidCfg, HumanoidCfgPPO
from gpugym.utils.helpers import get_load_path


class HumanoidMoE(Humanoid):
    """
    Humanoid Mixture of Experts class that dynamically selects between
    walking and running models based on terrain roughness.
    """

    def _custom_init(self, cfg):
        super()._custom_init(cfg)

        # Initialize additional variables for MoE
        self.terrain_heights = None
        self.roughness_window_size = cfg.moe.roughness_window_size
        self.roughness_threshold = cfg.moe.roughness_threshold
        self.current_expert = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.expert_blend_factor = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Load the expert policies
        self.experts = self._load_expert_policies(cfg.moe.expert_paths)

        # For tracking purposes
        self.roughness_scores = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _load_expert_policies(self, expert_paths):
        """
        Load the expert policies from the specified paths.

        Args:
            expert_paths: Dictionary mapping expert names to model paths

        Returns:
            Dictionary mapping expert names to loaded policies
        """
        experts = {}

        for expert_name, path in expert_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expert model not found at: {path}")

            print(f"Loading expert '{expert_name}' from {path}")
            loaded_dict = torch.load(path)
            actor_state_dict = loaded_dict['model_state_dict']

            # Create a policy network with the same architecture
            policy = self._create_policy_from_state_dict(actor_state_dict)
            experts[expert_name] = policy

        return experts

    def _create_policy_from_state_dict(self, state_dict):
        """
        Create a policy network from a state dictionary.

        Args:
            state_dict: State dictionary of a policy network

        Returns:
            Policy network with loaded weights
        """
        from gpugym.modules import ActorCritic

        # Create a new policy network
        num_obs = self.cfg.env.num_observations
        num_actions = self.cfg.env.num_actions
        actor_hidden_dims = self.cfg.policy.actor_hidden_dims
        activation = self.cfg.policy.activation

        policy = ActorCritic(num_obs, num_actions, actor_hidden_dims, activation).to(self.device)

        # Load only the actor part of the state dictionary
        actor_state_dict = {k: v for k, v in state_dict.items() if 'actor' in k}
        policy.load_state_dict(actor_state_dict, strict=False)
        policy.eval()  # Set to evaluation mode

        return policy

    def compute_terrain_roughness(self):
        """
        Compute terrain roughness based on height measurements around the robot.

        Returns:
            Tensor containing roughness scores for each environment
        """
        # If no terrain heights available, use height measurements if available
        if hasattr(self, 'height_measurements') and self.height_measurements is not None:
            # Use height measurements to determine roughness
            # Calculate standard deviation of heights in a window around the robot
            roughness = torch.std(self.height_measurements, dim=1)
        else:
            # Fallback to checking if we have direct terrain information
            if self.terrain_heights is not None:
                # Get robot position
                robot_pos = self.root_states[:, :2]  # x, y positions

                # Calculate roughness based on terrain heights in the vicinity of the robot
                # This is a simplified version - in practice, you'd sample the terrain heights
                # around each robot's position
                roughness = self._compute_local_terrain_roughness(robot_pos)
            else:
                # If no terrain information is available, use a default low roughness
                roughness = torch.zeros(self.num_envs, device=self.device)

        return roughness

    def _compute_local_terrain_roughness(self, robot_positions):
        """
        Compute local terrain roughness around each robot position.
        This is a placeholder function - the actual implementation depends on
        how terrain information is stored and accessed in your environment.

        Args:
            robot_positions: Tensor of shape [num_envs, 2] containing x,y positions

        Returns:
            Tensor of shape [num_envs] containing roughness scores
        """
        # This is just a placeholder implementation
        # In a real implementation, you would sample terrain heights in a window
        # around each robot position and compute the standard deviation

        # For now, return constant roughness
        return torch.ones(self.num_envs, device=self.device) * 0.05

    def select_expert(self, roughness_scores):
        """
        Select the appropriate expert based on terrain roughness.

        Args:
            roughness_scores: Tensor of shape [num_envs] containing roughness scores

        Returns:
            Tuple of (expert_indices, blend_factors)
        """
        # Determine which expert to use based on roughness threshold
        # 0 = walking (low roughness), 1 = running (high roughness)
        expert_indices = (roughness_scores < self.roughness_threshold).long()

        # Calculate blend factor for smooth transition between experts
        # This creates a smooth transition zone around the threshold
        blend_range = 0.2 * self.roughness_threshold
        blend_factors = torch.clamp(
            (roughness_scores - (self.roughness_threshold - blend_range)) / (2 * blend_range),
            0.0, 1.0
        )

        return expert_indices, blend_factors

    def compute_actions(self, obs):
        """
        Compute actions using the appropriate expert based on terrain roughness.

        Args:
            obs: Current observations

        Returns:
            Actions to be executed
        """
        # Compute terrain roughness
        roughness_scores = self.compute_terrain_roughness()
        self.roughness_scores = roughness_scores  # Store for logging/debugging

        # Select expert based on roughness
        expert_indices, blend_factors = self.select_expert(roughness_scores)
        self.current_expert = expert_indices
        self.expert_blend_factor = blend_factors

        # Get all expert actions
        walk_actions = None
        run_actions = None

        with torch.no_grad():
            if "walk" in self.experts:
                walk_distribution, _ = self.experts["walk"](obs)
                walk_actions = walk_distribution.sample()

            if "run" in self.experts:
                run_distribution, _ = self.experts["run"](obs)
                run_actions = run_distribution.sample()

        # If we're missing an expert, use the available one
        if walk_actions is None and run_actions is not None:
            return run_actions
        elif run_actions is None and walk_actions is not None:
            return walk_actions

        # Blend actions based on expert selection and blend factors
        # Use walking model for expert_indices == 0, running model for expert_indices == 1
        # Smoothly blend between them using blend_factors for a gradual transition
        blended_actions = torch.zeros_like(walk_actions)

        # Create masks for each expert
        walk_mask = (expert_indices == 0)
        run_mask = (expert_indices == 1)

        # Apply walking model where appropriate
        if walk_mask.any():
            walk_blend = 1.0 - blend_factors[walk_mask]
            blended_actions[walk_mask] += walk_actions[walk_mask] * walk_blend.unsqueeze(1)

            # If in transition zone, also blend in some running actions
            transition_mask = walk_mask & (blend_factors > 0)
            if transition_mask.any():
                run_blend = blend_factors[transition_mask]
                blended_actions[transition_mask] += run_actions[transition_mask] * run_blend.unsqueeze(1)

        # Apply running model where appropriate
        if run_mask.any():
            run_blend = blend_factors[run_mask]
            blended_actions[run_mask] += run_actions[run_mask] * run_blend.unsqueeze(1)

            # If in transition zone, also blend in some walking actions
            transition_mask = run_mask & (blend_factors < 1)
            if transition_mask.any():
                walk_blend = 1.0 - blend_factors[transition_mask]
                blended_actions[transition_mask] += walk_actions[transition_mask] * walk_blend.unsqueeze(1)

        return blended_actions

    def step(self, actions):
        """
        Override the step method to use our MoE computed actions.

        Args:
            actions: Actions from the policy

        Returns:
            Tuple of (obs, rewards, dones, info)
        """
        # For MoE, we ignore the input actions and compute our own from experts
        moe_actions = self.compute_actions(self.obs_buf)

        # Call the parent step method with our MoE actions
        return super().step(moe_actions)

    def reset_idx(self, env_ids):
        """
        Override reset_idx to handle MoE specific resets.

        Args:
            env_ids: Environment IDs to reset
        """
        # Reset parent environment
        super().reset_idx(env_ids)

        # Reset MoE specific variables
        if len(env_ids) > 0:
            self.current_expert[env_ids] = 0
            self.expert_blend_factor[env_ids] = 0.0
            self.roughness_scores[env_ids] = 0.0