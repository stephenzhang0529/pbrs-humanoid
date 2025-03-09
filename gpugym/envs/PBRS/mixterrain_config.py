# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

import numpy as np
from gpugym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from gpugym.envs.base.legged_robot import LeggedRobot
import torch


class MixedTerrainRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 16  # Reduced for testing
        num_observations = 235  # Keep the same as in the original config
        num_actions = 12  # Keep the same as in the original config
        episode_length_s = 20  # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'heightfield'  # Using heightfield for easier manipulation
        curriculum = False  # Disable curriculum so we can control terrain directly

        # Heightfield parameters
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]

        # Create larger terrain to accommodate both types
        num_rows = 20  # Split into flat (first half) and rough (second half)
        num_cols = 20

        # Terrain dimensions
        terrain_length = 16.0  # Larger area to explore
        terrain_width = 16.0

        # Enable height measurements
        measure_heights = True

        # Define transition between terrain types
        flat_to_rough_ratio = 0.5  # 50% flat, 50% rough

        # Terrain roughness parameters
        rough_amplitude = 0.1  # Amplitude of the rough terrain
        noise_scale = 0.2  # Scale of the noise for rough terrain


class MixedTerrainRobotCfgPPO(LeggedRobotCfgPPO):
    seed = 2
    runner_class_name = 'OnPolicyRunner'


class MixedTerrainRobot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_trimesh(self, terrain_type='mixed'):
        """
        Override the terrain creation method to create a mixed terrain with
        flat and rough sections.
        """
        # Get terrain dimensions
        num_rows = self.cfg.terrain.num_rows
        num_cols = self.cfg.terrain.num_cols
        terrain_width = self.cfg.terrain.terrain_width
        terrain_length = self.cfg.terrain.terrain_length
        horizontal_scale = self.cfg.terrain.horizontal_scale
        vertical_scale = self.cfg.terrain.vertical_scale
        flat_to_rough_ratio = self.cfg.terrain.flat_to_rough_ratio

        # Create heightfield
        height_samples = torch.zeros((num_rows, num_cols), device=self.device, dtype=torch.float)

        # Calculate the row where terrain changes from flat to rough
        transition_row = int(num_rows * flat_to_rough_ratio)

        # Flat terrain (first half)
        # Leave as zeros for flat terrain

        # Rough terrain (second half)
        noise_amplitude = self.cfg.terrain.rough_amplitude
        noise_scale = self.cfg.terrain.noise_scale

        # Create perlin noise for rough terrain
        for i in range(transition_row, num_rows):
            for j in range(num_cols):
                # Simple noise pattern (could be improved with Perlin noise)
                x_coord = j / num_cols * noise_scale
                y_coord = i / num_rows * noise_scale
                noise = torch.sin(x_coord * 20) * torch.cos(y_coord * 20)
                noise = noise * noise_amplitude
                height_samples[i, j] = noise

        # Create trimesh
        vertices, triangles = self._create_trimesh_from_heightfield(height_samples, horizontal_scale, vertical_scale,
                                                                    terrain_width, terrain_length)

        return vertices, triangles

    def _create_trimesh_from_heightfield(self, heightfield, horizontal_scale, vertical_scale, width, length):
        """
        Creates a trimesh from a heightfield.
        """
        num_rows = heightfield.shape[0]
        num_cols = heightfield.shape[1]

        # Create vertices from heightfield
        vertices = torch.zeros((num_rows * num_cols, 3), device=self.device, dtype=torch.float)

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                vertices[idx, 0] = j * horizontal_scale - width / 2
                vertices[idx, 1] = i * horizontal_scale - length / 2
                vertices[idx, 2] = heightfield[i, j] * vertical_scale

        # Create triangles (2 per grid cell)
        num_triangles = 2 * (num_rows - 1) * (num_cols - 1)
        triangles = torch.zeros((num_triangles, 3), device=self.device, dtype=torch.int32)

        triangle_idx = 0
        for i in range(num_rows - 1):
            for j in range(num_cols - 1):
                idx00 = i * num_cols + j
                idx01 = i * num_cols + (j + 1)
                idx10 = (i + 1) * num_cols + j
                idx11 = (i + 1) * num_cols + (j + 1)

                # First triangle
                triangles[triangle_idx, 0] = idx00
                triangles[triangle_idx, 1] = idx10
                triangles[triangle_idx, 2] = idx11
                triangle_idx += 1

                # Second triangle
                triangles[triangle_idx, 0] = idx00
                triangles[triangle_idx, 1] = idx11
                triangles[triangle_idx, 2] = idx01
                triangle_idx += 1

        return vertices, triangles

    def _init_custom_terrain_from_heightfield(self):
        """
        Initialize custom terrain from heightfield.
        """
        vertices, triangles = self._create_trimesh(terrain_type='mixed')
        self.gym.add_triangle_mesh(
            self.sim,
            vertices.cpu().numpy(),
            triangles.cpu().numpy(),
            self.terrain_material,
            self.get_terrain_transform(),
            "terrain_mesh"
        )

    def _setup_terrain(self):
        """
        Override to use custom terrain setup
        """
        # Create materials
        self.terrain_material = self.gym.create_material(
            self.sim, self.cfg.terrain.static_friction,
            self.cfg.terrain.dynamic_friction,
            self.cfg.terrain.restitution
        )

        # Create the terrain
        self._init_custom_terrain_from_heightfield()


# Register the new environment
from gpugym.utils.task_registry import task_registry


def register_mixed_terrain_env():
    """
    Register the mixed terrain environment with the task registry
    """
    # Create configurations
    mixed_terrain_cfg = MixedTerrainRobotCfg()
    mixed_terrain_train_cfg = MixedTerrainRobotCfgPPO()

    # Register the task
    task_registry.register(
        name="mixed_terrain",
        task_class=MixedTerrainRobot,
        cfg=mixed_terrain_cfg,
        train_cfg=mixed_terrain_train_cfg
    )