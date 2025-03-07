import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gpugym.utils.math import *
from gpugym.envs import *
from gpugym.envs import LeggedRobot
from gpugym.envs.PBRS.humanoid import Humanoid
from gpugym.envs.PBRS.humanoid_run import HumanoidRun



class HumanoidMoE(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Create the two expert models
        self.walk_expert = Humanoid(cfg, sim_params, physics_engine, sim_device, headless)
        self.run_expert = HumanoidRun(cfg, sim_params, physics_engine, sim_device, headless)

        # Initialize terrain roughness detection
        self.terrain_heights = None
        self.roughness_threshold = self.cfg.moe.roughness_threshold
        self.current_expert = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Gating network hidden state
        self.gating_hidden = None

    def _custom_init(self, cfg):
        self.dt_step = self.cfg.sim.dt * self.cfg.control.decimation
        self.pbrs_gamma = 0.99
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.eps = 0.2
        self.phase_freq = 1.5  # Between walk (1.0) and run (2.0)

        # MoE specific parameters
        self.gating_network_input_size = 39  # Base observation + terrain roughness
        self.expert_switching_cooldown = torch.zeros(self.num_envs, device=self.device)
        self.min_switching_time = self.cfg.moe.min_switching_time  # Minimum time between expert switches

    def compute_observations(self):
        """Compute observations including terrain roughness"""
        # Get base observations
        self.walk_expert.compute_observations()
        self.run_expert.compute_observations()

        # Calculate terrain roughness metric
        if self.cfg.terrain.measure_heights:
            heights = self.height_points
            # Compute local terrain roughness (std deviation of heights)
            roughness = torch.std(heights, dim=1, keepdim=True)
        else:
            roughness = torch.zeros(self.num_envs, 1, device=self.device)

        # Combine base observations with roughness
        walk_obs = self.walk_expert.obs_buf
        run_obs = self.run_expert.obs_buf

        # Choose observations based on current expert
        self.obs_buf = torch.where(
            self.current_expert.unsqueeze(1).repeat(1, walk_obs.shape[1]) == 0,
            walk_obs,
            run_obs
        )

        # Add roughness to observations
        self.obs_buf = torch.cat([self.obs_buf, roughness], dim=1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def select_expert(self):
        """Determine which expert to use based on terrain roughness"""
        if self.cfg.terrain.measure_heights:
            heights = self.height_points
            roughness = torch.std(heights, dim=1)

            # Apply cooldown to prevent rapid switching
            can_switch = self.expert_switching_cooldown <= 0

            # Switch experts based on roughness and cooldown
            new_expert = torch.where(
                roughness > self.roughness_threshold,
                torch.zeros_like(self.current_expert),  # Walk on rough terrain (0)
                torch.ones_like(self.current_expert)  # Run on smooth terrain (1)
            )

            # Only switch if cooldown allows
            expert_changed = (new_expert != self.current_expert) & can_switch
            self.current_expert = torch.where(
                can_switch,
                new_expert,
                self.current_expert
            )

            # Reset cooldown when expert changes
            self.expert_switching_cooldown = torch.where(
                expert_changed,
                torch.ones_like(self.expert_switching_cooldown) * self.min_switching_time,
                self.expert_switching_cooldown - self.dt_step
            )

    def pre_physics_step(self):
        """Select expert before physics step"""
        self.select_expert()

        # Call pre_physics_step for the currently active experts
        walk_mask = (self.current_expert == 0)
        run_mask = (self.current_expert == 1)

        if torch.any(walk_mask):
            self.walk_expert.pre_physics_step()

        if torch.any(run_mask):
            self.run_expert.pre_physics_step()

        # Store previous rewards for PBRS
        self.rwd_oriPrev = torch.where(
            self.current_expert.unsqueeze(1) == 0,
            self.walk_expert._reward_orientation(),
            self.run_expert._reward_orientation()
        )

        self.rwd_baseHeightPrev = torch.where(
            self.current_expert.unsqueeze(1) == 0,
            self.walk_expert._reward_base_height(),
            self.run_expert._reward_base_height()
        )

        self.rwd_jointRegPrev = torch.where(
            self.current_expert.unsqueeze(1) == 0,
            self.walk_expert._reward_joint_regularization(),
            self.run_expert._reward_joint_regularization()
        )

    def _post_physics_step_callback(self):
        """Update phase and commands after physics step"""
        # Update phase
        self.phase = torch.fmod(self.phase + self.dt, 1.0)

        # Update command resampling
        env_ids = (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt) == 0) \
            .nonzero(as_tuple=False).flatten()

        if self.cfg.commands.resampling_time == -1:
            pass  # when the joystick is used
        else:
            self._resample_commands(env_ids)
            if (self.cfg.domain_rand.push_robots and
                    (self.common_step_counter
                     % self.cfg.domain_rand.push_interval == 0)):
                self._push_robots()

        # Decrement expert switching cooldown
        self.expert_switching_cooldown = torch.max(
            self.expert_switching_cooldown - self.dt_step,
            torch.zeros_like(self.expert_switching_cooldown)
        )

    def _compute_reward(self, actions):
        """Compute rewards based on the active expert"""
        # Compute rewards from each expert
        walk_rewards = torch.zeros(self.num_envs, device=self.device)
        run_rewards = torch.zeros(self.num_envs, device=self.device)

        # Calculate rewards only for envs using that expert
        walk_mask = (self.current_expert == 0)
        run_mask = (self.current_expert == 1)

        if torch.any(walk_mask):
            walk_envs = torch.where(walk_mask)[0]
            walk_actions = actions[walk_envs]
            walk_rewards[walk_envs] = self.walk_expert._compute_reward(walk_actions)

        if torch.any(run_mask):
            run_envs = torch.where(run_mask)[0]
            run_actions = actions[run_envs]
            run_rewards[run_envs] = self.run_expert._compute_reward(run_actions)

        # Add a small reward for appropriate expert selection
        if self.cfg.terrain.measure_heights:
            heights = self.height_points
            roughness = torch.std(heights, dim=1)

            correct_expert_selection = torch.where(
                roughness > self.roughness_threshold,
                self.current_expert == 0,  # Should walk on rough terrain
                self.current_expert == 1  # Should run on smooth terrain
            ).float()

            expert_selection_reward = correct_expert_selection * self.cfg.rewards.scales.expert_selection
        else:
            expert_selection_reward = torch.zeros(self.num_envs, device=self.device)

        # Combine rewards
        rewards = torch.where(
            self.current_expert == 0,
            walk_rewards,
            run_rewards
        ) + expert_selection_reward

        return rewards

    def check_termination(self):
        """Check if environments need to be reset"""
        # Use the more conservative termination conditions from both experts
        walk_reset = self.walk_expert.reset_buf
        run_reset = self.run_expert.reset_buf

        # Combine reset conditions
        self.reset_buf = torch.logical_or(walk_reset, run_reset)

        # Time out condition
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _reward_ori_pb(self):
        """PBRS for orientation"""
        walk_ori = self.walk_expert._reward_orientation()
        run_ori = self.run_expert._reward_orientation()

        current_ori = torch.where(
            self.current_expert == 0,
            walk_ori,
            run_ori
        )

        delta_phi = ~self.reset_buf * (current_ori - self.rwd_oriPrev)
        return delta_phi / self.dt_step

    def _reward_jointReg_pb(self):
        """PBRS for joint regularization"""
        walk_joint = self.walk_expert._reward_joint_regularization()
        run_joint = self.run_expert._reward_joint_regularization()

        current_joint = torch.where(
            self.current_expert == 0,
            walk_joint,
            run_joint
        )

        delta_phi = ~self.reset_buf * (current_joint - self.rwd_jointRegPrev)
        return delta_phi / self.dt_step

    def _reward_baseHeight_pb(self):
        """PBRS for base height"""
        walk_height = self.walk_expert._reward_base_height()
        run_height = self.run_expert._reward_base_height()

        current_height = torch.where(
            self.current_expert == 0,
            walk_height,
            run_height
        )

        delta_phi = ~self.reset_buf * (current_height - self.rwd_baseHeightPrev)
        return delta_phi / self.dt_step

    # Forward all other reward functions to the appropriate expert
    def _reward_tracking_lin_vel(self):
        walk_reward = self.walk_expert._reward_tracking_lin_vel()
        run_reward = self.run_expert._reward_tracking_lin_vel()
        return torch.where(self.current_expert == 0, walk_reward, run_reward)

    def _reward_tracking_ang_vel(self):
        walk_reward = self.walk_expert._reward_tracking_ang_vel()
        run_reward = self.run_expert._reward_tracking_ang_vel()
        return torch.where(self.current_expert == 0, walk_reward, run_reward)

    def _reward_base_height(self):
        walk_reward = self.walk_expert._reward_base_height()
        run_reward = self.run_expert._reward_base_height()
        return torch.where(self.current_expert == 0, walk_reward, run_reward)

    def _reward_orientation(self):
        walk_reward = self.walk_expert._reward_orientation()
        run_reward = self.run_expert._reward_orientation()
        return torch.where(self.current_expert == 0, walk_reward, run_reward)

    def _reward_dof_vel(self):
        walk_reward = self.walk_expert._reward_dof_vel()
        run_reward = self.run_expert._reward_dof_vel()
        return torch.where(self.current_expert == 0, walk_reward, run_reward)

    def _reward_joint_regularization(self):
        walk_reward = self.walk_expert._reward_joint_regularization()
        run_reward = self.run_expert._reward_joint_regularization()
        return torch.where(self.current_expert == 0, walk_reward, run_reward)

    def _reward_ankle_regularization(self):
        walk_reward = self.walk_expert._reward_ankle_regularization()
        run_reward = self.run_expert._reward_ankle_regularization()
        return torch.where(self.current_expert == 0, walk_reward, run_reward)

    def sqrdexp(self, x):
        """Squared exponential helper function"""
        return torch.exp(-torch.square(x) / self.cfg.rewards.tracking_sigma)

    def smooth_sqr_wave(self, phase):
        """Smooth square wave based on current expert"""
        # Get intermediate phase frequency between walk and run based on terrain
        p = 2. * torch.pi * phase * self.phase_freq
        return torch.sin(p) / (1.5 * torch.sqrt(torch.sin(p) ** 2. + self.eps ** 2.)) + 0.5