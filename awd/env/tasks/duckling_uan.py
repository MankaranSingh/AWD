import torch
import numpy as np
import time
import os
from isaacgym.torch_utils import *

from isaacgym import gymtorch

from env.tasks.duckling_amp import DucklingAMP


class DucklingUAN(DucklingAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.uan_history_steps = cfg["env"]["uan_history_steps"]  
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.episode_reward_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                    for name in ["r_sim_to_real", "r_smoothness"]}
        self.extras["episode"] = {}

        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )
        self._default_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )
        
        self.trajectory_size = int(self.max_episode_length_s/self.sim_dt)
        self.reference_positions = torch.zeros((self.num_envs, self.trajectory_size), device=self.device, dtype=torch.float)
        self.reference_velocities = self.reference_positions.clone()
        self.real_velocities = self.reference_positions.clone()
        self.real_positions = self.reference_positions.clone()

        self.pos_vel_errors = torch.zeros((self.num_envs, self.uan_history_steps, 2), device=self.device, dtype=torch.float)
        self.target_positions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        self.target_dof = cfg["env"]["target_dof"]
        self.phase = 0
        self.waves = None
        self.load_uan_data()
        return
    
    def load_uan_data(self):
        data_root = self.cfg["env"]["asset"]["uanDataRoot"]
        sample_paths = os.listdir(data_root)

        waves = []
        for sample_path in sample_paths:
            sample = np.load(os.path.join(data_root, sample_path), allow_pickle=True).item()
            waves.append(sample)
        self.waves = np.array(waves)
        return
    
    def get_obs_size(self):
        return 2*self.uan_history_steps

    def pre_physics_step(self, actions):
        self.actions = actions.clone()
        self.corrective_torque = actions
    
        self.render()
        for _ in range(self.control_freq_inv):
            self.target_positions[:, self.target_dof] = self.reference_positions[:, self.phase]
            self.phase += 1
            self.phase = np.clip(self.phase, 0, self.trajectory_size-1)

            # control strategy
            if self.custom_control: # custom position control
                if self._mask_joint_values is not None:
                    self.target_positions[:, self._mask_joint_ids] = self._mask_joint_values
                self.torques = self.p_gains*(self.target_positions*self.power_scale + self._default_dof_pos - self._dof_pos) - (self.d_gains * self._dof_vel)
                if self.randomize_torques:
                    self.torques *= self.randomize_torques_factors
                self.torques[:, self.target_dof] += self.corrective_torque.squeeze(1)
                self.torques = torch.clip(self.torques, -self.max_efforts, self.max_efforts)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            elif (self._pd_control): # isaac based position contol
                pd_tar = self._action_to_pd_targets(actions) + self._initial_dof_pos
                if self._mask_joint_values is not None:
                    pd_tar[:, self._mask_joint_ids] = self._mask_joint_values
                pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
                self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
            else: # isaac based torque control
                forces = actions * self.motor_efforts.unsqueeze(0) * self.power_scale
                force_tensor = gymtorch.unwrap_tensor(forces)
                if self._mask_joint_values is not None:
                    force_tensor[:, self._mask_joint_ids] = self._mask_joint_values
                self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

            self.gym.simulate(self.sim)
            if self.cfg["args"].test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.projected_gravity = quat_rotate_inverse(self._duckling_root_states[:, 3:7], self.gravity_vec) # update imu at simulation freq.

            if self.cfg["task"]["add_obs_latency"]:
                self.update_obs_latency_buffer()

            self.pos_vel_errors[:, 0, 0] = self._dof_pos[:, self.target_dof] - self.reference_positions[:, self.phase]
            self.pos_vel_errors[:, 0, 1] = self._dof_vel[:, self.target_dof] - self.reference_velocities[:, self.phase]
            self.pos_vel_errors[:, 1:, :] = self.pos_vel_errors[:, :-1, :].clone()
        return

    def post_physics_step(self):
        super().post_physics_step()
        return

    def _compute_observations(self, env_ids=None):
        self.obs_buf[:] = self.pos_vel_errors.reshape(self.num_envs, -1)
    
    def _get_duckling_collision_filter(self):
        return 1 # disable self collisions

    def _compute_reset(self):
        self.reset_buf[:] = self.progress_buf > self.max_episode_length
        return
    
    def _compute_reward(self, actions):
        r_sim_to_real, r_smoothness = uan_reward(self.real_positions[:, self.target_dof], self._dof_pos[:, self.target_dof], self.last_actions.squeeze(1), self.actions.squeeze(1))
        self.rew_buf[:] = r_sim_to_real + r_smoothness
        self.episode_reward_sums["r_sim_to_real"] += r_sim_to_real
        self.episode_reward_sums["r_smoothness"] += r_smoothness 

    def _reset_env_tensors(self, env_ids):       
        super()._reset_env_tensors(env_ids) 
        self.phase = 0
        self.pos_vel_errors[:] = 0
        self.target_positions[:] = 0

        rand_indices = np.random.randint(0, len(self.waves), self.num_envs)
        waves = self.waves[rand_indices]
        
        for i in range(self.num_envs):
            self.reference_positions[i] = torch.tensor(waves[i]["position_targets"][:self.trajectory_size, 0], device=self.device, dtype=torch.float)
            self.reference_velocities[i] = torch.diff(self.reference_positions[i], prepend=self.reference_positions[i, :1]) / self.sim_dt
            self.real_velocities[i] = torch.tensor(waves[i]["actual_velocities"][:self.trajectory_size, 0], device=self.device, dtype=torch.float)
            self.real_positions[i] = torch.tensor(waves[i]["actual_positions"][:self.trajectory_size, 0], device=self.device, dtype=torch.float)
        
        for key in self.episode_reward_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_reward_sums[key][env_ids]/self.max_episode_length)
            self.episode_reward_sums[key][env_ids] = 0.
        return


@torch.jit.script
def uan_reward(q_real, q_sim, prev_action, action):
    """
    Compute the reward for the Unsupervised Actuator Net (UAN) using PyTorch.

    Parameters:
    - q_real: Tensor of shape (num_envs,), real-world joint positions
    - q_sim: Tensor of shape (num_envs,), simulated joint positions
    - prev_action: Tensor of shape (num_envs,), corrective torque from the previous timestep
    - action: Tensor of shape (num_envs,), corrective torque at the current timestep

    Returns:
    - reward: Tensor of shape (num_envs,), computed reward values
    """
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # Compute position error (L1 norm)
    error = torch.abs(q_real - q_sim)

    # Sim-to-real matching reward with multi-scale exponentials
    r_sim_to_real = -1.5 * error \
                    + 4.0 * torch.exp(-100 * error**2) \
                    + 4.0 * torch.exp(-300 * error**2) \
                    + 5.0 * torch.exp(-1000 * error**2)

    # Smoothness reward (penalizing sudden torque changes)
    r_smoothness = 0.5 * torch.exp(-0.5 * torch.abs(action - prev_action))

    # Total reward
    return r_sim_to_real, r_smoothness