import torch
import numpy as np
import time
import os
from isaacgym.torch_utils import *

from isaacgym import gymtorch

from env.tasks.duckling_amp import DucklingAMP

def sine_wave(frequency=1, amplitude=45, duration=5, dt=0.01, degrees=False):
    t = np.arange(0, duration, dt)
    if degrees:
        amplitude = amplitude * np.pi / 180  # convert degrees to radians
    return t, amplitude * np.sin(2 * np.pi * frequency * t)

def square_wave(frequency=1, amplitude=45, duration=5, dt=0.01, degrees=False):
    t = np.arange(0, duration, dt)
    if degrees:
        amplitude = amplitude * np.pi / 180  # convert degrees to radians
    return t, amplitude * np.sign(np.sin(2 * np.pi * frequency * t))

def gaussian_noise(new_sample_s=0.4, mean=0, std_dev=15, duration=5, dt=0.01, degrees=False):
    """
    Generates a piecewise constant Gaussian noise signal.
    The noise value is updated every 'frequency' seconds.
    
    Parameters:
        new_sample_s: update interval in seconds (e.g., new_sample_s=0.4 means update every 0.4 sec)
        mean: mean value (in degrees if degrees=True, otherwise in radians)
        std_dev: standard deviation (in degrees if degrees=True, otherwise in radians)
        duration: total duration of the signal in seconds
        dt: time step
        degrees: if True, convert mean and std_dev from degrees to radians, and clip bounds to ±45° in radians
    """
    t = np.arange(0, duration, dt)
    if degrees:
        mean = mean * np.pi / 180
        std_dev = std_dev * np.pi / 180
    
    y = np.zeros_like(t)
    last_update_time = -new_sample_s  # ensures update at t=0
    noise_value = np.clip(np.random.normal(mean, std_dev), -std_dev, std_dev)
    
    for i, time in enumerate(t):
        if time - last_update_time >= new_sample_s:
            noise_value = np.clip(np.random.normal(mean, std_dev), -std_dev, std_dev)
            last_update_time = time
        y[i] = noise_value
    
    return t, y

class DucklingModelJoints(DucklingAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
    
        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )
        self._default_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )
        
        self.position_targets = []
        self.actual_positions = []
        self.actual_velocities = []

        self.wave_types = cfg["env"]["waveTypes"]  
        sine_params = cfg["env"]["sine_params"]
        square_params = cfg["env"]["square_params"]
        gaussian_params = cfg["env"]["gaussian_params"]

        self.waves = []

        if "sine" in self.wave_types:
            for freq, amplitude in zip(sine_params["frequencies"], sine_params["amplitudes"]):
                _, wave = sine_wave(freq, amplitude, self.max_episode_length_s, self.sim_dt, degrees=True)
                self.waves.append(wave)
        if "square" in self.wave_types:
            for freq, amplitude in zip(square_params["frequencies"], square_params["amplitudes"]):
                _, wave = square_wave(freq, amplitude, self.max_episode_length_s, self.sim_dt, degrees=True)
                self.waves.append(wave)
        if "gaussian" in self.wave_types:
            for freq, mean, std in zip(gaussian_params["new_sample_s"], gaussian_params["mean"], gaussian_params["std"]):
                _, wave = gaussian_noise(freq, mean, std, self.max_episode_length_s, self.sim_dt, degrees=True)
                self.waves.append(wave)

        self.phase = -self.control_freq_inv
        self.current_dof = -1
        self.current_wave = 0
        return
    
    def pre_physics_step(self, actions):
        actions = torch.zeros_like(actions)
    
        self.render()
        for _ in range(self.control_freq_inv):
            actions[:, self.current_dof] = self.waves[self.current_wave][self.phase]
            self.phase += 1

            # control strategy
            if self.custom_control: # custom position control
                if self._mask_joint_values is not None:
                    actions[:, self._mask_joint_ids] = self._mask_joint_values
                self.torques = self.p_gains*(actions*self.power_scale + self._default_dof_pos - self._dof_pos) - (self.d_gains * self._dof_vel)
                if self.randomize_torques:
                    self.torques *= self.randomize_torques_factors
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

            self.position_targets.append(actions[:, self.current_dof].detach().cpu().numpy())
            self.actual_positions.append(self._dof_pos[:, self.current_dof].detach().cpu().numpy())
            self.actual_velocities.append(self._dof_vel[:, self.current_dof].detach().cpu().numpy())
        return


    def post_physics_step(self):
        super().post_physics_step()
        return
    
    def _get_duckling_collision_filter(self):
        return 1 # disable self collisions

    def _compute_reset(self):
        self.reset_buf[:] = self.progress_buf > self.max_episode_length
        return
    
    def _save_data(self):
        save_dir = "output/UAN_data"
        os.makedirs(save_dir, exist_ok=True)
        data = {
            "position_targets": np.array(self.position_targets),
            "actual_positions": np.array(self.actual_positions),
            "actual_velocities": np.array(self.actual_velocities)
        }
        np.save(save_dir + f"/{self.dof_names[self.current_dof]}_{self.current_wave}.npy", data)
        return

    def _reset_env_tensors(self, env_ids):       
        super()._reset_env_tensors(env_ids) 
        self.phase = -self.control_freq_inv

        self._save_data()

        self.position_targets = []
        self.actual_positions = []
        self.actual_velocities = []
        
        # Cycle through DOFs and waves
        self.current_dof += 1
        if self.current_dof >= self.num_dof:
            self.current_dof = 0
            self.current_wave += 1

        # End the program if the maximum wave is reached
        if self.current_wave >= len(self.waves):
            print("All waves and DOFs have been cycled through. Ending program.")
            exit(0)

        print("Current DOF: ", self.current_dof, "| Current Wave: ", self.current_wave)
        return
