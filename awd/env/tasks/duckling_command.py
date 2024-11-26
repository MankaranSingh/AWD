import torch

import env.tasks.duckling_amp_task as duckling_amp_task
from isaacgym.torch_utils import *

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2

class DucklingCommand(duckling_amp_task.DucklingAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_x"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"][0]
        self.rew_scales["lin_vel_y"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"][1]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["standstill"] = self.cfg["env"]["learn"]["standStillRewardScale"]

        # reward episode sums
        self.episode_reward_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.rew_scales.keys()}

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self._command_change_steps = self.cfg["task"]["randomize"]

        self._command_change_steps_min = cfg["env"]["commandChangeStepsMin"]
        self._command_change_steps_max = cfg["env"]["commandChangeStepsMax"]
        self._command_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        self.use_average_velocities = self.cfg["env"]["learn"]["useAverageVelocities"]
        
        # for key in self.rew_scales.keys():
        #      self.rew_scales[key] *= self.dt

        self.rew_scales["torque"] *= self.dt
        self.rew_scales["action_rate"] *= self.dt

        # rename variables to maintain consistency with anymal env
        self.root_states = self._root_states
        self.dof_state = self._dof_state
        self.dof_pos = self._dof_pos
        self.dof_vel = self._dof_vel
        self.contact_forces = self._contact_forces
        self.torques = self.dof_force_tensor

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.commands_scale = torch.tensor([self.lin_vel_scale[0], self.lin_vel_scale[1], self.ang_vel_scale], requires_grad=False, device=self.commands.device)
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        
        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 3
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, duckling_asset):
        super()._build_env(env_id, env_ptr, duckling_asset)
        return

    def _update_task(self):
        # TODO: change commands after certain steps.
        reset_task_mask = self.progress_buf >= self._command_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        change_steps = torch.randint(low=self._command_change_steps_min, high=self._command_change_steps_max,
                                     size=(len(env_ids),), device=self.device, dtype=torch.int64)
        
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)
        self._command_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

        return

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        # distance = torch.norm(self._duckling_root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.1)
        # self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 4)
        # self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0, self.terrain.env_rows)
        # self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        # self._initial_duckling_root_states[env_ids, :3] = self.env_origins[env_ids]

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            obs = self.commands * self.commands_scale
        else:
            obs = self.commands[env_ids] * self.commands_scale
        return obs

    def _compute_reward(self, actions):
        
        contact = self.contact_forces[:, self._contact_body_ids, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground
        #rew_airTime *= torch.norm(self.commands, dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        rew_lin_vel_x, rew_lin_vel_y, rew_ang_vel_z, rew_torque = compute_task_reward(self._duckling_root_states, self.commands,  self.torques, self.avg_velocities, self.rew_scales, self.use_average_velocities)

        # Penalize motion at zero commands
        rew_standstill = (torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)) * self.rew_scales["standstill"]
    
        self.rew_buf[:] = torch.clip(rew_lin_vel_x + rew_lin_vel_y + rew_ang_vel_z + rew_torque, 0., None) + rew_action_rate + rew_airTime + rew_standstill

        self.episode_reward_sums["lin_vel_x"] += rew_lin_vel_x
        self.episode_reward_sums["lin_vel_y"] += rew_lin_vel_y
        self.episode_reward_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_reward_sums["torque"] += rew_torque 
        self.episode_reward_sums["air_time"] += rew_airTime 
        self.episode_reward_sums["action_rate"] += rew_action_rate
        self.episode_reward_sums["standstill"] += rew_standstill
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_task_reward(
    # tensors
    root_states,
    commands,
    torques,
    avg_velocities,
    # Dict
    rew_scales,
    use_average_velocities
):
    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, float], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    if not use_average_velocities:
        base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    else:
        base_lin_vel = quat_rotate_inverse(base_quat, avg_velocities[:, :3])
        base_ang_vel = quat_rotate_inverse(base_quat, avg_velocities[:, 3:])

    # velocity tracking reward
    lin_vel_error_x = torch.square(commands[:, 0] - base_lin_vel[:, 0])
    lin_vel_error_y = torch.square(commands[:, 1] - base_lin_vel[:, 1])
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_x = torch.exp(-lin_vel_error_x/0.25) * rew_scales["lin_vel_x"]
    rew_lin_vel_y = torch.exp(-lin_vel_error_y/0.25) * rew_scales["lin_vel_y"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    return rew_lin_vel_x, rew_lin_vel_y, rew_ang_vel_z, rew_torque
