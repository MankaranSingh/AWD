import torch

import env.tasks.duckling as duckling
import env.tasks.duckling_amp as duckling_amp
import env.tasks.duckling_amp_task as duckling_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class DucklingJump(duckling_amp_task.DucklingAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        # reward scales
        self.rew_scales = {}
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["height_tracking"] = self.cfg["env"]["learn"]["heightTrackingRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self._command_change_steps = self.cfg["task"]["randomize"]

        self.target_height = self.cfg["env"]["learn"]["targetHeight"]

        self._command_change_steps_min = cfg["env"]["commandChangeStepsMin"]
        self._command_change_steps_max = cfg["env"]["commandChangeStepsMax"]
        self._command_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)

        # rename variables to maintain consistency with anymal env
        self.root_states = self._root_states
        self.dof_state = self._dof_state
        self.dof_pos = self._dof_pos
        self.dof_vel = self._dof_vel
        self.contact_forces = self._contact_forces
        self.torques = self.dof_force_tensor
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        
        return

    def get_task_obs_size(self):
        return 0

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
        return

    def _compute_reward(self, actions):
        
        contact = self.contact_forces[:, self._contact_body_ids, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time-0.01), dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground
        #rew_airTime *= torch.norm(self.commands, dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact

        pos_error = torch.abs(self._duckling_root_states[:, 2] - 0.5)
        pos_reward = self.rew_scales["height_tracking"] / (10. * pos_error + 0.01)

        self.rew_buf[:] = pos_reward + rew_airTime
        return
