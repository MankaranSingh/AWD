import torch
from isaacgym.torch_utils import *
from env.tasks.duckling_amp_task import DucklingAMPTask

class DucklingStanding(DucklingAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        # Define commands for position and orientation
        self.commands = torch.zeros(self.num_envs, 6, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([cfg["env"]['learn']["linearScale"]]*3 + [cfg["env"]['learn']["angularScale"]]*3, device=self.device)
        self.command_linear_range = torch.tensor(cfg["env"]["randomCommandRanges"]["linear"], device=self.device)
        self.command_angular_range = torch.deg2rad(torch.tensor(cfg["env"]["randomCommandRanges"]["angular"], device=self.device))

        self.reward_linear_scale = cfg["env"]['learn']["lineardRewardScale"]
        self.reward_angular_scale = cfg["env"]['learn']["angularRewardScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin"] = self.reward_linear_scale
        self.rew_scales["ang"] = self.reward_angular_scale
        self.rew_scales["air_time"] = cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["action_rate"] = cfg["env"]["learn"]["actionRateRewardScale"]

        # Initialize target root states
        self.target_root_states = self._initial_duckling_root_states.clone()

        # Command change steps
        self._command_change_steps_min = cfg["env"]["commandChangeStepsMin"]
        self._command_change_steps_max = cfg["env"]["commandChangeStepsMax"]
        self._command_change_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

        # Adjust reward scales for dt
        self.rew_scales["action_rate"] *= self.dt

    def _compute_reward(self, actions):
        # Compute action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        # Compute main reward using jit function
        main_reward = compute_standing_reward(
            self._duckling_root_states,
            self.target_root_states,
            self.rew_scales["lin"],
            self.rew_scales["ang"]
        )

        # Compute feet position deviation penalty
        feet_pos = self._rigid_body_pos[:, self._contact_body_ids]
        initial_feet_pos = self._initial_rigid_body_pos[:, self._contact_body_ids]
        feet_pos_error = torch.sum((feet_pos - initial_feet_pos)**2, dim=1)
        rew_feet_pos = torch.sum(torch.exp(-0.25 * feet_pos_error), dim=1) * self.rew_scales.get("feet_pos", 1.0)

        rew_lin_vel = torch.sum(torch.abs(self._rigid_body_vel[:, 0, :2]), dim=1) * -1.0
        rew_ang_vel = torch.sum(torch.abs(self._rigid_body_ang_vel[:, 0]), dim=1) * -0.5
        # orientation penalty
        orient_error = torch.sum(torch.abs(self.projected_gravity[:, :2]), dim=1)
        rew_orient = torch.exp(-0.25 * orient_error)

        # Combine all rewards
        self.rew_buf[:] = main_reward + rew_action_rate + rew_lin_vel + rew_ang_vel + rew_orient + rew_feet_pos

        # print(main_reward, rew_action_rate, rew_lin_vel, rew_ang_vel, rew_orient, sep="\n")
        # print(main_reward.shape, rew_action_rate.shape, rew_lin_vel.shape, rew_ang_vel.shape, rew_orient.shape, sep="\n")

    def get_task_obs_size(self):
        return 6

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            obs = self.commands * self.commands_scale
        else:
            obs = self.commands[env_ids] * self.commands_scale
        return obs

    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._command_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        change_steps = torch.randint(low=self._command_change_steps_min, high=self._command_change_steps_max,
                                     size=(len(env_ids),), device=self.device, dtype=torch.int64)
        
        # Sample new commands
        self.commands[env_ids, 2] = torch_rand_float(self.command_linear_range[0], self.command_linear_range[1], (len(env_ids), 1), device=self.device).squeeze()
        #self.commands[env_ids, 0] = torch_rand_float(self.command_linear_range[0]/3, self.command_linear_range[1]/3, (len(env_ids), 1), device=self.device).squeeze()
        #self.commands[env_ids, 3:] = torch_rand_float(self.command_angular_range[0], self.command_angular_range[1], (len(env_ids), 3), device=self.device)

        # Update target root states
        self.target_root_states[env_ids, :3] = self._initial_duckling_root_states[env_ids, :3] + self.commands[env_ids, :3]
        # delta_quat = quat_from_euler_xyz(self.commands[env_ids, 3], self.commands[env_ids, 4], self.commands[env_ids, 5])
        # self.target_root_states[env_ids, 3:7] = quat_mul(delta_quat, self._initial_duckling_root_states[env_ids, 3:7])
        # self._command_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps


@torch.jit.script
def compute_standing_reward(
    duckling_root_states,
    target_root_states,
    reward_linear_scale,
    reward_angular_scale
):
    # type: (Tensor, Tensor, float, float) -> Tensor
    # Compute position error
    pos_error = torch.abs(duckling_root_states[:, 2] - target_root_states[:, 2])
    pos_reward = torch.exp(-0.25 * pos_error)

    # Compute orientation error
    # current_quat = duckling_root_states[:, 3:7]
    # target_quat = target_root_states[:, 3:7]
    # quat_error = quat_diff_rad(current_quat, target_quat)
    # rot_error = torch.sum(quat_error**2, dim=-1)
    # rot_reward = torch.exp(-4.0 * rot_error) * reward_angular_scale

    # Calculate reward (higher when error is lower)
    reward = pos_reward

    return reward