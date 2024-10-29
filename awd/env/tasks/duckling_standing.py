import torch
from isaacgym.torch_utils import *
from env.tasks.duckling_amp_task import DucklingAMPTask

class DucklingStanding(DucklingAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        
        # Define commands for position and orientation
        self.commands = torch.zeros(self.num_envs, 6, device=self.device, requires_grad=False)
        self.command_linear_range = torch.tensor(cfg["env"]["randomCommandRanges"]["linear"], device=self.device)
        self.command_angular_range = torch.deg2rad(torch.tensor(cfg["env"]["randomCommandRanges"]["angular"], device=self.device))

        # Reward scales
        self.rew_scales = {
            "lin_tracking": cfg["env"]['learn']["linTrackingRewardScale"],
            "ang_tracking": cfg["env"]['learn']["angTrackingRewardScale"],
            "action_rate": cfg["env"]["learn"]["actionRateRewardScale"] * self.dt,
            "lin_vel_penalize": cfg["env"]["learn"]["linVelPenalizeScale"],
            "ang_vel_penalize": cfg["env"]["learn"]["angVelPenalizeScale"], 
            "orient_penalize": cfg["env"]["learn"]["orientPenalizeScale"]
        }

        # Initialize target root states
        self.target_root_states = self._initial_duckling_root_states.clone()

        # Command change steps
        self._command_change_steps_min = cfg["env"]["commandChangeStepsMin"]
        self._command_change_steps_max = cfg["env"]["commandChangeStepsMax"]
        self._command_change_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

    def _compute_reward(self, actions):
        # Compute action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        # Compute main reward using jit function
        tracking_reward = compute_standing_reward(
            self._duckling_root_states,
            self.target_root_states,
            self.rew_scales["lin_tracking"],
            self.rew_scales["ang_tracking"],
        )

        #self.commands[:, 4] += 0.005

        # Compute velocity and orientation penalties
        rew_lin_vel = torch.sum(torch.square(self._rigid_body_vel[:, 0, :2]), dim=1) * self.rew_scales["lin_vel_penalize"]
        rew_ang_vel = torch.sum(torch.square(self._rigid_body_ang_vel[:, 0]), dim=1) * self.rew_scales["ang_vel_penalize"]
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient_penalize"]

        # Combine all rewards
        self.rew_buf[:] = tracking_reward + rew_action_rate + rew_lin_vel + rew_ang_vel + rew_orient

        #print(tracking_reward[:1], rew_action_rate[:1], rew_lin_vel[:1], rew_ang_vel[:1], rew_orient[:1])

    def get_task_obs_size(self):
        return 6

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            obs = self.commands 
        else:
            obs = self.commands[env_ids] 
        return obs

    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._command_change_steps
        reset_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self._reset_task(reset_env_ids)

    def _reset_task(self, env_ids):
        change_steps = torch.randint(
            low=self._command_change_steps_min,
            high=self._command_change_steps_max,
            size=(len(env_ids),),
            device=self.device,
            dtype=torch.int64
        )
        
        # Sample height commands
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_linear_range[0],
            self.command_linear_range[1],
            (len(env_ids), 1),
            device=self.device
        ).squeeze()

        # Sample pitch commands
        self.commands[env_ids, 3:6] = torch_rand_float(
            self.command_angular_range[0],
            self.command_angular_range[1],
            (len(env_ids), 3),
            device=self.device
        ).squeeze()
        
        # self.commands[env_ids, 0] = torch_rand_float(
        #     self.command_linear_range[0]/3,
        #     self.command_linear_range[1]/3,
        #     (len(env_ids), 1),
        #     device=self.device
        # ).squeeze()

        # Update target root height
        self.target_root_states[env_ids, 2] = self._initial_duckling_root_states[env_ids, 2] + self.commands[env_ids, 2]

        # Update target root orientation
        delta_quat = quat_from_euler_xyz(
            self.commands[env_ids, 3],
            self.commands[env_ids, 4],
            self.commands[env_ids, 5]
        )
        self.target_root_states[env_ids, 3:7] = quat_mul(self._initial_duckling_root_states[env_ids, 3:7], delta_quat)
        self._command_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

@torch.jit.script
def lgsk_kernel(x: torch.Tensor, scale: float = 50.0, eps:float=2) -> torch.Tensor:
    """Defines logistic kernel function to bound input to [-0.25, 0)

    Ref: https://arxiv.org/abs/1901.08652 (page 15)

    Args:
        x: Input tensor.
        scale: Scaling of the kernel function (controls how wide the 'bell' shape is')
        eps: Controls how 'tall' the 'bell' shape is.

    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    return 1.0 / (scaled.exp() + eps + (-scaled).exp())

@torch.jit.script
def compute_standing_reward(
    duckling_root_states,
    target_root_states,
    reward_linear_scale,
    reward_angular_scale,
):
    # type: (Tensor, Tensor, float, float) -> Tensor
    # Compute position error
    pos_error = torch.norm(duckling_root_states[:, 2] - target_root_states[:, 2], p=2, dim=-1)
    pos_reward = reward_linear_scale * lgsk_kernel(pos_error, scale=50., eps=2.)

    # Compute orientation error
    error_quat = torch.abs(quat_diff(target_root_states[:, 3:7], duckling_root_states[:, 3:7]))
    rot_reward =  reward_angular_scale / (3. * torch.abs(error_quat) + 0.01)

    return pos_reward + rot_reward

