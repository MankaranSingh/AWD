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
            "action_rate": cfg["env"]["learn"]["actionRateRewardScale"] * self.control_dt, # TODO: check if this is correct
            "lin_vel_penalize": cfg["env"]["learn"]["linVelPenalizeScale"],
            "ang_vel_penalize": cfg["env"]["learn"]["angVelPenalizeScale"], 
        }

        # Initialize target root states
        self.target_root_states = self._initial_duckling_root_states.clone()

        # Command change steps
        self._command_change_steps_min = int(cfg["env"]["commandChangeStepsMin"] / self.control_dt)
        self._command_change_steps_max = int(cfg["env"]["commandChangeStepsMax"] / self.control_dt)
        self._command_change_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

    def _compute_reward(self, actions):
        # Compute all rewards using jit function
        self.rew_buf[:] = compute_standing_reward(
            self._duckling_root_states,
            self.target_root_states,
            self.last_actions,
            self.actions,
            self._rigid_body_vel,
            self._rigid_body_ang_vel,
            self.rew_scales["lin_tracking"],
            self.rew_scales["ang_tracking"],
            self.rew_scales["action_rate"],
            self.rew_scales["lin_vel_penalize"],
            self.rew_scales["ang_vel_penalize"]
        )

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
        mask = torch.rand(len(env_ids), device=self.device) > 0.5
        height_commands = torch_rand_float(
            self.command_linear_range[0],
            self.command_linear_range[1], 
            (len(env_ids), 1),
            device=self.device
        ).squeeze()
        self.commands[env_ids, 2] = height_commands * mask

        # Calculate angular range scale based on height command
        # Scale linearly from 0 at extremes to 1 at height=0
        height_commands = self.commands[env_ids, 2]
        max_height = self.command_linear_range[1]
        min_height = self.command_linear_range[0] 
        
        # Normalize height to [-1,1] range and take absolute value
        height_norm = torch.abs(2 * (height_commands) / (max_height - min_height))
        # Convert to scale factor that goes from 0 at extremes to 1 at center
        angular_scale = 1 - height_norm

        # Sample pitch commands and scale them
        base_angles = torch_rand_float(
            self.command_angular_range[0],
            self.command_angular_range[1],
            (len(env_ids), 3),
            device=self.device
        ).squeeze()
        
        # scale base angles based on height command
        self.commands[env_ids, 3:6] = base_angles * angular_scale.unsqueeze(-1)

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
def lgsk_kernel(x: torch.Tensor, scale: float = 2.0, eps:float=0.01) -> torch.Tensor:
    scaled = x * scale
    return 1.0 / (scaled.exp() + eps + (-scaled).exp())

@torch.jit.script
def quat_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * torch.asin(
        torch.clamp(
            torch.norm(
                mul[:, 0:3],
                p=2, dim=-1), max=1.0)
    )

@torch.jit.script
def compute_standing_reward(
    duckling_root_states: torch.Tensor,
    target_root_states: torch.Tensor,
    last_actions: torch.Tensor,
    actions: torch.Tensor,
    rigid_body_vel: torch.Tensor,
    rigid_body_ang_vel: torch.Tensor,
    reward_linear_tracking_scale: float,
    reward_angular_tracking_scale: float,
    reward_action_rate_scale: float,
    reward_lin_vel_penalize_scale: float,
    reward_ang_vel_penalize_scale: float,
) -> torch.Tensor:
    # Compute position error
    pos_error = torch.abs(duckling_root_states[:, 2] - target_root_states[:, 2])
    pos_reward = reward_linear_tracking_scale / (10. * pos_error + 0.01)

    # Compute orientation error
    error_quat = torch.abs(quat_diff(target_root_states[:, 3:7], duckling_root_states[:, 3:7]))
    rot_reward = reward_angular_tracking_scale / (3. * error_quat + 0.01)

    # Compute action rate penalty
    rew_action_rate = torch.sum(torch.square(last_actions - actions), dim=1) * reward_action_rate_scale

    # Compute velocity penalties
    rew_lin_vel = torch.sum(torch.square(rigid_body_vel[:, 0, :2]), dim=1) * reward_lin_vel_penalize_scale
    rew_ang_vel = torch.sum(torch.square(rigid_body_ang_vel[:, 0]), dim=1) * reward_ang_vel_penalize_scale

    return pos_reward + rot_reward + rew_action_rate + rew_lin_vel + rew_ang_vel
