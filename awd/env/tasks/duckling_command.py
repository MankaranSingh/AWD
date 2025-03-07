ROS=False

import torch
import threading
import numpy as np

if ROS:
    import rospy
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import Int32

import env.tasks.duckling_amp_task as duckling_amp_task
from isaacgym.torch_utils import *
from utils import torch_utils


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
        self.rew_scales["foot_slide"] = self.cfg["env"]["learn"]["footSlideRewardScale"]

        # reward episode sums
        self.episode_reward_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.rew_scales.keys()}

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self._command_change_steps = self.cfg["task"]["randomize"]

        self._command_change_steps_min = int(cfg["env"]["commandChangeStepsMin"] / self.control_dt)
        self._command_change_steps_max = int(cfg["env"]["commandChangeStepsMax"] / self.control_dt)
        self._command_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        self.use_average_velocities = self.cfg["env"]["learn"]["useAverageVelocities"]
        
        # for key in self.rew_scales.keys():
        #      self.rew_scales[key] *= self.dt

        self.rew_scales["torque"] *= self.control_dt
        self.rew_scales["action_rate"] *= self.control_dt

        # rename variables to maintain consistency with anymal env
        self.root_states = self._root_states
        self.dof_state = self._dof_state
        self.dof_pos = self._dof_pos
        self.dof_vel = self._dof_vel
        self.contact_forces = self._contact_forces

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.commands_scale = torch.tensor([self.lin_vel_scale[0], self.lin_vel_scale[1], self.ang_vel_scale], requires_grad=False, device=self.commands.device)
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        
        if ROS:
            # Initialize ROS node in a separate thread
            self.ros_thread = None
            self.ros_running = False
            self.ros_publish_rate = cfg.get("ros_publish_rate", 100)  # Hz
            
            # Initialize target joint positions for ROS control
            self.target_joint_pos_external = torch.zeros_like(self.dof_pos[0], dtype=torch.float, device=self.device)
            self.use_ros_control = True
            
            # Start ROS thread once everything is initialized
            self._start_ros_thread()
        return

    def _start_ros_thread(self):
        """Start the ROS thread in a non-blocking way"""
        self.rospy = rospy
        self.Imu = Imu
        self.JointState = JointState
        self.Int32 = Int32
        
        # Initialize the ROS node
        rospy.init_node('duckling_sim', anonymous=True, disable_signals=True)
        
        # Create publishers
        self.imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=1)
        self.foot_contacts_pub = rospy.Publisher('/feet_switch', Int32, queue_size=1)
        self.joint_state_pub = rospy.Publisher('/current_joint_states', JointState, queue_size=1)
        
        # Create subscriber for target joint states
        self.joint_target_sub = rospy.Subscriber('/target_joint_states', JointState, self._target_joint_states_callback, queue_size=1)
        
        # Start the thread
        self.ros_running = True
        self.ros_thread = threading.Thread(target=self._ros_publish_loop)
        self.ros_thread.daemon = True
        self.ros_thread.start()        
        print("ROS node started successfully.")
    
    def _target_joint_states_callback(self, msg):
        """Callback for receiving target joint states"""
        for i, name in enumerate(msg.name):
            self.target_joint_pos_external[i] = msg.position[i]
        self.target_joint_pos_external -= self._initial_dof_pos[0]
        self.target_joint_pos_external /= self.power_scale
                
    def _ros_publish_loop(self):
        """Main loop for publishing ROS messages"""
        rate = self.rospy.Rate(self.ros_publish_rate)
        
        while self.ros_running and not self.rospy.is_shutdown():
            try:
                self._publish_ros_messages()
                rate.sleep()
            except Exception as e:
                print(f"Error in ROS publish loop: {e}")
                break
    
    def _publish_ros_messages(self):
        """Publish all ROS messages"""
        # Get first environment's data (for single robot case)
        now = self.rospy.Time.now()
        
        # Publish IMU data
        imu_msg = self.Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "base_link"
        
        # Convert quaternion to ROS format (x, y, z, w)
        quat = self._duckling_root_states[0, 3:7].cpu().numpy()
        imu_msg.orientation.x = float(quat[0])
        imu_msg.orientation.y = float(quat[1])
        imu_msg.orientation.z = float(quat[2])
        imu_msg.orientation.w = float(quat[3])
        
        # Angular velocities
        root_ang_vel = self._rigid_body_ang_vel[:, 0]
        root_rot = self._duckling_root_states[:, 3:7]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)[0].cpu().numpy()

        imu_msg.angular_velocity.x = float(local_root_ang_vel[0])
        imu_msg.angular_velocity.y = float(local_root_ang_vel[1])
        imu_msg.angular_velocity.z = float(local_root_ang_vel[2])
        
        self.imu_pub.publish(imu_msg)
        
        # Publish foot contacts as a single Int32
        contact_forces = self._contact_forces[0, self._contact_body_ids, 2].cpu().numpy()
        # Simple encoding: bit 0 = right foot, bit 1 = left foot
        right_foot_contact = int(contact_forces[0] > 1.0)
        left_foot_contact = int(contact_forces[1] > 1.0)
        contact_state = (1 if left_foot_contact else 0) | ((1 if right_foot_contact else 0) << 1)
        self.foot_contacts_pub.publish(self.Int32(contact_state))
        
        # Publish joint states
        joint_state_msg = self.JointState()
        joint_state_msg.header.stamp = now
        
        # Add joint names
        joint_state_msg.name = self.dof_names
        
        # Add joint positions and velocities
        joint_state_msg.position = self._dof_pos[0].cpu().numpy().tolist()
        joint_state_msg.velocity = self._dof_vel[0].cpu().numpy().tolist()
                
        self.joint_state_pub.publish(joint_state_msg)

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 3
        return obs_size

    def pre_physics_step(self, actions):
        if ROS:
            # Check if we should use ROS joint targets instead of policy actions
            if self.use_ros_control:
                # Replace actions with the target joint positions from ROS
                actions = self.target_joint_pos_external.unsqueeze(0).clone().repeat(self.num_envs, 1)
        
        # Continue with normal control
        super().pre_physics_step(actions)
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, duckling_asset):
        super()._build_env(env_id, env_ptr, duckling_asset)
        return

    def _update_task(self):
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
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.05).unsqueeze(1)
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
        self.feet_air_time += self.control_dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground
        #rew_airTime *= torch.norm(self.commands, dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact

        foot_vel = self._rigid_body_vel[:, self._contact_body_ids, :2]
        foot_slide_reward = torch.sum(foot_vel.norm(dim=-1) * contact, dim=1) * self.rew_scales["foot_slide"]

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        rew_lin_vel_x, rew_lin_vel_y, rew_ang_vel_z, rew_torque = compute_task_reward(self._duckling_root_states, self.commands,  self.torques, self.avg_velocities, self.rew_scales, self.use_average_velocities)

        # Penalize motion at zero commands
        rew_standstill = (torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.05)) * self.rew_scales["standstill"]
    
        self.rew_buf[:] = torch.clip(rew_lin_vel_x + rew_lin_vel_y + rew_ang_vel_z, 0., None) + rew_torque + + rew_action_rate + rew_airTime + rew_standstill + foot_slide_reward

        self.episode_reward_sums["lin_vel_x"] += rew_lin_vel_x
        self.episode_reward_sums["lin_vel_y"] += rew_lin_vel_y
        self.episode_reward_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_reward_sums["torque"] += rew_torque 
        self.episode_reward_sums["air_time"] += rew_airTime 
        self.episode_reward_sums["action_rate"] += rew_action_rate
        self.episode_reward_sums["standstill"] += rew_standstill
        self.episode_reward_sums["foot_slide"] += foot_slide_reward
        return

    def close(self):
        # Cleanup ROS resources
        self.ros_running = False
        if self.ros_thread:
            self.ros_thread.join(timeout=1.0)
        
        if hasattr(self, 'rospy') and self.rospy is not None:
            try:
                self.rospy.signal_shutdown("Simulation ended")
            except:
                pass
        
        # Call parent's close method if it exists
        if hasattr(super(), 'close'):
            super().close()

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
