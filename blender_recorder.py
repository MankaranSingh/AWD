import bpy
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

# Set parameters
FPS = bpy.context.scene.render.fps
# Access the scene's unit settings
units = bpy.context.scene.unit_settings

# Get the current unit system
unit_system = units.system  # Options: 'NONE', 'METRIC', 'IMPERIAL'
unit_scale = units.scale_length  # Default is 1.0 for meters

print(unit_scale)
episode = {
    "LoopMode": "Wrap",
    "FPS": FPS,
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Joints": [],
    "Vel_x": [],
    "Vel_y": [],
    "Yaw": [],
    "Frame_offset": [{}],
    "Frame_size": [{}],
    "Frames": [],
    "MotionWeight": 1,
}

# Joint and object names
joint_names = [
    'left_hip_yaw.revolute.bone',
    'left_hip_roll.revolute.bone',
    'left_hip_pitch.revolute.bone',
    'left_knee.revolute.bone',
    'left_ankle.revolute.bone',
    'neck_pitch.revolute.bone',
    'head_pitch.revolute.bone',
    'head_yaw.revolute.bone',
    'head_roll.revolute.bone',
    'left_antenna.revolute.bone',
    'right_antenna.revolute.bone',
    'right_hip_yaw.revolute.bone',
    'right_hip_roll.revolute.bone',
    'right_hip_pitch.revolute.bone',
    'right_knee.revolute.bone',
    'right_ankle.revolute.bone',
]
actual_joint_names = [joint_name.split('.')[0] for joint_name in joint_names]
object_names = {
    "pelvis": "pelvis",
    "left_toe": "left_foot_link",
    "right_toe": "right_foot_link"
}

# Initialize storage arrays
prev_joint_angles = None
prev_left_toe_pos = None
prev_right_toe_pos = None
prev_pelvis_pos = None
prev_pelvis_quat = None
pelvis_positions = []
frames_data = []

# Helper function to calculate angular velocity
def compute_angular_velocity(quat, prev_quat, dt):
    if prev_quat is None:
        return [0.0, 0.0, 0.0]
    r1 = R.from_quat(quat)
    r0 = R.from_quat(prev_quat)
    r_rel = r0.inv() * r1
    axis, angle = r_rel.as_rotvec(), np.linalg.norm(r_rel.as_rotvec())
    angular_velocity = axis * (angle / dt)
    return list(angular_velocity)

# Animation frame range
start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end

# Loop through frames to capture data
for frame in range(start_frame, end_frame + 1):
    bpy.context.scene.frame_set(frame)
    
    frame_joint_angles = [0.0] * len(joint_names)
    frame_data = {}

    # Get joint angles
    for bone in bpy.context.object.pose.bones:
        if bone.name in joint_names:
            frame_joint_angles[joint_names.index(bone.name)] = round(bone.matrix.to_euler().y, 4)

    # Get object positions and quaternion for pelvis
    pelvis_obj = bpy.data.objects[object_names["pelvis"]]
    pelvis_position = pelvis_obj.matrix_world.translation * unit_scale  # Scale the position
    pelvis_quat = pelvis_obj.matrix_world.to_quaternion()

    left_toe_pos = bpy.data.objects[object_names["left_toe"]].matrix_world.translation * unit_scale  # Scale the position
    right_toe_pos = bpy.data.objects[object_names["right_toe"]].matrix_world.translation * unit_scale  # Scale the position

    # Store pelvis position for average velocity and yaw calculation
    pelvis_positions.append([pelvis_position.x, pelvis_position.y, pelvis_position.z])

    # Calculate velocities
    if prev_pelvis_pos is not None:
        world_linear_vel = list((np.array([pelvis_position.x, pelvis_position.y, pelvis_position.z]) - np.array(prev_pelvis_pos)) * FPS)
    else:
        world_linear_vel = [0.0, 0.0, 0.0]

    world_angular_vel = compute_angular_velocity(
        quat=[pelvis_quat.x, pelvis_quat.y, pelvis_quat.z, pelvis_quat.w],
        prev_quat=prev_pelvis_quat,
        dt=1 / FPS
    )

    joints_vel = [0.0] * len(frame_joint_angles) if prev_joint_angles is None else list(
        (np.array(frame_joint_angles) - np.array(prev_joint_angles)) * FPS
    )
    
    left_toe_vel = [0.0, 0.0, 0.0] if prev_left_toe_pos is None else list(
        (np.array([left_toe_pos.x, left_toe_pos.y, left_toe_pos.z]) - np.array(prev_left_toe_pos)) * FPS
    )
    
    right_toe_vel = [0.0, 0.0, 0.0] if prev_right_toe_pos is None else list(
        (np.array([right_toe_pos.x, right_toe_pos.y, right_toe_pos.z]) - np.array(prev_right_toe_pos)) * FPS
    )

    # Append frame data to episode["Frames"]
    frame_data["root_pos"] = [pelvis_position.x, pelvis_position.y, pelvis_position.z]
    frame_data["root_quat"] = [pelvis_quat.x, pelvis_quat.y, pelvis_quat.z, pelvis_quat.w]
    frame_data["joints_pos"] = frame_joint_angles
    frame_data["left_toe_pos"] = [left_toe_pos.x, left_toe_pos.y, left_toe_pos.z]
    frame_data["right_toe_pos"] = [right_toe_pos.x, right_toe_pos.y, right_toe_pos.z]
    frame_data["world_linear_vel"] = world_linear_vel
    frame_data["world_angular_vel"] = world_angular_vel
    frame_data["joints_vel"] = joints_vel
    frame_data["left_toe_vel"] = left_toe_vel
    frame_data["right_toe_vel"] = right_toe_vel

    frames_data.append(frame_data)

    # Update previous frame values
    prev_joint_angles = frame_joint_angles
    prev_left_toe_pos = [left_toe_pos.x, left_toe_pos.y, left_toe_pos.z]
    prev_right_toe_pos = [right_toe_pos.x, right_toe_pos.y, right_toe_pos.z]
    prev_pelvis_pos = [pelvis_position.x, pelvis_position.y, pelvis_position.z]
    prev_pelvis_quat = [pelvis_quat.x, pelvis_quat.y, pelvis_quat.z, pelvis_quat.w]

# Calculate Vel_x, Vel_y, and Yaw based on pelvis positions
pelvis_positions = np.array(pelvis_positions)
velocities = np.diff(pelvis_positions, axis=0) * FPS
episode["Joints"] = actual_joint_names
episode["Vel_x"] = np.mean(velocities[:, 0]).tolist()
episode["Vel_y"] = np.mean(velocities[:, 1]).tolist()
episode["Yaw"] = np.arctan2(np.mean(velocities[:, 1]), np.mean(velocities[:, 0])).tolist()

# Calculate offsets and sizes programmatically
offset = 0
for key, value in frames_data[0].items():
    episode["Frame_offset"][0][key] = offset
    size = len(value) if isinstance(value, list) else 1
    episode["Frame_size"][0][key] = size
    offset += size
    
# Add collected frame data to episode
for frame in frames_data:
    episode["Frames"].append(
        frame["root_pos"] + frame["root_quat"] + frame["joints_pos"] +
        frame["left_toe_pos"] + frame["right_toe_pos"] +
        frame["world_linear_vel"] + frame["world_angular_vel"] + frame["joints_vel"] +
        frame["left_toe_vel"] + frame["right_toe_vel"]
    )

print("Episode data filled.")

# Define the output file path
output_path = "blender.json"

# Save the episode dictionary as a JSON file
with open(output_path, 'w') as f:
    json.dump(episode, f, indent=4)

print(f"Episode data saved to {output_path}")
