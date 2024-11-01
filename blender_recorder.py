import bpy

# Ensure we're in pose mode
bpy.ops.object.mode_set(mode='POSE')

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

all_frames = []
joint_angles = [0.0]*len(joint_names)


start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end


for frame in range(start_frame, end_frame + 1):
    bpy.context.scene.frame_set(frame)  # Set the frame    
    for bone in bpy.context.object.pose.bones:
        if bone.name in joint_names:
            joint_angles[joint_names.index(bone.name)] = round(bone.matrix.to_euler().y, 4)
    all_frames.append(joint_angles.copy())

print(all_frames)