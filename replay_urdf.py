import argparse
import json
import placo
import time
import threading
import webbrowser
import numpy as np
from placo_utils.visualization import *

def open_browser():
    try:
        webbrowser.open_new('http://127.0.0.1:7000/static/')
    except:
        print("Failed to open the default browser. Trying Google Chrome.")
        try:
            webbrowser.get('google-chrome').open_new('http://127.0.0.1:7000/static/')
        except:
            print("Failed to open Google Chrome. Make sure it's installed and accessible.")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("urdf_path", help="Path to the URDF")
arg_parser.add_argument("motion_file", help="Path to the motion JSON file")
arg_parser.add_argument("--frames", help="Frame to display", nargs="+")
args = arg_parser.parse_args()

# Load the robot
robot = placo.RobotWrapper(args.urdf_path, placo.Flags.ignore_collisions)
robot.update_kinematics()

# Load the motion data
episode = json.load(open(args.motion_file))
frames = episode["Frames"]
frame_duration = episode["FrameDuration"]
frame_offsets = episode["Frame_offset"][0]

# Get the joint position slice
joints_pos_slice = slice(frame_offsets["joints_pos"], frame_offsets["left_toe_pos"])

# Setup visualization
viz = robot_viz(robot)
threading.Timer(1, open_browser).start()

print("Joint names:")
print(list(robot.joint_names()))

print("Frame names:")
print(list(robot.frame_names()))

try:
    while True:  # Outer loop for continuous replay
        for frame in frames:
            # Extract joint positions from frame
            joint_positions = frame[joints_pos_slice]
            
            # Update robot state
            for joint_idx, joint_name in enumerate(robot.joint_names()):
                if joint_idx < len(joint_positions):  # Ensure we don't exceed joint positions array
                    robot.set_joint(joint_name, joint_positions[joint_idx])
            
            robot.update_kinematics()
            
            # Display
            viz.display(robot.state.q)
            
            if args.frames:
                for frame_name in args.frames:
                    robot_frame_viz(robot, frame_name)
            
            time.sleep(frame_duration)
except KeyboardInterrupt:
    print("\nPlayback stopped by user")