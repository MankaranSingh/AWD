import argparse
import placo
import time
import threading
import webbrowser
from ischedule import schedule, run_loop
from placo_utils.visualization import *


def rotate_matrix(matrix, angle_degrees, axis='x'):
    # Convert angle to radians
    angle = np.radians(angle_degrees)

    # Define rotation matrices
    if axis == 'x':
        rotation = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Apply the rotation by matrix multiplication
    return matrix @ rotation

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
arg_parser.add_argument("path", help="Path to the URDF")
arg_parser.add_argument("--target", help="Target frame name", default="head")
args = arg_parser.parse_args()

start_time = time.perf_counter()
robot = placo.RobotWrapper(args.path, placo.Flags.ignore_collisions)
# Creating the solver
solver = placo.KinematicsSolver(robot)
target_frame_name = args.target
if "trunk" in target_frame_name:
    solver.mask_fbase(False)
else:
    solver.mask_fbase(True)

# Adding a custom regularization task
regularization_task = solver.add_regularization_task(1e-4)

effector_task = solver.add_frame_task(target_frame_name, np.eye(4))
effector_task.configure(target_frame_name, "soft", 1.0, 1.0)

robot.update_kinematics()
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"loading {args.path} took {elapsed_time:.6f} seconds.")

print("Joint names:")
print(list(robot.joint_names()))

print("Frame names:")
print(list(robot.frame_names()))

viz = robot_viz(robot)
t = 0
dt = 0.01
solver.dt = dt
threading.Timer(1, open_browser).start()

default_head_transform = robot.get_T_world_frame(target_frame_name)


# Retrieving initial position of the feet, com and trunk orientation
T_world_left = robot.get_T_world_frame("left_foot")
T_world_right = robot.get_T_world_frame("right_foot")

# Keep left and right foot on the floor
left_foot_task = solver.add_frame_task("left_foot", T_world_left)
left_foot_task.configure("left_foot", "soft", 1.0, 1.0)

right_foot_task = solver.add_frame_task("right_foot", T_world_right)
right_foot_task.configure("right_foot", "soft", 1.0, 1.0)

@schedule(interval=dt)
def loop():
    global t
    t += dt
    
    target = left_foot_task.position().target_world
    left_foot_task.position().target_world = target

    target = right_foot_task.position().target_world
    right_foot_task.position().target_world = target
    
    # Updating the target
    head_target = default_head_transform.copy()
    # linear_range = [-0.02, 0.02] # forward/backward [m]
    linear_range = [-0.04, 0.01] # up/down [m]

    # linear_range = [-10, 10] # roll [deg]
    # linear_range = [-15, 15] # pitch [deg]
    # linear_range = [-12, 12] # yaw [deg]

    head_target[0, 3] += np.clip(0.15*np.sin(t), linear_range[0], linear_range[1])
    head_target = rotate_matrix(head_target, np.clip(15*np.sin(t), -10, 10), "y")
    # head_target = rotate_matrix(head_target, np.clip(15*np.sin(t), linear_range[0], linear_range[1]), "y")
    # head_target = rotate_matrix(head_target, np.clip(15*np.sin(t), linear_range[0], linear_range[1]), "x")
    effector_task.T_world_frame = head_target

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    joints = []
    for joint_name in robot.joint_names():
        joints.append(robot.get_joint(joint_name))
    
    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, target_frame_name)
    frame_viz("target", effector_task.T_world_frame)


run_loop()