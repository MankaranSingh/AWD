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
    return matrix @ rotation[:len]

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
args = arg_parser.parse_args()

start_time = time.perf_counter()
robot = placo.RobotWrapper(args.path, placo.Flags.ignore_collisions)

print("Joint names:")
print(list(robot.joint_names()))

print("Frame names:")
print(list(robot.frame_names()))

# Creating the solver
solver = placo.KinematicsSolver(robot)

T_world_base = robot.get_T_world_frame("trunk")

# Trunk
com_task = solver.add_com_task(T_world_base[:3, 3])
com_task.configure("com", "soft", 1.0)

# Adding a custom regularization task
regularization_task = solver.add_regularization_task(1e-4)

trunk_orientation_task = solver.add_orientation_task("trunk", np.eye(3))
trunk_orientation_task.configure("trunk_orientation", "soft", 1.0)

# Retrieving initial position of the feet, com and trunk orientation
T_world_left = robot.get_T_world_frame("left_foot")
T_world_right = robot.get_T_world_frame("right_foot")

# Keep left and right foot on the floor
left_foot_task = solver.add_frame_task("left_foot", T_world_left)
left_foot_task.configure("left_foot", "soft", 1.0, 1.0)

right_foot_task = solver.add_frame_task("right_foot", T_world_right)
right_foot_task.configure("right_foot", "soft", 1.0, 1.0)

viz = robot_viz(robot)
t = 0
dt = 0.01
last = 0
solver.dt = dt
start_t = time.time()
robot.update_kinematics()

threading.Timer(1, open_browser).start()

@schedule(interval=dt)
def loop():
    global t
    
    target = left_foot_task.position().target_world
    left_foot_task.position().target_world = target

    target = right_foot_task.position().target_world
    right_foot_task.position().target_world = target

    # Updating the com target with lateral sinusoidal trajectory

    #linear_range = [-0.02, 0.02] # forward/backward [m]
    linear_range = [-0.03, 0.03] # up/down [m]

    # linear_range = [-10, 10] # roll [deg]
    # linear_range = [-15, 15] # pitch [deg]
    # linear_range = [-12, 12] # yaw [deg]

    com_target = T_world_base[:3, 3].copy()
    com_target[[0, 2]] += np.clip(0.1*np.sin(t), linear_range[0], linear_range[1])
    com_task.target_world = com_target
    
    solver.solve(True)
    robot.update_kinematics()

    point_viz("com", robot.com_world(), radius=0.025, color=0xAAAAAA)

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    #robot_frame_viz(robot, "trunk")

    t += dt

run_loop()

