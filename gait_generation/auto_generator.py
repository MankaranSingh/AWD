import json
import os
import subprocess
import numpy as np
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import re

def run_command_with_logging(cmd_log_tuple):
    cmd, log_file = cmd_log_tuple
    if log_file is None:
        print(f"cmd: {cmd}")
        subprocess.run(cmd)
    else:
        with open(log_file, "w") as outfile:
            subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT)

def numeric_prefix_sort_key(item):
    total_speed, preset_name = item
    match = re.match(r"(\d+)(.*)", preset_name)
    if match:
        number_part = int(match.group(1))
        rest_part = match.group(2)
        return (number_part, rest_part)
    return (float("inf"), preset_name)

def main(args):
    start_time = time.time()

    if args.bdx_type == "go_bdx":
        slow = 0.221
        medium = 0.336
        fast = 0.568
        dx_max = [0, 0.1]
        dy_max = [0, 0.1]
        dtheta_max = [0, 0.25]
        # Hardcoded sweep parameters for go_bdx
        min_sweep_x = -0.08
        max_sweep_x = 0.12
        min_sweep_y = -0.06
        max_sweep_y = 0.06
        min_sweep_theta = -0.3
        max_sweep_theta = 0.3
        sweep_xy_granularity = 0.04
        sweep_theta_granularity = 0.1
    elif args.bdx_type == "mini_bdx" or args.bdx_type == "mini2_bdx":
        slow = 0.05
        medium = 0.1
        fast = 0.15
        dx_max = [0, 0.05]
        dy_max = [0, 0.05]
        dtheta_max = [0, 0.25]
        # Hardcoded sweep parameters for mini_bdx and mini2_bdx
        min_sweep_x = -0.04
        max_sweep_x = 0.06
        min_sweep_y = -0.03
        max_sweep_y = 0.03
        min_sweep_theta = -0.3
        max_sweep_theta = 0.3
        sweep_xy_granularity = 0.01
        sweep_theta_granularity = 0.03
    else:
        raise ValueError("Invalid bdx_type. Choose either 'go_bdx', 'mini_bdx', or 'mini2_bdx'.")

    presets_dir = f"../awd/data/assets/{args.bdx_type}/placo_presets"
    tmp_dir = os.path.join(presets_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Create log directory
    script_path = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_path, "../recordings")
    log_dir = os.path.join(default_output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    preset_speeds = ["medium"]  # Using medium as primary preset like in auto_waddle

    if args.sweep:
        dxs = np.arange(min_sweep_x, max_sweep_x + sweep_xy_granularity, sweep_xy_granularity)
        dys = np.arange(min_sweep_y, max_sweep_y + sweep_xy_granularity, sweep_xy_granularity)
        dthetas = np.arange(min_sweep_theta, max_sweep_theta + sweep_theta_granularity, sweep_theta_granularity)
        all_n = len(dxs) * len(dys) * len(dthetas)
    else:
        all_n = args.num

    nb_moves_message = f"=== GENERATING {all_n} MOVES ==="
    spacer = "=" * len(nb_moves_message)
    print(spacer)
    print(nb_moves_message)
    print(spacer)

    commands = []
    for i in range(all_n):
        # Randomly select a preset speed
        selected_speed = np.random.choice(preset_speeds)
        # Load the selected preset
        with open(os.path.join(presets_dir, f"{selected_speed}.json")) as file:
            data = json.load(file)

        if args.sweep:
            dx_idx = i % len(dxs)
            dy_idx = (i // len(dxs)) % len(dys)
            dtheta_idx = (i // (len(dxs) * len(dys))) % len(dthetas)

            data["dx"] = round(dxs[dx_idx], 2)
            data["dy"] = round(dys[dy_idx], 2)
            data["dtheta"] = round(dthetas[dtheta_idx], 2)
        else:
            # Modify dx, dy, dtheta randomly
            data["dx"] = round(np.random.uniform(dx_max[0], dx_max[1]) * np.random.choice([-1, 1]), 2)
            data["dy"] = round(np.random.uniform(dy_max[0], dy_max[1]) * np.random.choice([-1, 1]), 2)
            data["dtheta"] = round(np.random.uniform(dtheta_max[0], dtheta_max[1]) * np.random.choice([-1, 1]), 2)

        tmp_preset = os.path.join(tmp_dir, f"{i}_{selected_speed}.json")
        with open(tmp_preset, 'w') as file:
            json.dump(data, file, indent=4)

        if args.bdx_type in ["mini_bdx", "mini2_bdx"]:
            cmd = ['python', "gait_generator.py", "--preset", f"{tmp_preset}", 
                   "--name", f"{i}", f"--{args.bdx_type.split('_')[0]}"]
        else:
            cmd = ['python', "gait_generator.py", "--preset", f"{tmp_preset}", 
                   "--name", f"{i}"]
        
        log_file = None if args.verbose else os.path.join(log_dir, f"{i}.log")
        commands.append((cmd, log_file))

    # Execute commands either sequentially or in parallel
    if args.jobs > 1:
        print(f"Running {args.jobs} parallel jobs")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            executor.map(run_command_with_logging, commands)
    else:
        for cmd in commands:
            run_command_with_logging(cmd)

    # Check and filter recordings
    totals = []
    for filename in os.listdir(default_output_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(default_output_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                
            # Extract the relevant information
            placo_data = data.get("Placo", {})
            avg_x_vel = placo_data.get("avg_x_lin_vel", 0)
            avg_y_vel = placo_data.get("avg_y_lin_vel", 0)
            preset_name = placo_data.get("preset_name", "unknown")
            
            # Calculate the total speed
            total_speed = np.sqrt(avg_x_vel**2 + avg_y_vel**2)
            totals.append((total_speed, preset_name))

            if ('slow' in preset_name and total_speed > slow) or \
                ('medium' in preset_name and (total_speed <= slow or total_speed > fast)) or \
                ('fast' in preset_name and total_speed <= medium):
                # delete the file
                # os.remove(file_path)
                print(f"Deleted {file_path}")

    # Sort and display the results
    totals = sorted(totals, key=numeric_prefix_sort_key)
    for speed, preset_name in totals:
        print(f"Preset: {preset_name}, Total Speed: {speed:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP data")
    parser.add_argument("--bdx_type", choices=["go_bdx", "mini_bdx", "mini2_bdx"], 
                        required=True, help="Type of BDX to generate data for")
    parser.add_argument("--num", type=int, default=100, help="Number of motion files to generate.")
    parser.add_argument("--sweep", action="store_true", 
                       help="Sweep through the dx, dy, dtheta values.")
    parser.add_argument("-j", "--jobs", nargs="?", type=int, const=os.cpu_count(), default=1,
                       help="Number of parallel jobs. If -j is provided without a number, "
                            "uses the number of CPU cores available. Default is 1.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    main(args)