import json
import os
import subprocess
import numpy as np
import argparse

def main(bdx_type, n):
    if bdx_type == "go_bdx":
        slow = 0.221
        medium = 0.336
        fast = 0.568
        dx_max = [0, 0.1]
        dy_max = [0, 0.1]
        dtheta_max = [0, 0.25]
    elif bdx_type == "mini_bdx":
        slow = 0.05
        medium = 0.1
        fast = 0.15
        dx_max = [0, 0.05]
        dy_max = [0, 0.05]
        dtheta_max = [0, 0.25]
    elif bdx_type == "mini2_bdx":
        slow = 0.05
        medium = 0.1
        fast = 0.15
        dx_max = [0, 0.05]
        dy_max = [0, 0.05]
        dtheta_max = [0, 0.25]
    else:
        raise ValueError("Invalid bdx_type. Choose either 'go_bdx' or 'mini_bdx'.")

    presets_dir = f"../awd/data/assets/{bdx_type}/placo_presets"
    tmp_dir = os.path.join(presets_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    preset_speeds = ["medium", "fast"]

    for i in range(n):
        # Randomly select a preset speed
        selected_speed = np.random.choice(preset_speeds)
        # Load the selected preset
        with open(os.path.join(presets_dir, f"{selected_speed}.json")) as file:
            data = json.load(file)

        # Modify dx, dy, dtheta randomly
        data["dx"] = round(np.random.uniform(dx_max[0], dx_max[1]) * np.random.choice([-1, 1]), 2)
        data["dy"] = round(np.random.uniform(dy_max[0], dy_max[1]) * np.random.choice([-1, 1]), 2)
        data["dtheta"] = round(np.random.uniform(dtheta_max[0], dtheta_max[1]) * np.random.choice([-1, 1]), 2)

        tmp_preset = os.path.join(tmp_dir, f"{selected_speed}.json")
        with open(tmp_preset, 'w') as file:
            json.dump(data, file, indent=4)

        if bdx_type in ["mini_bdx", "mini2_bdx"]:
            subprocess.run(['python', "gait_generator.py", "--preset", f"{tmp_preset}", "--name", f"{i}", f"--{bdx_type.split('_')[0]}"])
        else:
            subprocess.run(['python', "gait_generator.py", "--preset", f"{tmp_preset}", "--name", f"{i}"])
    

    speeds = []
    preset_names = []

    script_path = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_path, "../recordings")

    # Iterate through all JSON files in the directory
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

            print(total_speed, preset_name)
            

        if (preset_name == 'slow' and total_speed > slow) or \
            (preset_name == 'medium' and (total_speed <= slow or total_speed > fast)) or \
            (preset_name == 'fast' and total_speed <= medium):
        
            # delete the file
            os.remove(file_path)
            print(f"Deleted {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP data")
    parser.add_argument("--bdx_type", choices=["go_bdx", "mini_bdx", "mini2_bdx"], required=True, help="Type of BDX to generate data for")
    parser.add_argument("--num", type=int, default=100, help="Number of motion files to generate.")
    args = parser.parse_args()
    main(args.bdx_type, args.num)