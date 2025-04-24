#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import re

def find_latest_stats_files(stats_dir="stats/"):
    """Finds the latest set of oracle statistics files based on timestamp."""
    if not os.path.isdir(stats_dir):
        print(f"Error: Statistics directory '{stats_dir}' not found.")
        return None, None, None

    # Find all tracking error files and extract timestamps
    error_files = glob.glob(os.path.join(stats_dir, "oracle_*_tracking_errors.npy"))
    if not error_files:
        print("No oracle tracking error files found.")
        return None, None, None

    timestamps = []
    for f in error_files:
        match = re.search(r"oracle_(\d{8}_\d{6})_tracking_errors\.npy", os.path.basename(f))
        if match:
            timestamps.append(match.group(1))

    if not timestamps:
        print("Could not extract timestamps from file names.")
        return None, None, None

    latest_timestamp = sorted(timestamps)[-1]
    print(f"Using latest timestamp: {latest_timestamp}")

    base_pattern = os.path.join(stats_dir, f"oracle_{latest_timestamp}")
    errors_file = f"{base_pattern}_tracking_errors.npy"
    positions_file = f"{base_pattern}_ee_positions.npy"
    ref_positions_file = f"{base_pattern}_ee_ref_positions.npy"

    # Check if all corresponding files exist
    if not os.path.exists(errors_file):
        print(f"Error: File not found: {errors_file}")
        errors_file = None
    if not os.path.exists(positions_file):
        print(f"Error: File not found: {positions_file}")
        positions_file = None
    if not os.path.exists(ref_positions_file):
        print(f"Error: File not found: {ref_positions_file}")
        ref_positions_file = None
        
    if not all([errors_file, positions_file, ref_positions_file]):
        print("Missing one or more data files for the latest timestamp.")
        return None, None, None

    return errors_file, positions_file, ref_positions_file

def plot_statistics(errors_file, positions_file, ref_positions_file):
    """Loads data and generates plots."""
    try:
        tracking_errors = np.load(errors_file)
        ee_positions = np.load(positions_file)
        ee_ref_positions = np.load(ref_positions_file)
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    if ee_positions.shape[0] == 0 or ee_ref_positions.shape[0] == 0 or tracking_errors.shape[0] == 0:
        print("One or more data files are empty. Skipping plotting.")
        return
        
    if ee_positions.shape != ee_ref_positions.shape:
         print(f"Warning: Position data shapes mismatch. ee_pos: {ee_positions.shape}, ee_ref: {ee_ref_positions.shape}")
         min_len = min(ee_positions.shape[0], ee_ref_positions.shape[0], tracking_errors.shape[0])
         print(f"Truncating data to shortest length: {min_len}")
         tracking_errors = tracking_errors[:min_len]
         ee_positions = ee_positions[:min_len, :]
         ee_ref_positions = ee_ref_positions[:min_len, :]


    time_steps = np.arange(len(tracking_errors))

    # --- Plot 1: Tracking Error ---
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, tracking_errors)
    plt.title("End-Effector Tracking Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Tracking Error (norm)")
    plt.grid(True)

    # --- Plot 2: 3D Trajectories (Two Views) ---
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle("End-Effector Trajectory (Actual vs. Reference)")

    # View 1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], label='Actual EE Path', c='b')
    ax1.plot(ee_ref_positions[:, 0], ee_ref_positions[:, 1], ee_ref_positions[:, 2], label='Reference EE Path', c='r', linestyle='--')
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.set_zlabel("Z position")
    ax1.set_title("View 1 (elev=30, azim=-60)")
    ax1.legend()
    ax1.view_init(elev=30., azim=-60)
    # Make axes equal
    max_range = np.array([ee_positions[:,0].max()-ee_positions[:,0].min(), 
                          ee_positions[:,1].max()-ee_positions[:,1].min(), 
                          ee_positions[:,2].max()-ee_positions[:,2].min()]).max() / 2.0
    mid_x = (ee_positions[:,0].max()+ee_positions[:,0].min()) * 0.5
    mid_y = (ee_positions[:,1].max()+ee_positions[:,1].min()) * 0.5
    mid_z = (ee_positions[:,2].max()+ee_positions[:,2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)


    # View 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], label='Actual EE Path', c='b')
    ax2.plot(ee_ref_positions[:, 0], ee_ref_positions[:, 1], ee_ref_positions[:, 2], label='Reference EE Path', c='r', linestyle='--')
    ax2.set_xlabel("X position")
    ax2.set_ylabel("Y position")
    ax2.set_zlabel("Z position")
    ax2.set_title("View 2 (elev=10, azim=45)")
    ax2.legend()
    ax2.view_init(elev=10., azim=45)
    # Make axes equal using same limits as ax1
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_zlim(ax1.get_zlim())


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

if __name__ == "__main__":
    stats_dir = "stats/" 
    # Ensure the stats directory exists
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        print(f"Created statistics directory: {stats_dir}")
        print("Please run the oracle node first to generate data.")
    else:
        errors_f, positions_f, ref_positions_f = find_latest_stats_files(stats_dir)
        if errors_f and positions_f and ref_positions_f:
             plot_statistics(errors_f, positions_f, ref_positions_f)
        else:
             print("Could not find or load the necessary statistics files. Please run the oracle node.") 