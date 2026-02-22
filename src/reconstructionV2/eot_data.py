# optimized trajectory generator for eot network

import numpy as np
import os
from multiprocessing import Pool, cpu_count

# gravity constant for vertical motion
g = 9.81

# timestep for simulation (60 fps)
dt = 1 / 60

# number of consecutive points given to model
window_size = 120

# directory to save batches
save_dir = "src/reconstructionV2/synthetic_eot_data"
os.makedirs(save_dir, exist_ok=True)

# spatial bounds for normalization
space_x = 50.0
space_y = 50.0

# simulation parameters
total_time = 5.0  # seconds
min_z, max_z = 0.5, 5.0
v_min, v_max = 5.0, 25.0
vxvy_max = 20.0
restitution_range = (0.6, 0.95)
friction_range = (0.95, 0.995)

def simulate_trajectory():
    
    # initial position
    x = np.random.uniform(0, space_x)
    y = np.random.uniform(0, space_y)
    z = np.random.uniform(min_z, max_z)
    
    # initial velocities
    v_x = np.random.uniform(-vxvy_max, vxvy_max)
    v_y = np.random.uniform(-vxvy_max, vxvy_max)
    v_z = np.random.uniform(v_min, v_max)
    
    # physical properties
    restitution = np.random.uniform(*restitution_range)
    friction = np.random.uniform(*friction_range)
    
    trajectory = []
    bounce_points = []

    num_steps = int(total_time / dt)
    
    for step in range(num_steps):
        # update position
        x += v_x * dt
        y += v_y * dt
        z += v_z * dt
        
        # apply gravity
        v_z -= g * dt
        
        # bounce detection
        if z <= 0:
            z = 0
            v_z = -restitution * v_z
            v_x *= np.random.uniform(0.85, 0.98)
            v_y *= np.random.uniform(0.85, 0.98)
            bounce_points.append((len(trajectory), x, y))
            if abs(v_z) < 1.0:
                v_z = 0

        # rolling phase
        if z == 0 and v_z == 0:
            v_x *= friction
            v_y *= friction
            if np.sqrt(v_x**2 + v_y**2) < 0.1:
                break

        trajectory.append((x, y))
    
    trajectory = np.array(trajectory)
    # add vectorized gaussian noise
    noise = np.random.normal(0, 0.01, size=trajectory.shape)
    trajectory += noise
    
    return trajectory, bounce_points

def extract_windows(trajectory, bounce_points, window_size=window_size):
    # skip if too short
    if len(trajectory) < window_size + 1 or len(bounce_points) == 0:
        return np.empty((0, window_size, 2)), np.empty((0, 2))
    
    # precompute next bounce for each frame
    next_bounce_idx = np.full(len(trajectory), -1, dtype=int)
    bounce_iter = iter(bounce_points)
    bx_idx, bx_x, bx_y = next(bounce_iter)
    
    for i in range(len(trajectory)):
        while bx_idx <= i:
            try:
                bx_idx, bx_x, bx_y = next(bounce_iter)
            except StopIteration:
                bx_idx = -1
                break
        next_bounce_idx[i] = bx_idx

    x_windows = []
    y_labels = []

    for start in range(len(trajectory) - window_size):
        end_index = start + window_size - 1
        next_b = next_bounce_idx[end_index]
        if next_b == -1:
            continue
        # get bounce position
        for idx, bx, by in bounce_points:
            if idx == next_b:
                x_windows.append(trajectory[start:start + window_size])
                y_labels.append([bx, by])
                break

    return np.array(x_windows), np.array(y_labels)

def normalize_data(x_data, y_data):
    x_data[:, :, 0] /= space_x
    x_data[:, :, 1] /= space_y
    y_data[:, 0] /= space_x
    y_data[:, 1] /= space_y
    return x_data.astype(np.float32), y_data.astype(np.float32)

def generate_dataset(target_samples):
    x_all = np.empty((target_samples, window_size, 2), dtype=np.float32)
    y_all = np.empty((target_samples, 2), dtype=np.float32)
    filled = 0

    # multiprocessing pool
    with Pool(cpu_count()) as pool:
        while filled < target_samples:
            # generate 2× target to avoid too many skips
            results = pool.map(simulate_trajectory, [None]*100)
            for trajectory, bounces in results:
                x_win, y_lab = extract_windows(trajectory, bounces)
                n = len(x_win)
                if n == 0:
                    continue
                # number of samples to copy
                n_copy = min(n, target_samples - filled)
                x_all[filled:filled+n_copy] = x_win[:n_copy]
                y_all[filled:filled+n_copy] = y_lab[:n_copy]
                filled += n_copy
                if filled >= target_samples:
                    break
    return normalize_data(x_all, y_all)

def generate_and_save_batches(total_samples=200_000, batch_size=10_000):
    num_batches = total_samples // batch_size
    for b in range(num_batches):
        x_batch, y_batch = generate_dataset(batch_size)
        np.save(f"{save_dir}/x_batch_{b+1}.npy", x_batch)
        np.save(f"{save_dir}/y_batch_{b+1}.npy", y_batch)
        print(f"saved batch {b+1}/{num_batches}")
    print("all batches saved successfully.")

if __name__ == "__main__":
    generate_and_save_batches()