# synthetic eot dataset generator
# generates full trajectories with multiple bounces
# saves batches to disk with normalization
# uses multiprocessing for speed

import numpy as np
import os
import multiprocessing as mp

# constants for simulation
g = 9.81  # gravity
dt = 1/60  # simulation timestep (60 fps)
window_size = 120  # sequence length for model
total_time = 5.0  # simulate 5 seconds per trajectory

# bounds for normalization
space_x = 50.0
space_y = 50.0
space_z = 10.0

# directory to save batches
save_dir = "src/reconstructionV2/synthetic_eot_data"
os.makedirs(save_dir, exist_ok=True)

# number of trajectories per batch
trajectories_per_batch = 20000

# physics ranges
vel_range_xy = (-20, 20)
vel_range_z = (5, 25)
restitution_range = (0.6, 0.95)
friction_range = (0.95, 0.995)
noise_std = 0.01

def simulate_trajectory(_=None):
    # random initial position
    x = np.random.uniform(0, space_x)
    y = np.random.uniform(0, space_y)
    z = np.random.uniform(0.5, space_z/2)

    # random velocities
    v_x = np.random.uniform(*vel_range_xy)
    v_y = np.random.uniform(*vel_range_xy)
    v_z = np.random.uniform(*vel_range_z)

    # physical properties
    restitution = np.random.uniform(*restitution_range)
    friction = np.random.uniform(*friction_range)

    traj = []
    bounces = []

    t = 0
    while t < total_time:
        # update position
        x += v_x * dt
        y += v_y * dt
        z += v_z * dt

        # gravity
        v_z -= g * dt

        # bounce
        if z <= 0:
            z = 0
            v_z = -restitution * v_z
            v_x *= np.random.uniform(0.85, 0.98)
            v_y *= np.random.uniform(0.85, 0.98)
            bounces.append((len(traj), x, y))

            # stop if vertical energy too low
            if abs(v_z) < 1.0:
                v_z = 0

        # rolling
        if z == 0 and v_z == 0:
            v_x *= friction
            v_y *= friction
            if np.sqrt(v_x**2 + v_y**2) < 0.1:
                break

        # observation noise
        x_obs = x + np.random.normal(0, noise_std)
        y_obs = y + np.random.normal(0, noise_std)

        traj.append((x_obs, y_obs))
        t += dt

    return np.array(traj), bounces

def extract_windows(traj, bounces, window_size=window_size):
    n_windows = max(len(traj) - window_size, 0)
    if n_windows == 0 or len(bounces) == 0:
        return np.empty((0, window_size, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    x_windows = np.zeros((n_windows, window_size, 2), dtype=np.float32)
    y_labels = np.zeros((n_windows, 2), dtype=np.float32)

    bounce_idx = 0
    for i in range(n_windows):
        end_idx = i + window_size - 1

        # advance to the next bounce after the window
        while bounce_idx < len(bounces) and bounces[bounce_idx][0] <= end_idx:
            bounce_idx += 1

        if bounce_idx >= len(bounces):
            break

        next_bounce = bounces[bounce_idx]
        x_windows[i] = traj[i:i + window_size]
        y_labels[i] = next_bounce[1:]  # x, y of bounce

    # remove any empty rows if no future bounce
    valid = np.any(x_windows != 0, axis=(1, 2))
    return x_windows[valid], y_labels[valid]

def normalize(x_data, y_data):
    x_data[:, :, 0] /= space_x
    x_data[:, :, 1] /= space_y
    y_data[:, 0] /= space_x
    y_data[:, 1] /= space_y
    return x_data.astype(np.float32), y_data.astype(np.float32)

def generate_dataset(target_windows):
    x_list = []
    y_list = []

    pool_size = mp.cpu_count()
    with mp.Pool(pool_size) as pool:
        while True:
            # generate many trajectories at once
            results = pool.map(simulate_trajectory, [None]*pool_size)
            for traj, bounces in results:
                x_win, y_lab = extract_windows(traj, bounces)
                if len(x_win) == 0:
                    continue
                x_list.append(x_win)
                y_list.append(y_lab)

            # check if we have enough windows
            total_windows_collected = sum(x.shape[0] for x in x_list)
            if total_windows_collected >= target_windows:
                break

    # concatenate all windows
    x_all = np.concatenate(x_list, axis=0)[:target_windows]
    y_all = np.concatenate(y_list, axis=0)[:target_windows]

    return normalize(x_all, y_all)

def generate_and_save_batches(total_windows=200_000, batch_size=10_000):
    num_batches = total_windows // batch_size
    for b in range(num_batches):
        x_batch, y_batch = generate_dataset(batch_size)
        np.save(f"{save_dir}/x_batch_{b+1}.npy", x_batch)
        np.save(f"{save_dir}/y_batch_{b+1}.npy", y_batch)
        print(f"saved batch {b+1}/{num_batches}")
    print("all batches saved.")

if __name__ == "__main__":
    generate_and_save_batches()