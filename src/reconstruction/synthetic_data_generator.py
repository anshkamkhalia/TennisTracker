# generates synthetic data to train a model to transform 2D->3D

import numpy as np
import os

G = 9.81 # gravity constant
# realistic meters-to-pixels
meters_to_pixels_x = 1280 / 30   # scale X to image width
meters_to_pixels_y = 720 / 50    # scale Y to image height
n_frames = 60  # exactly 60 frames per trajectory

def simulate_trajectory(
        initial_position, # (X0, Y0, Z0)
        initial_velocity, # (vx, vy, vz)
        num_frames,       # number of frames you want
        total_time,
        e,
):

    """simulates 3d trajectory over time"""

    dt = total_time / num_frames  # spread total_time evenly across frames
    t = np.linspace(0, total_time, num_frames)

    # Unpack initial values
    X = initial_position[0]
    Y = initial_position[1]
    Z = initial_position[2]

    vx = initial_velocity[0]
    vy = initial_velocity[1]
    vz = initial_velocity[2]

    positions = []

    for _ in t:
        # update X and Y linearly
        X += vx * dt
        Y += vy * dt

        # update Z using physics
        Z += vz * dt
        vz -= G * dt  # gravity

        # bounce check
        if Z < 0:
            Z = 0
            vz = -e * vz  # reverse velocity and apply damping

        positions.append([X, Y, Z])

    return np.array(positions)  # shape (num_frames, 3)

# METERS — realistic tennis court ranges
X_RANGE = (0, 32) # left-right across court
Y_RANGE = (5, 51) # baseline → net (in meters)
Z_RANGE = (1, 9) # initial hit height

# Velocity ranges (m/s)
VX_RANGE = (-5, 5) # horizontal, side-to-side
VY_RANGE = (10, 35) # toward net
VZ_RANGE = (5, 17) # vertical

# number of trajectories
N = 10

rng = np.random.default_rng()

X_train = []
y_train = []

N_total = 170_000  # total number of trajectories
chunk_size = 10_000  # save 10k per file
output_dir = "src/reconstruction/synthetic_data"
os.makedirs(output_dir, exist_ok=True)

for chunk_idx in range(N_total // chunk_size):
    X_train = []
    y_train = []

    for _ in range(chunk_size):
        # random starting position
        start = (rng.uniform(*X_RANGE),
                 rng.uniform(*Y_RANGE),
                 rng.uniform(*Z_RANGE))

        # random velocities
        velocity = (rng.uniform(*VX_RANGE),
                    rng.uniform(*VY_RANGE),
                    rng.uniform(*VZ_RANGE))

        # random dampening factor
        e = rng.uniform(0.6, 0.9)

        # random total flight time
        total_time = 2 * velocity[2] / G + rng.uniform(0.5, 1.5)

        # simulate trajectory
        traj = simulate_trajectory(start, velocity, n_frames, total_time, e)

        # convert to pixel space
        uv = np.zeros_like(traj[:, :2])
        uv[:,0] = traj[:,0] * meters_to_pixels_x
        uv[:,1] = traj[:,1] * meters_to_pixels_y
        z = traj[:,2]

        X_train.append(uv)
        y_train.append(z)

    # convert to arrays
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    # save chunk
    np.save(os.path.join(output_dir, f"X_train_chunk{chunk_idx+1}.npy"), X_train)
    np.save(os.path.join(output_dir, f"y_train_chunk{chunk_idx+1}.npy"), y_train)

    print(f"saved chunk {chunk_idx+1}/{N_total // chunk_size}")