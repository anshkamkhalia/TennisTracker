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
Y_RANGE = (0, 5)  # slightly behind baseline to allow full rally
Z_RANGE = (0.8, 2.2) # initial hit height

# velocity ranges (m/s) tuned for tennis groundstrokes
VX_RANGE = (-4, 4)  # horizontal, side-to-side
VY_MEAN = 29        # forward velocity mean
VY_STD = 2.5        # forward velocity std
VY_MIN = 18         # clip minimum forward speed
VY_MAX = 33         # clip maximum forward speed
VZ_MEAN = 3.2       # vertical velocity mean
VZ_STD = 0.7        # vertical velocity std
VZ_MIN = 1.8        # clip minimum vertical speed
VZ_MAX = 5.0        # clip maximum vertical speed

# number of trajectories
N = 10

rng = np.random.default_rng()


def main():
    N_total = (12_800) * 100  # total number of trajectories
    chunk_size = 12_800  # save 10k per file
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

            # sample realistic forward velocity with bias for fast rally
            vy = rng.normal(loc=VY_MEAN, scale=VY_STD)
            vy = np.clip(vy, VY_MIN, VY_MAX)

            # sample vertical velocity for flatter arc
            vz = rng.normal(loc=VZ_MEAN, scale=VZ_STD)
            vz = np.clip(vz, VZ_MIN, VZ_MAX)

            # random horizontal velocity
            vx = rng.uniform(*VX_RANGE)

            # pack velocities
            velocity = (vx, vy, vz)

            # vertical restitution for tennis ball-court bounce
            e = rng.uniform(0.45, 0.62)

            # keep first bounce later in the sequence without overextending clip duration
            # (first-bounce time from initial height z0 under gravity)
            z0 = start[2]
            t_first_bounce = (vz + np.sqrt(vz**2 + 2 * G * z0)) / G
            total_time = t_first_bounce + rng.uniform(0.35, 0.65)

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


if __name__ == "__main__":
    main()
