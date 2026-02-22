# creates simulated trajectories for an EoT network

import numpy as np
import os

# gravity constant used for vertical motion
g = 9.81

# simulation timestep (60 fps equivalent)
dt = 1 / 60

# number of consecutive points given to the model
window_size = 120

# directory where batches will be saved
save_dir = "src/reconstructionV2/synthetic_eot_data"
os.makedirs(save_dir, exist_ok=True)

# spatial bounds used for normalization
# these represent the max possible area the object can move in
space_x = 50.0
space_y = 50.0

def simulate_long_motion(total_time=10.0):

    # randomly initialize starting position
    x = np.random.uniform(0, space_x)
    y = np.random.uniform(0, space_y)
    z = np.random.uniform(0.5, 5.0)

    # randomly initialize velocities
    v_x = np.random.uniform(-20, 20)
    v_y = np.random.uniform(-20, 20)
    v_z = np.random.uniform(5, 25)

    # randomize physical properties for variety
    restitution = np.random.uniform(0.6, 0.95)
    friction = np.random.uniform(0.95, 0.995)

    trajectory = []
    bounce_points = []

    t = 0
    while t < total_time:

        # randomly simulate a new external force
        if np.random.rand() < 0.05:
            v_x = np.random.uniform(-20, 20)
            v_y = np.random.uniform(-20, 20)
            v_z = np.random.uniform(5, 25)

        # update position based on velocity
        x += v_x * dt
        y += v_y * dt
        z += v_z * dt

        # apply gravity to vertical velocity
        v_z -= g * dt

        # detect and handle bounce
        if z <= 0:
            z = 0

            # invert vertical velocity with energy loss
            v_z = -restitution * v_z

            # apply horizontal damping at bounce
            v_x *= np.random.uniform(0.85, 0.98)
            v_y *= np.random.uniform(0.85, 0.98)

            # store bounce index and position
            bounce_points.append((len(trajectory), x, y))

            # if vertical energy becomes very small, transition to rolling
            if abs(v_z) < 1.0:
                v_z = 0

        # rolling phase after vertical motion ends
        if z == 0 and v_z == 0:

            # gradually slow down due to friction
            v_x *= friction
            v_y *= friction

            # if nearly stopped, end simulation early
            if np.sqrt(v_x**2 + v_y**2) < 0.1:
                break

        # add small gaussian noise to simulate detection error
        x_obs = x + np.random.normal(0, 0.01)
        y_obs = y + np.random.normal(0, 0.01)

        trajectory.append((x_obs, y_obs))
        t += dt

    return np.array(trajectory), bounce_points


def extract_windows(trajectory, bounce_points, window_size=window_size):

    x_windows = []
    y_labels = []

    # skip if trajectory too short to form a window
    if len(trajectory) < window_size + 1:
        return np.array([]), np.array([])

    for start in range(0, len(trajectory) - window_size):

        end_index = start + window_size - 1

        # find first bounce that occurs after window ends
        next_bounce = None
        for idx, bx, by in bounce_points:
            if idx > end_index:
                next_bounce = (bx, by)
                break

        # skip if no future bounce exists
        if next_bounce is None:
            continue

        window = trajectory[start:start + window_size]

        x_windows.append(window)
        y_labels.append(next_bounce)

    return np.array(x_windows), np.array(y_labels)


def normalize_data(x_data, y_data):

    # normalize x and y coordinates
    x_data[:, :, 0] /= space_x
    x_data[:, :, 1] /= space_y

    # normalize bounce labels
    y_data[:, 0] /= space_x
    y_data[:, 1] /= space_y

    return x_data.astype(np.float32), y_data.astype(np.float32)


def generate_dataset(target_samples):

    x_all = []
    y_all = []

    while len(x_all) < target_samples:

        trajectory, bounces = simulate_long_motion()
        x_windows, y_labels = extract_windows(trajectory, bounces)

        if len(x_windows) == 0:
            continue

        x_all.extend(x_windows)
        y_all.extend(y_labels)

    # trim to exact requested amount
    x_all = np.array(x_all[:target_samples])
    y_all = np.array(y_all[:target_samples])

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