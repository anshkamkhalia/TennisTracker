# synthetically generated data
# will be used to train an EoT network (end of trajectory)

import numpy as np
import os

os.makedirs("synthetic_data", exist_ok=True)

# define constants
COURT_LENGTH = 23.77 # standard court dimensions
COURT_WIDTH = 8.23
G = 9.81 # gravity constant

# x -> side to side (alley to alley)
# y -> forward to backward (baseline to net)
# z -> height

# adjust as needed

# typical groundstroke, medium contact, medium speed, wide arc
contact_height_range = (0.8, 1.5)
forward_velocity_range = (20, 39)
vertical_velocity_range = (2, 6)
resitution = 0.8

# serves, higher contact, higher speed, reduced arc
contact_height_range_serve = (2.2, 2.8)
forward_velocity_range_serve = (40, 60)
vertical_velocity_range_serve = (1, 5)
resitution_serve = 0.95

# lob, smaller range for contact, low speed, high arc
contact_height_range_lob = (1, 2)
forward_velocity_range_lob = (12, 20)
vertical_velocity_range_lob = (10, 18)
resitution_lob = 0.7

def simulate_trajectory(initial_position, initial_velocities, seq_len=120, restitution=0.7, jitter_std=0.001):
    x0, y0, z0 = initial_position
    v_x, v_y, v_z = initial_velocities

    # first flight
    a = -0.5 * G
    b = v_z
    c = z0

    disc = b**2 - 4*a*c
    if disc < 0:
        return None

    t_land1 = max((-b + np.sqrt(disc)) / (2*a),
                  (-b - np.sqrt(disc)) / (2*a))
    if t_land1 <= 0:
        return None

    # velocity right before bounce
    v_impact = v_z - G * t_land1

    # bounce physics
    v_z2 = -restitution * v_impact

    # horizontal damping
    v_x2 = v_x * 0.95
    v_y2 = v_y * 0.93

    # bounce position
    x_bounce = x0 + v_x * t_land1
    y_bounce = y0 + v_y * t_land1
    z_bounce = 0.0

    # second flight
    a2 = -0.5 * G
    b2 = v_z2
    c2 = 0.0

    disc2 = b2**2 - 4*a2*c2
    if disc2 < 0:
        return None

    t_land2 = max((-b2 + np.sqrt(disc2)) / (2*a2),
                  (-b2 - np.sqrt(disc2)) / (2*a2))
    if t_land2 <= 0:
        return None

    # stop halfway through second bounce
    second_fraction = 0.5
    t_cutoff = t_land1 + second_fraction * t_land2

    # generate sequence
    times = np.linspace(0, t_cutoff, seq_len)
    x_vals, y_vals, z_vals = [], [], []

    for t in times:
        if t <= t_land1:
            x = x0 + v_x * t
            y = y0 + v_y * t
            z = z0 + v_z * t - 0.5 * G * t**2
        else:
            t2 = t - t_land1
            x = x_bounce + v_x2 * t2
            y = y_bounce + v_y2 * t2
            z = v_z2 * t2 - 0.5 * G * t2**2

        # add small jitter
        x += np.random.normal(0, jitter_std)
        y += np.random.normal(0, jitter_std)
        z += np.random.normal(0, jitter_std)

        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(max(z, 0))

    # convert to array
    sequence = np.stack([x_vals, y_vals, z_vals], axis=-1)

    # interpolate dropped points to keep seq_len exact
    for dim in range(3):
        arr = sequence[:, dim]
        nans = np.isnan(arr)
        if np.any(nans):
            # linear interpolation for missing points
            not_nans = ~nans
            arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), arr[not_nans])
            sequence[:, dim] = arr

    # label = first bounce location
    label = np.array([x_bounce, y_bounce], dtype=np.float32)

    return sequence.astype(np.float32), label

def generate_dataset(num_samples, stroke_type="groundstroke", seq_len=60):

    sequences = []
    labels = []

    # pick velocity/contact ranges based on stroke type
    if stroke_type == "groundstroke":
        h_range = contact_height_range
        v_f_range = forward_velocity_range
        v_v_range = vertical_velocity_range
        rs = resitution
    elif stroke_type == "serve":
        h_range = contact_height_range_serve
        v_f_range = forward_velocity_range_serve
        v_v_range = vertical_velocity_range_serve
        rs = resitution_serve
    elif stroke_type == "lob":
        h_range = contact_height_range_lob
        v_f_range = forward_velocity_range_lob
        v_v_range = vertical_velocity_range_lob
        rs = resitution_lob
    else:
        raise ValueError("Unknown stroke type")

    for _ in range(num_samples):
        # randomly sample initial position
        x0 = np.random.uniform(0, COURT_WIDTH)
        y0 = np.random.uniform(0, COURT_LENGTH)
        z0 = np.random.uniform(*h_range)
        initial_position = (x0, y0, z0)

        # randomly sample initial velocities
        v_x = np.random.uniform(-2, 2)  # small side-to-side variation
        v_y = np.random.uniform(*v_f_range)
        v_z = np.random.uniform(*v_v_range)
        initial_velocities = (v_x, v_y, v_z)

        # simulate trajectory
        sample = simulate_trajectory(initial_position, initial_velocities, seq_len, restitution=rs)
        if sample is None:
            continue  # skip impossible trajectory

        sequence, label = sample
        sequences.append(sequence)
        labels.append(label)

    # convert to numpy arrays
    sequences = np.array(sequences)  # shape (num_samples, seq_len, 3)
    labels = np.array(labels)        # shape (num_samples, 2)

    return sequences, labels

def generate_mixed_dataset_save_batches(total_samples_per_type=150_000, seq_len=120, batch_size=10_000, save_dir="src/reconstructionV2/synthetic_data"):
    
    for stroke in ["groundstroke", "serve", "lob"]:
        num_batches = total_samples_per_type // batch_size
        
        for b in range(num_batches):
            # generate a batch
            seqs, lbls = generate_dataset(num_samples=batch_size,
                                          stroke_type=stroke,
                                          seq_len=seq_len)
            
            # convert to float32
            seqs = np.array(seqs, dtype=np.float32)
            lbls = np.array(lbls, dtype=np.float32)
            
            # save each batch separately
            seq_file = f"{save_dir}/X_train_batch_{b+1}.npy"
            lbl_file = f"{save_dir}/y_train_batch_{b+1}.npy"
            np.save(seq_file, seqs)
            np.save(lbl_file, lbls)
            
            print(f"saved batch {b+1}/{num_batches} for {stroke}")
            
    print("batches saved")

if __name__ == "__main__":
    generate_mixed_dataset_save_batches()