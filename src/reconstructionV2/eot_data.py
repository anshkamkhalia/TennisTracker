# synthetically generated data
# will be used to train an EoT network (end of trajectory)

import numpy as np
import os

os.makedirs("src/reconstructionV2/synthetic_data", exist_ok=True)

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
forward_velocity_range = (22, 35)
vertical_velocity_range = (3, 7)
resitution = 0.8

# serves, higher contact, higher speed, reduced arc
contact_height_range_serve = (2.2, 2.8)
forward_velocity_range_serve = (40, 60)
vertical_velocity_range_serve = (2, 6)
resitution_serve = 0.95

# lob, smaller range for contact, low speed, high arc
contact_height_range_lob = (1, 2)
forward_velocity_range_lob = (12, 20)
vertical_velocity_range_lob = (10, 18)
resitution_lob = 0.7

def simulate_trajectory(initial_position, initial_velocities, seq_len=60, restitution=0.7):

    x0, y0, z0 = initial_position
    v_x, v_y, v_z = initial_velocities

    # first flight
    a = -0.5 * G
    b = v_z
    c = z0

    disc = b**2 - 4*a*c # get discriminant
    if disc < 0:
        return None

    root_1 = (-b + np.sqrt(disc)) / (2*a)
    root_2 = (-b - np.sqrt(disc)) / (2*a)
    t_land1 = max(root_1, root_2)

    # positions during first flight
    times1 = np.linspace(0, t_land1, seq_len // 2)
    x1 = x0 + v_x * times1
    y1 = y0 + v_y * times1
    z1 = z0 + v_z * times1 - 0.5 * G * times1**2
    z1 = np.maximum(z1, 0)

    # velocity after first bounce
    v_z2 = -v_z * restitution
    v_x2 = v_x * 0.95  # small horizontal loss
    v_y2 = v_y * 0.95
    x0_2, y0_2, z0_2 = x1[-1], y1[-1], 0  # start at first bounce

    # second flight
    a2 = -0.5 * G
    b2 = v_z2
    c2 = z0_2

    disc2 = b2**2 - 4*a2*c2
    if disc2 < 0:
        return None # impossible

    # get roots
    root_1b = (-b2 + np.sqrt(disc2)) / (2*a2)
    root_2b = (-b2 - np.sqrt(disc2)) / (2*a2)
    t_land2 = max(root_1b, root_2b)

    times2 = np.linspace(0, t_land2, seq_len - len(times1))
    x2 = x0_2 + v_x2 * times2
    y2 = y0_2 + v_y2 * times2
    z2 = z0_2 + v_z2 * times2 - 0.5 * G * times2**2
    z2 = np.maximum(z2, 0)

    # combine flights
    x_t = np.concatenate([x1, x2])
    y_t = np.concatenate([y1, y2])
    z_t = np.concatenate([z1, z2])

    sequence = np.stack([x_t, y_t, z_t], axis=-1)

    # final landing label (after second bounce)
    x_land = x_t[-1]
    y_land = y_t[-1]
    label = np.array([x_land, y_land], dtype=np.float32)

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

def generate_mixed_dataset_save_batches(total_samples_per_type=300_000, seq_len=60, batch_size=10_000, save_dir="src/reconstructionV2/synthetic_data"):
    
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