# generates synthetic data to train a model to transform 2D->3D
import numpy as np
import os

G = 9.81  # gravity constant
n_frames = 120  # longer trajectories for better temporal context

# principal point (image center)
CX = 640.0
CY = 360.0
# Focal length (pixels)
FX = 800.0  # 50mm equivalent on standard sensor
FY = 800.0

# Image dimensions
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# realistic meters-to-world (for ground truth generation)
meters_to_pixels_x = IMAGE_WIDTH / 30
meters_to_pixels_y = IMAGE_HEIGHT / 50

# tennis ball spatial ranges (in meters)
X_RANGE = (0, 32)  # court length
Y_RANGE = (0, 5)   # court width
Z_RANGE = (1.07, 1.85)  # starting height (racket to net height)

# velocities for realistic tennis shots
VX_RANGE = (-8, 8)  # lateral velocity (side to side)
VY_MEAN = 35  # forward velocity (toward baseline) - faster shots
VY_STD = 2.5
VY_MIN = 30
VY_MAX = 45

VZ_MEAN = 1.4  # upward velocity (lower arc for faster shots)
VZ_STD = 0.3
VZ_MIN = 1.3
VZ_MAX = 1.5

COEFF_RESTITUTION_RANGE = (0.45, 0.62)

rng = np.random.default_rng()

def simulate_trajectory(start, velocity, num_frames, total_time, e):
    """simulate a realistic 3d tennis ball trajectory a bounce"""
    dt = total_time / num_frames
    t = np.linspace(0, total_time, num_frames)

    X, Y, Z = start
    vx, vy, vz = velocity

    positions = []
    for time_step in t:
        # update position with current velocity
        X += vx * dt
        Y += vy * dt
        Z += vz * dt
        
        # apply gravity to vertical velocity (creates the arc)
        vz -= G * dt
        
        # air resistance (slight dampening of horizontal velocities)
        air_resistance = 0.99
        vx *= air_resistance
        vy *= air_resistance
        
        # bounce detection and physics
        if Z < 0:
            # reflect position above ground
            Z = -Z
            
            # reverse vertical velocity and apply energy loss with coefficient of restitution
            vz = -e * vz
            
            # ninimal energy loss on horizontal components due to spin grip on court
            # spin causes the ball to accelerate forward/laterally after bounce (real tennis physics)
            bounce_damping = 0.98  # reduced from 0.92 to allow acceleration
            vx *= bounce_damping
            vy *= bounce_damping
            
            # add spin-induced acceleration boost to forward velocity after bounce
            # this simulates how topspin/backspin causes the ball to grab the court and accelerate
            vy *= rng.uniform(1.05, 1.15)  # 5-15% speed boost from spin effect

        positions.append([X, Y, Z])

    return np.array(positions)

def normalize_with_intrinsics(uv, fx=FX, fy=FY, cx=CX, cy=CY):
    """normalize pixel coordinates using camera intrinsics (proper camera model)"""
    u_norm = (uv[:,0] - cx) / fx
    v_norm = (uv[:,1] - cy) / fy
    return np.stack([u_norm, v_norm], axis=1)

def add_observation_noise(uv, noise_std=0.5):
    """add realistic detection/tracking noise (gaussian, in pixels)"""
    noise = np.random.normal(0, noise_std, uv.shape)
    return uv + noise

def main():
    output_dir = "src/reconstruction/synthetic_data"
    os.makedirs(output_dir, exist_ok=True)

    N_total = 16_384 * 50  # total trajectories
    chunk_size = 16_384

    for chunk_idx in range(N_total // chunk_size):
        X_train, y_train = [], []

        for _ in range(chunk_size):
            start = (
                rng.uniform(*X_RANGE),
                rng.uniform(*Y_RANGE),
                rng.uniform(*Z_RANGE)
            )

            vy = np.clip(rng.normal(VY_MEAN, VY_STD), VY_MIN, VY_MAX)
            vz = np.clip(rng.normal(VZ_MEAN, VZ_STD), VZ_MIN, VZ_MAX)
            vx = rng.uniform(*VX_RANGE)
            velocity = (vx, vy, vz)
            
            # coefficient of restitution for realistic bounces
            e = rng.uniform(*COEFF_RESTITUTION_RANGE)

            z0 = start[2]
            # time to first bounce (using quadratic formula for projectile motion)
            discriminant = vz**2 + 2*G*z0
            if discriminant < 0:
                discriminant = 0
            t_first_bounce = (vz + np.sqrt(discriminant)) / G
            
            # capture the initial arc and one bounce only
            # stop shortly after the first bounce
            total_time = t_first_bounce * rng.uniform(1.05, 1.65)

            traj = simulate_trajectory(start, velocity, n_frames, total_time, e)

            # convert to pixel space
            uv = np.zeros_like(traj[:, :2])
            uv[:,0] = traj[:,0] * meters_to_pixels_x
            uv[:,1] = traj[:,1] * meters_to_pixels_y

            # clip to image bounds (optional, for realism)
            uv[:,0] = np.clip(uv[:,0], 0, IMAGE_WIDTH)
            uv[:,1] = np.clip(uv[:,1], 0, IMAGE_HEIGHT)
            
            # add observation noise (realistic detector/tracker jitter)
            uv_noisy = add_observation_noise(uv, noise_std=rng.uniform(0.3, 1.5))
            uv_noisy[:,0] = np.clip(uv_noisy[:,0], 0, IMAGE_WIDTH)
            uv_noisy[:,1] = np.clip(uv_noisy[:,1], 0, IMAGE_HEIGHT)

            # normalize with camera intrinsics to normalized image plane
            uv_norm = normalize_with_intrinsics(uv_noisy, fx=FX, fy=FY, cx=CX, cy=CY)

            X_train.append(uv_norm)
            y_train.append(traj[:,2])  # keep Z in meters

        # convert to arrays
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        # save chunk
        np.save(os.path.join(output_dir, f"X_train_chunk{chunk_idx+1}.npy"), X_train)
        np.save(os.path.join(output_dir, f"y_train_chunk{chunk_idx+1}.npy"), y_train)

        print(f"saved chunk {chunk_idx+1}/{N_total // chunk_size}")

if __name__ == "__main__":
    main()
