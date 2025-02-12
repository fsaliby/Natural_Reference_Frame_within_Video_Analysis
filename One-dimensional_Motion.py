import numpy as np
from scipy.stats import linregress, norm, shapiro
import matplotlib.pyplot as plt

def load_data(filename):
    """Load time and position data from file."""
    t, x, y = np.loadtxt(filename, unpack=True)
    print(f'Number of data points: {len(t)}')
    print(f'Duration: {np.max(t)-np.min(t):.1f} seconds')
    return t, x, y

def calculate_trajectory_angle(x, y):
    """
    Calculate the angle of the trajectory from linear regression.
    Returns:
        theta: angle in radians
        slope: trajectory slope
        intercept: trajectory intercept
        theta_err: angle uncertainty in radians (using error propagation)
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    theta = np.arctan(slope)
    # Error propagation for θ = arctan(m)
    # σθ = σm/(1 + m²), where σm is std_err from linear regression
    theta_err = std_err/(1 + slope**2)
    return theta, slope, intercept, theta_err

def rotate_coordinates(x, y, theta):
    """Rotate coordinates by given angle."""
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    coords = np.vstack((x, y))
    coords_rot = np.dot(R, coords)
    return coords_rot[0], coords_rot[1]

def calculate_velocities(t, pos):
    """Calculate velocity and its uncertainty from position vs time."""
    slope, intercept, r_value, p_value, std_err = linregress(t, pos)
    return slope, intercept, std_err

def calculate_velocity_angle(vx, vy, vx_err, vy_err):
    """
    Calculate angle from velocity components and its uncertainty.
    Returns angle in radians and its uncertainty.
    """
    theta = np.arctan2(vy, vx)  # using arctan2 for correct quadrant
    
    # Error propagation for θ = arctan(vy/vx)
    dvx = -vy/(vx**2 + vy**2)  # ∂θ/∂vx
    dvy = vx/(vx**2 + vy**2)   # ∂θ/∂vy
    theta_err = np.sqrt((dvx*vx_err)**2 + (dvy*vy_err)**2)
    
    return theta, theta_err

def calculate_velocity_magnitude(vx, vy, vx_err, vy_err):
    """Calculate velocity magnitude and its uncertainty."""
    v_mag = np.sqrt(vx**2 + vy**2)
    # Error propagation: σ|v| = sqrt[(∂|v|/∂vx)²σvx² + (∂|v|/∂vy)²σvy²]
    dvx = vx/v_mag  # ∂|v|/∂vx
    dvy = vy/v_mag  # ∂|v|/∂vy
    v_mag_err = np.sqrt((dvx*vx_err)**2 + (dvy*vy_err)**2)
    return v_mag, v_mag_err

def plot_analysis(t, x, y, x_rot, y_rot, fits):
    """Create plots for trajectory and velocity analysis."""
    plt.figure(figsize=(12, 8))
    
    # Original trajectory
    plt.subplot(221)
    plt.plot(x, y, 'k.', label='Data')
    plt.plot(x, fits['slope_traj']*x + fits['intercept_traj'], 'r-', label='Fit')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.legend()
    plt.title('Original Trajectory')

    # Rotated trajectory
    plt.subplot(222)
    plt.plot(x_rot, y_rot, 'k.', label='Data')
    plt.axhline(0, c='r')
    plt.xlabel('x_rot (cm)')
    plt.ylabel('y_rot (cm)')
    plt.legend()
    plt.title('Rotated Trajectory')

    # Original velocities
    plt.subplot(223)
    plt.plot(t, x, 'b.', label='x data')
    plt.plot(t, y, 'r.', label='y data')
    plt.plot(t, fits['vx']*t + fits['intercept_x'], 'b-', label='x fit')
    plt.plot(t, fits['vy']*t + fits['intercept_y'], 'r-', label='y fit')
    plt.xlabel('t (s)')
    plt.ylabel('Position (cm)')
    plt.legend()
    plt.title('Original Position vs Time')

    # Rotated velocities
    plt.subplot(224)
    plt.plot(t, x_rot, 'b.', label='x_rot data')
    plt.plot(t, y_rot, 'r.', label='y_rot data')
    plt.plot(t, fits['vx_rot']*t + fits['intercept_x_rot'], 'b-', label='x_rot fit')
    plt.plot(t, fits['vy_rot']*t + fits['intercept_y_rot'], 'r-', label='y_rot fit')
    plt.xlabel('t (s)')
    plt.ylabel('Position (cm)')
    plt.legend()
    plt.title('Rotated Position vs Time')

    plt.tight_layout()
    plt.show()

def main():
    # Load data
    t, x, y = load_data('One-dimensional_Motion.txt')
    
    # Calculate trajectory angle
    theta_traj, slope_traj, intercept_traj, theta_err = calculate_trajectory_angle(x, y)
    
    # Rotate coordinates
    x_rot, y_rot = rotate_coordinates(x, y, theta_traj)
    x_rot -= x_rot[0]  # Set initial position to zero
    y_rot -= np.mean(y_rot)  # Center around zero
    
    # Calculate velocities in both reference frames
    vx, intercept_x, vx_err = calculate_velocities(t, x)
    vy, intercept_y, vy_err = calculate_velocities(t, y)
    vx_rot, intercept_x_rot, vx_rot_err = calculate_velocities(t, x_rot)
    vy_rot, intercept_y_rot, vy_rot_err = calculate_velocities(t, y_rot)
    
    # Calculate velocity magnitude
    v_mag, v_mag_err = calculate_velocity_magnitude(vx, vy, vx_err, vy_err)
    
    # Store fit parameters for plotting
    fits = {
        'slope_traj': slope_traj, 'intercept_traj': intercept_traj,
        'vx': vx, 'vy': vy, 'intercept_x': intercept_x, 'intercept_y': intercept_y,
        'vx_rot': vx_rot, 'vy_rot': vy_rot,
        'intercept_x_rot': intercept_x_rot, 'intercept_y_rot': intercept_y_rot
    }
    
    # Calculate angle from velocity components
    theta_vel, theta_vel_err = calculate_velocity_angle(vx, vy, vx_err, vy_err)
    
    # Print results
    print('\nAngle Analysis:')
    print('From trajectory fit:')
    print(f'Trajectory angle (degrees): {np.degrees(theta_traj):.3f} ± {np.degrees(theta_err):.3f}')
    print('From velocity components:')
    print(f'Velocity angle (degrees): {np.degrees(theta_vel):.3f} ± {np.degrees(theta_vel_err):.3f}')
    print(f'Standard deviation of y_rot: {np.std(y_rot, ddof=1):.2f} cm')
    print(f'\nVelocities in original reference frame (cm/s):')
    print(f'vx = {vx:.2f} ± {vx_err:.2f}')
    print(f'vy = {vy:.2f} ± {vy_err:.2f}')
    print(f'|v| = {v_mag:.3f} ± {v_mag_err:.3f}')
    print(f'\nVelocities in rotated reference frame (cm/s):')
    print(f'v_parallel = {vx_rot:.3f} ± {vx_rot_err:.3f}')
    print(f'v_perpendicular = {vy_rot:.3f} ± {vy_rot_err:.3f}')
    
    # Create plots
    plot_analysis(t, x, y, x_rot, y_rot, fits)

if __name__ == "__main__":
    main()
