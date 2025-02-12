import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.stats import linregress
import pandas as pd

def rotate_coordinates(x, y, theta):
    """Rotate coordinates by angle theta (in radians)"""
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot

def fit_quadratic(t, x):
    """Fit quadratic function and return coefficients [a, b, c] and R²"""
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    coeffs = np.linalg.lstsq(A, x, rcond=None)[0]
    
    # Calculate R²
    x_fit = coeffs[0]*t**2 + coeffs[1]*t + coeffs[2]
    r_squared = 1 - np.sum((x - x_fit)**2) / np.sum((x - np.mean(x))**2)
    
    return coeffs, r_squared

def find_acceleration(theta, t, x, y):
    """Return acceleration coefficient for given angle"""
    x_rot, _ = rotate_coordinates(x, y, theta)
    coeffs, _ = fit_quadratic(t, x_rot)
    return coeffs[0]

def find_natural_frame_and_params(t, x, y):
    """Find natural reference frame and physical parameters"""
    # Find angle where horizontal acceleration vanishes
    theta_0 = brentq(lambda theta: find_acceleration(theta, t, x, y), -np.pi/2, np.pi/2, xtol=1.e-12, rtol=1.e-14)
    
    # Get coordinates in rotated frame
    x_rot, y_rot = rotate_coordinates(x, y, theta_0)
    
    # Fit horizontal motion (should be linear)
    slope, intercept, r_value_x, _, _ = linregress(t, x_rot)
    v_x = slope
    
    # Fit vertical motion (quadratic)
    coeffs_y, r_squared_y = fit_quadratic(t, y_rot)
    a, b, c = coeffs_y
    
    # Physical parameters
    g = -2*a  # acceleration is -g/2 in quadratic fit
    v0_y = b  # initial vertical velocity
    
    # Calculate residuals
    x_fit = v_x*t + intercept
    y_fit = a*t**2 + b*t + c
    residuals_x = x_rot - x_fit
    residuals_y = y_rot - y_fit
    
    return {
        'theta': np.degrees(theta_0),
        'g': g,
        'R_squared_x': r_value_x**2,
        'R_squared_y': r_squared_y,
        'residuals_x': residuals_x,
        'residuals_y': residuals_y,
        'x_rot': x_rot,
        'y_rot': y_rot
    }

def split_throws(t, x, y, gap_threshold=0.2):
    """Divide trajectory into individual throws"""
    time_diffs = np.diff(t)
    splits = []
    start = 0
    
    for i, diff in enumerate(time_diffs):
        if diff > gap_threshold:
            splits.append({
                't': t[start:i+1],
                'x': x[start:i+1],
                'y': y[start:i+1]
            })
            start = i + 1
    
    # Add last throw
    splits.append({
        't': t[start:],
        'x': x[start:],
        'y': y[start:]
    })
    return splits

def analyze_ball(t, x, y, ball_name):
    """Analyze all throws for a single ball"""
    # Split into individual throws
    throws = split_throws(t, x, y)
    
    # Analyze each throw
    results = []
    for i, throw in enumerate(throws):
        try:
            params = find_natural_frame_and_params(throw['t'], throw['x'], throw['y'])
            params['ball'] = ball_name
            params['throw_number'] = i + 1
            results.append(params)
        except ValueError as e:
            print(f"Error analyzing {ball_name} throw {i+1}: {e}")
            continue
    
    return results

def plot_residuals_distribution(all_results):
    """Plot residuals distribution for all throws"""
    # Combine all residuals
    all_residuals_x = np.concatenate([r['residuals_x'] for r in all_results])
    all_residuals_y = np.concatenate([r['residuals_y'] for r in all_results])
    
    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # X residuals
    ax1.hist(all_residuals_x, bins='auto', density=True, alpha=0.7)
    mu_x, std_x = norm.fit(all_residuals_x)
    x = np.linspace(min(all_residuals_x), max(all_residuals_x), 100)
    ax1.plot(x, norm.pdf(x, mu_x, std_x), 'r-', lw=2)
    ax1.set_title('Residuals Distribution (x)')
    
    # Y residuals
    ax2.hist(all_residuals_y, bins='auto', density=True, alpha=0.7)
    mu_y, std_y = norm.fit(all_residuals_y)
    y = np.linspace(min(all_residuals_y), max(all_residuals_y), 100)
    ax2.plot(y, norm.pdf(y, mu_y, std_y), 'r-', lw=2)
    ax2.set_title('Residuals Distribution (y)')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'x': {'mean': mu_x, 'std': std_x},
        'y': {'mean': mu_y, 'std': std_y}
    }

# Main analysis
if __name__ == "__main__":
    # Load data
    t_blue, x_blue, y_blue = np.loadtxt('blue.txt', unpack=True)
    t_yellow, x_yellow, y_yellow = np.loadtxt('yellow.txt', unpack=True)
    t_red, x_red, y_red = np.loadtxt('red.txt', unpack=True)
    
    # Analyze each ball
    results_blue = analyze_ball(t_blue, x_blue, y_blue, "Blue")
    results_yellow = analyze_ball(t_yellow, x_yellow, y_yellow, "Yellow")
    results_red = analyze_ball(t_red, x_red, y_red, "Red")
    
    # Combine all results
    all_results = results_blue + results_yellow + results_red
    
    # Convert to DataFrame
    df_results = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in all_results])
    
    print("\nSummary Statistics:")
    print(df_results.describe())
    
    residuals_stats = plot_residuals_distribution(all_results)
    df_results.to_csv('juggling_3_balls_results.csv', index=False)

