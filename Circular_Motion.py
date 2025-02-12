import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from scipy import stats

def load_data(filename):
    """
    Load and prepare data from the Tracker file.
    
    Args:
        filename (str): Path to the data file
        
    Returns:
        tuple: (time, x coordinates, y coordinates)
    """
    t, x, y = np.loadtxt(filename, unpack=True)
    x = 100 * x  # converting to cm
    y = 100 * y
    return t, x, y

def fit_circle(x, y):
    """
    Fit a circle to the data points.
    
    Args:
        x, y (np.array): Coordinates of points
        
    Returns:
        dict: Circle parameters, residuals and parameter uncertainties
    """
    def circle_residuals(params, x, y):
        """Calculate residuals between points and circle."""
        x0, y0, R = params
        return np.sqrt((x - x0)**2 + (y - y0)**2) - R
    
    # Initial estimates
    x_mean, y_mean = np.mean(x), np.mean(y)
    R_guess = np.mean(np.sqrt((x - x_mean)**2 + (y - y_mean)**2))
    p0 = [x_mean, y_mean, R_guess]
    
    # Fitting
    res = least_squares(circle_residuals, p0, args=(x, y), jac='3-point')
    x0, y0, R = res.x
    
    # Calculate parameter uncertainties
    residuals = res.fun 
    
    # Estimated variance of the residuals
    s_sq = np.sum(residuals**2) / (len(residuals) - len(p0))
    
    # Covariance matrix
    pcov = s_sq * np.linalg.inv(res.jac.T.dot(res.jac))
    
    # Standard errors
    perr = np.sqrt(np.diag(pcov))
    
    return {
        'x0': x0,
        'y0': y0,
        'R': R,
        'x0_err': perr[0],
        'y0_err': perr[1],
        'R_err': perr[2],
        'residuals': residuals,
        'pcov': pcov
    }

def fit_ellipse(x, y):
    """
    Fit an ellipse to the data points.
    
    Args:
        x, y (np.array): Coordinates of points
        
    Returns:
        dict: Ellipse parameters and residuals
    """
    def ellipse_residuals(params, x, y):
        """Calculate residuals between points and ellipse."""
        x0, y0, a, b, theta = params
        xt = x - x0
        yt = y - y0
        xr = xt*np.cos(theta) + yt*np.sin(theta)
        yr = -xt*np.sin(theta) + yt*np.cos(theta)
        return ((xr/a)**2 + (yr/b)**2) - 1
    
    # Initial estimates
    x_mean, y_mean = np.mean(x), np.mean(y)
    R_guess = np.mean(np.sqrt((x - x_mean)**2 + (y - y_mean)**2))
    p0 = [x_mean, y_mean, R_guess, R_guess*0.98, 0]
    
    # Fitting
    res = least_squares(ellipse_residuals, p0, args=(x, y), jac='3-point')
    x0, y0, a, b, theta = res.x

    # Calculate parameter uncertainties
    residuals = res.fun

    # Estimated variance of the residuals
    s_sq = np.sum(residuals**2) / (len(residuals) - len(p0))

    # Covariance matrix
    pcov = s_sq * np.linalg.inv(res.jac.T.dot(res.jac))

    # Standard errors
    perr = np.sqrt(np.diag(pcov))
    
    return {
        'x0': x0,
        'y0': y0,
        'a': a,
        'b': b,
        'tilt': theta,
        'x0_err': perr[0],
        'y0_err': perr[1],
        'a_err': perr[2],
        'b_err': perr[3],
        'tilt_err': perr[4],
        'residuals': residuals,
        'pcov': pcov
    }

def main():
    t, x, y = load_data('Circular_Motion.txt')

    r_tracker = np.sqrt(x**2 + y**2)
    print(f'Average radius (Tracker): {np.mean(r_tracker):.3f} cm')
    print(f'Standard deviation of the radius (Tracker): {np.std(r_tracker):.3f} cm')
    print(f'Total variation: {np.max(r_tracker) - np.min(r_tracker):.2f} cm')
 
    # Perform circle fit
    circle_params = fit_circle(x, y)
    
    # Print results
    print("\nCircle fit results:")
    print(f"Center: ({circle_params['x0']:.3f} ± {circle_params['x0_err']:.3f}, {circle_params['y0']:.3f} ± {circle_params['y0_err']:.3f}) cm")
    print(f"Radius: {circle_params['R']:.3f} ± {circle_params['R_err']:.3f} cm")
    print(f"Residuals standard deviation: {np.std(circle_params['residuals'], ddof=1):.3f} cm")

   # Perform elipse fit
    ellipse_params = fit_ellipse(x, y)

    # Print results
    print("\nEllipse fit results:")
    print(f"Center: ({ellipse_params['x0']:.3f} ± {ellipse_params['x0_err']:.3f}, {ellipse_params['y0']:.3f} ± {ellipse_params['y0_err']:.3f}) cm")
    print(f"a: {ellipse_params['a']:.3f} ± {ellipse_params['a_err']:.3f} cm")
    print(f"b: {ellipse_params['b']:.3f} ± {ellipse_params['b_err']:.3f} cm")
    print(f"Orientation angle: {np.degrees(ellipse_params['tilt']):.1f} ± {np.degrees(ellipse_params['tilt_err']):.1f} degrees.")
    print(f"Residuals standard deviation: {np.std(ellipse_params['residuals'], ddof=1):.4f} cm")

if __name__ == "__main__":
    main()
