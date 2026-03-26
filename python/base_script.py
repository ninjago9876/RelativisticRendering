# ----------------------- SYMBOLIC RESULT -----------------------
# Inverse Metric (g^-1): Matrix([[x_r/(rs - x_r), 0, 0, 0], [0, (-rs + x_r)/x_r, 0, 0], [0, 0, x_r**(-2), 0], [0, 0, 0, 1/(x_r**2*sin(x_theta)**2)]])
# H(x, p): p_phi**2/(2*x_r**2*sin(x_theta)**2) + p_r**2*(-rs + x_r)/(2*x_r) + p_t**2*x_r/(2*(rs - x_r)) + p_theta**2/(2*x_r**2)
# Tetrad: Matrix([[1/sqrt(-rs/x_r + 1), 0, 0, 0], [0, 1/sqrt(1/(-rs/x_r + 1)), 0, 0], [0, 0, 1/Abs(x_r), 0], [0, 0, 0, 1/(Abs(x_r)*Abs(sin(x_theta)))]])
#
# --- ODE ---
# ODE's derived from the Hamiltonian of the geodesic:
# dx[0]/dλ = p_t*x_r/(rs - x_r);
# dx[1]/dλ = p_r*(-rs + x_r)/x_r;
# dx[2]/dλ = p_theta/pow(x_r, 2);
# dx[3]/dλ = p_phi/(pow(x_r, 2)*pow(sin(x_theta), 2));
#
# dp[0]/dλ = 0;
# dp[1]/dλ = pow(p_phi, 2)/(pow(x_r, 3)*pow(sin(x_theta), 2)) - 1.0/2.0*pow(p_r, 2)/x_r + (1.0/2.0)*pow(p_r, 2)*(-rs + x_r)/pow(x_r, 2) - 1.0/2.0*pow(p_t, 2)*x_r/pow(rs - x_r, 2) - 1.0/2.0*pow(p_t, 2)/(rs - x_r) + pow(p_theta, 2)/pow(x_r, 3);
# dp[2]/dλ = pow(p_phi, 2)*cos(x_theta)/(pow(x_r, 2)*pow(sin(x_theta), 3));
# dp[3]/dλ = 0;

import numpy as np
import matplotlib.pyplot as plt
from wcwidth import width


# ------------------------- Utility ----------------------
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# ------------------------- Math -------------------------
# Function to calculate the tetrad at some x with some rs
def tetrad(x, rs: float):
    x_t = x[0]
    x_r = x[1]
    x_theta = x[2]
    x_phi = x[3]

    f = 1 - rs / x_r

    eps = 1e-8
    sin_x_theta = np.sin(x_theta)
    sin_x_theta = np.sign(sin_x_theta) * max(abs(sin_x_theta), eps)

    e = np.zeros((4, 4))
    #   [ 1/sqrt(-rs/x_r + 1), 0,                       0,          0                              ]
    #   [ 0,                   1/sqrt(1/(-rs/x_r + 1)), 0,          0                              ]
    #   [ 0,                   0,                       1/Abs(x_r), 0                              ]
    #   [ 0,                   0,                       0,          1/(Abs(x_r)*Abs(sin(x_theta))) ]
    e[0, 0] = 1 / np.sqrt(f)
    e[1, 1] = np.sqrt(f)
    e[2, 2] = 1 / x_r
    e[3, 3] = 1 / (x_r * sin_x_theta)

    return e


# Metric
def metric(x, rs):
    # [
    #   [rs/x_r - 1, 0, 0, 0],
    #   [0, 1/(-rs/x_r + 1), 0, 0],
    #   [0, 0, x_r**2, 0],
    #   [0, 0, 0, x_r**2*sin(x_theta)**2]
    # ]
    x_t = x[0]
    x_r = x[1]
    x_theta = x[2]
    x_phi = x[3]

    eps = 1e-8
    sin_x_theta = np.sin(x_theta)
    sin_x_theta = np.sign(sin_x_theta) * max(abs(sin_x_theta), eps)

    x_r = max(x_r, rs + 1e-6)  # Clamp near the horizon
    f = 1 - rs / x_r

    g = np.zeros((4, 4))
    g[0, 0] = -f
    g[1, 1] = 1 / f
    g[2, 2] = x_r ** 2
    g[3, 3] = x_r ** 2 * sin_x_theta ** 2

    return g


# Inverse Metric
def inv_metric(x, rs):
    x_t = x[0]
    x_r = x[1]
    x_theta = x[2]
    x_phi = x[3]

    eps = 1e-8
    sin_x_theta = np.sin(x_theta)
    sin_x_theta = np.sign(sin_x_theta) * max(abs(sin_x_theta), eps)

    g_inv = np.array([
        [x_r / (rs - x_r), 0, 0, 0],
        [0, (-rs + x_r) / x_r, 0, 0],
        [0, 0, x_r ** (-2), 0],
        [0, 0, 0, 1 / (x_r ** 2 * sin_x_theta ** 2)]
    ])
    return g_inv


# Function to calculate the dx/dλ and dp/dλ in the form tuple(dx, dp) for some state x and p and constant rs
def ode(x, p, rs):
    x_t = x[0]
    x_r = x[1]
    x_theta = x[2]
    x_phi = x[3]

    p_t = p[0]
    p_r = p[1]
    p_theta = p[2]
    p_phi = p[3]

    eps = 1e-8
    sin_x_theta = np.sin(x_theta)
    sin_x_theta = np.sign(sin_x_theta) * max(abs(sin_x_theta), eps)

    dx = np.zeros(4)
    # dx[0]/dλ = p_t*x_r/(rs - x_r);
    dx[0] = p_t * x_r / (rs - x_r)
    # dx[1]/dλ = p_r*(-rs + x_r)/x_r;
    dx[1] = p_r * (-rs + x_r) / x_r
    # dx[2]/dλ = p_theta/pow(x_r, 2);
    dx[2] = p_theta / (x_r ** 2)
    # dx[3]/dλ = p_phi/(pow(x_r, 2)*pow(sin(x_theta), 2));
    dx[3] = p_phi / (x_r ** 2 * sin_x_theta ** 2)

    dp = np.zeros(4)
    # dp[0]/dλ = 0;
    dp[0] = 0

    # dp[1]/dλ =
    #         pow(p_phi, 2) / (pow(x_r, 3)*pow(sin(x_theta), 2))
    dp[1] = p_phi ** 2 / (x_r ** 3 * sin_x_theta ** 2)
    #       - 1.0/2.0 * pow(p_r, 2) / x_r
    dp[1] -= 0.5 * p_r ** 2 / x_r
    #       + (1.0/2.0) * pow(p_r, 2) * (-rs + x_r) / pow(x_r, 2)
    dp[1] += 0.5 * p_r ** 2 * (-rs + x_r) / x_r ** 2
    #       - 1.0/2.0 * pow(p_t, 2) * x_r / pow(rs - x_r, 2)
    dp[1] -= 0.5 * p_t ** 2 * x_r / (rs - x_r) ** 2
    #       - 1.0/2.0 * pow(p_t, 2) / (rs - x_r)
    dp[1] -= 0.5 * p_t ** 2 / (rs - x_r)
    #       + pow(p_theta, 2) / pow(x_r, 3);
    dp[1] += p_theta ** 2 / x_r ** 3

    # dp[2]/dλ = pow(p_phi, 2)*cos(x_theta)/(pow(x_r, 2)*pow(sin(x_theta), 3));
    dp[2] = p_phi ** 2 * np.cos(x_theta) / (x_r ** 2 * sin_x_theta ** 3)
    # dp[3]/dλ = 0;
    dp[3] = 0

    return dx, dp


# Convert Minkowski Cartesian coordinates into Minkowski Spherical coordinates
def minkowski_cartesian_to_minkowski_spherical(x: np.ndarray):
    x_t = x[0]
    x_x = x[1]
    x_y = x[2]
    x_z = x[3]

    sx = np.zeros(4)
    sx[0] = x_t

    # r
    r = np.sqrt(x_x ** 2 + x_y ** 2 + x_z ** 2)
    sx[1] = r

    # theta (polar angle, from +z axis)
    if r == 0:
        sx[2] = 0.0  # convention
    else:
        sx[2] = np.arccos(x_z / r)

    # phi (azimuthal angle in xy-plane)
    sx[3] = np.arctan2(x_y, x_x)

    return sx


def initial_schwarzschild_condition_from_cartesian(x, dir, rs: float):
    x_schwarzschild = minkowski_cartesian_to_minkowski_spherical(x)
    x_r = x_schwarzschild[1]
    x_theta = x_schwarzschild[2]
    x_phi = x_schwarzschild[3]

    dir = dir / np.linalg.norm(dir)

    e_r = np.array((
        np.sin(x_theta) * np.cos(x_phi),
        np.sin(x_theta) * np.sin(x_phi),
        np.cos(x_theta)
    ))

    e_theta = np.array((
        np.cos(x_theta) * np.cos(x_phi),
        np.cos(x_theta) * np.sin(x_phi),
        -np.sin(x_theta)
    ))

    e_phi = np.array((
        -np.sin(x_phi),
        np.cos(x_phi),
        0
    ))

    spatial = np.array([
        np.dot(dir, e_r),
        np.dot(dir, e_theta),
        np.dot(dir, e_phi)
    ])

    p_local = np.array((1.0, *spatial))

    e = tetrad(x_schwarzschild, rs)
    g = metric(x_schwarzschild, rs)

    p_contra = e @ p_local
    p_cov = g @ p_contra

    null_val = check_null(x_schwarzschild, p_cov, rs)
    if abs(null_val) > 1e-6:
        print("Warning: null condition violated:", null_val)

    return x_schwarzschild, p_cov


def check_null(x, p, rs):
    r = x[1]
    theta = x[2]

    sin_theta = np.sin(theta)
    eps = 1e-8
    sin_theta = np.sign(sin_theta) * max(abs(sin_theta), eps)

    g_inv = np.array([
        [r / (rs - r), 0, 0, 0],
        [0, (r - rs) / r, 0, 0],
        [0, 0, 1 / r ** 2, 0],
        [0, 0, 0, 1 / (r ** 2 * sin_theta ** 2)]
    ])

    return p @ g_inv @ p


def p_cov_to_static_dir(p_cov, x, rs):
    g_inv = inv_metric(x, rs)
    e = tetrad(x, rs)

    p_contra = g_inv @ p_cov
    p_local = np.linalg.inv(e) @ p_contra
    d = p_local[1:] / np.linalg.norm(p_local[1:])

    return d


# ------------------------- Integrator -------------------------
def rk4_integrate_ray(x, dir, rs, iterations):
    x, p = initial_schwarzschild_condition_from_cartesian(x, dir, rs)
    trajectory = [(x, p)]
    for i in range(iterations):
        h = 0.01  # Step Size

        # RK4 Step
        k1 = ode(x, p, rs)  # (dx, dp)
        k2 = ode(x + k1[0] * h / 2, p + k1[1] * h / 2, rs)
        k3 = ode(x + k2[0] * h / 2, p + k2[1] * h / 2, rs)
        k4 = ode(x + h * k3[0], p + h * k3[1], rs)

        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        p += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

        if x[1] < rs + 0.05:
            print("hit")
            return trajectory

        trajectory.append((x.copy(), p.copy()))
    return trajectory


# ------------------------- Plot -------------------------
def plot_trajectory_2d(trajectory, ax):
    r_vals = []
    phi_vals = []

    for t in trajectory:
        if len(t) != 2:
            continue
        x = t[0]
        r_vals.append(x[1])
        phi_vals.append(x[3])

    r_vals = np.array(r_vals)
    phi_vals = np.unwrap(np.array(phi_vals))

    # Convert to Cartesian
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)

    # Detect jumps in r or phi to break the line
    max_jump = 5.0  # tweak this depending on your scale
    segments = [[0]]
    for i in range(1, len(x_vals)):
        if np.abs(x_vals[i] - x_vals[i - 1]) > max_jump or np.abs(y_vals[i] - y_vals[i - 1]) > max_jump:
            segments.append([i])
        else:
            segments[-1].append(i)

    # Plot each segment separately
    for seg in segments:
        ax.plot(x_vals[seg], y_vals[seg], color="blue")


def plot_trajectory_3d(trajectory, ax):
    x_vals = []
    y_vals = []
    z_vals = []

    for t in trajectory:
        if len(t) != 2:
            continue

        x = t[0]
        r = x[1]
        theta = x[2]
        phi = x[3]

        # Spherical → Cartesian
        x_vals.append(r * np.sin(theta) * np.cos(phi))
        y_vals.append(r * np.sin(theta) * np.sin(phi))
        z_vals.append(r * np.cos(theta))

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    # Detect jumps (same idea as before)
    max_jump = 5.0
    segments = [[0]]

    for i in range(1, len(x_vals)):
        if (
                abs(x_vals[i] - x_vals[i - 1]) > max_jump or
                abs(y_vals[i] - y_vals[i - 1]) > max_jump or
                abs(z_vals[i] - z_vals[i - 1]) > max_jump
        ):
            segments.append([i])
        else:
            segments[-1].append(i)

    # Plot each segment
    for seg in segments:
        ax.plot(x_vals[seg], y_vals[seg], z_vals[seg], color="blue")

    ax.set_box_aspect([1, 1, 1])  # equal scaling


# ------------------------- Runnable -------------------------
def initial_condition_test():
    x = np.array([0, 100, 98, 0])  # (t, x, y, z)
    dir = np.array([-1, -1, 0])  # toward black hole

    x_s, p = initial_schwarzschild_condition_from_cartesian(x, dir, rs=2)
    null = check_null(x_s, p, 2)
    print(f"Null Geodesic: Test {"passed" if null < 1e-6 else "failed"} with value {null}")


def integrate_single_ray():
    rs = 1.
    x = [0, 3, 3, 0]
    dir = [0, -1, 0]
    trajectory = rk4_integrate_ray(np.array(x), np.array(dir), rs, 1000)

    plt.figure()
    ax = plt.subplot(111)
    plot_trajectory_2d(trajectory, ax)

    ax.add_patch(plt.Circle((0, 0), rs))
    ax.set_aspect("equal")


def integrate_array_of_rays():
    rs = 1.
    X = [[0, x, 3, 0] for x in np.linspace(2.5, 10, 10)]
    dir = [0, -1, 0]

    plt.figure()
    ax = plt.subplot(111)

    for x in X:
        trajectory = rk4_integrate_ray(np.array(x), np.array(dir), rs, 1000)
        plot_trajectory_2d(trajectory, ax)

    ax.add_patch(plt.Circle((0, 0), rs))
    ax.set_aspect("equal")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")


def integrate_grid_of_rays():
    rs = 1.
    X = []
    for x in np.linspace(-2, 2, 4):
        for y in np.linspace(-2, 2, 4):
            X.append([0, x, y, 5])
    dir = [0, 0, -1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='3d')

    for x in X:
        trajectory = rk4_integrate_ray(np.array(x), np.array(dir), rs, 1000)
        plot_trajectory_3d(trajectory, ax)

    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color="red")

    ax.set(xlim=[-10, 10], ylim=[-10, 10], zlim=[-10, 10])


def render_image():
    resolution = (20, 20)

    rs = 1.
    X = []
    D = []
    for x in np.linspace(-4, 4, resolution[0]):
        for y in np.linspace(-4, 4, resolution[1]):
            X.append([0, x, y, 5])
            D.append(normalize(np.array([0, 0, -1])))

    grid = []

    for x, d in zip(X, D):
        trajectory = rk4_integrate_ray(np.array(x), np.array(d), rs, 1000)
        x0, p0 = trajectory[0]
        static_d = p_cov_to_static_dir(p0, x0, rs)
        grid.append(static_d)

    grid = np.array(grid)
    grid = np.reshape(grid, (*resolution, 3))

    img = np.array(grid)

    # Create figure and axis
    fig, ax = plt.subplots()

    # Show the image on the axis
    ax.imshow(img)

    # Set ticks
    ax.set_xticks(range(len(grid[0])))
    ax.set_yticks(range(len(grid)))

    # Add grid lines between squares
    ax.set_xticks(np.arange(-.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(grid), 1), minor=True)
    ax.grid(which='minor', linestyle='-', linewidth=1)

    # Hide minor tick marks (just keep grid lines)
    ax.tick_params(which='minor', size=0)

    plt.show()


if __name__ == "__main__":
    plt.rcParams['axes3d.mouserotationstyle'] = 'azel'

    initial_condition_test()
    # integrate_grid_of_rays()
    render_image()

    plt.show()
