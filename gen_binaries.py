import numpy as np

def keplerian_motion(mass1, mass2, semi_major_axis, eccentricity, phase, inclination, omega):
    """
    Calculate initial positions and velocities of binary stars using Kepler's laws.

    Parameters:
    - mass1: Mass of star 1 (solar masses)
    - mass2: Mass of star 2 (solar masses)
    - semi_major_axis: Semi-major axis of the orbit (astronomical units)
    - eccentricity: Eccentricity of the orbit
    - phase: Orbital phase (in radians)
    - inclination: Orbital inclination (in radians)

    Returns:
    - initial_positions_star1: Initial position of star 1 relative to the barycenter (astronomical units)
    - initial_positions_star2: Initial position of star 2 relative to the barycenter (astronomical units)
    - initial_velocities_star1: Initial velocity of star 1 relative to the barycenter (km/s)
    - initial_velocities_star2: Initial velocity of star 2 relative to the barycenter (km/s)
    """

    # Gravitational constant in solar masses, astronomical units, and days
    G = 4 * np.pi**2 / 365.25**2
    
    mu = G*(mass1+mass2)
    specific_energy = -mu/2./semi_major_axis
    specific_angular_momentum = mu*semi_major_axis*np.sqrt(1.-eccentricity**2)

    # Calculate orbital parameters using Kepler's laws
    mean_motion = np.sqrt(G * (mass1 + mass2) / semi_major_axis**3)
    true_anomaly = phase
    
    semi_latus_rectum = semi_major_axis*(1.-eccentricity*eccentricity)
    
    initial_separation = semi_latus_rectum/(1.+eccentricity*np.cos(true_anomaly))
    
    initial_radial_velocity = np.sqrt(mu/semi_latus_rectum)*eccentricity*np.sin(true_anomaly)
    initial_tangential_velocity = np.sqrt(mu/semi_latus_rectum)*(1.+eccentricity*np.cos(true_anomaly))
    
    
    
    #eccentric_anomaly = np.arccos((1.-initial_separation/semi_major_axis)/eccentricity)
    #semi_minor_axis = semi_major_axis*np.sqrt(1.-eccentricity**2)
    
    
    angular_momentum = mean_motion * semi_major_axis**2 * np.sqrt(1 - eccentricity**2)
    
    # Mass ratio
    mass_ratio = mass2 / (mass1 + mass2)

    # Initial positions and velocities of each star relative to the barycenter
    initial_positions_star1 = initial_separation * np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0])*mass2 / (mass1 + mass2)
    initial_positions_star2 = initial_separation * np.array([np.cos(true_anomaly + np.pi), np.sin(true_anomaly + np.pi), 0])*mass1 / (mass1 + mass2)

    initial_velocities_star1 = mass_ratio * angular_momentum / np.sqrt(1 - eccentricity**2) * np.array([-np.sin(true_anomaly), np.cos(true_anomaly), 0])
    initial_velocities_star2 = -(1 - mass_ratio) * angular_momentum / np.sqrt(1 - eccentricity**2) * np.array([-np.sin(true_anomaly + np.pi), np.cos(true_anomaly + np.pi), 0])

    
    # Calculate the velocity of the barycenter
    velocity_barycenter = (mass1 * initial_velocities_star1 + mass2 * initial_velocities_star2) / (mass1 + mass2)

    # Subtract the velocity of the barycenter from the individual velocities
    initial_velocities_star1 -= velocity_barycenter
    initial_velocities_star2 -= velocity_barycenter

    return (
        initial_positions_star1, initial_positions_star2,
        initial_velocities_star1, initial_velocities_star2
    )



# Example usage
mass1 = 2.0  # Mass of star 1 in solar masses
mass2 = 1.5  # Mass of star 2 in solar masses
semi_major_axis = 1.0  # Semi-major axis in astronomical units
eccentricity = 0.2  # Eccentricity of the orbit
phase = np.pi / 4  # Orbital phase in radians
inclination = np.pi / 6  # Orbital inclination in radians
omega = np.pi/9

position_star1, position_star2, velocities_star1, velocities_star2 = keplerian_motion(mass1, mass2, semi_major_axis, eccentricity, phase, inclination, omega)

# Print or use the calculated velocities
print("Velocities of Star 1:", velocities_star1, "km/s")
print("Velocities of Star 2:", velocities_star2, "km/s")

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def orbital_equations(t, y, mass1, mass2, semi_major_axis, eccentricity, inclination):
    """
    Equations for the motion of a binary system in 3D space.

    Parameters:
    - t: Time
    - y: State vector [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
    - mass1: Mass of star 1 (solar masses)
    - mass2: Mass of star 2 (solar masses)
    - semi_major_axis: Semi-major axis of the orbit (astronomical units)
    - eccentricity: Eccentricity of the orbit
    - inclination: Orbital inclination (in radians)

    Returns:
    - dydt: Derivative of the state vector
    """
    G = 4 * np.pi**2 / 365.25**2

    # Unpack the state vector
    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2 = y

    # Relative position vector
    r = np.array([x2 - x1, y2 - y1, z2 - z1])

    # Distance between stars
    distance = np.linalg.norm(r)

    # Gravitational force
    force = G * mass1 * mass2 / distance**3 * r

    # Equations of motion
    dydt = [vx1, vy1, vz1, force[0] / mass1, force[1] / mass1, force[2] / mass1,
            vx2, vy2, vz2, -force[0] / mass2, -force[1] / mass2, -force[2] / mass2]

    return dydt

def integrate_orbit(mass1, mass2, semi_major_axis, eccentricity, inclination, t_span, num_points):
    # Initial conditions
    initial_state = [semi_major_axis * (1 + eccentricity), 0, 0, 0, np.sqrt(G * (mass1 + mass2) / semi_major_axis), 0,
                     -semi_major_axis * (1 + eccentricity), 0, 0, 0, -np.sqrt(G * (mass1 + mass2) / semi_major_axis), 0]

    # Time span for integration
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    # Perform the numerical integration
    result = solve_ivp(
        orbital_equations,
        t_span=t_span,
        y0=initial_state,
        args=(mass1, mass2, semi_major_axis, eccentricity, inclination),
        t_eval=t_eval,
        method='Radau',
        rtol=1e-10, atol=1e-20
    )

    return result


def integrate_and_plot_orbit(mass1, mass2, semi_major_axis, eccentricity, phase, inclination,omaga,  t_span, num_points):
    # Calculate orbital velocities
    positions_star1, positions_star2, velocities_star1, velocities_star2 = keplerian_motion(mass1, mass2, semi_major_axis, eccentricity, phase, inclination, omega)

    # Initial conditions
    initial_state = [
        positions_star1[0], positions_star1[1], positions_star1[2], velocities_star1[0], velocities_star1[1], velocities_star1[2],
        positions_star2[0], positions_star2[1], positions_star2[2], velocities_star2[0], velocities_star2[1], velocities_star2[2]
    ]

    # Time span for integration
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    # Perform the numerical integration
    result = solve_ivp(
        orbital_equations,
        t_span=t_span,
        y0=initial_state,
        args=(mass1, mass2, semi_major_axis, eccentricity, inclination),
        t_eval=t_eval,
        method='Radau',
        atol=1e-10, rtol=1e-12
    )

    # Extracting position vectors
    x1, y1, z1, _, _, _, x2, y2, z2, _, _, _ = result.y

    # Plotting the motion of the stars
    plt.figure(figsize=(8, 8))
    plt.plot(x1, y1, label='Star 1', color='blue')
    plt.plot(x2, y2, label='Star 2', color='red')
    plt.plot(x1, z1, label='Star 1 z', color='blue', linestyle='dotted')
    plt.plot(x2, z2, label='Star 2 z', color='red', linestyle='dotted')
    plt.scatter([0], [0], color='black', marker='o', label='Barycenter')
    plt.title('Orbit of Binary Stars')
    plt.xlabel('X-axis (AU)')
    plt.ylabel('Y-axis (AU)')
    plt.xlim([-semi_major_axis, semi_major_axis])
    plt.ylim([-semi_major_axis, semi_major_axis])
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
mass1 = 2.0  # Mass of star 1 in solar masses
mass2 = 0.5  # Mass of star 2 in solar masses
semi_major_axis = 1.0  # Semi-major axis in astronomical units
eccentricity = 0.9  # Eccentricity of the orbit
phase = np.pi/2.  # Orbital phase in radians
inclination = 0.0 # Orbital inclination in radians
omega = np.pi/2.
t_span = (0, 200)  # Time span for integration (days)
num_points = 10000  # Number of points for integration

# Integrate and plot the orbit
integrate_and_plot_orbit(mass1, mass2, semi_major_axis, eccentricity, phase, inclination, omega, t_span, num_points)
