import rebound
import matplotlib.pyplot as plt
import numpy as np

# Function to set up a binary system with given mass ratio and orbital elements
def setup_binary(m1, m2, semimajor_axis, eccentricity, inclination, argument_of_periapsis, longitude_of_ascending_node, true_anomaly):
    sim = rebound.Simulation()
    sim.G = 6.6743e-8
    # Add primary and secondary bodies
    sim.add(m=m1)
    sim.add(m=m2, a=semimajor_axis, e=eccentricity, inc=inclination, omega=argument_of_periapsis, Omega=longitude_of_ascending_node, f=true_anomaly)

    return sim

# Function to extract initial Cartesian positions and velocities
def get_initial_conditions(simulation):
    particles = simulation.particles  # List of particles in the simulation

    # Extracting initial conditions for the primary (index 0) and secondary (index 1)
    p1 = np.asarray(particles[0].xyz)
    v1 = np.asarray(particles[0].vxyz)

    p2 = np.asarray(particles[1].xyz)
    v2 = np.asarray(particles[1].vxyz)

    return p1, p2, v1, v2

def get_binary_xv(m1, q, a, e, i, omega, Omega, f, centre_primary=False):
    
    m2 = q*m1  # mass of secondary
    sim = setup_binary(m1, m2, a, e, i, omega, Omega, f)
    x1, x2, v1, v2 = get_initial_conditions(sim)
    vbary = (v1*m1 + v2*m2)/(m1+m2)
    xbary = (x1*m1 + x2*m2)/(m1+m2)
    
    if not centre_primary:
        x1 -= xbary
        x2 -= xbary
        v1 -= vbary
        v2 -= vbary

        return x1, x2, v1, v2
    else:
        x2 -= x1
        x1 -= x1
        v2 -= v1
        v1 -= v1
        return x2, v2

def gen_binpop(m1_s, q_s, a_s, e_s, centre_primary=False, plot=False):
    cosi = np.random.uniform(size=len(m1_s))
    i_s = np.arccos(cosi)
    f_s = np.random.uniform(-np.pi, np.pi,size=len(m1_s))
    o_s = np.random.uniform(-np.pi, np.pi,size=len(m1_s))
    O_s = np.random.uniform(-np.pi, np.pi,size=len(m1_s))
    
    if not centre_primary:
        xv_pairs = np.zeros((len(m1_s), 4, 3))
    else:
        xv_pairs = np.zeros((len(m1_s), 2, 3))
    
    
    for i, m1 in enumerate(m1_s):
        xv_pairs[i] = get_binary_xv(m1, q_s[i], a_s[i], e_s[i], i_s[i], o_s[i], O_s[i], f_s[i], centre_primary=centre_primary)
    
    if plot:
        print(xv_pairs.shape)
        x1 = xv_pairs[:,0]
        x2 = xv_pairs[:,1]
        print(x1.shape)
        plt.scatter(x1[:,0], x1[:,1])
        plt.scatter(x2[:,0], x2[:,1])
        plt.show()

        plt.hist(np.linalg.norm(x2[:,:2], axis=1), bins=100, density=True)
        plt.show()

    if  centre_primary:  
        return xv_pairs[:,0, :], xv_pairs[:,1, :] 
    else:
        return None

"""N = 10000
marr  = np.ones(N)
qarr = np.ones(N)*0.5
aarr = np.ones(N)*5.0
earr = np.ones(N)*0.01
gen_binpop(marr, qarr, aarr, earr)"""

# Function to plot the orbit of the binary system
def plot_orbit(simulation):
    rebound.OrbitPlot(simulation, color=True, unitlabel="[AU]")
    plt.show()

    
if __name__=='__main__':
    # Example usage
    mass_ratio = 0.2
    semimajor_axis = 1.0
    eccentricity = 0.1
    inclination = 0.2
    argument_of_periapsis = 0.3
    longitude_of_ascending_node = 0.4

    simulation = setup_binary(1.0, mass_ratio, semimajor_axis, eccentricity, inclination, argument_of_periapsis, longitude_of_ascending_node)

    # Run the simulation
    sim_duration = 100  # in simulation time units
    simulation.integrate(sim_duration)

    # Plot the orbit
    plot_orbit(simulation)
