import gen_binary_rebound as gbr
import gaussian_field as gf
import scipy.interpolate as interpolate
import numpy as np
import matplotlib.pyplot as plt
import imf
import os
import math
import gen_vel as vg
import gen_vel_gf as vgf
import sys


scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(scriptdir+'/general')
sys.path.append(scriptdir+'/plot')

from common import *


import nbody6_interface as nbi 
import cluster_plot as cp
import run_rebound_sim as rb

from powerbox import get_power

from scipy.spatial.distance import cdist


distance = 140.0
deg2rad = np.pi/180.0

mas2rad = 4.848e-9


day2s = 60.*60.*24.0
year2s = day2s*365.
pc2cm = 3.086e18
Gcgs = 6.6743e-8
Msol2g = 1.988e33
km2cm = 1e5
km2pc = 3.241e-14
au2cm = 1.496e13

# Given parameters
alpha = 0.018
delta_logP = 0.7


sv0 = 10.**0.30530517 
p = 0.72082734
r0 = 1.0


r0 = deg2rad*distance
sv0 *= mas2rad*distance*pc2cm/year2s

no_hard_edge=True
minlogP = 5.0
maxlogP = 10.0

# Define the binary fraction functions
def f_logP_lt_1(M1):
    return 0.020 + 0.04 * np.log10(M1) + 0.07 * np.log10(M1)**2

def f_logP_2_7(M1):
    return 0.039 + 0.07 * np.log10(M1) + 0.01 * np.log10(M1)**2

def f_logP_5_5(M1):
    return 0.078 - 0.05 * np.log10(M1) + 0.04 * np.log10(M1)**2

def binary_fraction(logP,  M1):
    if logP<minlogP:
        return 0.0
    elif logP>maxlogP:
        return 0.0
    elif 0.2 <= logP < 1.0:
        return f_logP_lt_1(M1)
    elif 1.0 <= logP < 2.7 - delta_logP:
        return f_logP_lt_1(M1) + (logP - 1) / (1.7 - delta_logP) * (f_logP_2_7(M1) - f_logP_lt_1(M1) - alpha * delta_logP)
    elif 2.7 - delta_logP <= logP < 2.7 + delta_logP:
        return f_logP_2_7(M1) + alpha * (logP - 2.7)
    elif 2.7 + delta_logP <= logP < 5.5:
        return f_logP_2_7(M1) + alpha * delta_logP + (logP - 2.7 - delta_logP) / (2.8 - delta_logP) * (
                f_logP_5_5(M1) - f_logP_2_7(M1) - alpha * delta_logP)
    elif 5.5 <= logP < maxlogP: # or (5.5 <= logP and no_hard_edge):
        return f_logP_5_5(M1)  #* np.exp(-0.3 * (logP - 5.5))
    else:
        return 0.0
    
def get_q_func_Msol(gamma1=0.4, gamma2=-0.7):
    qsp = np.linspace(0.,1., 100)
    psp = qsp**gamma1
    psp[qsp>0.3] = (qsp[qsp>0.3]**gamma2)*(0.3**gamma1)/(0.3**gamma2)
    norm = np.trapz(psp, qsp)
    psp /= norm
    return interpolate.interp1d(qsp, psp, bounds_error=False, fill_value=0.0)

def get_gammavals(logP):
    return 0.4, -0.7

def get_flogP_func(M1):
    logPsp = np.linspace(1.0, 10., 100)
    flogP = np.asarray([binary_fraction(logP, M1=M1) for logP in logPsp])
    return interpolate.interp1d(logPsp, flogP, bounds_error=False, fill_value=0.0)

def make_icdf(func, xmin=0.0, xmax=8.):
    xsp = np.linspace(xmin, xmax, 1000)
    fv = func(xsp)
    fv /= np.trapz(fv, xsp)
    cdf = np.cumsum(fv)
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return interpolate.interp1d(cdf,xsp)


"""# Generate data for plotting
logP_values = np.linspace(0.2, 8.0, 500)
q_value = 0.5  # Replace with the desired value for mass ratio q

# Plotting for a specific value of M1
binary_fractions_M1 = [binary_fraction(logP, M1=1.0) for logP in logP_values]

plt.plot(logP_values, binary_fractions_M1, label=f'q > 0.3, M1 = 20')
plt.xlabel('log P (days)')
plt.ylabel('Binary Fraction')
plt.title('Binary Fraction as a function of log P (for M1=20)')
plt.legend()
plt.grid(True)
plt.show()"""

def get_kroupa_imf(m1=0.08, p1=0.3, m2=0.5, p2=1.3, m3=1.0, p3=2.3, mmin=0.08):
    
    msp = np.logspace(-2, 2., 1000)
    
    xi  = msp**-p1
    f1 = (m1**-p1)/(m2**-p2)
    xi[msp>m1] = f1*(msp[msp>m1]**-p2)
    f2 = f1 * (m2**-p2)/(m3**-p3)
    xi[msp>m2] = f2*(msp[msp>m2]**-p3)
    xi[msp<mmin] = 0.0
    
    xi /= np.trapz(xi, msp)
    
    return interpolate.interp1d(msp, xi)


def get_imf_icdf():
    
    msp = np.logspace(-2, 2, 1000)
    
    imf_func= get_kroupa_imf()
    
    imf = imf_func(msp)
    cdf = np.cumsum(imf)
    cdf /=cdf[-1]
    
    return  interpolate.interp1d(cdf, msp)


#Assign masses to stellar positions
def assign_masses(rstars):
    u = np.random.uniform(size=rstars.shape[-1])
    icdf =  get_imf_icdf()
    mstars = icdf(u)
    return mstars

def generate_binary_population(mstars):
    num_stars = len(mstars)

    # Arrays to store results
    binary_flags = np.zeros(num_stars, dtype=bool)
    logP_companions = np.full(num_stars, -1.0)
    q_companions = np.full(num_stars, -1.0)
    e_companions = np.full(num_stars, -1.0)

    flogPsp = np.linspace(0., 9., 150)
    for i in range(num_stars):
        
        flogP_func = get_flogP_func(mstars[i])
        pbinary = np.trapz(flogP_func(flogPsp), flogPsp)
        u = np.random.uniform()
        if u<pbinary:
            u2 = np.random.uniform()
            icdf_logP = make_icdf(flogP_func, xmin=1.0, xmax=8.)
            logP_i = icdf_logP(u2)
            
            g1, g2 = get_gammavals(logP_i)
            
            qfunc = get_q_func_Msol(gamma1=g1, gamma2=g2)
            icdf_q = make_icdf(qfunc, xmin=0.1, xmax=1.)
            u3 = np.random.uniform()
            q_i = icdf_q(u3)
            
            #Eccentricity randomly distributed.. appears to be! 
            e_companions[i] = 0.6 # np.random.uniform()*

            # Determine if a binary is present based on the probability
            binary_flags[i] = 1
            logP_companions[i] = logP_i
            q_companions[i] = q_i
        
    

    return binary_flags, logP_companions, q_companions, e_companions

def generate_binary_pv(ms, bf, logP, q, e):
    iss = bf==1
    ms_s = ms[iss]
    logP_s = logP[iss]
    q_s = q[iss]
    e_s = e[iss]
    
    P_s = 10.**logP_s
    P_s *= day2s
    ms_s *= Msol2g
    a_s = np.power(Gcgs*ms_s*(1.+q_s)*(P_s/2./np.pi)**2 , 1./3.)
    xb_s, vb_s = gbr.gen_binpop(ms_s, q_s, a_s, e_s, centre_primary=True)
    xb = np.zeros((len(ms), 3))
    vb = np.zeros((len(ms), 3))
    
    xb[iss] = xb_s
    vb[iss] = vb_s
    
    return xb, vb

def add_binaries(rs, ms, bf, xb, vb, q, vs=None):
    nbins = int(np.sum(bf))
    ndim = rs.shape[0]
    rs_b = np.zeros((nbins, ndim))
    ms_b = np.zeros(nbins)
    ibs = np.where(bf==1)[0]
    inbs = np.where(bf==0)[0]
    
    
    rs_all = []
    vs_all = []
    ms_all = []
    
    for i, ib in enumerate(ibs):
        rs_all.append(rs.T[ib])
        rs_all.append(rs.T[ib] + xb[ib][:ndim])
        ms_all.append(ms[ib])
        ms_all.append(ms[ib]*q[ib])
        vs_all.append(vs.T[ib])
        vs_all.append(vs.T[ib] + vb[ib][:ndim])
    
    for i, iss in enumerate(inbs):
        rs_all.append(rs.T[iss])
        ms_all.append(ms[iss])
        vs_all.append(vs.T[iss])
    
    rs_all = np.asarray(rs_all).T
    vs_all = np.asarray(vs_all).T
    ms_all = np.asarray(ms_all)
    
    return rs_all, vs_all, ms_all

def plot_pairs(rstars_phys):
    rstars = rstars_phys/(distance*deg2rad)
    
    plt.scatter(rstars[0], rstars[1], s=8, marker='+')
    plt.scatter(rstars[0][:500], rstars[1][:500], s=8, marker='o', color='r')
    plt.show()
    
    Rpc = np.array([rstars_phys[0], rstars_phys[1]])
    p_k_samples, bins_samples = get_power(Rpc.T,50.0,N=200, b=1.0)
    p_k_samples3D, bins_samples3D = get_power(rstars_phys.T,50.0,N=200, b=1.0)
   

    plt.plot(bins_samples, p_k_samples, marker='o', label="Model 2D PS")
    plt.plot(bins_samples3D, p_k_samples3D, marker='s', label='Model 3D PS')
    dbins, dPk = np.load('Taurus_Pk.npy')[:]

    plt.plot(dbins, dPk, marker='s', label='Taurus 2D')
    plt.plot(bins_samples, 1000.*bins_samples**(-5./3),label="Input")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('corrfuncs.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    Ndim = rstars.shape[0]

    print(rstars.shape)
    rstar_comp = rstars[:2].T
    print(rstar_comp.shape)
    dr = cdist(rstar_comp, rstar_comp, 'euclidean')
    dr = dr[np.triu_indices(len(rstars[0]), k=1)]
    
    binsr = np.logspace(-4.0, 2, 25)
    if Ndim==2:
        weights = len(dr)/(2.*np.pi*dr)
    else:
        weights = len(dr)/(4.*np.pi*dr*dr)
    
    fig, ax = plt.subplots(figsize=(5.,4.))
    hist, bin_edges = np.histogram(dr, bins=binsr, density=True, weights=weights)
    dbin = np.diff(bin_edges)
    bin_cent = (bin_edges[:-1]+bin_edges[1:])/2.
    hist /= np.trapz(hist*2.*np.pi*bin_cent, bin_cent)
    hist *= len(dr)
    # Create a histogram
    dsp = np.logspace(-3.5, 2., 100)
    plt.plot(bin_cent, hist, marker='o', color='k', linewidth=1, label='Model')
    
    dbins, dhist = np.load('Taurus_fpairs.npy')[:]
    
    plt.plot(dbins, dhist, marker='s', color='r', linewidth=1, label='Observed')
    
    plt.xlabel('Angular separation: $\Delta \\theta$ [deg]')
    plt.ylabel('Pair surface density: $\Sigma_*$ [pc$^{-2}$]')
    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim([1e-7, 1e3])
    plt.xlim([1e-4, 30.0])
    plt.grid(True)
    ax.tick_params(which='both', right=True, bottom=True, left=True, top=True, direction='out')
    # Display the histogram
    plt.show()


def calc_pl(L1, L2, xi1, xi2):
    pl_xi = np.log(xi2) - np.log(xi1)
    pl_xi /= np.log(L2)-np.log(L1)
    
    pl_Pk = 3. - pl_xi
    return pl_xi, pl_Pk
    
    
def plot_stars_with_velocity(rs, vs, cdim, title=''):
    """
    Plot the positions of stars in two dimensions not colored, with the third dimension representing velocity.

    Parameters:
    - rs: 2D numpy array, shape (ndim, nstars), positions of stars.
    - vs: 2D numpy array, shape (ndim, nstars), velocities of stars.
    - cdim: int, the dimension amlosslong which to color the stars based on velocity.
    - title: str, title for the plot.

    Returns:
    None (displays the plot).
    """
    # Determine the dimensions not colored
    other_dimensions = [dim for dim in range(rs.shape[0]) if dim != cdim]

    # Extract positions in the two dimensions not colored
    positions_x = rs[other_dimensions[0], :]
    positions_y = rs[other_dimensions[1], :]

    # Extract velocities in the specified dimension
    velocities = vs[cdim, :]

    # Create a scatter plot
    plt.scatter(positions_x, positions_y, c=velocities/1e5, cmap='viridis', marker='o', alpha=0.8)

    # Add labels and title
    plt.xlabel(f"Position in Dimension {other_dimensions[0] + 1}")
    plt.ylabel(f"Position in Dimension {other_dimensions[1] + 1}")
    plt.title(title)

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Velocity [km/s]')

    # Show the plot
    plt.show()


def plot_dvNN(rs, vs):
    positions = rs.T
    velocities = vs.T
    # Calculate distances between all pairs of stars
    distances = cdist(positions, positions)
    vdistances = cdist(velocities, velocities)

    # Set the diagonal elements to a large value (to exclude a star being its own nearest neighbor)
    np.fill_diagonal(distances, np.inf)
    
    dist1ct = distances[np.triu_indices(len(distances), k=1)]
    vdist1ct = vdistances[np.triu_indices(len(distances), k=1)]
    
    irand = np.random.choice(np.arange(len(dist1ct)), size=1000, replace=False)
    """plt.figure(figsize=(8, 6))
    plt.scatter(dist1ct[irand], vdist1ct[irand], c='blue', alpha=0.7, edgecolors='none')

    plt.title('Magnitude of Velocity Difference to Neighbours')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()"""

    # Find the index of the nearest neighbor for each star
    nearest_neighbors = np.argmin(distances, axis=1)
    num_stars = len(positions)
    nearest_neighbor_distances = distances[np.arange(num_stars), nearest_neighbors]
    
    weights=1./4./np.pi/nearest_neighbor_distances**2

    weights_d=1./4./np.pi/distances**2

    bins = np.logspace(-4.0, 1.5)
    plt.hist(distances.flatten(), bins=bins, density=True, histtype='step', weights=weights_d.flatten())
    plt.hist( nearest_neighbor_distances, bins=bins,density=True, histtype='step', weights=weights)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    # Calculate the magnitude of the difference in velocity between each star and its nearest neighbor
    velocity_differences = np.linalg.norm(velocities - velocities[nearest_neighbors], axis=1)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(nearest_neighbor_distances, velocity_differences, c='blue', alpha=0.7, edgecolors='none')

    plt.title('Magnitude of Velocity Difference to Nearest Neighbor')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

    
def select_istars(rstars, rmax, sharpness=10.0):
    istars  = np.arange(rstars.shape[1])
    
    rmag = np.linalg.norm(rstars, axis=0)
    
    pinc = np.exp(-rmag*rmag/2./rmax/rmax)**sharpness
    u = np.random.uniform(size=len(rmag))
    iinc = u<pinc
    
    ist = istars[iinc]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates for each dimension
    x = rstars[0, ist]
    y = rstars[1, ist]
    z = rstars[2, ist]
    
    xi= rstars[0, :]
    yi = rstars[1, :]
    zi = rstars[2, :]

    ax.scatter(xi, yi, zi, c='r', marker='o', s=1)
    #ax.scatter(xi, yi, zi, c='b', marker='o', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
    
    return istars[iinc]
    
    
    
if __name__=='__main__':
    
    if not os.path.isfile('sim_ics_r.npy') or not os.path.isfile('sim_ics_v.npy') or not os.path.isfile('sim_ics_m.npy'):
        binsep = distance*deg2rad*(10.**-1.5)/1.5
        Lbox  = 60.0
        lNbox_est = math.log2(Lbox/binsep)
        Nbox = int(2.**(int(lNbox_est)))

        print(Nbox)
        ndim=3
        seed=7483
        #seed =586
        vcalc=True
        
        """rsp = np.logspace(-2, 2.0)
        plt.plot(rsp, sv0*(rsp/r0)**p/1e5)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()"""

        
        if ndim==2:
            #Parameters that work for 2D
            Pk_norm = 20.0
            Pk_index = -1.3
            covmat = np.eye(2)
            mu = np.zeros(2)
        else:
            #Parameters that work for 3D
            Pk_norm = 300.0
            Pk_index= -1.66
            covmat = np.eye(3)
            mu = np.zeros(3)

        print('Nbox:', Nbox)
        if not os.path.isfile('rgbox.npy'):
            rs = gf.build_cluster(Nstars=10000, Nbox=Nbox,  Lbox=Lbox, Rcl = 30.0, \
                     sharp_edge= 10.0, Pk_norm=Pk_norm, Pk_index=Pk_index, normed_covmat=covmat, mu=mu, seed=seed)
            np.save('rgbox', rs)
        else:
            rs = np.load('rgbox.npy')

        istars = np.arange(rs.shape[1]) 
        istars = select_istars(rs, 30.0, sharpness=10.0)
        istars = np.random.choice(istars, size=700, replace=False)
        print(rs.shape)
        rs = rs[:, istars]

        print('Vcalc walk...')
        vs = vgf.velocity_gen(rs, r0=r0, p=p, sv0=sv0)

        
        #plot_pairs(rs)
        ms = assign_masses(rs)
        bf, logP, q, e = generate_binary_population(ms)
        xb, vb = generate_binary_pv(ms, bf, logP, q, e)

        rs_all, vs_all, ms_all = add_binaries(rs,  ms, bf, xb/pc2cm, vb, q, vs=vs)

        np.save('rstars_wbin.npy', rs_all)
        
        vs_all /= km2cm
        
        
        plot_dvNN(rs_all, vs_all)
        np.save('sim_ics_bins', np.array([bf, logP, q, e]))
        np.save('sim_ics_r', rs_all)
        np.save('sim_ics_v', vs_all)
        np.save('sim_ics_m', ms_all)
    
    else:
        rs_all = np.load('sim_ics_r.npy')
        vs_all = np.load('sim_ics_v.npy')
        ms_all = np.load('sim_ics_m.npy')
        bf, logP, q, e = np.load('sim_ics_bins.npy')
    
   

    """irand = np.random.choice(np.arange(len(rs_all.T)), size=len(rs_all.T), replace=False)
    rs_all[0]  =rs_all[0, irand]
    irand = np.random.choice(np.arange(len(rs_all.T)), size=len(rs_all.T), replace=False)
    rs_all[1]  =rs_all[1, irand]
    irand = np.random.choice(np.arange(len(rs_all.T)), size=len(rs_all.T), replace=False)
    rs_all[2]  =rs_all[2, irand]"""
    nbins0= 0
    
    print(np.median(np.absolute(rs_all)),np.median(np.absolute(vs_all)), np.median(ms_all))
    print(rs_all.shape, vs_all.shape)
    nbins0 = int(np.sum(bf))

    sim = nbi.nbody6_cluster(rs_all.T, vs_all.T, ms_all,  outname='clustersim', dtsnap_Myr =0.01, \
                tend_Myr = 3.0, gasparams=None, etai=0.005, etar=0.005, etau=0.01, dtmin_Myr=1e-8, \
                rmin_pc=1e-4,dtjacc_Myr=0.05, load=True, ctype='smooth', force_incomp = False, \
                    rtrunc=50.0, nbin0=nbins0, aclose_au=200.0)
    sim.evolve()

    #cp.encounter_analysis(sim)
    cp.plot_3dpos(sim)
    #sim = rb.setupSimulation(rs_all, vs_all*1e5*1e6*year2s/pc2cm, ms_all, units=('Myr', 'pc', 'Msun'))
    #sim.integrate(3.0)
