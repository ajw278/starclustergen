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
import build_cluster as bc
import shutil

import matplotlib.gridspec as gridspec

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(scriptdir+'/general')
sys.path.append(scriptdir+'/plot')

from common import *


import nbody6_interface as nbi 
import cluster_plot as cp
import run_rebound_sim as rb

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


def plummer_density(r, M, a):
    """
    Plummer density profile
    """
    return 3 * M / (4 * np.pi * a**3) * (1 + (r**2 / a**2))**(-5/2)

def plummer_Mfracenc(r, a):
    return r*r*r/(r*r + a*a)**1.5

def plummer_potential(r, M, a, G=Gcgs):
    """
    Gravitational potential for Plummer sphere
    """
    rmag=r
    if len(r.shape)>1:
        rmag = np.linalg.norm(r,axis=0)
    return -1 * M * G/ np.sqrt(rmag**2 + a**2)

def plummer_sigmav(r,M, a, G=Gcgs):
    return np.sqrt(G*M/6./np.sqrt(r*r+a*a))

def sigmav_wgas(r, Mst, ast, Mg, ag, G=Gcgs):
    potst = plummer_potential(r, Mst, ast, G=G)
    potg = plummer_potential(r, Mg, ag, G=G)
   
    return np.sqrt(-potst/3./2. - potg/3.) #np.sqrt((-potst-potg)/3./2.)



def define_gasjumps(minit, lossrate, njumps=10, tdelay=0.0, tend=1.0, ascale=1.0):
    tscale = minit/np.absolute(lossrate)
    tvals = np.linspace(tdelay, min(tscale+tdelay,tend), njumps)
    if tscale>tend:
        raise Warning('final time smaller than last gas mass jump')
    else:
        tvals = np.append(tvals, tend)

    gparams = np.zeros((len(tvals), 4))

    tlast =0.0
    for it in range(len(tvals)):
        MNEW = minit+lossrate*(tvals[it]-tdelay)

        if MNEW>0.0:
            dt = tvals[it]-tlast
            gparams[it] = np.array([MNEW, ascale, 0.0, dt])
            tlast += dt
        else:
            dt = tvals[-1]-tlast
            gparams[it] = np.array([0.0, ascale, 0.0, dt])
            gparams = gparams[:it+1]
            break

    return gparams


def generate_positions(N, a):
    """
    Generate positions from Plummer density profile using inverse transform sampling
    """
    # Generate random numbers from a uniform distribution
    u = np.random.uniform(0, 1, N)

    rspace = np.logspace(np.log10(a/1e4), np.log10(1e4*a), int(1e5))
    cdf = plummer_Mfracenc(rspace, a)
    inverse_cdf = interpolate.interp1d(cdf, rspace)
    radii = inverse_cdf(u)


    # Generate random angles
    theta = np.random.uniform(0, np.pi, size=N)
    phi = np.random.uniform(0, 2 * np.pi, size=N)

    # Convert spherical coordinates to Cartesian coordinates
    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)

    return np.vstack((x, y, z))

def generate_velocities(N, radii, M, a, Mg0=0.0, ag0=None, G=Gcgs):
    """
    Generate velocities using the derived velocity dispersion
    """
    rmag = np.linalg.norm(radii, axis=0)
    # Calculate velocity dispersion using the relation 1/2 sigma^2 = potential
    if Mg0==0.0:
        sigma = plummer_sigmav(rmag, M, a, G=G)
    else:
        print('Generating velocities with gas...')
        if ag0 is None:
            ag0 = a
        sigma = sigmav_wgas(rmag, M, a, Mg0, ag0, G=G)
    
    sigma = sigma[np.newaxis,:]*np.ones((3, N))

    # Generate random velocities assuming isotropic distribution
    velocities = np.random.normal(loc=0, scale=sigma, size=(3, N))

    return velocities 

if __name__=='__main__':
    


    tend = 100.0

    # Parameters
    a = 5.0    # Scale radius
    M = 1e5    # Total mass

    texp = 2.0
    trem=20.0
    alphas = np.array([0.1, 0.25, 0.5])
    betas = np.array([0.1, 0.25, 0.5])
    num_rows = len(alphas)
    num_cols = len(betas)

    figsize = (12, 12)  # Adjust the figure size as needed

    # Create the figure and gridspec
    fig1 = plt.figure(figsize=figsize)
    fig2 = plt.figure(figsize=figsize)
    fig3 = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(num_rows, num_cols, figure=fig1, hspace=0, wspace=0)
    gs2 = gridspec.GridSpec(num_rows, num_cols, figure=fig2, hspace=0, wspace=0)
    gs3 = gridspec.GridSpec(num_rows, num_cols, figure=fig2, hspace=0, wspace=0)

    homdir = os.getcwd()
    for irow, alpha in enumerate(alphas):
        for icol, beta in enumerate(betas):
            print('Current working directory:', os.getcwd())
            dname = 'alpha_%.2lf_beta_%.2lf'%(alpha, beta)
            if not os.path.isdir(dname):
                print('No directory %s found, creating...'%dname)
                os.makedirs(dname)
            os.chdir(dname)
            if os.path.isfile(homdir+'/nbody_path.txt'):
                print('Found nbody_path, copying to current directory')
                shutil.copy(homdir+'/nbody_path.txt', './nbody_path.txt')
            else:
                print('No nbody_path found, assuming default is correct...')


            if not os.path.isfile('sim_ics_r.npy') or not os.path.isfile('sim_ics_v.npy') or not os.path.isfile('sim_ics_m.npy') or not os.path.isfile('sim_gparams.npy'):
                
                Mgas = alpha*M
                agas = beta*a

                #gparams = define_gasjumps(Mgas, -Mgas/trem, njumps=1, tdelay=1.0, tend=tend, ascale=1.0)
                if texp==0.0:
                	gparams = np.array([Mgas, agas,0.0, trem])
                else:
                	gparams = np.array([Mgas, agas,Mgas/texp, trem])
                	
                mmin= 0.08

                imf_func = bc.get_kroupa_imf(mmin=mmin)
                msp = np.logspace(-2, 2, 1000)
                mmass = np.trapz(msp*imf_func(msp), msp)/np.trapz(imf_func(msp), msp)
                
                mmass = 1.
                N = int(M/mmass)

                # Generate positions
                rs = generate_positions(N, a)

                # Generate velocities
                vs = generate_velocities(N, rs*pc2cm, M*Msol2g, a*pc2cm, Mg0=Mgas*Msol2g, ag0=agas*pc2cm, G=Gcgs)

                ms = np.ones(N) #assign_masses(rs, mmin=mmin)
                
                vs /= km2cm

                np.save('sim_ics_r', rs)
                np.save('sim_ics_v', vs)
                np.save('sim_ics_m', ms)
                np.save('sim_gparams', gparams)
            
            else:
                rs = np.load('sim_ics_r.npy')
                vs = np.load('sim_ics_v.npy')
                ms = np.load('sim_ics_m.npy')
                gparams=  np.load('sim_gparams.npy')
            
            nbins0= 0
            

            sim = nbi.nbody6_cluster(rs.T, vs.T, ms,  outname='clustersim', dtsnap_Myr =10.0, \
                        tend_Myr = tend, gasparams=gparams, etai=0.005, etar=0.005, etau=0.01, dtmin_Myr=1e-8, \
                        rmin_pc=1e-5,dtjacc_Myr=1.0, load=True, ctype='smooth', force_incomp = False, \
                        rtrunc=50.0, nbin0=nbins0, aclose_au=200.0)
            sim.store_arrays(reread=True)
            sim.evolve(reread=True, suppress_restart=False)
            #exit()
            txt = f'$\\alpha = {alpha}$, $\\beta = {beta}$'
            if os.path.isdir('pre_exp'):
            	shutil.rmtree('pre_exp')
            if os.path.isdir('post_exp'):
            	shutil.rmtree('post_exp')

            if irow==0 and icol==0:
                ax1 = fig1.add_subplot(gs1[irow, icol])
                ax2 = fig2.add_subplot(gs2[irow, icol])
                ax3 = fig3.add_subplot(gs3[irow, icol])
            else:
                ax1 = fig1.add_subplot(gs1[irow, icol], sharex=ax1, sharey=ax1)
                ax2 = fig2.add_subplot(gs2[irow, icol], sharex=ax2, sharey=ax2)
                ax3 = fig3.add_subplot(gs3[irow, icol], sharex=ax3, sharey=ax3)
                
            cp.plot_radii(sim, agas=gparams[1], Mgas=gparams[0], axtext=txt, axhmr=ax1, axrif=ax2, axErf=ax3)

            # Remove tick labels except for the left and bottom-most panels
            # Adjust tick labels for the left-most column and bottom-most row
            if icol == 0:
                ax1.tick_params(axis='y', which='both', labelleft=True)
                ax2.tick_params(axis='y', which='both', labelleft=True)
                ax3.tick_params(axis='y', which='both', labelleft=True)
            else:
                ax1.tick_params(axis='y', which='both', labelleft=False)
                ax2.tick_params(axis='y', which='both', labelleft=False)
                ax3.tick_params(axis='y', which='both', labelleft=False)
                ax1.set_ylabel('')
                ax2.set_ylabel('')
                ax3.set_ylabel('')
                
            if irow == num_rows - 1:
                ax1.tick_params(axis='x', which='both', labelbottom=True)
                ax2.tick_params(axis='x', which='both', labelbottom=True)
                ax3.tick_params(axis='x', which='both', labelbottom=True)
            else:
                ax1.tick_params(axis='x', which='both', labelbottom=False)
                ax2.tick_params(axis='x', which='both', labelbottom=False)
                ax3.tick_params(axis='x', which='both', labelbottom=False)
                ax1.set_xlabel('')
                ax2.set_xlabel('')
                ax3.set_xlabel('')
           
            os.chdir(homdir)
            del sim

            #plt.show()

    fig1.tight_layout(pad=0)
    fig2.tight_layout(pad=0)
    fig3.tight_layout(pad=0)
    fig1.savefig('hmr_all.pdf', bbox_inches='tight', format='pdf')
    fig2.savefig('rinit_v_rfinal.pdf', bbox_inches='tight', format='pdf')
    fig3.savefig('Einit_v_rfinal.pdf', bbox_inches='tight', format='pdf')
    plt.show()
