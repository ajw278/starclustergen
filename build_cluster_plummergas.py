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
    


    tend = 50.0

    # Parameters
    a = 1.0    # Scale radius
    M = 1e4    # Total mass

    alphas = np.array([0.25, 0.5, 1.0 , 2.0])
    betas = np.array([0.25, 0.5, 1.0 , 2.0])

    homdir = os.getcwd()
    for alpha in alphas:
        for beta in betas:
            os.chdir(homdir)
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

                trem=0.3

                #gparams = define_gasjumps(Mgas, -Mgas/trem, njumps=1, tdelay=1.0, tend=tend, ascale=1.0)
                gparams = np.array([Mgas, agas,0.0, trem])
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
            

            sim = nbi.nbody6_cluster(rs.T, vs.T, ms,  outname='clustersim', dtsnap_Myr =0.2, \
                        tend_Myr = tend, gasparams=gparams, etai=0.005, etar=0.005, etau=0.01, dtmin_Myr=1e-8, \
                        rmin_pc=1e-5,dtjacc_Myr=1.0, load=True, ctype='smooth', force_incomp = False, \
                        rtrunc=50.0, nbin0=nbins0, aclose_au=200.0)
            #sim.store_arrays(reread=True)

            sim.evolve(reread=True)


            cp.plot_radii(sim)
