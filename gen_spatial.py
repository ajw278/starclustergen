import numpy as np

import matplotlib.pyplot as plt
import math
from astroML.correlation import two_point
from astroML.correlation import bootstrap_two_point

plt.rc('text', usetex=True)

ND = 3
#15 pc length scale (root) 
LSCALE=15.0
LSCALE2= LSCALE*LSCALE

NSTARS = 500

#Power-law index for correlation function
def_p =1.5
def_dmin = 0.1
def_dmax = LSCALE
def_d0 = 1.0


if ND==3:
    #Define the covariance matrix and mean for the initial points
    def_covmat = np.array([[LSCALE2, 0.0, 0.0],[0.0, LSCALE2, 0.0], [0.0, 0.0, LSCALE2]])
    def_mu = np.array([0.0,0.,0.])
else:
    def_covmat = np.array([[LSCALE2, 0.0],[0.0, LSCALE2]])
    def_mu = np.array([0.0,0.])


#Functional form of the 2-point correlation we try to reproduce
def twopoint_corr(d, dmin, dmax, p=def_p, d0=1.0):
    return ((d+dmin)**-p)*np.exp(-d/dmax)

#Distance calculation between a position and a grid of positions (will update for Cov)
def dist_metric_gr(r, rgr, density_func=None):
    ndim =rgr.shape[0]
    dvec = r[:, *(np.newaxis,)*ndim]-rgr
    dmag = np.linalg.norm(dvec, axis=0)
    #ctf = plt.contourf(rgr[0], rgr[1], dmag, cmap='viridis')
    return dmag

#Distance calc. between star and other stars
def dist_metric_st(r1, r2, density_func=None):
    dvec = r1[:, np.newaxis]-r2
    dmag = np.linalg.norm(dvec, axis=0)
    #ctf = plt.contourf(rgr[0], rgr[1], dmag, cmap='viridis')
    return dmag

#Considering weighting the seed stars, but ignoring now...
def calc_probseeds(rstars, dmin=0.005, dmax=10.0, p=def_p, d0=1.0):
    #Initiate probabaility for each star
    Nst = rstars.shape[1]
    probs = np.zeros(Nst)
    """for ist in range(Nst):
        d_ = dist_metric_st(rstars[:,ist], rstars, density_func=None)
        p_ = twopoint_corr(d_, dmin, dmax, p=p, d0=1.0)
        probs[ist]+=np.sum(p_)
        plt.scatter(d_, probs)
    plt.show()"""
    probs=np.ones(Nst)
    return probs/np.sum(probs)

def calc_probpos(rpts, rgr, istars, dmin=0.01, dmax=5.0, p=2., d0=1.0):
    Nstars = rpts.shape[1]
    ptot = np.zeros(rgr.shape[1:])
    for istar in istars:
        d = dist_metric_gr(rpts[:,istar], rgr)
        p2point = twopoint_corr(d,dmin, dmax, p=p, d0=d0)
        ptot+=p2point
    return ptot/np.sum(ptot)

#Return the indices of the NN nearest neighbours
def find_nneighbours( rall, istar, NN=5):
    dr = dist_metric_st(rall[:,istar], rall)
    isort = np.argsort(dr)
    return isort[1:NN+1]


#Draw stars from a pdf that approximates the correlation function
def draw_corrfunc(rpts, istar, dmin=0.05, Ndraw=1, dmax=5.0, p=2., d0=1.0, covmat=None):
    
    #Create an array of possible distances to select from
    dsp = np.logspace(np.log10(dmin)-2.0, np.log10(dmax), 80)
    
    #Offset by a small random factor so that we're not always choosing the same distances
    dfact = np.median(dsp[1:]/dsp[:-1])
    dsp *= np.random.uniform(dfact)
    
    
    pdr = twopoint_corr(dsp, dmin, dmax, p=def_p, d0=1.0)
    
    istar = int(istar)
    Ndim = rpts.shape[0]
    
    #generate a grid
    dphi=0.1
    phi = np.arange(0., 2.*np.pi, dphi)
    #phi = np.fmod(phi, 2.*np.pi)
    #phi = np.sort(np.unique(phi))
    if Ndim==3:
        dth = 0.1
        theta = np.arange(0., np.pi+dth, dth)
        rg, pg, tg = np.meshgrid(dsp,phi, theta, indexing='ij')
        dr = np.gradient(dsp)[:, np.newaxis, np.newaxis]
        dV = np.absolute(rg*rg*np.sin(tg)*dr*dth*dphi)
        
        #Multiply probability in r by volume
        pgr = np.ones(rg.shape)*pdr[:, np.newaxis, np.newaxis]*dV #rg*dr*np.absolute(np.sin(tg)) #/2./np.pi/rg/rg
        
        #Grid offsets
        pg +=0.5*dphi*(1.-2.*np.random.uniform())
        pg = np.fmod(pg, 2.*np.pi)
        
        tg += .5*dth*(1.-2.*np.random.uniform())
        tg = np.fmod(pg, np.pi)
        
        rgcart = np.array([rg*np.sin(tg)*np.cos(pg), rg*np.sin(tg)*np.sin(pg), rg*np.cos(tg)])
        
        
    else:
        rg, pg = np.meshgrid(dsp,phi, indexing='ij')
        dr = np.gradient(dsp)[:, np.newaxis]
        dV = rg*dr*dphi
        
        #Multiply probability in r by volume in r
        pgr = np.ones(rg.shape)*pdr[:, np.newaxis]*dV #/2./np.pi/rg
        
        #Offset from grid
        pg +=0.5*dphi*(1.-2.*np.random.uniform())
        pg = np.fmod(pg, 2.*np.pi)
        
        rgcart = np.array([rg*np.cos(pg), rg*np.sin(pg)])
    
    
    rpts_st = rpts[:,istar]
    rgcart = rgcart + rpts_st[:, *(np.newaxis,)*Ndim]
  
  
    #We need to take into account the other neighbouring stars that change the probability 
    ineigh = find_nneighbours(rpts, istar, NN=10)
    for iN in ineigh:
        drN =  dist_metric_gr(rpts[:, iN], rgcart)
        pN = twopoint_corr(drN+dmin, dmin, dmax, p=def_p, d0=1.0)
        pgr+=pN
    

    pgr/=np.sum(pgr)

    idsp = weighted_random_choice(pgr, size=Ndraw)
    
    
 
    dr_new = np.array([rgcart[idim][idsp] for idim in range(Ndim)])
    
    #Used to offset from a grid here, but better to do it earlier
    dd = 0.0 #0.5*(np.absolute(dV[idsp])**(1./float(Ndim)))*(1.-2.*np.random.uniform(size=dr_new.shape))
    dr_new = dr_new #+dd 
    
    #plt.scatter(rrand[0], rrand[1])
    #plt.show()
    
    rdraw = dr_new
    """
    if Ndim==3:
        print(rgcart.shape, pgr.shape)
        print(rgcart[0,:,:,int(rgcart.shape[-1]/2)])
        ctf = plt.contourf(rgcart[0,:,:,int(rgcart.shape[-1]/2)], rgcart[1,:,:,int(rgcart.shape[-1]/2)], np.log10(pgr[:,:,int(rgcart.shape[-1]/2)]), cmap='viridis',levels=30)
        plt.scatter(rgcart[0,:,:,int(rgcart.shape[-1]/2)], rgcart[1,:,:,int(rgcart.shape[-1]/2)], c='k', s=1, alpha=0.2)
    else:
        ctf = plt.contourf(rgcart[0], rgcart[1], np.log10(pgr/dV), cmap='viridis',levels =30)
        plt.scatter(rgcart[0], rgcart[1], c='k', s=1, alpha=0.2)
        
    
    plt.scatter(rpts[0, ineigh], rpts[1, ineigh], c='r', s=2)
    plt.scatter(rpts[0, istar], rpts[1, istar], c='cyan', s= 2)
    plt.scatter(rdraw[0], rdraw[1], c='pink', marker='*', s=10)
    print(rdraw, rdraw.shape)
    plt.colorbar(ctf) 
    #plt.xlim([rpts[0,istar]-1.0, rpts[0,istar]+1.0])
    #plt.ylim([rpts[1,istar]-1.0, rpts[1,istar]+1.0])
    plt.show()
    
    if Ndim==3:
        
        ctf = plt.contourf(rgcart[0,:,int(rgcart.shape[2]/2),:], rgcart[2,:,int(rgcart.shape[2]/2),:], np.log10(pgr[:,int(rgcart.shape[1]/2),:]/dV), cmap='viridis', levels=30)
        ctf = plt.contourf(rgcart[0,:,0,:], rgcart[2,:,int(rgcart.shape[2]/2),:], np.log10(pgr[:,0,:]/dV), cmap='viridis', levels=30)

        
        plt.scatter(rgcart[0,:,int(rgcart.shape[2]/2),:], rgcart[2,:,int(rgcart.shape[2]/2),:], c='k', s=1)
        plt.scatter(rgcart[0,:,0,:], rgcart[2,:,0,:], c='r', s=1)
        
        plt.scatter(rpts[0, ineigh], rpts[2, ineigh], c='r', s=2)
        plt.scatter(rpts[0, istar], rpts[2, istar], c='cyan', s= 2)
        plt.scatter(rdraw[0], rdraw[2], c='blue', marker='*', s=10)
        print(rdraw, rdraw.shape)
        plt.colorbar(ctf) 
        #plt.xlim([rpts[0,istar]-1.0, rpts[0,istar]+1.0])
        #plt.ylim([rpts[1,istar]-1.0, rpts[1,istar]+1.0])
        plt.show()
    """
    
    return rdraw

#Initialise meshgrid    
def initialise_grid(x, y, z=None):
    if z is None:
        return np.array(np.meshgrid(x, y, indexing='ij'))  
    else:
        return np.array(np.meshgrid(x, y, z, indexing='ij'))


#Multi-dimensional Gaussian pdf values with convariance
def pmulti_gauss(rgr, mu, invcov):
    ndim = rgr.shape[0]
    
    # Calculate the difference between rgr and mu for all elements
    dx = rgr - mu[:, *(np.newaxis,)*ndim]  # Broadcast 'mu' to match the shape of 'rgr'

    # Calculate the quadratic form (dusq) element-wise
    dx_sig = np.einsum('ij...,jk...->ik...', invcov, dx)
    dusq = np.einsum('ij...,ij...->j...', dx_sig, dx)
  
    # Calculate the prefactor
    pref = (np.pi / 2.)**(-ndim / 2.) * np.sqrt(np.linalg.det(invcov))

    # Calculate the probability values for all elements in rgr
    pm = pref * np.exp(-0.5 * dusq)

    return pm


#Random choice with weights in array -- return the indices
def weighted_random_choice(arr, size=10):
    # Flatten the N-D array and calculate the weights
    flattened_arr = arr.flatten()
    weights = flattened_arr / np.sum(flattened_arr)
    
    # Choose a random index based on the calculated weights
    index = np.random.choice(len(flattened_arr), p=weights, size=size, replace=True)
    
    # Convert the 1D index back to N-D indices
    indices = np.unravel_index(index, arr.shape)
    
    
    return indices

#Add a random perturbation of size ~dr to a value
def random_dr(rpts, dr):
    return rpts+ (0.5-np.random.uniform(size=rpts.shape))*dr

def initialise_gausspos(Nnests, covmat = def_covmat,mu=def_mu, rmaxfact= 4.0):
    ndim = mu.shape[0]
    rmax = rmaxfact*(np.linalg.det(covmat)**(1./2./ndim))
    x = np.linspace(-rmax, rmax, 200)
    y = np.linspace(-rmax, rmax, 201)
    dr = x[1]-x[0]
    z=None
    if ndim>2:
        z = np.linspace(-rmax, rmax, 202)
    
    rgr = initialise_grid(x, y, z=z)
    invcov = np.linalg.inv(covmat)
    
    #Define a n-d gaussian probability distribution
    pdist = pmulti_gauss(rgr, mu, invcov)
    
    
    #Choose coordinate indices
    icoords = weighted_random_choice(pdist, size=Nnests)
    
    #Take the spatial coordinates of selected indices
    rpts = np.array([rs[icoords] for rs in rgr])
    
    """plt.scatter(rpts[0], rpts[1], color='cyan', s=5)
    ctf = plt.contourf(x, y, pdist[:, :, int(len(z)/2)].T, levels=10, cmap='viridis', alpha=0.4)
    plt.show()
    
    plt.scatter(rpts[0], rpts[2], color='cyan', s=5)
    ctf = plt.contourf(x, z, pdist[:, int(len(y)/2), :].T, levels=10, cmap='viridis', alpha=0.4)
    plt.show()"""
    
    
    #Apply a random perturbation to the points to ensure they're not on grid cells
    rpts =  random_dr(rpts, dr)
    
    return rpts, rgr
    

#Build a cluster
def gen_positions(rinit, Ntot,  Nstep=10, dmin=0.01, dmax=20.0, p=def_p, d0=1.0):
    
    Nstars= rinit.shape[1]
    rpts = rinit
    dr = np.median(np.diff(rgr))*2.
    
    while Nstars<Ntot:
        probs = calc_probseeds(rpts, dmin=dmin, dmax=dmax, p=p, d0=d0)
        
        istar =  weighted_random_choice(probs, size=1)[0]

        rpts_ = draw_corrfunc(rpts, istar, dmin=dmin, dmax=dmax, p=p, d0=d0, covmat=None)
        rpts = np.append(rpts, rpts_, axis=1)
        
        Nstars = rpts.shape[1]
        print(Nstars)
    
    np.save('star_positions', rpts)
    plt.scatter(rpts[0], rpts[-1], s=2, color='r')
    plt.show()
    
    return rpts



def plot_gauss(covmat= def_covmat, mu=def_mu):
    x = np.linspace(-10., 10.0, 50)
    y = np.linspace(-10., 10., 51)
    z = np.linspace(-10., 10., 52)
    
    rgr = initialise_grid(x, y, z=z)
    invcov = np.linalg.inv(covmat)
    
    pg = pmulti_gauss(rgr, mu, invcov)
    
    ctf = plt.contourf(x, y, pg[:, :, int(len(z)/2)].T, levels=10, cmap='viridis', alpha=0.4)
    plt.show()
    
def plot_corrfunc(rpts, dmin=def_dmin, dmax=def_dmax, p=def_p, d0=def_d0):

    Ndim = rpts.shape[0]
    rpg = np.array([np.meshgrid(rpts[idim], rpts[idim], indexing='ij') for idim in range(Ndim)])

    dr = np.linalg.norm(rpg - np.swapaxes(rpg, -1, -2), axis=0)

    nbins = 12
    ndp = len(dr) // nbins
    binsr = [dr[_ * ndp: (_+1)*ndp] for _ in range(nbins)]
    binsr = np.logspace(-2, 1, 25)

    corr, dcorr = bootstrap_two_point(rpts.T, binsr, Nbootstrap=5)

    corr_an = twopoint_corr(binsr[:-1], dmin, dmax, p=p, d0=d0)
    corr_an *= (corr[8] + 1.) / (corr_an[8])

    fig, ax = plt.subplots(figsize=(6., 4.))

    # Use 'linestyle' parameter to set the style for the data points
    plt.errorbar(binsr[:-1], corr + 1., yerr=dcorr, linestyle='None', fmt='o', color='b', markerfacecolor='w',markersize=4, markeredgecolor='b', label='Simulated cluster')
    plt.plot(binsr[:-1], corr_an, linewidth=1, color='k', label='Analytic PDF')
    plt.axhline(1., color='r', label='Uniform distribution')
    plt.xscale('log')
    plt.yscale('log')

    # Add labels to the plot
    plt.xlabel('Separation between pairs: $d$ [pc]')
    plt.ylabel('Offset two-point correlation function $(1+\\xi)$')

    # Show axis ticks on all sides
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    # Set the ticks to be on the upper and right sides of the plot
    ax.yaxis.set_tick_params(which='both', direction='in')
    ax.xaxis.set_tick_params(which='both', direction='in')

    plt.legend()
    plt.show()
    
          
    
if __name__=='__main__':
    #Initialise the 'seed' positions, or the number of nests - I've chosen 20 like Taurus
    rinit, rgr = initialise_gausspos(20)
    
    #Generate the star positions. I have chosen a minimum of the correlation function to be 0.1, and max 20
    #I am drawing 600 stars
    rpos = gen_positions(rinit,  NSTARS,  Nstep=1, dmin=def_dmin, dmax=def_dmax, p=def_p, d0=def_d0)
    
    #Check the correlation function
    plot_corrfunc(rpos, dmin=def_dmin, dmax=def_dmax, p=def_p, d0=def_d0)
    #plot_gauss()