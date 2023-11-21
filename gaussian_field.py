import numpy as np
import matplotlib.pyplot as plt

from sympy import fourier_transform, inverse_fourier_transform
import scipy as scipy
from astroML.correlation import bootstrap_two_point
from astroML.correlation import two_point
from scipy.signal import windows
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from powerbox import get_power
import powerbox as pbox

#Number of dimensions
ND=3

#Length scale of the grid (in x)
Lscale= 50.0

#Number of grid points in x, y
Ngrid = 200


alpha_def = -0.8


#Stetch the structure as well as the cluster shape? (Not implemented)
struct_stretch = True

cluster_size_factor =0.5 #0.2


#The (square) size scale of the cluster
CLX2 = (cluster_size_factor*Lscale)**2

#Thi parameter determines how sharp the edges of the cluster are
SHARPEDGE=10.0


#Covariance of 2 dimensions, and flattening in y
covxy = 0.
xyrat = 1.

#Third dimension if needed
if ND==3:
    covxz = 0.0
    covyz = 0.0
    xzrat = 1.0

#Define the covariance matrix and mean position of the cluster
if ND==3:
    def_covmat = np.array([[CLX2, covxy*xyrat*CLX2, covxz*xzrat*CLX2],\
                           [covxy*xyrat*CLX2, xyrat*xyrat*CLX2,  covyz*xyrat*xzrat*CLX2], \
                           [covxz*xzrat*CLX2, covyz*xyrat*xzrat*CLX2, xzrat*xzrat*CLX2]])
    def_mu = np.array([0.0,0.,0.])
else:
    def_covmat = np.array([[CLX2, covxy*xyrat*CLX2],[covxy*xyrat*CLX2, xyrat*xyrat*CLX2]])
    def_mu = np.array([0.0,0.])

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
    print(np.sum(flattened_arr))
    
    weights = flattened_arr / np.sum(flattened_arr)
    
    print(np.sum(flattened_arr))
    # Choose a random index based on the calculated weights
    #We do not replace each cell. This ensures that the normalisation of the field
    #is not so important as long as the grid cells are sufficiently small
    index = np.random.choice(len(flattened_arr), p=weights, size=size, replace=True)
    
    
    # Convert the 1D index back to N-D indices
    indices = np.unravel_index(index, arr.shape)
    
    
    return indices

#Add a random perturbation of size ~dr to a value
def random_dr(rpts, dr):
    return rpts+ (0.5-np.random.uniform(size=rpts.shape))*dr


def rnorm_cov(rgr, covmat):
    if not covmat is None:
        invcov = np.linalg.inv(covmat)
        det = np.linalg.det(covmat)
        # Calculate the quadratic form (dusq) element-wise
        rsig = np.einsum('ij...,jk...->ik...', invcov, rgr)
        rnorm2 = np.einsum('ij...,ij...->j...', rgr,  rsig)
        rnorm2 *= (det)**(1./covmat.shape[0])
        return np.sqrt(rnorm2)
    
    return np.linalg.norm(rgr, axis=0)

def Wfunc(u):
    return (3/u/u/u)*(np.sin(u)-u*np.cos(u))
    
def integral_ps_old(alpha_ps, Lbox, Ndim, kmin, kmax,  scale=1.0):
    ksp = np.logspace(np.log10(kmin), np.log10(kmax), 100 )
    factor = (1./2./np.pi)
    intfact = np.power(ksp, Ndim-1)
    sigR2 =  scale*(1./norm)*(np.trapz(Pk*intfact, ksp))
    return sigR2

def integral_ps_3D(alpha_ps, Lbox, Nbox, scale=1.0):
    print(Nbox, Nbox/2, Nbox//2)
    kbox = np.arange(-Nbox/2, Nbox/2, Nbox//2 )/Lbox
    kbox3 = np.asarray(np.meshgrid(kbox, kbox, kbox, indexing='ij'))
    
    kmag = np.linalg.norm(kbox3, axis=0)
    
    dv = kbox[1]-kbox[0]
    dv3 = dv*dv*dv
    
    Pk = scale*kmag**alpha_ps
    Pk[kmag==0] = 0.0
    
    sigR2 = np.sum(Pk*dv3) #/np.power(2.*np.pi, 3)
    
    return sigR2
    

def lognormal(x, mu, sigma):
    return (1./x/sigma/np.sqrt(2.*np.pi))*np.exp(-(np.log(x)-mu)**2/2./sigma/sigma)

#Generate a Gaussian random field with covariance
def gen_gfield(covmat=def_covmat, mu=def_mu, Ndim=ND, Nbox =Ngrid , Lbox=Lscale, Pk_norm=10.0, Pk_index=-1.5, sharp_edge=SHARPEDGE,plot=True, seed=None):
    
    Ndim = covmat.shape[0]
    dLbox = Lbox/Nbox
    
    psfunc = lambda k: Pk_norm*k**Pk_index
    
    lnpb = pbox.LogNormalPowerBox(
        N=Nbox,                     # Number of grid-points in the box
        dim=Ndim,                     # 2D box
        pk = psfunc, # The power-spectrum
        boxlength = Lbox,           # Size of the box (sets the units of k in pk)
        seed = seed            # Fix the seed to get the same 
    )
   
   
                                   
    delta_x = lnpb.delta_x()
    
    """ntspect = integral_ps_3D(alpha_ps, Lbox, Nbox, factor)
    print('SigR2', np.std(delta_x)**2, np.std(1.+delta_x)**2)
    print('Scale:', intspect, np.var(delta_x))
    print('Quantities:', Nbox*Nbox*Nbox, Lbox*Lbox*Lbox, np.power(2.*np.pi, 3))
    
    bins = np.linspace(0., 10.0, 20)
    weights = (1.+delta_x).flatten()""" 
                             
    args = [lnpb.x] * lnpb.dim
    rmgr = np.asarray(np.meshgrid(*args, indexing="ij"))
    
    print(dir(lnpb))
    
    density_ = 1.+delta_x

    invcov = np.linalg.inv(covmat)
    det = np.linalg.det(covmat)
    volume = det
    cluster_shape = pmulti_gauss(rmgr, mu, invcov)
    
    density = density_*(cluster_shape**sharp_edge)
    density /= np.amax(density)
    
    if plot and False:
        if Ndim==3:
            
            fig = plt.figure()

            # Scatter plot with color based on density
            ctf = plt.contourf(rmgr[0, :, :, 0], rmgr[1, :, :, 0], np.log10(np.sum(density_[:,:], axis=0)).T)

            # Add a color bar
            plt.colorbar(ctf)

            # Show the plot
            plt.show()
            
            fig = plt.figure()

            # Scatter plot with color based on density
            ctf = plt.contourf(rmgr[0, :, :, 0], rmgr[1, :, :, 0], np.log10(np.sum(density[:,:], axis=0)).T, levels=np.arange(-5., 0., 0.5))

            # Add a color bar
            plt.colorbar(ctf)

            # Show the plot
            plt.show()
        else:

            plt.contourf(rmgr[0], rmgr[1], np.log10(density).T)
            plt.colorbar()
            plt.show()

    return rmgr, density, psfunc


def draw_stars(rgr, density, Nstars=500, plot=True, Pkfunc=None):
    
    
    istars = weighted_random_choice(density, size=Nstars)
    
    levels = np.linspace(-5., -1.)
    
    ndim = rgr.shape[0]
    
    rstars = np.array([rgr[idim][istars] for idim in range(ndim)])
    drgr = np.array([np.gradient(rgr[idim],axis=idim) for idim in range(ndim)])
    drstars = np.array([drgr[idim][istars] for idim in range(ndim)])
    rst = random_dr(rstars, drstars)
    
    def inv_Abel(F, R):
        f = np.zeros(R.shape)
        dFdR  = np.gradient(F, R)

        for i, R_ in enumerate(R):
            iR = R>R_
            f[i] = -np.trapz(dFdR[iR]/np.sqrt(R[iR]**2-R_**2), R[iR])

        return f/np.pi

    Rpc = np.array([rst[0], rst[1]])
    # The number of grid points are also required when passing the samples
    print(Rpc)
    
    print(Rpc.T.shape, rst.T.shape)
    p_k_samples, bins_samples = get_power(Rpc.T,50.0,N=30, b=1.0)
    p_k_samples3D, bins_samples3D = get_power(rst.T,50.0,N=30, b=1.0)
   

    
    if plot:
        print(bins_samples)
        #plt.plot(bins_samples, 30.0*bins_samples**-1.6,label="Guess Power")
        plt.plot(bins_samples, p_k_samples, marker='o', label="Normal Sample Power")


        #plt.plot(bins_samples, inv_Abel(p_k_samples, bins_samples), marker='s', label='Inverse Abel')
        plt.plot(bins_samples3D, p_k_samples3D, marker='s', label='3D True')
        if not Pkfunc is None:
            plt.plot(bins_samples3D, Pkfunc(bins_samples3D),label="Input Power")

        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('corrfuncs.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        
        if ndim==3:

            #If 3D, sum over the y-dimension
            field_plt = np.sum(density, axis=-1)
            xplt = rgr[0, :, :, -1]
            yplt = rgr[1, :,  :, -1]
        else:
            field_plt = density
            xplt = rgr[0]
            yplt = rgr[1]    


        ctf = plt.contourf(xplt, yplt, np.log10(field_plt), cmap='viridis', levels=np.arange(-38., -34, 0.5))
        plt.scatter(rst[0], rst[1], color='r', s=1, alpha=0.3)


        rgunx = np.unique(rgr[0].flatten())
        rguny = np.unique(rgr[1].flatten())
        for rgux in rgunx:
            plt.axvline(rgux, color='k', linewidth=0.01)
        for rguy in rguny:
            plt.axhline(rguy, color='k', linewidth=0.01)
        plt.colorbar(ctf)
        
        plt.savefig('gfield.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    return rst


def plot_corrfunc(rpts, rgr):

    Ndim = rpts.shape[0]
    rpg = np.array([np.meshgrid(rpts[idim], rpts[idim], indexing='ij') for idim in range(Ndim)])

    dr = cdist(rpts.T, rpts.T, 'euclidean')
    dr = dr[np.triu_indices(len(rpts[0]), k=1)]
    
    binsr = np.logspace(-1.2, 2, 35)
    if Ndim==2:
        weights = 1./(2.*np.pi*dr)
    else:
        weights = 1./(4.*np.pi*dr*dr)
        
    gamma = alpha_def+3.
    
    xsp  = np.logspace(-1.2, 2, 1000)
    unn = 10.*np.power(xsp, -gamma)
    unn /= np.trapz(unn*xsp*2.*np.pi, xsp)
    #plt.plot(xsp, unn)
    plt.hist(dr, bins=binsr, edgecolor='k', alpha=0.5, weights=weights, density=True)
    plt.xlabel('Angular separation: $\Delta \\theta$ [deg]')
    plt.ylabel('Probability density function: $\mathrm{d}P /\mathrm{d} A = \\tilde{f}(R)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([3e-4, 1e2])
    plt.xlim([2e-2, 1e2])
    plt.grid(True)
    # Display the histogram
    plt.show()

    dr = dr[dr>0]

    nbins = 12
    ndp = len(dr) // nbins
    binsr = [dr[_ * ndp: (_+1)*ndp] for _ in range(nbins)]
    binsr = np.logspace(-2, 1, 35)
    

    #corr, dcorr = bootstrap_two_point(rpts.T, binsr, Nbootstrap=5)
    rptsxy = np.swapaxes(rpts, 0, -1)
    corr = two_point(rptsxy, binsr)
    
    dx = 1e-2
    Lx = np.amax(rpts)*2.
    
    # sample spacing
    Nx = 1000
    Lx = binsr[-1]
    dLx = Lx/Nx
    
    # Compute the 2PCF at the specified distances
    xvals = binsr[:-1]+binsr[1:]
    xvals *= 0.5 #np.logspace(np.log10(drgrid), np.log10(Lx), Nx)
    plxi_values = xvals**-alpha_def 
    #Renormalise?
    plxi_values *= np.absolute(corr[len(corr)//2]/plxi_values[len(corr)//2])

    fig, ax = plt.subplots(figsize=(6., 4.))

    # Use 'linestyle' parameter to set the style for the data points
    #plt.errorbar(binsr[:-1], corr , yerr=dcorr, linestyle='None', fmt='o', color='b', markerfacecolor='w',markersize=4, markeredgecolor='b', label='Simulated cluster')
    plt.scatter(binsr[:-1], corr, marker='o', color='w', edgecolor='b')
    plt.plot(xvals,plxi_values, linewidth=1, color='k',linestyle='dashed', label='Power-law PDF estimate')
    #plt.axhline(1., color='r', label='Uniform distribution', linewidth=0.5, linestyle='dashed')
    plt.axvline(drgrid, color='purple', linestyle='dotted', label='Grid cell size')
    plt.axvline(rmin, color='purple', linestyle='dashed', label='Min r')
    plt.axvline(rmax, color='purple', linestyle='solid', label='Max r')
    plt.xscale('log')
    plt.yscale('log')


    # Add labels to the plot
    plt.xlabel('Separation between pairs: $d$ [pc]')
    plt.ylabel('Correlation function $\\xi$')

    # Show axis ticks on all sides
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    # Set the ticks to be on the upper and right sides of the plot
    ax.yaxis.set_tick_params(which='both', direction='in')
    ax.xaxis.set_tick_params(which='both', direction='in')

    plt.legend()
    plt.show()

def build_cluster(Nstars=500, Nbox=Ngrid,  Lbox=Lscale, Rcl = cluster_size_factor*Lscale, \
                  normed_covmat=def_covmat, sharp_edge=10.0, mu=def_mu, Pk_norm=10.0, Pk_index=-1.5, seed=None):
    
    ndim = normed_covmat.shape[0]
    normed_covmat /= np.linalg.det(normed_covmat)**(1./ndim)
    covmat = normed_covmat*(Rcl)**2.
    rgr, field, pkfunc = gen_gfield(covmat=covmat, mu=mu, Ndim=ndim, Nbox =Nbox, Pk_norm=Pk_norm, Pk_index=Pk_index, sharp_edge=sharp_edge, seed=seed)
    rst = draw_stars(rgr, field, Nstars=Nstars, Pkfunc=pkfunc)
    return rst
    
    
if __name__=='__main__':
    
    rgr, field, pkfunc = gen_gfield()
    rst = draw_stars(rgr, field, Nstars=2000)
    plot_corrfunc(rst, rgr)