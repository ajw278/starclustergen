import numpy as np
import matplotlib.pyplot as plt

from sympy import fourier_transform, inverse_fourier_transform
import scipy as scipy
from astroML.correlation import bootstrap_two_point
from astroML.correlation import two_point
from scipy.signal import windows
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

#Number of dimensions
ND=2

#Length scale of the grid (in x)
LSCALE= 20.0

#Number of grid points in x, y
Nx_def = 201
Ny_def = 201

#Covariance of 2 dimensions, and flattening in y
covxy = 0.
xyrat = 1.

#Stetch the structure as well as the cluster shape? (Not implemented)
struct_stretch = True

#Third dimension if needed
if ND==3:
    Nz_def = 201
    covxz = 0.0
    covyz = 0.0
    xzrat = 1.0
else:
    Nz_def=None

#Filling factor of the cluster
cluster_size_factor =0.4 #0.2

#Minimum size of sub-structure
rmin = LSCALE*2e-2

#Power law index of the correlaton function -- must be greater than 1! 
alpha_def = 2.0

#The (square) size scale of the cluster
CLX2 = (cluster_size_factor*LSCALE)**2

#Thi parameter determines how sharp the edges of the cluster are
SHARPEDGE=5.0

#Grid cell size (in x)
drgrid = LSCALE/Nx_def

#Is the grid fine enough to resolve structure?
if rmin<5.*drgrid:
    print('Warning: minimum size scale of correlated structure \
    should be much larger than the grid cell size')
    print('Grid cell size (x, y):', LSCALE/Nx_def, LSCALE/Ny_def)
    print('Mininum correlation scale:',  rmin)


    
#Define the covariance matrix and mean position of the cluster
if ND==3:
    def_covmat = np.array([[CLX2, covxy*xyrat*CLX2, covxz*xzrat*CLX2],\
                           [covxy*xyrat*CLX2, xyrat*xyrat*CLX2,  covyz*xyrat*xzrat*CLX2], \
                           [covxz*xzrat*CLX2, covyz*xyrat*xzrat*CLX2, xzrat*xzrat*CLX2]])
    def_mu = np.array([0.0,0.,0.])
else:
    def_covmat = np.array([[CLX2, covxy*xyrat*CLX2],[covxy*xyrat*CLX2, xyrat*xyrat*CLX2]])
    def_mu = np.array([0.0,0.])


def unnormed_pdf(x, f0=100.0, xmin=rmin, xmax = 10.*rmin,  alpha=alpha_def):
    
    return f0*np.power((x+xmin)/xmin, -alpha) + 0.5*np.tanh((x-xmax)/xmax) + 0.5
    
#Define the spherically symmetric correlation function
def correlation_function(x, xmin=rmin,  alpha=alpha_def, xedge=np.sqrt(CLX2), ndim=ND):
    
    
    #This section is written to find a normalisation constant and maximum scale of the structure
    #we need this because otherwise the correlation function doesn't have the correct properties
    xsp = np.linspace(0., max(np.amax(x), xedge), 10000)
    if ndim == 3:
        intfact = 4.*xsp*xsp*np.pi
    else:
        intfact = 2.*xsp*np.pi
    
    
    def solve_f0(th):
        lf0_, xmax_ = th[:]
        f0_ =10.**lf0_
        pdf = unnormed_pdf(xsp,f0=f0_, xmin=xmin, xmax = xmax_,  alpha=alpha)
        #pdf[xsp>xedge] = 0.0
        intpdf = np.trapz(pdf*intfact, xsp)/np.trapz(intfact, xsp)
        return np.absolute(intpdf-1.)/100.0
    
    xmax_sp = np.linspace(xmin, xedge, 20)
    f0_sp = np.logspace(0., 4., 1001)
    
    fgr, xmgr = np.meshgrid(xmax_sp, f0_sp, indexing='ij')
    fgf = fgr.flatten()
    xmf = xmgr.flatten()

    dI = np.zeros(fgf.shape)
    for i in range(len(fgf)):
        dI[i] = solve_f0([fgf[i], xmf[i]])
    
    imin = np.argmin(dI)
    
    init_guess = [fgf[imin], xmf[imin]]
    
    res = minimize(solve_f0, init_guess, bounds = [(0., 4.), (xmin, xedge)], tol=1e-6)
    f0 = 10.**res['x'][0]
    xmax = res['x'][1]
    
    #We now have the norma
    pdf = unnormed_pdf(xsp,f0=f0, xmin=xmin, xmax = xmax,  alpha=alpha)
    
    #Check pdf has the write properties
    intpdf = np.trapz(pdf*intfact, xsp)/np.trapz(intfact, xsp)
    if np.absolute(intpdf-1.0)>1e-2:
        print('Error defining normalisagtion constant in the correlation function')
        print('Integrated value after attempt:', intpdf)
        print('f0 = ', f0)
        exit()
    
    #Interpolate to the given grid points
    pdf_xv = np.interp(x, xsp, pdf)
    
    #Subtract one to give the correlation function from pdf
    xi_r = pdf_xv -1.0
    
    #Check the correlation function doesn't go below one -shouldn't! 
    if np.any(xi_r<-1.):
        print('Error: parameters chosen such that xi_r goes below -1')
        plt.scatter(x.flatten()[::10], xi_r.flatten()[::10])
        plt.xscale('log')
        plt.show()
        exit()
    
    return xi_r
    
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
    #We do not replace each cell. This ensures that the normalisation of the field
    #is not so important as long as the grid cells are sufficiently small
    index = np.random.choice(len(flattened_arr), p=weights, size=size, replace=False)
    
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

#Generate a Gaussian random field with covariance
def gen_gfield(covmat=def_covmat, mu=def_mu, dims = (Nx_def, Ny_def, Nz_def)):

    Ndim = len(dims)
    if dims[-1] is None:
        Ndim-=1

    # Field size and grid
    Nx = dims[0]
    Ny = dims[1]
    Nz = None
    if Ndim==3:
        Nz = dims[2]

    dL = LSCALE/float(Nx)  # Square size

    # Create a grid of coordinates
    rcoords = [np.linspace(-1.,1., int(dims[i]))*(dL*0.5*float(dims[i])) for i in range(Ndim)]
    rmgr = np.array(np.meshgrid(*rcoords, indexing='ij'))

    
    #Set rnorm
    if struct_stretch:
        rnorm = rnorm_cov(rmgr, covmat)
        """rnorm_alt = np.linalg.norm(rmgr, axis=0)
        levels = np.linspace(0., LSCALE, 10)
        if Ndim==3:
            plt.contourf(rnorm[:,:,Nz//2], levels=levels)
            plt.contour(rnorm_alt[:,:,Nz//2], levels=levels, colors='r')
        else:
            plt.contourf(rnorm, levels=levels)
            plt.contour(rnorm_alt, levels=levels, colors='r')
        plt.show()"""
    else:
        rnorm = np.linalg.norm(rmgr, axis=0)
    
    
    corr = correlation_function(rnorm/2./np.pi, xmin=rmin,alpha=alpha_def)
    
    
    if Ndim==3:
        corr_plt = np.sum(corr, axis=-1)
        xplt = rmgr[0, :, :, -1]
        yplt = rmgr[1, :, :, -1]
    else:
        corr_plt = corr
        xplt = rmgr[0]
        yplt = rmgr[1]
    
    axes = [i for i in range(Ndim)]
    
    # Compute the 2D power spectrum using the 2D Fourier transform
    ps = np.fft.fftshift(np.fft.fftn(corr, axes=axes))
    
    
    amplitudes = np.sqrt(ps)
    
    # Generate random complex values from a Gaussian distribution
    # with the same shape as the power spectrum
    #random_field = np.random.normal(0, 1, size=ps.shape) + 1j * np.random.normal(0, 1, size=ps.shape)
    phases = np.random.uniform(0., 2.*np.pi, size=ps.shape)
    
    random_field =np.exp(-phases*1j)

    # Multiply the random field by the square root of the power spectrum
    field = np.fft.ifftn(np.fft.ifftshift(amplitudes * random_field), axes=axes).real
    
    # Normalize the field
    mean_value = np.mean(field)
    std_deviation = np.std(field)
    field = (field - mean_value) / std_deviation
    
    if not cluster_size_factor is None:
        invcov = np.linalg.inv(covmat)
        det = np.linalg.det(covmat)
        cluster_shape = pmulti_gauss(rmgr, mu, invcov)
        cluster_shape = np
        factor = np.log(cluster_shape)*SHARPEDGE
        ctf = plt.contourf(factor)
        plt.colorbar(ctf)
        plt.show()
        factor[factor<-10.] = -10.0
        field +=factor
    dl = 0.2
    levels = np.arange(-3., 3.+dl, dl)
    if Ndim==3:
        #fieldplt = np.sum(field, axis=-1)
        fieldplt = field[:,:, Nz//2]
    else:
        fieldplt=field
    
    return rmgr, field


def draw_stars(rgr, field, Nstars=10000, normfactor=4.*np.pi*np.pi):
    
    
    #It doesn't matter what operation we do to the field as long as it is
    #(a) monotonic and (b) gives a large enough spread so that we pick each cell
    #in order of probability. Should just change the selection function to do this directly...
    istars = weighted_random_choice(np.exp(normfactor*field), size=Nstars)
    
    levels = np.linspace(-5., -1.)
    
    ndim = rgr.shape[0]
    
    rstars = np.array([rgr[idim][istars] for idim in range(ndim)])
    drgr = np.array([np.gradient(rgr[idim],axis=idim) for idim in range(ndim)])
    drstars = np.array([drgr[idim][istars] for idim in range(ndim)])
    rst = random_dr(rstars, drstars)
    
    if ndim==3:
        #If 3D, sum over the y-dimension
        field_plt = np.sum(field, axis=2)
        xplt = rgr[0, :, -1, :]
        yplt = rgr[2, :, -1, :]
    else:
        field_plt = field
        xplt = rgr[0]
        yplt = rgr[1]    
    
    
    plt.contourf(xplt, yplt, field_plt, cmap='viridis', alpha=0.5)
    plt.scatter(rst[0], rst[1], color='r', s=1)
    
    
    rgunx = np.unique(rgr[0].flatten())
    rguny = np.unique(rgr[1].flatten())
    for rgux in rgunx:
        plt.axvline(rgux, color='k', linewidth=0.01)
    for rguy in rguny:
        plt.axhline(rguy, color='k', linewidth=0.01)
    plt.colorbar()
    plt.show()
    
    return rst


def plot_corrfunc(rpts, rgr):

    Ndim = rpts.shape[0]
    rpg = np.array([np.meshgrid(rpts[idim], rpts[idim], indexing='ij') for idim in range(Ndim)])

    dr = cdist(rpts.T, rpts.T, 'euclidean')
    
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
    xi_values =correlation_function(xvals) #transform_PS2CF(xvals, kvals, P_k)
    

    fig, ax = plt.subplots(figsize=(6., 4.))

    # Use 'linestyle' parameter to set the style for the data points
    #plt.errorbar(binsr[:-1], corr , yerr=dcorr, linestyle='None', fmt='o', color='b', markerfacecolor='w',markersize=4, markeredgecolor='b', label='Simulated cluster')
    plt.scatter(binsr[:-1], corr+1., marker='o', color='w', edgecolor='b')
    plt.plot(xvals,xi_values+1., linewidth=1, color='k',linestyle='dotted', label='Analytic PDF estimate')
    #plt.axhline(1., color='r', label='Uniform distribution', linewidth=0.5, linestyle='dashed')
    plt.axvline(drgrid, color='purple', linestyle='dotted', label='Grid cell size')
    plt.axvline(rmin, color='purple', linestyle='dashed', label='Min r')
    plt.axvline(cluster_size_factor*LSCALE, color='purple', linestyle='solid', label='Max r')
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
    
    rgr, field = gen_gfield()
    rst = draw_stars(rgr, field, Nstars=3000)
    plot_corrfunc(rst, rgr)