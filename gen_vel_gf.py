import numpy as np
from scipy.special import gamma

import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import dok_matrix, csr_matrix

def MB_dist(v_, sig, nd=3):
    signd = sig 
    norm = 0.5*gamma(nd/2.)*(4.*signd*signd)**(nd/2.)
    return (1./norm) *(v_**(nd-1))* np.exp(-v_*v_/4./signd/signd)


def sigv_pl(dr, r0=1.0, p=1., sv0=1.0):
    return sv0*(dr/r0)**p

def correlation_function(dr, svmax=2e5,lmin=1.0, **svparams):
    
    return np.exp(-dr*dr/lmin/lmin/2.)

def RQ_Kernel(dr, lmin=0.5, alpha=1.0, **svparams):
    return (1.+(dr/lmin)**2)**-alpha

"""def force_psd(matrix, epsilon=1e-8):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < epsilon] = epsilon
    psd_matrix = np.dot(np.dot(eigvecs, np.diag(eigvals)), eigvecs.T)
    return psd_matrix"""

def force_psd(matrix, alpha=1e-2):
    psd_matrix = matrix*(1.+ alpha*np.eye(len(matrix)))
    return psd_matrix

def plot_dvNN(rs, vs, **svparams):
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
    
    bins = np.logspace(-1.5, 1.5)
    """plt.hist(distances.flatten(), bins=bins, density=True, histtype='step')
    plt.hist( nearest_neighbor_distances, bins=bins,density=True, histtype='step')
    plt.yscale('log')
    plt.xscale('log')
    plt.show"""
    
    

    # Calculate the magnitude of the difference in velocity between each star and its nearest neighbor
    velocity_differences = np.linalg.norm(velocities - velocities[nearest_neighbors], axis=1)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    
    dsp = np.logspace(-2, 1.5, 40)
    vsp = np.logspace(-2, 1.5, 30)
    D, V = np.meshgrid(dsp, vsp, indexing='ij')

    """dv_gr = np.vstack([np.log10(ds), np.log10(dv)])
    kde = gaussian_kde(dv_gr,bw_method='scott')

    Z = kde(np.vstack([D.ravel(), V.ravel()])).reshape(D.shape)

    
    
    ctf = plt.contourf(D, V, np.log10(Z), levels=levels)"""
    sigvs  = sigv_pl(D, **svparams)
    
    print(sigvs, D)
    
    levels = np.arange(-4.0, 0.2, 0.1)
    pdist = V*V*MB_dist(V*1e5, sigvs, nd=2)
    pdist /= np.amax(pdist, axis=1)[:, np.newaxis]
    ctf=plt.contourf(D, V, np.log10(pdist), levels=levels)
    plt.colorbar(ctf, label='Normalised MB probability: $\log [v g(v)]$')

    plt.scatter(nearest_neighbor_distances, velocity_differences, c='cyan', edgecolor='gray', s=5)
    # Add labels and title

    rsp = np.logspace(-2, 2.0)
    plt.plot(rsp, sigv_pl(rsp, **svparams)/1e5, color='r', linewidth=2)
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Magnitude of Velocity Difference to Nearest Neighbor')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([5e-2, 20.])
    plt.ylim([5e-2, 20.])
    plt.show()
    

def velocity_gen(rstars, sigv=1.5e5, **svparams):
    
    positions = rstars.T
    
    # Assuming you have positions stored in the array 'positions'
    num_particles = len(positions)

    drmat = cdist(positions, positions)
    

    # Apply the covariance function element-wise to obtain the covariance matrix
    covmat = correlation_function(drmat, lmin=0.1*np.median(drmat), **svparams)
    
    print(covmat)
    print(covmat.shape)
    
    covmat *= (sigv*sigv)
    
    
    #np.fill_diagonal(covmat, 1.0)
    
    from scipy.linalg import eigh

    def is_positive_semidefinite(matrix, tol=1e-10):
        # Step 1: Check Symmetry
        if not np.allclose(matrix, matrix.T, atol=tol):
            return False

        # Step 2: Eigenvalue Decomposition
        eigenvalues, _ = eigh(matrix)
        
        print(eigenvalues)

        # Step 3: Tolerance Handling
        return np.all(eigenvalues >= -tol)

    # Example usage:
    result = is_positive_semidefinite(covmat)

    if result:
        print("The matrix is positive semi-definite.")
    else:
        print("The matrix is NOT positive semi-definite.")
        print('Projecting onto the PSD cone...')
        covmat = force_psd(covmat, alpha=1e-2)

    
    
    # Perform Cholesky decomposition on the sparse covariance matrix
    cholesky_matrix = np.linalg.cholesky(covmat)

    # Generate standard normal samples
    ux, uy, uz = np.random.normal(0, 1, size=(3,num_particles))

    # Transform to desired covariance
    vx = np.dot(cholesky_matrix, ux)
    vy = np.dot(cholesky_matrix, uy)
    vz = np.dot(cholesky_matrix, uz)
    
    vstars = np.array([vx, vy, vz])
    
    plot_dvNN(rstars, vstars/1e5, **svparams)

    
  

    return vstars
