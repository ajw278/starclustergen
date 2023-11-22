import numpy as np
from scipy.special import gamma

import matplotlib.pyplot as plt

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


def MB_dist(v_, sig, nd=3):
    signd = sig 
    norm = 0.5*gamma(nd/2.)*(4.*signd*signd)**(nd/2.)
    return (1./norm) *(v_**(nd-1))* np.exp(-v_*v_/4./signd/signd)


def sigv_pl(dr, r0=1.0, p=1., sv0=1.0):
    return sv0*(dr/r0)**p

def draw_velocity(drs, dvs, **svparams):
    
    svall = sigv_pl(drs,**svparams)
    imax =np.argmax(svall)
    imin = np.argmin(svall)
    
    res = min(max(int(2.*3.*svall[imax]/svall[imin]), 50), 200)
    
    vmax = 5.*svall[imax]
    
    vcent = dvs[:, imin]
    
    dv1d = np.linspace(-vmax, vmax, res)
    
    delta_v = dv1d[1]-dv1d[0]
    
    vgrid = np.asarray(np.meshgrid(dv1d+vcent[0], dv1d+vcent[1], dv1d+vcent[2], indexing='ij'))
    
    probs = np.ones(vgrid[0].shape)

    for i, dr in enumerate(drs):
        dvi = np.linalg.norm(vgrid - dvs[:, i, np.newaxis, np.newaxis, np.newaxis], axis=0)
        svi = svall[i]
        probs *= MB_dist(dvi, svi)
    
    probs/=np.sum(probs)
    
    iv = weighted_random_choice(probs, size=1)
    
    vstar = vgrid[:, iv[0], iv[1], iv[2]]
    vstar = random_dr(vstar, delta_v*np.ones(3)[:, np.newaxis])
    
    """ctf = plt.contourf(vgrid[0, :, :, 0],vgrid[1, :, :, 0], np.log10(np.sum(probs, axis=2)), levels=np.linspace(-6., -2. ,15) )
    plt.scatter(vgrid[0, :, :, 0],vgrid[1, :, :, 0], color='k', s=0.1)
    plt.scatter(dvs[0], dvs[1], color='cyan', marker='o', s=3)
    plt.scatter(vstar[0], vstar[1], color='red', s=10, marker='*')
    plt.colorbar(ctf)
    plt.show()"""
    
    
    return vstar[:,0]
    
    
        

def velocity_walk(rs, first_istar=None,**svparams):

    Nstars = rs.shape[1]
    istars = [i for i in np.arange(Nstars)]
    
    added = []
    
    if first_istar is None:
        iadd = np.random.choice(istars)
    
    vs = np.zeros(rs.shape)
        
    added.append(iadd)
    istars.remove(iadd)
    
    while len(istars)>0:
        
        inext = np.random.choice(istars)
        
        drs  = np.linalg.norm(np.array([rs[:, ia] - rs[:, inext] for ia in added]), axis=1)
        dvs = vs[:, added]
        #print('Before:', drs, dvs)
        if len(drs)>5:
            iclose = np.argsort(drs)[:5]
            drs = drs[iclose]
            dvs = dvs[:, iclose]
        #print('After:', drs, dvs)
        
        
        vs[:, inext] = draw_velocity(drs, dvs, **svparams)
        
        istars.remove(inext)
        added.append(inext)
        if len(added)%100==0:
            print('Velocities generated for %d / %d stars'%(len(added), Nstars))
    
    return vs