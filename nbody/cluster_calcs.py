
import numpy as np
import scipy.spatial as spatial


def com_calc(rvstars, mstars):
	rvx, rvy, rvz = np.swapaxes(rvstars,0,1)

	com_x = np.sum(rvx*mstars)/np.sum(mstars)
	com_y = np.sum(rvy*mstars)/np.sum(mstars)
	com_z = np.sum(rvz*mstars)/np.sum(mstars)

	return np.array([com_x, com_y, com_z])

def adjust_com(rvstars, mstars):

    com = com_calc(rvstars, mstars)
    rvstars -= com[np.newaxis, :]
    maxrv = np.amax(rvstars)
    check_com = com_calc(rvstars, mstars)

    return rvstars

def stellar_potential(rstars, mstars):
	
    gpot_tot = 0.0
    dr = spatial.distance.cdist(rstars, rstars)
    dr[np.diag_indices(len(dr), ndim=2)] = np.inf
    gpot = mstars[np.newaxis,:]*mstars[:, np.newaxis]/dr

    return np.sum(np.triu(gpot,k=1))

def total_kinetic(vstars, mstars):
    vsq = np.linalg.norm(vstars, axis=1)**2
    ke = np.sum(0.5*mstars*vsq)
    return ke


#from scipy.ndimage import gaussian_filter1d


def encounter_history_istar(istar, rstars, vstars,  mstars,nclose=3):
	
	#Want rstars in form [star][time][dimension]
	#rstars = gaussian_filter1d(rstars,1, axis=0)
	#vstars = gaussian_filter1d(vstars, 1, axis=0)
	rstars = np.swapaxes(rstars, 0,1)
	vstars = np.swapaxes(vstars, 0,1)

	

	nghbrs = np.zeros((rstars.shape[1], nclose))
	closest_x = np.zeros((rstars.shape[1], nclose, rstars.shape[2]))
	closest_v = np.zeros((rstars.shape[1], nclose, rstars.shape[2]))
	closest_m = np.zeros((rstars.shape[1], nclose))

	cx = []
	cv = []
	cm = []


	drstars = rstars - rstars[istar]
	dvstars = vstars - vstars[istar]
	drmags = np.linalg.norm(drstars, axis=2)

	drstars = np.swapaxes(drstars, 0,1)
	dvstars = np.swapaxes(dvstars, 0,1)
	drmags = np.swapaxes(drmags, 0,1)

	nblst = []
	
	for itime in range(len(drmags)):
		ninds = np.argsort(drmags[itime])[1:nclose+1]
		for nb in ninds:
			if not (nb in nblst):
				nblst.append(nb)
		nghbrs[itime] = ninds
		closest_x[itime] = drstars[itime][ninds]
		closest_v[itime] = dvstars[itime][ninds]
		closest_m[itime] = mstars[ninds]

	for nb in nblst:
		#[star][time][dimension] -> time, star, dim
		cxval = np.swapaxes(drstars,0,1)[nb]
		cx.append(cxval)
		cv.append(np.swapaxes(dvstars,0,1)[nb])
		cm.append(mstars[nb])

	cx = np.array(cx)
	cv = np.array(cv)
	
	return cx, cv, cm, nblst #closest_x, closest_v, closest_m, nghbrs



def calc_e_rp(dr, dv, m1, m2, hh, G=1):
    
    mu = G * (m1 + m2)
    a = 1 / (2 / dr - dv * dv / mu)
    v_infinity = np.sqrt(abs( -mu / a))
    b = hh / v_infinity
    ee = np.sqrt(1 - np.sign(a)*b**2 / a**2)
    rr_p = -a * (ee - 1)
	#     pp = np.sqrt(abs(4 * np.pi * a**3 / mu))
    
    return ee, rr_p

def calc_h(dx, dy, dz, dvx, dvy, dvz):
    
    dr = np.array([dx, dy, dz])
    dv = np.array([dvx, dvy, dvz])
    
    hh = np.cross(dr, dv)
    
    return np.linalg.norm(hh) 

def get_closeapproach(dr, dv, m1, m2, T, add=1, G=1.0):
	#Gravitational parameter
	mu = G*(m1 + m2)

	#Calculate specific angular momentum
	h = np.cross(dr, dv, axis=-1)
	drmag = np.linalg.norm(dr, axis=-1)[:,np.newaxis]
	dvmag = np.linalg.norm(dv, axis=-1)[:,np.newaxis]

	print(drmag.shape, dvmag.shape, h)

	#Calculate eccentricity vector
	e_vec = (np.cross(dv, h, axis=-1) / mu) - (dr / drmag)


	#Calculate eccentricity
	e = np.linalg.norm(e_vec, axis=-1)


	# Calculate specific orbital energy
	eps = 0.5*(dvmag ** 2)  - (mu / drmag)

	eps = eps.flatten()
	drmag = drmag.flatten()
	dvmag = dvmag.flatten()

	# Calculate semi-major axis and the pericentre distance
	a = -mu / (2 * eps)
	rp = a*(1.-e)


	#Eccentric anomaly
	print(drmag/a, e)
	E = np.arccos((1. - np.absolute(drmag/a))/e)

	# Calculate mean motion
	n = np.sqrt(mu / np.absolute(a)**3)

	# Calculate the mean anomaly
	M =  e*np.sinh(E) -E

	#Finally, get periastron itme
	T_peri = T - np.sign(add)*M / n


	return e.flatten(), rp.flatten(), T_peri.flatten()

def encounter_params(cx, cv, cm, ct, mstar):
	diff = np.amin(np.diff(ct))

	ce_eccs = []
	ce_x = []
	ce_time = []

	import matplotlib.pyplot as plt
	

	cxmag = np.apply_along_axis(np.linalg.norm, 2, cx)
	cxmagmin = np.amin(cxmag, axis=0)
	for ix in range(len(cx)):

		cvdotx = np.einsum('ij,ij->i', cx[ix], cv[ix])
		
		lm = []
		dts = []
		for ixmag in range(1,len(cxmag[ix])-1):
			
			if cxmag[ix][ixmag] == cxmagmin[ixmag]:
				if cvdotx[ixmag-1]<0.0 and cvdotx[ixmag]>0.0:
					lm.append(ixmag-1)
					dts.append(ct[ixmag]-ct[ixmag-1])

	 	#local_minima = np.array(local_minima)
		if len(lm)>0:
			xtmp = np.zeros(len(lm))
			etmp = np.zeros(xtmp.shape)
			ttmp = np.zeros(xtmp.shape)

			#hsmag= np.apply_along_axis(np.linalg.norm, 1, hs)
			#mu = cm[ix]+mstar
			#ls = hsmag*hsmag/mu
			e, rp, dtperi = get_closeapproach(cx[ix][lm], cv[ix][lm], mstar,cm[ix],0.0)

			for ienc, dt in enumerate(dts):
				if dtperi[ienc]<dt:
					ttmp[ienc] = ct[lm[ienc]]+dtperi[ienc]
					etmp[ienc] = e[ienc]
					xtmp[ienc] = rp[ienc]
				else:
					ttmp[ienc] = ct[lm[ienc]]s
					etmp[ienc] = e[ienc]
					xtmp[ienc] = cxmag[ix][lm[ienc]]
			ce_x.append(xtmp)
			ce_eccs.append(etmp)
			ce_time.append(ttmp)
		else:
			
			ce_eccs.append(np.array([]))
			ce_x.append(np.array([]))
			ce_time.append(np.array([]))
			
		
	
	return ce_x, ce_eccs, ce_time
