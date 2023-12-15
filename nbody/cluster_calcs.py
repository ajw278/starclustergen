import copy
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


def binary_sma(r1, r2, v1, v2, m1,m2, G=1.0):
	dr = r1-r2
	dv = v1-v2
	mu = m1+m2
	pot = G*mu/np.linalg.norm(dr, axis=-1)
	kin = 0.5*np.linalg.norm(dv, axis=-1)**2
	eps =  kin - pot
	return -mu/2./eps

def encounter_history_istar(istar, rstars, vstars,  mstars,nclose=3):

	istar=  np.atleast_1d(istar)

	drstars = rstars - rstars[:, istar]
	dvstars = vstars - vstars[:, istar]

	drmags = np.linalg.norm(drstars, axis=-1)

	ninds = np.argsort(drmags, axis=1)[:, 1:nclose+1]
	nblst = np.unique(ninds)

	cx = np.swapaxes(drstars[:, nblst], 0,1)
	cv = np.swapaxes(dvstars[:, nblst],0,1)
	cm = mstars[nblst]
	
	return cx, cv, cm, nblst 

def binary_filter(cx, cv, cm, sepfilt=0.1, G=1.0):
	cx_alt = copy.copy(cx)
	cv_alt = copy.copy(cv)
	cx_mag = np.linalg.norm(cx, axis=-1)
	cm_alt = copy.copy(cm)

	for j_s in range(len(cx)-1):
		for i_s in range(j_s+1, len(cx)):
			abin = binary_sma(cx[j_s], cx[i_s], cv[j_s], cv[i_s], cm[j_s],cm[i_s], G=G)
			ifilt = (abin>0.0)&(abin<sepfilt*cx_mag[j_s])
			bcm = (cm[j_s]+cm[i_s])
			bcx = (cx[j_s]*cm[j_s] + cx[i_s]*cm[i_s])/bcm
			bcv = (cv[j_s]*cm[j_s] + cv[i_s]*cm[i_s])/bcm
			if np.sum(ifilt)>0:
				cx_alt[j_s, ifilt] = bcx[ifilt]
				cx_alt[i_s, ifilt] = np.inf
				cv_alt[j_s, ifilt] = bcv[ifilt]
				cv_alt[i_s, ifilt] = np.inf
				cm_alt[j_s ] = cm[i_s]+cm[j_s]
	
	return cx_alt, cv_alt, cm_alt


				

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
				if dtperi[ienc]<dt and dtperi[ienc]>0.0:
					ttmp[ienc] = ct[lm[ienc]]+dtperi[ienc]
					etmp[ienc] = e[ienc]
					xtmp[ienc] = rp[ienc]
				else:
					ttmp[ienc] = ct[lm[ienc]]
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
