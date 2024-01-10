
import sys
import os
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(scriptdir)
sys.path.append(scriptdir+'/..')
import gen_binary_rebound as rbb

import copy
import numpy as np
import scipy.spatial as spatial
import scipy.special as sp
from scipy.optimize import newton

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

def split_true_chunks(iflat):
    # Find the indices where iflat is True
    true_indices = np.where(iflat)[0]

    # Split the true_indices into chunks
    if len(true_indices) > 0:
        chunks = np.split(true_indices, np.where(np.diff(true_indices) != 1)[0] + 1)
    else:
        chunks = []

    return chunks

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

def keplerian_orbital_elements(m1, m2, dr, dv, G=1.0):
	#Gravitational parameter
	mu = G*(m1 + m2)

	#Calculate specific angular momentum
	h = np.cross(dr, dv, axis=-1)
	hmag = np.linalg.norm(h)
	print('h r, v', h, dr, dv)

	drmag = np.linalg.norm(dr, axis=-1)
	dvmag = np.linalg.norm(dv, axis=-1)

	print('drm dvm', drmag, dvmag)

	#Calculate eccentricity vector
	e_vec = (np.cross(dv, h, axis=-1) / mu) - (dr / drmag)

	#Calculate eccentricity
	e = np.linalg.norm(e_vec, axis=-1)

	# Calculate specific orbital energy
	eps = 0.5*(dvmag ** 2)  - (mu / drmag)

	# Calculate semi-major axis and the pericentre distance
	a = -mu / (2 * eps)


	# Calculate inclination
	inclination = np.arccos(h[2] / hmag)

	# Calculate longitude of the ascending node
	Omega = np.arctan2(h[0], -h[1])
	n = np.array([-h[1], h[0], 0.0])
	nmag = np.linalg.norm(n, axis=0)
	ndote = np.dot(n, e_vec)

	arg_periapsis = np.arccos(max(min(ndote/nmag/e, 1.0),-1.0))

	#Eccentric anomaly
	E = np.arccos((1. - np.absolute(drmag/a))/e)

	# Calculate mean motion
	n = np.sqrt(mu / np.absolute(a)**3)

	# Calculate the mean anomaly
	M =  E - e*np.sin(E)

	print('E e sinh(E)', E, e, np.sin(E))

	#Finally, get periastron itme
	dt_peri =  - M / n

	return a, e, inclination, Omega, arg_periapsis, dt_peri

def eccentric_anomaly(M, e, N=50):
    E = M + 2 * np.sum([(sp.jn(n, n * e) / n) * np.sin(n * M) for n in range(1, N + 1)])
    return E


def keplerian_state_vector(a, e, i, Omega, arg_periapsis, m1, m2, t, tp, G=1.0):

	mu = G*(m1+m2)

	# Calculate mean motion
	nmm = np.sqrt(mu / a**3)

	# Calculate mean anomaly
	M = (nmm * (t - tp))%(2.*np.pi)

	# Solve Kepler's equation to find eccentric anomaly
	#E = eccentric_anomaly(M, e)

	E_guess = M if e < 0.8 else np.pi
	E = newton(lambda x: x - e * np.sin(x) - M, E_guess)

	# Calculate true anomaly
	f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

	print(t, E, M)

	#Calculate the separation:
	rmag  = a*(1.-e*e)/(1.+e*np.cos(E))
	vmag = np.sqrt(mu * (2./rmag - 1./a))

	# Calculate position in orbital plane
	#r_orbital_plane = np.array([a * (np.cos(f) - e), a * np.sqrt(1 - e**2) * np.sin(f), 0])

	# Orbital plane coordinates (x', y', z')
	x_prime = rmag * (np.cos(Omega) * np.cos(arg_periapsis + f) - np.sin(Omega) * np.sin(arg_periapsis + f) * np.cos(i))
	y_prime = rmag * (np.sin(Omega) * np.cos(arg_periapsis + f) + np.cos(Omega) * np.sin(arg_periapsis + f) * np.cos(i))
	z_prime = rmag * np.sin(arg_periapsis + f) * np.sin(i)

	rvec = np.array([x_prime, y_prime, z_prime])
	
	# Orbital plane velocity components (vx', vy', vz')
	vx_prime = vmag * (-np.sin(Omega) * np.cos(arg_periapsis + f) - np.cos(Omega) * np.sin(arg_periapsis + f) * np.cos(i))
	vy_prime = vmag * (np.cos(Omega) * np.cos(arg_periapsis + f) - np.sin(Omega) * np.sin(arg_periapsis + f) * np.cos(i))
	vz_prime = vmag * np.sin(arg_periapsis + f) * np.sin(i)
	
	vvec = np.array([vx_prime, vy_prime, vz_prime])

	raise Warning('Kep. state function not implemented correctly (velocities incorrect)')


def binary_state(a, e, i, Omega, arg_periapsis, m1, m2, t, tp, G=1.0):
	q = m2/m1
	mu = G*(m1+m2)

	# Calculate mean motion
	nmm = np.sqrt(mu / a**3)

	# Calculate mean anomaly
	M = (nmm * (t - tp))%(2.*np.pi)

	# Solve Kepler's equation to find eccentric anomaly
	#E = eccentric_anomaly(M, e)

	E_guess = M if e < 0.8 else np.pi
	E = newton(lambda x: x - e * np.sin(x) - M, E_guess)

	# Calculate true anomaly
	f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

	dr, dv = rbb.get_binary_xv(m1, q, a, e, i, arg_periapsis, Omega, f, centre_primary=True, G=1.0)
	return dr, dv

import matplotlib.pyplot as plt

def stable_binary_filter(m, cx, cv, cm, ct, G=1.0):
	cx_alt = copy.copy(cx)
	cv_alt = copy.copy(cv)
	cx_mag = np.linalg.norm(cx, axis=-1)
	cv_mag = np.linalg.norm(cv, axis=-1)
	cm_alt = copy.copy(cm)

	orbf = m*cm_alt[:, np.newaxis]/(m+cm_alt[:,np.newaxis])/cx_mag**2

	iflat = np.absolute(np.diff(cv_mag, axis=1)) < 1e-2*np.diff(ct)[np.newaxis,:]*orbf[:, :-1]

	for inghbr in range(0, cx_mag.shape[0]):
		stablist = split_true_chunks(iflat[inghbr])


		for stab in stablist:
			if len(stab)>2:
				print('Stab greater than 1 for neighbour', inghbr)
				print('Stab:', stab)

				ibef = stab[0]-1
				m2 = cm[inghbr]

				# Time array
				t_arr = ct[stab]
				t0 = ct[ibef]

				# Compute Keplerian orbital elements
				print('Params:', m, m2, cx[inghbr][ibef], cv[inghbr][ibef])
				a, e, inclination, Omega, arg_periapsis, dt_peri = keplerian_orbital_elements(m, m2, cx[inghbr][ibef], cv[inghbr][ibef], G=G)
				fact = 0.45/(4.5e-6)
				print(a*fact, e, inclination, Omega, arg_periapsis)
				if e<1:
					state_vectors = np.array([binary_state(a, e, inclination, Omega, arg_periapsis, m, m2, t, t0+dt_peri) for t in t_arr])
					#state_vectors = np.array([keplerian_state_vector(a, e,1e-4, 1e-4, 1e-4, m, m2, t, t0+dt_peri) for t in t_arr])
					
					cx_alt[inghbr][stab], cv_alt[inghbr][stab] = np.swapaxes(state_vectors,0,1)


	return cx_alt, cv_alt, cm_alt


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
	M =  E - e*np.sin(E)

	print(E, e, np.sinh(E))

	#Finally, get periastron itme
	T_peri = T - np.sign(add)*M / n


	return e.flatten(), rp.flatten(), T_peri.flatten()

def encounter_params(cx, cv, cm, ct, mstar, G=1.0):
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
					if cxmag[ix][ixmag]< cxmag[ix][ixmag-1]:
						lm.append(ixmag)
					else:
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
			print(e, rp, dtperi, ct)
			for ienc, dt in enumerate(dts):
				if (np.absolute(dtperi[ienc])<np.absolute(dt)):
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
