
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

def encounter_history_istar(istar, rstars, vstars,  mstars, nclose=3):
	
	#Want rstars in form [star][time][dimension]
	rstars = np.swapaxes(rstars, 0,1)
	vstars = np.swapaxes(vstars, 0,1)

	nghbr_list = []

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


def encounter_params(cx, cv, cm, ct, mstar):
	diff = np.amin(np.diff(ct))

	ce_eccs = []
	ce_x = []
	ce_time = []

	import matplotlib.pyplot as plt
	
	for ix in range(len(cx)):
		cxmag = np.apply_along_axis(np.linalg.norm, 1, cx[ix])
		cvmag = np.apply_along_axis(np.linalg.norm, 1, cv[ix])
		cvdotx = np.einsum('ij,ij->i', cx[ix], cv[ix])
		print(cvdotx.shape)
		local_minima = []
		for ixmag in range(1,len(cxmag)-1):
			if cvdotx[ixmag-1]<0.0 and cvdotx[ixmag]>0.0:
				local_minima.append(ixmag-1)

		
	 	#local_minima = np.array(local_minima)
		hs = np.cross(cx[ix][local_minima], cv[ix][local_minima])
		if len(local_minima)>0:
			hsmag= np.apply_along_axis(np.linalg.norm, 1, hs)
			mu = cm[ix]+mstar
			ls = hsmag*hsmag/mu
			smas = 1./((2./cxmag[local_minima])-(np.power(cvmag[local_minima],2)/mu))
			eccs = np.sqrt(1.-hsmag*hsmag/(smas*mu))
			xmins = smas*(1.-eccs)


			times = []
			idt=0
			for ivec in local_minima:
				vr = -np.dot(cx[ix][ivec], cv[ix][ivec])/cxmag[ivec]
				dt = (cxmag[ivec]-xmins[idt])/vr
				if np.absolute(dt)>diff/2.:
					dt = 0.0
					xmins[idt] = cxmag[ivec]

				times.append(ct[ivec]+dt)
				if times[idt]<0.0:
					print('cxmag', cxmag[ivec])
					print('cxest', xmins[idt])
					print(smas[idt] )
					print(eccs[idt])
					print(vr)
					print('dt', dt)
					print('t', ct[ivec])
				idt+=1
			ce_eccs.append(eccs)
			ce_x.append(xmins)

			ce_time.append(np.array(times))
		else:
			
			ce_eccs.append(np.array([]))
			ce_x.append(np.array([]))

			ce_time.append(np.array([]))
			
		
	
	return ce_x, ce_eccs, ce_time
