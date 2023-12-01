
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
