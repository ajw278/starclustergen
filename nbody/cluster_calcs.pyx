from libc.math cimport sqrt
import numpy as np
cimport numpy as np

#import encounter_mc_v2
import sys
import scipy.spatial as spatial


def com_calc(np.ndarray[double, ndim=2] rvstars, np.ndarray[double, ndim=1] mstars):
	rvx, rvy, rvz = np.swapaxes(rvstars,0,1)

	com_x = np.sum(rvx*mstars)/np.sum(mstars)
	com_y = np.sum(rvy*mstars)/np.sum(mstars)
	com_z = np.sum(rvz*mstars)/np.sum(mstars)

	return np.array([com_x, com_y, com_z])

def adjust_com(np.ndarray[double, ndim=2] rvstars, np.ndarray[double, ndim=1] mstars):

	cdef int istar

	com = com_calc(rvstars, mstars)
	
	for istar in range(len(mstars)):
		rvstars[istar] -= com
		
	maxrv = np.amax(rvstars)

	check_com = com_calc(rvstars, mstars)

	for check in check_com:
		if np.absolute(check/maxrv)>1e-4:
			print 'Error in com calc'
			print check
			print np.absolute(check/maxrv)
			print com
			print check_com
			sys.exit() 

	return rvstars

def Morans_I_vcorr(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=2] vstars, float dlim, float signoise):
	
	Ndat = len(rstars)

	v_x, v_y = np.swapaxes(vstars,0,1)[:2]
	
	if signoise>0.:
		vxadd = np.random.normal(loc=0.0, scale=signoise, size=len(v_x))
		vyadd = np.random.normal(loc=0.0, scale=signoise, size=len(v_y))
		v_x += vxadd
		v_y += vyadd

	mv_x = np.mean(v_x)
	mv_y = np.mean(v_y) 
	rproj = np.swapaxes(np.swapaxes(rstars,0,1)[:2],0,1)
	Numx = 0.
	Numy = 0.
	Denomx1 = np.sum((v_x-mv_x)**2.)
	Denomy1 = np.sum((v_y-mv_y)**2.)
	Denom2 = 0.0
	
	sumdist = 0.
	for istar in range(Ndat):
		rproj_tmp = np.delete(rproj, istar, axis=0)
		v_x_tmp =  np.delete(v_x, istar, axis=0)
		v_y_tmp =  np.delete(v_y, istar, axis=0)
		dist_ij = np.apply_along_axis(np.linalg.norm, 1, rproj_tmp-rproj[istar])
		subind = np.where(dist_ij>dlim)[0]
		dist_ij = dist_ij[subind]
		v_x_tmp = v_x_tmp[subind]
		v_y_tmp = v_y_tmp[subind]
		w_ij = np.power(dist_ij,-1.)
		
		Numx += (v_x[istar]-mv_x)*np.sum(w_ij*(v_x_tmp-mv_x))
		Numy += (v_y[istar]-mv_y)*np.sum(w_ij*(v_y_tmp-mv_y))
		
		Denom2 += np.sum(w_ij)
	
	I_x = float(Ndat)*Numx/(Denomx1*Denom2)
	I_y = float(Ndat)*Numy/(Denomy1*Denom2)
	
	return I_x, I_y
	
	

def empirical_centre(np.ndarray[double, ndim=2] rstars, float radius, int ndim, int nres, int niter):

	rproj = np.swapaxes(np.swapaxes(rstars, 0,1)[:ndim],0,1)
	rmax = np.amax(rproj)


	centre = np.zeros(ndim)
	
	for iiter in range(niter):
		gridvals = np.linspace(-rmax, rmax, nres)
		gridvals_x = gridvals +centre[0]
		gridvals_y = gridvals +centre[1]
		pop = np.zeros((nres,nres))
		if ndim==3:
			gridvals_z = gridvals +centre[2]
			pop = np.zeros((nres,nres,nres))


		for ix in range(nres):
			for iy in range(nres):
				if ndim==2:
					rptmp = rproj - np.array([gridvals_x[ix], gridvals_y[iy]])
					rpmag = np.apply_along_axis(np.linalg.norm, 1, rptmp)
					inrad = np.where(rpmag<radius)[0]
					pop[ix][iy] = len(inrad)
				else:
					for iz in range(nres):
						rptmp = rproj - np.array([gridvals_x[ix], gridvals_y[iy], gridvals_z[iz]])
						rpmag = np.apply_along_axis(np.linalg.norm, 1, rptmp)
						inrad = np.where(rpmag<radius)[0]
						pop[ix][iy][iz] = len(inrad)


		maxind = np.unravel_index(np.argmax(pop, axis=None), pop.shape)
		if ndim==2:
			centre = np.array([gridvals_x[maxind[0]], gridvals_y[maxind[1]]])
		else:
			centre = np.array([gridvals_x[maxind[0]], gridvals_y[maxind[1]], gridvals_z[maxind[2]]])
			
		
		rmax = gridvals[1]-gridvals[0]

	return centre


def single_energy(np.ndarray[double, ndim=1] rrel, np.ndarray[double, ndim=1] vrel, double m1, double m2):
	
	cdef double vsq, rsq, E_tot, mu

	mu = m1*m2/(m1+m2)

	vsq = np.dot(vrel, vrel)
	rsq = np.dot(rrel, rrel)

	E_tot = 0.5*mu*vsq - m1*m2/np.sqrt(rsq)

	return E_tot

def binary_distances(np.ndarray[double, ndim=2] rstars):

	bins = np.zeros(len(rstars), dtype=int)
	dists = np.zeros(len(bins))
	idrmin_srt = np.zeros(len(bins))

	for istar in range(len(rstars)):
		drmags = np.apply_along_axis(np.linalg.norm, 1, rstars-rstars[istar])
		idrmin_srt = np.argsort(drmags)
		dists[istar] = drmags[idrmin_srt[1]]
		bins[istar] = int(idrmin_srt[1])
		
	return bins, dists


def vdisp(np.ndarray[double, ndim=2] vstars, np.ndarray[double, ndim=1] mstars):
	#Using the method in P. Girichidis et al. (2012)
	cdef double mean_vx, mean_vy, mean_vz, mtot
	cdef double sigmasq_x, sigmasq_y, sigmasq_z
	cdef double sigma_3D
	
	mtot = np.sum(mstars)

	vx, vy, vz  = np.swapaxes(vstars,0,1)
	mean_vx = np.sum(mstars*vx)/mtot
	mean_vy = np.sum(mstars*vy)/mtot
	mean_vz = np.sum(mstars*vz)/mtot

	
	sigmasq_x = np.sum(mstars*np.power(vx-mean_vx, 2))/mtot
	sigmasq_y = np.sum(mstars*np.power(vy-mean_vy, 2))/mtot
	sigmasq_z = np.sum(mstars*np.power(vz-mean_vz, 2))/mtot

	sigma_3D = np.sqrt(np.sum(np.array([sigmasq_x, sigmasq_y, sigmasq_z])))

	return sigma_3D

def single_potential(int istar, np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars):
	cdef double gpot, drsq_tmp
	
	dr = np.apply_along_axis(np.linalg.norm, 1, rstars - rstars[istar])
	noti = np.where(dr>1e-10)[0]
	gpot = mstars[istar]*np.sum(mstars[noti]/dr[noti])

	return gpot

def pos_potential(np.ndarray[double, ndim=1] starpos, np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars):
	cdef int jstar
	cdef double gpot, drsq_tmp
	
	gpot = 0.0
	dr = rstars - starpos


	for jstar in range(len(rstars)):
		drsq_tmp = np.dot(rstars[jstar], rstars[jstar])
		if drsq_tmp>1e-5:
			gpot += -mstars[jstar]/np.sqrt(drsq_tmp)

	return gpot

def single_gacc(int istar, np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars):
	cdef int jstar
	cdef double drsq_tmp

	gacc = np.zeros(rstars.shape[1])
	
	dr = rstars - rstars[istar]


	for jstar in range(len(rstars)):
		if jstar!=istar:
			drsq_tmp = np.dot(rstars[jstar], rstars[jstar])
			gacc += dr[jstar]*mstars[jstar]/np.power(drsq_tmp, 3./2.)

	return gacc


def single_flux(np.ndarray[double, ndim=1] dr_arr, np.ndarray[double, ndim=1] Lstars):
		
	Linds = np.where(dr_arr>1e-15)[0]
	tot_flux = Lstars[Linds]/(4.*np.pi*dr_arr[Linds]*dr_arr[Linds])
	return tot_flux, Linds

def plummer_density(np.ndarray[double, ndim=1] r, float mplummer, float aplummer):
	return 3.*mplummer*np.power(1+(r*r/(aplummer*aplummer)), -5./2.)/(4.*np.pi*aplummer)


def UVextinction(np.ndarray[double, ndim=1] r1, np.ndarray[double, ndim=2] rLums, float mplummer, float aplummer):
	
	fluxes_factors = np.zeros(len(rLums))	
	delta_mags = np.zeros(len(rLums))

	#(AUV/AV) = 2.7
	#1/(AV/NH) = 1/1.8e21
	#(Msol/mH) = 1.19e57
	#(cm/pc)^2 = 1.05e-37
	#Whole factor (product the above): 0.1872

	factor = 0.1872

	uvals = np.linspace(0.,1., 20)
	muvals = 1.-uvals
	drs = np.apply_along_axis(np.linalg.norm, 1, rLums-r1)
	
	for iL in range(len(rLums)):
		rsp = np.dot(muvals[:, None], rLums[iL][None,:]) +np.dot(uvals[:, None],r1[None,:])
		rvals = np.apply_along_axis(np.linalg.norm, 1,rsp)
		delta_mags[iL] = np.absolute(np.trapz(plummer_density(rvals, mplummer, aplummer)*uvals*drs[iL], uvals*drs[iL]))
		

	delta_mags *= factor

	flux_factors = np.power(10., -delta_mags/2.5)

	return flux_factors

def flux_wext(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] Lstars, np.ndarray[double, ndim=2] rLstars, float mplummer, float aplummer):
	
	fstars = np.zeros(len(rstars))

	m2pc = 3.24078e-17

	for istar in range(len(rstars)):
		dr = np.apply_along_axis(np.linalg.norm, 1, rLstars-rstars[istar])
		isub = np.where(dr>1e-10)[0]
		drsub = dr[isub]
		Ltmp = Lstars[isub]
		rLtmp = rLstars[isub]
		fltmp, linds = single_flux(drsub*1e2/m2pc, Ltmp)
		if mplummer>1.:
			ffact = UVextinction(rstars[istar], rLtmp[linds], mplummer, aplummer)
		else:
			ffact = np.ones(len(fltmp))
		fstars[istar] = max(np.sum(fltmp*ffact), 1e-10)
		if (istar+1)%100==0 and mplummer>1.:
			print 'Min/max/mean {0}/{1}: {2} {3} {4}'.format(istar+1, len(rstars), np.amin(fstars[:istar+1]), np.amax(fstars[:istar+1]), np.mean(fstars[:istar+1]))

	return fstars

def flux_wext_tseries(np.ndarray[double, ndim=3] rstars, np.ndarray[double, ndim=1] Lstars, np.ndarray[double, ndim=3] rLstars,  np.ndarray[double, ndim=1] mplummer, np.ndarray[double, ndim=1] aplummer):


	fstars = np.zeros((rstars.shape[0],rstars.shape[1]))
	
	for it in range(len(rstars)):
		print '{0}/{1}'.format(it, len(rstars))
		fstars[it] = flux_wext(rstars[it],  Lstars, rLstars[it],  float(mplummer[it]),  float(aplummer[it]))
		

	return fstars


def flux(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] Lstars, np.ndarray[double, ndim=2] rLstars, int dim):
	
	rstars_proj = np.swapaxes(np.swapaxes(rstars,0,1)[:dim],0,1)
	rLstars_proj = np.swapaxes(np.swapaxes(rLstars,0,1)[:dim],0,1)
	
	fstars = np.zeros(len(rstars))

	for istar in range(len(rstars)):
		dr = np.apply_along_axis(np.linalg.norm, 1, rLstars_proj-rstars_proj[istar])
		fstars[istar] = np.sum(single_flux(dr, Lstars)[0])


	return fstars
		

def massbins(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars, int res):
	
	rstars = adjust_com(rstars, mstars)

	
	rmag = np.apply_along_axis(np.linalg.norm, 1,rstars)
	if len(rmag)<=3:
		print('Massbins norm error.')
		sys.exit()

	rbins = np.linspace(0.0,1.00000001*np.amax(rmag), res+1)

	mbins = np.zeros(res)
	
	for ir in range(res):
		xbool = rmag>=rbins[ir]
		ybool = rmag<rbins[ir+1]
		mbins[ir] = np.sum(mstars[np.where(np.logical_and(xbool,ybool))])


	return mbins, rbins


def mdensity(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars, int res):

	rstars = adjust_com(rstars, mstars)

	mbins, rbins = massbins(rstars, mstars, res)
	rvals = (rbins[1:]-rbins[:-1])/2.
	shell_vols = (4.*np.pi/3.)*(np.power(rbins[1:], 3) - np.power(rbins[:-1], 3))
	mdensity = mbins/shell_vols

	return mdensity, rvals

def plummer_pot(float rmag, float mplummer, float aplummer):
	return -mplummer/(aplummer*np.sqrt(1.+(rmag/aplummer)**2))

def gas_potential(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars, float mplummer, float aplummer):
	cdef int istar
	cdef double gpot_tot 
	
	gpot_tot = 0.0
	rmags = np.apply_along_axis(np.linalg.norm, 1, rstars)
	
	
	for istar in range(len(rstars)):
		gpot_tot += np.absolute(mstars[istar]*plummer_pot(rmags[istar], mplummer, aplummer))
	
	return gpot_tot

def stellar_potential(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars):
	cdef int istar
	cdef double gpot_tot 
	
	gpot_tot = 0.0
	rmags = np.apply_along_axis(np.linalg.norm, 1, rstars)
	
	
	for istar in range(len(rstars)):
		gpot_tot += single_potential(0, rstars[istar:], mstars[istar:])
	
	return gpot_tot


#This function is for the unupdated version of the potential calculation in Nbody6++
"""def total_potential_nbody6_wgas(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars, float mplummer, float aplummer):
	cdef int istar
	cdef double gpot_tot 
	
	gpot_tot = 0.0
	rmags = np.linalg.norm(rstars, axis=1)
	
	
	def plummer_pot_nbody6(float rmag, float mplummer, float aplummer):
		return -mplummer*rmag*rmag/np.power(aplummer**2+rmag**2, 3./2.)

	for istar in range(len(rstars)):
		gpot_tot += 0.5*np.absolute(single_potential(istar, rstars, mstars))
		gpot_tot += np.absolute(mstars[istar]*plummer_pot_nbody6(rmags[istar], mplummer, aplummer))
	
	return gpot_tot"""


def minimum_sep(np.ndarray[double, ndim=1] r, np.ndarray[double, ndim=2] rcomp):
	dr = rcomp-r
	drmag = np.apply_along_axis(np.linalg.norm, 1, dr)
	imin = np.argmin(drmag)
	
	return drmag[imin], imin

	
def mst(np.ndarray[double, ndim=2] rstars, int ndim):

	def compile_list(np.ndarray[double, ndim=2] rs, np.ndarray[double, ndim=2] rcomp):
		dist = []
		inds = []
		for ir in range(len(rs)):
			dtmp, itmp = minimum_sep(rs[ir], rcomp)
			dist.append(dtmp)
			inds.append(itmp)
		
		return np.array(dist), np.array(inds)
			

	
	rproj = np.swapaxes(np.swapaxes(rstars, 0, 1)[:ndim], 0,1)


	mst_inds = [] #np.zeros(len(rstars)-1, dtype=int)
	out_inds = np.arange(len(rstars))

	start = 0
	mst_inds.append(start)

	line_inds = np.zeros((len(rproj),2), dtype=int)
	
	
	for im in range(len(rproj)):
		#rkdtree = spatial.KDTree(rproj[out_inds])
		#dist, inds = rkdtree.query(np.array(rproj[mst_inds]))
		dist, inds = compile_list(rproj[mst_inds], rproj[out_inds])
		
		mindist = np.argmin(dist)
		minind = inds[mindist]
		line_inds[im] = np.array([mst_inds[mindist], out_inds[minind]],dtype=int)
		
		mst_inds.append(out_inds[minind])
		out_inds = np.delete(out_inds, minind)

	return line_inds

def mst_side_lengths_all(np.ndarray[long, ndim=2] line_inds, np.ndarray[double, ndim=2] rstars, int ndim):
	rproj = np.swapaxes(np.swapaxes(rstars, 0, 1)[:ndim], 0,1)
	
	nlines = float(len(line_inds))
	lentot = 0.0  
	sls = np.zeros(len(line_inds))
	for im in range(len(line_inds)):
		rvals = rproj[line_inds[im]]
		dr = np.linalg.norm(rvals[0]-rvals[1])
		sls[im] = dr

	return sls

def mst_side_length(np.ndarray[long, ndim=2] line_inds, np.ndarray[double, ndim=2] rstars, int ndim):
	rproj = np.swapaxes(np.swapaxes(rstars, 0, 1)[:ndim], 0,1)
	
	nlines = float(len(line_inds))
	lentot = 0.0  
	
	for im in range(len(line_inds)):
		rvals = rproj[line_inds[im]]
		dr = np.linalg.norm(rvals[0]-rvals[1])
		lentot+=dr

	"""print 'MST length'
	print lentot
	print 'Number lines'"""
	#print nlines
	

	return lentot/nlines

def avg_sep(np.ndarray[double, ndim=2] rstars, int ndim, int nsamp):
	
	rproj = np.swapaxes(np.swapaxes(rstars, 0, 1)[:ndim], 0,1)
	
	rproj_sub = rproj[np.random.choice(np.arange(len(rproj)), size= min(nsamp, len(rproj)), replace=False)]

	sbar = np.zeros(nsamp)

	for ir in range(min(nsamp, len(rproj))):
		dr = rproj-rproj_sub[ir]
		drmag = np.apply_along_axis(np.linalg.norm, 1, dr)
		sbar[ir] = np.mean(drmag)

	print 'Mean separation'
	print np.mean(sbar)

	return np.mean(sbar)

def Qvalue(double sbar, double mbar, double Rclust, double Nclust):
	
	area = np.pi*Rclust*Rclust
	
	normfactor = np.sqrt(Nclust*area)/(Nclust-1.)

	mnorm = mbar/normfactor

	snorm = sbar/Rclust

	
	return mnorm/snorm
	

def lambda_mseg(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars, double mseg, int ndim, int nsamp):
	

	ibig = np.where(mstars>mseg)[0]
	ismall = np.where(mstars<mseg)[0]


	
	lbig = 0.0

	print('Split:', len(ibig), len(ismall))
	ntries = 40

	allbig = np.zeros(ntries)

	for i in range(ntries):
		subset = np.random.choice(ibig, size= nsamp, replace=False)
		linds = mst(rstars[subset], ndim)
		slength = mst_side_length(linds, rstars[subset], ndim)
		allbig[i] =slength
	

	lbig = np.mean(allbig)
	print('Error1:', np.std(allbig))


	lsmall = 0.0
	ntries = 40
	allsmall = np.zeros(ntries)
	for i in range(ntries):
		subset = np.random.choice(ismall, size= nsamp, replace=False)
		linds = mst(rstars[subset], ndim)
		slength = mst_side_length(linds, rstars[subset], ndim)
		allsmall[i] =slength
	

	lsmall =np.mean(allsmall)
	print('Error2:', np.std(allsmall))

	print 'big small'
	print lbig
	print lsmall

	return lsmall/lbig
	
	

def isotropic_potential(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=1] mstars):
	
	gpot = np.zeros(len(mstars))

	rstars = adjust_com(rstars, mstars)

	mbins, rbins = massbins(rstars, mstars, 50)
	rvals = (rbins[1:]+rbins[:-1])/2.
	menc = np.cumsum(mbins)
	gpot = np.trapz(menc*mbins/rvals, rvals)

	return gpot

def local_ndensity(np.ndarray[double, ndim=2] rstars, int idense, int ndim):

	rproj = np.swapaxes(np.swapaxes(rstars, 0,1)[:ndim],0,1)
	
	nstars = np.zeros(len(rstars))

	for istar in range(len(rstars)):
		dr = np.apply_along_axis(np.linalg.norm, 1, rproj-rproj[istar])
		rsplit = np.sort(dr)[idense]
		if ndim==3:
			volsplit = 4.*np.pi*rsplit*rsplit*rsplit/3.
		if ndim==2:
			volsplit = np.pi*rsplit*rsplit
			
		nstars[istar] = float(idense-1)/volsplit
	
	return nstars


def total_kinetic(np.ndarray[double, ndim=2] vstars, np.ndarray[double, ndim=1] mstars):

	cdef double ke
	
	#vstars = adjust_com(vstars, mstars)
	vswap = np.swapaxes(vstars, 0,1)
	ke = np.sum(0.5*mstars*(vswap[0]*vswap[0]+vswap[1]*vswap[1]+vswap[2]*vswap[2]))

	return ke

def find_partners(int istar, np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=2] vstars, np.ndarray[double, ndim=1] mstars):
	
	cdef int jstar, ibound, etot
	cdef double mmass
	cdef list bound 
	bound = []
	
	dr = rstars-rstars[istar]
	dv = vstars-vstars[istar]
	
	drmag = np.apply_along_axis(np.linalg.norm, 1, dr)
	if len(drmag)<=3:
		print 'Norm error for drmag'
		sys.exit()

	argclose = np.argsort(drmag)[1:]

	ibound=0
	
	mmass = mstars[istar]
	mcom = dr[istar]
	mcov = dv[istar]

	for jstar in argclose:
		#print(dv[jstar]-mcov)
		#print drmag, rstars, mstars
		etot = single_energy(dr[jstar] - mcom, dv[jstar]-mcov, mmass, mstars[jstar])
		if etot<0.0:
			mcom = mcom*mmass+dr[jstar]*mstars[jstar]
			mcov= mcov*mmass+dv[jstar]*mstars[jstar]
			mmass += mstars[jstar]
			bound.append(jstar)
		else:
			break


	return bound
	
def multiplicity(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=2] vstars, np.ndarray[double, ndim=1] mstars):
	
	cdef int istar 
	cdef list bound_all

	bound_all = []
	nbound = np.zeros(len(rstars))
	
	for istar in range(len(rstars)):
		bound_all.append(find_partners(istar, rstars, vstars, mstars))

		nbound[istar] = len(bound_all[istar])

	return nbound, bound_all





def encounter_history(np.ndarray[double, ndim=3] rstars, np.ndarray[double, ndim=3] vstars, np.ndarray[double, ndim=1] mstars, int nclose):

	cdef int itime, istar

	
	#Want rstars in form [star][time][dimension]
	rstars = np.swapaxes(rstars, 0,1)
	vstars = np.swapaxes(vstars, 0,1)

	nghbr_list = []

	nghbrs = np.zeros((rstars.shape[0], rstars.shape[1], nclose))
	closest_x = np.zeros((rstars.shape[0], rstars.shape[1], nclose, rstars.shape[2]))
	closest_v = np.zeros((rstars.shape[0], rstars.shape[1], nclose, rstars.shape[2]))
	closest_m = np.zeros((rstars.shape[0], rstars.shape[1], nclose))
	
	cx = []
	cv = []
	cm = []

	nblst_all = []
	
	
	for istar in range(len(mstars)):

		cx.append([])
		cv.append([])
		cm.append([])


		drstars = rstars - rstars[istar]
		dvstars = vstars - vstars[istar]
		drmags = np.apply_along_axis(np.linalg.norm, 2, drstars)

		drstars = np.swapaxes(drstars, 0,1)
		dvstars = np.swapaxes(dvstars, 0,1)
		drmags = np.swapaxes(drmags, 0,1)

		nblst = []
		
		
		for itime in range(len(drmags)):
			ninds = np.argsort(drmags[itime])[1:nclose+1]
			for nb in ninds:
				if not (nb in nblst):
					nblst.append(nb)
			nghbrs[istar][itime] = ninds
			closest_x[istar][itime] = drstars[itime][ninds]
			closest_v[istar][itime] = dvstars[itime][ninds]
			closest_m[istar][itime] = mstars[ninds]


		for nb in nblst:
			#[star][time][dimension] -> time, star, dim
			cxval = np.swapaxes(drstars,0,1)[nb]
			cx[istar].append(cxval)
			cv[istar].append(np.swapaxes(dvstars,0,1)[nb])
			cm[istar].append(mstars[nb])

		nblst_all.append(nblst)
			
	return cx, cv, cm, nblst_all #closest_x, closest_v, closest_m, nghbrs



def encounter_history_istar(int istar, np.ndarray[double, ndim=3] rstars, np.ndarray[double, ndim=3] vstars, np.ndarray[double, ndim=1] mstars, int nclose):

	cdef int itime
	
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
	drmags = np.apply_along_axis(np.linalg.norm, 2, drstars)

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

	print 'neighbours', nblst

	for nb in nblst:
		#[star][time][dimension] -> time, star, dim
		cxval = np.swapaxes(drstars,0,1)[nb]
		cx.append(cxval)
		cv.append(np.swapaxes(dvstars,0,1)[nb])
		cm.append(mstars[nb])

	cx = np.array(cx)
	cv = np.array(cv)
	
			
	return cx, cv, cm, nblst #closest_x, closest_v, closest_m, nghbrs


def separation_history(int iflux, np.ndarray[double, ndim=3] rstars,  np.ndarray[double, ndim=1] mstars):

	cdef int itime, istar

	rstars = np.swapaxes(rstars, 0,1)
	

	drstars = rstars - rstars[iflux]
	drstars_mag = np.linalg.norm(drstars, axis=2)

	
	return drstars_mag


def encounter_params(np.ndarray[double, ndim=3] cx, np.ndarray[double, ndim=3] cv, np.ndarray[double, ndim=1] cm, np.ndarray[double, ndim=1] ct, float mstar):
	cdef int idt

	diff = np.amin(np.diff(ct))

	ce_eccs = []
	ce_x = []
	ce_time = []
	
	for ix in range(len(cx)):
		cxmag = np.apply_along_axis(np.linalg.norm, 1, cx[ix])
		cvmag = np.apply_along_axis(np.linalg.norm, 1, cv[ix])
		local_minima = []
		for ixmag in range(1,len(cxmag)-1):
			if cxmag[ixmag-1]>cxmag[ixmag] and cxmag[ixmag+1]>cxmag[ixmag]:
				local_minima.append(ixmag)

	 	#local_minima = np.array(local_minima)
		hs = np.cross(cx[ix][local_minima], cv[ix][local_minima])
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
				print 'cxmag', cxmag[ivec]
				print 'cxest', xmins[idt]
				print smas[idt] 
				print eccs[idt]
				print vr
				print 'dt', dt
				print 't', ct[ivec]
			idt+=1
		ce_eccs.append(eccs)
		ce_x.append(xmins)

		ce_time.append(np.array(times))
		
		
			
			
	return ce_x, ce_eccs, ce_time



"""def disc_evol(np.ndarray[double, ndim=1] xmins, np.ndarray[double, ndim=1] eccs,np.ndarray[double, ndim=1] mperts, np.ndarray[double, ndim=1] times, double Rinit, double mstar, double xthresh, double ethresh):
	rout = np.ones(len(xmins))*Rinit
	rfunc = encounter_mc_v2.r_func('model.npy', grid=False)

	for ix in range(len(xmins)):
		if xmins[ix]<xthresh:
			if eccs[ix]>ethresh : 
				eccenc = max(1.0, eccs[ix])
				rout[ix:] = min(xmins[ix]*rfunc(xmins[ix]/rout[ix],mperts[ix]/mstar,eccenc),rout[ix])
			
			else:
				return None

	return rout"""

	
	

	
	


def fit_king(np.ndarray[double, ndim=2] rstars, np.ndarray[double, ndim=2] vstars, np.ndarray[double, ndim=1] mstars):

	pot_es = np.zeros(len(mstars))
	k_es = np.zeros(len(mstars))

	#vstars_adj = vstars #adjust_com(vstars, mstars)
	vstars_adj_mag = np.apply_along_axis(np.linalg.norm, 1, vstars)

	rs = np.apply_along_axis(np.linalg.norm, 1, rstars)

	k_es  = 0.5*np.power(vstars_adj_mag, 2)

	for istar in range(len(mstars)):
		pot_es[istar] = single_potential(istar, rstars, mstars)/mstars[istar]

	tot_es = k_es+pot_es

	gdense, bin_edges = np.histogram(tot_es, bins=10, weights = mstars)




	binspace = np.append(np.array([0.0]), np.logspace(-2.,np.log10(min(4.0, np.amax(rs))), 10))

	rho, binr_edges = np.histogram(rs, bins=binspace, weights=mstars)

	rho = np.array(rho, dtype=np.float64)

	ndense, binn_edges = np.histogram(rs, bins=binspace)

	ndense = np.array(ndense, dtype=np.float64)


	return gdense, bin_edges, rho, binr_edges, ndense, binn_edges

