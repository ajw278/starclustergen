from __future__ import print_function

import os
import sys

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(scriptdir)
sys.path.append(scriptdir+'../general')

from common import *


import time
import numpy as np
import scipy.interpolate as interpolate





def boltz_pdf(v, g=1.):
	return v*v*np.exp(-1.*np.power(2., 2.*g)*v*v)


def IMF_pdf(m):
	S0 = 0.08
	S1 = 0.5
	S2 = 1.0

	P1 = -1.3
	P2 = -2.2
	P3 = -2.7

	F1 = 0.035
	F2 = F1*np.power(S1, P1)/np.power(S1, P2)
	F3 = F2*np.power(S2, P2)/np.power(S2, P3)

	if m<=S0:
		return 0.
	elif m>S0 and m<S1:
		return F1*np.power(m, P1)
	elif m>=S1 and m<S2:
		return F2*np.power(m, P2)
	elif m>=S2:
		return F3*np.power(m, P3)


def IMF_pdf_np(m, mmax=50.0, pl3=2.7):
	S0 = 0.08
	S1 = 0.5
	S2 = 1.0

	P1 = -1.3
	P2 = -2.2
	P3 = -1.*np.absolute(pl3)

	F1 = 0.035
	F2 = F1*np.power(S1, P1)/np.power(S1, P2)
	F3 = F2*np.power(S2, P2)/np.power(S2, P3)

	imf = np.zeros(len(m))

	imf[(m > S0) & (m <= S1)] = F1*np.power(m[(m > S0) & (m <= S1)], P1)
	imf[(m > S1) & (m <= S2)] = F2*np.power(m[(m > S1) & (m <= S2)], P2)
	imf[m > S2] = F3*np.power(m[m > S2], P3)
	imf[m>mmax] = 0.0

	return imf


#Function to calculate the maximum mass probability distribution for a given cluster of size nclust
def pmmax(nclust, mrange=None, res=None):


	if mrange==None:
		mrange = [0.08, 100.*np.log(2.16*nclust/1e2)]
	if res==None:
		res = max(int(200.*np.log(2.16*nclust/1e2)),400)
	massarray = np.logspace(np.log10(mrange[0]), np.log10(mrange[1]),res)

	imf_arr = IMF_pdf_np(massarray, mmax=mrange[1])
	imf_arr /= np.trapz(imf_arr, massarray)

	integral_arr = np.zeros(len(imf_arr))

	for ip in range(len(imf_arr)):
		integral_arr[ip] = np.trapz(imf_arr[:ip], massarray[:ip])


	integral_arr /= integral_arr[-1]
	
	pmmass_arr =nclust*np.power(integral_arr, nclust-1)*imf_arr

	

	mmass_mean = np.trapz(massarray*pmmass_arr, massarray)
	int_pmmass_arr = np.zeros(len(massarray))
	for ip in range(len(massarray)):
		int_pmmass_arr[ip] = np.trapz(pmmass_arr[:ip], massarray[:ip])
	inv_int_pmmass_func = interpolate.interp1d(int_pmmass_arr, massarray)

	mmass_med = inv_int_pmmass_func(0.5)
	sig_low = inv_int_pmmass_func(1./6.)
	sig_high = inv_int_pmmass_func(5./6.)


	return mmass_mean, mmass_med, sig_low, sig_high


def get_vstars(nstars, g=1.):
	vspace = np.linspace(0.0, 10.0, 5000)
	imf_vals_aux = np.zeros(len(vspace))

	for im in range(len(vspace)):
		imf_vals_aux[im] = np.trapz(boltz_pdf(vspace[:im], g=g), vspace[:im])


	imf_vals_aux = imf_vals_aux/imf_vals_aux[-1]
	inverse_cdf = interpolate.interp1d(imf_vals_aux, vspace)


	Rands  =  np.random.rand(nstars)	

	vstars= inverse_cdf(Rands)
	
	return vstars




def get_mstars(nstars, mmax=50.0, mmin=0.08, pl3=2.7):
	if mmax==None:
		mmax=100.0
	if mmin==None:
		mmin=0.08

	mspace = np.logspace(np.log10(mmin),np.log10(mmax), 5000)
	imf_vals_aux = np.zeros(len(mspace))

	for im in range(len(mspace)):
		imf_vals_aux[im] = np.trapz(IMF_pdf_np(mspace[:im], mmax=mmax, pl3=pl3), mspace[:im])


	imf_vals_aux = imf_vals_aux/imf_vals_aux[-1]
	inverse_cdf = interpolate.interp1d(imf_vals_aux, mspace)
	Rands  =  np.random.rand(nstars)

	mstars= inverse_cdf(Rands)

	return mstars


def random_dir(magnitude):
	vector = 1.-2.*np.random.random(3)
	vector /= np.linalg.norm(vector)
	return vector*magnitude


def fractal_grid(Nc, D, correlated=True, Ninit=1):

	Ngens = np.log(2.*Nc)/np.log(8.) +1.

	if D<=2.:
		Ngens+=1.

	Ngens = int(Ngens+0.5)
	parent_prob = np.power(2., D-3.)

	tree = []
	tree_vel = []
	tree_ass = []
	g=0
	Npoints = 0
	drct = np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.],[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.],[-1.,-1.,-1.]])
	edge = 1.0 - 1./float(Ninit)/2.
	side_pts = np.arange(-edge*Ninit, edge*Ninit+1.)/Ninit
	drct_init = np.array(np.meshgrid(side_pts, side_pts, side_pts)).T.reshape(-1,3)

	
	tree_ass.append(np.arange(len(drct_init)))

	while Npoints<Nc*2.:
		if correlated:
			vmags = get_vstars(int(((2.*Ninit)**3.)*np.power(2,(g+1)*3))+1, g=g)
		else:
			vmags =  get_vstars(int(((2.*Ninit)**3.)*np.power(2, (g+1)*3))+1, g=1)

		print('Generation {0}'.format(g))
		g+=1
		disp_factor= np.power(2., -g-1)
		tree.append([])
		tree_vel.append([])
		if g==1:
			for dr in drct_init:
				mag = disp_factor*np.random.random()
				tree[g-1].append(dr+disp_factor*random_dir(mag/float(int(Ninit))))
				vind = np.random.randint(0, len(vmags))
				tree_vel[g-1].append(random_dir(vmags[vind]))
				vmags = np.delete(vmags, vind)
				Npoints+=1

		elif g>1:
			t0 = time.time()
			tree_ass.append([])
			us = np.random.random(len(tree[g-2]))
			uinds = np.where(us<parent_prob)[0]
			mags = disp_factor*np.random.random((len(uinds),len(drct)))
			vind_all = np.random.choice(np.arange(len(vmags)), size=(len(uinds), len(drct)), replace=False)
			print('Number of parents: {0}/{1}'.format(len(uinds), len(us)))
			iu = 0
			for ibr in uinds:
				idr=0
				for dr in drct:
					mag = mags[iu][idr]
					tree[g-1].append(tree[g-2][ibr]+disp_factor*dr*2.+random_dir(mag/float(int(Ninit))))
					tree_ass[g-1].append(tree_ass[g-2][ibr])
					tree_vel[g-1].append(tree_vel[g-2][ibr]+random_dir(vmags[vind_all[iu][idr]]))
					Npoints +=1
					idr+=1
				iu+=1
					
			if len(tree_vel[g-1])==0:
				break
			
			t1= time.time()
			print('Time for generation: {0}'.format(t1-t0))

		print('Generated {0}/{1} positions.'.format(Npoints, Nc))

	pos_arr= np.array(tree[0])
	vel_arr= np.array(tree_vel[0])
	g_arr = np.ones(len(tree[0]))
	f_arr = np.array(tree_ass[0])
	
	for ibr in range(len(tree)):
		if ibr>=1:
			pos_arr = np.append(pos_arr, np.array(tree[ibr]), axis=0)
			vel_arr =np.append(vel_arr, np.array(tree_vel[ibr]), axis=0)
			g_arr = np.append(g_arr, (ibr+1)*np.ones(len(tree[ibr])))
			f_arr = np.append(f_arr, np.array(tree_ass[ibr]))

	#pos_arr -= np.mean(pos_arr,axis=0)
	#vel_arr -= np.mean(vel_arr,axis=0)

	return pos_arr, vel_arr, g_arr, f_arr





def Nbodyunits(Msi, Rsi, Vsi):
	
	m_sim = np.sum(Msi)
	r_sim = np.power(np.sum(Msi/m_sim),2)/(2.*np.absolute(cc.stellar_potential(Rsi, Msi/m_sim)))
	t_sim = np.sqrt(np.power(r_sim, 3)/(G_si*m_sim))

	print('M_sim: {0} , R_sim = {1}, T_sim = {2}'.format(m_sim, r_sim, t_sim))

	return t_sim, m_sim, r_sim
