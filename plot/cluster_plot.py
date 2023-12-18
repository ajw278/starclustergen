import os
scriptdir = os.path.dirname(os.path.realpath(__file__))

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plot_utils as cpu
import numpy as np
import cluster_calcs as cc
from scipy.spatial.distance import cdist
import copy
from scipy.special import gamma
import binary_reader as bread

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{color}')


NBODYDIR = '/data/ajwinter/Source/Nbody6ppGPU/build/'

BARPOINT = 80
BARLINE =2
BARMARK = 2
s2myr = 3.17098e-8*1e-6
m2au = 6.68459e-12
m2pc = 3.24078e-17
kg2sol = 1./2e30
G_si = 6.67e-11
g0=1.6e-3
au2pc = m2pc/m2au

MLIM = 1e-5
MDLIM=2.5e-10

mpl_cols = ['k','b','g','r','orange', 'c', 'm', 'y']
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a','#999999','#f781bf', '#a65628', '#e41a1c','#984ea3', '#e41a1c', '#dede00']


def resample_times(times, dt):
    """
    Resample the times array to have approximately uniform time steps.

    Parameters:
    - times (numpy array): Array of time values.
    - dt (float): Target time step.

    Returns:
    - resampled_times (numpy array): Resampled array of time values.
    - inc_inds (numpy array): Indices of times with approximately uniform time steps.
    """

    # Resample times based on the target time step
    resampled_times = np.arange(times[0], times[-1], dt)

    # Find the indices in the original times array corresponding to the resampled times
    inc_inds = np.searchsorted(times, resampled_times)
    print(inc_inds, len(times))

    return inc_inds
    
def plot_dvNN(r, v, ndim=2, **svparams):
	
	positions  = r[:, :ndim]
	velocities = v[:, :ndim]
	# Calculate distances between all pairs of stars
	distances = cdist(positions, positions)
	vdistances = cdist(velocities, velocities)

	# Set the diagonal elements to a large value (to exclude a star being its own nearest neighbor)
	np.fill_diagonal(distances, np.inf)

	dist1ct = distances[np.triu_indices(len(distances), k=1)]
	vdist1ct = vdistances[np.triu_indices(len(distances), k=1)]


	# Find the index of the nearest neighbor for each star
	nearest_neighbors = np.argmin(distances, axis=1)
	num_stars = len(positions)
	nearest_neighbor_distances = distances[np.arange(num_stars), nearest_neighbors]

	bins = np.logspace(-1.5, 1.5)
	# Calculate the magnitude of the difference in velocity between each star and its nearest neighbor
	velocity_differences = np.linalg.norm(velocities - velocities[nearest_neighbors], axis=1)

	# Create a scatter plot
	fig, ax = plt.subplots(figsize=(8, 6))

	dsp = np.logspace(-2, 1.5, 40)
	vsp = np.logspace(-2, 1.5, 30)
	D, V = np.meshgrid(dsp, vsp, indexing='ij')

	sigvs  = sigv_pl(D, **svparams)

	print(sigvs, D)

	if len(svparams)>0:
		levels = np.arange(-4.0, 0.2, 0.1)
		pdist = V*V*MB_dist(V*1e5, sigvs, nd=2)
		pdist /= np.amax(pdist, axis=1)[:, np.newaxis]
		ctf=plt.contourf(D, V, np.log10(pdist), levels=levels)
		plt.colorbar(ctf, label='Normalised MB probability: $\log [v g(v)]$')

		rsp = np.logspace(-4, 2.0, 40)
		plt.plot(rsp, sigv_pl(rsp, **svparams)/1e5, color='r', linewidth=2)
		plt.scatter(nearest_neighbor_distances, velocity_differences, c='cyan', edgecolor='gray', s=5)
	else:
		plt.scatter(nearest_neighbor_distances, velocity_differences, c='k', edgecolor='gray', s=5)
		
	# Add labels and title

	ax.tick_params(which='both', axis='both', right=True, top=True, left=True, bottom=True)

	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('%dD separation: $\Delta r$ [pc]'%ndim)
	plt.ylabel('%dD velocity difference: $\Delta v$ [km/s]'%ndim)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim([1e-2, 10.])
	plt.ylim([2e-2, 5.])
	plt.savefig('vdist_%dD.pdf'%ndim, bbox_inches='tight', format='pdf')

	plt.show()

def MB_dist(v_, sig, nd=3):
    signd = sig 
    norm = 0.5*gamma(nd/2.)*(4.*signd*signd)**(nd/2.)
    return (1./norm) *(v_**(nd-1))* np.exp(-v_*v_/4./signd/signd)


def svkep(dr):
	return 2933.0*np.sqrt(1./dr)

def sigv_pl(dr, r0=1.0, p=1., sv0=1.0):
    return np.sqrt((sv0*(dr/r0)**p)**2+ svkep(dr)**2)

def plot_dvNN_fromsim(simulation, time=2.0, **plparams):

	
	t = simulation.t
	r = simulation.r
	v = simulation.v
	munits, runits, tunits, vunits = simulation.units_astro
	
	it = np.argmin(np.absolute(t*tunits - time))
	
	plot_dvNN(r[it], v[it], **plparams)
	


def plot_3dpos(simulation, dim=None, save=True, rlim=20.0, dtmin=0.01):

	def update(frame, data, times, sc, time_text):
		sc._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame,:, 2])
		time_text.set_text('Time: {:.2f} Myr'.format(times[frame]))
		return sc

	def create_3d_stars_animation(rstars, times, filename='stars_animation.mp4'):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		sc = ax.scatter([], [], [], c='r', marker='o', s=1)

		time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
		
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		
		ax.set_xlim([-rlim, rlim])
		ax.set_ylim([-rlim, rlim])
		ax.set_zlim([-rlim, rlim])
		
		num_frames = rstars.shape[0]
		print(num_frames, rstars.shape, times.shape)

		ani = FuncAnimation(fig, update, frames=num_frames, fargs=(rstars, times, sc, time_text),
				interval=1, blit=False)

		ani.save(filename, writer='ffmpeg', fps=20, dpi=800)
		plt.show()

	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	m = copy.copy(simulation.m)
	
	munits, runits, tunits, vunits = simulation.units_astro
	print(tunits, munits, runits, vunits)
	print(r.shape)
	
	print(t)
	print(np.median(np.absolute(r[~np.isnan(r)])))
	r*=runits
	t*=tunits
	print(t)
	print(np.median(np.absolute(r[~np.isnan(r)])))
	if dtmin is None:
		dtmin = np.amin(np.diff(t))
	incinds = resample_times(t, dtmin)
	r_ = r[incinds]
	t_ = t[incinds]
	
	create_3d_stars_animation(r_, t_,filename='stars_animation.mp4')
	


def pairwise_analysis(simulation, ndim=2):
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	munits, runits, tunits, _ = simulation.units_astro

	t *= tunits
	r *= runits

	rbins = np.logspace(-4., 2.0, 25)

	# Set up a colormap for coloring by 't'
	cmap = plt.get_cmap('viridis')
	norm = plt.Normalize(t.min(), t.max())
	scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	scalar_map.set_array([])

	fig, ax = plt.subplots(figsize=(5.,4.))
	nplot=10
	skip = len(t)//nplot
	print('Number of times:', len(t))
	for it in range(0, len(t), skip):
		rsep = cdist(r[it, :, :ndim], r[it, :, :ndim])
		rsep = rsep[np.triu_indices(len(rsep), k=1)]
		rsep = rsep.flatten()

		if ndim == 2:
			weights = 1. / 2. / np.pi / rsep
		elif ndim == 3:
			weights = 1. / 4. / np.pi / rsep / rsep

		hist, be = np.histogram(rsep, density=True, bins=rbins, weights=weights)
		bc = 0.5 * (be[1:] + be[:-1])
		hist /= np.sum(hist*np.diff(be)*2.*np.pi*bc)
		# Plotting with color and markers
		plt.plot(bc, hist, color=scalar_map.to_rgba(t[it]), alpha=0.7)
		plt.scatter(bc, hist, marker='o', s=20, edgecolors='gray', facecolors='white', linewidths=0.5)


	if os.path.isfile('Taurus_distances.npy'):
		d=140.0
		au2pc = 4.84814e-6
		deg2arcsec = 3600.0
		dtaur = np.load('Taurus_distances.npy')
		dtaur *= d*au2pc*deg2arcsec
		
		weights = 1. / 2. / np.pi / dtaur
		
		hist, be = np.histogram(dtaur, density=True, bins=rbins, weights=weights)
		bc = 0.5 * (be[1:] + be[:-1])
		hist /= np.sum(hist*np.diff(be)*2.*np.pi*bc)
		# Plotting with color and markers
		plt.plot(bc, hist, color='r', linewidth=1, label='Taurus (observed)')
		plt.scatter(bc, hist, marker='o', s=20, edgecolors='red', facecolors='white', linewidths=0.5)
	# Color bar
	cbar = plt.colorbar(scalar_map, label='Time (t)',  ax=plt.gca())

	# Logarithmic Scaling
	plt.xscale('log')
	plt.yscale('log')
	
	plt.xlim([1e-4, 50.0])
	plt.ylim([1e-5, 1e3])
	# Adding grid
	plt.grid(True, which='major', linestyle='--', linewidth=0.5)

	ax.tick_params(which='both', axis='both', left=True, right=True, bottom=True, top=True)
	# Display
	plt.xlabel('Pair Separation Distance')
	plt.ylabel('Pair Distribution Function')
	plt.legend(loc='best')
	plt.savefig('pairwise_separation.pdf', bbox_inches='tight', format='pdf')
	plt.show()


def encounter_analysis_binaries(simulation, direct='enchist_bins'):

	wbin_snap = bread.WideBinarySnapshot()
	wbin_snap.create_database()

	binary_snapshot = bread.BinarySnapshot()
	binary_snapshot.create_database() 
	
	munit, runit, tunit, vunit = simulation.units_astro
	munit_phys, runit_phys, tunit_phys = simulation.units_SI
	G = 6.67e-11*(tunit_phys**2)*munit_phys/(runit_phys**3)
	print('GRavitational constant:', G)




	istars = np.arange(1100)
	allbin = bread.AllBinaries(istars)
	#allbin.create_binary_arrays()

	m = simulation.ms

	isub = np.arange(len(m))

	fname = simulation.out+'_enchist_binaries_{0}'


	if not os.path.isdir(direct):
		os.makedirs(direct)
	
	isub = np.sort(isub)
	for istar in isub:
		print('Binary encounter history for %d / %d'%(istar, len(isub)))
		if not os.path.isfile(direct+'/'+fname+'.npy'):
			t,bf, ic, a, e, m2 = allbin.get_history(istar+1)
			a = a.flatten()
			e = e.flatten()
			m2 = m2.flatten()
			bf = bf.flatten()
			ic = ic.flatten()
			t = t.flatten()
			if np.sum(bf)>0:
				dt = np.diff(t, prepend=0.0)
				period = np.sqrt(4.*np.pi*np.pi*(a*au2pc/runit)**3/(G*(m[istar]+m2/munit)))
				prob_enc = dt/period
				print('Prob_encs completed')
				prob_enc[np.isnan(prob_enc)] = 0.0
				print('Generating random number')
				urand = np.random.uniform(size=t.shape)
				print('performing monte carlo')
				iienc = urand<prob_enc
				tenc = t[iienc]
				eenc = e[iienc]
				rpenc = a[iienc]*(1.-e[iienc])
				menc = m2[iienc]/munit
				print(tenc)
				print('Saving')
				np.save(direct+'/'+fname.format(istar), np.array([rpenc, menc, eenc, tenc]))
			else:
				np.save(direct+'/'+fname.format(istar), np.array([[], [], [], []]))


def compare_encanalysis(simulation, istars, direct_s='enchist', direct_b='enchist_bins'):

	munits, runits, tunits, vunits = copy.copy(simulation.units_astro)
	fname_b = simulation.out+'_enchist_binaries_{0}'
	fname_s = simulation.out+'_enchist_{0}'
	icol = 0
	for ist in istars:
		
		x_order, m_order, e_order, t_order = np.load(direct_s+'/'+fname_s.format(ist))
		x_order_b, m_order_b, e_order_b, t_order_b = np.load(direct_b+'/'+fname_b.format(ist))
		plt.scatter(t_order*tunits, x_order*runits/au2pc, color=mpl_cols[icol%len(mpl_cols)], marker='+')
		plt.scatter(t_order_b*tunits, x_order_b*runits/au2pc, color=mpl_cols[icol%len(mpl_cols)], marker='^')

		icol+=1
	plt.yscale('log')
	plt.show()


	munits, runits, tunits, vunits = copy.copy(simulation.units_astro)
	icol = 0
	for ist in istars:
		
		x_order, m_order, e_order, t_order = np.load(simulation.out+'_enchist_{0}.npy'.format(ist))
		x_order_b, m_order_b, e_order_b, t_order_b = np.load(simulation.out+'_enchist_binaries_{0}.npy'.format(ist))
		plt.scatter(t_order*tunits, e_order, color=mpl_cols[icol%len(mpl_cols)], marker='+')
		plt.scatter(t_order_b*tunits, e_order, color=mpl_cols[icol%len(mpl_cols)], marker='^')

		icol+=1
	
	plt.show()
		

def encounter_analysis(simulation, save=False, init_rad = 100.0, res=300,subset=1000, rmax = None,  time=3.0, plotall=True, direct='enchist'):

	print('Copying simulation arrays (may take time if they are large...)')
	t = simulation.t
	r =simulation.r
	v =simulation.v
	m = simulation.m
	print('Encounter analysis...')
	munits, runits, tunits, vunits = copy.copy(simulation.units_astro)
	munits_SI, runits_SI, tunits_SI  = copy.copy(simulation.units_SI)
	print('G:', 6.67e-11*munits_SI*(tunits_SI**2)/(runits_SI)**3)

	rsep = cdist(r[0], r[0])
	rsep = rsep[np.triu_indices(len(r[0]),k=1)]
	rsep = rsep.flatten()


	#istars = np.arange(1100)
	#allbin = bread.AllBinaries(istars)
	#allbin.create_binary_arrays()

	bins=  np.logspace(-4., 1.5, 25)

	isub = np.arange(len(m))

	isub = np.sort(isub)

	all_x = np.array([])
	all_e = np.array([])

	evolve_rall = []

	if not os.path.isdir(direct):
		os.makedirs(direct)

	print('Starting encounter analysis...')

	print('Finding close encounters...')
	ict=0
	for istar in isub[::-1]:
		print('Scanning encounters for i={2} ({0}/{1})'.format(ict+1, len(isub), istar))
		if not os.path.isfile(direct+'/'+simulation.out+'_enchist_{0}.npy'.format(istar)):
			print('Generating global encounter history for star {0}... '.format(istar))
			cx, cv, cm, cn = cc.encounter_history_istar(istar, r, v, m, 2)
			cxb, cvb, cmb = cc.binary_filter(cx, cv, cm)
			x_order = np.array([])
			e_order = np.array([])
			m_order = np.array([])
			t_order = np.array([])

			print('Obtaining neighbour lists for star {0}.'.format(istar))
		
			#logsx_s, logse_s, logst_s = cc.encounter_params(np.array(cx), np.array(cv), np.array(cm), t, float(m[istar]))
			logsx, logse, logst = cc.encounter_params(np.array(cxb), np.array(cvb), np.array(cmb), t, float(m[istar]))
			icol=0
			
			if plotall:
				plt.figure(figsize=(4.,4.))

			print('Organising neighbour interactions...')
			print('Number of neighbours:', cn)
			for inghbr in range(len(cn)):
				all_x = np.append(all_x, logsx[inghbr])
				all_e = np.append(all_e, logse[inghbr])

				x_order= np.append(x_order, logsx[inghbr]*runits) 
				e_order= np.append(e_order, logse[inghbr]) 
				t_order = np.append(t_order, logst[inghbr]*tunits) 
				m_order = np.append(m_order, np.ones(len(logsx[inghbr]))*cm[inghbr])

				if plotall:
					plt.plot(t*tunits, np.linalg.norm(cx[inghbr], axis=1)*runits/au2pc, color=mpl_cols[icol%len(mpl_cols)], linewidth=2)
					plt.plot(t*tunits, np.linalg.norm(cxb[inghbr], axis=1)*runits/au2pc, linestyle='dashed', color='r', linewidth=1)
					plt.scatter(np.array(logst[inghbr])*tunits, np.array(logsx[inghbr])*runits/au2pc, color=mpl_cols[icol%len(mpl_cols)], marker='+')
				icol+=1

			if plotall:
				plt.xlabel('Time [Myr]')
				plt.ylabel('Separation [au]')
				plt.yscale('log')
				plt.savefig(simulation.out+'_enchist_{0}.pdf'.format(istar), format='pdf', bbox_inches='tight')
				plt.close()
			print(x_order,t_order, e_order)
			chron = np.argsort(t_order)
			x_order = x_order[chron]
			m_order = m_order[chron]
			e_order = e_order[chron]
			t_order = t_order[chron]

			if len(x_order)>0:
				print('Closest encounter: ', np.amin(x_order), m2au, runits)

			np.save(direct+'/'+simulation.out+'_enchist_{0}'.format(istar), np.array([x_order, m_order, e_order, t_order]))

			"""print('Calculating disc evolution for star {0}'.format(istar))

			revol = cluster_calcs.disc_evol(x_order, e_order,m_order, t_order, init_rad, m[istar],5000., 0.8)"""
			"""
		

			rout_evol = np.ones(res)*init_rad
			times_evol = np.linspace(0.0, t[-1]*tunits*s2myr, res)


			itime=0
			if revol!=None:
				for iev in range(res):
					if itime<len(t_order):
						while times_evol[iev]>t_order[itime]:
							rout_evol[iev:] = revol[itime]
							itime+=1
							if itime>=len(t_order):
								break


				#plt.plot(times_evol, rout_evol)
				evolve_rall.append(rout_evol)
				np.save(simulation.out+'_encrevol_{0}'.format(istar), rout_evol)
				np.save('time_revol', times_evol)
			else:
				print(np.amin(e_order), np.amin(x_order))
				print('Binary phase detected for {0}.'.format(istar))
				np.save(simulation.out+'_encrevol_{0}'.format(istar), np.array([]))"""
			
		
		else:
			
			x_order, m_order, e_order, t_order = np.load(direct+'/'+simulation.out+'_enchist_{0}.npy'.format(istar))
			"""print('Previous radius calculation found for {0}'.format(istar))
			rout_evol = np.load('revol_{0}.npy'.format(istar))
			if len(rout_evol)>1:
				evolve_rall.append(rout_evol)"""

		if len(x_order)>0:
			xmin  = np.amin(x_order)
			print('Closest encounter for i=%d : %.2e'%(istar, xmin))
		
		ict+=1

	



def plot_discani3d(simulation, restrict=None, rinit=100.0, nprocs=10, rmax=5.0, mbig=30.0, ptype='radius'):
	t = simulation.t
	r = simulation.r
	v = simulation.v
	m = simulation.m
	tunits, munits, runits = simulation.units

	t *= tunits*s2myr

	rout_all = simulation.phot_r
	mdisc_all = simulation.phot_m

	rmean = np.mean(rout_all, axis=0)

	rswap = np.swapaxes(r, 0,2)
	#rswap = np.swapaxes(rswap, 1,2)
	x = rswap[0]
	y = rswap[1]
	z = rswap[2]

	x= np.swapaxes(x,0,1)
	y = np.swapaxes(y,0,1)
	z = np.swapaxes(z,0,1)

	rout_all = np.swapaxes(rout_all, 0,1)

	cm = plt.cm.get_cmap('autumn')

	# create the figure
	fig=plt.figure()
	ax=fig.gca(projection='3d')

	# create the first plot

	biginds = np.where(m>mbig)[0]

	SFACT = 20.0
	point=ax.scatter(x[1], y[1], z[1],s=SFACT*m,  marker=(5, 2), c=rout_all[0], vmin=0, vmax=100.0, cmap=cm, zorder=1)
	#pointb =ax.scatter(x[0][biginds], y[0][biginds], z[0][biginds],s=10.*SFACT*m[biginds]*munits*kg2sol, c='c', zorder=3)
	ax.legend()
	ax.set_xlim([-rmax, rmax])
	ax.set_ylim([-rmax, rmax])
	ax.set_zlim([-rmax, rmax])

	ax.set_xlabel('x (pc)')
	ax.set_ylabel('y (pc)')
	ax.set_zlabel('z (pc)')


	if float(mpl.__version__.split('.')[0])>=2.0:
		ax.set_facecolor('black')
	else:
		ax.set_axis_bgcolor('black')
	ax.xaxis.label.set_color('white')
	ax.yaxis.label.set_color('white')
	ax.zaxis.label.set_color('white')
	ax.tick_params(axis='x',colors='white')
	ax.tick_params(axis='y',colors='white')
	ax.tick_params(axis='z',colors='white')
	R=.0
	G=.0
	B=.0
	A=.0
	ax.w_xaxis.set_pane_color((R,G,B,A))
	ax.w_yaxis.set_pane_color((R,G,B,A))
	ax.w_zaxis.set_pane_color((R,G,B,A))

	ttext = ax.text2D(0.05, 0.95, "$t = $ {0} Myrs".format(0.0), transform=ax.transAxes, color='w')

	cb1 = fig.colorbar(point)
	cb1.set_label('Disc Radius (au)')

	cbytick_obj = plt.getp(cb1.ax.axes, 'yticklabels') 
	#cbylab_obj = plt.getp(cb1.ax.axes, 'label')                #tricky
	#plt.setp(cbytick_obj, color='w')               
	#plt.setp(cbylab_obj, color='w')
	# first option - remake the plot at every frame
	def update_axes(n, x, y, z,m, rad,times,ax):
		ax.cla()
		ax.set_xlim([-rmax, rmax])
		ax.set_ylim([-rmax, rmax])
		ax.set_zlim([-rmax, rmax])
		ax.text2D(0.05, 0.95, "$t = $ {:03.2f} Myrs".format(times[n]), transform=ax.transAxes,color='w')
		print('Time: {:03.2f}, mean rad:  {:.2f}, disp: {:.2f}'.format(times[n], np.mean(rad[n]), np.std(rad[n])))

		ininds = np.where((np.absolute(x[n])<rmax)&(np.absolute(y[n])<rmax)&(np.absolute(z[n])<rmax))[0]
		binds = np.where((m>30.0)&(np.absolute(x[n])<rmax)&(np.absolute(y[n])<rmax)&(np.absolute(z[n])<rmax))[0]
		point=ax.scatter(x[n][ininds], y[n][ininds], z[n][ininds], s=SFACT*m, marker=(5, 2), c=rad[n][ininds], vmin=0, vmax=100.0, cmap=cm, zorder=1)
		#pointb=ax.scatter(x[n][binds], y[n][binds], z[n][binds],s=30, c='c', zorder=3)
		
		ax.legend()
		return point

	nstep =20
	ani=animation.FuncAnimation(fig, update_axes, len(t[1::nstep]), fargs=(x[1::nstep], y[1::nstep], z[1::nstep],m, rout_all[::nstep],t[::nstep], ax))

	# make the movie file demo.mp4

	writer=animation.writers['ffmpeg'](fps=5)
	dpi = 500
	ani.save(simulation.out+'_3D_photoevap.mp4',writer=writer,dpi=dpi)
	
	plt.show()

def core_density_evol(simulation):
	
	
	t = simulation.t
	r =simulation.r 
	v = simulation.v 
	m = simulation.m

	tunits, munits, runits = simulation.units

	rs = np.linalg.norm(r,axis=2)

	ns = np.zeros(len(t))

	for itime in range(len(t)):
		binspace = np.linspace(0.0, 4.0, 41)# np.logspace(-2.,np.log10(4.0), 5)
		ndense, binn_edges = np.histogram(rs[itime], bins=binspace)
		binn_vol = (4.*np.pi/3.)*(binn_edges[1:]**3-binn_edges[:1]**3)
		binn_cent = (binn_edges[:1]+binn_edges[1:])/2.
		ndense = np.array(ndense, dtype=np.float64)
		ndense/= binn_vol

		ns[itime] = np.amax(ndense)


	plt.plot(t*tunits*s2myr, ns/np.power(runits*m2pc, 3))
	plt.ylabel('$n$ (pc$^{-3}$)')
	plt.xlabel('Time (Myr)')
	plt.show()



def plot_starani2d(simulation,rmax=12.5, centre=(.0,.0), mfilt=0.5, show_centre=False, radcent=5.0, dtsize=0.1):


	t = copy.copy(simulation.t)
	r =copy.copy(simulation.r)
	v = copy.copy(simulation.v)
	m = copy.copy(simulation.m)


	
	"""if hasattr(simulation, 'starinds'):
		popinds = copy.copy(simulation.starinds)
		prev_inds = np.array([])
		for stinds in popinds:
			iesc  = prev_inds[~np.in1d(prev_inds, stinds)]
			print('Escaper indices:',iesc)
			if len(iesc)>1:
				break
			prev_inds = stinds

	
	

	r = np.swapaxes(np.swapaxes(r,0,1)[iesc],0,1)
	m = m[iesc]
	v = np.swapaxes(np.swapaxes(v,0,1)[iesc],0,1)"""

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol


	

	tnext = 0.0
	isteps = []
	for it in range(len(t)):
		if tarch[it]>=tnext:
			isteps.append(it)
			tnext = tarch[it] + dtsize

	

	print('Tsnaps:', tarch[isteps])

	rswitch = np.swapaxes(r,0,1)

	rswap = np.swapaxes(r, 0,2)
	#rswap = np.swapaxes(rswap, 1,2)
	x = rswap[0]
	y = rswap[1]
	z = rswap[2]


	x= np.swapaxes(x,0,1)
	y = np.swapaxes(y,0,1)
	z = np.swapaxes(z,0,1)


	x -= centre[0]
	y -= centre[1]

	
	# create the figure
	mpl.rc('axes',edgecolor='k')
	fig, ax = plt.subplots()

	# create the first plot
	biginds = np.where(m>mfilt)[0]

	
	pointb =ax.scatter(x[0][biginds], y[0][biginds],s=20, c='c', zorder=3)

	ax.legend()
	ax.set_xlim([-rmax, rmax])
	ax.set_ylim([-rmax, rmax])



	if float(mpl.__version__.split('.')[0])>=2.0:
		ax.set_facecolor('black')
	else:
		ax.set_axis_bgcolor('black')
	"""ax.tick_params(axis='x',colors='white')
	ax.tick_params(axis='y',colors='white')"""
	ax.get_xaxis().set_tick_params(direction='out', width=1)
	ax.get_yaxis().set_tick_params(direction='out', width=1)

	ax.set_xlabel('x (pc)', color='k')
	ax.set_ylabel('y (pc)', color='k')
	if float(mpl.__version__.split('.')[0])>=2.0:
		ax.set_facecolor('black')
	else:
		ax.set_axis_bgcolor('black')

	ttext = ax.text(0.05, 0.95, "$t = $ {0} Myrs".format(0.0), transform=ax.transAxes, color='k')

		
	def update_axes(n, x, y, m, times,ax):
		ax.cla()
		ax.set_xlim([-rmax, rmax])
		ax.set_ylim([-rmax, rmax])
		ax.text(0.25, 1.01, "$t =  {:03.2f}$ Myr".format(times[n]), transform=ax.transAxes,color='k')

		filt1 = np.where((m>mfilt)&(np.absolute(x[n])<rmax)&(np.absolute(y[n])<rmax))[0]
		
		xt = x[n][filt1]
		yt = y[n][filt1]
		zt = z[n][filt1]
		zsort = np.argsort(zt)
		xt = xt[zsort]
		yt = yt[zsort]


		point=ax.scatter(xt,yt, s=20,c='c')
		if show_centre:
			pos3d = np.swapaxes(np.array([x[n],y[n],z[n]]), 0,1)
			centre = cluster_calcs.empirical_centre(pos3d, radcent, 2, 20, 3)
			print('Centre at t={0}: {1}'.format(times[n], centre))
			ax.scatter(centre[0], centre[1], s=40, c='r')
			
		ax.legend()
		return point

	nstep =1
	ani=animation.FuncAnimation(fig, update_axes, len(tarch[isteps]), fargs=(x[isteps], y[isteps],  m,tarch[isteps],ax))

	# make the movie file demo.mp4

	#writer=animation.writers['ffmpeg'](fps=3)
	#dpi = 500
	#ani.save(simulation.out+'_stars_2D.mp4',writer=writer,dpi=dpi)

	plt.show()


def plot_dft(simulation,rmax=12.5, centre=(.0,.0), sp_time=None, wext=False, cent=0.4):

	
	if wext:
		pinds = copy.copy(simulation.photoevap_inds)
		disc_ms = np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		disc_rs = np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		disc_ms = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		disc_rs = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)

	t = copy.copy(simulation.t)
	r =copy.copy(simulation.r)
	v = copy.copy(simulation.v)
	m = copy.copy(simulation.m)

	
	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol

	if sp_time!=None:
		list_indices = time_inds(tarch, dt=sp_time)
		nsnaps = len(list_indices)
	else:
		nsnaps = len(tarch)
		list_indices = np.arange(nsnaps)



	rdisc = np.swapaxes(np.swapaxes(r,0,1)[pinds],0,1)
	vdisc= np.swapaxes(np.swapaxes(v,0,1)[pinds],0,1)
	
	rmassive = np.swapaxes(np.swapaxes(r,0,1)[np.where(m>20.0)],0,1)
	vmassive= np.swapaxes(np.swapaxes(v,0,1)[np.where(m>20.0)],0,1)


	rswap = np.swapaxes(rdisc, 0,2)
	rswapm = np.swapaxes(rmassive, 0,2)
	#rswap = np.swapaxes(rswap, 1,2)
	x = rswap[0]
	y = rswap[1]
	z = rswap[2]

	xm = rswapm[0]
	ym = rswapm[1]
	zm = rswapm[2]


	x= np.swapaxes(x,0,1)
	y = np.swapaxes(y,0,1)
	z = np.swapaxes(z,0,1)

	xm= np.swapaxes(xm,0,1)
	ym = np.swapaxes(ym,0,1)
	zm = np.swapaxes(zm,0,1)


	x -= centre[0]
	y -= centre[1]

	xm -= centre[0]
	ym -= centre[1]

	dfrac_cent = np.zeros(len(list_indices))
	dfrac_all = np.zeros(len(list_indices))#

	tvals = tarch[list_indices]

	ival = 0
	for n in list_indices:
		filt1 = np.where(np.absolute(x[n]**2+y[n]**2)<rmax**2)[0]
		filt2 = np.where(np.absolute(xm[n]**2+ym[n]**2)<rmax**2)[0]


		xt = x[n][filt1]
		yt = y[n][filt1]
		#zsort = np.argsort(zt)
		#xt = xt[zsort]
		#yt = yt[zsort]
		
		
		rvals = np.sqrt(xt**2+yt**2)
		icdisc = np.where((disc_ms[n][filt1]>MLIM)&(rvals<cent))[0]
		icall = np.where(rvals<cent)[0]

		idisc = np.where(disc_ms[n][filt1]>MLIM)[0]
		
		dfrac_cent[ival] = float(len(icdisc))/float(len(icall))
		dfrac_all[ival] = float(len(idisc))/float(len(filt1))
		print('Central/all disc fraction at t={0}: {1}'.format(tarch[n], dfrac_cent[ival], dfrac_all[ival]))
		ival+=1

	np.save(simulation.out+'_dfracs', np.array([tvals, dfrac_cent, dfrac_all]))

	print(plt.rcParams['interactive'])
	print(plt.get_backend())

	plt.rc('font', family='serif')
	plt.rc('text', usetex=True)
	plt.figure(figsize=(4.,4.))
	plt.plot(tarch[list_indices], dfrac_cent, color=CB_color_cycle[3], label='$d<%.1lf$ pc'%(cent))
	plt.plot(tarch[list_indices], dfrac_all, color=CB_color_cycle[4], label='All')
	plt.legend(loc='best', fontsize=10)
	plt.xlabel('Time (Myr)')
	plt.ylabel('Disc fraction ($M_\mathrm{disc}>10^{-5} \, M_\odot$)')
	plt.savefig(simulation.out+'_dft.pdf', bbox_inches='tight', format='pdf')
	plt.show()

	return tvals, dfrac_cent, dfrac_all

	

				
def plot_discani2d(simulation,rmax=12.5, centre=(.0,.0), mfilt=0.1, ptype='radius', wext=False,nstep=1, save=False):

	
	if wext:
		pinds = copy.copy(simulation.photoevap_inds)
		disc_ms = np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		disc_rs = np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		disc_ms = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		disc_rs = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)

	t = copy.copy(simulation.t)
	r =copy.copy(simulation.r)
	v = copy.copy(simulation.v)
	m = copy.copy(simulation.m)

	
	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol


	rdisc = np.swapaxes(np.swapaxes(r,0,1)[pinds],0,1)
	vdisc= np.swapaxes(np.swapaxes(v,0,1)[pinds],0,1)
	
	rmassive = np.swapaxes(np.swapaxes(r,0,1)[np.where(m>10.0)],0,1)
	vmassive= np.swapaxes(np.swapaxes(v,0,1)[np.where(m>10.0)],0,1)


	SFACT = 20.0

	if ptype=='radius':
		print('Assigning radius evolution...')	
		plot_vals = disc_rs
		psize=None
	if ptype=='mass':
		print('Assigning mass evolution...')	
		plot_vals = disc_ms
		psize=None
	elif ptype=='g0':
		print('Assigning G0...')	
		plot_vals = g0vals
		psize= None
	elif ptype=='g0m':
		plot_vals = g0vals
		psize = 1e5*disc_ms
		psize = np.log10(psize)
		psize[np.where(psize<=0.)]=0.0
		psize[np.where(np.isnan(psize))]=0.0
	else:
		print('Plot type not recognised: "{0}"'.format(ptype))
		print(ptype, 'mass', ptype=='mass', type(ptype))
		exit()

	print('Evolution assigned.')

	rmean = np.mean(plot_vals, axis=0)

	rswap = np.swapaxes(rdisc, 0,2)
	rswapm = np.swapaxes(rmassive, 0,2)
	#rswap = np.swapaxes(rswap, 1,2)
	x = rswap[0]
	y = rswap[1]
	z = rswap[2]

	xm = rswapm[0]
	ym = rswapm[1]
	zm = rswapm[2]


	x= np.swapaxes(x,0,1)
	y = np.swapaxes(y,0,1)
	z = np.swapaxes(z,0,1)

	xm= np.swapaxes(xm,0,1)
	ym = np.swapaxes(ym,0,1)
	zm = np.swapaxes(zm,0,1)


	x -= centre[0]
	y -= centre[1]

	xm -= centre[0]
	ym -= centre[1]


	cm = plt.cm.get_cmap('autumn')
	"""print(rout_all[0].shape, x[0].shape, y[0].shape)
	sc  = plt.scatter(x[0], y[0],  s=5, marker='+', c=rout_all[-1], vmin=0, vmax=100.0, cmap=cm)
	cb1 = plt.colorbar(sc)
	cb1.set_label('Disc Radius (au)')		
	plt.show()"""

	# create the figure
	plt.rc('axes',edgecolor='k')
	fig, ax = plt.subplots()

	# create the first plot

	if ptype=='radius':
		point=ax.scatter(x[0], y[0], s=20,  c=plot_vals[0], vmin=20.0, vmax=100.0, cmap=cm)
	elif ptype=='mass':
		point=ax.scatter(x[0], y[0], s=20,  c=plot_vals[0],norm=LogNorm(vmin=1e-6, vmax=1e-1), cmap=cm)
	elif ptype=='g0':
		point=ax.scatter(x[0], y[0], s=20,  c=plot_vals[0], norm=LogNorm(vmin=1e2, vmax=1e5), cmap=cm)
	elif ptype=='g0m':
		ax.scatter(x[0], y[0], c='g',marker='+', s=1)
		point=ax.scatter(x[0], y[0], c=plot_vals[0], norm=LogNorm(vmin=1e2, vmax=1e5), cmap=cm, s=SFACT*psize[0])

	ax.scatter(xm[0], ym[0], s=50.0, c='cyan',marker='+')

	ax.legend()
	ax.set_xlim([-rmax, rmax])
	ax.set_ylim([-rmax, rmax])




	"""ax.tick_params(axis='x',colors='white')
	ax.tick_params(axis='y',colors='white')"""
	ax.get_xaxis().set_tick_params(direction='out', width=1)
	ax.get_yaxis().set_tick_params(direction='out', width=1)

	ax.set_xlabel('x (pc)', color='k')
	ax.set_ylabel('y (pc)', color='k')
	print(mpl.__version__)
	if float(mpl.__version__.split('.')[0])>=2.0:
		ax.set_facecolor('black')
	else:
		ax.set_axis_bgcolor('black')

	ttext = ax.text(0.05, 0.95, "$t = $ {0} Myrs".format(0.0), transform=ax.transAxes, color='w')

	
	cb1 = fig.colorbar(point)
	if ptype=='g0' or ptype=='g0m':
		cb1.set_label('FUV Flux ($G_0$)')
	elif ptype=='radius':
		cb1.set_label('Disc Radius (au)')
	elif ptype=='mass':
		cb1.set_label('Disc mass ($M_\\odot$)')

	cbytick_obj = plt.getp(cb1.ax.axes, 'yticklabels') 

	
	def update_axes(n, xs, ys,xms,yms, pvals,pointsize, times,ax):
		ax.cla()
		ax.set_xlim([-rmax, rmax])
		ax.set_ylim([-rmax, rmax])
		ax.text(0.05, 0.95, "$t = {:03.2f}$ Myrs".format(times[n]), transform=ax.transAxes, color='w')
		print('Time: {:03.2f}, mean rad:  {:.2f}, disp: {:.2f}'.format(times[n], np.mean(pvals[n]), np.std(pvals[n])))
		filt1 = np.where((np.absolute(xs[n])<rmax)&(np.absolute(ys[n])<rmax))[0]
		filt2 = np.where((np.absolute(xms[n])<rmax)&(np.absolute(yms[n])<rmax))[0]


		xt = xs[n][filt1]
		yt = ys[n][filt1]
		#zsort = np.argsort(zt)
		#xt = xt[zsort]
		#yt = yt[zsort]
		

		
		"""plt.show()
		plt.scatter(np.sqrt(xt**2+yt**2), pvals[n][filt1])
		plt.yscale('log')
		plt.xlim([0.,3.0])
		plt.ylim([1e0, 1e5])
		rsort = np.sort(np.sqrt(xt**2+yt**2))
		
		plt.show()

		rspace = np.linspace(0., 10.0, 100)
		rvals = np.sqrt(xt**2+yt**2)
		cumdist_resfix = np.zeros(100)
		for ir in range(len(rspace)):
			cumdist_resfix[ir] = len(np.where(rvals<rspace[ir])[0])

		plt.plot(rspace, cumdist_resfix)
		plt.show()"""

		if ptype=='radius':
			point=ax.scatter(xt,yt, s=20, c=pvals[n][filt1], vmin=0.0, vmax=100.0, cmap=cm)
		elif ptype=='mass':
			point=ax.scatter(xt,yt, s=20, c=pvals[n][filt1], norm=LogNorm(vmin=1e-6, vmax=1e-1), cmap=cm)
		elif ptype=='g0':
			point=ax.scatter(xt,yt, s=20, c=pvals[n][filt1], norm=LogNorm(vmin=1e2, vmax=1e5), cmap=cm)
		elif ptype=='g0m':
			ax.scatter(xt, yt, c='g',marker='+', s=1)
			point=ax.scatter(xt,yt, c=pvals[n][filt1], norm=LogNorm(vmin=1e2, vmax=1e5), cmap=cm, s=SFACT*pointsize[n][filt1])
			propinds = np.where((pvals[n][filt1]>1e4)&(pointsize[n][filt1]>MLIM))[0]
			ax.scatter(xt[propinds], yt[propinds], s=10.0, c='blue',marker='+')
			
			
			rvals = np.sqrt(xt**2+yt**2)
			icent = np.where((pointsize[n][filt1]>MLIM)&(rvals<0.2))[0]
			iall = np.where(rvals<0.2)[0]
			surv_frac = float(len(icent))/float(len(iall))
			print('Central disc fraction at t={0}: {1}'.format(times[n], surv_frac))

		ax.scatter(xms[n][filt2], yms[n][filt2], s=50.0, c='cyan',marker='+')

		ax.set_xlabel('x (pc)', color='k')
		ax.set_ylabel('y (pc)', color='k')
		#pointb=ax.scatter(x[n][biginds], y[n][biginds], z[n][biginds],s=10.*SFACT*m[biginds]*munits*kg2sol, c='c', zorder=3)
	
		ax.legend()
		return point

	#cdfrac = np.array(CENT_DFRAC)
	#cdfrac_ts = np.array(CENT_DFRAC_T)

	#np.save(simulation.out+'_cdfrac', np.array([cdfrac_ts,cdfrac]))

	ani=animation.FuncAnimation(fig, update_axes, len(tarch[::nstep]), fargs=(x[::nstep], y[::nstep],xm[::nstep], ym[::nstep], plot_vals[::nstep],psize[::nstep],tarch[::nstep], ax))

	# make the movie file demo.mp4
	if save:
		writer=animation.writers['ffmpeg'](fps=5)
		dpi = 500
		ani.save(simulation.out+'_{0}_2D_photoevap.mp4'.format(ptype),writer=writer,dpi=dpi)

	plt.show()

	"""plt.rc('font', family='serif')
	plt.rc('text', usetex=True)
	plt.figure(figsize=(4.,4.))
	plt.plot(cdfrac_ts, cdfrac)
	plt.xlabel('Time (Myr)')
	plt.ylabel('Disc fraction within $0.2$ pc')
	plt.show()"""


def plot_dfrac_time(simulation, wext=False, g0=1.6e-3, show=False, rlim=3.0):
	tunits, munits, runits = copy.copy(simulation.units)
	t = copy.copy(simulation.t)
	m= copy.copy(simulation.m)
	

	tarch = t*tunits*s2myr
	m *= munits*kg2sol

		
	if not wext:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		dmacc = np.swapaxes(copy.copy(simulation.phot_dmacc[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
		euvvals = np.swapaxes(copy.copy(simulation.EUV[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc =np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		dmacc = np.swapaxes(copy.copy(simulation.phot_wext_dmacc[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)
		euvvals = np.swapaxes(copy.copy(simulation.EUVwext[pinds]),0,1)

	rst = copy.copy(simulation.r)*runits*m2pc

	MLLIM = 1e-5
	dfrac = np.ones(len(t))

	for tind in range(len(t)):
		mdt = mdisc[tind]
		rmag = np.linalg.norm(rst[tind][pinds],axis=1)
		print('Rmag:', rmag)
		mdt = mdt[np.where(rmag<rlim)[0]]

		df = float(len(np.where(mdt>MLLIM)[0]))/float(len(mdt))
		dfrac[tind] = df

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	plt.figure(figsize=(4.,4.))
	plt.plot(tarch, dfrac, c='k')


	

	plt.xlabel('Time (Myr)')
	plt.ylabel('Disc fraction')
	
	plt.savefig('dfrac_time.pdf', bbox_inches='tight', format='pdf')
	
	if show:
		plt.show()
	else:
		plt.close()

	return None
	
def plot_maccmwind_time(simulation, wext=False, g0=1.6e-3, show=False):
	tunits, munits, runits = copy.copy(simulation.units)
	t = copy.copy(simulation.t)
	m= copy.copy(simulation.m)
	

	tarch = t*tunits*s2myr
	m *= munits*kg2sol

		
	if not wext:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		dmacc = np.swapaxes(copy.copy(simulation.phot_dmacc[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
		euvvals = np.swapaxes(copy.copy(simulation.EUV[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc =np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		dmacc = np.swapaxes(copy.copy(simulation.phot_wext_dmacc[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)
		euvvals = np.swapaxes(copy.copy(simulation.EUVwext[pinds]),0,1)

	rst = copy.copy(simulation.r)*runits
	vst = copy.copy(simulation.v)*runits/tunits

	mrat_med = np.zeros(len(t))
	mrat_up = np.zeros(len(t))
	mrat_low = np.zeros(len(t))

	MLLIM = 1e-5
	dfrac = np.ones(len(t))

	for tind in range(len(t)):
		mdt = mdisc[tind]
		rdt = rdisc[tind]
		mdat = dmacc[tind]
	
		fuv_alt = g0vals[tind]
		euv_alt = euvvals[tind]
		if not os.path.isfile(simulation.out+'_mdotphot_%d.npy'%(tind)):
			mdpt_alt = np.zeros(rdt.shape)
			
			iim=0
			for im in pinds:
				if rdt[iim]>.1:
					mlf = mlc.mloss_function(mstar=m[im])
					mlFUV = mlf.mdot(g0vals[tind][iim], rdt[iim], mdt[iim])
					mlEUV = mlc.mloss_euv(euvvals[tind][iim], rdt[iim])
					print('Mdot FUV:', mlFUV)
					print('Mdot EUV:', mlEUV)
					mdpt_alt[iim] = max(mlFUV, mlEUV)
				print(iim,'/', len(pinds))
				iim+=1
			np.save(simulation.out+'_mdotphot_%d'%(tind), mdpt_alt)
		else:
			mdpt_alt = np.load(simulation.out+'_mdotphot_%d.npy'%(tind))

		
		ipe = np.where(mdpt_alt>3e-10)[0]
			
		mrat = mdpt_alt[ipe]/mdat[ipe]
		mrat_med[tind] = np.median(mrat)
		mrat_low[tind] = np.percentile(mrat, 16.0)
		mrat_up[tind] = np.percentile(mrat, 84.0)

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	plt.figure(figsize=(4.,4.))
	plt.fill_between(tarch, mrat_up, y2=mrat_low, alpha=0.5, color='grey')
	plt.plot(tarch, mrat_med, c='k')


	mlONC = 0.8799622020355788 
	mhONC = 2.6159865034115772
	plt.fill_between([0.8,3.0], [mhONC, mhONC], y2=[mlONC, mlONC], alpha=0.2, color='red')
	plt.text(0.85, 3.0, 'ONC', color='red')


	ymin= mrat_low[-1]
	ymax = mrat_up[-1]
	plt.fill_between([3.0,5.0], [ymax, ymax], y2=[ymin,ymin], alpha=0.2, color='orange')
	plt.text(3.2, 1.2*ymax, '$\sigma$ Orionis', color='orange')

	plt.xlabel('Time (Myr)')
	plt.ylabel('$\dot{M}_\mathrm{wind}/\dot{M}_\mathrm{acc}$')
	
	plt.yscale('log')
	plt.xlim([0., 5.])
	plt.ylim([0.5, 10.])

	plt.savefig('mdotrat_time.pdf', bbox_inches='tight', format='pdf')
	
	if show:
		plt.show()
	else:
		plt.close()
	plt.show()


def plot_maccmwind_snap(simulation, wext=False, g0=1.6e-3, time=0.0, rmax=None, show=False):
	tunits, munits, runits = copy.copy(simulation.units)
	t = copy.copy(simulation.t)
	m= copy.copy(simulation.m)
	

	tarch = t*tunits*s2myr
	m *= munits*kg2sol

	imax=  np.argmax(m)

		
	if not wext:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		dmacc = np.swapaxes(copy.copy(simulation.phot_dmacc[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
		euvvals = np.swapaxes(copy.copy(simulation.EUV[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc =np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		dmacc = np.swapaxes(copy.copy(simulation.phot_wext_dmacc[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)
		euvvals = np.swapaxes(copy.copy(simulation.EUVwext[pinds]),0,1)

	rst = copy.copy(simulation.r)*runits*m2pc



	tind = np.argmin(np.absolute(tarch-time))

	print('Time: ', tind, time, tarch[tind])

	
	mdt = mdisc[tind]
	rdt = rdisc[tind]
	mdat = dmacc[tind]

	fuv_alt = g0vals[tind]
	euv_alt = euvvals[tind]
	if not os.path.isfile(simulation.out+'_mdotphot_%d.npy'%(tind)):
		mdpt_alt = np.zeros(rdt.shape)
		
		iim=0
		for im in pinds:
			if rdt[iim]>.1:
				mlf = mlc.mloss_function(mstar=m[im])
				mlFUV = mlf.mdot(g0vals[tind][iim], rdt[iim], mdt[iim])
				mlEUV = mlc.mloss_euv(euvvals[tind][iim], rdt[iim])
				print('Mdot FUV:', mlFUV)
				print('Mdot EUV:', mlEUV)
				mdpt_alt[iim] = max(mlFUV, mlEUV)
			print(iim,'/', len(pinds))
			iim+=1
		np.save(simulation.out+'_mdotphot_%d'%(tind), mdpt_alt)
	else:
		mdpt_alt = np.load(simulation.out+'_mdotphot_%d.npy'%(tind))

	
	

	rt = rst[tind]
	rmassive = rt[imax]
	rt -= rmassive
	rtd = rt[pinds]
	rmag = np.linalg.norm(rtd, axis=1)
	if type(rmax)==type(None):
		rmax = 1e10
	print(rmag)


	ipe = np.where((mdpt_alt>MDLIM)&(mdat>0.0)&(rmag<rmax))[0]
	nipe = np.where((mdpt_alt<MDLIM)&(mdat>0.0)&(rmag<rmax))[0]
	
	mrat = mdpt_alt[ipe]/mdat[ipe]

	#np.save('ONCcore_mrats', np.array([mdpt_alt[ipe],mdat[ipe]]))

	dmphONC, dmaccONC = np.load('ONCcore_mrats.npy')
	ipe_ONC = np.where((dmphONC>MDLIM)&(dmaccONC>0.0))[0]
	mrat_ONC = dmphONC[ipe_ONC]/dmaccONC[ipe_ONC]

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	bins = np.linspace(-1.,4, 50)
	fig, ax = plt.subplots(figsize=(5.,3.))#plt.figure() #figsize=(6.,4.)

	plt.scatter(np.log10(mdpt_alt[ipe]),np.log10(mdat[ipe]),c='orange', marker='o', label='$\sigma$ Orionis (5 Myr)')
	plt.scatter(np.log10(mdpt_alt[nipe]),np.log10(mdat[nipe]), c='orange', marker='o',alpha=0.2)
	plt.scatter(np.log10(dmphONC),np.log10(dmaccONC), c='blue', marker='s', label='ONC core (2.8 Myr)')

	
	xl = np.linspace(-11., -6.)
	plt.plot(xl, xl, c='k')
	plt.plot(xl+np.log10(np.median(mrat)), xl, c='orange')
	plt.plot(np.log10(np.median(mrat_ONC))+xl,xl, c='b')
	print('Median sigma Orionis:',np.median(mrat))
	print('Median ONC:',np.median(mrat_ONC))
	
	

	"""plt.hist(np.log10(mrat), bins=bins,facecolor='orange', alpha=0.2, label='$\sigma$ Ori (5 Myr)', normed=True)
	plt.hist(np.log10(mrat_ONC), bins=bins,facecolor='red', alpha=0.2, label='ONC core (2.8 Myr)', normed=True)
	plt.axvline(0.0, c='k')
	plt.axvline(np.median(np.log10(mrat)), c='orange') 
	plt.axvline(np.median(np.log10(mrat_ONC)), c='red') 
	
	plt.ylabel('Probability density ($\dot{M}_\mathrm{wind}>3\cdot10^{-10} \, M_\odot$~yr$^{-1}$)')
	plt.xlim([-1, 3])
	plt.legend(loc='best')"""
	plt.axvline(np.log10(MDLIM), c='r', linestyle='dashed')
	plt.axvline(np.log10(1e-10), c='r')
	plt.ylim([-10.5, -7.6])
	plt.xlim([np.log10(8e-11), -5.8])
	plt.legend(loc=4, fontsize=12)
	ax.set_aspect('equal')
	plt.ylabel('$\log \dot{M}_\mathrm{acc}$ ($M_\odot$ yr$^{-1}$)')
	plt.xlabel('$\log \dot{M}_\mathrm{wind}$ ($M_\odot$ yr$^{-1}$)')
	plt.savefig('mdotrat_snap.pdf', bbox_inches='tight', format='pdf')
	if show:
		plt.show()
	else:
		plt.close()
	

	

	
def plot_mstmdisc_ipops(simulation, time=0., wext=False, g0=1.6e-3, rmaxtype='dist', rmax=None,prop='mass', var='mst', popdiv=True, cbv='eccorb', uplims=True, show=False, ax=None, tvals=None):


	axparse=True
	if type(ax)==type(None):
		fig, ax = plt.subplots(figsize=(4.,4.))
		axparse=False


	tunits, munits, runits = copy.copy(simulation.units)
	t = copy.copy(simulation.t)
	m= copy.copy(simulation.m)
	

	tarch = t*tunits*s2myr
	tind = np.argmin(np.absolute(tarch-time))
	tval = tarch[tind]
	m *= munits*kg2sol

		
	if not wext:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		dmacc = np.swapaxes(copy.copy(simulation.phot_dmacc[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
		euvvals = np.swapaxes(copy.copy(simulation.EUV[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc =np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		dmacc = np.swapaxes(copy.copy(simulation.phot_wext_dmacc[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)
		euvvals = np.swapaxes(copy.copy(simulation.EUVwext[pinds]),0,1)

	rst = copy.copy(simulation.r)*runits
	vst = copy.copy(simulation.v)*runits/tunits

	

	rst_all =  np.swapaxes(copy.copy(simulation.r),0,1)*runits
	
	ipops = copy.copy(simulation.starinds)
	tends = np.cumsum(np.array(copy.copy(simulation.tends)))
	tends  = np.append(np.array([0.0]), tends)
 
	if popdiv:
		tfirst = get_tfirst(tends, ipops, len(simulation.m))
	else:
		tfirst=0.0

	tages = time-tfirst
	rt= runits*m2pc*copy.copy(simulation.r)[tind]
	vt= 1e-3*runits*copy.copy(simulation.v)[tind]/tunits
	
	r2d = np.swapaxes(np.swapaxes(rt,0,1)[:2],0,1)	
	rad = np.linalg.norm(r2d, axis=1)


	imax = np.argmax(m)
	dmax = np.linalg.norm(rt-rt[imax], axis=1)
	dmaxproj = np.linalg.norm(r2d-r2d[imax], axis=1)

	if popdiv:
		popages  = np.unique(tages)
		stpops = []
		for age in popages:
			stpops.append(np.where(tages==age)[0])
			
	else:
		
		stpops = [np.arange(len(simulation.m))]
	stpops_pind = []
	pind_stpops = []
	for stpop in stpops:
		stpops_pind.append(np.where(np.isin(pinds, stpop))[0])
		pind_stpops.append(np.where(np.isin(stpop, pinds))[0])

	if not os.path.isfile(simulation.out+'_ecc.npy') or not os.path.isfile(simulation.out+'_sma.npy'):
		rmags =np.linalg.norm(rst, axis=2)
		vmags =np.linalg.norm(vst, axis=2)
		eccs_all = np.zeros(rst.shape)
		sma_all = np.zeros(rst.shape)
		grav_param = np.zeros(rmags.shape)
		vsq = vmags*vmags
		r_inv  = 1./rmags

		print(vsq.shape)

		if not os.path.isfile(simulation.out+'_menc.npy'):
			for iit in range(len(rmags)):
				print('Computing m enc for time {0}/{1}'.format(iit+1, len(rmags)))
				for iir in range(len(rmags[iit])):
					iout = np.where(rmags[iit]<1e-8)[0]
					rmags[iit][iout] = 1e50
					iinside = np.where(rmags[iit]<rmags[iit][iir])[0]
					grav_param[iit][iir] = np.sum(m[iinside])
				print(grav_param[iit])
			np.save(simulation.out+'_menc', grav_param)
		else:
			grav_param = np.load(simulation.out+'_menc.npy')
		grav_param *= G_si/kg2sol
		if not os.path.isfile(simulation.out+'_f2.npy'):
			factor2 = np.zeros(rmags.shape)
			for iit in range(len(factor2)):
				for ir in range(len(factor2[iit])):
					factor2[iit][ir] =np.dot(rst[iit][ir], vst[iit][ir])/grav_param[iit][ir]
			np.save(simulation.out+'_f2', factor2)
		else:
			factor2 = np.load(simulation.out+'_f2.npy')
		factor1 = vsq/grav_param - r_inv
		for iit in range(len(eccs_all)):
			for ir in range(len(eccs_all[iit])):
				eccs_all[iit][ir]= factor1[iit][ir]*rst[iit][ir]-factor2[iit][ir]*vst[iit][ir]
		
		np.save(simulation.out+'_ecc', eccs_all)

		specE = vsq/2. - grav_param*r_inv
		sma_all = - grav_param/(2.*specE)
		np.save(simulation.out+'_sma', sma_all*m2pc)

	else:
		eccs_all = np.load(simulation.out+'_ecc.npy')
		sma_all = np.load(simulation.out+'_sma.npy')

	dvmax = vt - vt[imax]
	drmax = rt - rt[imax]
	dvabsmax  = np.array([np.inner(drmax[iv], dvmax[iv]) for iv in range(len(dvmax))])
	print(dvabsmax.shape)
	
	dvabsmax/=np.linalg.norm(drmax,axis=1)
	
	
	eccs_all_mag = np.linalg.norm(eccs_all,axis=2)
	# define our (line) fitting function
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y: (y - fitfunc(p, x))
		
	MS= 10
	mdt = mdisc[tind]
	rdt = rdisc[tind]
	mdat = dmacc[tind]
	eccs_t = eccs_all_mag[tind]
	eccs_t[np.where(np.invert(np.isfinite(eccs_t)))] = 1e-6
	sma_t = sma_all[tind]

	if prop=='mdot' or prop=='radiusIF' or prop=='SHalpha' or True:
		dmdt = -np.gradient(mdisc, tarch*1e6, axis=0)
		mdpt = dmdt[tind]-mdat
		fuv_alt = g0vals[tind]
		euv_alt = euvvals[tind]
		if not os.path.isfile(simulation.out+'_mdotphot_%d.npy'%(tind)):
			mdpt_alt = np.zeros(rdt.shape)
			
			iim=0
			for im in pinds:
				if rdt[iim]>.1:
					mlf = mlc.mloss_function(mstar=m[im])
					mlFUV = mlf.mdot(g0vals[tind][iim], rdt[iim], mdt[iim])
					mlEUV = mlc.mloss_euv(euvvals[tind][iim], rdt[iim])
					print('Mdot FUV:', mlFUV)
					print('Mdot EUV:', mlEUV)
					mdpt_alt[iim] = max(mlFUV, mlEUV)
				print(iim,'/', len(pinds))
				iim+=1
			np.save(simulation.out+'_mdotphot_%d'%(tind), mdpt_alt)
		else:
			mdpt_alt = np.load(simulation.out+'_mdotphot_%d.npy'%(tind))

		

	SBLIM=10.**7.5

	if prop=='radiusIF' or prop=='SHalpha':
		euvctsmassive = simulation.euvcts[np.where(simulation.bigstars==imax)[0]][0]
		euvctsall = simulation.euvcts
		cm2pc = 3.24e-19
		factor=1./(cm2pc**2.*1e49)

		rIF = 100.*np.power(4.*np.pi*euv_alt*factor,-1/3.)*np.power(mdpt_alt/1e-8,2./3.)
		rIF_alt = 100.*dmax[pinds]**(2./3.)*np.power(euvctsmassive/1e49,-1/3.)*np.power(mdpt_alt/1e-8,2./3.)

		alpha_Heff = 1.17e-13
		alpha_A  = 4.2e-13
		alpha_B  = 2.6e-13
		const = 7e11
		sB  = alpha_Heff/alpha_B*const*4.*np.pi*euv_alt*factor/1e2
		sB_alt =alpha_Heff/alpha_B*const*(euvctsmassive/1e49)/(dmax[pinds]*10.0)**2.
		
		euvqcalc = euvctsmassive/(4.*np.pi*(dmax[pinds]/cm2pc)**2)
		euvqcalc_alt = np.zeros(len(pinds))
		ieuv_ct = 0
		for ieuv in simulation.bigstars:
			dtmp = np.linalg.norm(rt-rt[ieuv], axis=1)
			euvqcalc_alt += euvctsall[ieuv_ct]/(4.*np.pi*(dtmp[pinds]/cm2pc)**2)
			ieuv_ct+=1

		plt.scatter(euv_alt, euvqcalc)
		plt.scatter(euv_alt, euvqcalc_alt, c='r', marker='+')
		plt.xscale('log')
		plt.yscale('log')
		plt.show()

		
		plt.scatter(rIF, rIF_alt)
		plt.show()
		sB = sB_alt

	
	

	icol=0
	markers = ['^', 's', 'o']

	dall = np.array([])
	rall = np.array([])

	dall_fuv = np.array([])
	rall_fuv = np.array([])

	out_text = 'ONC_discs_'
	HEAD = 'mhost_Msol, dx_pc, dy_pc, dz_pc, euv_ctscm2, fuv_G0, mdot_wind_Msolyr, mdisc_Msol, rdisc_au' 
	for ipop in stpops_pind:
		pst = np.array(pind_stpops[icol], dtype=int)
		rt_xf, rt_yf, rt_zf = np.swapaxes(rt[stpops[icol][pst]], 0,1)
		dt_xf, dt_yf, dt_zf = np.swapaxes(rt[stpops[icol][pst]]-rt[imax], 0,1)

		if not os.path.isfile(out_text+str(3-icol)+'.csv') or True:
			Xarr = np.swapaxes(np.array([m[stpops[icol][pst]], dt_xf, dt_yf, dt_zf,euv_alt[ipop],fuv_alt[ipop], mdpt_alt[ipop], mdt[ipop], rdt[ipop]]),0,1)
			np.savetxt(out_text+str(3-icol)+'.csv',Xarr,fmt='%.3e', delimiter=', ',header=HEAD)
			
		if rmaxtype=='square':
			iin_surv= np.where((np.absolute(rt_xf)<rmax)&(np.absolute(rt_yf)<rmax)&(mdt[ipop]>MLIM))[0]
		elif rmaxtype=='sepmassive':
			iin_surv= np.where(((dt_xf**2.+dt_yf**2.)<rmax**2.)&(mdt[ipop]>MLIM))[0]
		elif rmaxtype=='centre':
			iin_surv= np.where(((rt_xf**2.+rt_yf**2.)<rmax**2.)&(mdt[ipop]>MLIM))[0]
			
		if var=='mst':
			mstsub = m[stpops[icol][pst]]
			mstmp = np.linspace(0.1,2.0)
		elif var=='dmassive':
			mstsub = np.sqrt(dt_xf**2.+dt_yf**2.)#dmaxproj[stpops[icol][pst]]
			mstmp= np.logspace(-3, 1.)
		elif var=='eccorb':
			print('Computing eccentricities...')
			mstsub = eccs_t[stpops[icol][pst]]
			sma_i = sma_t[stpops[icol][pst]]
			mstmp = np.linspace(0.,2.)
		elif var=='periorb':
			mstsub = sma_t[stpops[icol][pst]]*np.sqrt(np.absolute(1.-eccs_t[stpops[icol][pst]]**2.))
			sma_i = eccs_t[stpops[icol][pst]]
			mstmp = np.linspace(0.,2.)
		elif var=='rdisc':
			mstsub = rdt[ipop]
		elif var=='mdisc':
			mstsub = mdt[ipop]
				
		
		fuvsub = fuv_alt[ipop]
		dvmax_i = dvabsmax[stpops[icol][pst]]
		if prop=='mass':
			mdsub =  mdt[ipop]
			pinit = [-2., 0.2]
			logxtmp = np.log10(mstsub[iin_surv])
			logytmp = np.log10(mdsub[iin_surv])
			out = optimize.leastsq(errfunc, pinit, args=(logxtmp, logytmp), full_output=1)
			pfinal = out[0]
			covar = out[1]

			index = pfinal[1]
			amp = 10.0**pfinal[0]
			print(3-icol, index)
			if popdiv:
				plt.plot(mstmp, amp*mstmp**index, c=CB_color_cycle[icol+3],linestyle='dashed')
		elif prop=='radius':
			mdsub=rdt[ipop]

			pinit = [1., 0.2]
			"""logxtmp = np.log10(mstsub[iin_surv])
			logytmp = np.log10(mdsub[iin_surv])
			out = optimize.leastsq(errfunc, pinit, args=(logxtmp, logytmp), full_output=1)
			pfinal = out[0]
			covar = out[1]

			index = pfinal[1]
			amp = 10.0**pfinal[0]
			print(3-icol, index)
			plt.plot(mstmp, amp*mstmp**index, c=CB_color_cycle[icol+3],linestyle='dashed')"""
		elif prop=='mdot':
			if var!='eccorb' and var!='periorb':
				mdsub=mdat[ipop]
				mdpsub = mdpt_alt[ipop]
				mrat = mdpsub/mdsub
				mrat = mrat[np.where(np.isfinite(mrat)&~np.isnan(mrat))]
				ax.scatter(mstsub[iin_surv], mdpsub[iin_surv], marker=markers[icol],s =MS,  facecolors='none', edgecolors=CB_color_cycle[icol+3], alpha=0.8)
				msttmp = mstsub[iin_surv]
				mdptmp = mdpsub[iin_surv]
				iab = np.where(mdptmp>MDLIM)[0]
				rho, p = spearmanr(msttmp[iab], b=mdptmp[iab])
			else:
				mdsub =   mdpt_alt[ipop]
		elif prop=='radiusIF':
			mdsub = rIF[ipop]
			euvsub = euv_alt[ipop]
			mstsub_act = m[stpops[icol][pst]]
		elif prop=='SHalpha':
			mdsub = sB[ipop]
			pinit = [1., 0.2]
			logxtmp = np.log10(mstsub[iin_surv])
			logytmp = np.log10(mdsub[iin_surv])
			out = optimize.leastsq(errfunc, pinit, args=(logxtmp, logytmp), full_output=1)
			pfinal = out[0]
			covar = out[1]

			index = pfinal[1]
			amp = 10.0**pfinal[0]
			print(3-icol, index)
			#plt.plot(mstmp, amp*mstmp**index, c=CB_color_cycle[icol+3],linestyle='dashed')
		else:
			print('Plot property not recognised ({0})'.format(prop))
		if prop=='radiusIF':
			#fracprops = int(100.*float(len(np.where(mdsub[iin_surv]>1.0)[0]))/float(len(iin_surv)))
			
			mstsubsv= mstsub[iin_surv]
			mdsubsv = mdsub[iin_surv]
			sBsubsv = sB[iin_surv]
			ifuv = np.where(sBsubsv>SBLIM)[0]
			ilowfuv = np.where(sBsubsv<SBLIM)[0]
			dall = np.append(dall, mstsubsv)
			rall = np.append(rall, mdsubsv)
			dall_fuv = np.append(dall_fuv, mstsubsv[ifuv])
			rall_fuv = np.append(rall_fuv, mdsubsv[ifuv])
			ax.scatter(mstsubsv[ifuv], mdsubsv[ifuv], marker=markers[icol],s =MS, c=CB_color_cycle[icol+3], alpha=0.8, label='Pop. {0}'.format(3-icol))
			ax.scatter(mstsubsv[ilowfuv], mdsubsv[ilowfuv], marker=markers[icol],s =MS, facecolors='None', edgecolor=CB_color_cycle[icol+3], alpha=0.8)
			#plt.scatter(mstsub[iin_surv], mdsub[iin_surv], marker=markers[icol],s =MS, c=CB_color_cycle[icol+3], alpha=0.8, label='Pop. {0}: {1} \%'.format(3-icol,fracprops))
		else:
			if popdiv or cbv==None:
				if len(stpops_pind)>1:
					ax.scatter(mstsub[iin_surv], mdsub[iin_surv], marker=markers[icol],s =MS, c=CB_color_cycle[icol+3], alpha=0.8, label='Pop. %d'%(3-icol))
				else:
					ax.scatter(mstsub[iin_surv], mdsub[iin_surv], marker=markers[icol],s =MS, c='k', alpha=0.8)
					rho_alt, p_alt = spearmanr(mstsub[iin_surv], b=mdsub[iin_surv])
					ax.axhline(np.median(mdsub[iin_surv]), color='k', alpha=0.2)
					if tvals==None:
						ax.text(0.51, 0.1, "$\\rho =$ %.2lf (%.2lf)\n$\log p = $%.1lf (%.1lf)"%(rho,rho_alt,np.log10(p),np.log10(p_alt )), transform=ax.transAxes, color='k')
					else:
						ax.text(0.51, 0.1, "$\\rho =$ %.2lf (%.2lf,\\textcolor{blue}{%.2lf})\n$\log p =$ %.1lf (%.1lf, \\textcolor{blue}{%.1lf})"%(rho,rho_alt,tvals[0], np.log10(p),np.log10(p_alt ), np.log10(tvals[1])), transform=ax.transAxes)
			else:
				if cbv=='eccorb':
					print(mstsub[iin_surv])
					scp = plt.scatter(mstsub[iin_surv], mdsub[iin_surv], marker=markers[icol],s =MS, c=sma_i[iin_surv], alpha=0.8, vmin=0.0, vmax=1.0)
					plt.colorbar(scp, label='Orbital eccentricity')
				elif cbv =='dvmax':
					scp = plt.scatter(mstsub[iin_surv], mdsub[iin_surv], marker=markers[icol],s =MS, c=dvmax_i[iin_surv], alpha=0.8, vmin=-10.0, vmax=10.0)
					plt.colorbar(scp, label='$\Delta v_{\\theta_1 C}$ (km/s)')
				elif cbv =='G0':
					scp = plt.scatter(mstsub[iin_surv], mdsub[iin_surv], marker=markers[icol],s =MS, c=fuvsub[iin_surv], alpha=0.8, norm=LogNorm(vmin=1e1, vmax=1e3))
					plt.colorbar(scp, label='$F_{\\mathrm{FUV}}$ ($G_0$)')
					dmpsub = dmaxproj[stpops[icol][pst]]
					np.save('SEPMDG0', np.array([mstsub[iin_surv], mdsub[iin_surv], fuvsub[iin_surv], dmpsub[iin_surv]]))
				else:
					print('Colour bar label not recognised: {0}'.format(cbv))
					exit()
			"""if var=='eccorb':
				#mdsub=mdat[ipop]
				xv =mstsub[iin_surv]
				yv =mdsub[iin_surv]
				ixs = np.argsort(xv)
				xvs = xv[ixs]
				yvs = yv[ixs]
				xsp = np.array_split(xvs, 5)
				ysp = np.array_split(yvs, 5)
				for ixsp in range(len(xsp)):
					xupl = np.percentile(xsp[ixsp], 84.0)
					xlol = np.percentile(xsp[ixsp], 16.0)
					xmed = np.percentile(xsp[ixsp], 50.0)

					yupl = np.percentile(ysp[ixsp], 84.0)
					ylol = np.percentile(ysp[ixsp], 16.0)
					ymed = np.percentile(ysp[ixsp], 50.0)
					print(xmed, ymed, ylol, yupl, xlol, xupl)
					plt.errorbar(xmed, ymed, yerr=[[ymed-ylol], [yupl-ymed]], xerr=[[xmed-xlol], [xupl-xmed]], color='r')"""

		icol+=1


	rt_xf, rt_yf, rt_zf = np.swapaxes(rt[pinds], 0,1)
	iin_surv_all = np.where((np.absolute(rt_xf)<rmax)&(np.absolute(rt_yf)<rmax)&(mdt>MLIM)&(rdt>0.1))[0]	

	print('Number of stars:', len(iin_surv_all))

	mst_sub = m[pinds]
	logx = np.log10(mst_sub[iin_surv_all])

	
	if prop=='mass':	
		pinit = [-10.,1.]
		logy = np.log10(mdt[iin_surv_all])
	elif prop=='radius':
		pinit = [1.,1.]
		logy = np.log10(rdt[iin_surv_all])
	elif prop=='mdot':
		pinit = [-7.0,1.]
		logy = np.log10(mdpt_alt[iin_surv_all])
	elif prop=='radiusIF':
		pinit = [1.,1.]
		logy = np.log10(rIF[iin_surv_all])
		logRIF = logy
		logSB = np.log10(sB[iin_surv_all])
	elif prop=='SHalpha':
		pinit = [10.,-1.]
		logy = np.log10(sB[iin_surv_all])
		logSB = logy
		logRIF = np.log10(rIF[iin_surv_all])


	ireal = np.where(np.isfinite(logy))[0]
	out = optimize.leastsq(errfunc, pinit, args=(logx[ireal], logy[ireal]), full_output=1)

	pfinal = out[0]
	covar = out[1]

	index = pfinal[1]
	amp = 10.0**pfinal[0]

	indexErr = np.sqrt( covar[1][1] )
	ampErr = np.sqrt( covar[0][0] )
	
	print('Index M_d (m_s)  - all stars: {0} pm {1}'.format(index, indexErr))
	print('with amplitude: {0} pm {1}'.format(amp, ampErr))

		
	m_d_sp = np.linspace(0.1,2.0,20)
	m_disc_sp = amp*m_d_sp**index
	pp =(amp*ampErr)*m_d_sp**(index+indexErr) 
	pm =  (amp/ampErr)*m_d_sp**(index+indexErr)
	mp =  (amp*ampErr)*m_d_sp**(index-indexErr)
	mm = (amp/ampErr)*m_d_sp**(index-indexErr) 
	m_disc_ul = np.amax(np.array([pp,mp,pm,mm]),axis=0)
	m_disc_ll = np.amin(np.array([pp,mp,pm,mm]),axis=0)
	
	#print(m_d_sp, m_disc_sp)
	#plt.plot(m_d_sp, m_disc_ul, c='k', linewidth=BARLINE, linestyle='dashed')
	#plt.plot(m_d_sp, m_disc_ll, c='k', linewidth=BARLINE, linestyle='dashed')
	if var=='mst':
		ax.set_xlabel('Stellar host mass -- $m_*$ ($M_\\odot$)')
		ax.set_xscale('log')
		ax.set_xlim([0.1,2.])
		plt.tick_params(axis='x', which='both', left=False, right=False,labelleft=False) # labels along the bottom edge are off
		ax.xaxis.set_major_formatter(NullFormatter())
		ax.xaxis.set_minor_formatter(NullFormatter())
		ax.set_xticks([0.1, 0.2, 0.5, 1.0,2.0]) 
		ax.set_xticklabels([0.1, 0.2, 0.5, 1.0,2.0]) 
	elif var=='dmassive':
		ax.set_xlim([0.0,rmax])


		#plt.xlabel('Projected distance to $\\theta^1$C (pc)')
		ax.set_xlabel('Projected distance to $\sigma$ Ori (pc)')
	elif var=='eccorb':
		ax.set_xscale('log')
		ax.set_xlim([0.01, 1.0])
		plt.xlabel('Orbital eccentricity')
	elif var=='periorb':
		#plt.xscale('log')
		#plt.xlim([0.01, 1.0])
		ax.set_xlabel('Instantaneous semi-minor axis (pc)')
	elif var=='rdisc':
		ax.set_xlabel('Outer disc radius -- $R_\mathrm{disc}$(au)')
		ax.set_xlim([0., 100.])
	elif var=='mdisc':
		ax.set_xlabel('Disc mass -- $M_\mathrm{disc}$($M_\odot$)')
		ax.set_xscale('log')
		ax.set_xlim([1e-4, 1e-1])
	


	if prop=='mass':
		plt.plot(m_d_sp, m_disc_sp, c='k', linewidth=BARLINE)
		mearth2msun = 3e-6
		m_disc_eis = 4.0*1e2*mearth2msun*m_d_sp**0.25
		#plt.plot(m_d_sp, m_disc_eis, c='r', linestyle='dashed', linewidth=BARLINE)
		plt.ylabel('$M_\\mathrm{disc}$ ($M_\\odot$)')
		plt.yscale('log')
		plt.ylim([MLIM, 1e-1])
		if popdiv:
			plt.legend(loc='best')
	elif prop=='radius':
		#plt.plot(m_d_sp, m_disc_sp, c='k', linewidth=BARLINE)
		plt.ylabel('$R_\\mathrm{disc}$ (au)')
		plt.ylim([5., 200.0])
		plt.tick_params(axis='y', which='both', bottom=False, top=False,labelbottom=False) # labels along the bottom edge are off
		
		ax.set_yscale('log')
		ax.yaxis.set_major_formatter(NullFormatter())
		ax.yaxis.set_minor_formatter(NullFormatter())
		ax.set_yticks([5.,10., 20., 50., 100., 200.])
		ax.set_yticklabels([5.,10.,20.,50.,100.,200.])
		if popdiv:
			plt.legend(loc='best')
	elif prop=='radiusIF':

		if var=='dmassive':
			rIFobs = np.array([18.2, 1.2, 3.2, 1.9, 17.0,2.5,6.3,11.3,16.6,4.4,5.,20.1,5.3, 3.5, 9.1,5.0,2.2,2.5,7.9,6.3,2.8,2.8,12.2,4.7,23.3, 4.1,6.9,20.4,12.2])
			rIFobserr = np.array([2.2,0.3,0.3,0.3,2.5,0.3,0.6,0.6,1.6,0.6,0.3,1.6,1.9,0.3,1.0,0.6,0.6,0.6,0.3,0.6,0.3,0.3,1.2,0.3,1.6, 0.3,1.0,1.6,1.2])
			dtheta1C = np.array([19.16,16.97,16.63,27.21,20.48,10.97,9.42,9.6,10.6,22.75,17.20,28.35, 10.24, 6.05, 7.74, 6.91,2.14,7.01,7.83,6.6,6.64, 16.47, 16.20, 14.29, 19.11, 22.48, 16.38, 25.84,25.12])

			rIFobs*=6.685
			rIFobserr*=6.685

			rIFvic = np.array([472.,326.,648.,268.,525.,302.,678.,543.,734.,2520.])
			rIFvic/=2.6

			#dva, rIFva = np.load('V_A_rIF.npy')
		
			dtheta1Cvic = np.array([0.38,0.27,0.27,0.13,0.31,0.09,0.12,0.17,0.23,0.31])

			asec2rad = 4.85e-6
			DIST=414.0
			dtheta1C *= asec2rad*DIST
			#dva *= asec2rad*DIST

			#rIFva*=2./2.6

			bins = [0.,0.1,0.2,0.3,0.4]
			RLIM = 100.0
			"""Ngtr50_VA = []
			Ngtr50_sim = []
			Ngtr50_sim_hfuv = []
			for ibin in range(len(bins)-1):
				Ngtr50_VA.append(float(len(np.where((dva<bins[ibin+1])&(dva>bins[ibin])&(rIFva>RLIM))[0])))
				Ngtr50_VA[-1] += float(len(np.where((dtheta1Cvic<bins[ibin+1])&(dtheta1Cvic>bins[ibin])&(rIFvic>RLIM))[0]))

				Ngtr50_sim.append(float(len(np.where((dall<bins[ibin+1])&(dall>bins[ibin])&(rall>RLIM))[0])))
				Ngtr50_sim_hfuv.append(float(len(np.where((dall_fuv<bins[ibin+1])&(dall_fuv>bins[ibin])&(rall_fuv>RLIM))[0])))
				

			bins = np.array(bins)
			bins_mp = (bins[:-1]+bins[1:])/2.
			Ngtr50_VA = np.array(Ngtr50_VA)
			Ngtr50_sim = np.array(Ngtr50_sim)
			Ngtr50_sim_hfuv = np.array(Ngtr50_sim_hfuv)"""

			plt.errorbar(dtheta1C, rIFobs, yerr=rIFobserr, marker='x',linestyle='None',markersize=5, color='k',label='Henney \& Arthur (1998)') 
			#plt.errorbar(dva, rIFva,yerr=67.5/2.6, marker='x',linestyle='None', markersize=5,label='Vicente \& Alves (2005)', color='g')
			#plt.errorbar(dtheta1Cvic, rIFvic,yerr=22.5, marker='x',markersize=5, linestyle='None', label='Silhouettes - V\&A (2005)', color='y')
		

		plt.ylabel('$R_\\mathrm{IF}$ (au)')
		plt.ylim([5., 2000.0])
		plt.tick_params(axis='y', which='both', bottom=False, top=False,labelbottom=False) # labels along the bottom edge are off
		
		ax.set_yscale('log')
		ax.yaxis.set_major_formatter(NullFormatter())
		ax.yaxis.set_minor_formatter(NullFormatter())
		ax.set_yticks([5., 10., 20., 50., 100., 200., 500.0, 1000.0, 2000.0])
		ax.set_yticklabels([5., 10.,20.,50., 100., 200., 500.0,1000.,2000.])
		plt.legend(loc=1, fontsize=8)
	elif prop=='mdot':
		ax.set_ylabel('Mass-loss rate ($M_\\odot$ yr$^{-1}$)')
		ax.set_yscale('log')
		ax.set_ylim([1e-11, 1e-7])
		ax.axhline(1e-10, c='r')
		ax.axhline(MDLIM, c='r', linestyle='dashed')
		if popdiv:
			plt.legend(loc='best')
	elif prop=='SHalpha':
		plt.ylabel('$\langle S(\mathrm{H}\\alpha) \\rangle$ (s$^{-1}$ cm$^{-2}$ s.r.$^{-1}$)')

		plt.axhline(10.**10.75, c='k', linestyle='dashed')
		plt.yscale('log')
		plt.ylim([3e9, 1e14])

		if var=='dmassive':
			sHalpha_obs_log = np.array([11.87,12.24,12.82,12.26,12.05, 12.06, 11.90, 11.36,13.34,12.77, 12.96,12.93,13.22,12.48,11.40,10.87, 12.03, 11.72, 11.87, 10.96, 10.75, 10.80])
		
			phi_asec = np.array([0.46,0.19,0.20,0.29,0.40,0.23,0.70,0.71,0.10,0.23,0.16,0.28, 0.13,0.37,0.59, 0.23,0.26,0.64,0.35,1.19,0.49,0.30])

			theta_asec = np.array([20.4,11.0,9.4,9.6,10.6,17.4,28.3,31.1,6.1,7.6,6.9,7.8,6.6,16.2,19.1,53.1, 16.6,25.8,25.1,57.3, 62.7,70.2])
			asec2rad = 4.85e-6
			DIST=414.0

	
			DOde = theta_asec*asec2rad*DIST


			plt.scatter(theta_asec*asec2rad*DIST, 10.**sHalpha_obs_log, marker='+', c='k', label="O'Dell (1998)")
			d_space = np.logspace(-2, 1)
			sHalpha = 5.56e14*(d_space/(DIST*asec2rad))**-2.
			plt.plot(d_space, sHalpha, c='k')
			#plt.xscale('log')
			#plt.xticks([0.01, 0.03,0.1], [0.01, 0.03,0.1])
			#plt.xlim([0.01,rmax])

		
		plt.legend(loc='best')

	sHalpha_obs_log = np.array([11.87,12.24,12.82,12.26,12.05, 12.06, 11.90, 11.36,13.34,12.77, 12.96,12.93,13.22,12.48,11.40,10.87, 12.03, 11.72, 11.87, 10.96, 10.75, 10.80])
	rhen = np.array([17.0, 2.5, 6.3, 11.3, 16.6, 5.0,20.1,-1., 3.5,9.1,5.0,7.9,6.3,12.2,23.3,-1.,6.9,20.4,12.2, -1.,-1.,-1.])
	rhenpm = np.array([2.5,0.3,0.6,0.6,1.6, 0.3,1.6, 0.0, 0.3,1.0, 0.6, 0.3, 0.6, 1.2, 1.6, 0.0, 1.0, 1.6, 1.2, 0.,0.,0.])
	Dhen = np.array([2.82, 0.87, 0.92, 2.55, 1.03, 1.23,2.02,-1.,0.59,0.75,0.67,0.62,0.52, 1.29,1.86,-1.,1.17,2.5,2.0,-1.,-1.,-1.])

	rhen*=6.685
	rhenpm*=6.685
	Dhen*=0.0324

	if var=='mst':
		vstr='ms'
	elif var=='dmassive':
		vstr = 'dm'
	elif var=='eccorb':
		vstr= 'ecc'
	elif var=='periorb':
		vstr= 'b'
	elif var=='rdisc':
		vstr= 'rd'
	elif var=='mdisc':
		vstr= 'md'

	if prop=='mass':
		pstr = 'md'
	elif prop=='radius':
		pstr='rd'
	elif prop=='radiusIF':
		pstr = 'rIF'
	elif prop=='mdot':
		pstr = 'mdot'
	elif prop=='SHalpha':
		pstr = 'SH'


	if not axparse:
		plt.savefig('paper_figure_'+vstr+pstr+'_1e-3_'+str(tind)+'.pdf', bbox_inches='tight', format='pdf')
		

		if show:
			plt.show()
		else:
			plt.close()
	else:
		return None

	if prop=='radiusIF' or prop=='SHalpha':
		pc2au =206265.
		plt.figure(figsize=(4.,4.))
		plt.errorbar(rhen, 10.**sHalpha_obs_log, xerr=rhenpm, marker='s',alpha=0.5,  c='r',linestyle='None', label="O'Dell (1998)")
		icol=0		
		for pop in stpops_pind:
			plt.scatter(rIF[pop], sB[pop], marker='+', c=CB_color_cycle[icol+3], s=10, label='Pop. %d'%(3-icol))
			icol+=1
		
		plt.xscale('log')
		plt.yscale('log')
		plt.ylim([1e9, 1e14])
		plt.xlim([1e1, 1e3])
		plt.xlabel('$R_\mathrm{IF}$ (au)')
		plt.ylabel('$\langle S(H\\alpha) \\rangle$ (s$^{-1}$ cm$^{-2}$ s.r.$^{-1}$)')
		plt.axvline(22.5, linestyle='dashed')
		plt.legend(loc=1)
		plt.savefig('paper_figure_RIFSH_1e-3_'+str(tind)+'.pdf', bbox_inches='tight', format='pdf')
		if show:
			plt.show()
		else:
			plt.close()
	

	if prop=='radiusIF':

		fig, ax = plt.subplots(figsize=(4.,4.))
		dbin =bins_mp[1]-bins_mp[0]
		plt.bar(bins_mp, Ngtr50_sim, dbin,align='center',facecolor='None',edgecolor='r', label='Model (all)')
		plt.bar(bins_mp, Ngtr50_sim_hfuv, dbin,align='center',facecolor='None',edgecolor='r',linestyle='dashed', label='Model ($\log \langle S(\mathrm{H}\\alpha) \\rangle > %.2lf$ cm$^{-2}$ s$^{-1}$ s.r.$^{-1}$)'%(np.log10(SBLIM)))
		plt.bar(bins_mp, Ngtr50_VA, dbin,align='center',facecolor='None', edgecolor='b', label='Vicente \& Alves (2005)')
		plt.ylabel('Number of proplyds ($R_\mathrm{IF}>50$ au)')
		plt.xlabel('Distance to $\\theta^1$C (pc)')
		plt.xlim([0.0,0.4])
		plt.legend(loc='best')
		plt.savefig('paper_figure_RIF50_1e-3_'+str(tind)+'.pdf', bbox_inches='tight', format='pdf')
		
		if show:
			plt.show()
		else:
			plt.close()

def plot_mstmdisc(simulation, time=0.0, wext=False, g0=1.6e-3, plot=True, fit='odr', show=False):
	
	tind=  0
	
		
	if not wext:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc = 1e-2/3.00273e-6*np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc =1e-2/3.00273e-6*np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)

	if len(simulation.assoc.shape)>=1:
		assoc = copy.copy(simulation.assoc[pinds])
	else:
		assoc = np.array(np.ones(len(pinds)), dtype=int)



	print('Subset size:', len(pinds))
	MS= 2

	t = copy.copy(simulation.t)
	m_d = copy.copy(simulation.m)[pinds]
	tunits, munits, runits = copy.copy(simulation.units)

	tarch = t*tunits*s2myr
	tind = np.argmin(np.absolute(tarch-time))
	tval = tarch[tind]
	print('Time:', tarch[tind])
	m_d *= munits*kg2sol


	mall=  munits*kg2sol*copy.copy(simulation.m)
	im_b =np.where(mall>=1.0)[0]
	m_b = mall[im_b]
	im_m = np.where(mall>=30.0)[0]

	mall=  munits*kg2sol*copy.copy(simulation.m)
	im_b =np.where(mall>=1.0)[0]
	m_b = mall[im_b]
	im_m = np.where(mall>=30.0)[0]

	rt_d = runits*m2pc*copy.copy(simulation.r)[tind][pinds]
	rt_b = runits*m2pc*copy.copy(simulation.r)[tind][im_b]
	rt_m = runits*m2pc*copy.copy(simulation.r)[tind][im_m]

	
	rt_i = runits*m2pc*copy.copy(simulation.r)[0][pinds]

	
	rt_xf, rt_yf, rt_zf = np.swapaxes(rt_d, 0,1)
	rtf_xm, rtf_ym, rtf_zm = np.swapaxes(rt_m, 0,1)
	rt_x, rt_y, rt_z = np.swapaxes(rt_i, 0,1)


	Lums = get_FUVluminosities(m_b)

	fluxes  = cluster_calcs.flux(rt_d*1e2/m2pc, Lums,rt_b*1e2/m2pc,2)	
	fluxes /= g0


	massbins = [[0.3, 0.6],[0.6,1.0],[1.0,2.0]]*5
	

	

	def logsig_func(logm, dmin=0.05, dmax1=0.1, dmax2=0.4, dist=True):
		sigs = np.ones(len(logm))*dmin
		dmax = dmax1 - dmax2*logm

		idist = np.where(dmax>dmin)[0]
		if dist:
			rnums = np.random.uniform(size=len(idist))
			print(rnums)
			sigs[idist]= dmin+rnums*dmax[idist]
		else:
			sigs[idist]= dmax[idist]
		return sigs
		
		


	def fit_pop_lm(indices, popcol, label='', plot=plot):
		M3sig = 0.01
		mdtmp = np.copy(mdisc[tind][indices])
		delta = mdtmp>M3sig
		notdelta = np.logical_not(delta)
		ind_d = np.where(delta)[0]
		mdtmp[np.where(mdtmp<M3sig)[0]] = M3sig
		y = np.log10(mdtmp)
		x = np.log10(m_d[indices])
		#ysig = logsig_func(y, dmin=0.03, dmax1=0.2, dmax2=0.2, dist=False)
		#xsig =  logsig_func(x, dmin=0.05, dmax1=0.1, dmax2=0.3, dist=True)
		if True:
			print("Don't use linmix fit")
			exit()

			xsig = np.load('obs_mst_sig.npy')
			xsig = np.random.choice(xsig, size=len(x))
			ysig = np.load('obs_md_sig.npy')
			ysig = np.random.choice(ysig, size=len(y))
		

		lmcens  = linmix.LinMix(x, y, xsig, ysig, delta=delta, K=2)
		lmcens.run_mcmc(silent=True, maxiter=10000)
		

		fbeta = []

		NTRUNC = 1000

		beta_val = np.mean(lmcens.chain[NTRUNC:]['beta'])
		ibeta = np.argmin(np.absolute(beta_val-lmcens.chain[NTRUNC:]['beta']))
		alpha_val = np.mean(lmcens.chain[NTRUNC:]['alpha'])
		berr= np.std(lmcens.chain[NTRUNC:]['beta'])
		aerr= np.std(lmcens.chain[NTRUNC:]['alpha'])
		
		xs =np.linspace(-2.0, 1.0)
		ys = beta_val*xs+alpha_val
		if plot:
			#ax.plot(10.**xs, 10.**ys, color=popcol,  label='$\\beta = %.2lf \pm %.2lf$, $\\alpha = %.2lf \pm %.2lf$'%(beta_val,berr, alpha_val,aerr))
			ax.plot(10.**xs, 10.**ys, color=popcol,  label='$\\beta = %.2lf$, $\\alpha = %.2lf$'%(beta_val, alpha_val))

			#ax.plot(10.**xs, 10.**ys_alt, color='k', ls ='dashed',label='$\\beta = 2$')
			ax.scatter(10.**x[delta], 10.**y[delta], marker='o', facecolor=popcol, s=60, edgecolor='gray', linewidth=1, zorder=100)
			ax.errorbar(10.**x[delta], 10.**y[delta], xerr=10.**x[delta]*xsig[delta]/0.434, yerr=10.**y[delta]*ysig[delta]/0.434, ls=' ', fmt='o', ms=0, ecolor='gray', zorder=90)
			ax.scatter(10.**x[notdelta], 10.**y[notdelta], marker='v', facecolor=popcol, edgecolor='gray')
		return beta_val, alpha_val


	def odr_fit(xv, yv):
		def f(B, x):
			print(B[0], B[1])
			return B[0]*x + B[1]
		linear = odr.Model(f)
		mydata = odr.RealData(xv, yv)
		odrobj = odr.ODR(mydata, linear, beta0=[1.5, 2.])
		out = odrobj.run()
		out.pprint()
		return out.beta, out.sd_beta




	def fit_pop_odr(indices, popcol, label='',llim=1.0, plot=plot):
		mdtmp = np.copy(mdisc[tind][indices])
		delta =  mdtmp>llim
		ind_d = np.where(delta)[0]


		
		y = np.log10(mdtmp)
		x = np.log10(m_d[indices])

		y = y[ind_d]
		x = x[ind_d]

		#ysig = logsig_func(y, dmin=0.03, dmax1=0.2, dmax2=0.2, dist=False)
		#xsig =  logsig_func(x, dmin=0.05, dmax1=0.1, dmax2=0.3, dist=True)

		ab, aberr = odr_fit(x,y)
		
		beta_val, alpha_val = ab
		xs =np.linspace(-2.0, 1.0)
		ys = beta_val*xs+alpha_val
		if plot:
			#ax.plot(10.**xs, 10.**ys, color=popcol,  label='$\\beta = %.2lf \pm %.2lf$, $\\alpha = %.2lf \pm %.2lf$'%(beta_val,berr, alpha_val,aerr))
			ax.plot(10.**xs, 10.**ys, color=popcol,  label='$\\beta = %.2lf$, $\\alpha = %.2lf$'%(beta_val, alpha_val))

			#ax.plot(10.**xs, 10.**ys_alt, color='k', ls ='dashed',label='$\\beta = 2$')
			ax.scatter(10.**x, 10.**y, marker='o', facecolor=popcol, s=60, edgecolor='gray', linewidth=1, zorder=100)
		return beta_val, alpha_val


	DIV = 400.0

	#inds_subsel = np.where((fluxes>1e2)&(fluxes<DIV))[0]
	#inds_subsel2 = np.where(fluxes>DIV)[0]
	if plot:
		fig = plt.figure(figsize=(4,4))
		ax = fig.add_subplot(111)
	inds_massive = np.where((m_d>0.1)&(m_d<2.0))[0]
	#inds_subsel = np.random.choice(inds_massive, size=200, replace=False)
	if fit=='linmix':
		beta, alpha = fit_pop_lm(inds_massive, 'orange')# label='$F_\mathrm{FUV} < %d \, G_0$'%(DIV))
	elif fit=='odr':
		beta, alpha = fit_pop_odr(inds_massive, 'orange')
	else:
		print('Fitting method not recognised.')
		exit()
	#fit_pop(inds_subsel2, 'green', label='$F_\mathrm{FUV} > %d \, G_0$'%(DIV))

	
	if plot:	
		ax.set_xlabel('$m_*$ ($M_\\odot$)')
		ax.set_ylabel('$M_\\mathrm{dust}$ ($M_\\oplus$)')
		ax.set_yscale('log')
		ax.set_xscale('log')
		plt.tick_params(axis='x', which='both', left=False, right=False,labelleft=False) # labels along the bottom edge are off
		ax.xaxis.set_major_formatter(NullFormatter())
		ax.xaxis.set_minor_formatter(NullFormatter())
		ax.set_xticks([0.1,0.2,0.3,0.5, 0.7, 1.0, 1.5, 2.0]) 
		ax.set_xticklabels([0.1,0.2, 0.3,0.5, 0.7, 1.0, 1.5, 2.0]) 
		#ax.set_xlim(0.1,2.0)
		#ax.set_ylim(3e-1,2e2)
		ax.legend(loc='best')
		#ax.plot([-6,6,6,-6,-6], [-3,-3,4,4,-3], color='k')
		fig.tight_layout()
		plt.savefig('paper_figure_betamupp.pdf', bbox_inches='tight', format='pdf')
		if show:
			plt.show()
		else:
			plt.close()

	return tval, beta, alpha

def plot_mstmdisc_times(simulation, wext=False, show=False):

	if not os.path.isfile(simulation.out+'_abvals.npy'):
		treq = np.linspace(0.0, 5.0, 50)
		times = np.zeros(len(treq))
		beta = np.zeros(len(treq))
		alpha = np.zeros(len(treq))
		for it in range(len(treq)):
			times[it], beta[it], alpha[it]= plot_mstmdisc(simulation, time=treq[it], wext=wext,plot=False)
		np.save(simulation.out+'_abvals', np.array([times, alpha, beta]))
	else:
		times, alpha, beta = np.load(simulation.out+'_abvals.npy')

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.figure(figsize=(4.,4.))
	plt.plot(times, beta, color='k', linestyle='solid', label='$\\beta$')
	plt.plot(times, alpha, color='k', linestyle='dashed', label='$\\alpha$')
	plt.xlabel('Time (Myr)')
	plt.ylabel('Power law params. ($\\alpha$, $\\beta$)')
	plt.xlim([0.0, 5.0])
	plt.ylim([1.0, 3.0])
	plt.legend(loc='best')
	plt.savefig(simulation.out+'_abevol.pdf', format='pdf', bbox_inches='tight')
	
	if show:
		plt.show()
	else:
		plt.close()

	

	


def plot_cumfrac(simulation, time=0.0, wext=False, g0=1.6e-3, rmax=None, centremmax=True):

	msol2earth = 3.3e5
	if not wext:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc =np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)


	r =copy.copy(simulation.r)
	t= copy.copy(simulation.t)
	m= copy.copy(simulation.m)

	
	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol


	rvals = np.swapaxes(np.swapaxes(r,0,1)[pinds],0,1)
	

	tind = np.argmin(np.absolute(t-time))

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

		
	plt.figure(figsize=(4.,4.))

	it=0
	if type(time)==list:
		lsts = ['solid', 'dashed', 'dotted']
		for tval in times:
			tind = np.argmin(np.absolute(tarch-tval))
			mdisc_copy = copy.copy(mdisc[tind])
			if type(rmax)!=type(None):
				rvals_copy = copy.copy(rvals[tind])
				x,y,z = np.swapaxes(rvals_copy,0,1)[:]
				invals = np.where(x**2+y**2<rmax**2)[0]
			else:
				invals = np.arange(len(rvals[tind]))

			X =  mdisc_copy[invals]*1e-2*msol2earth
			bins = np.unique(np.sort(X))
			n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=-1, bins=bins, edgecolor='k', linestyle=lsts[it])
			plt.plot([],[], linestyle=lsts[it],color='k', label='$t = %.1lf$ Myr'%(times[it]))
			it+=1
	else:
		tind = np.argmin(np.absolute(tarch-time))
		mdisc_copy = copy.copy(mdisc[tind])
		
		rvals_copy = copy.copy(rvals[tind])
		x,y,z = np.swapaxes(rvals_copy,0,1)[:]
		if centremmax==True:
			imax = np.argmax(m)
			xall, yall, zall = np.swapaxes(r[tind], 0,1)[:]
			xmmax = xall[imax]
			ymmax = yall[imax]

			print('Recentering: ({0},{1})'.format(xmmax, ymmax))
			
			x -= xmmax
			y -= ymmax

		if type(rmax)!=type(None):
			invals = np.where(x**2+y**2<rmax**2)[0]
		else:
			invals = np.arange(len(rvals[tind]))

		
		invals2 = np.where(x**2+y**2<9.0)[0]
		X =  mdisc_copy[invals2]*1e-2*msol2earth
		bins = np.unique(np.sort(X))
		n2, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=-1, bins=bins, edgecolor='k', label='All')

		X =  mdisc_copy[invals]*1e-2*msol2earth
		n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=-1, bins=bins, edgecolor='k', linestyle='dashed', label='$<0.15$ pc from $\\theta^1$C')

	np.save(simulation.out+'_mdcf', np.array([bins[1:], n, n2]))

	plt.xscale('log')
	plt.ylim([0.,1.])

	plt.axvline(MLIM*1e-2*msol2earth)
	
	if type(time)==list:
		plt.legend(loc='best')
	
	plt.xlabel('$M$ ($M_\oplus$)')
	plt.ylabel('$P(M_\mathrm{dust}\geq M)$')
	plt.legend(loc='best')
	plt.savefig('mdisc_cumfrac.pdf', bbox_inches='tight', format='pdf')

	
	plt.show()
	
	return None

	


def plot_dprops(simulation, time=0.0, wext=False, g0=1.6e-3, rmax=None):
	
	tind=  0
	
		
	if not wext:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc =np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)

	if len(simulation.assoc.shape)>=1:
		assoc = copy.copy(simulation.assoc[pinds])
	else:
		assoc = np.array(np.ones(len(pinds)), dtype=int)

	
	#print('Mean initial m_disc:', np.mean(np.log(mdisc[0])))
	#print('Standard deviation m_disc:', np.std(np.log(mdisc[0])))
	#exit()
	

	print('Subset size:', len(pinds))
	MS= 2

	"""print(g0[:][0])
	print(mdisc[:][0])



	print(g0[:][1])
	print(mdisc[:][1])


	print(g0[:][2])
	print(mdisc[:][2])
	exit()"""

	t = copy.copy(simulation.t)
	m_d = copy.copy(simulation.m)[pinds]
	tunits, munits, runits = copy.copy(simulation.units)

	tarch = t*tunits*s2myr
	tind = np.argmin(np.absolute(tarch-time))
	tval = tarch[tind]
	print('Time:', tarch[tind])
	m_d *= munits*kg2sol

	
	g0nonz = np.where(np.swapaxes(g0vals,0,1)[0]>1e-1)[0]
	tg0nonz = t[g0nonz]
	g0average =  np.trapz(g0vals[g0nonz],tg0nonz, axis=0)/(tg0nonz[-1]-tg0nonz[0])

	pres_hist = g0vals[tind]/g0average

	print('Shapes:', g0average.shape, g0vals[tind].shape, pres_hist.shape)

	mall=  munits*kg2sol*copy.copy(simulation.m)
	im_b =np.where(mall>=1.0)[0]
	m_b = mall[im_b]
	im_m = np.where(mall>=30.0)[0]

	rt_d = runits*m2pc*copy.copy(simulation.r)[tind][pinds]
	rt_b = runits*m2pc*copy.copy(simulation.r)[tind][im_b]
	rt_m = runits*m2pc*copy.copy(simulation.r)[tind][im_m]

	ctmp = cluster_calcs.empirical_centre(rt_d, 10.0, 2, 40, 2)

	
	rt_i = runits*m2pc*copy.copy(simulation.r)[0][pinds]

	
	rt_xf, rt_yf, rt_zf = np.swapaxes(rt_d, 0,1)
	rtf_xm, rtf_ym, rtf_zm = np.swapaxes(rt_m, 0,1)
	rt_x, rt_y, rt_z = np.swapaxes(rt_i, 0,1)

	vt_i = 1e-3*runits*copy.copy(simulation.v[0])[pinds]/tunits
	vtf  = 1e-3*runits*copy.copy(simulation.v[tind])[pinds]/tunits
	vtf_m = 1e-3*runits*copy.copy(simulation.v)[tind][im_m]/tunits
	
	vt_x, vt_y, vt_z = np.swapaxes(vt_i, 0,1)
	vtf_x, vtf_y, vtf_z = np.swapaxes(vtf,0,1)
	vtfm_x, vtfm_y, vtfm_z = np.swapaxes(vtf_m,0,1)
	unique_a = np.unique(assoc)
	
	mpl_cols = ['k', 'r', 'orange', 'lawngreen', 'brown', 'b']
	CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	"""plt.figure(figsize=(4.,4.))

	icol=0
	for iun in unique_a:

		a_inds = np.where(assoc==iun)[0]
		

		plt.scatter(vt_x[a_inds], vt_y[a_inds], marker='+', c=CB_color_cycle[icol])
		icol+=1

	plt.ylabel('$v_x$ (km/s)')
	plt.xlabel('$v_y$ (km/s)')
	plt.savefig('vcheck.pdf', bbox_inches='tight', format='pdf')

	plt.show()"""
	
	"""plt.figure(figsize=(4.,4.))

	icol=0
	for iun in unique_a:

		a_inds = np.where(assoc==iun)[0]
		

		plt.scatter(rt_x[a_inds]-ctmp[0], rt_y[a_inds]-ctmp[1], marker='+', c=CB_color_cycle[icol])
		icol+=1

	plt.ylabel('$x$ (pc)')
	plt.xlabel('$y$ (pc)')
	plt.savefig('rcheck.pdf', bbox_inches='tight', format='pdf')

	plt.show()"""
	"""plt.figure(figsize=(4.,4.))


	colM = plt.cm.get_cmap('RdYlBu')
	sc=plt.scatter(g0vals[tind], rdisc[tind], marker='+',s =MS, c=m_d,  vmin=0.5, vmax=2.0, cmap=colM)
	plt.colorbar(sc, label='$m_\\mathrm{star}$ ($M_\\odot$)')

	plt.xlabel('$F_\\mathrm{FUV}$ ($G_0$)')
	plt.ylabel('$R_\\mathrm{disc}$ (au)')
	plt.xscale('log')
	plt.xlim([5e2,1e4]) 
	plt.ylim([0.0, 150.0])
	plt.savefig('g0rdisc.pdf', bbox_inches='tight', format='pdf')

	"""

	g0vt = g0vals[tind]
	mdt = mdisc[tind]
	iin = np.where((rt_xf<rmax)&(rt_yf<rmax))[0]

	plt.figure(figsize=(4.,4.))

	
	colM = plt.cm.get_cmap('RdYlBu')
	sc=plt.scatter(g0t[iin], mdt[iin], marker='+',s =MS, c=m_d,  vmin=0.5, vmax=2.0, cmap=colM)
	plt.colorbar(sc, label='$m_*$ ($M_\\odot$)')

	plt.xlabel('$F_\\mathrm{FUV}$ ($G_0$)')
	plt.ylabel('$M_\\mathrm{disc}$ ($M_\\odot$)')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim([5e2,1e4]) 
	plt.ylim([1e-6, 1e-1])
	plt.savefig('g0mdisc.pdf', bbox_inches='tight', format='pdf')



	"""plt.figure(figsize=(4.,4.))

	rbins =  np.array([0.01,20.,40., 60., 80.,100.0]) 
	
	icol=0
	for iun in unique_a:

		a_inds = np.where(assoc==iun)[0]
		
		rdtmp = rdisc[tind][a_inds]
		fracs = []
		for ibin in range(len(rbins)-1):
			inds =np.where((rdtmp>rbins[ibin])&(rdtmp<rbins[ibin+1]))[0]
			fracs.append(float(len(inds))/float(len(rdisc[tind][a_inds])))
		
		plt.bar(rbins[:-1], np.array(fracs)*100.0, 20., fill=False, edgecolor = CB_color_cycle[icol], linewidth=BARLINE)

		#plt.scatter(m_d[a_inds], rdisc[tind][a_inds], marker='+',s =MS, c=CB_color_cycle[icol], alpha=0.8)
		plt.plot([],[], c=CB_color_cycle[icol], linewidth=BARLINE, label='Group %d'%(iun+1))
		icol+=1

	
	
	
	
	plt.ylabel('\% of Total Sample')
	plt.xlabel('$R_\\mathrm{disc}$ (au)')
	plt.xlim([0.,100.]) 
	plt.legend(loc=1, fontsize=10)
	plt.savefig('mstarrdisc.pdf', bbox_inches='tight', format='pdf')
	plt.show()"""


	plt.figure(figsize=(4.,4.))

	icol=0
	for iun in unique_a:

		a_inds = np.where(assoc==iun)[0]
		

		plt.scatter(m_d[a_inds], mdisc[tind][a_inds], marker='+',s =MS, c=CB_color_cycle[icol], alpha=0.8)
		plt.scatter([],[], marker='+',s =10, c=CB_color_cycle[icol], label='Group %d'%(iun+1))
		icol+=1

	LMLIM = 1e-5
	isurv = np.where(mdisc[tind]>LMLIM)[0]
	mdisct = mdisc[tind]
	logx = np.log10(m_d[isurv])
	logy = np.log10(mdisct[isurv])

	# define our (line) fitting function
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y: (y - fitfunc(p, x))
	pinit = [-5.,2.2]
	out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)

	pfinal = out[0]
	covar = out[1]

	index = pfinal[1]
	amp = 10.0**pfinal[0]

	indexErr = np.sqrt( covar[1][1] )
	ampErr = np.sqrt( covar[0][0] )
	
	print('Index M_d (m_s)  - all stars: {0} pm {1}'.format(index, indexErr))
	print('with amplitude: {0} pm {1}'.format(amp, ampErr))

	"""
	LMLIM = 1e-4
	NSUB = 10
	isurv = np.where(mdisc[tind]>LMLIM)[0]
	mdisct = mdisc[tind]
	logx = np.log10(m_d[isurv])
	logy = np.log10(mdisct[isurv])
	subinds = np.random.choice(np.arange(len(logx)), size=NSUB, replace=False)
	logx = logx[subinds]
	logy = logy[subinds]


	pinit = [-10.,1.]
	out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)

	pfinal = out[0]
	covar = out[1]

	index = pfinal[1]
	amp = 10.0**pfinal[0]

	indexErr = np.sqrt( covar[1][1] )
	ampErr = np.sqrt( covar[0][0] )
	
	
	print('Index M_d (m_s)  - subset {2} with M_d > {3}: {0} pm {1}'.format(index, indexErr, NSUB, LMLIM))
	print('with amplitude: {0} pm {1}'.format(amp, ampErr))"""

	plt.xlabel('$m_*$ ($M_\\odot$)')
	plt.ylabel('$M_\\mathrm{disc}$ ($M_\\odot$)')
	m_d_sp = np.linspace(0.5,2.0,20)
	m_disc_sp = amp*m_d_sp**index
	pp =(amp*ampErr)*m_d_sp**(index+indexErr) 
	pm =  (amp/ampErr)*m_d_sp**(index+indexErr)
	mp =  (amp*ampErr)*m_d_sp**(index-indexErr)
	mm = (amp/ampErr)*m_d_sp**(index-indexErr) 
	m_disc_ul = np.amax(np.array([pp,mp,pm,mm]),axis=0)
	m_disc_ll = np.amin(np.array([pp,mp,pm,mm]),axis=0)
	plt.plot(m_d_sp, m_disc_sp, c='k', linewidth=BARLINE)
	#plt.plot(m_d_sp, m_disc_ul, c='k', linewidth=BARLINE, linestyle='dashed')
	#plt.plot(m_d_sp, m_disc_ll, c='k', linewidth=BARLINE, linestyle='dashed')
	plt.yscale('log')
	plt.xlim([0.5,2.]) 
	plt.ylim([MLIM, 1e-1])
	plt.legend(loc=4, fontsize=10)
	plt.savefig('mstarmdisc.pdf', bbox_inches='tight', format='pdf')
	plt.show()


	

	"""
	plt.figure(figsize=(4.,4.))

	icol=0
	for iun in unique_a:

		a_inds = np.where(assoc==iun)[0]
		

		plt.scatter(m_d[a_inds], rdisc[tind][a_inds], marker='+',s =MS, c=CB_color_cycle[icol], alpha=0.5)
		plt.scatter([],[], marker='+',s =10, c=CB_color_cycle[icol], label='Group %d'%(iun))
		icol+=1

	plt.scatter(rtf_xm-ctmp[0], rtf_ym-ctmp[1], s=30, marker='*', c='k')

	plt.xlabel('$m_\\mathrm{star}$ ($M_\\odot$)')
	plt.ylabel('$R_\\mathrm{disc}$ (au)')
	plt.xlim([0.5,2.]) 
	plt.ylim([0., 100.0])
	#plt.legend(loc=4, fontsize=10)
	plt.savefig('mstarrdisc.pdf', bbox_inches='tight', format='pdf')



	plt.figure(figsize=(4.,4.))

	icol=0
	for iun in unique_a:

		a_inds = np.where(assoc==iun)[0]
		

		plt.scatter(rt_xf[a_inds]-ctmp[0], rt_yf[a_inds]-ctmp[1], marker='+', c=CB_color_cycle[icol], s=MS, alpha=0.5)
		icol+=1

	plt.ylabel('$x$ (pc)')
	plt.xlabel('$y$ (pc)')
	fsize=15.0
	if fsize!=None:
		plt.xlim([-fsize, fsize])
		plt.ylim([-fsize, fsize])

	plt.savefig('rcheck_final.pdf', bbox_inches='tight', format='pdf')

	
	plt.figure(figsize=(4.,4.))

	icol=0
	for iun in unique_a:

		a_inds = np.where(assoc==iun)[0]
		

		plt.scatter(rdisc[tind][a_inds], mdisc[tind][a_inds], marker='+',s =MS, c=CB_color_cycle[icol], alpha=0.5)
		plt.scatter([],[], marker='+',s =10, c=CB_color_cycle[icol], label='Group %d'%(iun))
		icol+=1

	plt.scatter(rtf_xm-ctmp[0], rtf_ym-ctmp[1], s=30, marker='*', c='k')

	plt.xlabel('$R_\\mathrm{disc}$ (au)')
	plt.ylabel('$M_\\mathrm{disc}$ ($M_\\odot$)')
	plt.yscale('log')
	plt.ylim([1e-6, 1e-1])
	plt.xlim([0., 150.])
	plt.legend(loc=4, fontsize=10)
	plt.savefig('disccheck.pdf', bbox_inches='tight', format='pdf')"""


	"""fig, ax = plt.subplots(figsize=(4.,4.))
	Rend = rdisc[tind]
	Mend = np.log10(mdisc[tind])
	nstep=1
	X = np.swapaxes(np.array([Rend,Mend]),0,1)[::nstep]
	N = [2,3,4,5,6]
	def compute_GMM(N, covariance_type='full', n_iter=1000):
		models = [None for n in N]
		for i in range(len(N)):
			print(N[i])
			models[i] = mixture.GMM(n_components=N[i], n_iter=n_iter,
			covariance_type=covariance_type)
			models[i].fit(X)
		return models

	models = compute_GMM(N)
	AIC = [m.aic(X) for m in models]
	BIC = [m.bic(X) for m in models]

	i_best = np.argmin(BIC)
	gmm_best = models[i_best]
	print("best fit converged:", gmm_best.converged_)
	print("BIC: n_components =  %i" % N[i_best])

	nRbins= 101
	nMbins = 101 
	H, R_bins, M_bins = np.histogram2d(Rend, Mend,
                                  (nRbins, nMbins))

	Xgrid = np.array(map(np.ravel,
             np.meshgrid(0.5 * (R_bins[:-1]
                                + R_bins[1:]),
                         0.5 * ( M_bins[:-1]
                                +  M_bins[1:])))).T
	
	log_dens = gmm_best.score(Xgrid).reshape((nRbins, nMbins))"""

	"""plt.imshow(np.exp(log_dens),
	origin='lower', interpolation='nearest', aspect='auto',
	extent=[R_bins[0], R_bins[-1],
	  M_bins[0], M_bins[-1]],
	cmap=plt.cm.binary)"""

	"""plt.scatter(gmm_best.means_[:, 0], gmm_best.means_[:, 1], c='w')
	icol = 0
	ingroups=[]
	subsets = gmm_best.predict(X)

	print(subsets.shape)

	sub_unique = np.unique(subsets)
	
	for mu, C, w in zip(gmm_best.means_, gmm_best.covars_, gmm_best.weights_):
		print(mu)
		if mu[0]>1.:
			draw_ellipse(mu, C, scales=[1.5], ax=ax, fc='none', ec='k')	
			ingroups.append(icol)
		icol+=1
	

	icol=0
	for ialph in ingroups:
		icol=ialph
		inds = np.where(subsets==ialph)[0]
		plt.scatter(Rend[inds], Mend[inds], marker='+', s=MS, color=CB_color_cycle[icol], label='Group %d'%(icol+1))

	plt.xlabel('$R_\\mathrm{disc}$ (au)')
	plt.ylabel('$M_\\mathrm{disc}$ ($M_\\odot$)')
	plt.ylim([-6, -1])
	plt.xlim([0., 150.])
	plt.legend(loc='best', fontsize=12)
	plt.savefig('gmm.pdf', bbox_inches='tight', format='pdf')


	icol=0
	for ialph in ingroups:

		plt.figure(figsize=(4.,4.))
		a_inds = np.where(subsets==ialph)[0]
		

		plt.scatter(rt_xf[a_inds]-ctmp[0], rt_yf[a_inds]-ctmp[1],s=MS, marker='+', c=CB_color_cycle[ialph])
		plt.ylabel('$x$ (pc)')
		plt.xlabel('$y$ (pc)')
		fsize=15.0
		if fsize!=None:
			plt.xlim([-fsize, fsize])
			plt.ylim([-fsize, fsize])
		plt.scatter(rtf_xm-ctmp[0], rtf_ym-ctmp[1], s=30, marker='*', c='k')
	
		plt.savefig('gmm_r_{0}.pdf'.format(ialph), bbox_inches='tight', format='pdf')
	
		icol+=1



	icol=0
	for ialph in ingroups:

		plt.figure(figsize=(4.,4.))
		a_inds = np.where(subsets==ialph)[0]
		plt.scatter(vtf_x[a_inds]-np.mean(vtf_x), vtf_y[a_inds]-np.mean(vtf_y), s=MS, marker='+', c=CB_color_cycle[ialph])
		plt.scatter(vtfm_x-np.mean(vtf_x), vtfm_y-np.mean(vtf_y), s=30, marker='*', c='k')
		plt.xlim([-20.,20.])
		plt.ylim([-20.0,20.])	
		plt.ylabel('$v_x$ (km/s)')
		plt.xlabel('$v_y$ (km/s)')

		plt.savefig('gmm_v_{0}.pdf'.format(int(ialph)), bbox_inches='tight', format='pdf')
		icol+=1
	"""

	"""fsize=20.
	fig = plt.figure(figsize=(4.,4.))
	ax = p3.Axes3D(fig)
	for ialph in ingroups:
		a_inds = np.where(subsets==ialph)[0]

		xvals = rt_xf[a_inds]-ctmp[0]
		yvals = rt_yf[a_inds]-ctmp[1]
		zvals = rt_zf[a_inds]
		sub = np.where((np.absolute(xvals)<fsize)&(np.absolute(yvals)<fsize)&(np.absolute(zvals)<fsize))[0]
	
		ax.scatter(xvals[sub], yvals[sub], zvals[sub], s=MS, marker='+', c=CB_color_cycle[ialph])
	
	xm = rtf_xm-ctmp[0]
	ym = rtf_ym-ctmp[1]
	zm = rtf_zm

	subm = np.where((np.absolute(xm)<fsize)&(np.absolute(ym)<fsize)&(np.absolute(zm)<fsize))[0]
	
	ax.scatter(xm[subm], ym[subm],zm[subm],  s=30, marker='*', c='k')


	ax.set_title('Disc Substructure')
	#ax.view_init(elev=-8., azim=43.0)


	ax.set_xlim3d([-fsize,fsize])
	ax.set_xlabel('X')

	ax.set_ylim3d([-fsize,fsize])
	ax.set_ylabel('Y')

	ax.set_zlim3d([-fsize,fsize])
	ax.set_zlabel('Z')

	plt.savefig('gmm_r_3d.pdf', bbox_inches='tight', format='pdf')

	plt.show()"""
	
	

	"""plt.figure(figsize=(4.,4.))
	XLIM = [-.1,.1]

	bin_divs = np.linspace(XLIM[0], XLIM[1], 10)
	
	icol = 0
	for ialph in ingroups:
		icol = ialph
		pres_hist_sub = pres_hist[np.where(subsets==ialph)[0]]
		pfrac = np.zeros(len(bin_divs)-1)
		for idiv in range(len(bin_divs)-1):
			in_div =  np.where((np.log10(pres_hist_sub)>bin_divs[idiv])&(np.log10(pres_hist_sub)<bin_divs[idiv+1]))[0]
			pfrac[idiv] = float(len(in_div))/float(len(pres_hist_sub))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			plt.bar(bin_divs[idiv],pfrac[idiv], width=w, edgecolor=CB_color_cycle[icol],color='None')

	
		plt.plot([],[], label='Group %d'%(ialph+1), color=CB_color_cycle[icol])
		icol +=1

	plt.legend(loc='best', fontsize=12)
	plt.xlim(XLIM)
	plt.ylabel('Fraction of Group')
	plt.xlabel('$\\log \\left( F_\\mathrm{FUV, f}/\\langle F_\\mathrm{FUV} \\rangle \\right)$')
	plt.savefig('gmm_hist.pdf', bbox_inches='tight', format='pdf')

	

	plt.figure(figsize=(4.,4.))

	bin_divs = np.linspace(1., 5., 6)
	
	icol = 0
	for ialph in ingroups:
		icol = ialph
		pres_hist_sub = g0average[np.where(subsets==ialph)[0]]
		pfrac = np.zeros(len(bin_divs)-1)
		for idiv in range(len(bin_divs)-1):
			in_div =  np.where((np.log10(pres_hist_sub)>bin_divs[idiv])&(np.log10(pres_hist_sub)<bin_divs[idiv+1]))[0]
			pfrac[idiv] = float(len(in_div))/float(len(pres_hist_sub))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			plt.bar(bin_divs[idiv],pfrac[idiv], width=w, edgecolor=CB_color_cycle[icol],color='None')

	
		plt.plot([],[], label='Group %d'%(ialph+1), color=CB_color_cycle[icol])
		icol +=1

	plt.legend(loc='best', fontsize=12)
	plt.ylabel('Fraction of Group')
	plt.xlabel('$\\log \\left( \\langle F_\\mathrm{FUV} \\rangle \\right)$ ($G_0$)')
	plt.show()"""
	
	Lums = get_FUVluminosities(m_b)

	fluxes  = cluster_calcs.flux(rt_d*1e2/m2pc, Lums,rt_b*1e2/m2pc,2)	
	fluxes /= g0

	xt_d, yt_d, zt_d = np.swapaxes(rt_d, 0,1)
	xt_b, yt_b, zt_b = np.swapaxes(rt_b, 0,1)

	print(np.amax(fluxes))
	g0vals_t = g0vals[tind]

	"""plt.figure(figsize=(4.,4.))


	plt.hist(m_d, bins=30)

	plt.xlabel('$m_\mathrm{star}$ ($G_0$)')
	plt.ylabel('$N$')
	plt.savefig('msthist.pdf', bbox_inches='tight', format='pdf')


	"""
	
	plt.figure(figsize=(4.,4.))


	colM = plt.cm.get_cmap('RdYlBu')
	sc=plt.scatter(fluxes, mdisc[tind], marker='+',s =MS, c=m_d,  vmin=0.5, vmax=2.0, cmap=colM)
	plt.colorbar(sc, label='$m_*$ ($M_\\odot$)')

	plt.xlabel('Projected $F_\\mathrm{FUV}$ ($G_0$)')
	plt.ylabel('$M_\\mathrm{disc}$ ($M_\\odot$)')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim([1e3,2e5]) 
	plt.ylim([1e-6, 1e-1])
	plt.savefig('projg0mdisc.pdf', bbox_inches='tight', format='pdf')

	plt.figure(figsize=(4.,4.))

	lowG0b = [3e3, 6e3]
	highG0b = [8e3,1e10]
	ilow = np.where((fluxes>lowG0b[0])&(fluxes<lowG0b[1]))[0]
	ihigh = np.where((fluxes>highG0b[0])&(fluxes<highG0b[1]))[0]
	
	LLOBS = 1e-3

	ilow_surv = np.where((fluxes>lowG0b[0])&(fluxes<lowG0b[1])&(mdisc[tind]>LLOBS))[0]
	ihigh_surv = np.where((fluxes>highG0b[0])&(fluxes<highG0b[1])&(mdisc[tind]>LLOBS))[0]
	rdisc_lowg0 = rdisc[tind][ilow]
	rdisc_highg0 = rdisc[tind][ihigh]
	rdisc_lowg0_s = rdisc[tind][ilow_surv]
	rdisc_highg0_s = rdisc[tind][ihigh_surv]

	X =  rdisc_lowg0
	bins = np.append(np.sort(X),3e2)
	bins = np.unique(bins)
	n, bins, patches = plt.hist(X+1e-10, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='r')
	
	X =  rdisc_highg0
	bins = np.append(np.sort(X), 3e2)
	bins = np.unique(bins)
	n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='b')
	
	
	X =  rdisc_lowg0_s
	bins = np.append(np.sort(X), 3e2)
	bins = np.unique(bins)
	n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='r', linestyle='dashed')
	
	X =  rdisc_highg0_s
	bins = np.append(np.sort(X), 3e2)
	bins = np.unique(bins)
	n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='b', linestyle='dashed' )

	l1, =plt.plot([],[], c='k', label='All')
	l2, = plt.plot([],[], c='k',linestyle='dashed', label='$M_\mathrm{disc}>10^{-3}\, M_\odot$')
	
	c1, =plt.plot([],[], c='r',  label='$10^3 \, G_0 < $ Projected $F_\mathrm{FUV}<2\cdot 10^3 \, G_0$')
	c2, =plt.plot([],[], c='b',  label='Projected $F_\mathrm{FUV}>5 \cdot 10^3 \, G_0$')

	plt.xlim([0.0,100.0])
	plt.ylim([0.,1.])
	leg1 = plt.legend(handles=[l1,l2], loc=2, fontsize=9)
	plt.legend(handles=[c1,c2],fontsize=9,loc=4)
	plt.gca().add_artist(leg1)
	plt.xlabel('$R$ (au)')
	plt.ylabel('$P(R_\mathrm{disc}<R)$')
	plt.savefig('rdisc_cumfrac.pdf', bbox_inches='tight', format='pdf')

	kstest1 = stats.ks_2samp(rdisc_lowg0_s, rdisc_highg0_s)
	#print('KS:', kstest1)
	#print('Sample sizes:', len(rdisc_lowg0_s), len(rdisc_highg0_s))
	subset = 20
	Ntest=100
	pvals= []
	for itest in range(Ntest):
		kstest2=stats.ks_2samp(np.random.choice(rdisc_lowg0_s,size=subset,replace=False), np.random.choice(rdisc_highg0_s,size=subset,replace=False))
		if kstest2.pvalue<0.05:
			pvals.append(1.)
		else:
			pvals.append(0.0)
	pvals = np.array(pvals)
	print('Chance of significant p-value for sample of {0} R_disc (M_disc > 1e-3 M_sol):'.format(subset), np.mean(pvals))



	kstest1 = stats.ks_2samp(rdisc_lowg0, rdisc_highg0)
	print('KS:', kstest1)
	print('Sample sizes:', len(rdisc_lowg0), len(rdisc_highg0))
	subset = 20
	Ntest=100
	pvals= []
	for itest in range(Ntest):
		kstest2=stats.ks_2samp(np.random.choice(rdisc_lowg0,size=subset,replace=False), np.random.choice(rdisc_highg0,size=subset,replace=False))
		if kstest2.pvalue<0.05:
			pvals.append(1.)
		else:
			pvals.append(0.0)
	pvals = np.array(pvals)
	print('Chance of significant p-value for sample of {0} R_disc:'.format(subset), np.mean(pvals))


	
	plt.figure(figsize=(4.,4.))

	lowG0b = [3e3, 5e3]
	highG0b = [8e3,1e10]
	ilow = np.where((fluxes>lowG0b[0])&(fluxes<lowG0b[1]))[0]
	ihigh = np.where((fluxes>highG0b[0])&(fluxes<highG0b[1]))[0]
	
	LLOBS = 1e-3

	ilow_surv = np.where((fluxes>lowG0b[0])&(fluxes<lowG0b[1])&(mdisc[tind]>LLOBS))[0]
	ihigh_surv = np.where((fluxes>highG0b[0])&(fluxes<highG0b[1])&(mdisc[tind]>LLOBS))[0]
	mdisc_lowg0 = mdisc[tind][ilow]
	mdisc_highg0 = mdisc[tind][ihigh]

	mdisc_copy = copy.copy(mdisc[tind])
	mdisc_copy[np.where(mdisc_copy<1e-3)[0]] = 1e-3

	mdisc_lowg0_s = mdisc[tind][ilow_surv]
	mdisc_highg0_s = mdisc[tind][ihigh_surv]

	X =  mdisc_lowg0
	bins = np.append(np.sort(X),1.)
	n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='r')
	
	X =  mdisc_highg0
	bins = np.append(np.sort(X), 1.)
	n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='b')
	
	
	X =  mdisc_lowg0_s
	bins = np.append(np.sort(X), 1.)
	n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='r', linestyle='dashed')
	
	X =  mdisc_highg0_s
	bins = np.append(np.sort(X), 1.)
	n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='b', linestyle='dashed' )

	l1, =plt.plot([],[], c='k', label='All')
	l2, = plt.plot([],[], c='k',linestyle='dashed', label='$M_\mathrm{disc}>10^{-3}\, M_\odot$')
	
	c1, =plt.plot([],[], c='r',  label='$3000 \, G_0 < $ Projected $F_\mathrm{FUV}<5000 \, G_0$')
	c2, =plt.plot([],[], c='b',  label='Projected $F_\mathrm{FUV}>8000 \, G_0$')

	plt.xscale('log')
	plt.xlim([1e-3,1e-2])
	plt.ylim([0.,1.])
	leg1 = plt.legend(handles=[l1,l2], fontsize=9, bbox_to_anchor=(0.9,0.35) ,bbox_transform=plt.gcf().transFigure)
	plt.legend(handles=[c1,c2], loc=4, fontsize=9)
	plt.gca().add_artist(leg1)
	plt.xlabel('$M$ ($M_\odot$)')
	plt.ylabel('$P(M_\mathrm{disc}<M)$')
	plt.savefig('mdisc_cumfrac.pdf', bbox_inches='tight', format='pdf')

	kstest1 = stats.ks_2samp(mdisc_lowg0_s, mdisc_highg0_s)
	print('KS:', kstest1)
	print('Sample sizes:', len(mdisc_lowg0_s), len(mdisc_highg0_s))
	Ntest=100
	pvals= []
	for itest in range(Ntest):
		kstest2=stats.ks_2samp(np.random.choice(mdisc_lowg0_s,size=subset,replace=False), np.random.choice(mdisc_highg0_s,size=subset,replace=False))
		if kstest2.pvalue<0.05:
			pvals.append(1.)
		else:
			pvals.append(0.0)
	pvals = np.array(pvals)
	print('Pvals:', np.mean(pvals))


	kstest1 = stats.ks_2samp(mdisc_lowg0_s, mdisc_highg0_s)
	print('KS:', kstest1)
	print('Sample sizes:', len(mdisc_lowg0_s), len(mdisc_highg0_s))
	subset = 100
	Ntest=100
	pvals= []
	for itest in range(Ntest):
		kstest2=stats.ks_2samp(np.random.choice(mdisc_lowg0_s,size=subset,replace=False), np.random.choice(mdisc_highg0_s,size=subset,replace=False))
		if kstest2.pvalue<0.05:
			pvals.append(1.)
		else:
			pvals.append(0.0)
	pvals = np.array(pvals)
	print('Chance of significant p-value for sample of {0} M_disc (M_disc > 1e-3 M_sol):'.format(subset), np.mean(pvals))



	mdisc_lowg0 = mdisc_copy[ilow]
	mdisc_highg0 = mdisc_copy[ihigh]

	kstest1 = stats.ks_2samp(mdisc_lowg0, mdisc_highg0)
	print('KS:', kstest1)
	print('Sample sizes:', len(mdisc_lowg0), len(mdisc_highg0))
	subset = 200
	Ntest=100
	pvals= []
	for itest in range(Ntest):
		s1 = np.random.choice(mdisc_lowg0,size=subset,replace=False)
		s2 = np.random.choice(mdisc_highg0,size=subset,replace=False)
		print('s1 number above LLIM:', len(np.where(s1>LLOBS)[0]))
		print('s2 number above LLIM:', len(np.where(s2>LLOBS)[0]))
		kstest2=stats.ks_2samp(s1,s2)
		if kstest2.pvalue<0.05:
			pvals.append(1.)
		else:
			pvals.append(0.0)
	pvals = np.array(pvals)
	print('Chance of significant p-value for sample of {0} M_disc:'.format(subset), np.mean(pvals))

	"""plt.figure(figsize=(4.,4.))


	colM = plt.cm.get_cmap('RdYlBu')
	sc=plt.scatter(fluxes, rdisc[tind], marker='+',s =MS, c=m_d,  vmin=0.5, vmax=2.0, cmap=colM)
	plt.colorbar(sc, label='$m_\\mathrm{star}$ ($M_\\odot$)')

	plt.xlabel('Projected $F_\\mathrm{FUV}$ ($G_0$)')
	plt.ylabel('$R_\\mathrm{disc}$ (au)')
	plt.xscale('log')
	plt.xlim([1e3,3e4]) 
	plt.ylim([0.,100.])
	plt.savefig('projg0rdisc.pdf', bbox_inches='tight', format='pdf')"""

	plt.show()

	"""
	icol=0
	for iun in unique_a:

		a_inds = np.where(assoc==iun)[0]
		

		plt.scatter(fluxes[a_inds], rdisc[tind][a_inds], marker='+', c=CB_color_cycle[icol])
		icol+=1

	plt.xlabel('Projected FUV Flux ($G_0$)')
	plt.ylabel('$R_\\mathrm{disc}$ (au)')
	plt.xlim([1e3, 2e4])
	plt.ylim([0., 100.])
	plt.savefig('disccheck_r.pdf', bbox_inches='tight', format='pdf')

	plt.show()"""

def plot_g0mdisc(simulation, time=0.0, mfilt=1.0, wext=False, g0=1.6e-3, plot=False, force=False, infile=None):
	if infile==None:
		INFILE = simulation.out
		calc=True
	else:
		INFILE = infile
		calc=False
	
	tind=  0
	if (not os.path.isfile(INFILE+'_projg0_discs.npy') or \
	not os.path.isfile(INFILE+'_projg0_OBstars.npy') or force) and calc:
		
		if not wext:
			pinds = copy.copy(simulation.photoevap_inds)
			mdisc = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
			rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
			g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
		else:
			pinds = copy.copy(simulation.photoevap_inds)
			mdisc =np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
			rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
			g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)

		if len(simulation.assoc.shape)>=1:
			assoc = copy.copy(simulation.assoc[pinds])
		else:
			assoc = np.array(np.ones(len(pinds)), dtype=int)



		print('Subset size:', len(pinds))
		MS= 1

		"""print(g0[:][0])
		print(mdisc[:][0])


	
		print(g0[:][1])
		print(mdisc[:][1])


		print(g0[:][2])
		print(mdisc[:][2])
		exit()"""
	
		t = copy.copy(simulation.t)
		m_d = copy.copy(simulation.m)[pinds]
		tunits, munits, runits = copy.copy(simulation.units)

		tarch = t*tunits*s2myr
		tind = np.argmin(np.absolute(tarch-time))
		tval = tarch[tind]
		print('Time:', tarch[tind])
		m_d *= munits*kg2sol

		
		g0nonz = np.where(np.swapaxes(g0vals,0,1)[0]>1e-1)[0]
		tg0nonz = t[g0nonz]
		g0average =  np.trapz(g0vals[g0nonz],tg0nonz, axis=0)/(tg0nonz[-1]-tg0nonz[0])

		pres_hist = g0vals[tind]/g0average
	
		print('Shapes:', g0average.shape, g0vals[tind].shape, pres_hist.shape)

		mall=  munits*kg2sol*copy.copy(simulation.m)
		im_b =np.where(mall>=1.0)[0]
		m_b = mall[im_b]
		im_m = np.where(mall>=30.0)[0]

		rt_d = runits*m2pc*copy.copy(simulation.r)[tind][pinds]
		rt_b = runits*m2pc*copy.copy(simulation.r)[tind][im_b]
		rt_m = runits*m2pc*copy.copy(simulation.r)[tind][im_m]

		ctmp = cluster_calcs.empirical_centre(rt_d, 10.0, 2, 40, 2)

		
		rt_i = runits*m2pc*copy.copy(simulation.r)[0][pinds]

		
		rt_xf, rt_yf, rt_zf = np.swapaxes(rt_d, 0,1)
		rtf_xm, rtf_ym, rtf_zm = np.swapaxes(rt_m, 0,1)
		rt_x, rt_y, rt_z = np.swapaxes(rt_i, 0,1)

		vt_i = 1e-3*runits*copy.copy(simulation.v[0])[pinds]/tunits
		vtf  = 1e-3*runits*copy.copy(simulation.v[tind])[pinds]/tunits
		vtf_m = 1e-3*runits*copy.copy(simulation.v)[tind][im_m]/tunits
		
		vt_x, vt_y, vt_z = np.swapaxes(vt_i, 0,1)
		vtf_x, vtf_y, vtf_z = np.swapaxes(vtf,0,1)
		vtfm_x, vtfm_y, vtfm_z = np.swapaxes(vtf_m,0,1)
		unique_a = np.unique(assoc)
		


		Lums = get_FUVluminosities(m_b)

		fluxes  = cluster_calcs.flux(rt_d*1e2/m2pc, Lums,rt_b*1e2/m2pc,2)	
		fluxes /= g0

		xt_d, yt_d, zt_d = np.swapaxes(rt_d, 0,1)
		xt_b, yt_b, zt_b = np.swapaxes(rt_b, 0,1)

		print(np.amax(fluxes))
		g0vals_t = g0vals[tind]
		
		
		

		np.save(simulation.out+'_projg0_discs', np.array([g0vals, fluxes, mdisc, rdisc, xt_d, yt_d, zt_d , m_d, assoc]))
		np.save(simulation.out+'_projg0_OBstars', np.array([xt_b, yt_b, zt_b, m_b]))
	else:
		g0vals, fluxes, mdisc, rdisc, xt_d, yt_d, zt_d , m_d, assoc = np.load(INFILE+'_projg0_discs.npy')
		#print(dmass, g0proj)
		#print(dmass.shape, g0proj.shape)
		#plt.scatter(g0proj, dmass)NFILE+'_projg0_discs.npy')
		rt_d = np.swapaxes(np.array([xt_d,yt_d, zt_d]),0,1)
		xt_b, yt_b, zt_b, m_b = np.load(INFILE+'_projg0_OBstars.npy')
		tind=-1
	#mbig = np.where(m>mfilt)[0]
	if plot:

		unique_a = np.unique(assoc)
		print(unique_a)
		
		mpl_cols = ['k', 'r', 'orange', 'lawngreen', 'brown', 'b']
		CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
	

		"""plt.figure(figsize=(4.,4.))

		icol=0
		for iun in unique_a:
	
			a_inds = np.where(assoc==iun)[0]
			
	
			plt.scatter(g0vals[tind][a_inds], mdisc[tind][a_inds], marker='+', c=CB_color_cycle[icol])
			icol+=1

		plt.xscale('log')
		plt.yscale('log')
		plt.xlim([5e2, 2e4])
		plt.ylim([1e-7, 2e-2])
		plt.ylabel('Disc Mass ($M_\\odot$)')
		plt.xlabel('FUV Flux ($G_0$)')
		plt.savefig(INFILE+'_g0vmdisc_{0}.pdf'.format('fudge'), bbox_inches='tight', format='pdf')
	
		plt.show()

		
		plt.figure(figsize=(4.,4.))
	
		icol=0
		for iun in unique_a:
			a_inds = np.where(assoc==iun)[0]
	
			plt.scatter(fluxes[a_inds], mdisc[tind][a_inds], marker='+', c=CB_color_cycle[icol])
			icol+=1

		plt.xscale('log')
		plt.yscale('log')
		plt.xlim([1e3, 2e4])
		plt.ylim([1e-7, 2e-2])
		plt.ylabel('Disc Mass ($M_\\odot$)')
		plt.xlabel('Projected FUV Flux ($G_0$)')
		plt.savefig(INFILE+'_projg0vmdisc_{0}.pdf'.format('fudge'), bbox_inches='tight', format='pdf')
	
		plt.show()"""



		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		bin_divs = [3.0, 3.35, 3.5, 3.75, 4.0, 4.35,4.7]
	
		surv_frac = np.zeros(len(bin_divs)-1)

		for idiv in range(len(bin_divs)-1):
			in_div =  np.where((np.log10(g0vals[tind])>bin_divs[idiv])&(np.log10(g0vals[tind])<bin_divs[idiv+1]))[0]
			dm_div = mdisc[tind][in_div]
			nsurv = len(np.where(dm_div>MLIM)[0])
			print('Survival fraction {0}/{1}'.format(nsurv, len(in_div)))
			if len(in_div)>0.:
				surv_frac[idiv] = float(nsurv)/float(len(in_div))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			plt.bar(bin_divs[idiv],surv_frac[idiv], width=w, edgecolor=mpl_cols[idiv],color='None')


		#print(dmass, g0proj)
		#print(dmass.shape, g0proj.shape)
		#plt.scatter(g0proj, dmass)
		#plt.xscale('log')
		#plt.yscale('log')
		plt.xlim([3.0, 4.7])
		plt.ylabel('$N_\mathrm{disc}/N_\mathrm{tot}$')
		plt.xlabel('$\log(\mathrm{FUV flux})$  ($G_0$)')
		plt.savefig(simulation.out+'_g0_1.pdf', bbox_inches='tight', format='pdf')

		plt.show()

	
		fig, (ax1,ax2)= plt.subplots(1,2,figsize=(9.,4.))
	
		surv_frac_2 = np.zeros(len(bin_divs)-1)
		xguar = []
		
		for idiv in range(len(bin_divs)-1):
			if idiv<len(bin_divs)-2:
				in_div =  np.where((np.log10(fluxes)>bin_divs[idiv])&(np.log10(fluxes)<bin_divs[idiv+1]))[0]
			else:
				print('IDIV:', idiv) 
				in_div =  np.where((np.log10(fluxes)>bin_divs[idiv]))[0]
				
			dm_div = mdisc[tind][in_div]
			nsurv = len(np.where(dm_div>MLIM)[0])
			print('Survival fraction {0}/{1}'.format(nsurv, len(in_div)))
			if len(in_div)>0.:
				surv_frac_2[idiv] = float(nsurv)/float(len(in_div))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			ax1.bar(bin_divs[idiv],surv_frac_2[idiv], width=w, edgecolor=mpl_cols[idiv],color='None',linewidth=BARLINE)
			
			xguar.append(bin_divs[idiv]+w/2.)

		yguar = [0.39, 0.35, 0.31, 0.27, 0.22, 0.18]

		#ax1.scatter(xguar, yguar, marker='+',linewidths = BARMARK,s=BARPOINT, color='k')
		ax1.set_ylim([0.,0.7])
		ax1.set_xlim([3.0, 4.7])

		#print(dmass, g0proj)
		#print(dmass.shape, g0proj.shape)
		#plt.scatter(g0proj, dmass)
		#plt.xscale('log')
		#plt.yscale('log')
		ax1.set_ylabel('$N_\mathrm{disc}/N_\mathrm{tot}$')
		ax1.set_xlabel('$\log(\mathrm{projected } \, F_\mathrm{FUV})$  ($G_0$)')
		#plt.savefig(INFILE+'_g0bar.pdf', bbox_inches='tight', format='pdf')
	
		#plt.show()


		fsize=15.0

		ctmp = cluster_calcs.empirical_centre(rt_d, 10.0, 2, 40, 2)

		#plt.figure(figsize=(4.,4.))


		for idiv in range(len(bin_divs)-1):
			in_div =  np.where((np.log10(fluxes)>bin_divs[idiv])&(np.log10(fluxes)<bin_divs[idiv+1]))[0]
			dm_div = mdisc[tind][in_div]
			isurv = np.where(dm_div>MLIM)[0]
			idest = np.where(dm_div<=MLIM)[0]
			nsurv = len(isurv)
			print('Disc fraction: {0}/{1}'.format(nsurv, len(in_div)))
			if len(in_div)>0.:
				surv_frac[idiv] = float(nsurv)/float(len(in_div))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			xtss = xt_d[in_div]
			ytss = yt_d[in_div]
			ax2.scatter(xtss[isurv]-ctmp[0],ytss[isurv]-ctmp[1], c=mpl_cols[idiv], edgecolors=mpl_cols[idiv], s=7)
			ax2.scatter(xtss[idest]-ctmp[0],ytss[idest]-ctmp[1], c='None', edgecolors=mpl_cols[idiv], s=7)

		im = np.where(m_b>10.0)[0]
		ax2.scatter(xt_b[im]-ctmp[0], yt_b[im]-ctmp[1], marker='*', s=80, c='k')
	
		ax2.set_ylabel('$y$ (pc)')
		ax2.set_xlabel('$x$  (pc)')
		if fsize!=None:
			ax2.set_xlim([-fsize, fsize])
			ax2.set_ylim([-fsize, fsize])

		plt.savefig(INFILE+'_g0proposal.pdf', bbox_inches='tight', format='pdf')

		plt.show()

		"""plt.figure(figsize=(4.,4.))
	
		plt.scatter(mdisc[0], mdisc[tind], marker='+', c='k')

		#plt.xscale('log')
		#plt.yscale('log')
		#plt.xlim([3e1, 3e3])
		#plt.ylim([1e-7, 2e-2])
		plt.ylabel('Final Disc Mass ($M_\\odot$)')
		plt.xlabel('Initial Disc Mass ($M_\\odot$)')
	
		plt.show()


	
		plt.figure(figsize=(4.,4.))
	
		plt.scatter(m,mdisc[0], marker='+', c='k')

		#plt.xscale('log')
		#plt.yscale('log')
		#plt.xlim([3e1, 3e3])
		#plt.ylim([1e-7, 2e-2])
		plt.ylabel('Initial Disc Mass ($M_\\odot$)')
		plt.xlabel('Stellar Mass ($M_\\odot$)')
	
		plt.show()"""

	return tval, tind, g0vals, fluxes, mdisc, rdisc, xt_d, yt_d, zt_d , m_d, assoc, xt_b, yt_b, zt_b, m_b



def plot_g0mdisc_tseries(simulation, times=0.0,wext=False, g0=1.6e-3, plot=False, force=False, infile=None):
	
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	

	plt.figure(figsize=(6.,6.))
	lines = ['solid', 'dashed', 'dashdot', 'dotted']
	mpl_cols = ['k']*10#['k', 'r', 'orange', 'lawngreen', 'brown', 'b']
	it =0
	xguar = []
	for t in times:
		tval, tind, g0vals, fluxes, mdisc, rdisc, xt_d, yt_d, zt_d , m_d, assoc, xt_b, yt_b, zt_b, m_b =  plot_g0mdisc(simulation, time=t, wext=wext, plot=False, force=True)


		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		bin_divs = [3.0, 3.35, 3.5, 3.75, 4.0, 4.35,4.7, 5.0, 5.5]
	
		surv_frac = np.zeros(len(bin_divs)-1)


	
		surv_frac_2 = np.zeros(len(bin_divs)-1)
		
		for idiv in range(len(bin_divs)-1):
			if idiv<len(bin_divs)-2:
				in_div =  np.where((np.log10(fluxes)>bin_divs[idiv])&(np.log10(fluxes)<bin_divs[idiv+1]))[0]
			else:
				in_div =  np.where((np.log10(fluxes)>bin_divs[idiv]))[0]
			dm_div = mdisc[tind][in_div]
			nsurv = len(np.where(dm_div>MLIM)[0])
			print('Survival fraction {0}/{1}'.format(nsurv, len(in_div)))
			if len(in_div)>0.:
				surv_frac_2[idiv] = float(nsurv)/float(len(in_div))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			plt.bar(bin_divs[idiv],surv_frac_2[idiv], width=w, edgecolor=mpl_cols[idiv],color='None', linestyle=lines[it],linewidth=BARLINE)
			
			if it==0:
				xguar.append(bin_divs[idiv]+w/2.)

		plt.plot([],[], linestyle=lines[it], label = '$t=%.2lf$ Myr'%(tval), color='k')
		it+=1

	yguar = [0.39, 0.35, 0.31, 0.27, 0.22, 0.18]

	#plt.scatter(xguar, yguar, marker='+',linewidths = BARMARK,s=BARPOINT,  color='k')
	plt.ylim([0.,1.0])
	plt.xlim([3.0, 5.5])

	plt.legend(loc=2)

	#print(dmass, g0proj)
	#print(dmass.shape, g0proj.shape)
	#plt.scatter(g0proj, dmass)
	#plt.xscale('log')
	#plt.yscale('log')
	plt.ylabel('$N_\mathrm{disc}/N_\mathrm{tot}$')
	plt.xlabel('$\log(\mathrm{projected }\,  F_\mathrm{FUV})$  ($G_0$)')
	plt.savefig('g0bartseries.pdf', bbox_inches='tight', format='pdf')
	
	plt.show()



	return None


def plot_disc2d(simulation,rmax=12.5, centre=(.0,.0), mfilt=1.0,time=0.0, ptype='radius', size_fact=4e4, wext=False):


	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	m = copy.copy(simulation.m)

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	
	tind = np.argmin(np.absolute(tarch-time))


	rswitch = np.swapaxes(r,0,1)

	rout_all = np.ones((len(m),len(tarch)))*100.0

	def psize_calc(dmass):
		return size_fact*dmass

	if not wext:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		rdisc = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		mdisc =np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		rdisc =np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)

	print('Not implemented correctly...')
	exit()


	if ptype=='radius':
		print('Assigning radius evolution...')	
		plot_vals = rdisc
	if ptype=='mass':
		print('Assigning mass evolution...')	
		plot_vals = mdisc
	elif ptype=='g0':
		print('Assigning G0...')	
		plot_vals = simulation.FUV
	elif ptype=='g0m':
		
		plot_vals = np.swapaxes(simulation.FUV,0,1)
		psize = np.swapaxes(psize_calc(simulation.phot_m),0,1)

		print(plot_vals.shape, psize.shape)
	else:
		print('Plot type not recognised: "{0}"'.format(ptype))
		exit()

	print('Evolution assigned.')

	rmean = np.mean(plot_vals, axis=0)

	rswap = np.swapaxes(r, 0,2)
	#rswap = np.swapaxes(rswap, 1,2)
	x = rswap[0]
	y = rswap[1]
	z = rswap[2]


	x= np.swapaxes(x,0,1)
	y = np.swapaxes(y,0,1)
	z = np.swapaxes(z,0,1)


	x -= centre[0]
	y -= centre[1]

	rout_all = np.swapaxes(rout_all, 0,1)


	cm = plt.cm.get_cmap('autumn')
	# create the figure
	plt.rc('axes',edgecolor='k')
	
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
	fig, ax = plt.subplots()

	# create the first plot
	biginds = np.where(m>mfilt)[0]
	massinds = np.where(m>30.0)[0]
	xt = x[tind][biginds]
	yt = y[tind][biginds]
	zt = z[tind][biginds]
	zsort = np.argsort(zt)
	xt = xt[zsort]
	yt = yt[zsort]

	if ptype=='radius':
		point=ax.scatter(xt, yt, s=20,  c=plot_vals[tind][biginds][zsort], vmin=20.0, vmax=100.0, cmap=cm)
	if ptype=='mass':
		point=ax.scatter(xt, yt,  s=20,  c=plot_vals[tind][biginds][zsort],norm=LogNorm(vmin=1e-6, vmax=1e-1), cmap=cm)
	elif ptype=='g0':
		point=ax.scatter(xt, yt,  s=20,  c=plot_vals[tind][biginds][zsort], norm=LogNorm(vmin=5e1, vmax=1e4), cmap=cm)
	elif ptype=='g0m':
		point=ax.scatter(xt, yt,  c=plot_vals[tind][biginds][zsort], norm=LogNorm(vmin=5e1, vmax=1e4), cmap=cm, s=psize[tind][biginds])
	pointb =ax.scatter(x[tind][massinds], y[tind][massinds],s=50., c='c', zorder=3)

	ax.legend()
	ax.set_xlim([-rmax, rmax])
	ax.set_ylim([-rmax, rmax])


	#ax.set_axis_bgcolor('black')
	"""ax.tick_params(axis='x',colors='white')
	ax.tick_params(axis='y',colors='white')"""
	ax.get_xaxis().set_tick_params(direction='out', width=1)
	ax.get_yaxis().set_tick_params(direction='out', width=1)

	ax.set_xlabel(r'x (pc)', color='k')
	ax.set_ylabel(r'y (pc)', color='k')
	#ax.set_axis_bgcolor('black')

	cb1 = fig.colorbar(point)
	if ptype=='g0' or ptype=='g0m':
		cb1.set_label('FUV Flux ($G_0$)')
	elif ptype=='radius':
		cb1.set_label('Disc Radius (au)')
	elif ptype=='mass':
		cb2.set_label('Disc mass ($M_\\odot$)')

	if ptype=='g0m':
		psize_calc(simulation.phot_m)
		l1 = plt.scatter([],[], s=psize_calc(1e-4), edgecolors='none')
		l2 = plt.scatter([],[], s=psize_calc(1e-3), edgecolors='none')
		l3 = plt.scatter([],[], s=psize_calc(1e-2), edgecolors='none')

		labels = ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$']

		leg = plt.legend([l1, l2, l3], labels, ncol=3, frameon=True, fontsize=12, loc ='best', title='Disc Masses ($M_\\odot$)', scatterpoints=1)
	

	plt.savefig('discsnap_{0}_{1}.pdf'.format(ptype,tarch[tind]), bbox_inches='tight', format='pdf')
	
	plt.show()

def time_inds(time_arr, dt=0.1):
	inds = [0]
	tprev = 0.0
	for itime in range(len(time_arr)):
		if tprev+dt<=time_arr[itime]:
			inds.append(itime)
			tprev = time_arr[itime]
	
	return np.array(inds, dtype=int)

def prof_evol(simulation, rout=13.0, autocent=True, radcentre=10.0, dim=2):

	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v)
	m = copy.copy(simulation.m)

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= (runits/tunits)*1e-3
	
	nres =1000

	if not os.path.isfile(simulation.out+'_rprof.npy') or not os.path.isfile(simulation.out+'_rparams.npy'):
		rparams = np.zeros((len(tarch), 3))
		rprofile = np.zeros((len(tarch), 2, nres))

		print('Fitting profiles...')

		for it in range(len(tarch)):
			rproj = np.swapaxes(np.swapaxes(r[it],0,1)[:dim], 0,1)
			if autocent:
				ctmp = cluster_calcs.empirical_centre(r[it], radcentre, 2, 20, 3)
			else:
				if dim==2:
					ctmp = np.array([0.,0.])
				else:
					ctmp=np.array([0.,0.,0.])
			rvals = np.linalg.norm(rproj-ctmp,axis=1)
			rparams[it] = fit_elson(rvals, nres, ndim=dim, rmax=rout, init_guess=np.array([2.0, 0.2]))
			print('Params at t={0}:'.format(tarch[it]), rparams[it])
			hst, be = np.histogram(rvals, nres)
			rprofile[it][0] = (be[:-1]+be[1:])/2.
			diff =  (be[1:]- be[:-1])
			rprofile[it][1] = hst/(2.*np.pi*rprofile[it][0]*diff)

		np.save(simulation.out+'_rprof', rprofile)
		np.save(simulation.out+'_rparams', rparams)
	else:
		rprofile = np.load(simulation.out+'_rprof.npy')
		rparams = np.load(simulation.out+'_rparams.npy')


	if not os.path.isfile(simulation.out+'_vdisp.npy'):
		vdisp = np.zeros((len(tarch), 3))
		"""vtst = np.swapaxes(v, 0,1)
		rtst = np.swapaxes(r, 0,1)
		
		for iv in range(max(len(v), 4)):
			vtest(vtst[iv]*1e3, rtst[iv]/m2pc, tarch/s2myr)
		exit()"""
		for it in range(len(tarch)):
			vx, vy, vz  = np.swapaxes(v[it],0,1)
			sigma_x = np.sqrt(np.mean(vx**2.))
			sigma_y = np.sqrt(np.mean(vy**2.))
			sigma_z = np.sqrt(np.mean(vz**2.))
			vdisp[it] = np.array([sigma_x, sigma_y, sigma_z])
		np.save(simulation.out+'_vdisp', vdisp)
	else:
		vdisp = np.load(simulation.out+'_vdisp.npy')

	cumact = np.zeros(rprofile.shape)
	cummod = np.zeros(rprofile.shape) 


	if dim==2:
		ansig = rparams[2]*np.power(1.+np.power(rprofile[0]/rparams[1], 2.), -rparams[0]/2.)
		for ir in range(len(rprofile[0])):
			cumact[1][ir]= np.trapz(2.*np.pi*rprofile[0][:ir]*rprofile[1][:ir], rprofile[0][:ir])
			cummod[1][ir] = np.trapz(2.*np.pi*rprofile[0][:ir]*ansig[:ir], rprofile[0][:ir])


		rspace = np.linspace(0.0, 20.0, 100)
		ansigfix = np.power(1.+np.power(rspace/0.2, 2.), -2.)
		cumfix = np.zeros(len(rspace))
		for ir in range(len(rspace)):
			cumfix[ir] = np.trapz(2.*np.pi*rspace[:ir]*ansigfix[:ir], rspace[:ir])
	else:
		
		ansig = rparams[2]*np.power(1.+np.power(rprofile[0]/rparams[1], 2.), -(rparams[0]+1.)/2.)
		for ir in range(len(rprofile[0])):
			cumact[1][ir]= np.trapz(4.*np.pi*rprofile[0][:ir]*rprofile[0][:ir]*rprofile[1][:ir], rprofile[0][:ir])
			cummod[1][ir] = np.trapz(4.*np.pi*rprofile[0][:ir]*rprofile[0][:ir]*ansig[:ir], rprofile[0][:ir])


		rspace = np.linspace(0.0, 20.0, 100)
		ansigfix = np.power(1.+np.power(rspace/0.2, 2.), -2.)
		cumfix = np.zeros(len(rspace))
		for ir in range(len(rspace)):
			cumfix[ir] = np.trapz(4.*np.pi*rspace[:ir]*rspace[:ir]*ansigfix[:ir], rspace[:ir])

	cumfix *= 8e3/cumfix[-1]
	
	fig, ax = plt.subplots()

	# create the first plot
	
	
	ax.set_ylabel('Cumulative No. Stars', color='k')
	ax.set_xlabel('Radius (pc)', color='k')

	ttext = ax.text(0.05, 0.95, "$t = $ {0} Myrs".format(0.0), transform=ax.transAxes, color='k')

	
	def update_axes(n, cmod, cact, cfix, rspace, rparams, sigv, times,ax):
		ax.cla()
		ax.plot(rspace, cfix, 'k')
		ax.plot(cmod[n][0], cmod[n][1], 'r')
		ax.plot(cact[n][0], cact[n][1], 'g')
		ttext = ax.text(0.05, 0.95, "$t =${:03.2f} Myrs".format(times[n]), transform=ax.transAxes, color='k')
		ttext = ax.text(0.25, 0.95, "$\\gamma = ${:03.2f}".format(rparams[n][0]), transform=ax.transAxes, color='k')
		ttext = ax.text(0.45, 0.95, "$a = ${:03.2f} pc".format(rparams[n][1]), transform=ax.transAxes, color='k')
		ttext = ax.text(0.65, 0.95, "$\\Sigma_0=$%.2lf pc$^{-2}$"%(rparams[n][2]), transform=ax.transAxes, color='k')
		
		ttext = ax.text(0.05, 0.85, "$\\sigma_x = ${:03.2f} km/s".format(vdisp[n][0]), transform=ax.transAxes, color='k')
		ttext = ax.text(0.25, 0.85, "$\\sigma_y = ${:03.2f} km/s".format(vdisp[n][1]), transform=ax.transAxes, color='k')
		ttext = ax.text(0.45, 0.85, "$\\sigma_z = ${:03.2f} km/s".format(vdisp[n][2]), transform=ax.transAxes, color='k')
		
		ax.set_xlim([np.amin(rspace), np.amax(rspace)])
		ax.set_ylim([0.0, 3.2e4])
		ax.set_xlim([0.0,30.0])
		ax.set_ylabel('Cumulative No. Stars', color='k')
		ax.set_xlabel('Radius (pc)', color='k')

		ax.legend()
		return None
	
	nstep =1
	ani=animation.FuncAnimation(fig, update_axes, len(tarch[::nstep]), fargs=( cummod[::nstep], cumact[::nstep],  cumfix, rspace,rparams[::nstep], vdisp[::nstep], tarch[::nstep],ax))

	# make the movie file demo.mp4

	writer=animation.writers['ffmpeg'](fps=1)
	dpi = 500
	ani.save(simulation.out+'_density_profile.mp4',writer=writer,dpi=dpi)
	
	plt.show()


	return rparams

def prof_snap(simulation, rout=13.0, autocent=True, radcentre=10.0, time=0.0, dim=2):

	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v)
	m = copy.copy(simulation.m)

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= (runits/tunits)*1e-3

	
	it = np.argmin(np.absolute(tarch-time))
	
	nres =1000

	rprofile = np.zeros((2, nres))

	print('Fitting profiles...')

	rproj = np.swapaxes(np.swapaxes(r[it],0,1)[:dim], 0,1)
	if autocent:
		ctmp = cluster_calcs.empirical_centre(r[it], radcentre, dim, 20, 3)
	else:
		if dim==2:
			ctmp = np.array([0.,0.])
		else:
			ctmp=np.array([0.,0.,0.])
	rvals = np.linalg.norm(rproj-ctmp,axis=1)
	rparams= fit_elson(rvals, nres, ndim=dim, rmax=rout, init_guess=np.array([4.0, 0.2]))

	
	print('Params at t={0}: a={1}, gamma={2}, n0={3}'.format(tarch[it], rparams[1], rparams[0], rparams[2]))
	hst, be = np.histogram(rvals, nres)
	rprofile[0] = (be[:-1]+be[1:])/2.
	diff =  (be[1:]- be[:-1])
	if dim==2:
		rprofile[1] = hst/(2.*np.pi*rprofile[0]*diff)
	else:
		rprofile[1] = hst/(4.*np.pi*rprofile[0]*rprofile[0]*diff)
		

	cumact = np.zeros(rprofile.shape)
	cummod = np.zeros(rprofile.shape) 


	
	cummod[0] = rprofile[0]
	cumact[0] = rprofile[0]
	if dim==2:
		ansig = rparams[2]*np.power(1.+np.power(rprofile[0]/rparams[1], 2.), -rparams[0]/2.)
		for ir in range(len(rprofile[0])):
			cumact[1][ir]= np.trapz(2.*np.pi*rprofile[0][:ir]*rprofile[1][:ir], rprofile[0][:ir])
			cummod[1][ir] = np.trapz(2.*np.pi*rprofile[0][:ir]*ansig[:ir], rprofile[0][:ir])


		rspace = np.linspace(0.0, 20.0, 100)
		ansigfix = np.power(1.+np.power(rspace/0.2, 2.), -2.)
		cumfix = np.zeros(len(rspace))
		for ir in range(len(rspace)):
			cumfix[ir] = np.trapz(2.*np.pi*rspace[:ir]*ansigfix[:ir], rspace[:ir])
	else:
		
		ansig = rparams[2]*np.power(1.+np.power(rprofile[0]/rparams[1], 2.), -(rparams[0]+1.)/2.)
		for ir in range(len(rprofile[0])):
			cumact[1][ir]= np.trapz(4.*np.pi*rprofile[0][:ir]*rprofile[0][:ir]*rprofile[1][:ir], rprofile[0][:ir])
			cummod[1][ir] = np.trapz(4.*np.pi*rprofile[0][:ir]*rprofile[0][:ir]*ansig[:ir], rprofile[0][:ir])


		rspace = np.linspace(0.0, 20.0, 100)
		ansigfix = np.power(1.+np.power(rspace/0.2, 2.), -2.)
		cumfix = np.zeros(len(rspace))
		for ir in range(len(rspace)):
			cumfix[ir] = np.trapz(4.*np.pi*rspace[:ir]*rspace[:ir]*ansigfix[:ir], rspace[:ir])

	cumfix *= 8e3/cumfix[-1]
	

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig = plt.figure(figsize=(5.,5.))

	# create the first plot
	
	
	plt.ylabel('Cumulative No. Stars', color='k')
	if dim==2:
		plt.xlabel('Projected radius (pc)', color='k')
	else:
		plt.xlabel('Radius (pc)', color='k')


	#plt.plot(rspace, cumfix, 'y', label='Observed')

	plt.plot(cumact[0], cumact[1], 'k', label='$N$-body model')
	
	cdens_n = rparams[2]
	cdens_i = int(np.log10(rparams[2]))
	while cdens_n>10.0:
		cdens_n/=10.0

	plt.plot(cummod[0], cummod[1], 'r', label='$a=%.2lf$ pc, $\gamma=%.2lf$, $n_0 =%.1lf \\times 10^{%d}$ pc$^{-3}$'%(rparams[1], rparams[0], cdens_n, cdens_i))
		
	plt.ylim([0.0, 8000.0])
	plt.xlim([0.0,rout])

	plt.legend(loc='best', prop={'size': 10})
	
	# make the movie file demo.mp4
	plt.savefig(simulation.out+'_profsnap.pdf', bbox_inches='tight', format='pdf')
	
	plt.show()


	return None





def plot_hmr(simulation, dim=2, plot=True, autocent=False, radcentre=5.0, sp_time=0.1):

	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v)
	m = copy.copy(simulation.m)


	if hasattr(simulation, 'starinds'):
		stit_flag=True
		stinds = simulation.starinds
		tends = simulation.tends
		tends = np.cumsum(tends)
		inext=0
		stit = []
		for it in range(len(t)):
			stit.append(stinds[inext])
			if t[it]>tends[inext] and inext<len(stinds)-1:
				inext+=1
	else:
		stit = [np.arange(len(m)) for it in range(len(t))]


	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= (runits/tunits)*1e-3
	
	
	if sp_time!=None:
		list_indices = time_inds(tarch, dt=sp_time)
		nsnaps = len(list_indices)
	else:
		nsnaps = len(tarch)
		list_indices = np.arange(nsnaps)

	
	rproj = np.swapaxes(np.swapaxes(r, 0,2)[:dim],0,2)
	xp, yp, zp = np.swapaxes(np.swapaxes(r,0,1), 0,2)[:]

	nmembers = np.zeros(len(list_indices))
	

	if not os.path.isfile(simulation.out+'_HM.npy'):
		t_return = np.zeros(nsnaps)
		r_hm = np.zeros(nsnaps)
		r_75 = np.zeros(nsnaps)
		r_25 = np.zeros(nsnaps)
		ival = 0
		for it in list_indices:
			t_return[ival] = tarch[it]
			nmembers[ival] = len(stit[it])
			if autocent:
				ctmp = cluster_calcs.empirical_centre(r[it][stit[it]], radcentre, dim, 20, 3)
			else:
				if dim==2:				
					ctmp = np.array([0.,0.])
				else:				
					ctmp = np.array([0.,0.,0.])
			
			rmags = np.linalg.norm(rproj[it][stit[it]]-ctmp,axis=1)
			irsrt = np.argsort(rmags)
			rsort = rmags[irsrt]
			msort = m[irsrt]
			cumdist = np.cumsum(msort)
			ncumdist = cumdist/cumdist[-1]
			frac_func = interpolate.interp1d(ncumdist, rsort)
			r_hm[ival] = frac_func(0.5)
			r_75[ival] = frac_func(0.75)
			r_25[ival] = frac_func(0.25)
			print('At {0} Myr HM: {1}'.format(tarch[it], r_hm[ival]))
			ival +=1

		np.save(simulation.out+'_HM', np.array([t_return, r_hm, r_75, r_25]))
	else:
		t_return, r_hm,r_75, r_25= np.load(simulation.out+'_HM.npy')
	
	if plot:
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.figure(figsize=(4.,4.))
	
		plt.plot(t_return, r_75, 'k', linestyle='dashed', label='$R_{75}$')
		plt.plot(t_return, r_hm, 'k', label='$R_{50}$')
		plt.plot(t_return, r_25, 'k', linestyle='dotted', label='$R_{25}$')
		plt.xlabel('Time (Myr)')
		plt.ylabel('Radius (pc)')
		plt.legend(loc='best')
		plt.savefig(simulation.out+'_HM.pdf', bbox_inches='tight', format='pdf')
		plt.show()



	return t_return, r_hm



def rotate_vector(x0, y0, z0, th, ph):
	
	def Rx_theta(x,y,z, theta):
		xnew = x
		ynew = np.cos(theta)*y - np.sin(theta)*z
		znew = np.sin(theta)*y +np.cos(theta)*z
		return xnew, ynew, znew

	
	def Ry_phi(x,y,z, phi):
		xnew = x*np.cos(phi)+z*np.sin(phi)
		ynew = y 
		znew = -x*np.sin(phi)+z*np.cos(phi)
		return xnew, ynew, znew

	
	x1, y1, z1 = Rx_theta(x0,y0,z0, th)
	x1, y1, z1 = Ry_phi(x1,y1,z1, ph)

	return x1, y1, z1
	


def get_KE(x, y, vx, vy, m):
	
	
	def radial_velocity(xtmp,ytmp,vxtmp,vytmp):
		vr = (vxtmp*xtmp+ytmp*vytmp)/np.sqrt(xtmp**2.+ytmp**2.)
		return vr

	def azim_velocity(xtmp,ytmp,vxtmp,vytmp):
		vt = (vytmp*xtmp-ytmp*vxtmp)/np.sqrt(xtmp**2.+ytmp**2.)
		return vt

	
	rvst =  radial_velocity(x, y, vx, vy)
	tvst =  azim_velocity(x, y, vx, vy)

	tneg = np.where(tvst<0.)[0]
	tpos = np.where(tvst>0.)[0]
	tvn = tvst[tneg]
	tvp = tvst[tpos]
	tmp = m[tpos]
	tmn = m[tneg]

	rneg = np.where(rvst<0.)[0]
	rpos = np.where(rvst>0.)[0]
	rvn = rvst[rneg]
	rmn = m[rneg]
	rmp = m[rpos]
	rvp = rvst[rpos]
	print('Print sum 1 (1):', np.sum(rvn), np.sum(rvp))
	print('Print sum 2 (1):', np.sum(tvn), np.sum(tvp))

	KEr = np.sum(rmp*rvp**2.)/(np.sum(rmn*rvn**2.)+np.sum(rmp*rvp**2.))
	KEt= np.sum(tmp*tvp**2.)/(np.sum(tmn*tvn**2.)+np.sum(tmp*tvp**2.))

	print('Print sum 1 (2):', np.sum(rmn*rvn**2.)+np.sum(rmp*rvp**2.))
	print('Print sum 2 (2):', np.sum(tmn*tvn**2.)+np.sum(tmp*tvp**2.))
	KErt = (np.sum(rmn*rvn**2.)+np.sum(rmp*rvp**2.))/(np.sum(rmn*rvn**2.)+np.sum(rmp*rvp**2.)+np.sum(tmn*tvn**2.)+np.sum(tmp*tvp**2.))

	return KEr, KEt, KErt


def get_CMASS_fromsim(simulation,  centre = (.0, .0),  autocent=False, radcentre=5.0, dims=2,time=0.0):
	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	m = copy.copy(simulation.m)

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol

	it = np.argmin(np.absolute(tarch-time))
	

	rswap = np.swapaxes(r, 0,2)
	#rswap = np.swapaxes(rswap, 1,2)
	x = rswap[0]
	y = rswap[1]
	z = rswap[2]

	x= np.swapaxes(x,0,1)
	y = np.swapaxes(y,0,1)
	z = np.swapaxes(z,0,1)

	if autocent:
		ctmp = cluster_calcs.empirical_centre(r[it], radcentre, dims, 20, 3)
	else:
		ctmp = np.array([centre[0],centre[1]])

	xt = x[it]-ctmp[0]
	yt = y[it]-ctmp[1]
	if dims==3:
		zt = x[it]-ctmp[0]
		subvals = np.where((xt**2.+yt**2.+zt**2.)<radcentre**2.)[0]
	else:
		subvals = np.where((xt**2.+yt**2.)<radcentre**2.)[0]
	
	cmass = np.sum(m[subvals])
			
	return cmass

def get_KE_fromsim(simulation,  fsize=None, centre = (.0, .0), mfilt=1.0,  autocent=False, radcentre=5.0, time=0.0,hm=False, radlim=None, fix_n=None):
	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= 1e-3*runits/tunits

	it = np.argmin(np.absolute(tarch-time))
	vswap = np.swapaxes(v, 0,2)
	rswap = np.swapaxes(r, 0,2)
	vx = vswap[0]
	vy = vswap[1]
	vz = vswap[2]
	x = rswap[0]
	y = rswap[1]
	
	if mfilt!=None:
		biginds = np.where(m>mfilt)[0]
		x = x[biginds]
		y = y[biginds]
		vx = vx[biginds]
		vy = vy[biginds]
	
	vx= np.swapaxes(vx,0,1)
	vy = np.swapaxes(vy,0,1)
	
	x= np.swapaxes(x,0,1)
	y = np.swapaxes(y,0,1)

	if autocent:
		ctmp = cluster_calcs.empirical_centre(r[it], radcentre, 2, 20, 3)
	else:
		ctmp = np.array([centre[0],centre[1]])

	xt = x[it]-ctmp[0]
	yt = y[it]-ctmp[1]
	if hm:
		rmags = np.sqrt(xt**2+yt**2)
		irsrt = np.argsort(rmags)
		rsort = rmags[irsrt]
		msort = m[irsrt]
		cumdist = np.cumsum(msort)
		ncumdist = cumdist/cumdist[-1]
		frac_func = interpolate.interp1d(ncumdist, rsort)
		r_hm = frac_func(0.5)
		subvals = np.where(rmags<r_hm)[0]
		xt = xt[subvals]
		yt = yt[subvals]
		vxt = vx[it][subvals]
		vyt = vy[it][subvals]
	elif radlim!=None:
		rmags = np.sqrt(xt**2+yt**2)
		subvals = np.where(rmags<radlim)[0]
		xt = xt[subvals]
		yt = yt[subvals]
		vxt = vx[it][subvals]
		vyt = vy[it][subvals]
	elif fsize!=None:
		subvals = np.where(np.absolute(xt)<fsize)[0]
		xt = xt[subvals]
		yt = yt[subvals]
		vxt = vx[it][subvals]
		vyt = vy[it][subvals]
		subvals = np.where(np.absolute(yt)<fsize)[0]
		xt = xt[subvals]
		yt = yt[subvals]
		vxt = vxt[subvals]
		vyt = vyt[subvals]
	else:
		vxt = vx[it]
		vyt = vy[it]

	if fix_n!=None:
		if len(xt)>fix_n:
			subvals = np.random.choice(np.arange(len(xt)),size=fix_n, replace=False)
			xt = xt[subvals]
			yt = yt[subvals]
			vxt = vxt[subvals]
			vyt = vyt[subvals]
			print('Restricting consideration to {0}/{1} stars.'.format(fix_n, len(xt)))
		else:
			if hm:
				print('Warning: too few stars in hm radius - {0}/{1}.'.format(len(xt), fix_n))
			elif radlim!=None:
				print('Warning: too few stars in radius {2} pc - {0}/{1}.'.format(len(xt), fix_n, radlim))
			elif fsize!=None:
				print('Warning: too few stars in field size {2}x{2} pc - {0}/{1}.'.format(len(xt), fix_n, fsize*2.))
			else:
				print('Warning: too few stars in simulation.')
	
	KEr, KEt, KErt = get_KE(xt, yt, vxt, vyt)

	return KEr, KEt, KErt

def get_iesc(popinds):

	iesc_all = np.array([])
	for stinds in popinds:
		iesc  = prev_inds[~np.in1d(prev_inds, stinds)]
		if len(iesc)>0:
			iesc_all = np.append(iesc_all, iesc)
		prev_inds = stinds
	
	return iesc_all

	

	

def plot_vdisp(simulation,  sp_time=None):
	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)


	if hasattr(simulation, 'starinds'):
		stit_flag=True
		stinds = simulation.starinds
		tends = simulation.tends
		tends = np.cumsum(tends)
		inext=0
		stit = []
		for it in range(len(t)):
			stit.append(stinds[inext])
			if t[it]>tends[inext] and inext<len(stinds)-1:
				inext+=1
	else:
		stit = [np.arange(len(m)) for it in range(len(t))]

	
		 

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= 1e-3*runits/tunits

	if type(sp_time)!=type(None):
		list_indices = time_inds(tarch, dt=sp_time)
		nsnaps = len(list_indices)
	else:
		nsnaps = len(tarch)
		list_indices = np.arange(nsnaps)

	times = tarch[list_indices]
	vels = v[list_indices]
	stit_cut = []
	for ist in range(len(stit)):
		if ist in list_indices:
			stit_cut.append(stit[ist])

	it=0
	vdisps = np.zeros((len(list_indices),3))
	for vel in vels:
		vdisps[it] = np.std(vel[stit_cut[it]], axis=0)
		it+=1

	vx, vy, vz = np.swapaxes(vdisps,0,1)[:]
	plt.rc('font', family='serif')
	plt.rc('text', usetex=True)
	plt.figure(figsize=(4.,4.))
	plt.plot(times, vx, c='k', linestyle='solid')
	plt.plot(times, vy, c='k', linestyle='dashed')
	plt.plot(times, vz, c='k', linestyle='dotted')
	plt.xlabel('Time (Myr)')
	plt.ylabel('$\sigma_v$ (km/s)')
	plt.show()

def plot_n0(simulation, centrad=0.1, sp_time=None):
	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)


	if hasattr(simulation, 'starinds'):
		stit_flag=True
		stinds = simulation.starinds
		tends = simulation.tends
		tends = np.cumsum(tends)
		inext=0
		stit = []
		for it in range(len(t)):
			stit.append(stinds[inext])
			if t[it]>tends[inext] and inext<len(stinds)-1:
				inext+=1
	else:
		stit = [np.arange(len(m)) for it in range(len(t))]

	
		 

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= 1e-3*runits/tunits

	if type(sp_time)!=type(None):
		list_indices = time_inds(tarch, dt=sp_time)
		nsnaps = len(list_indices)
	else:
		nsnaps = len(tarch)
		list_indices = np.arange(nsnaps)

	times = tarch[list_indices]
	rvals_all = r[list_indices]
	stit_cut = []
	for ist in range(len(stit)):
		if ist in list_indices:
			stit_cut.append(stit[ist])

	it=0
	ncore = np.zeros(len(list_indices))
	Vol = (4./3.)*np.pi*centrad**3.
	for rvals in rvals_all:
		rmags = np.linalg.norm(rvals[stit_cut[it]], axis=1)
		ncore[it] = float(len(np.where(rmags<centrad)[0]))/Vol
		it+=1


	plt.rc('font', family='serif')
	plt.rc('text', usetex=True)
	plt.figure(figsize=(4.,4.))
	plt.plot(times, ncore, c='k', linestyle='solid')
	plt.xlabel('Time (Myr)')
	plt.ylabel('$n_0$ (stars/pc$^{3}$)')
	plt.yscale('log')
	plt.show()

	np.save(simulation.out+'_n0', np.array([times, ncore]))
	return times, ncore
	

def get_tfirst(tarray, popinds, nstars):
	tfirst_arr = np.zeros(nstars)
	for istar in range(nstars):
		tcheck = True
		ipop=0
		while tcheck and ipop<len(popinds):
			if np.in1d(istar, popinds[ipop])[0]:
				tfirst_arr[istar] = tarray[ipop]
				tcheck=False
			ipop+=1
		if tcheck:
			print('Star not found:', istar)
			exit()

	return tfirst_arr
			

def plot_stages(simulation,  time=0.0, rmax=20.0):
	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)
	ipops = copy.copy(simulation.starinds)
	tends = np.cumsum(np.array(copy.copy(simulation.tends)))
	tends  = np.append(np.array([0.0]), tends)
	
	tunits, munits, runits = simulation.units

	tends *= tunits*s2myr
	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= 1e-3*runits/tunits

	itime = np.argmin(np.absolute(tarch-time))
	
	rvals = r[itime]
	
	print(tends)
	tfirst = get_tfirst(tends, ipops, len(m))
	print(tfirst)

	tages = time-tfirst

	rad = np.linalg.norm(rvals, axis=1)
	x,y,z = np.swapaxes(rvals,0,1)[:]

	popages  = np.unique(tages)
	stpops = []
	for age in popages:
		
		stpops.append(np.where((tages==age)&(rad<rmax))[0])
		

	mu0, std0 = norm.fit(x[stpops[0]])
	mu1, std1 = norm.fit(x[stpops[1]])
	mu2, std2 = norm.fit(x[stpops[2]])

	xspace = np.linspace(-20, 20, 100)
	plt.hist(x[stpops[0]], facecolor='none', edgecolor='k', bins=20)
	p0 =  len(stpops[0])*norm.pdf(xspace, mu0, std0)
	plt.plot(xspace, p0, c='k')
	plt.show()
	
	plt.hist(x[stpops[1]], facecolor='none', edgecolor='g', bins=20)
	p1 =  len(stpops[1])*norm.pdf(xspace, mu1, std1)
	plt.plot(xspace, p1, c='g')
	plt.show()
	plt.hist(x[stpops[2]], facecolor='none', edgecolor='r', bins=20)
	p2 = len(stpops[2])*norm.pdf(xspace, mu2, std2)
	plt.plot(xspace, p2, c='r')

	#plt.scatter(x[stpops[0]], y[stpops[0]], c='k')
	#plt.scatter(x[stpops[1]], y[stpops[1]], c='g')
	#plt.scatter(x[stpops[2]], y[stpops[2]], c='r')
	plt.show()


def plot_popevol(simulation,  sp_time=None, rmax=5000.0):
	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)
	ipops = copy.copy(simulation.starinds)
	tends = np.cumsum(np.array(copy.copy(simulation.tends)))
	tends  = np.append(np.array([0.0]), tends)
	
	tunits, munits, runits = simulation.units

	tends *= tunits*s2myr
	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= 1e-3*runits/tunits

	if sp_time!=None:
		list_indices = time_inds(tarch, dt=sp_time)
		nsnaps = len(list_indices)
	else:
		nsnaps = len(tarch)
		list_indices = np.arange(nsnaps)
		

	
	tfirst = get_tfirst(tends, ipops, len(m))


	popages  = np.unique(tfirst)
	sigpops = np.zeros((len(list_indices), len(popages)+1))
	itruncs = np.zeros(len(popages)+1, dtype=int)

	rmags = np.linalg.norm(r, axis=2)
	rmags_max = np.zeros(len(m))
	ipop=0
	for tfirst_v in popages:
		ipops_v  = np.where(tfirst==tfirst_v)[0]
		maxvals = np.amax(rmags[np.where(tarch>=tfirst_v)[0]], axis=0)
		rmags_max[ipops_v] = maxvals
		itruncs[ipop] = int(np.amin(np.where(tarch[list_indices]-tfirst_v>0.)[0]))
		ipop+=1
	
	print(rmags_max.shape, len(m))
	
	it=0
	for itime in list_indices:
		rvals = r[itime]
		rad = np.linalg.norm(rvals, axis=1)
		x,y,z = np.swapaxes(rvals,0,1)[:]
		stpops = []
		ipop=0
		
		for tfirst_v in popages:
			stpops.append(np.where((tfirst==tfirst_v)&(rad<rmax))[0])
			if len(stpops[ipop])>0:
				rhm = np.percentile(rad[stpops[ipop]],50.0)
				std0 = np.std(x[stpops[ipop]])
				std1 = np.std(y[stpops[ipop]])
				std2 = np.std(z[stpops[ipop]])
				sigpops[it][ipop] = rhm
			ipop+=1
		rhm = np.percentile(rad[np.where(rad<rmax)[0]],50.0)
		sigpops[it][-1] = rhm
		it+=1

	tvals = tarch[list_indices]

	sigpops = np.swapaxes(sigpops,0,1)

	plt.rc('font', family='serif')
	plt.rc('text', usetex=True)
	plt.figure(figsize=(4.,4.))
	

	plt.plot(tvals[itruncs[0]:], sigpops[0][itruncs[0]:], c=CB_color_cycle[0], label='Pop. 1')
	plt.plot(tvals[itruncs[1]:], sigpops[1][itruncs[1]:], c=CB_color_cycle[1], label='Pop. 2')
	plt.plot(tvals[itruncs[2]:], sigpops[2][itruncs[2]:], c=CB_color_cycle[2], label='Pop. 3')
	plt.plot(tvals, sigpops[3], c='k', label='All')
	
	plt.xlabel('Time (Myr)')
	plt.ylabel('$R_{50}$ (pc)')
	plt.legend(loc='best', fontsize=10)
	plt.savefig(simulation.out+'_rhmpops.pdf', bbox_inches='tight', format='pdf')
	plt.show()
	
	np.save(simulation.out+'_sigpop', np.array([tvals,sigpops]))
	np.save(simulation.out+'_sigpop_itruncs', itruncs)

	return tvals, sigpops
		
	


def get_KEMI_distr(simulation,  fsize=10.0, centre = (.0, .0), mfilt=None,  autocent=False, radcentre=10.0, time=0.0,hm=False, radlim=None, fix_n=100, nhist=100, theta=0.0, phi=0.0, minKE=False, minVZ=False, verror=0.):
	
	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= 1e-3*runits/tunits

	
	if type(time)==list:
		KE_MI_all = []
		vdisp_all = []
		times_all = []

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.figure(figsize=(4.,4.))
		itime=0
		lines = ['dashed', 'solid', 'dashdot', 'dotted']
		
		for t in time:
			it = np.argmin(np.absolute(tarch-t))
			times_all.append(tarch[it])

			vswap = np.swapaxes(v, 0,2)
			rswap = np.swapaxes(r, 0,2)
			vx = vswap[0]
			vy = vswap[1]
			vz = vswap[2]
			x = rswap[0]
			y = rswap[1]
			z = rswap[2]
			if mfilt!=None:
				biginds = np.where(m>mfilt)[0]
				x = x[biginds]
				y = y[biginds]
				z = z[biginds]
				vx = vx[biginds]
				vy = vy[biginds]
				vz = vz[biginds]
				m = m[biginds]
	
			vx= np.swapaxes(vx,0,1)
			vy = np.swapaxes(vy,0,1)
			vz = np.swapaxes(vz,0,1)
	
			x= np.swapaxes(x,0,1)
			y = np.swapaxes(y,0,1)
			z = np.swapaxes(z,0,1)

			print(x.shape)

			if minKE:
				if not os.path.isfile(simulation.out+'_KEmin.npy'):
					print('Minimising KE. This may take several minutes...')
					ke_min, theta, phi = minimise_KE(x[it], y[it], z[it], vx[it], vy[it], vz[it], m, KEtype=0, res=10, autocent=autocent, fsize=fsize, radlim=radlim, radcentre=radcentre, hm=hm, centre=centre)
					print('Minimised KE bias for theta = {0}, phi = {1}: {2}'.format(theta, phi, ke_min))
					np.save(simulation.out+'_KEmin', np.array([theta, phi, ke_min]))
				else:
					theta, phi, ke_min = np.load(simulation.out+'_KEmin.npy')
					print('Minimised KE bias for theta = {0}, phi = {1}: {2}'.format(theta, phi, ke_min))
			elif minVZ:
				if not os.path.isfile(simulation.out+'_minVZ.npy'):
					print('Minimising v_z. This may take several minutes...')
					vz_min, theta, phi = minimise_VZ(x[it], y[it], z[it], vx[it], vy[it], vz[it], m, res=10, autocent=False, fsize=None, radlim=radlim, radcentre=radcentre, hm=hm, centre=centre)
					print('Minimised v_z for theta = {0}, phi = {1}: {2}'.format(theta, phi, vz_min))
					np.save(simulation.out+'_minVZ', np.array([theta, phi, vz_min]))
				else:
					theta, phi, vz_min = np.load(simulation.out+'_minVZ.npy')
					print('Minimised v_z bias for theta = {0}, phi = {1}: {2}'.format(theta, phi, vz_min))

			print('Vdisp before:', np.std(vx[it]), np.std(vy[it]), np.std(vz[it]))

			xt,  yt, zt, vxt, vyt, vzt = rotate_rx(x[it], y[it], z[it], vx[it],vy[it],vz[it],theta, phi)
	
			print('Vdisp after:', np.std(vxt), np.std(vyt), np.std(vzt))

			print(xt.shape)
	

			if autocent:
				rt = np.swapaxes(np.array([xt,yt,zt]),0,1)
				ctmp = cluster_calcs.empirical_centre(rt, radcentre, 2, 50, 3)
			else:
				ctmp = np.array([centre[0],centre[1]])

			xt = xt-ctmp[0]
			yt = yt-ctmp[1]
			if hm:
				rmags = np.sqrt(xt**2+yt**2)
				irsrt = np.argsort(rmags)
				rsort = rmags[irsrt]
				msort = m[irsrt]
				cumdist = np.cumsum(msort)
				ncumdist = cumdist/cumdist[-1]
				frac_func = interpolate.interp1d(ncumdist, rsort)
				r_hm = frac_func(0.5)
				subvals = np.where(rmags<r_hm)[0]
				xt = xt[subvals]
				yt = yt[subvals]
				zt = zt[subvals]
				vxt = vxt[subvals]
				vyt = vyt[subvals]
				vzt = vzt[subvals]
				mt = m[subvals]
			elif radlim!=None:
				rmags = np.sqrt(xt**2+yt**2)
				subvals = np.where(rmags<radlim)[0]
				xt = xt[subvals]
				yt = yt[subvals]
				zt = zt[subvals]
				vxt = vxt[subvals]
				vyt = vyt[subvals]
				vzt = vzt[subvals]
				mt = m[subvals]
			elif fsize!=None:
				print('Field size: {0}'.format(fsize))
				subvals = np.where((np.absolute(xt)<fsize)&(np.absolute(yt)<fsize))[0]
				xt = xt[subvals]
				yt = yt[subvals]
				zt = zt[subvals]
				vxt = vxt[subvals]
				vyt = vyt[subvals]
				vzt = vzt[subvals]
				mt = m[subvals]
	
			if len(xt)>fix_n:
				print('Restricting consideration to {0}/{1} stars.'.format(fix_n, len(xt)))
			else:
				print('Error: too few stars in field ({0}/{1}).'.format(len(xt), fix_n))
				exit()

			KE_MI = np.zeros((nhist,3))
			vdisp = np.zeros((nhist,3))
			for ihist in range(nhist):
				subvals = np.random.choice(np.arange(len(xt)),size=fix_n, replace=False)
				xtmp = xt[subvals]
				ytmp = yt[subvals]
				vxtmp = vxt[subvals]
				vytmp = vyt[subvals]
				vztmp = vzt[subvals]
				mtmp = mt[subvals]

				KEr, KEt, KErt = get_KE(xtmp, ytmp, vxtmp, vytmp, mtmp)
				MIx, MIy=  cluster_calcs.Morans_I_vcorr(r[it][subvals], v[it][subvals], 0.*au2pc, verror)
				vdispx = np.sqrt(np.mean(vxtmp*vxtmp))
				vdispy = np.sqrt(np.mean(vytmp*vytmp))
				vdispz = np.sqrt(np.mean(vztmp*vztmp))
				KE_MI[ihist] = np.array([KEr,MIx,MIy])
				vdisp[ihist] = np.array([vdispx,vdispy,vdispz])
				if (ihist+1)%100==0:
					KEMI_tmp = np.swapaxes(KE_MI[:ihist+1],0,1)
					KEmean = np.mean(KEMI_tmp[0])
					MIxmean = np.mean(KEMI_tmp[1])
					MIymean = np.mean(KEMI_tmp[2])
					KErsigma = 0.5*(np.percentile(KEMI_tmp[0], 84.2)-np.percentile(KEMI_tmp[0], 15.8))
					MIxsigma = 0.5*(np.percentile(KEMI_tmp[1], 84.2)-np.percentile(KEMI_tmp[1], 15.8))
					MIysigma = 0.5*(np.percentile(KEMI_tmp[2], 84.2)-np.percentile(KEMI_tmp[2], 15.8))
					print('At iteration {3}: <E>={0}, <Ix>={1}, <Iy>={2}'.format(KEmean, MIxmean, MIymean, ihist+1))
					print('sigma_E={0}, sigma_Ix={1}, sigma_Iy={2}'.format(KErsigma, MIxsigma, MIysigma))

				
					vdisp_tmp = np.swapaxes(vdisp[:ihist+1],0,1)
					vxmean = np.mean(vdisp_tmp[0])
					vymean = np.mean(vdisp_tmp[1])
					vzmean = np.mean(vdisp_tmp[2])
					vxsigma = 0.5*(np.percentile(vdisp_tmp[0], 84.2)-np.percentile(vdisp_tmp[0], 15.8))
					vysigma = 0.5*(np.percentile(vdisp_tmp[1], 84.2)-np.percentile(vdisp_tmp[1], 15.8))
					vzsigma = 0.5*(np.percentile(vdisp_tmp[2], 84.2)-np.percentile(vdisp_tmp[2], 15.8))
					print('(sqrt) <<vx^2>>={0}, <<vy^2>>={1}, <<vz^2>>={2}'.format(vxmean, vymean, vzmean))
					print('(sqrt) sigma_<vx^2>={0}, sigma_<vy^2>={1}, sigma_<vz^2>={2}\n_________\n'.format(vxsigma, vysigma, vzsigma))

			KE_MI_all.append(KE_MI)
			vdisp_all.append(vdisp)

			X =  np.swapaxes(KE_MI, 0,1)[0]
			bins = np.append(np.sort(X),1e1)
			bins = np.unique(bins)
			n, bins, patches = plt.hist(X, normed=True, histtype='step', cumulative=True, bins=bins, edgecolor='k', linestyle=lines[itime])
			plt.plot([],[], linestyle=lines[itime], color='k', label='$t = %.1lf$ Myr'%(times_all[-1]))
			itime+=1

		plt.xlim([0.3,0.7])
		plt.ylim([0.,1.])
		
		plt.ylabel('Cum. Frac. Ensembles')
		plt.xlabel('$\\mathcal{E}$')
		plt.axvline(0.5, color='r')
		plt.legend(loc=2)
		plt.savefig(simulation.out+'_KEt_cumfrac.pdf', bbox_inches='tight', format='pdf')
		plt.show()

	else:
		if not os.path.isfile(simulation.out+'_KEMIdist.npy') or not os.path.isfile(simulation.out+'_vdisp.npy'):
			it = np.argmin(np.absolute(tarch-time))
			vswap = np.swapaxes(v, 0,2)
			rswap = np.swapaxes(r, 0,2)
			vx = vswap[0]
			vy = vswap[1]
			vz = vswap[2]
			x = rswap[0]
			y = rswap[1]
			z = rswap[2]
			if mfilt!=None:
				biginds = np.where(m>mfilt)[0]
				x = x[biginds]
				y = y[biginds]
				z = z[biginds]
				vx = vx[biginds]
				vy = vy[biginds]
				vz = vz[biginds]
				m = m[biginds]
	
			vx= np.swapaxes(vx,0,1)
			vy = np.swapaxes(vy,0,1)
			vz = np.swapaxes(vz,0,1)
	
			x= np.swapaxes(x,0,1)
			y = np.swapaxes(y,0,1)
			z = np.swapaxes(z,0,1)

			print(x.shape)

			if minKE:
				if not os.path.isfile(simulation.out+'_KEmin.npy'):
					print('Minimising KE. This may take several minutes...')
					ke_min, theta, phi = minimise_KE(x[it], y[it], z[it], vx[it], vy[it], vz[it], m, KEtype=0, res=10, autocent=autocent, fsize=fsize, radlim=radlim, radcentre=radcentre, hm=hm, centre=centre)
					print('Minimised KE bias for theta = {0}, phi = {1}: {2}'.format(theta, phi, ke_min))
					np.save(simulation.out+'_KEmin', np.array([theta, phi, ke_min]))
				else:
					theta, phi, ke_min = np.load(simulation.out+'_KEmin.npy')
					print('Minimised KE bias for theta = {0}, phi = {1}: {2}'.format(theta, phi, ke_min))
			elif minVZ:
				if not os.path.isfile(simulation.out+'_minVZ.npy'):
					print('Minimising v_z. This may take several minutes...')
					vz_min, theta, phi = minimise_VZ(x[it], y[it], z[it], vx[it], vy[it], vz[it], m, res=10, autocent=False, fsize=None, radlim=radlim, radcentre=radcentre, hm=hm, centre=centre)
					print('Minimised v_z for theta = {0}, phi = {1}: {2}'.format(theta, phi, vz_min))
					np.save(simulation.out+'_minVZ', np.array([theta, phi, vz_min]))
				else:
					theta, phi, vz_min = np.load(simulation.out+'_minVZ.npy')
					print('Minimised v_z bias for theta = {0}, phi = {1}: {2}'.format(theta, phi, vz_min))

			print('Vdisp before:', np.std(vx[it]), np.std(vy[it]), np.std(vz[it]))

			xt,  yt, zt, vxt, vyt, vzt = rotate_rx(x[it], y[it], z[it], vx[it],vy[it],vz[it],theta, phi)
	
			print('Vdisp after:', np.std(vxt), np.std(vyt), np.std(vzt))

			print(xt.shape)
	

			if autocent:
				rt = np.swapaxes(np.array([xt,yt,zt]),0,1)
				ctmp = cluster_calcs.empirical_centre(rt, radcentre, 2, 50, 3)
			else:
				ctmp = np.array([centre[0],centre[1]])

			xt = xt-ctmp[0]
			yt = yt-ctmp[1]
			if hm:
				rmags = np.sqrt(xt**2+yt**2)
				irsrt = np.argsort(rmags)
				rsort = rmags[irsrt]
				msort = m[irsrt]
				cumdist = np.cumsum(msort)
				ncumdist = cumdist/cumdist[-1]
				frac_func = interpolate.interp1d(ncumdist, rsort)
				r_hm = frac_func(0.5)
				subvals = np.where(rmags<r_hm)[0]
				xt = xt[subvals]
				yt = yt[subvals]
				zt = zt[subvals]
				vxt = vxt[subvals]
				vyt = vyt[subvals]
				vzt = vzt[subvals]
				mt = m[subvals]
			elif radlim!=None:
				rmags = np.sqrt(xt**2+yt**2)
				subvals = np.where(rmags<radlim)[0]
				xt = xt[subvals]
				yt = yt[subvals]
				zt = zt[subvals]
				vxt = vxt[subvals]
				vyt = vyt[subvals]
				vzt = vzt[subvals]
				mt = m[subvals]
			elif fsize!=None:
				print('Field size: {0}'.format(fsize))
				subvals = np.where((np.absolute(xt)<fsize)&(np.absolute(yt)<fsize))[0]
				xt = xt[subvals]
				yt = yt[subvals]
				zt = zt[subvals]
				vxt = vxt[subvals]
				vyt = vyt[subvals]
				vzt = vzt[subvals]
				mt = m[subvals]
	
			if len(xt)>fix_n:
				print('Restricting consideration to {0}/{1} stars.'.format(fix_n, len(xt)))
			else:
				print('Error: too few stars in field ({0}/{1}).'.format(len(xt), fix_n))
				exit()

			KE_MI = np.zeros((nhist,3))
			vdisp = np.zeros((nhist,3))
			for ihist in range(nhist):
				subvals = np.random.choice(np.arange(len(xt)),size=fix_n, replace=False)
				xtmp = xt[subvals]
				ytmp = yt[subvals]
				vxtmp = vxt[subvals]
				vytmp = vyt[subvals]
				vztmp = vzt[subvals]
				mtmp = mt[subvals]

				KEr, KEt, KErt = get_KE(xtmp, ytmp, vxtmp, vytmp, mtmp)
				MIx, MIy=  cluster_calcs.Morans_I_vcorr(r[it][subvals], v[it][subvals], 0.*au2pc, verror)
				vdispx = np.sqrt(np.mean(vxtmp*vxtmp))
				vdispy = np.sqrt(np.mean(vytmp*vytmp))
				vdispz = np.sqrt(np.mean(vztmp*vztmp))
				KE_MI[ihist] = np.array([KEr,MIx,MIy])
				vdisp[ihist] = np.array([vdispx,vdispy,vdispz])
				if (ihist+1)%100==0:
					KEMI_tmp = np.swapaxes(KE_MI[:ihist+1],0,1)
					KEmean = np.mean(KEMI_tmp[0])
					MIxmean = np.mean(KEMI_tmp[1])
					MIymean = np.mean(KEMI_tmp[2])
					KErsigma = 0.5*(np.percentile(KEMI_tmp[0], 84.2)-np.percentile(KEMI_tmp[0], 15.8))
					MIxsigma = 0.5*(np.percentile(KEMI_tmp[1], 84.2)-np.percentile(KEMI_tmp[1], 15.8))
					MIysigma = 0.5*(np.percentile(KEMI_tmp[2], 84.2)-np.percentile(KEMI_tmp[2], 15.8))
					print('At iteration {3}: <E>={0}, <Ix>={1}, <Iy>={2}'.format(KEmean, MIxmean, MIymean, ihist+1))
					print('sigma_E={0}, sigma_Ix={1}, sigma_Iy={2}'.format(KErsigma, MIxsigma, MIysigma))

				
					vdisp_tmp = np.swapaxes(vdisp[:ihist+1],0,1)
					vxmean = np.mean(vdisp_tmp[0])
					vymean = np.mean(vdisp_tmp[1])
					vzmean = np.mean(vdisp_tmp[2])
					vxsigma = 0.5*(np.percentile(vdisp_tmp[0], 84.2)-np.percentile(vdisp_tmp[0], 15.8))
					vysigma = 0.5*(np.percentile(vdisp_tmp[1], 84.2)-np.percentile(vdisp_tmp[1], 15.8))
					vzsigma = 0.5*(np.percentile(vdisp_tmp[2], 84.2)-np.percentile(vdisp_tmp[2], 15.8))
					print('(sqrt) <<vx^2>>={0}, <<vy^2>>={1}, <<vz^2>>={2}'.format(vxmean, vymean, vzmean))
					print('(sqrt) sigma_<vx^2>={0}, sigma_<vy^2>={1}, sigma_<vz^2>={2}\n_________\n'.format(vxsigma, vysigma, vzsigma))
		
			np.save(simulation.out+'_KEMIdist', KE_MI)
			np.save(simulation.out+'_vdisp', vdisp)
		
		else:
			KE_MI = np.load(simulation.out+'_KEMIdist.npy')
			vdisp = np.load(simulation.out+'_vdisp.npy')
	


	
	
		KEr, MIx,MIy = np.swapaxes(KE_MI, 0,1)

		MIxbar = np.mean(MIx)
		MIxsigma = 0.5*(np.percentile(MIx, 84.2)-np.percentile(MIx, 15.8))
	
		MIybar = np.mean(MIy)
		MIysigma = 0.5*(np.percentile(MIy, 84.2)-np.percentile(MIy, 15.8))

		KErbar = np.mean(KEr)
		KErsigma = 0.5*(np.percentile(KEr, 84.2)-np.percentile(KEr, 15.8))


		vdisp_tmp = np.swapaxes(vdisp,0,1)
		vxmean = np.mean(vdisp_tmp[0])
		vymean = np.mean(vdisp_tmp[1])
		vzmean = np.mean(vdisp_tmp[2])
		vxsigma = 0.5*(np.percentile(vdisp_tmp[0], 84.2)-np.percentile(vdisp_tmp[0], 15.8))
		vysigma = 0.5*(np.percentile(vdisp_tmp[1], 84.2)-np.percentile(vdisp_tmp[1], 15.8))
		vzsigma = 0.5*(np.percentile(vdisp_tmp[2], 84.2)-np.percentile(vdisp_tmp[2], 15.8))

		print('Final values:  <E>={0}, <Ix>={1}, <Iy>={2}'.format(KErbar, MIxbar, MIybar))
		print('Final values:  sigma_E={0}, sigma_Ix={1}, sigma_Iy={2}'.format(KErsigma, MIxsigma, MIysigma))
		print('(sqrt) <<vx^2>>={0}, <<vy^2>>={1}, <<vz^2>>={2}'.format(vxmean, vymean, vzmean))
		print('(sqrt) sigma_<vx^2>={0}, sigma_<vy^2>={1}, sigma_<vz^2>={2}'.format(vxsigma, vysigma, vzsigma))


		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.figure(figsize=(4.,4.))
	
		plt.hist(KEr, bins=10, normed=True, cumulative=True)
		plt.axvline(0.5, color='r')
		plt.axvline(0.43, color='r', linestyle='--')
		plt.axvline(0.59, color='r', linestyle='--')
		plt.xlabel("$\\mathcal{E}$")
		plt.ylabel('Probability')
		plt.savefig(simulation.out+'_MIhist.pdf', bbox_inches='tight', format='pdf')
		plt.show()


		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.figure(figsize=(4.,4.))
		plt.hist2d(MIx, MIy, bins=10, norm=LogNorm(), normed=True, cmap='Greys')

		plt.axvline(0.034, c='r', linestyle='--')
		plt.axvline(0.021, c='r', linestyle='--')
		plt.axhline(0.034, c='r', linestyle='--')
		plt.axhline(0.021, c='r', linestyle='--')

		plt.xlabel("$I_x$")
		plt.ylabel("$I_y$")
		cb = plt.colorbar()
		cb.set_label('PDF Value')
		plt.savefig(simulation.out+'_KEhist.pdf', bbox_inches='tight', format='pdf')
		plt.show()

	return KE_MI

def rotate_rx(x, y,z, vx, vy, vz, theta, phi):
	xtmp, ytmp, ztmp = rotate_vector(x, y, z, theta, phi)
	sumxv, sumyv, sumzv = rotate_vector(x+vx, y+vy, z+vz, theta, phi)
	vxtmp = sumxv - xtmp
	vytmp = sumyv - ytmp
	vztmp = sumzv - ztmp
	return xtmp, ytmp, ztmp, vxtmp, vytmp, vztmp
	
def minimise_VZ(x, y, z, vx, vy, vz, m, res=10, fsize=None, autocent=False, radcentre=10.0,hm=False, radlim=None, centre=(0.,0.)):
	
	theta_vals = np.linspace(0., 2.*np.pi, res)
	phi_vals = np.linspace(0., np.pi, int(res/2.))

	vz_arr = np.zeros((len(theta_vals), len(phi_vals)))

	for itheta in range(len(theta_vals)):
		for iphi in range(len(phi_vals)):
			xtmp, ytmp, ztmp, vxtmp, vytmp, vztmp=rotate_rx(x,y,z,vx,vy,vz,theta_vals[itheta], phi_vals[iphi])

			if autocent:
				rt = np.swapaxes(np.array([xtmp,ytmp,ztmp]),0,1)
				ctmp = cluster_calcs.empirical_centre(rt, radcentre, 2, 50, 3)
			else:
				ctmp = np.array([centre[0],centre[1]])

			xtmp = xtmp-ctmp[0]
			ytmp = ytmp-ctmp[1]
			
			if hm:
				rmags = np.sqrt(xtmp**2+ytmp**2)
				irsrt = np.argsort(rmags)
				rsort = rmags[irsrt]
				msort = m[irsrt]
				cumdist = np.cumsum(msort)
				ncumdist = cumdist/cumdist[-1]
				frac_func = interpolate.interp1d(ncumdist, rsort)
				r_hm = frac_func(0.5)
				subvals = np.where(rmags<r_hm)[0]
				vzt = vztmp[subvals]
			elif radlim!=None:
				rmags = np.sqrt(xtmp**2+ytmp**2)
				subvals = np.where(rmags<radlim)[0]
				vzt = vztmp[subvals]
			elif fsize!=None:
				subvals = np.where((np.absolute(xtmp)<fsize)&(np.absolute(ytmp)<fsize))[0]
				vzt = vztmp[subvals]
			else:
				vzt = vztmp
				

			vz_arr[itheta][iphi] =  np.std(vzt)
		print('Angle exploration complete for {0}/{1} theta values'.format(itheta+1,len(theta_vals)))


	min_ind = np.unravel_index(np.argmin(np.absolute(vz_arr), axis=None), vz_arr.shape)

	vz_min = vz_arr[min_ind]
	
	theta_min = theta_vals[min_ind[0]]
	phi_min = phi_vals[min_ind[1]]

	return vz_min, theta_min, phi_min

	
def minimise_KE(x, y, z, vx, vy, vz, m, KEtype=0, res=50, fsize=None, autocent=False, radcentre=10.0,hm=False, radlim=None, centre=(0.,0.), KEval=0.5):
	
	theta_vals = np.linspace(0., 2.*np.pi, res)
	phi_vals = np.linspace(0., np.pi, int(res/2.))

	ke_arr = np.zeros((len(theta_vals), len(phi_vals)))

	for itheta in range(len(theta_vals)):
		for iphi in range(len(phi_vals)):
			xtmp, ytmp, ztmp, vxtmp, vytmp, vztmp=rotate_rx(x,y,z,vx,vy,vz,theta_vals[itheta], phi_vals[iphi])

			if autocent:
				rt = np.swapaxes(np.array([xtmp,ytmp,ztmp]),0,1)
				ctmp = cluster_calcs.empirical_centre(rt, radcentre, 2, 50, 3)
			else:
				ctmp = np.array([centre[0],centre[1]])

			xtmp = xtmp-ctmp[0]
			ytmp = ytmp-ctmp[1]
			
			if hm:
				rmags = np.sqrt(xtmp**2+ytmp**2)
				irsrt = np.argsort(rmags)
				rsort = rmags[irsrt]
				msort = m[irsrt]
				cumdist = np.cumsum(msort)
				ncumdist = cumdist/cumdist[-1]
				frac_func = interpolate.interp1d(ncumdist, rsort)
				r_hm = frac_func(0.5)
				subvals = np.where(rmags<r_hm)[0]
				xtmp = xtmp[subvals]
				ytmp = ytmp[subvals]
				vxtmp = vxtmp[subvals]
				vytmp = vytmp[subvals]
				mtmp = m[subvals]
			elif radlim!=None:
				rmags = np.sqrt(xtmp**2+ytmp**2)
				subvals = np.where(rmags<radlim)[0]
				xtmp = xtmp[subvals]
				ytmp = ytmp[subvals]
				vxtmp = vxtmp[subvals]
				vytmp = vytmp[subvals]
				mtmp = m[subvals]
			elif fsize!=None:
				subvals = np.where((np.absolute(xtmp)<fsize)&(np.absolute(ytmp)<fsize))[0]
				xtmp = xtmp[subvals]
				ytmp = ytmp[subvals]
				vxtmp = vxtmp[subvals]
				vytmp = vytmp[subvals]
				mtmp = m[subvals]

			ke_arr[itheta][iphi] =  get_KE(xtmp, ytmp, vxtmp, vytmp, mtmp)[KEtype]
		print('Angle exploration complete for {0}/{1} theta values'.format(itheta+1,len(theta_vals)))


	min_ind = np.unravel_index(np.argmin(np.absolute(ke_arr-KEval), axis=None), ke_arr.shape)

	ke_min = ke_arr[min_ind]
	
	theta_min = theta_vals[min_ind[0]]
	phi_min = phi_vals[min_ind[1]]

	return ke_min, theta_min, phi_min

def convfield_KE(simulation, time=0.0, rrange = [1.0, 100.0], fix_n=700,  autocent=True, radcentre=5.0, plot=True):
	
	#rspace = np.logspace(np.log10(rrange[0]), np.log10(rrange[1]), 10)
	rspace = np.linspace(rrange[0], rrange[1], 20)
	KE_r = np.zeros(len(rspace))

	if not os.path.isfile(simulation.out+'_KE_convergence.npy'):
		for ir in range(len(rspace)):
	
			ker, ket, kert = get_KE_fromsim(simulation, mfilt=None,  autocent=True,hm=False, radcentre=radcentre, time=time, radlim=rspace[ir], fix_n=fix_n)
			KE_r[ir] = ker
			print('KE for r={0}: {1}'.format(rspace[ir], ker))
		np.save(simulation.out+'_KE_convergence', np.array([rspace, KE_r]))
	else:
		rspace, KE_r = np.load(simulation.out+'_KE_convergence.npy')


	if plot:
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.figure(figsize=(4.,4.))
	
		plt.plot(rspace, KE_r, 'k')

		plt.xlabel('Projected Field Radius (pc)')
		plt.ylabel('$\mathcal{E}$')
		plt.ylim([0.4,0.75])

		plt.legend(loc='best')
	
		plt.savefig(simulation.out+'_KE_conv.pdf', bbox_inches='tight', format='pdf')
		plt.show()

	return rspace, KE_r



def get_FUVluminosities(mstars):
	fuvdat = np.load(photoevapdir+'/FUV_lum.npy')
			
	mspacefuv = fuvdat[0]
	fuv = fuvdat[1]

	thresh = np.amin(mspacefuv)
	fuvfunc = interpolate.interp1d(mspacefuv, fuv) 

	msttmp = copy.copy(mstars)
	msttmp[np.where(msttmp>99.99)[0]]=99.99
	indices = np.where(msttmp>thresh)[0]

	fuvlum = np.zeros(len(mstars))
	fuvlum[indices] = fuvfunc(msttmp[indices])

	return fuvlum


def plot_g0proj(simulation, fsize=None, centre = (.0, .0), massrange=[0.5,2.0], plot=True,autocent=False, radcentre=10.0, time=0., fix_n=None, force=False, theta=0., phi=0., wext=False):
	
	print('Projected G0/disc calculations...')
	MLLIM= 1.0

	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)

	tunits, munits, runits = simulation.units

	subset_inds = simulation.photoevap_inds


	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol




	if wext:
		pinds = copy.copy(simulation.photoevap_inds)
		disc_ms = np.swapaxes(copy.copy(simulation.phot_wext_m[pinds]),0,1)
		disc_rs = np.swapaxes(copy.copy(simulation.phot_wext_r[pinds]),0,1)
		
		g0vals = np.swapaxes(copy.copy(simulation.FUVwext[pinds]),0,1)
	else:
		pinds = copy.copy(simulation.photoevap_inds)
		disc_ms = np.swapaxes(copy.copy(simulation.phot_m[pinds]),0,1)
		disc_rs = np.swapaxes(copy.copy(simulation.phot_r[pinds]),0,1)
		g0vals = np.swapaxes(copy.copy(simulation.FUV[pinds]),0,1)

	it = np.argmin(np.absolute(tarch-time))

	print('Snap time (Myr):', tarch[it])
	
	dmass = disc_ms[it]
	
	drad = disc_rs[it]
	



	if not os.path.isfile(simulation.out+'_projg0.npy') or force:
		
		if theta!=0. or phi!=0.:
			rt = np.swapaxes(r[it],0,1)
			xt, yt, zt, vxt, vyt, vzt =rotate_rx(rt[0], rt[1], rt[2], 0.,0.,0.,theta, phi)
			rt = np.swapaxes(np.array([xt,yt,zt]),0,1)
		else:
			rt= r[it]
			vt = v[it]
			

		if autocent:
			ctmp = cluster_calcs.empirical_centre(rt, radcentre, 2, 50, 3)
		else:
			ctmp = np.array([centre[0],centre[1]])

	
		im_b = np.where(m>MLLIM)[0]
		m_d = m[pinds]
		m_b = m[im_b]
		rt_d = rt[pinds]
		rt_b = rt[im_b]
	
		xt_d, yt_d, zt_d = np.swapaxes(rt_d, 0,1)
		xt_b, yt_b, zt_b = np.swapaxes(rt_b, 0,1)
		
		xt_d = xt_d-ctmp[0]
		yt_d = yt_d-ctmp[1]

		xt_b = xt_b-ctmp[0]
		yt_b = yt_b-ctmp[1]

		

		Lums = get_FUVluminosities(m_b)

		fluxes  = cluster_calcs.flux(rt_d*1e2/m2pc, Lums,rt_b*1e2/m2pc,3)
		fluxes /= g0

		g0proj = fluxes
		
		
		np.save(simulation.out+'_projg0', np.array([g0proj, dmass, drad, xt_d, yt_d, zt_d , m_d, xt_b, yt_b, zt_b, m_b]))
	else:
		g0proj, dmass, drad, xt_d, yt_d, zt_d , m_d, xt_b, yt_b, zt_b, m_b = np.load(simulation.out+'_projg0.npy')

	
	if plot:
		
		if fsize==None:
			subvals = np.arange(len(xt_d))
			subLvals = np.arange(len(xt_b))
		else:
			subvals = np.where((np.absolute(xt_d)<fsize)&(np.absolute(yt_d)<fsize))[0]
			subLvals = np.where((np.absolute(xt_b)<fsize)&(np.absolute(yt_b)<fsize))[0]

			
		
		xtsub = xt_d[subvals]
		ytsub = yt_d[subvals]
		ztsub = zt_d[subvals]
		mtsub = m_d[subvals]
		disc_mst = dmass[subvals]
		disc_rst = drad[subvals]
		g0proj = g0proj[subvals]
		
		xlargesub = xt_b[subLvals]
		ylargesub = yt_b[subLvals]
		mlarge = m_b[subLvals]


		mpl_cols = ['k', 'r', 'orange', 'lawngreen', 'brown', 'b']
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.figure(figsize=(4.,4.))

		bin_divs = [3.0, 3.35, 3.5, 3.75, 4.0, 4.35,4.7]
		
		surv_frac = np.zeros(len(bin_divs)-1)

		for idiv in range(len(bin_divs)-1):
			in_div =  np.where((np.log10(g0proj)>bin_divs[idiv])&(np.log10(g0proj)<bin_divs[idiv+1]))[0]
			dm_div = dmass[in_div]
			print(len(in_div))
			nsurv = len(np.where(dm_div>MLIM)[0])
			print('Nsurv:', nsurv)
			if len(in_div)>0.:
				surv_frac[idiv] = float(nsurv)/float(len(in_div))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			plt.bar(bin_divs[idiv],surv_frac[idiv], width=w, edgecolor=mpl_cols[idiv],color='None',linewidth=BARLINE)


		#print(dmass, g0proj)
		#print(dmass.shape, g0proj.shape)
		#plt.scatter(g0proj, dmass)
		#plt.xscale('log')
		#plt.yscale('log')
		plt.xlim([3.0, 4.7])
		plt.ylabel('$N_\mathrm{disc}/N_\mathrm{tot}$')
		plt.xlabel('$\log(\mathrm{FUV flux})$  ($G_0$)')
		plt.savefig(simulation.out+'_g0proj.pdf', bbox_inches='tight', format='pdf')
		plt.show()


		plt.figure(figsize=(4.,4.))


		for idiv in range(len(bin_divs)-1):
			in_div =  np.where((np.log10(g0proj)>bin_divs[idiv])&(np.log10(g0proj)<bin_divs[idiv+1]))[0]
			dm_div = dmass[in_div]
			isurv = np.where(dm_div>MLIM)[0]
			idest = np.where(dm_div<=MLIM)[0]
			nsurv = len(isurv)
			print('Disc fraction: {0}/{1}'.format(nsurv, len(in_div)))
			if len(in_div)>0.:
				surv_frac[idiv] = float(nsurv)/float(len(in_div))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			xtss = xtsub[in_div]
			ytss = ytsub[in_div]
			plt.scatter(xtss[isurv],ytss[isurv], c=mpl_cols[idiv], edgecolors=mpl_cols[idiv])
			plt.scatter(xtss[idest],ytss[idest], c='None', edgecolors=mpl_cols[idiv])

		ibig = np.where(mlarge>20.0)[0]
		plt.scatter(xlargesub, ylargesub, marker='*', s=80, c='k')
		
		plt.ylabel('$y$ (pc)')
		plt.xlabel('$x$  (pc)')
		if fsize!=None:
			plt.xlim([-fsize, fsize])
			plt.ylim([-fsize, fsize])

		plt.savefig(simulation.out+'_g0_phys_calc.pdf', bbox_inches='tight', format='pdf')


		plt.figure(figsize=(4.,4.))


		for idiv in range(len(bin_divs)-1):
			in_div =  np.where((np.log10(g0vals[it][subvals])>bin_divs[idiv])&(np.log10(g0vals[it][subvals])<bin_divs[idiv+1]))[0]
			dm_div = dmass[in_div]
			isurv = np.where(dm_div>MLIM)[0]
			idest = np.where(dm_div<=MLIM)[0]
			nsurv = len(isurv)
			print('Disc fraction: {0}/{1}'.format(nsurv, len(in_div)))
			if len(in_div)>0.:
				surv_frac[idiv] = float(nsurv)/float(len(in_div))
			w = bin_divs[idiv+1]-bin_divs[idiv]
			xtss = xtsub[in_div]
			ytss = ytsub[in_div]
			plt.scatter(xtss[isurv],ytss[isurv], c=mpl_cols[idiv], edgecolors=mpl_cols[idiv])
			plt.scatter(xtss[idest],ytss[idest], c='None', edgecolors=mpl_cols[idiv])

		ibig = np.where(mlarge>20.0)[0]
		plt.scatter(xlargesub, ylargesub, marker='*', s=80, c='k')
		
		plt.ylabel('$y$ (pc)')
		plt.xlabel('$x$  (pc)')
		if fsize!=None:
			plt.xlim([-fsize, fsize])
			plt.ylim([-fsize, fsize])

		plt.savefig(simulation.out+'_g0_phys_code.pdf', bbox_inches='tight', format='pdf')

		plt.show()
	return None




def get_cinds(simulation, autocent=True, fsize= 20.0, massrange = [0.1,2.0],radlim=None, hm=False, radcentre=15.0, time=.0, theta=0.0, phi=0., fix_n=None):

	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= 1e-3*runits/tunits

	it = np.argmin(tarch-t)

	rt = r[it]
	xt, yt, zt = np.swapaxes(rt, 0,1)


	if autocent:
		ctmp = cluster_calcs.empirical_centre(rt, radcentre, 2, 50, 3)
	else:
		ctmp = np.array([centre[0],centre[1]])

	xt = xt-ctmp[0]
	yt = yt-ctmp[1]
	if hm:
		print('Finding stars inside half mass..')
		rmags = np.sqrt(xt**2+yt**2)
		irsrt = np.argsort(rmags)
		rsort = rmags[irsrt]
		msort = m[irsrt]
		cumdist = np.cumsum(msort)
		ncumdist = cumdist/cumdist[-1]
		frac_func = interpolate.interp1d(ncumdist, rsort)
		r_hm = frac_func(0.5)
		subvals = np.where((rmags<r_hm)&(m>massrange[0])&(m<massrange[1]))[0]
	elif radlim!=None:
		print('Finding stars inside {0} pc..'.format(radlim))
		rmags = np.sqrt(xt**2+yt**2)
		subvals = np.where((rmags<radlim)&(m>massrange[0])&(m<massrange[1]))[0]
	elif fsize!=None:
		print('Finding stars inside field size: {0}'.format(fsize))
		subvals = np.where((np.absolute(xt)<fsize)&(np.absolute(yt)<fsize)&(m>massrange[0])&(m<massrange[1]))[0]
	else:
		subvals = np.where((m>massrange[0])&(m<massrange[1]))[0]
		
		
	
	
	if fix_n!=None:
		if fix_n>len(subvals):
			print('Subset greater than number of stars ({0}/{1})'.format(fix_n, len(subvals)))
		else:
			subvals = np.random.choice(subvals, size=fix_n, replace=False)


	return subvals
		

	
def plot_KE(simulation, fsize=None, centre = (.0, .0), mfilt=1.0, plot=True,autocent=False, radcentre=5.0, sp_time=0.1, fix_n=None, force=False, theta=0., phi=0.):
	
	print('Kinetic energy plot: Loading simulation...')	

	t = copy.copy(simulation.t)
	r = copy.copy(simulation.r)
	v = copy.copy(simulation.v) 
	m = copy.copy(simulation.m)

	if hasattr(simulation, 'starinds'):
		stit_flag=True
		stinds = simulation.starinds
		tends = simulation.tends
		tends = np.cumsum(tends)
		inext=0
		stit = []
		for it in range(len(t)):
			stit.append(stinds[inext])
			if t[it]>tends[inext] and inext<len(stinds)-1:
				inext+=1
	else:
		stit = [np.arange(len(m)) for it in range(len(t))]

	tunits, munits, runits = simulation.units

	tarch = t*tunits*s2myr
	r *= runits*m2pc
	m *= munits*kg2sol
	v *= 1e-3*runits/tunits

	if sp_time!=None:
		list_indices = time_inds(tarch, dt=sp_time)
		nsnaps = len(list_indices)
	else:
		nsnaps = len(tarch)
		list_indices = np.arange(nsnaps)
	
	biginds = np.where(m>mfilt)[0]

	vswap = np.swapaxes(v, 0,2)
	
	vx = vswap[0][biginds]
	vy = vswap[1][biginds]
	vz = vswap[2][biginds]

	vx= np.swapaxes(vx,0,1)
	vy = np.swapaxes(vy,0,1)
	vz = np.swapaxes(vz,0,1)
	

	rswap = np.swapaxes(r, 0,2)
	#rswap = np.swapaxes(rswap, 1,2)
	x = rswap[0][biginds]
	y = rswap[1][biginds]
	z = rswap[2][biginds]
	x= np.swapaxes(x,0,1)
	y = np.swapaxes(y,0,1)
	z = np.swapaxes(z,0,1)


	t_return = np.zeros(nsnaps)

	if not os.path.isfile(simulation.out+'_KE.npy') or force:
		KEr_rat = np.zeros(nsnaps)
		KEt_rat = np.zeros(nsnaps)
		KErt_rat = np.zeros(nsnaps)
		ival=0
		for it in list_indices:

			if theta!=0. or phi!=0.:
				rt = np.swapaxes(r[it][stit[it]],0,1)
				vt = np.swapaxes(v[it][stit[it]],0,1)
				xt, yt, zt, vxt, vyt, vzt =rotate_rx(rt[0], rt[1], rt[2], vt[0],vt[1],vt[2],theta, phi)
				rt = np.swapaxes(np.array([xt,yt,zt]),0,1)
				vt = np.swapaxes(np.array([vxt,vyt,vzt]),0,1)
			else:
				rt= r[it][stit[it]]
				vt = v[it][stit[it]]
				xt, yt, zt = np.swapaxes(rt, 0,1)
				vxt, vyt, vzt = np.swapaxes(vt,0,1)

			t_return[ival] = tarch[it]
			if autocent:
				ctmp = cluster_calcs.empirical_centre(rt, radcentre, 2, 20, 3)
			else:
				ctmp = np.array([centre[0],centre[1]])

			print(ctmp)
		
			if fsize!=None:
				xt = xt-ctmp[0]
				yt = yt-ctmp[1]
				subvals = np.where((np.absolute(xt)<fsize)&(np.absolute(yt)<fsize))[0]
				xt = xt[subvals]
				yt = yt[subvals]
				zt = zt[subvals]
				vxt = vxt[subvals]
				vyt = vyt[subvals]
				vzt  = vzt[subvals]
				mt = m[subvals]
			else:
				xt = xt-ctmp[0]
				yt = yt-ctmp[1]
				zt = zt
				vxt = vxt
				vyt = vyt
				vzt = vzt
				mt = m

			if fix_n!=None:
				if len(xt)>fix_n:
					print('Restricting consideration to {0}/{1} stars.'.format(fix_n, len(xt)))
					subvals = np.random.choice(np.arange(len(xt)),size=fix_n, replace=False)
					xt = xt[subvals]
					yt = yt[subvals]
					zt = zt[subvals]
					vxt = vxt[subvals]
					vyt = vyt[subvals]
					vzt = vzt[subvals]
					mt = mt[subvals]
				else:
					print('Warning: too few stars ({0}/{1}).'.format(fix_n, len(xt)))

			print(xt[0], len(xt), vxt[0], len(vxt), mt[0], len(mt))

			KEr_rat[ival], KEt_rat[ival], KErt_rat[ival] = get_KE(xt, yt, vxt, vyt, mt)
			print(KEr_rat[ival], KEt_rat[ival], KErt_rat[ival])
			print('At {0} Myr KE in/out: {1}'.format(tarch[it], KEr_rat[ival]))

			ival+=1

		np.save(simulation.out+'_KE', np.array([t_return, KEr_rat, KEt_rat, KErt_rat]))
	else:
		t_return, KEr_rat, KEt_rat, KErt_rat = np.load(simulation.out+'_KE.npy')

	
	if plot:
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.figure(figsize=(4.,4.))
	
		plt.plot(t_return, KEr_rat, 'k', label='Out/In')
		plt.plot(t_return, KEt_rat, 'k:', label='Clockwise/Counter')
		plt.plot(t_return, KErt_rat, 'k--', label='Radial/Azimuthal')

		plt.xlabel('Time (Myr)')
		plt.ylabel('$T^{+}/(T^{+}+T^{-})$')
		plt.ylim([0.,1.])

		plt.legend(loc='best')
	
		plt.savefig(simulation.out+'_KE.pdf', bbox_inches='tight', format='pdf')
		plt.show()
	

	return t_return, KEr_rat, KEt_rat, KErt_rat


		


	
