from __future__ import print_function

import os
import numpy as np
#import pyximport; pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import copy
import sys
import shutil
import glob 


scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(scriptdir)
sys.path.append(scriptdir+'/../general')

import subprocess
#import cluster_calcs as cc

from common import *

import command_class as cclass
import cluster_utils as cu
import cluster_calcs as cc
import nbody6_template
import saveload


#mpl_cols = ['k','b','g','r','orange', 'c', 'm', 'y']

class nbody6_cluster:
	def __init__(self, rstars_pc, vstars_kms, mstars_msol, outname='clustersim', dtsnap_Myr =1e-1, tend_Myr = 1.0, assoc=None, gasparams=None, etai=0.02, etar=0.02, etau=0.2, dtmin_Myr=5e-7, dtadj_Myr=1.0, rmin_pc=1e-6, dtjacc_Myr=0.05, load=False, ctype='clumpy', force_incomp = False, starinds = None, rtrunc=50.0, nbin0=0, aclose_au=50.0):
		self.out = outname
		self.idir = 0
		self.ctype = ctype
		if load:
			load_succ = self.load()
		else:
			load_succ =False

		if not load_succ:
			self.complete=False
			print('No previous simulation found...')

			self.etai = etai
			self.etar = etar
			self.etau = etau
			print('Assigning initial conditions...')
			self.n= len(mstars_msol)
			self.nbin0 = nbin0
			
			rstars, vstars, mstars, runit, tunit, munit = cu.get_nbody_units(mstars_msol, rstars_pc, vstars_kms)
			
			aclose_nb = aclose_au/m2au/runit
			
			self.eclose = np.sqrt(np.median(mstars)/aclose_nb)
			
			self.dtjacc = dtjacc_Myr/tunit/s2myr
			self.tend = tend_Myr/tunit/s2myr
			print('Tend:', self.tend, tend_Myr)
			self.rmin = rmin_pc/runit/m2pc
			self.dtmin = dtmin_Myr/tunit/s2myr
			self.dt = dtsnap_Myr/tunit/s2myr
			print('dt:', self.dt, self.dtmin)
			self.dtopt = self.dt
			self.dtadj = dtjacc_Myr/tunit/s2myr

	
			self.units_SI = np.array([munit, runit, tunit])
			self.units_astro = np.array([munit*kg2sol, runit*m2pc, tunit*s2myr, 1e3*runit/tunit])
			
			
			self.ms = mstars
			self.rs = rstars
			self.vs = vstars
			
			self.minit = self.ms	
			self.vinit =  self.vs
			self.rinit = self.rs

			if type(gasparams)!=type(None):
				gasparams = np.array(gasparams)
				if len(gasparams.shape)==1:
					gasparams[0] /= self.units_astro[0]
					gasparams[1] /= self.units_astro[1]
					gasparams[2] /= self.units_astro[0]/self.units_astro[2]
					gasparams[3] /= self.units_astro[2]
				elif len(gasparams.shape)==2:
					for igas in range(len(gasparams)):
						print(gasparams[igas])
						gasparams[igas][0] /= self.units_astro[0]
						gasparams[igas][1] /= self.units_astro[1]
						gasparams[igas][2] /=  self.units_astro[0]/self.units_astro[2]
						gasparams[igas][3] /= self.units_astro[2]
						print(gasparams[igas])
						
				else:
					print('Error: gas parameter input incorrect')
					print(gasparams)
					sys.exit()
			
			if not hasattr(self,'gasparams'):
				if type(gasparams)!=type(None):
					self.gasparams = np.array(gasparams)
				else:
					self.gasparams = gasparams
			
			
			if not hasattr(self,'starinds'):
				self.starinds =  starinds
				if not starinds is None:
					self.rs = np.ones((len(mstars),3))*1e3
					self.vs = np.zeros((len(mstars),3))
					self.rs[self.starinds[0]] = self.rinit[self.starinds[0]]
					self.vs[self.starinds[0]] = self.vinit[self.starinds[0]]

			self.indicts = []

			if type(self.gasparams)!=type(None):
				print('Assigning gas potential parameters...')
				if len(self.gasparams.shape)==1:
					if self.gasparams[2]==0.0:
						self.dirs = ['pre_exp', 'post_exp']
						self.tends = [self.gasparams[3], self.tend-self.gasparams[3]]
						self.gasparams = [self.gasparams]
						self.gasparams.append(None)
					else:
						self.dirs = ['run_dir']
						self.gasparams = [self.gasparams]
						self.tends = [self.tend]
				elif len(self.gasparams.shape)==2:
					self.dirs = ['g{0}'.format(ig) for ig in range(self.gasparams.shape[0])]
					tvalues = np.swapaxes(self.gasparams, 0,1)[3]
					self.tends = [tval for tval in tvalues]
					self.gasparams = [gpar for gpar in self.gasparams]
					final_dt = self.tend
					for it in range(len(self.tends)):
						final_dt -= self.tends[it]
					print('Final tjump:', final_dt, np.sum(self.tends), self.tend)
					
					if final_dt > self.tend/1e2:
						self.tends.append(final_dt)
						self.gasparams.append([0., self.gasparams[-1][1], 0.0, self.tend])
						self.dirs.append('g_final')
				else:
					print('Error: gas parameter input incorrect')
					print(self.gasparams)
					sys.exit()
			else:
				self.gasparams = [None]
				self.dirs = ['run_dir']
				self.tends = [self.tend]
	
			print('Creating run directories...')
			print(self.dirs, self.tends)
			
			if self.tends[0]<self.tend*1e-10:
				print('First time jump small, removing...')
				self.tends = self.tends[1:]
				self.dirs = self.dirs[1:]
				self.gasparams = self.gasparams[1:]


			if self.tends[-1]<self.tend*1e-10:
				print('Last time jump small, removing...')
				self.tends = self.tends[:-1]
				self.dirs = self.dirs[:-1]
				self.gasparams = self.gasparams[:-1]
			
			self.adjust_dt()
							
			for dname in self.dirs:
				if not os.path.isdir(dname):
					os.makedirs(dname)
				"""fnames = glob.glob(self.out+'*')
				for fle in fnames:
					shutil.copyfile(fle, dname+'/'+fle)"""

			print('Saving simulation setup...')

			self.save()

		if load_succ:
				print('Load successful. ')
				if self.complete:
						print('Simulation has been flagged as complete...')
        
		if load_succ and hasattr(self, 'tends'):
			if np.sum(self.tends)>self.tend and len(self.tends)>1:
				lg = -1
				while np.sum(self.tends)>self.tend:	
					self.tends[lg] -= min(self.tends[lg], np.sum(self.tends)- self.tend)
					lg -= 1
			elif self.tends[-1]>self.tend and len(self.tends)==1:
				self.tends[-1] = self.tend

			if self.tends[0]<self.tend*1e-10:
				print('First time jump small, removing...')
				self.tends = self.tends[1:]
				self.dirs = self.dirs[1:]
				self.gasparams = self.gasparams[1:]


			while self.tends[-1]<self.tend*1e-10:
				print('Last time jump small, removing...')
				self.tends = self.tends[:-1]
				self.dirs = self.dirs[:-1]
				self.gasparams = self.gasparams[:-1]
			print('Recovered simulation segments with the following end times (nbody units):')
			print(self.tends)
			if hasattr(self, 't'):
				if self.t[-1]>self.tends[-1]*0.99:
					print('Last t:', self.t[-1])
					print('Specified end time: ', self.tend)
					print('Setting simulation to "complete"')
					self.complete=True
		if not hasattr(self, 'r') or force_incomp:
			print('No positional array found, assuming simulation is incomplete...')
			self.complete=False
	

	def save(self):
		if not os.path.exists('obj'):
			os.makedirs('obj')

		saveload.save_obj(self, self.out+'.sim')
		
		return None

	def load(self):

		if os.path.exists('obj/'+self.out+'.sim.pkl'):

			oldclass = saveload.load_obj(self.out+'.sim')
			oldprops = oldclass.__dict__ 
			for okey in oldprops:
				setattr(self, okey, getattr(oldclass,okey))
			
			return True
		else:
			return False

	def adjust_dt(self, min_snaps=None):
		
		dt_round = 10.0
		itry =0
		while dt_round > self.dt:
			itry+=1
			dt_round /=10.0
				
			if itry>10:
				print('Error: check timestep definition ({0})'.format(self.dt))
				sys.exit()

		self.dt = dt_round

		if hasattr(self, 'tends') and hasattr(self, 'idir'):
			if len(self.tends)>1:
				iatt = 1
				tfrac = float(self.tends[self.idir]/self.dt)
				
				while np.absolute(int(tfrac)-tfrac)/tfrac >self.dtjacc:
					self.dt/=10.0
					
					tfrac = float(self.tends[self.idir]/self.dt)
					iatt+=1
					print('Adjusting timestep for gas jump:', self.dt)
					print('dt error:', np.absolute(int(tfrac)-tfrac)/tfrac)
					if iatt>20:
						print('Error: check gas jump times ({0})'.format(self.tends[self.idir]))
						print('dt_err/dt_acc: ', np.absolute(int(tfrac)-tfrac)/(tfrac*self.dtjacc))
						sys.exit()
		
		print('Snapshot timestep: {0}'.format(self.dt))
		
		return None


	def write_to_input(self, restart=None):
	
		self.adjust_dt()

		
		if type(self.starinds)!=type(None):
			stinds =self.starinds[self.idir]
		else:
			stinds = np.arange(len(self.ms))

		"""

		1 1000000.0 1.E6 40 40 640
		16000 1 10 43532 100 1
		0.02 0.02 0.1 1.0 1.0 1000.0 2.0E-05 1.0 0.7
		0 1 1 0 1 0 4 0 0 2
		0 1 0 1 2 0 0 0 3 6
		1 0 2 0 0 2 0 0 0 2
		1 0 2 1 1 0 1 1 0 0
		0 0 0 0 0 0 0 0 0 0
		4.0E-06 5E-4 0.1 1.0 1.0E-06 0.01 1.0
		2.35 20.0 0.08 0 0 0.001 0 1.0
		0.5 0.0 0.0 0.0
		"""

		indict  = {}
		if restart is None:
			indict['KSTART'] = 1 #1
		else:
			indict['KSTART'] = int(2+(restart%3)) #2
			if os.path.isfile('comm.1'):
				if os.path.isfile('comm.2'):
					sdate1 = os.path.getmtime('comm.1')
					sdate2 = os.path.getmtime('comm.2')
					ssize1 = os.path.getsize('comm.1')
					ssize2 = os.path.getsize('comm.2')
					if sdate1<sdate2 and ssize2>0.5*ssize1:
						shutil.copyfile('comm.1','comm_backup.1')
						shutil.copyfile('comm.2', 'comm.1')
			elif os.path.isfile('comm.2'):
				shutil.copyfile('comm.2','comm.1')
			else:
				raise Exception('Restart attempted without restart file.')
			
		indict['TCOMP'] = 1000000.0 #2
		#End time in Myr - don't use, use nbody units instead (TCRIT)
		indict['TCRITP'] = self.tend*self.units_astro[2] #3
		
		if type(self.starinds)==type(None):
			indict['N'] = int(self.n) #4
		else:
			indict['N'] = len(self.starinds[self.idir])
		indict['NFIX'] = 1 #5
		indict['NCRIT'] = -1 #6
		indict['NNBOPT'] = int(min(max(2.*(float(indict['N'])/100.)**0.5, 10.), 300.)) #7
		indict['NRUN'] = 1 #8
		indict['NCOMM'] = 10

		#I think this takes ages....
		#ETAI - tstep factor for irregulart force poly 1
		indict['ETAI'] = self.etai
		#ETAR - time step factor for reg	2
		indict['ETAR'] = self.etar
		#RS0 - guess for all radii of nghbour spheres (nbody units)	3
		rmags = np.apply_along_axis(np.linalg.norm, 1, self.rs[stinds])
		bigR = np.median(rmags)
		bigN = len(np.where(rmags<bigR)[0])
		rho = float(bigN/(4.*np.pi*bigR*bigR*bigR/3.))
		rguess = 0.05*float(indict['NNBOPT'])*np.power(3./(4.*np.pi*rho),0.3333)
		print('Rguess:', rguess)

		indict['RS0'] =  rguess #rguess
		
		
		indict['RBAR'] = self.units_astro[1]
		indict['ZMBAR'] = np.mean(self.ms)*self.units_astro[0]
		#TCRIT - termination time in nbody units	6
		indict['TCRIT'] = self.tends[self.idir]
		
		print('Length scale [pc]:',indict['RBAR'])
		print('Units:', self.units_astro)
		#QE - energy tolerance	7
		if type(self.gasparams[self.idir])!=type(None):
			indict['QE'] = 5.0E-02
		else:
			indict['QE'] = 5.0E-02


		#NEW BLOCK _______________________________
		#0 1 1 0 1 0 4 0 0 2
		indict['KZ'] = []
		#KZ(1) - save file to fort.1 (1 - end of run or when dummy file STOP, 2- every 100*NMAX steps 8
		indict['KZ'].append(1) 
		#KZ(2) - save file to fort.2 (output time 1, output time and restart of energy error>5*QE) 9
		indict['KZ'].append(2)
		#KZ(3) - save basic data to file conf.3 at output time 10
		indict['KZ'].append(1)
		#KZ(4) - supress (?) binary diagnostics on bdat.4 11
		indict['KZ'].append(0)
		#KZ(5) - initial conditions of the particle distribution if KZ(22) = 0 12
		indict['KZ'].append(0)
		#KZ(6) - bodief.f output significant binaries at main output 13
		indict['KZ'].append(1)
		#KZ(7) - determine Lag. radii avaerage mass, particle counters, average velocity, dispersion and rotational, within Lagrangian radii 14
		indict['KZ'].append(0)
		#KZ(8) - Primordial binaries initializations and output 15
		indict['KZ'].append(2)
		#KZ(9) - binary diagnositics 16
		indict['KZ'].append(1)
		#KZ(10) - K.S. regularizations diagnostics 17
		indict['KZ'].append(2)


		#0 1 0 0 2 0 0 3 3 6
		#NEW BLOCK _______________________________
		#0 1 0 1 2 0 0 0 3 6

		#supressed 18
		indict['KZ'].append(0)
		#KZ(12) - >0 HR diagnositics of evolving stars with output time interval DTPLOT --> -1 used if KX(19)=0 19
		indict['KZ'].append(0)
		#KZ(13) - interstellar clouds 20
		indict['KZ'].append(0)	
		#KZ(14) - external tidal force 21
		if type(self.gasparams[self.idir])==type(None):
			indict['KZ'].append(0)
		else:
		
			indict['KZ'].append(4)
			indict['MP'] = self.gasparams[self.idir][0]
			indict['AP'] = self.gasparams[self.idir][1]
			indict['MPDOT'] = self.gasparams[self.idir][2]
			indict['TDELAY'] = self.gasparams[self.idir][3]
			
		#KZ(15) - Triple, quad, chain and merger searc 22
		indict['KZ'].append(2)		
		#KZ(16) -Auto-adjustment regularisation parameters 23
		indict['KZ'].append(1)
		#KZ(17) - auto asjust ETAI etc. 24
		indict['KZ'].append(1)
		#KZ(18) - hierarchical systems 25 
		indict['KZ'].append(0)
		#KZ(19) - stellar evolution mass loss 26
		indict['KZ'].append(0)
		#KZ(20) - IMF, needs KZ(22)=0 or 9 27 +8 (negative for preserved RBAR, ZMBAR)
		indict['KZ'].append(-1)

		
		#NEW BLOCK _______________________________
		#1 2 2 0 0 2 0 0 0 2
		#KZ(21) - extra diagnostics information at main output every DELTAT 28
		indict['KZ'].append(1)
		#KZ(22) -  INITIALIZATION OF BASIC PARTICLE DATA, MASS POSITION VELOCITY
		#[0; based on KZ(5), IMF on KZ(2), 1; write ICs in dat.10, 2: nbody-format (7 params per line, mas, pos, vel)
		#3: tree format, 4:starlab, ... , 9: Nbody, ignore mass and use Kz(20), 10: Nbody and units astrophysical (M_sol, pc, km/s)]   29
		indict['KZ'].append(2)
			
		#KZ(23) - Removal of escapers 30
		indict['KZ'].append(0)
		#KS(24) Initial conditions for subsystems 31
		indict['KZ'].append(0)
		#KS(25) Vel kicks for wds 32
		indict['KZ'].append(0)
		#KS(26) Slow-down of two-body motion, increase the regularization integration efficiency = 3: Rectify to get better energy conservation 33 +8
		indict['KZ'].append(2)
		#KZ(27) Two-body tidal circularization 34
		indict['KZ'].append(0)
		#KZ(28) Magnetic braking and gravitational radiation for NS or BH binaries 35
		indict['KZ'].append(0)
		#KZ(29) suppressed (boundary reflection) 36
		indict['KZ'].append(0)
		#KZ(30) hierarchical reg if not 37
		indict['KZ'].append(2)

		#NEW BLOCK _______________________________
		#1 0 2 1 1 0 1 1 2 0
		#KZ(31) com correction after energy check 38
		indict['KZ'].append(0)
		#KZ(32) adjustment of DTADJ based on binding energy of cluster 39 +8 (=47)
		indict['KZ'].append(0)
		#KZ(33) block-step stats at main output
		indict['KZ'].append(2)
		#KZ(34) roche-lobe overflow
		indict['KZ'].append(0)
		#KZ(35) TIME reset to zero 
		indict['KZ'].append(0)
		#KZ(36) (supressed) step reduction for hierarchical
		indict['KZ'].append(0)
		#KZ(37) nbr list additions #was 1
		indict['KZ'].append(1)
		#KZ(38) nbr force poly corrections during reg block step calc #was 1
		indict['KZ'].append(1)
		#KZ(39) nbr radius adjustment method # was 3
		#Use 0 if system has unique density centre and smooth density profgile
		indict['KZ'].append(1)
		if hasattr(self, 'ctype'):
			if self.ctype =='clumpy':
				indict['KZ'][-1]=1
		#0 0 0 0 0 2 -3 0 0 0
		#KZ(40) = 0: For the initialization of particle time steps, use only force and its first derivative, to estimate. 
		#This is very efficent. > 0: Use Fploy2 (second and third order force derivatives calculation) to estimate the initial time steps. #was 1
		indict['KZ'].append(0)

		#KZ(41) proto-star evol
		indict['KZ'].append(0)
		#KZ(42) init binary dist
		indict['KZ'].append(0)
		#KZ(43) unused
		indict['KZ'].append(0)
		#KZ(44) unused
		indict['KZ'].append(0)
		#KZ(45) unused		
		indict['KZ'].append(0)
		#KZ(46) HDF5/BINARY/ANSI format output and global param output
		indict['KZ'].append(0)
		#Frequency for 46
		indict['KZ'].append(0)
		#KZ(48) unused
		indict['KZ'].append(0)
		#KZ(49) Computation of moments of INertia 
		indict['KZ'].append(0)
		#KZ(50) Unused
		indict['KZ'].append(0)


		#1.0E-5 2E-4 0.1 1.0 1.0E-06 0.01 0.125
		
		#DTMIN tstep criterion for reg search 
		indict['DTMIN'] = self.dtmin #self.dtmin
		#Distance creiterion for reg search
		indict['RMIN'] = self.rmin
		#Reg tstep param (2*pi/ETAU steps/orbit)
		indict['ETAU'] =  self.etau
		#binding energy per unit mass fror hard binary
		indict['ECLOSE'] = self.eclose
		#Gmin relative two-body pert for unperturbed motion
		indict['GMIN'] = 1e-7
		#Secondary termination param for soft binaries
		indict['GMAX'] = 0.01
		#Max time-step 
		indict['SMAX'] = 1.0
		#2.35 20.0 0.08 0 0 0.001 0 1.0
		
		#Power-law index for initial mass function, routine data.F
		indict['ALPHA'] = 2.35
		#Maximum particle mass before scaling (based on KZ(20); solar mass unit)
		indict['BODY1'] = 100.0
		#Minimum particle mass before scaling
		indict['BODYN'] = 0.08
		# Number primordial binaries
		indict['NBIN0'] = self.nbin0	
		#Number of prim hierarchichal 
		indict['NHI0'] = 0
		#Metal abundance
		indict['ZMET'] = 0.001
		#EPOCH0 ecolutionary epoch
		indict['EPOCH0'] = 0
		#Plotting interval for stellar evolution HRDIAG (N-body units; >= DELTAT)
		indict['DTPLOT'] = 1.0
		
		if type(self.gasparams[self.idir])!=type(None) and indict['KZ'][21]!=10:
			stell_pot = cc.stellar_potential(self.rs[stinds], self.ms[stinds])
			gas_pot = cc.gas_potential(self.rs[stinds], self.ms[stinds], self.gasparams[self.idir][0], self.gasparams[self.idir][1])
			gpot = np.absolute(stell_pot)+np.absolute(gas_pot)
		else:
			gpot = np.absolute(cc.stellar_potential(self.rs[stinds], self.ms[stinds]))
        
		ke = cc.total_kinetic(self.vs[stinds],  self.ms[stinds])
		Qvir = np.absolute(ke/gpot)
        
		print('Virial Ratio:', Qvir)
		if Qvir<1e-4:
			print('Error: virial ratio too small.')
			print('Q: {0}, GPOT: {1}, TKIN: {2}'.format(Qvir, gpot, ke))
			sys.exit()
		if Qvir>5e3:
			print('Error: virial ratio too large.')
			print('Q: {0}, GPOT: {1}, TKIN: {2}'.format(Qvir, gpot, ke))
			sys.exit()

		indict['Q'] =  Qvir

		if not restart is None and indict['KSTART']>2:
			if indict['KSTART']==3 or indict['KSTART']==5:
				indict['DELTAT'] = max(self.dt/10., min(self.dt, 1e-2))
				indict['DTADJ'] = max(self.dtadj/10.,min(self.dtadj, 1e-2))
				
			elif indict['KSTART']==4 or indict['KSTART']==5:
				indict['ETAI'] /= 10.0 
				indict['ETAI'] = max(indict['ETAI'], 0.001)
				indict['ETAR'] /= 10.0 
				indict['ETAR'] = max(indict['ETAR'], 0.002)
				#indict['ETAU']
				
		else:
			#DTADJ - time interval for parameter adjustment (nbody units)	4 (or maybe this takes ages)
			indict['DTADJ'] = self.dt
			#DELTAT  - time interval for writing output data (*NFIX) -nunits	5
			indict['DELTAT'] = self.dt
		
		print(indict)
	

		
		indict['VXROT'] = 0.0
		indict['VZROT'] = 0.0
		indict['RTIDE'] = 0.0

		if indict['KSTART']==1 or indict['KSTART']==2:
			INSTRING = nbody6_template.infile_string(indict, self.ms[stinds], self.rs[stinds], self.vs[stinds])
		elif indict['KSTART']==3:
			INSTRING = nbody6_template.infile_string_r3(indict)
		elif indict['KSTART']==4:
			INSTRING = nbody6_template.infile_string_r4(indict)
		elif indict['KSTART']==5:
			INSTRING = nbody6_template.infile_string_r5(indict)

		if restart==None:
			instring = self.out
		else:		
			instring= self.out+'_restart'+str(restart)
		
		with open(instring+".input", "w") as f:
			f.write(INSTRING)
		
		if hasattr(self, 'indicts'):
			if len(self.indicts)>self.idir:
				self.indicts[self.idir] = indict
			else:
				self.indicts.append(indict)
		
		
		return instring
		
		
		
		
		
	def read_to_npy(self, full=False, force=False, maxfiles=200, checkQ=False, checkT=True, checkScale=False):

		tunits_ast, munits_ast, runits_ast, vunits_ast = self.units_astro

		if type(self.starinds)!=type(None):
			stinds =self.starinds[self.idir]
		else:
			stinds = np.arange(len(self.ms))
		

		ms  = self.ms[stinds]

		if not os.path.isfile(self.out+'_t.npy') or not os.path.isfile(self.out+'_r.npy') or not os.path.isfile(self.out+'_v.npy') or force:
			read=True
			def read_header(file):
				newline = file.read(4)
				header1 = np.fromfile(file, dtype=np.int32, count=4)
				newline = file.read(4)
				header2 = np.fromfile(file, dtype=np.float32, count=20)
				newline = file.read(4)
				return header1, header2


			files_all = []
			conf_list = []
			subconf_list = []
			for iconf in range(1000):
				files_tmp = glob.glob('conf.3_'+str(iconf)+'.*')
				
				if len(files_tmp)>0:
					fle_nums = np.zeros(len(files_tmp))
					for ifle in range(len(files_tmp)):
						fle_nums[ifle] = float(files_tmp[ifle].split('.')[-1])

					ifile_srt = np.argsort(fle_nums)
						
						#conf_list.append(iconf)
						#subconf_list.append([])
					for ifile in ifile_srt:
						#subconf_list.append(int(fname.split('.')[-1]))
						files_all.append(files_tmp[ifile])

			print('All files:', files_all)
			print('CWD:', os.getcwd())
			while(len(files_all))>maxfiles:
				files_all = files_all[::2]
			fints = [int(x.split('.')[-1]) for x in files_all]


			rs_all = np.zeros((len(fints), len(stinds) , 3))
			vs_all = np.zeros((len(fints), len(stinds) , 3))
			times = np.zeros(len(fints))

			itime = 0
			for fname in files_all:
				datfile = open(fname, 'rb')

				h1, h2 = read_header(datfile)
				M = np.fromfile(datfile, dtype=np.float32, count=h1[0])
				RHO = np.fromfile(datfile, dtype=np.float32, count=h1[0])
				XNS = np.fromfile(datfile, dtype=np.float32, count=h1[0])
				X = np.reshape(np.fromfile(datfile, dtype=np.float32, count=3*h1[0]),(h1[0],3))
				V = np.reshape(np.fromfile(datfile, dtype=np.float32, count=3*h1[0]), (h1[0],3))
				POT = np.fromfile(datfile, dtype=np.float32, count=h1[0])
				NAME = np.fromfile(datfile, dtype=np.int32, count=h1[0])

				NS_arg = (NAME<=self.n)*(NAME>=1)
				NAME = NAME[NS_arg]
				X = X[NS_arg]
				V = V[NS_arg]
				

				NAME -= 1

				#print(X.shape, h1,h2, h2[1])


				times[itime] = h2[1]
				print('{0}: t={1} Myr'.format(fname, times[itime]*self.units_astro[2]))
				for iitime in range(itime, len(rs_all)):
					rs_all[iitime][NAME] =X
					vs_all[iitime][NAME] =V
				
				itime+=1

			#f = FortranFile(fname, 'r')
			#print(fname, f.read_record(float))
			asort = np.argsort(times)
			rs_all = rs_all[asort]
			vs_all = vs_all[asort]
			times = times[asort]

			if hasattr(self, 'tends') and len(times)>0:
				tendtmp = np.asarray(self.tends[self.idir])
				iend = np.argmin(np.absolute(tendtmp-times))
				if iend+1<len(times) and times[iend]<=tendtmp:
				    iend+=1
				
				print('Trimming output to time: {0}/{1} ({2}/{3})'.format(times[iend], tendtmp, iend+1, len(times)))			

				times = times[:iend+1]
				rs_all = rs_all[:iend+1]
				vs_all = vs_all[:iend+1]
			

			if checkScale:
				with open(self.out+'.output') as f:
					for line in f:
						if 'PHYSICAL SCALING:' in line:
							elements = line.split('=')
							scales = []
							for el in elements:
								try:
									flel = float(el[:-5])							
									scales.append(flel)
								except:
									tmp=0
							break


				tnorm = scales[3]/tunits_ast
				rnorm = scales[0]/runits_ast
				vnorm = scales[2]/vunits_ast
				mnorm = scales[1]/munits_ast

				if abs(tnorm-1.0)>2e-2 or abs(rnorm-1.0)>2e-2 or abs(vnorm-1.0)>2e-2 or abs(mnorm-1.0)>2e-2:
					print('Physical units error:')
					print('Time:',tnorm)
					print('Velocity:', vnorm)
					print('Radius:', rnorm)
					print('Mass:', mnorm)
					sys.exit()


			if checkQ:
				print('Checking virial ratio consistency...')
				iline=0
				adj_lines = []
				all_lines = []
				with open(self.out+'.output') as f:
					for line in f:
						if 'ADJUST:' in line:
							adj_lines.append(iline)

						all_lines.append(line)

						iline+=1
				
					last_adj = all_lines[adj_lines[-1]]

				ilet=0
				for let in last_adj:
					if let=='Q':
						virstr = last_adj[ilet+1:ilet+7]
						break
					ilet+=1

				if ilet==len(last_adj):
					raise Exception('Error: Virial ratio not found in ouput.')
				else:
					Qvirout = float(virstr)
			
				print('Calculating potential..')
				if type(self.gasparams[self.idir])!=type(None):
					stell_pot = cc.stellar_potential(rs_all[-1], ms)
					gas_pot = cc.gas_potential(rs_all[-1], ms, self.gasparams[self.idir][0], self.gasparams[self.idir][1])
					gpot = np.absolute(stell_pot)+np.absolute(gas_pot)
				else:
					gpot = np.absolute(cc.stellar_potential(rs_all[-1], ms))

				ke = cc.total_kinetic(vs_all[-1],  ms)
				Qvir = np.absolute(ke/gpot)
				print('Virial Ratio Sim/PP: {0}/{1}'.format(Qvirout, Qvir))
				if abs(Qvir-Qvirout)/Qvirout>= 5e-2:
					raise Exception('virial ratio calculation mismatch - sim output = {0}, pp calc = {1}'.format(Qvirout, Qvir))
			if checkT:
				print(times, self.tends[self.idir])
				itmax = np.argmin(np.absolute(times-self.tends[self.idir]))
				print('Final time in sim section: {0}/{1}'.format(times[itmax], self.tends[self.idir]))
				if hasattr(self, 'dtjacc'):
					chkval = self.dtjacc
				else:
					chkval = 0.05
				
				if self.idir< len(self.tends)-1:
					if np.absolute(times[itmax]-self.tends[self.idir])/self.tends[self.idir] >chkval:
						raise Exception('finish time inaccuracy')
				else:
					if (self.tends[self.idir]- times[itmax])/self.tends[self.idir] >chkval:
						raise Exception('finish time inaccuracy')
				
				itimes = np.where(times<=times[itmax])[0]
				times = times[itimes]
				rs_all = rs_all[itimes]
				vs_all = vs_all[itimes]
				
			if type(self.starinds)!=type(None):
				all_inds = np.arange(len(self.ms))
				nstinds =  all_inds[~np.in1d(all_inds, stinds)]
				
				#Remove escapers
				iesc_sim = np.where(np.apply_along_axis(np.linalg.norm, 1, rs_all[-1])>self.rtrunc)[0]
				iesc = stinds[iesc_sim]

				if len(iesc)>0:
					print('Removing %d escapers'%(len(iesc)))
					print(rs_all[-1][iesc_sim])

					for iid in range(self.idir+1, len(self.starinds)):			
						stinds_nxt= self.starinds[iid]				
						stinds_nxt = stinds_nxt[~np.in1d(stinds_nxt, iesc)]
						self.starinds[iid] = stinds_nxt
					self.save()

				
				#Indices which are not in present pop but are in the next pop:
				#set to initial value
				if self.idir<len(self.starinds)-1:
					stinds_nxt = self.starinds[self.idir+1]
				else:
					stinds_nxt = np.array([], dtype=int)
							
				

				newinds = np.intersect1d(nstinds, stinds_nxt)

				rs_all_tmp = np.zeros((self.n, len(times),3))
				vs_all_tmp = np.zeros((self.n, len(times),3))

				rs_all_tmp[stinds] = np.swapaxes(rs_all,0,1)
				vs_all_tmp[stinds] = np.swapaxes(vs_all,0,1)
				
				rs_all_tmp[nstinds]  = np.ones(rs_all_tmp[nstinds].shape)*1e3
				vs_all_tmp[nstinds]  = np.ones(rs_all_tmp[nstinds].shape)*1e3

				
				rs_all_tmp = np.swapaxes(rs_all_tmp,0,1)	
				vs_all_tmp = np.swapaxes(vs_all_tmp,0,1)

				if len(newinds)>0:
					rs_all_tmp[-1][newinds]= self.rinit[newinds]
					vs_all_tmp[-1][newinds] = self.vinit[newinds]

				rs_all = rs_all_tmp
				vs_all = vs_all_tmp

			
			np.save(self.out+'_t', times)
			np.save(self.out+'_r', rs_all)
			np.save(self.out+'_v', vs_all)
						
			if full:
				self.r = rs_all
				self.t = times
				self.v = vs_all
				self.save()
		else:
			times = np.load(self.out+'_t.npy')
			rs_all = np.load(self.out+'_r.npy')
			vs_all = np.load(self.out+'_v.npy')

			if full:
				self.r = rs_all
				self.t = times
				self.v = vs_all
				self.save()	
		
		if len(rs_all)>0:
			self.rs= rs_all
			self.vs = vs_all
			self.t = times
		else:
			self.t =0.0
		#self.save()

		return self.rs[-1], self.vs[-1], self.ms, self.t[-1], tunits_ast, munits_ast, runits_ast 


	def magas_tseries(self):
		tgas = copy.copy(self.t)
		mgas = np.zeros(len(self.t))
		agas = np.zeros(len(self.t))

		gpar_tmp  =copy.copy(self.gasparams)

		
		mgas[:] = self.gasparams[0][0]
		agas[:] =  self.gasparams[0][1]
		
		ignext = 0		
		tlast = 0.
		for it in range(len(tgas)):
			if tgas[it]>self.gasparams[ignext][3]+tlast:
				mgas[it:] = self.gasparams[ignext+1][0]
				agas[it:] = self.gasparams[ignext+1][1]
				ignext+=1
				tlast=  tgas[it]

			if ignext>=len(self.gasparams)-1:
				break
		
		
		return mgas, agas
	

	
	def combine(self, reread=False):
		if not self.complete or reread:
			idir = 0
			for idir in range(len(self.dirs)):
				d = self.dirs[idir]
				if os.path.isdir(d):
					if idir==0:
						t = np.load(d+'/'+self.out+'_t.npy')
						r = np.load(d+'/'+self.out+'_r.npy')
						v = np.load(d+'/'+self.out+'_v.npy')
						#tunits, munits, runits = np.load(self.out+'_units.npy')
					else:
						r = np.append(r, np.load(d+'/'+self.out+'_r.npy'), axis=0)
						v = np.append(v, np.load(d+'/'+self.out+'_v.npy'), axis=0)
						ttmp =np.load(d+'/'+self.out+'_t.npy')
						ttmp += t[-1]
						t = np.append(t, ttmp, axis=0)
				else:
					raise Exception('"{0}" not found.'.format(d))
		
			np.save(self.out+'_t', t)
			np.save(self.out+'_r', r)
			np.save(self.out+'_v', v)
			#np.save(self.out+'_units', np.array([tunits, munits, runits]))

			self.r = r 
			self.v = v
			self.t = t
			self.m = self.ms
			#self.units_SI = np.array([tunits, munits, runits])
			self.save()
		
		return None

	
	def run_nbody(self, reread=False):
		homedir = os.getcwd()

		if not self.complete or reread:
	
			for idir in range(len(self.dirs)):
				self.idir=idir

				os.chdir(homedir)
				if not os.path.isdir(self.dirs[idir]):
					os.makedirs(self.dirs[idir])
				os.chdir(self.dirs[idir])
				print('Current directory:', os.getcwd())
			
				outfiles = glob.glob('conf.3_0.*')


				
				if not os.path.isfile(self.out+'.input'):
					self.write_to_input()
				else:
					print('Input file detected.')
					
				if len(outfiles)==0:
					RUN_STR =  NBODYEXE + " < {0} > {1}".format(self.out+'.input', self.out+'.output')
					print(RUN_STR)
					#command = cclass.Command(RUN_STR)

					#command.run(timeout=20000)
					subprocess.run(RUN_STR) 

				else:
					print('Output file detected.')	
				
				if hasattr(self, 'tends'):
					ttmp = 0.0
					iatt=0
					tend = self.tends[idir]
				else:
					tend = self.tend
				
				
				while (tend-ttmp)/tend > 0.05 and iatt<3:		
					rtmp, vtmp, mtmp, ttmp, tunits, munits, runits = self.read_to_npy(force=reread, checkT=False)
					
					if (tend-ttmp)/tend > 0.05 and iatt==0:
						rtmp, vtmp, mtmp, ttmp, tunits, munits, runits = self.read_to_npy(force=True, checkT=False)

						print('Output files found up to time {0}'.format(ttmp))
			
					if (tend-ttmp)/tend > 0.05:

						print('Did not make it to end time on previous attempt.')
						print('New attempt {0} starting at time {1}'.format(iatt, ttmp))
						print(self.tends, ttmp)
						print('T_end = {0}/{1}'.format(ttmp, tend))
						inname = self.write_to_input(restart=0)
						RUN_STR_NEW =  NBODYEXE + " < {0} > {1}".format(inname+'.input', inname+'.output')
						print(RUN_STR_NEW)
						#command = cclass.Command(RUN_STR_NEW)
						#command.run(timeout=20000)
						subprocess.run(RUN_STR) 
						rtmp, vtmp, mtmp, ttmp, tunits, munits, runits = self.read_to_npy(force=True, checkT=False)
					
					iatt+=1
				if (self.tends[idir]-ttmp)/self.tends[idir] > self.dtjacc and iatt>=3:
					raise Exception('Failure to run for {0} after {1} attempts...'.format(self.dirs[idir], iatt))
			os.chdir(homedir)
		return None

	def evolve(self, reread=True):
		self.run_nbody(reread=reread)
		self.combine(reread=reread)
