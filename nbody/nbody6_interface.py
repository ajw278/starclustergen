from __future__ import print_function

import os
import numpy as np
#import pyximport; pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import time
import copy
import sys
import shutil
import glob 
from multiprocessing import Process, Queue
import scipy.interpolate as interpolate

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, scriptdir)
sys.path.insert(0, scriptdir+'../general')

#import cluster_calcs as cc

from common import *

import command_class as cclass
import cluster_utils as cu
import cluster_calcs as cc
import nbody6_template
import saveload


#mpl_cols = ['k','b','g','r','orange', 'c', 'm', 'y']

class nbody6_cluster:
	def __init__(self, rstars, vstars, mstars, tunit=(1./s2myr), munit=(1./kg2sol), runit=(1./m2pc), outname='clustersim', dtsnap =1e-1, tend = 1.0, assoc=None, gasparams=None, etai=0.005, etar=0.01, etau=0.2, dtmin=5e-7, dtadj=1.0, rmin=1e-6, astrounits=False, dtjacc=0.05, load=False, ctype='smooth', force_incomp = False, starinds = None, rtrunc=50.0):
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
			if not tunit>0. and munit>0. and runit>0.:
				print('Unit definition incorrect.')
				sys.exit()

			self.etai = etai
			self.etar = etar
			self.etau = etau
			self.dtjacc =dtjacc
			self.tend = tend


			self.astunits = astrounits

			print('Assigning initial conditions...')
			self.n= len(mstars)
			if not self.astunits:
				self.rs = rstars
				self.vs  = vstars
				self.ms = mstars			
				self.units = np.array([tunit, munit, runit])
			else:
				self.rs = rstars*runit*m2pc
				self.vs = vstars*runit*1e-3/tunit
				self.ms = mstars*munit*kg2sol
				
				self.units = np.array([(1./s2myr), (1./kg2sol), (1./m2pc)])
				dtmin *= tunit*s2myr
				tend *= tunit*s2myr
				dtadj *= tunit*s2myr
				rmin *= runit*m2pc

				if type(gasparams)!=type(None):
					gasparams = np.array(gasparams)
					if len(gasparams.shape)==1:
						gasparams[0] *= munit*kg2sol
						gasparams[1] *= runit*m2pc
						gasparams[2] *= munit*kg2sol/(tunit*s2myr)
						gasparams[3] *= tunit*s2myr
					elif len(gasparams.shape)==2:
						for igas in range(len(gasparams)):
							print(gasparams[igas])
							gasparams[igas][0] *= munit*kg2sol
							gasparams[igas][1] *= runit*m2pc
							gasparams[igas][2] *= munit*kg2sol/(tunit*s2myr)
							gasparams[igas][3] *= tunit*s2myr
							print(gasparams[igas])
							
					else:
						print('Error: gas parameter input incorrect')
						print(gasparams)
						sys.exit()

			self.minit = self.ms	
			self.vinit =  self.vs
			self.rinit = self.rs

			self.rmin = rmin
			self.dtmin = dtmin
			print('Assigning output timestep...')
			#Note, the values of dt for snapshots etc. should multiply by an integer to get 1

			if not hasattr(self, 'assoc'):
				self.assoc= assoc
			if not hasattr(self,'gasparams'):
				if type(gasparams)!=type(None):
					self.gasparams = np.array(gasparams)
				else:
					self.gasparams = gasparams
			
			
			if not hasattr(self,'starinds'):
				if not type(starinds)==type(None):
					self.starinds =  starinds
					self.rs = np.ones((len(mstars),3))*1e3
					self.vs = np.zeros((len(mstars),3))
					self.rs[self.starinds[0]] = self.rinit[self.starinds[0]]
					self.vs[self.starinds[0]] = self.vinit[self.starinds[0]]
					
				else:
					self.starinds = None


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
			
			self.dt = dtsnap
			self.dtopt = dtsnap
			self.dtadj = dtadj
			
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

		self.rmin = rmin
		self.dtmin = dtmin

		
		if not hasattr(self, 'r') or force_incomp:
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
		if hasattr(self, 'dtopt'):
			self.dt = self.dtopt
		
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
		
		self.dtadj = min(self.dtadj, self.dt)
		print('Snapshot timestep: {0}'.format(self.dt))
		
		return None


	def write_to_input(self, restart=None):
		
		"""for ir in range(len(self.rs)):
			dr = np.linalg.norm(self.rs-self.rs[ir],axis=1)
			dr = np.append(dr[:ir], dr[ir+1:])
			drsmall = np.where(dr<1e-10)[0]
			if np.amin(dr)<1e-4:
				print('Minsep = {0} au'.format(np.amin(dr)*self.units[2]*m2au))
			if len(drsmall)>1:
				print('Number close neighbours: {0}'.format(len(drsmall)))
				print(dr[drsmall])
		sys.exit()"""


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
		if restart==None:
			indict['KSTART'] = 1 #1
		else:
			indict['KSTART'] = int(2+(restart%3)) #2
			if os.path.isfile('fort.1'):
				if os.path.isfile('fort.2'):
					sdate1 = os.path.getmtime('fort.1')
					sdate2 = os.path.getmtime('fort.2')
					ssize1 = os.path.getsize('fort.1')
					ssize2 = os.path.getsize('fort.2')
					if sdate1<sdate2 and ssize2>0.5*ssize1:
						shutil.copyfile('fort.1','fort_backup.1')
						shutil.copyfile('fort.2', 'fort.1')
			elif os.path.isfile('fort.2'):
				shutil.copyfile('fort.2','fort.1')
			else:
				print('Restart attempted without restart file.')
				sys.exit()
			
		indict['TCOMP'] = 1000000.0 #2
		#End time in Myr - don't use, use nbody units instead (TCRIT)
		indict['TCRITP'] = 1E6 #3
		
		if type(self.starinds)==type(None):
			indict['N'] = int(self.n) #4
		else:
			indict['N'] = len(self.starinds[self.idir])
		indict['NFIX'] = 1 #5
		indict['NCRIT'] = -1 #6
		indict['NNBOPT'] = int(min(max(2.*(float(indict['N'])/100.)**0.5, 50.), 300.)) #7
		indict['NRUN'] = 1 #8

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
		rguess = 0.1*float(indict['NNBOPT'])*np.power(3./(4.*np.pi*rho),0.3333)
		print('Rguess:', rguess)

		indict['RS0'] =  rguess #rguess
		
			
		#TCRIT - termination time in nbody units	6
		indict['TCRIT'] = self.tends[self.idir]
	
		#QE - energy tolerance	7
		if type(self.gasparams[self.idir])!=type(None):
			indict['QE'] = 5.0E-02
		else:
			indict['QE'] = 1.0E-03
		indict['RBAR'] = self.units[2]*m2pc
		indict['ZMBAR'] = np.mean(self.ms)*kg2sol*self.units[1]


		indict['KZ'] = []
		#KZ(1) - save file to fort.1 (1 - end of run or when dummy file STOP, 2- every 100*NMAX steps 8
		indict['KZ'].append(0) 
		#KZ(2) - save file to fort.2 (output time 1, output time and restart of energy error>5*QE) 9
		indict['KZ'].append(2)
		#KZ(3) - save basic data to file conf.3 at output time 10
		indict['KZ'].append(1)
		#KZ(4) - supress (?) binary diagnostics on bdat.4 11
		indict['KZ'].append(0)
		#KZ(5) - initial conditions of the particle distribution if KZ(22) = 0 12
		indict['KZ'].append(0)
		#KZ(6) - bodief.f output significant binaries at main output 13
		indict['KZ'].append(0)
		#KZ(7) - determine Lag. radii avaerage mass, particle counters, average velocity, dispersion and rotational, within Lagrangian radii 14
		indict['KZ'].append(1)
		#KZ(8) - Primordial binaries initializations and output 15
		indict['KZ'].append(0)
		#KZ(9) - binary diagnositics 16
		indict['KZ'].append(0)
		#KZ(10) - K.S. regularizations diagnostics 17
		indict['KZ'].append(2)


		#NOTE: Currently has tidal field


		#supressed 18
		indict['KZ'].append(0)
		#KZ(12) - >0 HR diagnositics of evolving stars with output time interval DTPLOT --> -1 used if KX(19)=0 19
		indict['KZ'].append(1)
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
		indict['KZ'].append(0)
		#KZ(17) - auto asjust ETAI etc. 24
		indict['KZ'].append(0)
		#KZ(18) - hierarchical systems 25 
		indict['KZ'].append(0)
		#KZ(19) - stellar evolution mass loss 26
		indict['KZ'].append(0)
		#KZ(20) - IMF, needs KZ(22)=0 or 9 27 +8 (negative for preserved RBAR, ZMBAR)
		indict['KZ'].append(-1)



		#KZ(21) - extra diagnostics information at main output every DELTAT 28
		indict['KZ'].append(1)
		#KZ(22) -  INITIALIZATION OF BASIC PARTICLE DATA, MASS POSITION VELOCITY
		#[0; based on KZ(5), IMF on KZ(2), 1; write ICs in dat.10, 2: nbody-format (7 params per line, mas, pos, vel)
		#3: tree format, 4:starlab, ... , 9: Nbody, ignore mass and use Kz(20), 10: Nbody and units astrophysical (M_sol, pc, km/s)]   29
		if self.astunits:
			indict['KZ'].append(10)
		else:
			indict['KZ'].append(2)
			
		#KZ(23) - Removal of escapers 30
		indict['KZ'].append(0)
		#KS(24) Initial conditions for subsystems 31
		indict['KZ'].append(0)
		#KS(25) Vel kicks for wds 32
		indict['KZ'].append(0)
		#KS(26) Slow-down of two-body motion, increase the regularization integration efficiency = 3: Rectify to get better energy conservation 33 +8
		indict['KZ'].append(3)
		#KZ(27) Two-body tidal circularization 34
		indict['KZ'].append(0)
		#KZ(28) Magnetic braking and gravitational radiation for NS or BH binaries 35
		indict['KZ'].append(0)
		#KZ(29) suppressed (boundary reflection) 36
		indict['KZ'].append(0)
		#KZ(30) hierarchical reg if not 37
		indict['KZ'].append(0)

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
		indict['KZ'].append(0)
		#KZ(38) nbr force poly corrections during reg block step calc #was 1
		indict['KZ'].append(0)
		#KZ(39) nbr radius adjustment method # was 3
		#Use 0 if system has unique density centre and smooth density profgile
		indict['KZ'].append(0)
		if hasattr(self, 'ctype'):
			if self.ctype =='clumpy':
				indict['KZ'][38]=2

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


		#4.0E-06 5E-4 0.1 1.0 1.0E-06 0.01 1.0

		#DTMIN tstep criterion for reg search 
		indict['DTMIN'] = self.dtmin
		#Distance creiterion for reg search
		indict['RMIN'] = self.rmin
		#Reg tstep param (2*pi/ETAU steps/orbit)
		indict['ETAU'] = self.etau
		#binding energy per unit mass fror hard binary
		indict['ECLOSE'] = 1.0
		#Gmin relative two-body pert for unperturbed motion
		indict['GMIN'] = 1e-6
		#Secondary termination param for soft binaries
		indict['GMAX'] = 0.001
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
		indict['NBIN0'] = 0	
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
		if Qvir<1e-2:
			print('Error: virial ratio too small.')
			print('Q: {0}, GPOT: {1}, TKIN: {2}'.format(Qvir, gpot, ke))
			sys.exit()
		if Qvir>5e3:
			print('Error: virial ratio too large.')
			print('Q: {0}, GPOT: {1}, TKIN: {2}'.format(Qvir, gpot, ke))
			sys.exit()

		indict['Q'] =  Qvir



		if restart!=None and indict['KSTART']>2:
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
			indict['DTADJ'] = self.dtadj
			#DELTAT  - time interval for writing output data (*NFIX) -nunits	5
			indict['DELTAT'] = self.dt
		
		"""elif Qvir>=1.0:
			print('Warning: high virial ratio QVIR={0}, switching on hot boundary conditions'.format(Qvir))
			indict['KZ'][28]=1
			indict['Q'] = Qvir
			vswap = np.swapaxes(self.vs, 0,1)
			vdisp =  np.sum(self.ms*vswap[0]*vswap[0])/np.sum(self.ms)
			vdisp += np.sum(self.ms*vswap[1]*vswap[1])/np.sum(self.ms)
			vdisp += np.sum(self.ms*vswap[2]*vswap[2])/np.sum(self.ms)
			indict['SIGMA0'] = np.sqrt(vdisp)*1e2
			print('Sigma_0 = {0}'.format(indict['SIGMA0']))
			print('Need to understand units in the hotsys.f subroutine.')
		else:
			indict['KZ'][28]=0
			indict['Q'] = Q"""

	

		
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

		tunits, munits, runits = self.units

		if type(self.starinds)!=type(None):
			stinds =self.starinds[self.idir]
		else:
			stinds = np.arange(len(self.ms))
		

		ms  = self.ms[stinds]

		read = False

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
				print('{0}: t={1} Myr'.format(fname, times[itime]*tunits*s2myr))
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
				print(self.idir, self.tends, self.tends[self.idir])
				tendtmp = np.asarray(self.tends[self.idir])
				print(tendtmp, times)
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


				tnorm = scales[3]/(tunits*s2myr)
				rnorm = scales[0]/(runits*m2pc)
				vnorm = scales[2]/(runits*1e-3/tunits)
				mnorm = scales[1]/(munits*kg2sol)

				if abs(tnorm-1.0)>2e-2 or abs(rnorm-1.0)>2e-2 or abs(vnorm-1.0)>2e-2:
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
					print('Error: Virial ratio not found in ouput.')
					sys.exit()
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
					print('Error: virial ratio calculation mismatch - sim output = {0}, pp calc = {1}'.format(Qvirout, Qvir))
					sys.exit()

			if checkT:
				itmax = np.argmin(np.absolute(times-self.tends[self.idir]))
				print('Final time in sim section: {0}/{1}'.format(times[itmax], self.tends[self.idir]))
				if hasattr(self, 'dtjacc'):
					chkval = self.dtjacc
				else:
					chkval = 0.05
				
				if self.idir< len(self.tends)-1:
					if np.absolute(times[itmax]-self.tends[self.idir])/self.tends[self.idir] >chkval:
						print('Error: finish time inaccuracy')
						sys.exit()
				else:
					if (self.tends[self.idir]- times[itmax])/self.tends[self.idir] >chkval:
						print('Error: finish time inaccuracy')
						sys.exit()
				
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

				
				"""import matplotlib.pyplot as plt
			
				plt.hist(np.linalg.norm(rs_all_tmp[-1][stinds_nxt], axis=1))
				plt.yscale('log')
				plt.show()"""

				

				

				
				
			
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
			self.rs= rs_all[-1]
			self.vs = vs_all[-1]
			self.t = times[-1]
		else:
			self.t =0.0
		#self.save()

		return self.rs, self.vs, self.ms, self.t, tunits, munits, runits 


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
	

	#basic for now
	def flux_analysis(self, g0=1.6e-3, massrange=[0.05, 2.0], wext=False, reset=False, subset=1000, dt =0.05, mgassume=1e5, recalc=False, inds=None, ext_t=None):

		big_stars = self.bigstars
		LFUV =  self.Lfuv
		EUVcts =  self.euvcts
		#Units in erg/s
		EUVcts = np.array(EUVcts)
		LFUV = np.array(LFUV)

		tunits, munits, runits = self.units

		t = copy.copy(self.t)*tunits*s2myr
		r = copy.copy(self.r)*runits*m2pc
		m = munits*kg2sol*copy.copy(self.m)

		if type(inds)!=type(None):
			subset=len(inds)


		rswitch = np.swapaxes(r,0,1)
		print('Getting flux for each star..')

		if ext_t==None:
			ext_t=0.0

		if type(inds)!=type(None):
			self.photoevap_inds = np.sort(inds)
			self.save()
			reset=True
		elif hasattr(self,'photoevap_inds') and not reset:
			print('Loading subset for flux analysis...')
			self.photoevap_inds = np.sort(np.unique(self.photoevap_inds))
			inds = self.photoevap_inds
		elif subset!=None:
			print('Selecting subset for flux analysis..')
			minds = np.where((m<massrange[1])&(m>massrange[0]))[0]
			if subset<len(minds):
				inds = np.sort(np.random.choice(minds,size=subset, replace=False))
				self.photoevap_inds = inds
				self.save()
			else:
				print('Not enough stars in correct mass range. Selecting all appropriate stars...')
				inds = np.sort(minds)
				self.photoevap_inds = inds
				self.save()
			reset=True
		else:
			reset=True
			print('No subset selected for flux analysis (computing all in mass range)..')
			minds = np.where((m<massrange[1])&(m>massrange[0]))[0]
			inds = np.sort(minds)
			self.photoevap_inds = inds
			self.save()

		inds = self.photoevap_inds
		
		if wext:
			fdir='flux_wext'
			subattr = 'wext'
		else:
			fdir='flux'
			subattr=''


		if not hasattr(self, 'FUV'+subattr) or not hasattr(self, 'EUV'+subattr) or recalc or reset:
			r_phot = rswitch[inds]
			r_big = rswitch[big_stars]	
			r_phot = np.swapaxes(r_phot,0,1)
			r_big = np.swapaxes(r_big,0,1)

			if wext:
				mgas_t, agas_t =  self.magas_tseries()
				mgas_t *= munits*kg2sol
				agas_t *= runits*m2pc
			else:
				mgas_t = np.zeros(len(t))
				agas_t = np.ones(len(t))

			
			flux_wext = np.zeros((len(t),len(m)))
			euv_wext = np.zeros((len(t), len(m)))
			if wext:
				if os.path.isfile(self.out+'_fwext.npy'):
					flux_wext = np.load(self.out+'_fwext.npy')
				if os.path.isfile(self.out+'_euvwext.npy'):
					euv_wext = np.load(self.out+'_euvwext.npy')
			else:
				if os.path.isfile(self.out+'_flux.npy'):
					flux_wext = np.load(self.out+'_flux.npy')
				if os.path.isfile(self.out+'_euv.npy'):
					euv_wext = np.load(self.out+'_euv.npy')

			if type(dt)!=type(None):
				tinds = [0]
				tlast = 0.0
				for it in range(len(t)):
					if t[it]>=tlast+dt:
						tinds.append(it)
						tlast = t[it]
						#elif mgas_t[it]<1.0:
						#tinds.append(it)
					elif it==len(t)-1:
						tinds.append(it)
				tinds_p = tinds+[len(t)-1]
			else:
				tinds = np.arange(len(t))
				tinds_p = tinds
			tinds_p =np.sort(np.array(list(dict.fromkeys(tinds_p))))

			tends_units = np.cumsum(np.array(self.tends))*tunits*s2myr
			tends_zer = np.append(np.array([0.]), tends_units)

			ian = 0
			for it in tinds_p:
				#print('{0}/{1}'.format(ian+1, len(tinds)))
				if (np.count_nonzero(flux_wext[it][inds])<len(inds) or recalc) and  float(mgas_t[it])<mgassume  and t[it]>=ext_t:
					print('Gas mass {0} M_sol at {1} Myr, found {2}/{3} ({4}/{3})- calculating...'.format(mgas_t[it], t[it],np.count_nonzero(flux_wext[it][inds]),len(inds), np.count_nonzero(euv_wext[it][inds])))
					if not recalc:
						doinds = np.where(flux_wext[it][inds]==0.)[0]
					else:
						doinds = np.arange(len(inds))

					
					tmp = cc.flux_wext(r_phot[it][doinds], LFUV, r_big[it],  float(mgas_t[it]),  float(agas_t[it]))
					flux_wext[it][inds[doinds]] = tmp

					"""rtmp = r_phot[it][doinds]
					rtmp2d = np.swapaxes(np.swapaxes(rtmp,0,1)[:2],0,1)
					rmags = np.linalg.norm(rtmp, axis=1)
					rmag2 = np.linalg.norm(rtmp2d, axis=1)
					import matplotlib.pyplot as plt
					print(tmp)
					plt.scatter(rmags, tmp/g0, marker='+', color='b')
					plt.scatter(rmag2, tmp/g0, marker='+', color='r')
					plt.yscale('log')
					plt.xlim([0.,3.0])
					plt.ylim([1e0, 1e5])
					plt.show()

					
					rspace = np.linspace(0., 10.0, 100)
					cumdist_resfix = np.zeros(100)
					for ir in range(len(rspace)):
						cumdist_resfix[ir] = len(np.where(rmag2/(runits*m2pc)<rspace[ir])[0])

					plt.plot(rspace, cumdist_resfix)
					plt.show()"""


					if mgas_t[it]>1e-10 and wext:
						euv_wext[it][inds] = 1e-5
					else:
						tmp_euv = cc.flux_wext(r_phot[it][doinds], EUVcts, r_big[it],  float(0.0),  float(agas_t[it]))
						euv_wext[it][inds[doinds]] = tmp_euv
					
					if wext:
						np.save(self.out+'_fwext', flux_wext)
						np.save(self.out+'_euvwext', euv_wext)
					else:
						np.save(self.out+'_flux', flux_wext)
						np.save(self.out+'_euv', euv_wext)
				elif np.count_nonzero(flux_wext[it][inds])==len(self.photoevap_inds) and not recalc:
					print('Gas mass {0} M_sol at {1} Myr - already calculated.'.format(mgas_t[it], t[it]))
				elif t[it]<ext_t:
					flux_wext[it][inds] = 1e-5
					euv_wext[it][inds] = 1e-5
					print('Time delay for flux {0}/{1}'.format(t[it],ext_t))
				else:
					flux_wext[it][inds] = 1e-5
					euv_wext[it][inds] = 1e-5
					print('Gas mass {0} M_sol at {1} Myr - assuming negligible flux.'.format(mgas_t[it], t[it]))	
				#print('Median flux {0} ({1}):'.format(it, np.count_nonzero(flux_wext[it][inds])), np.median(flux_wext[it][inds]))
				if ian >0:
					flux_0 = flux_wext[tinds_p[ian-1]]
					flux_1 = flux_wext[it]
					euv_0 = euv_wext[tinds_p[ian-1]]
					euv_1 = euv_wext[it]
					DT = t[it]- t[tinds_p[ian-1]]
					if DT>0.:
						grad = (flux_1-flux_0)/DT 
						gradeuv = (euv_1-euv_0)/DT 
					elif DT<0.:
						print('Flux calc. error: time-step reversal')
						sys.exit()
					else:
						print('Grad set to zero: dt = ', DT)
						grad = np.zeros(self.n)
						gredeuv =np.zeros(self.n)
					for itt in range(tinds_p[ian-1]+1, it):
						delta_t = t[itt]- t[tinds_p[ian-1]]
						if recalc:
							flux_wext[itt][inds] = flux_0[inds]+grad[inds]*delta_t
							euv_wext[itt][inds] = euv_0[inds]+gradeuv[inds]*delta_t
						elif np.count_nonzero(flux_wext[itt][inds])<len(inds):
							flux_wext[itt][inds]=flux_0[inds]+grad[inds]*delta_t
							euv_wext[itt][inds]=euv_0[inds]+gradeuv[inds]*delta_t
						#print('Median flux {0}:'.format(itt),  np.median(flux_wext[itt][inds]))
						#tmp = flux_wext[itt][np.where(flux_wext[itt]!=0.0)[0]]

				if type(self.starinds)!=type(None):
					
					idir=0
					for idd in range(len(tends_units)):
						if t[it]>=tends_zer[idd]:
							idir=min(idd, len(tends_units)-1)				
					print('Time: {0}, Idir: {1}'.format(t[it], idir))

					#Take subset of inds which are not in the current pop, and set flux to zero
					nstinds =  inds[~np.in1d(inds, self.starinds[idir])]
					flux_wext[it][nstinds] =1e-5
					euv_wext[it][nstinds] = 1e-5
					
						
			
				ian+=1
	
			if wext:
				np.save(self.out+'_fwext', flux_wext)
				np.save(self.out+'_euvwext', euv_wext)
			else:
				np.save(self.out+'_flux', flux_wext)
				np.save(self.out+'_euv', euv_wext)

			#flux_wext = cc.flux_wext_tseries(r_phot, LFUV, r_big,  mgas_t,  agas_t)
			fuv_flux_all = np.swapaxes(flux_wext/g0, 0,1)
			euv_counts_all= np.swapaxes(euv_wext, 0,1)
			if wext:
				self.FUVwext = fuv_flux_all
				self.EUVwext = euv_counts_all
			else:
				self.FUV =  fuv_flux_all
				self.EUV = euv_counts_all

			self.save()

		
		return None

	#groups[igr], m,FUV_arr, t_arr, minit, rinit0, rscale, alpha,tinit_arr,t0, igr, seed_arr[igr], addname, wext
	def thread_photoevap(self,grp, mstars, FUV_arr, EUV_arr, t_arr,  minit, rinit0, rscale, alpha, tinits, t0, igr, seed, addname, wext, recalc):
		if not os.path.isdir('proc{0}'.format(igr)):
			os.mkdir('proc{0}'.format(igr))
		os.chdir('proc{0}'.format(igr))

		
		icalc=0
		print('Random number seed for proc {0} = {1}'.format(igr, seed))
		np.random.seed(seed)
		mfrac_5e3 = []
		mfrac_1e3 = []
		mfrac_5e2 = []
		mfrac_1e2 = []
		tsurv_5e3 = []
		tsurv_1e3 = []
		tsurv_5e2 = []
		tsurv_1e2 = []


		if type(tinits)==type(None):
			tinits = np.zeros(len(mstars))


		for istar in grp:
			
			outname_r = self.out+'_photrdisc'+addname+'_{0}'.format(istar)
			outname_m = self.out+'_photmdisc'+addname+'_{0}'.format(istar)
			outname_dmacc =  self.out+'_photdmacc'+addname+'_{0}'.format(istar)

			
			if (not os.path.isfile('../'+outname_r+'.npy') or not os.path.isfile('../'+outname_m+'.npy') or not os.path.isfile('../'+outname_dmacc+'.npy')) or recalc:
				#print('Photoevaporation calculation for {0}...'.format(istar))
				if type(minit)==float:
					m0 = minit
				elif minit=='auto':
					m0 = 0.1*(mstars[istar])
				elif minit=='distfix':
					mu = -3.25
					sigma = 0.7
					m0 = np.exp(mu+sigma*np.random.normal())
				elif minit=='dist':
					factor = 0.1+0.9*np.random.rand()
					m0 = mstars[istar]*0.1*factor
				else:
					print('Error: initial mass setting not recognised.')	
					exit()

				if rscale=='auto':
					r0 = np.sqrt(mstars[istar])*(rinit0/2.5)
					rinit = r0*2.5
				elif rscale=='fix':
					r0 = rinit0/2.5
					rinit=  rinit0
				elif rscale=='distfix':
					rinit = rinit0 +100.*np.random.rand()
					r0= rinit/2.5
					#rinit = rinit0/1.4+(1.4*rinit0-(rinit0/1.4))*np.random.rand()
					#r0 = rinit0/2.5
				elif rscale=='dist':
					rinit = mstars[istar]*(rinit0/5.+(2.*rinit0-(rinit0/5.))*np.random.rand())
					r0 = rinit0/2.5
				elif type(rscale)==float:
					r0=rscale
					rinit = rinit0
				elif rscale=='mdep':
					r0 = np.sqrt(m0/0.1)*(rinit0/2.5)
					rinit = r0*2.5
				else:
					print('Error: radius scale setting not recognised')
					exit()

				print('Mstar = {0:.2f}: R0 = {1:.2f}, Rout = {2:.2f}, M_disc = {3:.2e}'.format(mstars[istar], r0, rinit, m0))


				if np.count_nonzero(FUV_arr[istar])<=1:
					print('G0s not detected for {0}'.format(istar))
					exit()

				FUVtmp = copy.copy(FUV_arr[istar])	
				EUVtmp = copy.copy(EUV_arr[istar])
				ttmp = copy.copy(t_arr)
				istart  = np.argmin(np.absolute(ttmp-tinits[istar]))
				ttmp = ttmp[istart:]
				tinittmp = ttmp[0]
				ttmp -= tinittmp
				FUVtmp = FUVtmp[istart:]
				EUVtmp = EUVtmp[istart:]
				
				g0arr = np.array([ttmp,FUVtmp])
				euvarr = np.array([ttmp, EUVtmp])

				disc = vdclass.viscous_disc(mstar=mstars[istar], iswitch=0, amdin=m0,  oname='pdisc_{0}'.format(istar), restart=False, nsave=100, mmin=5e-6, g0ts=g0arr, tend=ttmp[-1], alpha=alpha, scaleparam=r0, redgeinit=rinit, euvts=euvarr)
				
				t = t0+tinittmp+disc.grid_loader('tw')/1e6
				rout = disc.grid_loader('redge')
				mout = disc.grid_loader('amtotw')
				mdotacc = disc.grid_loader('amlossviw')
				
				if t0+tinittmp>1e-10:
					t = np.append(np.array([0.]), t)
					rout = np.append(np.array([rout[0]]), rout)
					mout = np.append(np.array([mout[0]]), mout)
					mdotacc = np.append(np.array([mdotacc[0]]), mdotacc)
				
				mout_surv = np.where(mout>1e-5)[0]
				tdisp = t[mout_surv[-1]]
				
				if wext:
					#print('{0} ({4} - G0={5}): {1} {2} {3}'.format(istar, t[-1], rout[-1], np.log10(mout[-1]), m[istar], self.FUVwext[istar][-1]))
					dsurv=1.							
					if mout[-1]<1e-5:
						dsurv=0.

					if  self.FUVwext[istar][-1]>5e3:
						mfrac_5e3.append(dsurv)
						tsurv_5e3.append(tdisp)
					elif  self.FUVwext[istar][-1]>1e3:
						mfrac_1e3.append(dsurv)
						tsurv_1e3.append(tdisp)
					elif  self.FUVwext[istar][-1]>5e2:
						mfrac_5e2.append(dsurv)
						tsurv_5e2.append(tdisp)
					elif  self.FUVwext[istar][-1]>1e2:
						mfrac_1e2.append(dsurv)
						tsurv_1e2.append(tdisp)
					
				else:
					#print('{0} (mstar = {4} - G0={5}): tend={1}, rend= {2}, log(mend)= {3}'.format(istar, t[-1], rout[-1], np.log10(mout[-1]), m[istar], self.FUV[istar][-1]))
					dsurv=1.							
					if mout[-1]<1e-5:
						dsurv = 0.
					
					if  self.FUV[istar][-1]>5e3:
						mfrac_5e3.append(dsurv)
						tsurv_5e3.append(tdisp)
					elif  self.FUV[istar][-1]>1e3:
						mfrac_1e3.append(dsurv)
						tsurv_1e3.append(tdisp)
					elif  self.FUV[istar][-1]>5e2:
						mfrac_5e2.append(dsurv)
						tsurv_5e2.append(tdisp)
					elif  self.FUV[istar][-1]>1e2:
						mfrac_1e2.append(dsurv)
						tsurv_1e2.append(tdisp)
				sample=0	
				if len(mfrac_5e3)>0:
					mm1 = np.mean(np.array(mfrac_5e3))
					mt1 = np.mean(np.array(tsurv_5e3))
					sample+=len(mfrac_5e3)
				else:
					mm1=-1
					mt1=-1
				if len(mfrac_1e3)>0:
					mm2 = np.mean(np.array(mfrac_1e3))
					mt2 = np.mean(np.array(tsurv_1e3))
					sample+=len(mfrac_1e3)
				else:
					mm2=-1
					mt2=-1
				if len(mfrac_5e2)>0:
					mm3 = np.mean(np.array(mfrac_5e2))
					mt3 = np.mean(np.array(tsurv_5e2))
					sample+=len(mfrac_5e2)
				else:
					mm3=-1
					mt3=-1
				if len(mfrac_1e2)>0:
					mm4 = np.mean(np.array(mfrac_1e2))
					mt4 = np.mean(np.array(tsurv_1e2))
					sample+=len(mfrac_1e2)
				else:
					mm4=-1
					mt4=-1
					
				
				#print('****Fractions (%d) - 5e3: %.2lf, 1e3: %.2lf, 5e2: %.2lf, 1e2: %.2lf****'%(sample, mm1,mm2,mm3, mm4))
				print('****Dispersal times - 5e3: %.2lf, 1e3: %.2lf, 5e2: %.2lf, 1e2: %.2lf****'%(mt1,mt2,mt3, mt4))


				np.save('../'+outname_r, np.array([t, rout]))
				np.save('../'+outname_m, np.array([t, mout]))
				np.save('../'+outname_dmacc, np.array([t, mdotacc]))

				disc.clean()
				icalc+=1
		
		os.chdir('..')
		shutil.rmtree('proc{0}'.format(igr)) 


		return None


	def photoevap_analysis(self, rinit0=100.0, minit='auto',rscale='fix', nprocs=NPROCS, rmax=15.0, tstart=0.0, wext=False, recalc=False, alpha=5e-3):
		

		tunits, munits, runits = self.units
		tarch = copy.copy(self.t)
		r = copy.copy(self.r)
		v =copy.copy(self.v)
		m = copy.copy(self.m)


		tarch *= tunits*s2myr
		r *= runits*m2pc
		m *= munits*kg2sol

		if hasattr(self,'photoevap_inds'):
			print('Loading subset for photoevaporation analysis...')
			self.photoevap_inds = np.sort(self.photoevap_inds)
			inds = self.photoevap_inds
			#minds = np.where((m[inds]>massrange[0])&(m[inds]<massrange[1]))[0]
			
			"""if len(inds)!=len(minds):
				print('Stars outside mass range detected. Cutting sample...')
				self.photoevap_inds = np.sort(inds[minds])
				inds = np.sort(inds[minds])
				self.save()"""
		else:
			print('Error: no photoevaporation indices defined for analysis.')
			exit()
		
		if wext:
			addname = '_wext'
		else:
			addname=''
		
		if recalc:
			incomplete_list = inds
		elif hasattr(self, 'phot'+addname+'_m'):
			if wext:
				dm0, dm1 =np.swapaxes(self.phot_wext_m[inds],0,1)[[0,-1]]
			else:
				dm0, dm1 =np.swapaxes(self.phot_m[inds],0,1)[[0,-1]]
			
			i_incomp = np.where(dm0-dm1<1e-15)[0]
			
			incomplete_list = inds[i_incomp]
		else:
			incomplete_list = inds
		incomplete_list=  inds

		self.incomp_phot_inds = incomplete_list
		
		
		print('Photoevaporation calculation incomplete for {0}/{1}'.format(len(incomplete_list), len(inds)))

		


		"""for ibig in big_stars:
			index = np.argwhere(restrict==ibig)
			restrict = np.delete(restrict, index)	"""


		rswitch = np.swapaxes(r,0,1)


		print('Beginning photoevaporation analysis...')
	
		wdir = os.getcwd()

		if wext:
			pdir = 'photoevap_wext'
		else:
			pdir = 'photoevap'

		if not os.path.isdir(pdir):
			os.makedirs(pdir)
		os.chdir(pdir)

		if len(self.incomp_phot_inds)>0 or recalc:
			
			inlist = []
			for istar in inds:
				if not wext:
					outname_r = self.out+'_photrdisc_{0}'.format(istar)
					outname_m = self.out+'_photmdisc_{0}'.format(istar)
					outname_dmacc = self.out+'_photdmacc_{0}'.format(istar)
				else:
					outname_r = self.out+'_photrdisc_wext_{0}'.format(istar)
					outname_m = self.out+'_photmdisc_wext_{0}'.format(istar)
					outname_dmacc = self.out+'_photdmacc_wext_{0}'.format(istar)
				if (not os.path.isfile(outname_r+'.npy') or not os.path.isfile(outname_m+'.npy') or not os.path.isfile(outname_dmacc+'.npy')) or recalc:
					inlist.append(istar)

			print('Calculations running for {0}/{1}'.format(len(inlist), len(inds)))
			
				
		
			inlist = np.array(inlist, dtype='int')
		
			t_arr = tarch
			itmin = np.argmin(np.absolute(t_arr-tstart))
			t0 = t_arr[itmin]
			t_arr = t_arr[itmin:] - t_arr[itmin]
			
			
			tinit_arr = np.zeros(self.n)
			tends_units = np.cumsum(np.array(self.tends))*s2myr*tunits
			tends_zer = np.append(np.array([0.]), tends_units)
			if type(self.starinds)!=type(None):
				#If  pops of stars, then set start time for disc evol calcs
				all_inds = inds
				
				oldinds = np.array([])
				for idir in range(len(tends_units)):
						#First time stars are present?
						popinds = self.starinds[idir]
						newinds = popinds[~np.in1d(popinds, oldinds)]

						tinit_arr[newinds] = tends_zer[idir]
						oldinds = popinds

			
			seed_arr = np.random.randint(0, high=int(1e4), size=nprocs)


			if wext:
				FUV_arr = self.FUVwext
				EUV_arr = self.EUVwext
			else:
				FUV_arr  = self.FUV
				EUV_arr = self.EUV

			EUV_arr = EUV_arr[:][itmin:]
			FUV_arr = FUV_arr[:][itmin:] # np.swapaxes(np.swapaxes(FUV_arr,0,1)[itmin:],0,1)


			ngroups = nprocs
			groups = np.array_split(inlist, ngroups)
			igr= 0
			procs = []
			if len(inlist)>0:
				while (igr<ngroups) or (len(procs)>0):
					if len(procs) < nprocs and igr<ngroups:
						print('Running for processor {0}:'.format(igr), len(groups[igr]))
						# thread_photoevap(self,grp, mstars, FUV_arr, EUV_arr, t_arr,  minit, rinit0, rscale, alpha, tinits, t0, igr, seed, addname, wext, recalc)
						procs.append(Process(target=self.thread_photoevap, args=(groups[igr], m,FUV_arr,EUV_arr, t_arr, minit, rinit0, rscale, alpha,tinit_arr,t0, igr, seed_arr[igr], addname, wext, recalc)))
						procs[-1].start()
						igr+=1

					tmp = []
					# Check for finished jobs:
					for sim in procs:
						if sim.exitcode is not None:
							sim.join()
						else:	
							tmp.append(sim)
					procs = tmp

		else:
			print('Disc photoevaporation results found...')

		os.chdir(wdir)

		if len(incomplete_list)>0 or recalc:
			print('Assigning radius/mass evolution...')

			rout_all = -1.*np.ones((len(m),len(tarch)))
			mdisc_all = -1.*np.ones((len(m),len(tarch)))
			dmadisc_all =  -1.*np.ones((len(m),len(tarch)))
			icount = 0
		
			outnames_r = glob.glob(pdir+'/'+self.out+'_photrdisc'+addname+'_*')
			outnames_m = glob.glob(pdir+'/'+self.out+'_photmdisc'+addname+'_*')
			outnames_dmacc= glob.glob(pdir+'/'+self.out+'_photdmacc'+addname+'_*')
		
			for istar in self.photoevap_inds:
				fflg = True
				for outname in outnames_r:
					if self.out+'_photrdisc'+addname+'_{0}'.format(istar) in outname:
						data_r = np.load(outname)
						fflg=False
						break

				if fflg:
					print('Error: could not find radial evolution of disc {0}'.format(istar))
					#print('Fnames: ', outnames_r)
					exit()

				fflg = True
				for outname in outnames_m:
					if self.out+'_photmdisc'+addname+'_{0}'.format(istar) in outname:
						data_m = np.load(outname)
						fflg=False
						break

				if fflg:
					print('Error: could not find mass evolution of disc {0}'.format(istar))
					#print('Fnames: ', outnames_m)
					exit()

				fflg = True
				for outname in outnames_dmacc:
					if self.out+'_photdmacc'+addname+'_{0}'.format(istar) in outname:
						data_dmacc = np.load(outname)
						fflg=False
						break

				if fflg:
					print('Error: could not find radial evolution of disc {0}'.format(istar))
					#print('Fnames: ', outnames_r)
					exit()
			
				t = data_r[0]
				rout = data_r[1]
		

				inext= 0
				tnext = t[inext]

				for it in range(len(tarch)):
					if inext>=len(t):
						break
					while tarch[it]>=tnext:
						rout_all[istar][it:] = rout[inext]
						inext+=1
						if inext>=len(t):
							break
						else:
							tnext=t[inext]


			
				t = data_m[0]
				mdisc = data_m[1]
		

				inext= 0
				tnext = t[inext]

				for it in range(len(tarch)):
					if inext>=len(t):
						break
					while tarch[it]>=tnext:
						mdisc_all[istar][it:] = mdisc[inext]
						inext+=1
						if inext>=len(t):
							break
						else:
							tnext=t[inext]

				t = data_dmacc[0]
				dmacc = data_dmacc[1]

				inext= 0
				tnext = t[inext]

				for it in range(len(tarch)):
					if inext>=len(t):
						break
					while tarch[it]>=tnext:
						dmadisc_all[istar][it:] = dmacc[inext]
						inext+=1
						if inext>=len(t):
							break
						else:
							tnext=t[inext]

				#plt.plot(tarch, rout_all[icount])
				#plt.show()

				if icount%100==0:
					print('Complete for {0}'.format(icount))
				icount+=1

			if not wext:
				self.phot_r = rout_all
				self.phot_m = mdisc_all
				self.phot_dmacc = dmadisc_all
			else:
				self.phot_wext_r = rout_all
				self.phot_wext_m = mdisc_all
				self.phot_wext_dmacc = dmadisc_all
				
			np.save('rtmp'+addname, rout_all)
			np.save('mtmp'+addname, mdisc_all)
			np.save('dmatmp'+addname, dmadisc_all)
			self.save()
		elif os.path.isfile('rtmp'+addname+'.npy') and os.path.isfile('mtmp'+addname+'.npy')\
			 and (not hasattr(self, 'phot'+addname+'_r') or not hasattr(self, 'phot'+addname+'_m')):

			if wext:
				self.phot_wext_r = np.load('rtmp'+addname+'.npy')
				self.phot_wext_m = np.load('mtmp'+addname+'.npy')
				self.phot_wext_dmacc = np.load('dmatmp'+addname+'.npy')
			else:
				self.phot_r = np.load('rtmp.npy')
				self.phot_m = np.load('mtmp.npy')
				self.phot_dmacc = np.load('dmatmp.npy')
			self.save()

		else:
			print('Disc property calculations already complete.')

		


		return None
		
	def get_lums(self):
		if not hasattr(self, 'bigstars') or not hasattr(self, 'Lfuv') or not hasattr(self, 'euvcts'):
			print('Getting stellar luminosities..')
			m = copy.copy(self.m)

			tunits, munits, runits = self.units
			euvdat = np.load(photoevapdir+'/EUV_counts.npy')
			fuvdat = np.load(photoevapdir+'/FUV_lum.npy')
			m*=munits*kg2sol

			mspaceeuv = euvdat[0]
			mspacefuv = fuvdat[0]
			euv = euvdat[1]
			fuv = fuvdat[1]

			thresh = min(np.amin(mspaceeuv), np.amin(mspacefuv))
			euvfunc = interpolate.interp1d(mspaceeuv, euv)
			fuvfunc = interpolate.interp1d(mspacefuv, fuv) 

			indices = np.where(m>thresh)[0]
			toobig = np.where(m>99.99)[0]
			mtmp = copy.copy(m)
			mtmp[toobig]  = 99.99
			fuvlum = fuvfunc(mtmp[indices])
			euvlum = euvfunc(mtmp[indices])

			self.bigstars = indices
			self.Lfuv= fuvlum
			self.euvcts = euvlum
			self.save()
		return self.bigstars, self.Lfuv, self.euvcts

	
	def combine(self):
		if not self.complete:
			if not hasattr(self, 'r') or not hasattr(self, 'v') or not hasattr(self, 't'):
				idir = 0
				for idir in range(len(self.dirs)):
					d = self.dirs[idir]
					if os.path.isdir(d):
						if idir==0:
							t = np.load(d+'/'+self.out+'_t.npy')
							r = np.load(d+'/'+self.out+'_r.npy')
							v = np.load(d+'/'+self.out+'_v.npy')
							tunits, munits, runits = np.load(self.out+'_units.npy')
						else:
							r = np.append(r, np.load(d+'/'+self.out+'_r.npy'), axis=0)
							v = np.append(v, np.load(d+'/'+self.out+'_v.npy'), axis=0)
							ttmp =np.load(d+'/'+self.out+'_t.npy')
							ttmp += t[-1]
							t = np.append(t, ttmp, axis=0)

						print('Shape r:', r.shape)
					else:
						print('Error: "{0}" not found.'.format(d))
			
				np.save(self.out+'_t', t)
				np.save(self.out+'_r', r)
				np.save(self.out+'_v', v)
				np.save(self.out+'_units', np.array([tunits, munits, runits]))

				self.r = r 
				self.v = v
				self.t = t
				self.m = self.ms
				self.units = np.array([tunits, munits, runits])
				self.save()
		
		return None

	
	def run_nbody(self):
		homedir = os.getcwd()

		if not self.complete:
	
			for idir in range(len(self.dirs)):
				self.idir=idir

				os.chdir(homedir)
				if not os.path.isdir(self.dirs[idir]):
					os.makedirs(self.dirs[idir])
				os.chdir(self.dirs[idir])
				print('Current directory:', os.getcwd())
			
				if not os.path.isfile(self.out+'.input'):
					self.write_to_input()
				else:
					print('Input file detected.')
					
				RUN_STR =  NBODYDIR + "nbody6++.avx < {0} 2>&1 {1}".format(self.out+'.input', self.out+'.output')
				if not os.path.isfile(self.out+'.output'):
					print(RUN_STR)
					command = cclass.Command(RUN_STR)
					command.run(timeout=20000)
				else:
					print('Output file detected.')	
				if hasattr(self, 'tends'):
					ttmp = 0.0
					iatt=0
					exit_flag=False
					while (self.tends[idir]-ttmp)/self.tends[idir] > self.dtjacc and iatt<3:			
						rtmp, vtmp, mtmp, ttmp, tunits, munits, runits = self.read_to_npy(force=False, checkT=False)

						if (self.tends[idir]-ttmp)/self.tends[idir] > self.dtjacc and iatt==0:
							rtmp, vtmp, mtmp, ttmp, tunits, munits, runits = self.read_to_npy(force=True, checkT=False)
				
						if (self.tends[idir]-ttmp)/self.tends[idir] > self.dtjacc:
							print('Simulation ended early for {0}. Restarting ({1})...'.format(self.dirs[idir], iatt))
							print(self.tends, ttmp)
							print('T_end = {0}/{1}'.format(ttmp, self.tends[idir]))
							inname = self.write_to_input(restart=0)
							RUN_STR_NEW =  NBODYDIR + "nbody6++.avx < {0} >& {1}".format(inname+'.input', inname+'.output')
							print(RUN_STR_NEW)
							command = cclass.Command(RUN_STR_NEW)
							command.run(timeout=20000)
							rtmp, vtmp, mtmp, ttmp, tunits, munits, runits = self.read_to_npy(force=True, checkT=False)
						

						"""ir=0
						rtmpt=rtmp[self.starinds[idir+1]]
						vtmpt=vtmp[self.starinds[idir+1]]
						mtmpt = mtmp[self.starinds[idir+1]]
						itmpt = self.starinds[idir+1]
						print(idir, rtmpt.shape)
						for rval in rtmpt:
							if ir>0 and ir<len(rtmp)-1:
								rtmp_trunc = np.append(rtmpt[:ir], rtmpt[ir+1:], axis=0)
								vtmp_trunc = np.append(vtmpt[:ir], vtmpt[ir+1:], axis=0)
								mtmp_trunc = np.append(mtmpt[:ir], mtmpt[ir+1:], axis=0)
								itmp_trunc =  np.append(itmpt[:ir], itmpt[ir+1:], axis=0)
							elif ir==0:
								rtmp_trunc =rtmpt[ir+1:]
								mtmp_trunc =mtmpt[ir+1:]
								vtmp_trunc =vtmpt[ir+1:]
								itmp_trunc =itmpt[ir+1:]
							else:
								rtmp_trunc =rtmpt[:ir]
								mtmp_trunc =mtmpt[:ir]
								vtmp_trunc =vtmpt[:ir]
								itmp_trunc =itmpt[:ir]
							drvals = np.linalg.norm(rtmp_trunc-rval,axis=1)
							
							if np.amin(drvals)<1e-8:
								imin = np.argmin(drvals)
								print(drvals[imin])
								print(itmp_trunc[imin], itmpt[ir])
								print(rtmp_trunc[imin], rtmpt[ir])
								print(mtmp_trunc[imin]/np.mean(mtmpt), mtmpt[ir]/np.mean(mtmpt))
								print(vtmp_trunc[imin], vtmpt[ir])
								test_arr = np.array([itmp_trunc[imin],itmpt[ir]])
								for iinds in self.starinds:
									print(np.in1d(test_arr, iinds))
									print(np.count_nonzero(iinds == itmp_trunc[imin])
								exit_flag=True
							ir+=1
						#if exit_flag:
						#	exit()
						"""
						iatt+=1
					if (self.tends[idir]-ttmp)/self.tends[idir] > self.dtjacc and iatt>=3:
						print('Error: Failure to run for {0} after {1} attempts...'.format(self.dirs[idir], iatt))
						sys.exit()
		os.chdir(homedir)
		
	def evolve(self, photo=False,tstart=0.,wext=False,  reset=False, recalc_flux=False, recalc_phot=False, subset=1000, inds=None, ext_t =None, alpha=1e-2, minit='dist', rscale='fix', rinit0=100.0):
		self.run_nbody()
		self.combine()

		if photo:
			self.get_lums()	
			self.flux_analysis(wext=wext, subset=subset, reset=reset, recalc=recalc_flux, inds=inds, ext_t=ext_t)
			self.photoevap_analysis(rinit0=rinit0, tstart=tstart, minit=minit, rscale=rscale, recalc=recalc_phot, wext=wext, alpha=alpha)

		
