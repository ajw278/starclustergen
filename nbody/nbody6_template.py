from __future__ import print_function

from datetime import datetime
import numpy as np
import sys
import os

def infile_string_r3(indict, mpi=False, KZ_change=None):

	if not mpi:
		STRING = "{0} {1} {2} 40 40 640\n".format(indict['KSTART'], indict['TCOMP'], indict['TCRITP'])
	else:
		STRING = "{0} {1} {2} {3} {4} {5}\n".format(indict['KSTART'], indict['TCOMP'], indict['TCRITP'], indict['isernb'],indict['iserreg'],indict['iserks'])

	STRING += "{0} {1} {2} {3} {4} {5}\n".format(indict['DTADJ'], indict['DELTAT'], 0, 0,indict['TCRIT'],indict['QE'])

	if KZ_change==None:
		STRING += "{0} {1} \n".format(0, indict['KZ'][0])
	else:
		for KZ in KZ_change:
			STRING += "{0} {1} \n".format(KZ[0], KZ[1])
			
	return STRING


def infile_string_r4(indict, mpi=False):

	if not mpi:
		STRING = "{0} {1} {2} 40 40 640\n".format(indict['KSTART'], indict['TCOMP'], indict['TCRITP'])
	else:
		STRING = "{0} {1} {2} {3} {4} {5}\n".format(indict['KSTART'], indict['TCOMP'], indict['TCRITP'], indict['isernb'],indict['iserreg'],indict['iserks'])

	

	STRING += "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(indict['ETAI'], indict['ETAR'], indict['ETAU'], indict['DTMIN'],indict['RMIN'],indict['NCRIT'], indict['NNBOPT'], indict['SMAX'])

	return STRING

def infile_string_r5(indict, mpi=False, KZ_change=None):

	if not mpi:
		STRING = "{0} {1} {2} 40 40 640\n".format(indict['KSTART'], indict['TCOMP'], indict['TCRITP'])
	else:
		STRING = "{0} {1} {2} {3} {4} {5}\n".format(indict['KSTART'], indict['TCOMP'], indict['TCRITP'], indict['isernb'],indict['iserreg'],indict['iserks'])
	
	STRING += "{0} {1} {2} {3} {4} {5}\n".format(indict['DTADJ'], indict['DELTAT'], 0.0, 0.0,indict['TCRIT'],indict['QE'])

	
	if KZ_change==None:
		STRING += "{0} {1} \n".format(0, indict['KZ'][0])
	else:
		for KZ in KZ_change:
			STRING += "{0} {1} \n".format(KZ[0], KZ[1])


	STRING += "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(indict['ETAI'], indict['ETAR'], indict['ETAU'], indict['DTMIN'],indict['RMIN'],indict['NCRIT'], indict['NNBOPT'], indict['SMAX'])

	return STRING

def infile_string(indict, masses, pos, vel,  mpi=False):

	if not mpi:
		STRING = "{0} {1} {2} 40 40 640\n".format(indict['KSTART'], indict['TCOMP'], indict['TCRITP'])
	else:
		STRING = "{0} {1} {2} {3} {4} {5}\n".format(indict['KSTART'], indict['TCOMP'], indict['TCRITP'], indict['isernb'],indict['iserreg'],indict['iserks'])

	NRAND = np.random.randint(0, 100000)
	RBAR = 0
	ZMBAR = 0

	STRING += "{0} {1} {2} {3} {4} {5} {6}\n".format(indict['N'], indict['NFIX'], indict['NCRIT'], NRAND, indict['NNBOPT'], indict['NRUN'], indict['NCOMM'])

	STRING += "{0} {1} {2} {3} {4} {5} {6} {7} {8}\n".format(indict['ETAI'], indict['ETAR'], indict['RS0'],indict['DTADJ'], indict['DELTAT'],indict['TCRIT'], indict['QE'], indict['RBAR'], indict['ZMBAR'])

	STRING += "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(indict['KZ'][0],indict['KZ'][1],indict['KZ'][2],indict['KZ'][3], indict['KZ'][4], indict['KZ'][5],indict['KZ'][6], indict['KZ'][7], indict['KZ'][8], indict['KZ'][9])
	#0 1 1 0 1 0 4 0 0 2
	
	STRING += "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(indict['KZ'][10],indict['KZ'][11],indict['KZ'][12],indict['KZ'][13], indict['KZ'][14], indict['KZ'][15],indict['KZ'][16], indict['KZ'][17], indict['KZ'][18], indict['KZ'][19])
	#0 1 0 1 2 0 0 0 3 6
	STRING += "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(indict['KZ'][20],indict['KZ'][21],indict['KZ'][22],indict['KZ'][23], indict['KZ'][24], indict['KZ'][25],indict['KZ'][26], indict['KZ'][27], indict['KZ'][28], indict['KZ'][29])
	#1 0 2 0 0 2 0 0 0 2

	STRING += "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(indict['KZ'][30],indict['KZ'][31],indict['KZ'][32],indict['KZ'][33], indict['KZ'][34], indict['KZ'][35],indict['KZ'][36], indict['KZ'][37], indict['KZ'][38], indict['KZ'][39])
	#1 0 2 1 1 0 1 1 0 0
	
	STRING += "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(indict['KZ'][40],indict['KZ'][41],indict['KZ'][42],indict['KZ'][43], indict['KZ'][44], indict['KZ'][45],indict['KZ'][46], indict['KZ'][47], indict['KZ'][48], indict['KZ'][49])
	#0 0 0 0 0 0 0 0 0 0

	#SKIPPING: setup.F
	"""if indict['KZ'][4] !=0:
		print('Error: Not implemented for setup (5) params.')
		sys.exit()"""
	
	STRING += "{0} {1} {2} {3} {4} {5} {6}\n".format(indict['DTMIN'], indict['RMIN'], indict['ETAU'], indict['ECLOSE'], indict['GMIN'], indict['GMAX'], indict['SMAX'])

	if indict['KZ'][21]!=2 and indict['KZ'][21]!=6 and indict['KZ'][21]!=10:
		print('Error: Not implemented for setup (22) params.')
		sys.exit()
	else:
		PSTRING = particle_string(masses, pos, vel)
		if not os.path.isfile('dat.10'):
			with open('dat.10', "w") as f:
				f.write(PSTRING)
	
	STRING += "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(indict['ALPHA'], indict['BODY1'], indict['BODYN'], indict['NBIN0'], indict['NHI0'], indict['ZMET'], indict['EPOCH0'], indict['DTPLOT'])
	
	STRING += "{0} {1} {2} {3}\n".format(indict['Q'], indict['VXROT'], indict['VZROT'], indict['RTIDE'])
	

	if indict['KZ'][13]==4:
		STRING += "%.2e %.2lf %.5e %.2lf\n"%(indict['MP'], indict['AP'], indict['MPDOT'], indict['TDELAY'])	

	if indict['KZ'][28]>0:
		STRING += "{0}".format(indict['SIGMA0'])	

	"""if not indict['KZ'][13] in [0,1]:
		print('Error: Not implemented for tidal params.')
		sys.exit()"""
		

	#if indict['KZ'][7] != 0:
	#	print('Error: Not implemented for binary params.')
	#	sys.exit()

	if indict['KZ'][17] != 0:
		print('Error: Not implemented for hierarchical params.')
		sys.exit()

	if indict['KZ'][23] != 0:
		print('Error: Not implemented for black hole params.')
		sys.exit()

	if indict['KZ'][12] != 0:
		print('Error: Not implemented for cloud params.')
		sys.exit()


	return STRING


def particle_string(masses, pos, vel):
	STRING = ""

	for im in range(len(masses)):
		STRING += "{0} {1} {2} {3} {4} {5} {6}\n".format(masses[im], pos[im][0], pos[im][1], pos[im][2], vel[im][0], vel[im][1], vel[im][2])

	return STRING

