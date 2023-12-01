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

import cluster_calcs as cc





def get_nbody_units(ms_Msol, rs_pc, vs_kms):

	#Convert everything to SI
	
	ms = ms_Msol/kg2sol
	rs = rs_pc/m2pc
	vs = vs_kms*1e3
	
	#M_units in SI is easy
	m_units = np.sum(ms)
	
	#Now compute potential with G=1:
	potG1 = cc.stellar_potential(rs, ms)
	
	#Multiply by G in SI
	pot = potG1*G_si
	
	KE= cc.total_kinetic(vs, ms)
	
	#Now compute velocity units to give E=1/4
	v_units = np.sqrt((4./m_units)*(KE - np.absolute(pot)))
	
	
	r_units = G_si*m_units/v_units 
	
	t_units = 1./np.sqrt(G_si*m_units/np.power(r_units, 3))

	return rs/r_units, vs/v_units, ms/m_units, r_units, t_units, m_units