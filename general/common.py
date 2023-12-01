import os
import sys
import numpy as np


scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(scriptdir)
split = scriptdir.split('/')
split[0]  = split[0][1:]
split = split[1:]
rootdir = ''
for isplit in range(len(split)-1):
	rootdir+='/'+split[isplit]
photoevapdir = rootdir+'/photoevap'
nbodydir = rootdir+'/nbody'
plotdir = rootdir+'/plot'



sys.path.append(rootdir)
sys.path.append(photoevapdir)
sys.path.append(nbodydir)
sys.path.append(plotdir)

sys.path =list(dict.fromkeys(sys.path))

import pyximport; pyximport.install(setup_args={'include_dirs':[np.get_include()]})
#import cluster_calcs as cc
sys.path.append(scriptdir+'/nbody')


exepath = 'nbody_path.txt'
NBODYEXE=None
if os.path.isfile(exepath):
    with open(exepath) as f:
        lines = f.readlines()
        for line in lines:
            if len(line)>2:
                NBODYEXE = line.strip('\n')
                break
if NBODYEXE is None:
    print('No file (%s) with location of Nbody6++ found...'%exepath)
    NBODYEXE = '/usr/local/bin/nbody6++'
    print('Assuming: %s'%NBODYEXE)
else:
    print('Nbody6++ executable location:', NBODYEXE)

NPROCS=8
SYSPATH =sys.path


G_si = 6.67e-11
s2myr = 3.17098e-8*1e-6
m2au = 6.68459e-12
m2pc = 3.24078e-17
kg2sol = 1./2e30
