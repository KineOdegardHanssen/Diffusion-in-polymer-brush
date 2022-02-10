import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
import numpy as np
import random
import math
import time
import os
import glob

def rmsd(x,y):
    Nx = len(x)
    Ny = len(y)
    if Nx!=Ny:
        print('WARNING! Nx!=Ny. Could not calculate rmsd value')
        return 'WARNING! Nx!=Ny. Could not calculate rmsd value'
    delta = 0
    for i in range(Nx):
        delta += (x[i]-y[i])*(x[i]-y[i])
    delta = np.sqrt(delta/(Nx-1))
    return delta

Nintervals = 10

# Input parameters for file selection:
psigma   = 1 
spacings = [1,1.25,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,10,15,25,50,75,100]
damp     = 10
N        = len(spacings)
# Input booleans for file selection:
bulkdiffusion = False
substrate     = False
confignrs     = np.arange(1,1001)
bpl           = np.arange(10,12)

# Ds
DRs = np.zeros(N)
Dxs = np.zeros(N)
Dys = np.zeros(N)
Dzs = np.zeros(N)
Dparallel = np.zeros(N)
# Ds, stdv
DRs_stdv = np.zeros(N)
Dxs_stdv = np.zeros(N)
Dys_stdv = np.zeros(N)
Dzs_stdv = np.zeros(N)
Dparallel_stdv = np.zeros(N)

# bs
bRs = np.zeros(N)
bxs = np.zeros(N)
bys = np.zeros(N)
bzs = np.zeros(N)
bparallel = np.zeros(N)
# bs, stdv
bRs_stdv = np.zeros(N)
bxs_stdv = np.zeros(N)
bys_stdv = np.zeros(N)
bzs_stdv = np.zeros(N)
bparallel_stdv = np.zeros(N)


if bulkdiffusion==True:
    parentfolder = 'Pure_bulk/'
    filestext    = '_seed'+str(confignrs[0])+'to'+str(confignrs[-1])
    systemtype   = 'bulk'
    if substrate==True:
        parentfolder = 'Bulk_substrate/'
        systemtype   = 'substrate'
else:
    parentfolder = 'Brush/'
    systemtype   = 'brush'
    filestext    = 'config'+str(confignrs[0])+'to'+str(confignrs[-1])+'_placements'+str(bpl[0])+'to'+str(bpl[-1])

basepath_base   = '/Diffusion_staticbrush/'
endlocation_out = basepath_base+'D_vs_d/Nocut/'
outfilename  = endlocation_out+'D_vs_d_static_better_rms_Nestimates%i.txt' % Nintervals
plotname     = endlocation_out+'D_vs_d_static_better_rms_Nestimates%i.png' % Nintervals
plotname_fit = endlocation_out+'D_vs_d_fit_static_better_rms_Nestimates%i.png' % Nintervals
indfilename  = endlocation_out+'D_vs_d_fitindices_static_better_rms_Nestimates%i.txt' % Nintervals

outfile = open(outfilename, 'w')
outfile.write('d   D_R2   sigmaD_R2  b_R2 sigmab_R2; D_z2  sigmaD_z2 b_z2  sigmaD_z2; D_par2 sigmaD_par2  b_par2  sigmab_par2\n')

indexfile = open(indfilename, 'w')
indexfile.write('Start_index_R     end_index_R     Start_index_ort     end_index_ort     Start_index_par     end_index_par\n')

for i in range(N):
    spacing = spacings[i]
    basepath        = basepath_base+'Spacing'+str(spacing)+'/Radius' + str(psigma) + '/'
    endlocation_in  = basepath + 'Nocut/'
    
    infilename = endlocation_in+'diffusion'+filestext+'_better_rms_Nestimates%i.txt' % Nintervals
    metaname   = endlocation_in+'indices_for_fit_better_rms.txt'
    
    # Read in:
    #### Automatic part
    ## Find the extent of the polymers: Max z-coord of beads in the chains
    infile = open(infilename, "r")
    lines = infile.readlines() # This takes some time
    # Getting the number of lines, etc.
    line = lines[1]
    words = line.split()
    
    # Ds
    DRs[i] = float(words[0])
    Dzs[i] = float(words[2])
    Dparallel[i] = float(words[4])
    # Ds, stdv
    DRs_stdv[i] = float(words[1])
    Dzs_stdv[i] = float(words[3])
    Dparallel_stdv[i] = float(words[5])
    
    infile.close()
    
    outfile.write('%.5e %.5e %.5e %.5e %.5e %.5e %.5e\n' % (spacing, DRs[i], DRs_stdv[i],  Dzs[i], Dzs_stdv[i], Dparallel[i], Dparallel_stdv[i]))
    
    metafile = open(metaname, 'r')
    mlines      = metafile.readlines()
    startindex_R  = int(mlines[0].split()[1]) # New setup
    endindex_R    = int(mlines[0].split()[2])
    startindex_z  = int(mlines[1].split()[1])
    endindex_z    = int(mlines[1].split()[2])
    startindex_p  = int(mlines[2].split()[1])
    endindex_p    = int(mlines[2].split()[2])
    metafile.close()
    indexfile.write('%.2f %i %i %i %i %i %i\n' % (spacing, startindex_R, endindex_R, startindex_z, endindex_z, startindex_p, endindex_p))

outfile.close()

spacings = np.array(spacings)
    
plt.figure(figsize=(6,5))
plt.errorbar(spacings, DRs, yerr=DRs_stdv, capsize=2, label=r'$D_R$')
plt.errorbar(spacings, Dzs, yerr=Dzs_stdv, capsize=2, label=r'$D_\perp$')
plt.errorbar(spacings, Dparallel, yerr=Dparallel_stdv, capsize=2, label=r'$D_\parallel$')
plt.xlabel(r'$d$')
plt.ylabel(r'Diffusion constant $D$')
plt.title('Diffusion constant $D$, static %s, no cut' % systemtype)
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc='upper left')
plt.savefig(plotname)
plt.show() # For testing purposes

