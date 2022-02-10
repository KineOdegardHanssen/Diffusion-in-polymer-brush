import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
import numpy as np
import random
import math
import time
import os
import glob

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

Dzs_all  = []
Dpar_all = []
Dzs_stdv_all  = []
Dpar_stdv_all = []
spacings_all  = []

# Fixed parameters
psigma   = 1 # For instance 
damp     = 10
# Input booleans for file selection:
bulkdiffusion = False
substrate     = False
moresigmas    = False
big           = False
bulk_cut      = False
confignrs     = np.arange(1,1001)
filestext     = '_seed'+str(confignrs[0])+'to'+str(confignrs[-1])

charges  = [-2,-1,0,1,2]
colors   = ['r','b','g','k','darkviolet']
markers  = ['-X','-d','-o','-<','-*'] 
Ncharges = len(charges)

endlocation   = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/Nocut/'
plotname      = endlocation+'D_vs_d_charge.png'
plotname_tiff = endlocation+'D_vs_d_charge.tiff'

#Bulk:
bulklocation = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/'
bulkfilename  = bulklocation + 'diffusion_bulk'+filestext
if bulk_cut==True:
    bulkfilename = bulkfilename +'_cut.txt'
else:
    bulkfilename = bulkfilename +'_uncut.txt'

bulkfile  = open(bulkfilename, 'r')
# D_R2  sigmaD_R2 b_R2 sigmab_R2; D_z2  sigmaD_z2  b_z2  sigmaD_z2; D_par2 sigmaD_par2  b_par2  sigmab_par2
bulklines = bulkfile.readlines()
bulkline  = bulklines[1]
words     = bulkline.split()

# Ds
DRs_bulk = float(words[0])
Dzs_bulk = float(words[4])
Dparallel_bulk = float(words[8])

# Ds, stdv
DRs_stdv_bulk = float(words[1])
Dzs_stdv_bulk = float(words[5])
Dparallel_stdv_bulk = float(words[9])

bulkfile.close()


for charge in charges:
    if charge!=0:
        inlocation = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/Nocut/Charge'+str(charge)+'/'
        infilename = inlocation + 'D_vs_d_charge'+str(charge)+'.txt'
    else:
        infilename = endlocation + 'D_vs_d_better_rms_Nestimates10.txt'
    
    infile = open(infilename,'r')
    
    lines = infile.readlines()
    N_dyn = len(lines)-1
    
    # ds
    dens_dyn = np.zeros(N_dyn)
    spacings_dyn = np.zeros(N_dyn)
    # Ds
    DRs_dyn = np.zeros(N_dyn)
    Dxs_dyn = np.zeros(N_dyn)
    Dys_dyn = np.zeros(N_dyn)
    Dzs_dyn = np.zeros(N_dyn)
    Dparallel_dyn = np.zeros(N_dyn)
    # Ds, stdv
    DRs_stdv_dyn = np.zeros(N_dyn)
    Dxs_stdv_dyn = np.zeros(N_dyn)
    Dys_stdv_dyn = np.zeros(N_dyn)
    Dzs_stdv_dyn = np.zeros(N_dyn)
    Dparallel_stdv_dyn = np.zeros(N_dyn)
    
    for i in range(1,N_dyn+1):
        words = lines[i].split()
        j = i-1
        
        d = float(words[0])
        spacings_dyn[j] = d
        dens_dyn[j] = np.pi/float(d**2)
        DRs_dyn[j]  = float(words[1])
        Dzs_dyn[j]  = float(words[3])
        Dparallel_dyn[j] = float(words[5])
        # Ds, stdv
        DRs_stdv_dyn[j] = float(words[2])
        Dzs_stdv_dyn[j] = float(words[4])
        Dparallel_stdv_dyn[j] = float(words[6])
        
    infile.close()
    
    # Divide by bulk:
    
    for i in range(N_dyn):
        DRnew   = DRs_dyn[i]/DRs_bulk
        Dznew   = Dzs_dyn[i]/DRs_bulk
        Dparnew = Dparallel_dyn[i]/DRs_bulk 
        # Ds, stdv
        DRs_stdv_dyn[i] = abs(DRnew)*np.sqrt((DRs_stdv_dyn[i]/DRs_dyn[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
        Dzs_stdv_dyn[i] = abs(Dznew)*np.sqrt((Dzs_stdv_dyn[i]/Dzs_dyn[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
        Dparallel_stdv_dyn[i] = abs(Dparnew)*np.sqrt((Dparallel_stdv_dyn[i]/Dparallel_dyn[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
        # Ds
        DRs_dyn[i] = DRs_dyn[i]/DRs_bulk
        Dzs_dyn[i] = Dzs_dyn[i]/DRs_bulk
        Dparallel_dyn[i] =  Dparallel_dyn[i]/DRs_bulk
    
    Dzs_all.append(Dzs_dyn)
    Dpar_all.append(Dparallel_dyn)
    spacings_all.append(spacings_dyn)
    Dzs_stdv_all.append(Dzs_stdv_dyn)
    Dpar_stdv_all.append(Dparallel_stdv_dyn)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4),dpi=300)
for i in range(Ncharges):
    ax1.plot(spacings_all[i], Dpar_all[i], markers[i], color=colors[i], label=r'$q=$%.2f' % charges[i])
    ax1.fill_between(spacings_all[i], Dpar_all[i]+Dpar_stdv_all[i], Dpar_all[i]-Dpar_stdv_all[i], facecolor=colors[i], alpha=0.2)
ax1.set_xlabel(r'$d$ (nm)', fontsize=12)
ax1.set_ylabel(r'$D_\parallel/D_{\mathregular{bulk}}$', fontsize=12)
ax1.axis([1.9,10.1,-0.05,0.7])
ax1.legend(loc='upper left',fontsize=11,ncol=2)
ax1.set_title('A',loc='left', fontsize=12)
for i in range(Ncharges):
    ax2.plot(spacings_all[i], Dzs_all[i], markers[i], color=colors[i], label=r'$q=$%.2f' % charges[i])
    ax2.fill_between(spacings_all[i], Dzs_all[i]+Dzs_stdv_all[i], Dzs_all[i]-Dzs_stdv_all[i], facecolor=colors[i], alpha=0.2)
ax2.set_xlabel(r'$d$ (nm)', fontsize=12)
ax2.set_ylabel(r'$D_\perp/D_{\mathregular{bulk}}$', fontsize=12)
ax2.axis([1.9,10.1,-0.05,0.9])
ax2.set_title('B',loc='left', fontsize=12)
plt.tight_layout()
plt.savefig(plotname)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4),dpi=300)
for i in range(Ncharges):
    ax1.plot(spacings_all[i], Dpar_all[i], markers[i], color=colors[i], label=r'$q=$%.2f' % charges[i])
    ax1.fill_between(spacings_all[i], Dpar_all[i]+Dpar_stdv_all[i], Dpar_all[i]-Dpar_stdv_all[i], facecolor=colors[i], alpha=0.2)
ax1.set_ylabel(r'$D_\parallel/D_{\mathregular{bulk}}$', fontsize=12)
ax1.axis([1.9,10.1,-0.05,0.7])
ax1.legend(loc='upper left',fontsize=11,ncol=2)
ax1.set_title('A',loc='left', fontsize=12)
for i in range(Ncharges):
    ax2.plot(spacings_all[i], Dzs_all[i], markers[i], color=colors[i], label=r'$q=$%.2f' % charges[i])
    ax2.fill_between(spacings_all[i], Dzs_all[i]+Dzs_stdv_all[i], Dzs_all[i]-Dzs_stdv_all[i], facecolor=colors[i], alpha=0.2)
ax2.set_xlabel(r'$d$ (nm)', fontsize=12)
ax2.set_ylabel(r'$D_\perp/D_{\mathregular{bulk}}$', fontsize=12)
ax2.axis([1.9,10.1,-0.05,0.9])
ax2.set_title('B',loc='left', fontsize=12)
plt.tight_layout()
plt.savefig(plotname_tiff)

plt.show()
