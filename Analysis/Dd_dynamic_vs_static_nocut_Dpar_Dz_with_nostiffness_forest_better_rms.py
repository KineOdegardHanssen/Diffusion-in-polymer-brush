import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
import numpy as np
import random
import math
import time
import os
import glob

def myexponential(x,A,l):
    return A*np.exp(-x/l)+1

def myexponential_shifted(x,A,l):
    return A*np.exp(-(x-2)/l)+1

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
limitsmall = 11

# Input parameters for file selection: 
psigma   = 1 
pmass    = 1 
damp     = 10
# Input booleans for file selection:
bulkdiffusion = False
substrate     = False
moresigmas    = False
big           = False
bulk_cut      = False
confignrs     = np.arange(1,1001)
filestext     = '_seed'+str(confignrs[0])+'to'+str(confignrs[-1])
old_bulk      = False
endlocation = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/Nocut/'
if moresigmas==True:
    endlocation = '/Diffusion_bead_near_grid/D_vs_d/Brush/Moresigmas/Nocut/'
endlocation_nostiff = endlocation + 'Nostiffness/'
endlocation_f = '/Diffusion_forest/D_vs_d/Nocut/'

basepath_base      = '/Diffusion_staticbrush/'
endlocation_static = basepath_base+'D_vs_d/Nocut/'

bulklocation = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/'

bulkfilename  = bulklocation + 'diffusion_bulk'+filestext
if bulk_cut==True:
    bulkfilename = bulkfilename +'_cut.txt'
else:
    bulkfilename = bulkfilename +'_uncut.txt'
if old_bulk==False:
    bulkfilename  = bulklocation + 'diffusion_bulk'+filestext+'_new.txt'

## Files to read
brushfilename_dyn      = endlocation + 'D_vs_d_better_rms_Nestimates%i.txt' % Nintervals
brushfilename_f        = endlocation_f + 'D_vs_d_forest_better_rms_Nestimates%i.txt' % Nintervals
brushfilename_stat     = endlocation_static + 'D_vs_d_static_better_rms_Nestimates%i.txt' % Nintervals
brushfilename_nostiff  = endlocation_nostiff + 'D_vs_d_better_rms_Nestimates%i.txt' % Nintervals
## Files to write to
if big==False:
    plotname     = endlocation_static+'Dd_dyn_vs_stat_noDR_better_rms_Nestimates%i_all.png' % Nintervals
    plotname_cut = endlocation_static+'Dd_dyn_vs_stat_cut_noDR_better_rms_Nestimates%i_all.png' % Nintervals
    plotname_twoinone = endlocation_static+'Dd_dyn_vs_stat_noDR_twoinone_better_rms_Nestimates%i_all.png' % Nintervals
else:
    plotname     = endlocation_static+'Dd_dyn_vs_stat_big_noDR_better_rms_Nestimates%i_all.png' % Nintervals
    plotname_cut = endlocation_static+'Dd_dyn_vs_stat_cut_big_noDR_better_rms_Nestimates%i_all.png' % Nintervals
    plotname_twoinone = endlocation_static+'Dd_dyn_vs_stat_big_noDR_twoinone_better_rms_Nestimates%i_all.png' % Nintervals
plotname_Dz_fraction_dyn = endlocation_static+'D_vs_d_perp_div_parallel_dyn.png'

# Dynamic sims:
brushfile_dyn = open(brushfilename_dyn, 'r')

lines = brushfile_dyn.readlines()
N_dyn = len(lines)-1

# ds
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
    
    spacings_dyn[j] = float(words[0])
    DRs_dyn[j] = float(words[1])
    Dzs_dyn[j] = float(words[3])
    Dparallel_dyn[j] = float(words[5])
    # Ds, stdv
    DRs_stdv_dyn[j] = float(words[2])
    Dzs_stdv_dyn[j] = float(words[4])
    Dparallel_stdv_dyn[j] = float(words[6])
    
brushfile_dyn.close()

##########

# Static sims:
brushfile_stat = open(brushfilename_stat, 'r')

lines = brushfile_stat.readlines()
N_stat = len(lines)-1

# ds
spacings_stat = np.zeros(N_stat)
# Ds
DRs_stat = np.zeros(N_stat)
Dxs_stat = np.zeros(N_stat)
Dys_stat = np.zeros(N_stat)
Dzs_stat = np.zeros(N_stat)
Dparallel_stat = np.zeros(N_stat)
# Ds, stdv
DRs_stdv_stat = np.zeros(N_stat)
Dxs_stdv_stat = np.zeros(N_stat)
Dys_stdv_stat = np.zeros(N_stat)
Dzs_stdv_stat = np.zeros(N_stat)
Dparallel_stdv_stat = np.zeros(N_stat)

for i in range(1,N_stat+1):
    words = lines[i].split()
    j = i-1
    
    spacings_stat[j] = float(words[0])
    DRs_stat[j] = float(words[1])
    Dzs_stat[j] = float(words[3])
    Dparallel_stat[j] = float(words[5])
    # Ds, stdv
    DRs_stdv_stat[j] = float(words[2])
    Dzs_stdv_stat[j] = float(words[4])
    Dparallel_stdv_stat[j] = float(words[6])
    
brushfile_stat.close()

#######

# Forest sims:
brushfile_f = open(brushfilename_f, 'r')

lines = brushfile_f.readlines()
N_f = len(lines)-1

# ds
spacings_f = np.zeros(N_f)
# Ds
DRs_f = np.zeros(N_f)
Dxs_f = np.zeros(N_f)
Dys_f = np.zeros(N_f)
Dzs_f = np.zeros(N_f)
Dparallel_f = np.zeros(N_f)
# Ds, stdv
DRs_stdv_f = np.zeros(N_f)
Dxs_stdv_f = np.zeros(N_f)
Dys_stdv_f = np.zeros(N_f)
Dzs_stdv_f = np.zeros(N_f)
Dparallel_stdv_f = np.zeros(N_f)

for i in range(1,N_f+1):
    words = lines[i].split()
    j = i-1
    
    spacings_f[j] = float(words[0])
    DRs_f[j] = float(words[1])
    Dzs_f[j] = float(words[3])
    Dparallel_f[j] = float(words[5])
    # Ds, stdv
    DRs_stdv_f[j] = float(words[2])
    Dzs_stdv_f[j] = float(words[4])
    Dparallel_stdv_f[j] = float(words[6])
    
brushfile_f.close()

#####
# Nostiff sims:
brushfile_nostiff = open(brushfilename_nostiff, 'r')

lines = brushfile_nostiff.readlines()
N_nostiff = len(lines)-1

# ds
spacings_nostiff = np.zeros(N_nostiff)
# Ds
DRs_nostiff = np.zeros(N_nostiff)
Dxs_nostiff = np.zeros(N_nostiff)
Dys_nostiff = np.zeros(N_nostiff)
Dzs_nostiff = np.zeros(N_nostiff)
Dparallel_nostiff = np.zeros(N_nostiff)
# Ds, stdv
DRs_stdv_nostiff = np.zeros(N_nostiff)
Dxs_stdv_nostiff = np.zeros(N_nostiff)
Dys_stdv_nostiff = np.zeros(N_nostiff)
Dzs_stdv_nostiff = np.zeros(N_nostiff)
Dparallel_stdv_nostiff = np.zeros(N_nostiff)

for i in range(1,N_nostiff+1):
    words = lines[i].split()
    j = i-1
    
    spacings_nostiff[j] = float(words[0])
    DRs_nostiff[j] = float(words[1])
    Dzs_nostiff[j] = float(words[3])
    Dparallel_nostiff[j] = float(words[5])
    # Ds, stdv
    DRs_stdv_nostiff[j] = float(words[2])
    Dzs_stdv_nostiff[j] = float(words[4])
    Dparallel_stdv_nostiff[j] = float(words[6])
    
brushfile_nostiff.close()


###
#Bulk:

bulkfile  = open(bulkfilename, 'r')
# D_R2  sigmaD_R2 b_R2 sigmab_R2; D_z2  sigmaD_z2  b_z2  sigmaD_z2; D_par2 sigmaD_par2  b_par2  sigmab_par2
if old_bulk==True:
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
else:
    bulklines = bulkfile.readlines()
    bulkline  = bulklines[1]
    words     = bulkline.split()
    
    # Ds
    DRs_bulk = float(words[1])
    Dzs_bulk = float(words[3])
    Dparallel_bulk = float(words[5])
    
    # Ds, stdv
    DRs_stdv_bulk = float(words[2])
    Dzs_stdv_bulk = float(words[4])
    Dparallel_stdv_bulk = float(words[6])
    
bulkfile.close()

# Divide by bulk:
for i in range(N_stat):
    DRnew = DRs_stat[i]/DRs_bulk
    Dznew = Dzs_stat[i]/DRs_bulk
    Dparnew = Dparallel_stat[i]/DRs_bulk
    # Ds, stdv
    DRs_stdv_stat[i] = abs(DRnew)*np.sqrt((DRs_stdv_stat[i]/DRs_stat[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    Dzs_stdv_stat[i] = abs(Dznew)*np.sqrt((Dzs_stdv_stat[i]/Dzs_stat[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    Dparallel_stdv_stat[i] = abs(Dparnew)*np.sqrt((Dparallel_stdv_stat[i]/Dparallel_stat[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    # Ds
    DRs_stat[i] = DRnew
    Dzs_stat[i] = Dznew
    Dparallel_stat[i] = Dparnew

for i in range(N_dyn):
    DRnew = DRs_dyn[i]/DRs_bulk
    Dznew = Dzs_dyn[i]/DRs_bulk
    Dparnew = Dparallel_dyn[i]/DRs_bulk
    # Ds, stdv
    DRs_stdv_dyn[i] = abs(DRnew)*np.sqrt((DRs_stdv_dyn[i]/DRs_dyn[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    Dzs_stdv_dyn[i] = abs(Dznew)*np.sqrt((Dzs_stdv_dyn[i]/Dzs_dyn[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    Dparallel_stdv_dyn[i] = abs(Dparnew)*np.sqrt((Dparallel_stdv_dyn[i]/Dparallel_dyn[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    # Ds
    DRs_dyn[i] = DRnew
    Dzs_dyn[i] = Dznew
    Dparallel_dyn[i] =  Dparnew

####
for i in range(N_f):
    DRnew = DRs_f[i]/DRs_bulk
    Dznew = Dzs_f[i]/DRs_bulk
    Dparnew = Dparallel_f[i]/DRs_bulk
    # Ds, stdv
    DRs_stdv_f[i] = abs(DRnew)*np.sqrt((DRs_stdv_f[i]/DRs_f[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    Dzs_stdv_f[i] = abs(Dznew)*np.sqrt((Dzs_stdv_f[i]/Dzs_f[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    Dparallel_stdv_f[i] = abs(Dparnew)*np.sqrt((Dparallel_stdv_f[i]/Dparallel_f[i])**2 +(DRs_stdv_bulk/DRs_bulk)**2)
    # Ds
    DRs_f[i] = DRnew
    Dzs_f[i] = Dznew
    Dparallel_f[i] = Dparnew

for i in range(N_nostiff):
    DRnew = DRs_nostiff[i]/DRs_bulk
    Dznew = Dzs_nostiff[i]/DRs_bulk
    Dparnew = Dparallel_nostiff[i]/DRs_bulk
    # Ds, stdv
    DRs_stdv_nostiff[i] = abs(DRnew)*np.sqrt((DRs_stdv_nostiff[i]/DRs_nostiff[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    Dzs_stdv_nostiff[i] = abs(Dznew)*np.sqrt((Dzs_stdv_nostiff[i]/Dzs_nostiff[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    Dparallel_stdv_nostiff[i] = abs(Dparnew)*np.sqrt((Dparallel_stdv_nostiff[i]/Dparallel_nostiff[i])**2+(DRs_stdv_bulk/DRs_bulk)**2)
    # Ds
    DRs_nostiff[i] = DRnew
    Dzs_nostiff[i] = Dznew
    Dparallel_nostiff[i] = Dparnew

minforsmall = 0
maxforsmall = 0
i = 0
d = 1
while d<limitsmall:
    ## Max vals:
    Dthismax_zdyn = Dzs_dyn[i]+Dzs_stdv_dyn[i]
    Dthismax_pdyn = Dparallel_dyn[i]+Dparallel_stdv_dyn[i]
    Dthismax_zstat = Dzs_stat[i]+Dzs_stdv_stat[i]
    Dthismax_pstat = Dparallel_stat[i]+Dparallel_stdv_stat[i]
    ## Min vals:
    Dthismin_zdyn = Dzs_dyn[i]-Dzs_stdv_dyn[i]
    Dthismin_pdyn = Dparallel_dyn[i]-Dparallel_stdv_dyn[i]
    Dthismin_zstat = Dzs_stat[i]-Dzs_stdv_stat[i]
    Dthismin_pstat = Dparallel_stat[i]-Dparallel_stdv_stat[i]
    ###### Testing if larger:
    if Dthismax_zdyn>maxforsmall:
        maxforsmall=Dthismax_zdyn
    if Dthismax_pdyn>maxforsmall:
        maxforsmall=Dthismax_pdyn
    if Dthismax_zstat>maxforsmall:
        maxforsmall=Dthismax_zstat
    if Dthismax_pstat>maxforsmall:
        maxforsmall=Dthismax_pstat
    ##### Testing if smaller:
    if Dthismin_zdyn<minforsmall:
        minforsmall=Dthismin_zdyn
    if Dthismin_pdyn<minforsmall:
        minforsmall=Dthismin_pdyn
    if Dthismin_zstat<minforsmall:
        minforsmall=Dthismin_zstat
    if Dthismin_pstat<minforsmall:
        minforsmall=Dthismin_pstat
    i+=1
    d = spacings_dyn[i] 

# Double check for forest:
i = 0
d = 1
while d<limitsmall:
    # Max vals:
    Dthismax_zf = Dzs_f[i]+Dzs_stdv_f[i]
    Dthismax_pf = Dparallel_f[i]+Dparallel_stdv_f[i]
    # Min vals:
    Dthismin_zf = Dzs_f[i]-Dzs_stdv_f[i]
    Dthismin_pf = Dparallel_f[i]-Dparallel_stdv_f[i]
    #### Testing if larger:
    if Dthismax_zf>maxforsmall:
        maxforsmall=Dthismax_zf
    if Dthismax_pf>maxforsmall:
        maxforsmall=Dthismax_pf
    #### Testing if smaller:
    if Dthismin_zf<minforsmall:
        minforsmall=Dthismin_zf
    if Dthismin_pf<minforsmall:
        minforsmall=Dthismin_pf
    i+=1
    d = spacings_f[i] 


# Double check for nostiff
i = 0
d = 1
while d<11:
    # Max vals:
    Dthismax_zn = Dzs_nostiff[i]+Dzs_stdv_nostiff[i]
    Dthismax_pn = Dparallel_nostiff[i]+Dparallel_stdv_nostiff[i]
    # Min vals:
    Dthismin_zn = Dzs_nostiff[i]-Dzs_stdv_nostiff[i]
    Dthismin_pn = Dparallel_nostiff[i]-Dparallel_stdv_nostiff[i]
    #### Testing if larger:
    if Dthismax_zn>maxforsmall:
        maxforsmall=Dthismax_zn
    if Dthismax_pn>maxforsmall:
        maxforsmall=Dthismax_pn
    #### Testing if smaller:
    if Dthismin_zn<minforsmall:
        minforsmall=Dthismin_zn
    if Dthismin_pn<minforsmall:
        minforsmall=Dthismin_pn
    i+=1
    d = spacings_nostiff[i] 



if minforsmall<0:
    minforsmall*=1.2
else:
    minforsmall*=0.8
maxforsmall*=1.05

if big==False:
    plt.figure(figsize=(8,5))
    ax = plt.subplot(111)    
    # Data points
    ax.plot(spacings_dyn, Dzs_dyn, '-o' , color='b',label=r'$D_\perp$, dyn.')
    ax.plot(spacings_dyn, Dparallel_dyn, '-*', color='g', label=r'$D_\parallel$, dyn.')
    ax.plot(spacings_stat, Dzs_stat, '-s', color='c', label=r'$D_\perp$, stat.')
    ax.plot(spacings_stat, Dparallel_stat, '-d', color='limegreen',label=r'$D_\parallel$, stat.')
    ax.plot(spacings_f, Dzs_f, '-v' , color='r',label=r'$D_\perp$, straight')
    ax.plot(spacings_f, Dparallel_f, '-1', color='black', label=r'$D_\parallel$, straight')
    ax.plot(spacings_nostiff, Dzs_nostiff, '-X', color='lightcoral', label=r'$D_\perp$, no stiffness')
    ax.plot(spacings_nostiff, Dparallel_nostiff, '-<', color='lightgray',label=r'$D_\parallel$, no stiffness')
    # Fill between
    ax.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
    ax.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
    ax.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
    ax.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
    ax.fill_between(spacings_f, Dzs_f+Dzs_stdv_f, Dzs_f-Dzs_stdv_f, color='r',alpha=0.2)
    ax.fill_between(spacings_f, Dparallel_f+Dparallel_stdv_f, Dparallel_f-Dparallel_stdv_f, color='black',alpha=0.2)
    ax.fill_between(spacings_nostiff, Dzs_nostiff+Dzs_stdv_nostiff, Dzs_nostiff-Dzs_stdv_nostiff,color='lightcoral',alpha=0.2)
    ax.fill_between(spacings_nostiff, Dparallel_nostiff+Dparallel_stdv_nostiff, Dparallel_nostiff-Dparallel_stdv_nostiff, color='lightgray',alpha=0.5)
    if moresigmas==True:
        plt.xlabel(r'$d/\sigma_b$')
    else:
        plt.xlabel(r'$d$ (nm)')
    plt.ylabel(r'$D/D_{\mathregular{bulk}}$')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
else:
    plt.figure(figsize=(16,10))
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14) 
    ax = plt.subplot(111)
    # Data points
    ax.plot(spacings_dyn, Dzs_dyn, '-o' , color='b',label=r'$D_\perp$, dyn.')
    ax.plot(spacings_dyn, Dparallel_dyn, '-*', color='g', label=r'$D_\parallel$, dyn.')
    ax.plot(spacings_stat, Dzs_stat, '-s', color='c', label=r'$D_\perp$, stat.')
    ax.plot(spacings_stat, Dparallel_stat, '-d', color='limegreen',label=r'$D_\parallel$, stat.')
    ax.plot(spacings_f, Dzs_f, '-v' , color='r',label=r'$D_\perp$, straight')
    ax.plot(spacings_f, Dparallel_f, '-1', color='black', label=r'$D_\parallel$, straight')
    ax.plot(spacings_nostiff, Dzs_nostiff, '-X', color='lightcoral', label=r'$D_\perp$, no stiffness')
    ax.plot(spacings_nostiff, Dparallel_nostiff, '-<', color='lightgray',label=r'$D_\parallel$, no stiffness')
    # Fill between:
    ax.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
    ax.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
    ax.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
    ax.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
    if moresigmas==True:
        plt.xlabel(r'$d/\sigma_b$', fontsize=14)
    else:
        plt.xlabel(r'$d$ (nm)', fontsize=14)
    plt.ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=14)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 20})
plt.savefig(plotname)

if big==False:
    plt.figure(figsize=(6.4,5))
    ax = plt.subplot(111)    
    ax.plot(spacings_dyn, Dzs_dyn, '-o' , color='b',label=r'$D_\perp$, dyn.')
    ax.plot(spacings_dyn, Dparallel_dyn, '-*', color='g', label=r'$D_\parallel$, dyn.')
    ax.plot(spacings_stat, Dzs_stat, '-s', color='c', label=r'$D_\perp$, stat.')
    ax.plot(spacings_stat, Dparallel_stat, '-d', color='limegreen',label=r'$D_\parallel$, stat.')
    ax.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
    ax.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
    ax.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
    ax.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
    if moresigmas==True:
        plt.xlabel(r'$d/\sigma_b$')
    else:
        plt.xlabel(r'$d$ (nm)')
    plt.ylabel(r'$D/D_{\mathregular{bulk}}$')
    ax.axis([0,11,minforsmall,maxforsmall])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
else:
    plt.figure(figsize=(12.8,10))
    ax = plt.subplot(111)
    ax.plot(spacings_dyn, Dzs_dyn, '-o' , color='b',label=r'$D_\perp$, dyn.')
    ax.plot(spacings_dyn, Dparallel_dyn, '-*', color='g', label=r'$D_\parallel$, dyn.')
    ax.plot(spacings_stat, Dzs_stat, '-s', color='c', label=r'$D_\perp$, stat.')
    ax.plot(spacings_stat, Dparallel_stat, '-d', color='limegreen',label=r'$D_\parallel$, stat.')
    ax.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
    ax.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
    ax.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
    ax.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
    if moresigmas==True:
        plt.xlabel(r'$d/\sigma_b$', fontsize=14)
    else:
        plt.xlabel(r'$d$ (nm)', fontsize=14)
    plt.ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=14)
    ax.axis([0,11,minforsmall,maxforsmall]) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(plotname_cut)

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
################## twoinone ####################################
Nattempt = len(spacings_dyn) 
attempt = np.zeros(Nattempt)
for i in range(Nattempt):
    attempt[i] = 1-np.pi*(1/spacings_dyn[i])**2
if big==False:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4),dpi=300)
    ax1.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
    ax1.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
    ax1.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
    ax1.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
    ax1.plot(spacings_dyn, Dzs_dyn, '-o' , color='b',label=r'$D_\perp$, dyn.')
    ax1.plot(spacings_dyn, Dparallel_dyn, '-*', color='g', label=r'$D_\parallel$, dyn.')
    ax1.plot(spacings_stat, Dzs_stat, '-s', color='c', label=r'$D_\perp$, stat.')
    ax1.plot(spacings_stat, Dparallel_stat, '-d', color='limegreen',label=r'$D_\parallel$, stat.')
    if moresigmas==True:
        ax1.set_xlabel(r'$d/\sigma_b$', fontsize=13)
        ax1.set_ylabel('$D/D_{\mathregular{bulk}}$', fontsize=13)
    else:
        ax1.set_xlabel(r'$d$ (nm)', fontsize=13)
        ax1.set_ylabel('$D/D_{\mathregular{bulk}}$', fontsize=13)
    ax1.set_title('A',loc='left',fontsize=14)
    ax1.legend(loc="lower right",fontsize=12)
    # Fill between
    ax2.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
    ax2.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
    ax2.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
    ax2.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
    ax2.fill_between(spacings_f, Dzs_f+Dzs_stdv_f, Dzs_f-Dzs_stdv_f, color='r',alpha=0.2)
    ax2.fill_between(spacings_f, Dparallel_f+Dparallel_stdv_f, Dparallel_f-Dparallel_stdv_f, color='black',alpha=0.2)
    ax2.fill_between(spacings_nostiff, Dzs_nostiff+Dzs_stdv_nostiff, Dzs_nostiff-Dzs_stdv_nostiff,color='lightcoral',alpha=0.2)
    ax2.fill_between(spacings_nostiff, Dparallel_nostiff+Dparallel_stdv_nostiff, Dparallel_nostiff-Dparallel_stdv_nostiff, color='lightgray',alpha=0.5)
    # Plot data
    ax2.plot(spacings_dyn, Dzs_dyn, '-o' , color='b')
    ax2.plot(spacings_dyn, Dparallel_dyn, '-*', color='g')
    ax2.plot(spacings_stat, Dzs_stat, '-s', color='c')
    ax2.plot(spacings_stat, Dparallel_stat, '-d', color='limegreen')
    ax2.plot(spacings_f, Dzs_f, '-v' , color='r',label=r'$D_\perp$, straight')
    ax2.plot(spacings_f, Dparallel_f, '-1', color='black', label=r'$D_\parallel$, straight')
    ax2.plot(spacings_nostiff, Dzs_nostiff, '-X', color='lightcoral', label=r'$D_\perp$, no stiff.')
    ax2.plot(spacings_nostiff, Dparallel_nostiff, '-<', color='lightgray',label=r'$D_\parallel$, no stiff.')
    if moresigmas==True:
        ax2.set_xlabel(r'$d/\sigma_b$', fontsize=13)
        ax2.set_ylabel(ylabel=r'$D/D_{\mathregular{bulk}}$', fontsize=13)
    else:
        ax2.set_xlabel(r'$d$ (nm)', fontsize=13)
        ax2.set_ylabel(ylabel=r'$D/D_{\mathregular{bulk}}$', fontsize=13)
    ax2.axis([0,limitsmall,minforsmall,maxforsmall])
    ax2.set_title('B',loc='left',fontsize=14)
    ax2.legend(loc="upper left",fontsize=12)
    fig.tight_layout()
    plt.show()
    fig.savefig(plotname_twoinone)
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4),dpi=300)
    ax1.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
    ax1.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
    ax1.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
    ax1.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
    ax1.plot(spacings_dyn, Dzs_dyn, '-o' , color='b',label=r'$D_\perp$, dyn.')
    ax1.plot(spacings_dyn, Dparallel_dyn, '-*', color='g', label=r'$D_\parallel$, dyn.')
    ax1.plot(spacings_stat, Dzs_stat, '-s', color='c', label=r'$D_\perp$, stat.')
    ax1.plot(spacings_stat, Dparallel_stat, '-d', color='limegreen',label=r'$D_\parallel$, stat.')
    if moresigmas==True:
        ax1.set_xlabel(r'$d/\sigma_b$', fontsize=13)
        ax1.set_ylabel('$D/D_{\mathregular{bulk}}$', fontsize=13)
    else:
        ax1.set_xlabel(r'$d$ (nm)', fontsize=13)
        ax1.set_ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=13)
    ax1.set_title('A',loc='left',fontsize=14)
    # Fill between
    ax2.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
    ax2.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
    ax2.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
    ax2.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
    ax2.fill_between(spacings_f, Dzs_f+Dzs_stdv_f, Dzs_f-Dzs_stdv_f, color='r',alpha=0.2)
    ax2.fill_between(spacings_f, Dparallel_f+Dparallel_stdv_f, Dparallel_f-Dparallel_stdv_f, color='black',alpha=0.2)
    ax2.fill_between(spacings_nostiff, Dzs_nostiff+Dzs_stdv_nostiff, Dzs_nostiff-Dzs_stdv_nostiff,color='lightcoral',alpha=0.2)
    ax2.fill_between(spacings_nostiff, Dparallel_nostiff+Dparallel_stdv_nostiff, Dparallel_nostiff-Dparallel_stdv_nostiff, color='lightgray',alpha=0.5)
    # Plot data
    ax2.plot(spacings_dyn, Dzs_dyn, '-o' , color='b')
    ax2.plot(spacings_dyn, Dparallel_dyn, '-*', color='g')
    ax2.plot(spacings_stat, Dzs_stat, '-s', color='c')
    ax2.plot(spacings_stat, Dparallel_stat, '-d')
    ax2.plot(spacings_f, Dzs_f, '-v' , color='r',label=r'$D_\perp$, straight')
    ax2.plot(spacings_f, Dparallel_f, '-1', color='black', label=r'$D_\parallel$, straight')
    ax2.plot(spacings_nostiff, Dzs_nostiff, '-X', color='lightcoral', label=r'$D_\perp$, no stiff.')
    ax2.plot(spacings_nostiff, Dparallel_nostiff, '-<', color='lightgray',label=r'$D_\parallel$, no stiff.')
    if moresigmas==True:
        ax2.set_xlabel(r'$d/\sigma_b$', fontsize=13)
        ax2.set_ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=13)
    else:
        ax2.set_xlabel(r'$d$ (nm)', fontsize=13)
        ax2.set_ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=13)
    ax2.axis([0,limitsmall,minforsmall,maxforsmall])
    ax2.set_title('B',loc='left',fontsize=14)
    ax2.legend(loc="upper left",fontsize=12)
    fig.tight_layout()
    plt.show()
    fig.savefig(plotname_twoinone)

spacings_fraction = []
Dzs_fraction      = []
Dzs_diff          = []
Dzs_rms_fraction  = []
Dzs_rms_diff      = []
ones              = []

i = 0
spacing = spacings_f[i]
while spacing<=25:
    spacings_fraction.append(spacing)    
    Dznew = Dzs_f[i]/Dzs_nostiff[i]
    # Ds, stdv
    Dzs_stdv = abs(Dznew)*np.sqrt((Dzs_stdv_nostiff[i]/Dzs_nostiff[i])**2+(Dzs_stdv_f[i]/Dzs_f[i])**2)
    Dzs_stdv_diff = np.sqrt((Dzs_stdv_nostiff[i])**2+(Dzs_stdv_f[i])**2)
    Dzs_fraction.append(Dznew)
    Dzs_rms_fraction.append(Dzs_stdv)
    Dzs_diff.append(Dzs_f[i]-Dzs_nostiff[i])
    Dzs_rms_diff.append(Dzs_stdv_diff)
    ones.append(1)
    i+=1
    spacing = spacings_f[i]

ones = np.array(ones)
Dzs_diff = np.array(Dzs_diff)
Dzs_fraction = np.array(Dzs_fraction)
Dzs_rms_diff = np.array(Dzs_rms_diff)
Dzs_rms_fraction = np.array(Dzs_rms_fraction)
spacings_fraction = np.array(spacings_fraction)

plt.figure(figsize=(6,5),dpi=300)
plt.fill_between(spacings_fraction, Dzs_fraction+Dzs_rms_fraction, Dzs_fraction-Dzs_rms_fraction,color='lightcoral',alpha=0.2)
plt.plot(spacings_fraction, Dzs_fraction, '-o',color='lightcoral')
plt.plot(spacings_fraction, ones, '--',color='lightcoral')
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D_{straight}/D_{no stiff.}$', fontsize=14)
plt.show()

plt.figure(figsize=(6,5),dpi=300)
plt.fill_between(spacings_fraction, Dzs_diff+Dzs_rms_diff, Dzs_diff-Dzs_rms_diff,color='lightcoral',alpha=0.2)
plt.plot(spacings_fraction, Dzs_diff, '-o',color='lightcoral')
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D_{straight}/D_{bulk}-D_{no stiff.}/D_{bulk}$', fontsize=14)
plt.show()


### Dyn vs. stat.

spacings_fraction = []
Dzs_fraction      = []
Dzs_diff          = []
Dzs_rms_fraction  = []
Dzs_rms_diff      = []
ones              = []

i = 0
spacing = spacings_f[i]
while spacing<=25:
    spacings_fraction.append(spacing)    
    Dznew = Dzs_dyn[i]/Dzs_stat[i]
    # Ds, stdv
    Dzs_stdv = abs(Dznew)*np.sqrt((Dzs_stdv_dyn[i]/Dzs_dyn[i])**2+(Dzs_stdv_stat[i]/Dzs_stat[i])**2)
    Dzs_stdv_diff = np.sqrt((Dzs_stdv_dyn[i])**2+(Dzs_stdv_stat[i])**2)
    Dzs_fraction.append(Dznew)
    Dzs_rms_fraction.append(Dzs_stdv)
    Dzs_diff.append(Dzs_dyn[i]-Dzs_stat[i])
    Dzs_rms_diff.append(Dzs_stdv_diff)
    ones.append(1)
    i+=1
    spacing = spacings_dyn[i]

ones = np.array(ones)
Dzs_diff = np.array(Dzs_diff)
Dzs_fraction = np.array(Dzs_fraction)
Dzs_rms_diff = np.array(Dzs_rms_diff)
Dzs_rms_fraction = np.array(Dzs_rms_fraction)
spacings_fraction = np.array(spacings_fraction)

plt.figure(figsize=(6,5),dpi=300)
plt.fill_between(spacings_fraction, Dzs_fraction+Dzs_rms_fraction, Dzs_fraction-Dzs_rms_fraction,color='lightcoral',alpha=0.2)
plt.plot(spacings_fraction, Dzs_fraction, '-o',color='lightcoral')
plt.plot(spacings_fraction, ones, '--',color='lightcoral')
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D_{dyn}/D_{stat}$', fontsize=14)
plt.show()

plt.figure(figsize=(6,5),dpi=300)
plt.fill_between(spacings_fraction, Dzs_diff+Dzs_rms_diff, Dzs_diff-Dzs_rms_diff,color='lightcoral',alpha=0.2)
plt.plot(spacings_fraction, Dzs_diff, '-o',color='lightcoral')
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D_{dyn}/D_{bulk}-D_{stat}/D_{bulk}$', fontsize=14)
plt.show()

### Dyn, perp vs parallel

spacings_fraction = []
Dzs_fraction      = []
Dzs_diff          = []
Dzs_rms_fraction  = []
Dzs_rms_diff      = []
ones              = []

spacings_few = []
Dzs_fraction_few = []
Dzs_fraction_rms_few = []

spacings_plot = []
Dzs_fraction_plot = []
Dzs_fraction_rms_plot = []

i = 0
spacing = spacings_f[i]
while spacing<=25:
    spacings_fraction.append(spacing)    
    Dznew = Dzs_dyn[i]/Dparallel_dyn[i]
    # Ds, stdv
    Dzs_stdv = abs(Dznew)*np.sqrt((Dzs_stdv_dyn[i]/Dzs_dyn[i])**2+(Dparallel_stdv_dyn[i]/Dparallel_dyn[i])**2)
    Dzs_stdv_diff = np.sqrt((Dzs_stdv_dyn[i])**2+(Dparallel_stdv_dyn[i])**2)
    Dzs_fraction.append(Dznew)
    Dzs_rms_fraction.append(Dzs_stdv)
    Dzs_diff.append(Dzs_dyn[i]-Dparallel_dyn[i])
    Dzs_rms_diff.append(Dzs_stdv_diff)
    if spacing>1.9:
        spacings_plot.append(spacing)
        Dzs_fraction_plot.append(Dznew)
        Dzs_fraction_rms_plot.append(Dzs_stdv)
        ones.append(1)
    if spacing>2:
        spacings_few.append(spacing)
        Dzs_fraction_few.append(Dznew)
        Dzs_fraction_rms_few.append(Dzs_stdv)
    i+=1
    spacing = spacings_dyn[i]

ones = np.array(ones)
Dzs_diff = np.array(Dzs_diff)
Dzs_fraction = np.array(Dzs_fraction)
Dzs_rms_diff = np.array(Dzs_rms_diff)
spacings_few = np.array(spacings_few)
Dzs_fraction_few = np.array(Dzs_fraction_few)
Dzs_rms_fraction = np.array(Dzs_rms_fraction)
spacings_fraction = np.array(spacings_fraction)
Dzs_fraction_few_dyn = np.array(Dzs_fraction_plot)
Dzs_fraction_few_dyn_rms = np.array(Dzs_fraction_rms_plot)

Dzs_fraction_few_dyn_prms = Dzs_fraction_few_dyn+Dzs_fraction_few_dyn_rms
Dzs_fraction_few_dyn_mrms = Dzs_fraction_few_dyn-Dzs_fraction_few_dyn_rms


coeffs, covs = curve_fit(myexponential,spacings_few, Dzs_fraction_few) 
A_dyn = coeffs[0]
b_dyn = coeffs[1]
rms_A_dyn = np.sqrt(covs[0,0])
rms_b_dyn = np.sqrt(covs[1,1])

dstart = spacings_few[0]
dend   = spacings_few[-1]
spacings_fit = np.linspace(dstart,dend,100)
fit_dyn      = myexponential(spacings_fit,A_dyn,b_dyn)

# Shifted:
coeffs, covs = curve_fit(myexponential_shifted,spacings_few, Dzs_fraction_few) 
A_dyn_shifted = coeffs[0]
b_dyn_shifted = coeffs[1]
rms_A_dyn_shifted = np.sqrt(covs[0,0])
rms_b_dyn_shifted = np.sqrt(covs[1,1])

fit_dyn_shifted      = myexponential_shifted(spacings_fit,A_dyn_shifted,b_dyn_shifted)

#'''
plt.figure(figsize=(6,4),dpi=300)
plt.fill_between(spacings_plot, Dzs_fraction_few_dyn_prms, Dzs_fraction_few_dyn_mrms,color='lightcoral',alpha=0.2)
plt.plot(spacings_plot, Dzs_fraction_few_dyn, '-o',color='lightcoral',label='$D_{\perp, \mathregular{dyn}}/D_{\parallel, \mathregular{dyn}}$')
plt.plot(spacings_plot, ones, '--',color='lightcoral')
plt.plot(spacings_fit,fit_dyn,'--',label=r'Exponential fit')
plt.xlabel(r'$d$ (nm)', fontsize=15)
plt.ylabel(r'$D_{\perp}/D_{\parallel}$', fontsize=15)
plt.legend(loc='upper right',fontsize=12)
plt.tight_layout()
plt.savefig(plotname_Dz_fraction_dyn)

'''
plt.figure(figsize=(6,5),dpi=300)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
plt.fill_between(spacings_fraction, Dzs_diff+Dzs_rms_diff, Dzs_diff-Dzs_rms_diff,color='lightcoral',alpha=0.2)
plt.plot(spacings_fraction, Dzs_diff, '-o',color='lightcoral')
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D_{\perp, \mathregular{dyn}}/D_{\mathregular{bulk}}-D_{\parallel, \mathregular{dyn}}/D_{\mathregular{bulk}}$', fontsize=14)
plt.show()
#'''

### Stat, perp vs parallel

spacings_fraction = []
Dzs_fraction      = []
Dzs_diff          = []
Dzs_rms_fraction  = []
Dzs_rms_diff      = []
ones              = []
spacings_few      = []
Dzs_fraction_few  = []
Dzs_fraction_rms_few = []

i = 0
spacing = spacings_f[i]
while spacing<=25:
    spacings_fraction.append(spacing)    
    Dznew = Dzs_stat[i]/Dparallel_stat[i]
    # Ds, stdv
    Dzs_stdv = abs(Dznew)*np.sqrt((Dzs_stdv_stat[i]/Dzs_stat[i])**2+(Dparallel_stdv_stat[i]/Dparallel_stat[i])**2)
    Dzs_stdv_diff = np.sqrt((Dzs_stdv_stat[i])**2+(Dparallel_stdv_stat[i])**2)
    Dzs_fraction.append(Dznew)
    Dzs_rms_fraction.append(Dzs_stdv)
    Dzs_diff.append(Dzs_stat[i]-Dparallel_stat[i])
    Dzs_rms_diff.append(Dzs_stdv_diff)
    if spacing>1.5:
        spacings_few.append(spacing)
        Dzs_fraction_few.append(Dznew)
        Dzs_fraction_rms_few.append(Dzs_stdv)
        ones.append(1)
    i+=1
    spacing = spacings_stat[i]

ones = np.array(ones)
Dzs_diff = np.array(Dzs_diff)
spacings_few = np.array(spacings_few)
Dzs_fraction = np.array(Dzs_fraction)
Dzs_rms_diff = np.array(Dzs_rms_diff)
Dzs_fraction_few = np.array(Dzs_fraction_few)
Dzs_rms_fraction = np.array(Dzs_rms_fraction)
spacings_fraction = np.array(spacings_fraction)
Dzs_fraction_few_stat = np.array(Dzs_fraction_few)
Dzs_fraction_few_stat_rms = np.array(Dzs_fraction_rms_few)

Dzs_fraction_few_stat_prms = Dzs_fraction_few_stat+Dzs_fraction_few_stat_rms
Dzs_fraction_few_stat_mrms = Dzs_fraction_few_stat-Dzs_fraction_few_stat_rms

coeffs, covs = curve_fit(myexponential,spacings_few, Dzs_fraction_few) 
A_stat = coeffs[0]
b_stat = coeffs[1]
rms_A_stat = np.sqrt(covs[0,0])
rms_b_stat = np.sqrt(covs[1,1])


dstart = spacings_few[0]
dend   = spacings_few[-1]
spacings_fit = np.linspace(dstart,dend,100)
fit_stat     = myexponential(spacings_fit,A_stat,b_stat)

# Shifted
coeffs, covs = curve_fit(myexponential_shifted,spacings_few, Dzs_fraction_few)
A_stat_shifted = coeffs[0]
b_stat_shifted = coeffs[1]
rms_A_stat_shifted = np.sqrt(covs[0,0])
rms_b_stat_shifted = np.sqrt(covs[1,1])

fit_stat_shifted     = myexponential_shifted(spacings_fit,A_stat_shifted,b_stat_shifted)

#'''
plotname_x = endlocation_static+'D_vs_d_perp_div_parallel_stat_fitinterval_zoomed.png'
plt.figure(figsize=(6,4),dpi=300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.fill_between(spacings_few, Dzs_fraction_few_stat_prms, Dzs_fraction_few_stat_mrms,color='lightcoral',alpha=0.2)
plt.plot(spacings_few, Dzs_fraction_few_stat, '-o',color='lightcoral',label='$D_{\perp, \mathregular{stat}}/D_{\parallel, \mathregular{stat}}$')
plt.plot(spacings_few, ones, '--',color='lightcoral')
plt.plot(spacings_fit,fit_stat,'--',label=r'Exponential fit')
plt.xlabel(r'$d$ (nm)', fontsize=15)
plt.ylabel(r'$D_{\perp}/D_{\parallel}$', fontsize=15)
plt.legend(loc='upper right',fontsize=11)
plt.axis([1.7,25.3,-4,13])
plt.tight_layout()
plt.savefig(plotname_x)

'''
plt.figure(figsize=(6,5),dpi=300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.fill_between(spacings_fraction, Dzs_diff+Dzs_rms_diff, Dzs_diff-Dzs_rms_diff,color='lightcoral',alpha=0.2)
plt.plot(spacings_fraction, Dzs_diff, '-o',color='lightcoral')
plt.xlabel(r'$d$ (nm)', fontsize=15)
plt.ylabel(r'$D_{\perp, \mathregular{stat}}/D_{\mathregular{bulk}}-D_{\parallel, \mathregular{stat}}/D_{\mathregular{bulk}}$', fontsize=15)
plt.show()
#'''


print("A_dyn:",A_dyn, ", b_dyn:",b_dyn)
print("rms_A_dyn:",rms_A_dyn, ", rms_b_dyn:",rms_b_dyn)
print("----------------------------------")
print("A_stat:",A_stat, ", b_stat:",b_stat)
print("rms_A_stat:",rms_A_stat, ", rms_b_stat:",rms_b_stat)

print("Done.")

### No stiff, perp vs parallel

spacings_fraction = []
Dzs_fraction      = []
Dzs_diff          = []
Dzs_rms_fraction  = []
Dzs_rms_diff      = []
ones              = []
spacings_few      = []
Dzs_fraction_few  = []
Dzs_fraction_rms_few = []

i = 0
spacing = spacings_nostiff[i]
while spacing<25:
    spacing = spacings_nostiff[i]
    spacings_fraction.append(spacing)   
    Dznew = Dzs_nostiff[i]/Dparallel_nostiff[i]
    # Ds, stdv
    Dzs_stdv = abs(Dznew)*np.sqrt((Dzs_stdv_nostiff[i]/Dzs_nostiff[i])**2+(Dparallel_stdv_nostiff[i]/Dparallel_nostiff[i])**2)
    Dzs_stdv_diff = np.sqrt((Dzs_stdv_nostiff[i])**2+(Dparallel_stdv_nostiff[i])**2)
    Dzs_fraction.append(Dznew)
    Dzs_rms_fraction.append(Dzs_stdv)
    Dzs_diff.append(Dzs_nostiff[i]-Dparallel_nostiff[i])
    Dzs_rms_diff.append(Dzs_stdv_diff)
    if spacing>1.5:
        spacings_few.append(spacing)
        Dzs_fraction_few.append(Dznew)
        Dzs_fraction_rms_few.append(Dzs_stdv)
        ones.append(1)
    i+=1

ones = np.array(ones)
Dzs_diff = np.array(Dzs_diff)
spacings_few = np.array(spacings_few)
Dzs_fraction = np.array(Dzs_fraction)
Dzs_rms_diff = np.array(Dzs_rms_diff)
Dzs_fraction_few = np.array(Dzs_fraction_few)
Dzs_rms_fraction = np.array(Dzs_rms_fraction)
spacings_fraction = np.array(spacings_fraction)
Dzs_fraction_few_nostiff = np.array(Dzs_fraction_few)
Dzs_fraction_few_nostiff_rms = np.array(Dzs_fraction_rms_few)

Dzs_fraction_few_nostiff_prms = Dzs_fraction_few_nostiff+Dzs_fraction_few_nostiff_rms
Dzs_fraction_few_nostiff_mrms = Dzs_fraction_few_nostiff-Dzs_fraction_few_nostiff_rms

coeffs, covs = curve_fit(myexponential,spacings_few, Dzs_fraction_few) 
A_nostiff = coeffs[0]
b_nostiff = coeffs[1]
rms_A_nostiff = np.sqrt(covs[0,0])
rms_b_nostiff = np.sqrt(covs[1,1])

dstart = spacings_few[0]
dend   = spacings_few[-1]
spacings_fit = np.linspace(dstart,dend,100)
fit_nostiff  = myexponential(spacings_fit,A_nostiff,b_nostiff)

# Shifted:
coeffs, covs = curve_fit(myexponential_shifted,spacings_few, Dzs_fraction_few)
A_nostiff_shifted = coeffs[0]
b_nostiff_shifted = coeffs[1]
rms_A_nostiff_shifted = np.sqrt(covs[0,0])
rms_b_nostiff_shifted = np.sqrt(covs[1,1])

fit_nostiff_shifted  = myexponential_shifted(spacings_fit,A_nostiff_shifted,b_nostiff_shifted)

'''
plt.figure(figsize=(6,5))
plt.fill_between(spacings_few, Dzs_fraction_few_nostiff_prms, Dzs_fraction_few_nostiff_mrms,color='lightcoral',alpha=0.2)
plt.plot(spacings_few, Dzs_fraction_few_nostiff, '-o',color='lightcoral',label='$D_{\perp, \mathregular{no} \mathregular{stiff.}}/D_{\parallel, \mathregular{no} \mathregular{stiff.}}$')
plt.plot(spacings_few, ones, '--',color='lightcoral')
plt.plot(spacings_fit,fit_nostiff,'--',label=r'Exponential fit')
#plt.plot(spacings_fit,fit_nostiff_shifted,'--',label=r'Exp. fit with shift')
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D_{\perp, \mathregular{no} \mathregular{stiff.}}/D_{\parallel, \mathregular{no} \mathregular{stiff.}}$', fontsize=14)
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(6,5))
plt.fill_between(spacings_fraction, Dzs_diff+Dzs_rms_diff, Dzs_diff-Dzs_rms_diff,color='lightcoral',alpha=0.2)
plt.plot(spacings_fraction, Dzs_diff, '-o',color='lightcoral')
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D_{\perp, \mathregular{no stiff.}}/D_{\mathregular{bulk}}-D_{\parallel, \mathregular{no stiff.}}/D_{\mathregular{bulk}}$', fontsize=14)
plt.show()
'''

### Straight, perp vs parallel
spacings_fraction = []
Dzs_fraction      = []
Dzs_diff          = []
Dzs_rms_fraction  = []
Dzs_rms_diff      = []
ones              = []
spacings_few      = []
Dzs_fraction_few  = []
Dzs_fraction_rms_few = []

spacings_plot      = []
Dzs_fraction_plot  = []
Dzs_fraction_rms_plot = []

i = 0
spacing = spacings_f[i]
while spacing<25:
    spacing = spacings_f[i]
    spacings_fraction.append(spacing)   
    Dznew = Dzs_f[i]/Dparallel_f[i]
    # Ds, stdv
    Dzs_stdv = abs(Dznew)*np.sqrt((Dzs_stdv_f[i]/Dzs_f[i])**2+(Dparallel_stdv_f[i]/Dparallel_f[i])**2)
    Dzs_stdv_diff = np.sqrt((Dzs_stdv_f[i])**2+(Dparallel_stdv_f[i])**2)
    Dzs_fraction.append(Dznew)
    Dzs_rms_fraction.append(Dzs_stdv)
    Dzs_diff.append(Dzs_f[i]-Dparallel_f[i])
    Dzs_rms_diff.append(Dzs_stdv_diff)
    if spacing>1.9:
        spacings_plot.append(spacing)
        Dzs_fraction_plot.append(Dznew)
        Dzs_fraction_rms_plot.append(Dzs_stdv)
        ones.append(1)
    if spacing>3:
        spacings_few.append(spacing)
        Dzs_fraction_few.append(Dznew)
        Dzs_fraction_rms_few.append(Dzs_stdv)
    i+=1

Dzs_diff = np.array(Dzs_diff)
spacings_few = np.array(spacings_few)
Dzs_fraction = np.array(Dzs_fraction)
Dzs_rms_diff = np.array(Dzs_rms_diff)
Dzs_fraction_few = np.array(Dzs_fraction_few)
Dzs_rms_fraction = np.array(Dzs_rms_fraction)
spacings_fraction = np.array(spacings_fraction)
Dzs_fraction_few_straight = np.array(Dzs_fraction_plot)
Dzs_fraction_few_straight_rms = np.array(Dzs_fraction_rms_plot)

Dzs_fraction_few_straight_prms = Dzs_fraction_few_straight+Dzs_fraction_few_straight_rms
Dzs_fraction_few_straight_mrms = Dzs_fraction_few_straight-Dzs_fraction_few_straight_rms

coeffs, covs = curve_fit(myexponential,spacings_few, Dzs_fraction_few) 
A_f = coeffs[0]
b_f = coeffs[1]
rms_A_f = np.sqrt(covs[0,0])
rms_b_f = np.sqrt(covs[1,1])

dstart = spacings_few[0]
dend   = spacings_few[-1]
spacings_fit = np.linspace(dstart,dend,100)
fit_f        = myexponential(spacings_fit,A_f,b_f)

# Shifted
coeffs, covs = curve_fit(myexponential_shifted,spacings_few, Dzs_fraction_few) 
A_f_shifted = coeffs[0]
b_f_shifted = coeffs[1]
rms_A_f_shifted = np.sqrt(covs[0,0])
rms_b_f_shifted = np.sqrt(covs[1,1])

fit_f_shifted        = myexponential_shifted(spacings_fit,A_f_shifted,b_f_shifted)

ones = np.zeros(len(spacings_plot))+1

'''
plt.figure(figsize=(6,5))
plt.fill_between(spacings_plot, Dzs_fraction_few_straight_prms, Dzs_fraction_few_straight_mrms,color='lightcoral',alpha=0.2)
plt.plot(spacings_plot, Dzs_fraction_few_straight, '-o',color='lightcoral',label='$D_{\perp, \mathregular{straight}}/D_{\parallel, \mathregular{straight}}$')
plt.plot(spacings_plot, ones, '--',color='lightcoral')
plt.plot(spacings_fit,fit_f,'--',label=r'Exponential fit')
#plt.plot(spacings_fit,fit_f_shifted,'--',label=r'Exp. fit with shift')
plt.xlabel(r'$d$ (nm)', fontsize=15)
plt.ylabel(r'$D_{\perp, \mathregular{straight}}/D_{\parallel, \mathregular{straight}}$', fontsize=15)
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(6,5))
plt.fill_between(spacings_fraction, Dzs_diff+Dzs_rms_diff, Dzs_diff-Dzs_rms_diff,color='lightcoral',alpha=0.2)
plt.plot(spacings_fraction, Dzs_diff, '-o',color='lightcoral')
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D_{\perp, \mathregular{straight}}/D_{\mathregular{bulk}}-D_{\parallel, \mathregular{straight}}/D_{\mathregular{bulk}}$', fontsize=14)
'''


print("A_dyn:",A_dyn, ", b_dyn:",b_dyn)
print("rms_A_dyn:",rms_A_dyn, ", rms_b_dyn:",rms_b_dyn)
print("----------------------------------")
print("A_stat:",A_stat, ", b_stat:",b_stat)
print("rms_A_stat:",rms_A_stat, ", rms_b_stat:",rms_b_stat)
print("----------------------------------")
print("A_nostiff:",A_nostiff, ", b_nostiff:",b_nostiff)
print("rms_A_nostiff:",rms_A_nostiff, ", rms_b_nostiff:",rms_b_nostiff)
print("----------------------------------")
print("A_f:",A_f, ", b_f:",b_f)
print("rms_A_f:",rms_A_f, ", rms_b_f:",rms_b_f)

print('\n############## SHIFTED ####################\n')
print("A_dyn_shifted:",A_dyn_shifted, ", b_dyn_shifted:",b_dyn_shifted)
print("rms_A_dyn_shifted:",rms_A_dyn_shifted, ", rms_b_dyn_shifted:",rms_b_dyn_shifted)
print("----------------------------------")
print("A_stat_shifted:",A_stat_shifted, ", b_stat_shifted:",b_stat_shifted)
print("rms_A_stat_shifted:",rms_A_stat_shifted, ", rms_b_stat_shifted:",rms_b_stat_shifted)
print("----------------------------------")
print("A_nostiff_shifted:",A_nostiff_shifted, ", b_nostiff_shifted:",b_nostiff_shifted)
print("rms_A_nostiff_shifted:",rms_A_nostiff_shifted, ", rms_b_nostiff_shifted:",rms_b_nostiff_shifted)
print("----------------------------------")
print("A_f_shifted:",A_f_shifted, ", b_f_shifted:",b_f_shifted)
print("rms_A_f_shifted:",rms_A_f_shifted, ", rms_b_f_shifted:",rms_b_f_shifted)


## Extra plotting
# Stat vs dyn
plt.figure(figsize=(8,5))
ax = plt.subplot(111)    
ax.plot(spacings_dyn, Dzs_dyn, '-o' , color='b',label=r'$D_\perp$, dyn.')
ax.plot(spacings_stat, Dzs_stat, '-s', color='c', label=r'$D_\perp$, stat.')
ax.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='b',alpha=0.2)
ax.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='c', alpha=0.2)
if moresigmas==True:
    plt.xlabel(r'$d/\sigma_b$')
else:
    plt.xlabel(r'$d$ (nm)')
plt.ylabel(r'$D/D_{\mathregular{bulk}}$')
plt.legend(loc='lower right')

plt.figure(figsize=(8,5))
ax = plt.subplot(111)    
ax.plot(spacings_dyn, Dparallel_dyn, '-*', color='g', label=r'$D_\parallel$, dyn.')
ax.plot(spacings_stat, Dparallel_stat, '-d', color='limegreen',label=r'$D_\parallel$, stat.')
ax.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
ax.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen',alpha=0.2)
if moresigmas==True:
    plt.xlabel(r'$d/\sigma_b$')
else:
    plt.xlabel(r'$d$ (nm)')
plt.ylabel(r'$D/D_{\mathregular{bulk}}$')
plt.legend(loc='lower right')




plt.show()
