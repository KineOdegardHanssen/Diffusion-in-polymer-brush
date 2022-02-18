import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
import numpy as np
import random
import math
import time
import os
import glob


# change the default font family
plt.rcParams.update({'font.family':'Arial'})

set_sigma_errors = False #True
data_sigma_errors = True # False

thismaxfev = 10000

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

# Models:

def ordered_packings(X,sigma):
    d,Lzs = X
    phi = 1 - (4./3*np.pi*(sigma**3)*101./(d**2*Lzs))
    tau = (3-phi)/2.
    return 1./tau

def hyp_rev(X,sigma):
    d,Lzs = X
    phi = 1 - (4./3*np.pi*(sigma**3)*101./(d**2*Lzs))
    tau = 2-phi
    return 1./tau

def het_cat(X,sigma):
    d,Lzs = X
    phi = 1 - (4./3*np.pi*(sigma**3)*101./(d**2*Lzs))
    tau = phi/(1-(1-phi)**(1/3.))
    return 1./tau

def catex_resin_membr(X,sigma):
    phi = 1 - (4./3*np.pi*(sigma**3)*101./(d**2*Lzs))
    tau = ((2.-phi)/phi)**2
    return 1./tau

def mymodel1(X,k):
    d,Lzs = X
    # Assumes probability of neighbor being occupied when you're occupied: (1-k)*phi
    phi = 1 - (4./3*np.pi*(0.5**3)*101./(d**2*Lzs))
    teller = 1-phi*k
    nevner = phi*(1-k)
    tau = (teller/nevner)**2
    return 1./tau

def mymodel2(X,k):
    d,Lzs = X
    # Assumes probability of neighbor being occupied when you're occupied: vp*k
    phi = 1 - (4./3*np.pi*(0.5**3)*101./(d**2*Lzs))
    teller = 1+(1-phi)*(1-k)
    nevner = 1-(1-phi)*k
    tau = (teller/nevner)**2
    return 1./tau

def mymodel3(X,k):
    d,Lzs = X
    # Assumes probability of neighbor being occupied when you're occupied: k
    phi = 1 - (4./3*np.pi*(0.5**3)*101./(d**2*Lzs)) # sigma fixed
    teller = 1-phi
    nevner = 1-k
    tau = (1+teller/nevner)**2
    return 1./tau

def mymodel4(X,k,f):
    d,Lzs = X
    # Assumes probability of neighbor being occupied when you're occupied: kvp**f
    phi = 1 - (4./3*np.pi*(0.5**3)*101./(d**2*Lzs)) # sigma fixed
    vp = 1-phi
    nevner = 1-k*vp**f
    tau = (1+vp/nevner)**2
    return 1./tau

def mymodel5(X,k,f):
    d,Lzs = X
    # Assumes probability of neighbor being occupied when you're occupied: k+(k-1)vp**f
    phi = 1 - (4./3*np.pi*(0.5**3)/(d**2*Lzs)) # sigma fixed
    vp = 1-phi
    tau = (1+vp/(1-(k+((k-1)*vp**f))))**2
    return 1./tau

def mymodel6(X,k):
    # Assumes probability of neighbor being occupied when you're occupied: kvp**d
    phi = 1 - (4./3*np.pi*(0.5**3)*101./(d**2*Lzs)) # sigma fixed
    vp = 1-phi
    nevner = 1-k*vp**d
    tau = (1+vp/nevner)**2
    return 1./tau

def mymodel7(X,k):
    d,Lzs = X
    # Assumes probability of neighbor being occupied when you're occupied: kvp**(-1./d)
    phi = 1 - (4./3*np.pi*(0.5**3)*101./(d**2*Lzs))
    vp = 1-phi
    nevner = 1-k*vp**(-1./d)
    tau = (1+vp/nevner)**2
    return 1./tau

def mymodel8(X,k):
    d,Lzs = X
    # Assumes probability of neighbor being occupied when you're occupied: k+(k-1)vp
    phi = 1 - (4./3*np.pi*(0.5**3)*101./(d**2*Lzs))
    vp = 1-phi
    tau = (1+vp/(1-(k+((k-1)*vp))))**2
    return 1./tau

def mymodel9(X,k):
    d,Lzs = X
    # Assumes probability of neighbor being occupied when you're occupied: k+(1-k)vp
    phi = 1 - (4./3*np.pi*(0.5**3)*101./(d**2*Lzs))
    vp = 1-phi
    tau = (1+vp/(1-(k+((1-k)*vp))))**2
    return 1./tau

def powerlaw1(d,k,l,n):
    return 1 - (d/l)**n*k

def powerlaw2(d,l,n):
    return 1 - (d/l)**n

def powerlaw3(d,n,k):
    return 1 - d**n*k

def powerlaw4(d,n):
    return 1 - d**n

def powerlaw3_sigmas(d,sigma,n,k):
    return 1 - (d/sigma)**n*k

showplots = True #False

# Input parameters for file selection:
rsphere  = 1.0 # Old parameter

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

ds  = [2,2.5,3,3.5,4,4.5,5,6,7,8,10]
Lzs = []
fitds = np.array(ds)

for i in range(len(ds)):
    folder   = '/Diffusion_bead_near_grid/Spacing'+str(ds[i])+'/damp10_diffseedLgv/Brush/Sigma_bead_' +str(psigma) + '/Nocut/'
    filename = folder+'maxz_az_config1to1000_nocut.txt'
    infile   = open(filename,'r')
    line     = infile.readline()
    words    = line.split()
    Lz       = float(words[0])
    Lzs.append(Lz)

Lzs = np.array(Lzs)

endlocation = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/Nocut/'
if moresigmas==True:
    endlocation = '/Diffusion_bead_near_grid/D_vs_d/Brush/Moresigmas/Nocut/'

basepath_base      = '/Diffusion_staticbrush/'
endlocation_static = basepath_base+'D_vs_d/Nocut/'

bulklocation = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/'

bulkfilename  = bulklocation + 'diffusion_bulk'+filestext
if bulk_cut==True:
    bulkfilename = bulkfilename +'_cut.txt'
else:
    bulkfilename = bulkfilename +'_uncut.txt'

## Files to read
brushfilename_dyn  = endlocation + 'D_vs_d_better_rms_Nestimates10.txt'
brushfilename_stat = endlocation_static + 'D_vs_d_static_better_rms_Nestimates10.txt'

## Files to write to
if big==False:
    ### d
    plotname_d_dyn     = endlocation_static+'D_vs_d_dyn_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid.tiff'
    plotname_d_stat    = endlocation_static+'D_vs_d_stat_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid.tiff'
    plotname_d_f       = endlocation_static+'D_vs_d_f_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid.tiff'
    # perp:
    plotname_d_dyn_perp     = endlocation_static+'D_vs_d_dyn_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_perp_fittomid.tiff'
    plotname_d_stat_perp    = endlocation_static+'D_vs_d_stat_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_perp_fittomid.tiff'
    plotname_d_f_perp       = endlocation_static+'D_vs_d_f_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_perp_fittomid.tiff'
    plotname_cut_d = endlocation_static+'D_vs_d_dyn_vs_stat_vs_RWgeom_cut_norefl_rsphere'+str(rsphere)+'_modelr_findsigma_fittomid.tiff'
else:
    ### d
    plotname_d_dyn     = endlocation_static+'D_vs_d_dyn_big_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid.tiff'
    plotname_d_stat    = endlocation_static+'D_vs_d_stat_big_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid.tiff'
    plotname_d_f       = endlocation_static+'D_vs_d_f_big_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid.tiff'
    # perp:
    plotname_d_dyn_perp     = endlocation_static+'D_vs_d_dyn_big_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_perp_fittomid.tiff'
    plotname_d_stat_perp    = endlocation_static+'D_vs_d_stat_big_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_perp_fittomid.tiff'
    plotname_d_f_perp       = endlocation_static+'D_vs_d_f_big_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_perp_fittomid.tiff'
    plotname_cut_d = endlocation_static+'D_vs_d_dyn_vs_stat_cut_vs_RWgeom_big_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid.tiff'
plotname_panels = endlocation_static+'D_vs_d_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid_panels.tiff'
plotname_panels_all = endlocation_static+'D_vs_d_norefl_rsphere'+str(rsphere)+'_allmodels_modelr_findsigma_fittomid_panels_all.tiff'

# Dynamic sims:
brushfile_dyn = open(brushfilename_dyn, 'r')

lines = brushfile_dyn.readlines()
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

brushfile_dyn.close()

##########

# Static sims:
brushfile_stat = open(brushfilename_stat, 'r')

lines = brushfile_stat.readlines()
N_stat = len(lines)-1

# ds
dens_stat     = np.zeros(N_stat)
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

    d = float(words[0])
    spacings_stat[j] = d
    dens_stat[j] = np.pi/float(d**2)
    DRs_stat[j]  = float(words[1])
    Dzs_stat[j]  = float(words[3])
    Dparallel_stat[j] = float(words[5])
    # Ds, stdv
    DRs_stdv_stat[j] = float(words[2])
    Dzs_stdv_stat[j] = float(words[4])
    Dparallel_stdv_stat[j] = float(words[6])

brushfile_stat.close()

###
#Bulk:

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

# Divide by bulk:
for i in range(N_stat):
    DRnew   = DRs_stat[i]/DRs_bulk
    Dznew   = Dzs_stat[i]/DRs_bulk
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


indices_dyn = []
for i in range(N_dyn):
    if spacings_dyn[i]>=2 and spacings_dyn[i]<11:
        indices_dyn.append(i)
startindex_dyn = indices_dyn[0]
endindex_dyn   = indices_dyn[-1]+1
spacings_dyn   = spacings_dyn[startindex_dyn:endindex_dyn]
Dparallel_dyn  = Dparallel_dyn[startindex_dyn:endindex_dyn]
Dparallel_stdv_dyn  = Dparallel_stdv_dyn[startindex_dyn:endindex_dyn]
Dzs_dyn        = Dzs_dyn[startindex_dyn:endindex_dyn]
Dzs_stdv_dyn   = Dzs_stdv_dyn[startindex_dyn:endindex_dyn]

indices_stat = []
for i in range(N_stat):
    if spacings_stat[i]>=2 and spacings_stat[i]<11:
        indices_stat.append(i)
startindex_stat = indices_stat[0]
endindex_stat   = indices_stat[-1]+1
spacings_stat   = spacings_stat[startindex_stat:endindex_stat]
Dparallel_stat  = Dparallel_stat[startindex_stat:endindex_stat]
Dparallel_stdv_stat  = Dparallel_stdv_stat[startindex_stat:endindex_stat]
Dzs_stat        = Dzs_stat[startindex_stat:endindex_stat]
Dzs_stdv_stat   = Dzs_stdv_stat[startindex_stat:endindex_stat]

# Spacings
spacings_stat = np.array(spacings_stat)
spacings_dyn  = np.array(spacings_dyn)

#### Parameter estimates

cfsigmas = np.zeros(len(Dparallel_stat))+0.1
pardyn_sigmas   = Dparallel_stdv_dyn
perpdyn_sigmas  = Dzs_stdv_dyn
parstat_sigmas  = Dparallel_stdv_stat
perpstat_sigmas = Dzs_stdv_stat
# Ordered packing
if set_sigma_errors==True:
    popt, pcov           = curve_fit(ordered_packings, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(ordered_packings, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, sigma=parstat_sigmas, absolute_sigma=True)
sigma_op_stat        = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_op_stat_rms    = rms_params[0]
if set_sigma_errors==True:
    popt, pcov           = curve_fit(ordered_packings, (spacings_dyn,Lzs), Dparallel_dyn, p0=7, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(ordered_packings, (spacings_dyn,Lzs), Dparallel_dyn, p0=7, maxfev=thismaxfev, sigma=pardyn_sigmas, absolute_sigma=True)
sigma_op_dyn         = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_op_dyn_rms     = rms_params[0]
# Perp
if set_sigma_errors==True:
    popt, pcov             = curve_fit(ordered_packings, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov             = curve_fit(ordered_packings, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, sigma=perpstat_sigmas, absolute_sigma=True)
sigma_op_stat_perp     = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_op_stat_perp_rms = rms_params[0]
if set_sigma_errors==True:
    popt, pcov             = curve_fit(ordered_packings, (spacings_dyn,Lzs), Dzs_dyn, p0=7, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov             = curve_fit(ordered_packings, (spacings_dyn,Lzs), Dzs_dyn, p0=7, maxfev=thismaxfev, sigma=perpdyn_sigmas, absolute_sigma=True)
sigma_op_dyn_perp      = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_op_dyn_perp_rms  = rms_params[0]

# Ds
Ds_op_stat = ordered_packings((fitds,Lzs),sigma_op_stat)
Ds_op_dyn  = ordered_packings((fitds,Lzs),sigma_op_dyn)
Ds_op_stat_perp = ordered_packings((fitds,Lzs),sigma_op_stat_perp)
Ds_op_dyn_perp  = ordered_packings((fitds,Lzs),sigma_op_dyn_perp)

# Hyperbola of revolution
if set_sigma_errors==True:
    popt, pcov           = curve_fit(hyp_rev, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(hyp_rev, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, sigma=parstat_sigmas, absolute_sigma=True)
sigma_hr_stat        = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_hr_stat_rms    = rms_params[0]
if set_sigma_errors==True:
    popt, pcov           = curve_fit(hyp_rev, (spacings_dyn,Lzs), Dparallel_dyn, p0=5, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(hyp_rev, (spacings_dyn,Lzs), Dparallel_dyn, p0=5, maxfev=thismaxfev, sigma=pardyn_sigmas, absolute_sigma=True)
sigma_hr_dyn           = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_hr_dyn_rms       = rms_params[0]
# Perp
if set_sigma_errors==True:
    popt, pcov             = curve_fit(hyp_rev, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov             = curve_fit(hyp_rev, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, sigma=perpstat_sigmas, absolute_sigma=True)
sigma_hr_stat_perp     = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_hr_stat_perp_rms = rms_params[0]
if set_sigma_errors==True:
    popt, pcov             = curve_fit(hyp_rev, (spacings_dyn,Lzs), Dzs_dyn, p0=5, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov             = curve_fit(hyp_rev, (spacings_dyn,Lzs), Dzs_dyn, p0=5, maxfev=thismaxfev, sigma=perpdyn_sigmas, absolute_sigma=True)
sigma_hr_dyn_perp      = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_hr_dyn_perp_rms  = rms_params[0]

# Ds
Ds_hr_stat = hyp_rev((fitds,Lzs),sigma_hr_stat)
Ds_hr_dyn  = hyp_rev((fitds,Lzs),sigma_hr_dyn)
Ds_hr_stat_perp = hyp_rev((fitds,Lzs),sigma_hr_stat_perp)
Ds_hr_dyn_perp  = hyp_rev((fitds,Lzs),sigma_hr_dyn_perp)


# Heterogeneous catalyst
if set_sigma_errors==True:
    popt, pcov           = curve_fit(het_cat, (spacings_stat,Lzs), Dparallel_stat,maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(het_cat, (spacings_stat,Lzs), Dparallel_stat,maxfev=thismaxfev, sigma=parstat_sigmas, absolute_sigma=True)
sigma_hc_stat        = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_hc_stat_rms    = rms_params[0]
if set_sigma_errors==True:
    popt, pcov           = curve_fit(het_cat, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(het_cat, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, sigma=pardyn_sigmas, absolute_sigma=True)
sigma_hc_dyn         = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_hc_dyn_rms     = rms_params[0]
# Perp
if set_sigma_errors==True:
    popt, pcov           = curve_fit(het_cat, (spacings_stat,Lzs), Dzs_stat,maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(het_cat, (spacings_stat,Lzs), Dzs_stat,maxfev=thismaxfev, sigma=perpstat_sigmas, absolute_sigma=True)
sigma_hc_stat_perp   = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_hc_stat_perp_rms = rms_params[0]
if set_sigma_errors==True:
    popt, pcov             = curve_fit(het_cat, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov             = curve_fit(het_cat, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, sigma=perpdyn_sigmas, absolute_sigma=True)
sigma_hc_dyn_perp      = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_hc_dyn_perp_rms  = rms_params[0]

# Ds
Ds_hc_stat = het_cat((fitds,Lzs),sigma_hc_stat)
Ds_hc_dyn  = het_cat((fitds,Lzs),sigma_hc_dyn)
Ds_hc_stat_perp = het_cat((fitds,Lzs),sigma_hc_stat_perp)
Ds_hc_dyn_perp  = het_cat((fitds,Lzs),sigma_hc_dyn_perp)

# Cation exchange resin membrane
if set_sigma_errors==True:
    popt, pcov           = curve_fit(catex_resin_membr, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(catex_resin_membr, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, sigma=parstat_sigmas, absolute_sigma=True)
sigma_rm_stat        = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_rm_stat_rms    = rms_params[0]
if set_sigma_errors==True:
    popt, pcov           = curve_fit(catex_resin_membr, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(catex_resin_membr, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, sigma=pardyn_sigmas, absolute_sigma=True)
sigma_rm_dyn         = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_rm_dyn_rms     = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov             = curve_fit(catex_resin_membr, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov             = curve_fit(catex_resin_membr, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, sigma=perpstat_sigmas, absolute_sigma=True)
sigma_rm_stat_perp     = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_rm_stat_perp_rms = rms_params[0]
if set_sigma_errors==True:
    popt, pcov             = curve_fit(catex_resin_membr, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov             = curve_fit(catex_resin_membr, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, sigma=perpdyn_sigmas, absolute_sigma=True)
sigma_rm_dyn_perp      = popt[0]
rms_params = np.sqrt(np.diag(pcov))
sigma_rm_dyn_perp_rms  = rms_params[0]

# Ds
Ds_rm_stat = catex_resin_membr((fitds,Lzs),sigma_rm_stat)
Ds_rm_dyn  = catex_resin_membr((fitds,Lzs),sigma_rm_dyn)
Ds_rm_stat_perp = catex_resin_membr((fitds,Lzs),sigma_rm_stat_perp)
Ds_rm_dyn_perp  = catex_resin_membr((fitds,Lzs),sigma_rm_dyn_perp)

# My models:
### mymodel1(d,k)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel1, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel1, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=pardyn_sigmas, absolute_sigma=True)
k_m1_stat        = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m1_stat_stdv   = rms_params[0]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel1, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel1, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=pardyn_sigmas, absolute_sigma=True)
k_m1_dyn         = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m1_dyn_stdv    = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel1, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel1, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=perpstat_sigmas, absolute_sigma=True)
k_m1_stat_perp      = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m1_stat_perp_stdv = rms_params[0]
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel1, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel1, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=perpdyn_sigmas, absolute_sigma=True)
k_m1_dyn_perp       = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m1_dyn_perp_stdv  = rms_params[0]

# Ds
Ds_m1_stat = mymodel1((fitds,Lzs),k_m1_stat)
Ds_m1_dyn  = mymodel1((fitds,Lzs),k_m1_dyn)
Ds_m1_stat_perp = mymodel1((fitds,Lzs),k_m1_stat_perp)
Ds_m1_dyn_perp  = mymodel1((fitds,Lzs),k_m1_dyn_perp)

### mymodel2(d,k)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel2, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,np.inf), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel2, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,np.inf), sigma=parstat_sigmas, absolute_sigma=True)
k_m2_stat        = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m2_stat_stdv   = rms_params[0]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel2, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,np.inf), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel2, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,np.inf), sigma=pardyn_sigmas, absolute_sigma=True)
k_m2_dyn         = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m2_dyn_stdv    = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel2, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,np.inf), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel2, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,np.inf), sigma=perpstat_sigmas, absolute_sigma=True)
k_m2_stat_perp      = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m2_stat_perp_stdv = rms_params[0]
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel2, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,np.inf), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel2, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,np.inf), sigma=perpdyn_sigmas, absolute_sigma=True)
k_m2_dyn_perp       = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m2_dyn_perp_stdv  = rms_params[0]

# Ds
Ds_m2_stat = mymodel2((fitds,Lzs),k_m2_stat)
Ds_m2_dyn  = mymodel2((fitds,Lzs),k_m2_dyn)
Ds_m2_stat_perp = mymodel2((fitds,Lzs),k_m2_stat_perp)
Ds_m2_dyn_perp  = mymodel2((fitds,Lzs),k_m2_dyn_perp)

### mymodel3(d,k)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel3, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel3, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=parstat_sigmas, absolute_sigma=True)
k_m3_stat        = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m3_stat_stdv   = rms_params[0]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel3, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel3, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=pardyn_sigmas, absolute_sigma=True)
k_m3_dyn         = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m3_dyn_stdv    = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel3, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel3, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=perpstat_sigmas, absolute_sigma=True)
k_m3_stat_perp      = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m3_stat_perp_stdv = rms_params[0]
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel3, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel3, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=perpdyn_sigmas, absolute_sigma=True)
k_m3_dyn_perp       = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m3_dyn_perp_stdv  = rms_params[0]

# Ds
Ds_m3_stat = mymodel3((fitds,Lzs),k_m3_stat)
Ds_m3_dyn  = mymodel3((fitds,Lzs),k_m3_dyn)
Ds_m3_stat_perp = mymodel3((fitds,Lzs),k_m3_stat_perp)
Ds_m3_dyn_perp  = mymodel3((fitds,Lzs),k_m3_dyn_perp)


### mymodel4(d,k,f)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel4, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel4, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=parstat_sigmas, absolute_sigma=True)
k_m4_stat        = popt[0]
f_m4_stat        = popt[1]
rms_params       = np.sqrt(np.diag(pcov))
k_m4_stat_stdv   = rms_params[0]
f_m4_stat_stdv   = rms_params[1]
pcov4_statpar    = pcov
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel4, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel4, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=pardyn_sigmas, absolute_sigma=True)
k_m4_dyn         = popt[0]
f_m4_dyn         = popt[1]
rms_params       = np.sqrt(np.diag(pcov))
k_m4_dyn_stdv    = rms_params[0]
f_m4_dyn_stdv    = rms_params[1]
pcov4_dynpar     = pcov
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel4, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel4, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=perpstat_sigmas, absolute_sigma=True)
k_m4_stat_perp      = popt[0]
f_m4_stat_perp      = popt[1]
rms_params          = np.sqrt(np.diag(pcov))
k_m4_stat_perp_stdv = rms_params[0]
f_m4_stat_perp_stdv = rms_params[1]
pcov4_statperp      = pcov
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel4, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel4, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=perpdyn_sigmas, absolute_sigma=True)
k_m4_dyn_perp       = popt[0]
f_m4_dyn_perp       = popt[1]
rms_params          = np.sqrt(np.diag(pcov))
k_m4_dyn_perp_stdv  = rms_params[0]
f_m4_dyn_perp_stdv  = rms_params[1]
pcov4_dynperp       = pcov

# Ds
Ds_m4_stat = mymodel4((fitds,Lzs),k_m4_stat,f_m4_stat)
Ds_m4_dyn  = mymodel4((fitds,Lzs),k_m4_dyn,f_m4_dyn)
Ds_m4_stat_perp = mymodel4((fitds,Lzs),k_m4_stat_perp,f_m4_stat_perp)
Ds_m4_dyn_perp  = mymodel4((fitds,Lzs),k_m4_dyn_perp,f_m4_dyn_perp)



### mymodel5(d,k,f)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel5, (spacings_stat,Lzs), Dparallel_stat, maxfev=1000000, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel5, (spacings_stat,Lzs), Dparallel_stat, maxfev=1000000, sigma=parstat_sigmas, absolute_sigma=True)
k_m5_stat        = popt[0]
f_m5_stat        = popt[1]
rms_params       = np.sqrt(np.diag(pcov))
k_m5_stat_stdv   = rms_params[0]
f_m5_stat_stdv   = rms_params[1]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel5, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=1000000, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel5, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=1000000, sigma=pardyn_sigmas, absolute_sigma=True)
k_m5_dyn         = popt[0]
f_m5_dyn         = popt[1]
rms_params       = np.sqrt(np.diag(pcov))
k_m5_dyn_stdv    = rms_params[0]
f_m5_dyn_stdv    = rms_params[1]
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel5, (spacings_stat,Lzs), Dzs_stat, maxfev=1000000, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel5, (spacings_stat,Lzs), Dzs_stat, maxfev=1000000, sigma=perpstat_sigmas, absolute_sigma=True)
k_m5_stat_perp      = popt[0]
f_m5_stat_perp      = popt[1]
rms_params          = np.sqrt(np.diag(pcov))
k_m5_stat_perp_stdv = rms_params[0]
f_m5_stat_perp_stdv = rms_params[1]
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel5, (spacings_dyn,Lzs), Dzs_dyn, maxfev=1000000, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel5, (spacings_dyn,Lzs), Dzs_dyn, maxfev=1000000, sigma=perpdyn_sigmas, absolute_sigma=True)
k_m5_dyn_perp       = popt[0]
f_m5_dyn_perp       = popt[1]
rms_params          = np.sqrt(np.diag(pcov))
k_m5_dyn_perp_stdv  = rms_params[0]
f_m5_dyn_perp_stdv  = rms_params[1]

# Ds
Ds_m5_stat = mymodel5((fitds,Lzs),k_m5_stat,f_m5_stat)
Ds_m5_dyn  = mymodel5((fitds,Lzs),k_m5_dyn,f_m5_dyn)
Ds_m5_stat_perp = mymodel5((fitds,Lzs),k_m5_stat_perp,f_m5_stat_perp)
Ds_m5_dyn_perp  = mymodel5((fitds,Lzs),k_m5_dyn_perp,f_m5_dyn_perp)

### mymodel6(d,k)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel6, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel6, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, sigma=parstat_sigmas, absolute_sigma=True)
k_m6_stat        = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m6_stat_stdv   = rms_params[0]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel6, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel6, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, sigma=pardyn_sigmas, absolute_sigma=True)
k_m6_dyn         = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m6_dyn_stdv    = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel6, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel6, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, sigma=perpstat_sigmas, absolute_sigma=True)
k_m6_stat_perp      = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m6_stat_perp_stdv = rms_params[0]
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel6, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel6, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, sigma=perpstat_sigmas, absolute_sigma=True)
k_m6_dyn_perp       = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m6_dyn_perp_stdv  = rms_params[0]

# Ds
Ds_m6_stat = mymodel6((fitds,Lzs),k_m6_stat)
Ds_m6_dyn  = mymodel6((fitds,Lzs),k_m6_dyn)
Ds_m6_stat_perp = mymodel6((fitds,Lzs),k_m6_stat_perp)
Ds_m6_dyn_perp  = mymodel6((fitds,Lzs),k_m6_dyn_perp)

### mymodel7(d,k)
if set_sigma_errors==True:
    popt, pcov         = curve_fit(mymodel7, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,np.inf), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov         = curve_fit(mymodel7, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,np.inf), sigma=parstat_sigmas, absolute_sigma=True)
rms_params = np.sqrt(np.diag(pcov))
k_m7_stat          = popt[0]
k_m7_stat_stdv     = rms_params[0]
if set_sigma_errors==True:
    popt, pcov         = curve_fit(mymodel7, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,np.inf), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov         = curve_fit(mymodel7, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,np.inf), sigma=pardyn_sigmas, absolute_sigma=True)
rms_params = np.sqrt(np.diag(pcov))
k_m7_dyn           = popt[0]
k_m7_dyn_stdv      = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov              = curve_fit(mymodel7, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,np.inf), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov              = curve_fit(mymodel7, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,np.inf), sigma=perpstat_sigmas, absolute_sigma=True)
rms_params = np.sqrt(np.diag(pcov))
k_m7_stat_perp          = popt[0]
k_m7_stat_perp_stdv     = rms_params[0]
if set_sigma_errors==True:
    popt, pcov              = curve_fit(mymodel7, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,np.inf), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov              = curve_fit(mymodel7, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,np.inf), sigma=perpdyn_sigmas, absolute_sigma=True)
rms_params = np.sqrt(np.diag(pcov))
k_m7_dyn_perp           = popt[0]
k_m7_dyn_perp_stdv      = rms_params[0]

# Ds
Ds_m7_stat = mymodel7((fitds,Lzs),k_m7_stat)
Ds_m7_dyn  = mymodel7((fitds,Lzs),k_m7_dyn)
Ds_m7_stat_perp = mymodel7((fitds,Lzs),k_m7_stat_perp)
Ds_m7_dyn_perp  = mymodel7((fitds,Lzs),k_m7_dyn_perp)


### mymodel8(d,k)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel8, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel8, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=parstat_sigmas, absolute_sigma=True)
k_m8_stat        = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m8_stat_stdv   = rms_params[0]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel8, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel8, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=pardyn_sigmas, absolute_sigma=True)
k_m8_dyn         = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m8_dyn_stdv    = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel8, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel8, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=perpstat_sigmas, absolute_sigma=True)
k_m8_stat_perp      = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m8_stat_perp_stdv = rms_params[0]
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel8, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel8, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=perpdyn_sigmas, absolute_sigma=True)
k_m8_dyn_perp       = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m8_dyn_perp_stdv  = rms_params[0]

# Ds
Ds_m8_stat = mymodel8((fitds,Lzs),k_m8_stat)
Ds_m8_dyn  = mymodel8((fitds,Lzs),k_m8_dyn)
Ds_m8_stat_perp = mymodel8((fitds,Lzs),k_m8_stat_perp)
Ds_m8_dyn_perp  = mymodel8((fitds,Lzs),k_m8_dyn_perp)

### mymodel9(d,k)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel9, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel9, (spacings_stat,Lzs), Dparallel_stat, maxfev=thismaxfev, bounds=(0,1), sigma=parstat_sigmas, absolute_sigma=True)
k_m9_stat        = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m9_stat_stdv   = rms_params[0]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel9, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(mymodel9, (spacings_dyn,Lzs), Dparallel_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=pardyn_sigmas, absolute_sigma=True)
k_m9_dyn         = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
k_m9_dyn_stdv    = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel9, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel9, (spacings_stat,Lzs), Dzs_stat, maxfev=thismaxfev, bounds=(0,1), sigma=perpstat_sigmas, absolute_sigma=True)
k_m9_stat_perp      = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m9_stat_perp_stdv = rms_params[0]
if set_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel9, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(mymodel9, (spacings_dyn,Lzs), Dzs_dyn, maxfev=thismaxfev, bounds=(0,1), sigma=perpdyn_sigmas, absolute_sigma=True)
k_m9_dyn_perp       = popt[0]
rms_params          = np.sqrt(np.diag(pcov))
k_m9_dyn_perp_stdv  = rms_params[0]

# Ds
Ds_m9_stat = mymodel9((fitds,Lzs),k_m9_stat)
Ds_m9_dyn  = mymodel9((fitds,Lzs),k_m9_dyn)
Ds_m9_stat_perp = mymodel9((fitds,Lzs),k_m9_stat_perp)
Ds_m9_dyn_perp  = mymodel9((fitds,Lzs),k_m9_dyn_perp)

##### powerlaw3(d,n,k)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(powerlaw3, spacings_stat, Dparallel_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(powerlaw3, spacings_stat, Dparallel_stat, maxfev=thismaxfev, sigma=parstat_sigmas, absolute_sigma=True)
pcovp3_statpar   = pcov
n_p3_stat        = popt[0]
k_p3_stat        = popt[1]
rms_params       = np.sqrt(np.diag(pcov))
n_p3_stat_stdv   = rms_params[0]
k_p3_stat_stdv   = rms_params[1]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(powerlaw3, spacings_dyn, Dparallel_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(powerlaw3, spacings_dyn, Dparallel_dyn, maxfev=thismaxfev, sigma=pardyn_sigmas, absolute_sigma=True)
pcovp3_dynpar    = pcov
n_p3_dyn         = popt[0]
k_p3_dyn         = popt[1]
rms_params       = np.sqrt(np.diag(pcov))
n_p3_dyn_stdv    = rms_params[0]
k_p3_dyn_stdv    = rms_params[1]
#Perp
if set_sigma_errors==True:
    popt, pcov            = curve_fit(powerlaw3, spacings_stat, Dzs_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov            = curve_fit(powerlaw3, spacings_stat, Dzs_stat, maxfev=thismaxfev, sigma=perpstat_sigmas, absolute_sigma=True)
pcovp3_statperp       = pcov
n_p3_stat_perp        = popt[0]
k_p3_stat_perp        = popt[1]
rms_params            = np.sqrt(np.diag(pcov))
n_p3_stat_perp_stdv   = rms_params[0]
k_p3_stat_perp_stdv   = rms_params[1]
if set_sigma_errors==True:
    popt, pcov            = curve_fit(powerlaw3, spacings_dyn, Dzs_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov            = curve_fit(powerlaw3, spacings_dyn, Dzs_dyn, maxfev=thismaxfev, sigma=perpdyn_sigmas, absolute_sigma=True)
pcovp3_dynperp        = pcov
n_p3_dyn_perp         = popt[0]
k_p3_dyn_perp         = popt[1]
rms_params            = np.sqrt(np.diag(pcov))
n_p3_dyn_perp_stdv    = rms_params[0]
k_p3_dyn_perp_stdv    = rms_params[1]

# Ds
Ds_p3_stat = powerlaw3(fitds,n_p3_stat,k_p3_stat)
Ds_p3_dyn  = powerlaw3(fitds,n_p3_dyn,k_p3_dyn)
Ds_p3_stat_perp = powerlaw3(fitds,n_p3_stat_perp,k_p3_stat_perp)
Ds_p3_dyn_perp  = powerlaw3(fitds,n_p3_dyn_perp,k_p3_dyn_perp)


##### powerlaw4(d,n)
if set_sigma_errors==True:
    popt, pcov       = curve_fit(powerlaw4, spacings_stat, Dparallel_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(powerlaw4, spacings_stat, Dparallel_stat, maxfev=thismaxfev, sigma=parstat_sigmas, absolute_sigma=True)
n_p4_stat        = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
n_p4_stat_stdv   = rms_params[0]
if set_sigma_errors==True:
    popt, pcov       = curve_fit(powerlaw4, spacings_dyn, Dparallel_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov       = curve_fit(powerlaw4, spacings_dyn, Dparallel_dyn, maxfev=thismaxfev, sigma=pardyn_sigmas, absolute_sigma=True)
n_p4_dyn         = popt[0]
rms_params       = np.sqrt(np.diag(pcov))
n_p4_dyn_stdv    = rms_params[0]
#Perp
if set_sigma_errors==True:
    popt, pcov          = curve_fit(powerlaw4, spacings_stat, Dzs_stat, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov          = curve_fit(powerlaw4, spacings_stat, Dzs_stat, maxfev=thismaxfev, sigma=perpstat_sigmas, absolute_sigma=True)
n_p4_stat_perp       = popt[0]
rms_params           = np.sqrt(np.diag(pcov))
n_p4_stat_perp_stdv  = rms_params[0]
if set_sigma_errors==True:
    popt, pcov           = curve_fit(powerlaw4, spacings_dyn, Dzs_dyn, maxfev=thismaxfev, sigma=cfsigmas, absolute_sigma=True)
if data_sigma_errors==True:
    popt, pcov           = curve_fit(powerlaw4, spacings_dyn, Dzs_dyn, maxfev=thismaxfev, sigma=perpdyn_sigmas, absolute_sigma=True)
n_p4_dyn_perp        = popt[0]
rms_params           = np.sqrt(np.diag(pcov))
n_p4_dyn_perp_stdv   = rms_params[0]

# Ds
Ds_p4_stat = powerlaw4(fitds,n_p4_stat)
Ds_p4_dyn  = powerlaw4(fitds,n_p4_dyn)
Ds_p4_stat_perp = powerlaw4(fitds,n_p4_stat_perp)
Ds_p4_dyn_perp  = powerlaw4(fitds,n_p4_dyn_perp)



#### Plotting:
plt.figure(figsize=(14,5),dpi=300)
ax = plt.subplot(111)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.plot(spacings_dyn, Dparallel_dyn, color='g', label=r'$D_\parallel$, dyn.')
ax.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
ax.plot(fitds,Ds_op_dyn, '--', label=r'Ordered packings') # I have most values for the dynamic brush
ax.plot(fitds,Ds_hr_dyn, '-.', label=r'Hyperbola of revolution') # 2
ax.plot(fitds,Ds_hc_dyn, ':', label=r'Heterogeneous catalyst')
ax.plot(fitds,Ds_m1_dyn, '-.', label=r'Custom model 1')
ax.plot(fitds,Ds_m2_dyn, '-.', label=r'Custom model 2')
ax.plot(fitds,Ds_m3_dyn, '-.', label=r'Custom model 3')
ax.plot(fitds,Ds_m4_dyn, '-.', label=r'Custom model 4')
ax.plot(fitds,Ds_m5_dyn, '-.', label=r'Custom model 5')
ax.plot(fitds,Ds_m6_dyn, '-.', label=r'Custom model 6')
ax.plot(fitds,Ds_m7_dyn, '-.', label=r'Custom model 7')
ax.plot(fitds,Ds_m8_dyn, '-.', label=r'Custom model 8')
ax.plot(fitds,Ds_m9_dyn, ':', label=r'Custom model 9')
ax.plot(fitds,Ds_p3_dyn, '--', label=r'Power law')
ax.plot(fitds,Ds_p4_dyn, '-.', label=r'Power law 2')
line1, = ax.plot(fitds,Ds_rm_dyn, '--,', label=r'Cation-exchange resin membrane')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=14)
ax.yaxis.get_offset_text().set_fontsize(14)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
plt.savefig(plotname_d_dyn)

plt.figure(figsize=(14,5),dpi=300)
ax = plt.subplot(111)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.plot(spacings_stat, Dparallel_stat, color='limegreen',label=r'$D_\parallel$, stat.')
ax.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='limegreen', alpha=0.2)
ax.plot(fitds,Ds_op_stat, '--', label=r'Ordered packings')
ax.plot(fitds,Ds_hr_stat, '-.', label=r'Hyperbola of revolution') # 2
ax.plot(fitds,Ds_hc_stat, ':', label=r'Heterogeneous catalyst')
ax.plot(fitds,Ds_m1_stat, '-.', label=r'Custom model 1')
ax.plot(fitds,Ds_m2_stat, '-.', label=r'Custom model 2')
ax.plot(fitds,Ds_m3_stat, '-.', label=r'Custom model 3')
ax.plot(fitds,Ds_m4_stat, '-.', label=r'Custom model 4')
ax.plot(fitds,Ds_m5_stat, '-.', label=r'Custom model 5')
ax.plot(fitds,Ds_m6_stat, '-.', label=r'Custom model 6')
ax.plot(fitds,Ds_m7_stat, '-.', label=r'Custom model 7')
ax.plot(fitds,Ds_m8_stat, '-.', label=r'Custom model 8')
ax.plot(fitds,Ds_m9_stat, ':', label=r'Custom model 9')
ax.plot(fitds,Ds_p3_stat, '--', label=r'Power law')
ax.plot(fitds,Ds_p4_stat, '-.', label=r'Power law 2')
line1, = ax.plot(fitds,Ds_rm_stat, '--,', label=r'Cation-exchange resin membrane')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=14)
ax.yaxis.get_offset_text().set_fontsize(14)
#plt.legend(loc='lower right')
#plt.tight_layout()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
plt.savefig(plotname_d_stat)

plt.figure(figsize=(14,5),dpi=300)
ax = plt.subplot(111)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.plot(spacings_dyn, Dzs_dyn, color='g', label=r'$D_\perp$, dyn.')
ax.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='g', alpha=0.2)
ax.plot(fitds,Ds_op_dyn_perp, '--', label=r'Ordered packings')
ax.plot(fitds,Ds_hr_dyn_perp, '-.', label=r'Hyperbola of revolution') # 2
ax.plot(fitds,Ds_hc_dyn_perp, ':', label=r'Heterogeneous catalyst')
ax.plot(fitds,Ds_m1_dyn_perp, '-.', label=r'Custom model 1')
ax.plot(fitds,Ds_m2_dyn_perp, '-.', label=r'Custom model 2')
ax.plot(fitds,Ds_m3_dyn_perp, '-.', label=r'Custom model 3')
ax.plot(fitds,Ds_m4_dyn_perp, '-.', label=r'Custom model 4')
ax.plot(fitds,Ds_m5_dyn_perp, '-.', label=r'Custom model 5')
ax.plot(fitds,Ds_m6_dyn_perp, '-.', label=r'Custom model 6')
ax.plot(fitds,Ds_m7_dyn_perp, '-.', label=r'Custom model 7')
ax.plot(fitds,Ds_m8_dyn_perp, '-.', label=r'Custom model 8')
ax.plot(fitds,Ds_m9_dyn_perp, ':', label=r'Custom model 9')
ax.plot(fitds,Ds_p3_dyn_perp, '--', label=r'Power law')
ax.plot(fitds,Ds_p4_dyn_perp, '-.', label=r'Power law 2')
line1, = ax.plot(fitds,Ds_rm_dyn_perp, '--,', label=r'Cation-exchange resin membrane')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=14)
ax.yaxis.get_offset_text().set_fontsize(14)
#plt.tight_layout()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
plt.savefig(plotname_d_dyn_perp)


plt.figure(figsize=(14,5),dpi=300)
ax = plt.subplot(111)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.plot(spacings_stat, Dzs_stat, color='g', label=r'$D_\perp$, stat.')
ax.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='g', alpha=0.2)
ax.plot(fitds,Ds_op_stat_perp, '--', label=r'Ordered packings')
ax.plot(fitds,Ds_hr_stat_perp, '-.', label=r'Hyperbola of revolution')
ax.plot(fitds,Ds_hc_stat_perp, ':', label=r'Heterogeneous catalyst')
ax.plot(fitds,Ds_m1_stat_perp, '-.', label=r'Custom model 1')
ax.plot(fitds,Ds_m2_stat_perp, '-.', label=r'Custom model 2')
ax.plot(fitds,Ds_m3_stat_perp, '-.', label=r'Custom model 3')
ax.plot(fitds,Ds_m4_stat_perp, '-.', label=r'Custom model 4')
ax.plot(fitds,Ds_m5_stat_perp, '-.', label=r'Custom model 5')
ax.plot(fitds,Ds_m6_stat_perp, '-.', label=r'Custom model 6')
ax.plot(fitds,Ds_m7_stat_perp, '-.', label=r'Custom model 7')
ax.plot(fitds,Ds_m8_stat_perp, '-.', label=r'Custom model 8')
ax.plot(fitds,Ds_m9_stat_perp, ':', label=r'Custom model 9')
ax.plot(fitds,Ds_p3_stat_perp, '--', label=r'Power law')
ax.plot(fitds,Ds_p4_stat_perp, '-.', label=r'Power law 2')
line1, = ax.plot(fitds,Ds_rm_stat_perp, '--,', label=r'Cation-exchange resin membrane')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
plt.xlabel(r'$d$ (nm)', fontsize=14)
plt.ylabel(r'$D/D_{\mathregular{bulk}}$', fontsize=14)
ax.yaxis.get_offset_text().set_fontsize(14)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
plt.savefig(plotname_d_stat_perp)

################################################################################
# Panel plot
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(15,4.2),dpi=300)
ax1.set_title(r'Dynamic')
ax2.set_title(r'Static')

ax1.set_title(r'A',loc='left')
ax2.set_title(r'B',loc='left')
ax3.set_title(r'C',loc='left')
ax4.set_title(r'D',loc='left')
# Plot 1:
ax1.set_ylabel(r'$D_\parallel/D_{\mathregular{bulk}}$', fontsize=12)
ax3.set_ylabel(r'$D_\perp/D_{\mathregular{bulk}}$', fontsize=12)
ax3.set_xlabel(r'$d$ (nm)', fontsize=12)
ax4.set_xlabel(r'$d$ (nm)', fontsize=12)
ax.yaxis.get_offset_text().set_fontsize(12)
ax1.plot(spacings_dyn, Dparallel_dyn, color='g', label=r'$D$')
ax1.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
ax1.plot(fitds,Ds_op_dyn, '--', label=r'Ordered packings')
ax1.plot(fitds,Ds_hr_dyn, '-.', label=r'Hyperbola of revolution')
ax1.plot(fitds,Ds_hc_dyn, ':', label=r'Heterogeneous catalyst')
ax1.plot(fitds,Ds_m1_dyn, '-.', label=r'Custom model 1')
ax1.plot(fitds,Ds_m2_dyn, '-.', label=r'Custom model 2')
ax1.plot(fitds,Ds_m3_dyn, '-.', label=r'Custom model 3')
ax1.plot(fitds,Ds_m4_dyn, '-.', label=r'Custom model 4')
ax1.plot(fitds,Ds_m5_dyn, '-.', label=r'Custom model 5')
ax1.plot(fitds,Ds_m6_dyn, '-.', label=r'Custom model 6')
ax1.plot(fitds,Ds_m7_dyn, '-.', label=r'Custom model 7')
ax1.plot(fitds,Ds_m8_dyn, '-.', label=r'Custom model 8')
ax1.plot(fitds,Ds_m9_dyn, ':', label=r'Custom model 9')
ax.plot(fitds,Ds_p3_dyn, '--', label=r'Power law')
line1, = ax1.plot(fitds,Ds_rm_dyn, '--,', label=r'Cation-exchange resin membrane')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

# Plot 2:
ax2.plot(spacings_stat, Dparallel_stat, color='g')
ax2.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='g', alpha=0.2)
ax2.plot(fitds,Ds_op_stat, '--', label=r'Ordered packings')
ax2.plot(fitds,Ds_hr_stat, '-.', label=r'Hyperbola of revolution')
ax2.plot(fitds,Ds_hc_stat, ':', label=r'Heterogeneous catalyst')
ax2.plot(fitds,Ds_m1_stat, '-.', label=r'Custom model 1')
ax2.plot(fitds,Ds_m2_stat, '-.', label=r'Custom model 2')
ax2.plot(fitds,Ds_m3_stat, '-.', label=r'Custom model 3')
ax2.plot(fitds,Ds_m4_stat, '-.', label=r'Custom model 4')
ax2.plot(fitds,Ds_m5_stat, '-.', label=r'Custom model 5')
ax2.plot(fitds,Ds_m6_stat, '-.', label=r'Custom model 6')
ax2.plot(fitds,Ds_m7_stat, '-.', label=r'Custom model 7')
ax2.plot(fitds,Ds_m8_stat, '-.', label=r'Custom model 8')
ax2.plot(fitds,Ds_m9_stat, ':', label=r'Custom model 9')
ax2.plot(fitds,Ds_p3_stat, '--', label=r'Power law')
line1, = ax2.plot(fitds,Ds_rm_stat, '--,', label=r'Cation-exchange resin membrane')
# Plot 3:
ax3.plot(spacings_dyn, Dzs_dyn, color='g', label=r'$D_\perp$, dyn.')
ax3.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='g', alpha=0.2)
ax3.plot(fitds,Ds_op_dyn_perp, '--', label=r'Ordered packings')
ax3.plot(fitds,Ds_hr_dyn_perp, '-.', label=r'Hyperbola of revolution')
ax3.plot(fitds,Ds_hc_dyn_perp, ':', label=r'Heterogeneous catalyst')
ax3.plot(fitds,Ds_m1_dyn_perp, '-.', label=r'Custom model 1')
ax3.plot(fitds,Ds_m2_dyn_perp, '-.', label=r'Custom model 2')
ax3.plot(fitds,Ds_m3_dyn_perp, '-.', label=r'Custom model 3')
ax3.plot(fitds,Ds_m4_dyn_perp, '-.', label=r'Custom model 4')
ax3.plot(fitds,Ds_m5_dyn_perp, '-.', label=r'Custom model 5')
ax3.plot(fitds,Ds_m6_dyn_perp, '-.', label=r'Custom model 6')
ax3.plot(fitds,Ds_m7_dyn_perp, '-.', label=r'Custom model 7')
ax3.plot(fitds,Ds_m8_dyn_perp, '-.', label=r'Custom model 8')
ax3.plot(fitds,Ds_m9_dyn_perp, ':', label=r'Custom model 9')
ax3.plot(fitds,Ds_p3_dyn_perp, '--', label=r'Power law')
line1, = ax3.plot(fitds,Ds_rm_dyn_perp, '--,', label=r'Cation-exchange resin membrane')
# Plot 4:
ax4.plot(spacings_stat, Dzs_stat, color='g', label=r'$D_\perp$, stat.')
ax4.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='g', alpha=0.2)
ax4.plot(fitds,Ds_op_stat_perp, '--', label=r'Ordered packings') # I have most values for the static brush
ax4.plot(fitds,Ds_hr_stat_perp, '-.', label=r'Hyperbola of revolution') # 2
ax4.plot(fitds,Ds_hc_stat_perp, ':', label=r'Heterogeneous catalyst')
ax4.plot(fitds,Ds_m1_stat_perp, '-.', label=r'Custom model 1')
ax4.plot(fitds,Ds_m2_stat_perp, '-.', label=r'Custom model 2')
ax4.plot(fitds,Ds_m3_stat_perp, '-.', label=r'Custom model 3')
ax4.plot(fitds,Ds_m4_stat_perp, '-.', label=r'Custom model 4')
ax4.plot(fitds,Ds_m5_stat_perp, '-.', label=r'Custom model 5')
ax4.plot(fitds,Ds_m6_stat_perp, '-.', label=r'Custom model 6')
ax4.plot(fitds,Ds_m7_stat_perp, '-.', label=r'Custom model 7')
ax4.plot(fitds,Ds_m8_stat_perp, '-.', label=r'Custom model 8')
ax4.plot(fitds,Ds_m9_stat_perp, ':', label=r'Custom model 9')
ax4.plot(fitds,Ds_p3_stat_perp, '--', label=r'Power law')
line1, = ax4.plot(fitds,Ds_rm_stat_perp, '--,', label=r'Cation-exchange resin membrane')
box = ax4.get_position()
ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig(plotname_panels_all)
################################################################################

# Panel plot
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(10,4.2),dpi=300)
ax1.set_title(r'Dynamic')
ax2.set_title(r'Static')

ax1.set_title(r'A',loc='left')
ax2.set_title(r'B',loc='left')
ax3.set_title(r'C',loc='left')
ax4.set_title(r'D',loc='left')
# Plot 1:
ax1.set_ylabel(r'$D_\parallel/D_{\mathregular{bulk}}$', fontsize=12)
ax3.set_ylabel(r'$D_\perp/D_{\mathregular{bulk}}$', fontsize=12)
ax3.set_xlabel(r'$d$ (nm)', fontsize=12)
ax4.set_xlabel(r'$d$ (nm)', fontsize=12)
ax.yaxis.get_offset_text().set_fontsize(12)
ax1.plot(spacings_dyn, Dparallel_dyn, color='g', label=r'$D$')
ax1.fill_between(spacings_dyn, Dparallel_dyn+Dparallel_stdv_dyn, Dparallel_dyn-Dparallel_stdv_dyn, facecolor='g', alpha=0.2)
line1, = ax1.plot(fitds,Ds_hr_dyn, '--,', label=r'hr')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#ax1.plot(fitds,Ds_hc_dyn, ':', label=r'hc')
ax1.plot(fitds,Ds_m9_dyn, '-.', label=r'c')
ax1.plot(fitds,Ds_p3_dyn, '--', label=r'pl')
ax1.legend(loc='upper left')#,ncol=2)
plt.tight_layout()
# Plot 2:
ax2.plot(spacings_stat, Dparallel_stat, color='g')
ax2.fill_between(spacings_stat, Dparallel_stat+Dparallel_stdv_stat, Dparallel_stat-Dparallel_stdv_stat, facecolor='g', alpha=0.2)
line1, = ax2.plot(fitds,Ds_hr_stat, '--,', label=r'hr')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#ax2.plot(fitds,Ds_hc_stat, ':', label=r'hc')
ax2.plot(fitds,Ds_m9_stat, '-.')
ax2.plot(fitds,Ds_p3_stat, '--')
# Plot 3:
ax3.plot(spacings_dyn, Dzs_dyn, color='g', label=r'$D_\perp$, dyn.')
ax3.fill_between(spacings_dyn, Dzs_dyn+Dzs_stdv_dyn, Dzs_dyn-Dzs_stdv_dyn, facecolor='g', alpha=0.2)
line1, = ax3.plot(fitds,Ds_hr_dyn_perp, '--,', label=r'hr')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#ax3.plot(fitds,Ds_hc_dyn_perp, ':', label=r'hc')
ax3.plot(fitds,Ds_m9_dyn_perp, '-.')
ax3.plot(fitds,Ds_p3_dyn_perp, '--')
# Plot 4:
ax4.plot(spacings_stat, Dzs_stat, color='g', label=r'$D_\perp$, stat.')
ax4.fill_between(spacings_stat, Dzs_stat+Dzs_stdv_stat, Dzs_stat-Dzs_stdv_stat, facecolor='g', alpha=0.2)
line1, = ax4.plot(fitds,Ds_hr_stat_perp, '--,', label=r'hr')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#ax4.plot(fitds,Ds_hc_stat_perp, ':', label=r'hc')
ax4.plot(fitds,Ds_m9_stat_perp, '-.')
ax4.plot(fitds,Ds_p3_stat_perp, '--')
plt.tight_layout()
plt.savefig(plotname_panels)


print('\n\nModel 1')
print('k stat, par:', k_m1_stat, '+/-', k_m1_stat_stdv)
print('k stat, perp:', k_m1_stat_perp, '+/-', k_m1_stat_perp_stdv)
print('k dyn, par:', k_m1_dyn, '+/-', k_m1_dyn_stdv)
print('k dyn, perp:', k_m1_dyn_perp, '+/-', k_m1_dyn_perp_stdv)

print('\n\nModel 2')
print('k stat, par:', k_m2_stat, '+/-', k_m2_stat_stdv)
print('k stat, perp:', k_m2_stat_perp, '+/-', k_m2_stat_perp_stdv)
print('k dyn, par:', k_m2_dyn, '+/-', k_m2_dyn_stdv)
print('k dyn, perp:', k_m2_dyn_perp, '+/-', k_m2_dyn_perp_stdv)


# Model 3
print('\n\nModel 3')
print('k stat, par:', k_m3_stat, '+/-', k_m3_stat_stdv)
print('k stat, perp:', k_m3_stat_perp, '+/-', k_m3_stat_perp_stdv)
print('k dyn, par:', k_m3_dyn, '+/-', k_m3_dyn_stdv)
print('k dyn, perp:', k_m3_dyn_perp, '+/-', k_m3_dyn_perp_stdv)

#'''
# Model 4
print('\n\nModel 4')
print('k stat, par:', k_m4_stat, '+/-', k_m4_stat_stdv,'; f stat, par:', f_m4_stat, '+/-', f_m4_stat_stdv)
print('k stat, perp:', k_m4_stat_perp, '+/-', k_m4_stat_perp_stdv,'; f stat, perp:', f_m4_stat_perp, '+/-', f_m4_stat_perp_stdv)
print('k dyn, par:', k_m4_dyn, '+/-', k_m4_dyn_stdv,'; f dyn, par:', f_m4_dyn, '+/-', f_m4_dyn_stdv)
print('k dyn, perp:', k_m4_dyn_perp, '+/-', k_m4_dyn_perp_stdv,'; f dyn, perp:', f_m4_dyn_perp, '+/-', f_m4_dyn_perp_stdv)
#'''

# Model 6
print('\n\nModel 6')
print('k stat, par:', k_m6_stat, '+/-', k_m6_stat_stdv)
print('k stat, perp:', k_m6_stat_perp, '+/-', k_m6_stat_perp_stdv)
print('k dyn, par:', k_m6_dyn, '+/-', k_m6_dyn_stdv)
print('k dyn, perp:', k_m6_dyn_perp, '+/-', k_m6_dyn_perp_stdv)
print('  ')


# Model 7
print('\n\nModel 7')
print('k stat, par:', k_m7_stat, '+/-', k_m7_stat_stdv)
print('k stat, perp:', k_m7_stat_perp, '+/-', k_m7_stat_perp_stdv)
print('k dyn, par:', k_m7_dyn, '+/-', k_m7_dyn_stdv)
print('k dyn, perp:', k_m7_dyn_perp, '+/-', k_m7_dyn_perp_stdv)
print('  ')

# Model 8
print('\n\nModel 8')
print('k stat, par:', k_m8_stat, '+/-', k_m8_stat_stdv)
print('k stat, perp:', k_m8_stat_perp, '+/-', k_m8_stat_perp_stdv)
print('k dyn, par:', k_m8_dyn, '+/-', k_m8_dyn_stdv)
print('k dyn, perp:', k_m8_dyn_perp, '+/-', k_m8_dyn_perp_stdv)
print('  ')

# Power law 1
print('\n\nPower law 1')
print('k stat, par:', k_p1_stat, '+/-', k_p1_stat_stdv,'l stat, par:', l_p1_stat, '+/-', l_p1_stat_stdv, 'n stat, par:', n_p1_stat, '+/-', n_p1_stat_stdv)
print('k stat, perp:', k_p1_stat_perp, '+/-', k_p1_stat_perp_stdv,'l stat, perp:', l_p1_stat_perp, '+/-', l_p1_stat_perp_stdv, 'n stat, perp:', n_p1_stat_perp, '+/-', n_p1_stat_perp_stdv)
print('k dyn, par:', k_p1_dyn, '+/-', k_p1_dyn_stdv,'l dyn, par:', l_p1_dyn, '+/-', l_p1_dyn_stdv, 'n dyn, par:', n_p1_dyn, '+/-', n_p1_dyn_stdv)
print('k dyn, perp:', k_p1_dyn_perp, '+/-', k_p1_dyn_perp_stdv,'l dyn, perp:', l_p1_dyn_perp, '+/-', l_p1_dyn_perp_stdv, 'n dyn, perp:', n_p1_dyn_perp, '+/-', n_p1_dyn_perp_stdv)
print('  ')

# Power law 2
print('\n\nPower law 2')
print('l stat, par:', l_p2_stat, '+/-', l_p2_stat_stdv, 'n stat, par:', n_p2_stat, '+/-', n_p2_stat_stdv)
print('l stat, perp:', l_p2_stat_perp, '+/-', l_p2_stat_perp_stdv, 'n stat, perp:', n_p2_stat_perp, '+/-', n_p2_stat_perp_stdv)
print('l dyn, par:', l_p2_dyn, '+/-', l_p2_dyn_stdv, 'n dyn, par:', n_p2_dyn, '+/-', n_p2_dyn_stdv)
print('l dyn, perp:', l_p2_dyn_perp, '+/-', l_p2_dyn_perp_stdv, 'n dyn, perp:', n_p2_dyn_perp, '+/-', n_p2_dyn_perp_stdv)
print('  ')


# Power law 3
print('\n\nPower law 3')
print('n stat, par:', n_p3_stat, '+/-', n_p3_stat_stdv, 'k stat, par:', k_p3_stat, '+/-', k_p3_stat_stdv)
print('n stat, perp:', n_p3_stat_perp, '+/-', n_p3_stat_perp_stdv, 'k stat, perp:', k_p3_stat_perp, '+/-', n_p3_stat_perp_stdv)
print('n dyn, par:', n_p3_dyn, '+/-', n_p3_dyn_stdv, 'k dyn, par:', k_p3_dyn, '+/-', k_p3_dyn_stdv)
print('n dyn, perp:', n_p3_dyn_perp, '+/-', n_p3_dyn_perp_stdv, 'k dyn, perp:', k_p3_dyn_perp, '+/-', n_p3_dyn_perp_stdv)
print('  ')

# Power law 4
print('\n\nPower law 4')
print('n stat, par:', n_p4_stat, '+/-', n_p4_stat_stdv)
print('n stat, perp:', n_p4_stat_perp, '+/-', n_p4_stat_perp_stdv)
print('n dyn, par:', n_p4_dyn, '+/-', n_p4_dyn_stdv)
print('n dyn, perp:', n_p4_dyn_perp, '+/-', n_p4_dyn_perp_stdv)
print('  ')


### From Shen & Chen:

print('-----------------------')
print('Ordered packings:')
print('sigma, op, dyn, par:',sigma_op_dyn,'+/-',sigma_op_dyn_rms)
print('sigma, op, stat, par:',sigma_op_stat,'+/-',sigma_op_stat_rms)
print('sigma, op, dyn, perp:',sigma_op_dyn_perp,'+/-',sigma_op_dyn_perp_rms)
print('sigma, op, stat, perp:',sigma_op_stat_perp,'+/-',sigma_op_stat_perp_rms)
print('-----------------------')
print('Hyperbola of revolution:')
print('sigma, hr, dyn, par:',sigma_hr_dyn,'+/-',sigma_hr_dyn_rms)
print('sigma, hr, stat, par:',sigma_hr_stat,'+/-',sigma_hr_stat_rms)
print('sigma, hr, dyn, perp:',sigma_hr_dyn_perp,'+/-',sigma_hr_dyn_perp_rms)
print('sigma, hr, stat, perp:',sigma_hr_stat_perp,'+/-',sigma_hr_stat_perp_rms)
print('-----------------------')
print('Heterogeneous catalyst:')
print('sigma, hc, dyn, par:',sigma_hc_dyn,'+/-',sigma_hc_dyn_rms)
print('sigma, hc, stat, par:',sigma_hc_stat,'+/-',sigma_hc_stat_rms)
print('sigma, hc, dyn, perp:',sigma_hc_dyn_perp,'+/-',sigma_hc_dyn_perp_rms)
print('sigma, hc, stat, perp:',sigma_hc_stat_perp,'+/-',sigma_hc_stat_perp_rms)
print('-----------------------')
print('Cation-exchange resin membrane:')
print('sigma, rm, dyn, par:',sigma_rm_dyn,'+/-',sigma_rm_dyn_rms)
print('sigma, rm, stat, par:',sigma_rm_stat,'+/-',sigma_rm_stat_rms)
print('sigma, rm, dyn, perp:',sigma_rm_dyn_perp,'+/-',sigma_rm_dyn_perp_rms)
print('sigma, rm, stat, perp:',sigma_rm_stat_perp,'+/-',sigma_rm_stat_perp_rms)
print('-----------------------')



print('k_m4_stat_perp:',k_m4_stat_perp)

print('sigma_rm_stat_perp:',sigma_rm_stat_perp)
print('sigma_rm_dyn_perp:',sigma_rm_dyn_perp)
print('sigma_rm_stat:',sigma_rm_stat)
print('sigma_rm_dyn:',sigma_rm_dyn)

print('Covariance matrices:')
print('Modell 4:')
print('pcov4_statpar:',pcov4_statpar)
print('pcov4_dynpar:',pcov4_dynpar)
print('pcov4_statperp:',pcov4_statperp)
print('pcov4_dynperp:',pcov4_dynperp)

print('Power law 3:')
print('pcovp3_statpar:',pcovp3_statpar)
print('pcovp3_dynpar:',pcovp3_dynpar)
print('pcovp3_statperp:',pcovp3_statperp)
print('pcovp3_dynperp:',pcovp3_dynperp)


### sigma_walker, scaling the mass : #############################################################################################################################
plotname_sigma_free_massadj_dynamic                  = endlocation + 'D_vs_d_varioussigmas_sigmafree_dynamic_adjustingmass.tiff'
plotname_sigma_free_massadj_static                   = endlocation + 'D_vs_d_varioussigmas_sigmafree_static_adjustingmass.tiff'
plotname_difftime_sigma_free_massadj_dynamic         = endlocation + 'diffusiontime_dynamic_sigmafree_adjustingmass.tiff'
plotname_difftime_sigma_free_massadj_static          = endlocation + 'diffusiontime_static_sigmafree_adjustingmass.tiff'
plotname_difftime_sigma_free_massadj_dynamic_zooming = endlocation + 'diffusiontime_dynamic_sigmafree_adjustingmass_zoomed.tiff'
plotname_difftime_sigma_free_massadj_static_zooming  = endlocation + 'diffusiontime_static_sigmafree_adjustingmass_zoomzoom.tiff'

bulkfilename_II = '/Diffusion_bead_near_grid/D_vs_d/Bulk/Varysigmas/D_vs_sigma_better_rms_Nestimates10.txt'
bulkfile_II = open(bulkfilename_II,'r')
hline = bulkfile_II.readline()
lines = bulkfile_II.readlines()

sigma_walker = []
Dzs_bulk_vm = []
for line in lines:
    words = line.split()
    if len(words)>0:
        sigma_walker.append(float(words[0])/2.)
        Dzs_bulk_vm.append(float(words[3]))
bulkfile_II.close()

mylinestyles = [[1,1,1,1],[2,1,5,1],[3,1,3,1],[10,0],[2,2,10,2],[10,1,1,1]]

sigma_walker = sigma_walker[:6]

fitds = np.linspace(2,10,100)
thickness = 50e-9 # m
sigma_chain  = 1.0
sigma_walker = np.array(sigma_walker)
sigmas = (sigma_walker+sigma_chain)/2.
Nsigmas = len(sigmas)
fitds_dyn_store = []
diffusiontimes_dyn = []
j = 0
plt.figure(figsize=(6,5),dpi=300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
for sigma in sigma_walker:
    Ds_p3_dyn_perp = powerlaw3_sigmas(fitds,sigma,n_p3_dyn_perp,k_p3_dyn_perp)
    fitds_II          = []
    Ds_p3_dyn_perp_II = []
    diffusiontime_dyn = []
    for i in range(len(Ds_p3_dyn_perp)):
        if Ds_p3_dyn_perp[i]>0:
            thisD = Ds_p3_dyn_perp[i]
            fitds_II.append(fitds[i])
            Ds_p3_dyn_perp_II.append(thisD)
            D = thisD*Dzs_bulk_vm[j]
            difftime = thickness**2/(2.*D)
            diffusiontime_dyn.append(difftime)
    line1, = plt.plot(fitds_II,Ds_p3_dyn_perp_II,label=r'$a$ = %s nm' %str(sigma), linewidth=2.0)
    line1.set_dashes(mylinestyles[j])
    diffusiontimes_dyn.append(diffusiontime_dyn)
    fitds_dyn_store.append(fitds_II)
    j+=1
plt.xlabel('d (nm)', fontsize=12)
plt.ylabel(r'$D_\perp/D_{\mathregular{bulk}}$', fontsize=12)
ax.yaxis.get_offset_text().set_fontsize(12)
plt.legend(loc="lower right",fontsize=12)
plt.tight_layout())
plt.savefig(plotname_sigma_free_massadj_dynamic)

j = 0
fitds_stat_store = []
diffusiontimes_stat = []
plt.figure(figsize=(6,5),dpi=300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
for sigma in sigma_walker:
    Ds_p3_stat_perp = powerlaw3_sigmas(fitds,sigma,n_p3_stat_perp,k_p3_stat_perp)
    fitds_II           = []
    Ds_p3_stat_perp_II = []
    diffusiontime_stat = []
    for i in range(len(Ds_p3_stat_perp)):
        if Ds_p3_stat_perp[i]>0:
            thisD = Ds_p3_stat_perp[i]
            fitds_II.append(fitds[i])
            Ds_p3_stat_perp_II.append(thisD)
            D = thisD*Dzs_bulk_vm[j]
            difftime = thickness**2/(2.*D)
            diffusiontime_stat.append(difftime)
    line1, = plt.plot(fitds_II,Ds_p3_stat_perp_II,label=r'$a$ = %s nm' %str(sigma), linewidth=2.0)
    line1.set_dashes(mylinestyles[j])
    diffusiontimes_stat.append(diffusiontime_stat)
    fitds_stat_store.append(fitds_II)
    j+=1
plt.xlabel('d (nm)', fontsize=12)
plt.ylabel(r'$D_\perp/D_{\mathregular{bulk}}$', fontsize=12)
plt.legend(loc="lower right",fontsize=12)
ax.yaxis.get_offset_text().set_fontsize(12)
plt.tight_layout()
plt.savefig(plotname_sigma_free_massadj_static)

'''
plt.figure(figsize=(6,5))
for i in range(Nsigmas):
    plt.plot(fitds_dyn_store[i],diffusiontimes_dyn[i],label=r'$\sigma_{\mathregular{LJ}}$ = %s nm' %str(sigma_walker[i]))
plt.xlabel('d (nm)', fontsize=12)
plt.ylabel('Diffusion time (s)', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig(plotname_difftime_sigma_free_massadj_dynamic)

plt.figure(figsize=(6,5))
for i in range(Nsigmas):
    plt.plot(fitds_stat_store[i],diffusiontimes_stat[i],label=r'$\sigma_{\mathregular{LJ}}$ = %s nm' %str(sigma_walker[i]))
plt.xlabel('d (nm)', fontsize=12)
plt.ylabel('Diffusion time (s)', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig(plotname_difftime_sigma_free_massadj_static)
'''

###### Zoom-friendly: #######
#'''
fig, ax = plt.subplots(figsize=(6,5),dpi=300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
for i in range(Nsigmas):
    line1, = plt.plot(fitds_dyn_store[i],diffusiontimes_dyn[i],label=r'$a$ = %s nm' %str(sigma_walker[i]), linewidth=2.0)
    line1.set_dashes(mylinestyles[i])
plt.xlabel(r'$d$ (nm)', fontsize=12)
plt.ylabel('Diffusion time (s)', fontsize=12)
plt.legend(loc="upper_right", fontsize=12)
plt.axis([1.98,10.02,-1e-9,0.37e-7])
ax.yaxis.get_offset_text().set_fontsize(12)
plt.tight_layout()
plt.savefig(plotname_difftime_sigma_free_massadj_dynamic_zooming)
'''

plt.figure(figsize=(7.5,5))
for i in range(Nsigmas):#(5):#
    plt.plot(fitds_stat_store[i],diffusiontimes_stat[i],label=r'$\sigma_{\mathregular{LJ}}$ = %s nm' %str(sigma_walker[i]))
plt.xlabel('d (nm)')
plt.ylabel('Diffusion time (s)')
plt.legend(bbox_to_anchor=(1,0), loc="lower left",fontsize=12)
plt.tight_layout()
plt.savefig(plotname_difftime_sigma_free_massadj_static_zooming)

#'''
##################################################################################################################################################################

if showplots==True:
    plt.show()
