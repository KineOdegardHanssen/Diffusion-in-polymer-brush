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

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

# Height of net:
h = 50e-9 # m

# Constants. Fix later:
F  = 96485.33212       #       mol-1 C
R  = 8.31446261815324  # J K-1 mol-1
T  = 310               #   K
constant = F**2/(R*T)

# Radius of neuron:
r1 = 6e-6 # m
r2 = 8e-6 # m
r3 = 11e-6 # m
# For plotting:
r1_um = r1*1e6
r2_um = r2*1e6
r3_um = r3*1e6

def rho_to_R(rho,h,r):
    a = r
    b = r+h
    return rho/(4*np.pi)*(1./a-1./b)

psigma  = 1
charges = [1,2,-1]
concentrations = [149,2,145] # In mM or mol m-3
ds = []
volumes = []

vfilename = '/Diffusion_bead_near_grid/volume_vs_d.txt'
vfile     = open(vfilename,'r')
lines     = vfile.readlines()
for line in lines:
    words = line.split()
    if len(words)>0:
        ds.append(float(words[0]))
        #volumes.append(float(words[1]))
vfile.close()

N         = len(ds)
R1        = np.zeros(N)
R1_rms    = np.zeros(N)
R2        = np.zeros(N)
R2_rms    = np.zeros(N)
R3        = np.zeros(N)
R3_rms    = np.zeros(N)
sigma     = np.zeros(N)
rho       = np.zeros(N)
sigma_rms = np.zeros(N)
rho_rms   = np.zeros(N)

plotfolder            = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/Nocut/'
plotname_conductivity = plotfolder+'cond_vs_d.tiff'
plotname_resistivity  = plotfolder+'rho_vs_d.tiff'
plotname_resistance   = plotfolder+'R_vs_d.tiff'
plotname_resistivity_and_conductivity = plotfolder+'cond_and_rho_vs_d.tiff'

for j in range(len(charges)):
    charge        = charges[j]
    concentration = concentrations[j]
    Ds_temp       = []
    Ds_rms_temp   = []
    inlocation    = '/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/Nocut/'+ 'Charge' + str(charge)+'/'
    infilename    = inlocation+'D_vs_d_charge'+str(charge)+'.txt'
    infile        = open(infilename,'r')
    header        = infile.readline()
    lines         = infile.readlines()

    i = 0
    for line in lines:
        words = line.split()
        if len(words)>0:
            # Read in
            Dp = float(words[3])
            Dp_rms = float(words[4])
            Ds_temp.append(Dp)
            Ds_rms_temp.append(Dp_rms)
            # Do calculations while I'm on it
            sigma[i] += Dp*charge**2*concentration
            sigma_rms[i] += (Dp_rms*charge**2*concentration)**2
        i+=1
    infile.close()

for i in range(N):
    # Conductivity
    sigma[i] *= constant
    sigma_rms[i]  = np.sqrt(sigma_rms[i])
    sigma_rms[i] *= constant
    # Resistivity
    rho[i] = 1./sigma[i]
    rho_rms[i] = sigma_rms[i]/(sigma[i]**2)
    # Resistance
    R1[i] = rho_to_R(rho[i],h,r1)
    R1_rms[i] = R1[i]/rho[i]*rho_rms[i]
    R2[i] = rho_to_R(rho[i],h,r2)
    R2_rms[i] = R2[i]/rho[i]*rho_rms[i]
    R3[i] = rho_to_R(rho[i],h,r3)
    R3_rms[i] = R3[i]/rho[i]*rho_rms[i]

### Twoinone
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,6),dpi=300)

startat = 4
ax1.plot(ds[startat:], sigma[startat:])
ax1.fill_between(ds[startat:], sigma[startat:]+sigma_rms[startat:], sigma[startat:]-sigma_rms[startat:], alpha=0.2)
ax1.set_ylabel(r'$\sigma_e$ (S/m)', fontsize=12)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.get_offset_text().set_fontsize(12)
ax1.set_title('A',loc='left', x=-0.03, y=1.1, fontsize=12)

ax2.plot(ds[startat:], R1[startat:], '-o', label=r'$r_{\mathregular{neuron}}$=%i $\mu$m' % r1_um)
ax2.fill_between(ds[startat:], R1[startat:]+R1_rms[startat:], R1[startat:]-R1_rms[startat:], alpha=0.2)
ax2.plot(ds[startat:], R2[startat:], '-d', label=r'$r_{\mathregular{neuron}}$=%i $\mu$m' % r2_um)
ax2.fill_between(ds[startat:], R2[startat:]+R2_rms[startat:], R2[startat:]-R2_rms[startat:], alpha=0.2)
ax2.plot(ds[startat:], R3[startat:], '-X', label=r'$r_{\mathregular{neuron}}$=%i $\mu$m' % r3_um)
ax2.fill_between(ds[startat:], R3[startat:]+R3_rms[startat:], R3[startat:]-R3_rms[startat:], alpha=0.2)
ax2.set_xlabel(r'$d$ (nm)', fontsize=12)
ax2.set_ylabel(r'$R$ ($\Omega$)', fontsize=12)
ax2.legend(loc='upper right', fontsize=11)
plt.tight_layout()
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.yaxis.get_offset_text().set_fontsize(12)
ax2.set_title('B',loc='left', x=-0.03, y=1.1, fontsize=12)
plt.tight_layout()
plt.savefig(plotname_resistivity_and_conductivity)


plt.show()
