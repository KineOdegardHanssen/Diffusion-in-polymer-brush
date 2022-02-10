import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
import numpy as np
import random
import math
import time
import os
import glob

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 

# Height of net:
h = 50e-9 # m

# Constants. Fix later:
F  = 96485.33212       #       mol-1 C
R  = 8.31446261815324  # J K-1 mol-1
T  = 310               #   K
NA = 6.022e-13         # Avogadro's constant
constant = F**2/(R*T)#/NA

# Radius of neuron:
# Old: r1 = 1.5e-6  # m; r2 = 2.75e-6 # m; r3 = 4e-6    # m
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
charges = [1,2,-1] #,-2] # I haven't encountered an ion with charge -2e in the brain ECS in literature
concentrations = [149,2,145] # In mM or mol m-3
ds = []
volumes = []

vfilename = '/home/kine/Documents/Backup2_P2_PolymerMD/P2_PolymerMD/Planar_brush/Diffusion_bead_near_grid/volume_vs_d.txt'
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

plotfolder            = '/home/kine/Documents/Backup2_P2_PolymerMD/P2_PolymerMD/Planar_brush/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/Nocut/'
plotname_conductivity = plotfolder+'cond_vs_d.png'
plotname_resistivity  = plotfolder+'rho_vs_d.png'
plotname_resistance   = plotfolder+'R_vs_d.png'
plotname_resistivity_and_conductivity = plotfolder+'cond_and_rho_vs_d.png'

for j in range(len(charges)):
    charge        = charges[j]
    concentration = concentrations[j]
    Ds_temp       = []
    Ds_rms_temp   = []
    inlocation    = '/home/kine/Documents/Backup2_P2_PolymerMD/P2_PolymerMD/Planar_brush/Diffusion_bead_near_grid/D_vs_d/Brush/Sigma_bead_' +str(psigma) + '/Nocut/'+ 'Charge' + str(charge)+'/'
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
    print('charge**2:',charge**2)
    
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

fig, ax = plt.subplots(figsize=(6,3))
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
plt.plot(ds, sigma)
plt.fill_between(ds, sigma+sigma_rms, sigma-sigma_rms, alpha=0.2)
plt.xlabel(r'$d$ (nm)', fontsize=15)
plt.ylabel(r'$\sigma_e$ (S/m)', fontsize=15)
plt.tight_layout()
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(12)
plt.tight_layout()
plt.savefig(plotname_conductivity)

#plt.figure(figsize=(6,5))
fig, ax = plt.subplots(figsize=(6,3))
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
plt.plot(ds, rho)
plt.fill_between(ds, rho+rho_rms, rho-rho_rms, alpha=0.2)
plt.xlabel(r'$d$ (nm)', fontsize=15)
plt.ylabel(r'$\rho_e$ ($\Omega$ m)', fontsize=15)
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(12)
plt.tight_layout()
plt.savefig(plotname_resistivity)


#plt.figure(figsize=(6,5))
fig, ax = plt.subplots(figsize=(6,3))
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
plt.plot(ds, R1, '-o', label=r'$r_{\mathregular{neuron}}$=%i $\mu$m' % r1_um)
plt.fill_between(ds, R1+R1_rms, R1-R1_rms, alpha=0.2)
plt.plot(ds, R2, '-d', label=r'$r_{\mathregular{neuron}}$=%i $\mu$m' % r2_um)
plt.fill_between(ds, R2+R2_rms, R2-R2_rms, alpha=0.2)
plt.plot(ds, R3, '-X', label=r'$r_{\mathregular{neuron}}$=%i $\mu$m' % r3_um)
plt.fill_between(ds, R3+R3_rms, R3-R3_rms, alpha=0.2)
plt.xlabel(r'$d$ (nm)', fontsize=15)
plt.ylabel(r'$R$ ($\Omega$)', fontsize=15)
plt.tight_layout()
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc='upper right', fontsize=11)
ax.yaxis.get_offset_text().set_fontsize(12)
plt.tight_layout()
plt.savefig(plotname_resistance)


### Twoinone
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,6),dpi=300)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
ax1.plot(ds, sigma)
ax1.fill_between(ds, sigma+sigma_rms, sigma-sigma_rms, alpha=0.2)
#ax1.set_xlabel(r'$d$ (nm)', fontsize=15)
ax1.set_ylabel(r'$\sigma_e$ (S/m)', fontsize=17)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.get_offset_text().set_fontsize(14)
ax1.set_title('A',loc='left', x=-0.03, y=1.1, fontsize=15)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
ax2.plot(ds, rho)
ax2.fill_between(ds, rho+rho_rms, rho-rho_rms, alpha=0.2)
ax2.set_xlabel(r'$d$ (nm)', fontsize=17)
ax2.set_ylabel(r'$\rho_e$ ($\Omega$ m)', fontsize=17)
plt.tight_layout()
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.yaxis.get_offset_text().set_fontsize(14)
ax2.set_title('B',loc='left', x=-0.03, y=1.1, fontsize=15)
plt.tight_layout()
plt.savefig(plotname_resistivity_and_conductivity)

print('r1:',r1)
print('R1:',R1)

A3 = 4*np.pi*(r3*100)**2
print('A, r3 (in cm2):', A3)
print('120mScm-2 -->', 120e-3*A3)
print('36mScm-2 -->', 36e-3*A3)
print('0.3mScm-2 -->', 0.3e-3*A3)


#plt.show()
