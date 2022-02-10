import matplotlib.pyplot as plt                     # To plot
from pylab import *
import numpy as np
import random
import math
import time
import os
import glob
import copy

# Treats all the _all-files regardless of whether the diffusing bead displays an unphysical trajectory.

#
damp = 10
M = 9
Ninch = 101
# Input parameters for file selection:
popup_plots = False
long    = True
spacings = [1,1.25,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,10,15,25,50,75,100]

# long: 2.5, 3.5, 4.5
psigma  = 1

T        = 3
plotseed = 0
plotdirs = False
zhigh          = 290    # For omitting unphysical trajectories
zlow           = -50    
confignrs    = np.arange(1,1001)
Nseeds       = len(confignrs)      # So that I don't have to change that much
maxz_av      = 0
filescounter = 0
unitlength   = 1e-9
unittime     = 2.38e-11 # s
timestepsize = 0.00045*unittime
filestext    = 'config'+str(confignrs[0])+'to'+str(confignrs[-1])

endlocation = '/Diffusion_bead_near_grid/d_vs_Rg/'
outfilename = endlocation + 'd_vs_Rg_avg_rms.txt'
plotname    = endlocation + 'd_vs_Rg_avg_rms.tiff'
outfile = open(outfilename,'w')

Rg_avgs = []
Rg_rms  = []

for spacing in spacings:
    inlocation          = '/Diffusion_bead_near_grid/Spacing'+str(spacing)+'/damp%i_diffseedLgv/Brush/Sigma_bead_' % damp + str(psigma) + '/'
    # Text files
    infilename_radgyrs_avg = inlocation+filestext+'_radgyr_avg_rms'
    
    if spacing==2.5 or spacing==3.5 or spacing==4.5:
        long = True
    else:
        long = False
    
    if long==True:
        infilename_radgyrs_avg = infilename_radgyrs_avg +'_long.txt'
    else:
        infilename_radgyrs_avg = infilename_radgyrs_avg +'.txt'
    
    infile = open(infilename_radgyrs_avg,'r')
    line  = infile.readline()
    words = line.split()
    avg   = float(words[0])
    rms   = float(words[1])
    
    Rg_avgs.append(avg)
    Rg_rms.append(rms)
    
    outfile.write('%.2f %.16f %.16f\n' % (spacing,avg,rms))
outfile.close()

spacings = np.array(spacings)
Rg_avgs = np.array(Rg_avgs)
Rg_rms  = np.array(Rg_rms)

maxarray = Rg_avgs+Rg_rms
ymaxax   = max(maxarray)+2.9

spacings_plot = np.linspace(-4,106,4)
spacings_plot_div2 = spacings_plot/2

plt.figure(figsize=(6,3),dpi=300)
ax = plt.subplot()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.plot(spacings, Rg_avgs,'.-')
ax.fill_between(spacings, Rg_avgs+Rg_rms, Rg_avgs-Rg_rms, alpha=0.2)
ax.fill_between(spacings_plot, spacings_plot_div2, alpha=0.2)
ax.set_xlabel(r'$d$ (nm)',fontsize=12)
ax.set_ylabel(r'$<R_g>$ (nm)',fontsize=12)
ax.text(60,3,'Mushroom phase',fontsize=12)
ax.text(6,45,'Brush phase',fontsize=12)
plt.axis([-2,102,0,ymaxax])
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig(plotname)

plt.show()
