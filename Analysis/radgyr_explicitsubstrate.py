from skimage import morphology, measure, io, util   # For performing morphology operations (should maybe import more from skimage...)
from mpl_toolkits.mplot3d import Axes3D             # Plotting in 3D
import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
from scipy.ndimage import measurements, convolve    # Should have used the command below, but changing now will lead to confusion
from scipy import ndimage                           # For Euclidean distance measurement
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
long    = False
spacing = 100
psigma  = 1
print('spacing:', spacing)
print('psigma:', psigma)

T        = 3
plotseed = 0
plotdirs = False
zhigh          = 290    # For omitting unphysical trajectories
zlow           = -50    
confignrs    = np.arange(1,1001)
Nseeds       = len(confignrs)      # So that I don't have to change that much
maxz_av      = 0
filescounter = 0
if long==True:
    Nsteps   = 10001
else:
    Nsteps   = 2001 
unitlength   = 1e-9
unittime     = 2.38e-11 # s
timestepsize = 0.00045*unittime

endlocation          = '/Diffusion_bead_near_grid/Spacing'+str(spacing)+'/damp%i_diffseedLgv/Brush/Sigma_bead_' % damp + str(psigma) + '/'
filestext            = 'config'+str(confignrs[0])+'to'+str(confignrs[-1])
# Text files
outfilename_radgyrs_all = endlocation+filestext+'_radgyr_all'
outfilename_radgyrs_avg = endlocation+filestext+'_radgyr_avg_rms'

if long==True:
    outfilename_radgyrs_all = outfilename_radgyrs_all+'_long.txt'
    outfilename_radgyrs_avg = outfilename_radgyrs_avg +'_long.txt'
else:
    outfilename_radgyrs_all = outfilename_radgyrs_all+'.txt'
    outfilename_radgyrs_avg = outfilename_radgyrs_avg +'.txt'

outfile_radgyrs_all = open(outfilename_radgyrs_all,'w')

radgyrs_all = []
radgyrs_avg = 0
radgyrs_rms = 0

Nread        = 0
skippedfiles = 0

for confignr in confignrs:
    print('On config number:', confignr)
    if long==True:
        infilename_all  = endlocation+'long/'+'all_confignr'+str(confignr)+'_long.lammpstrj'
    else:
        infilename_all  = endlocation+'all_confignr'+str(confignr)+'.lammpstrj'
    
    # Read in:
    #### Automatic part
    ## Find the extent of the polymers: Max z-coord of beads in the chains
    try:
        infile_all = open(infilename_all, "r")
    except:
        try:
            infilename_all = endlocation+'long/'+'all_confignr'+str(confignr)+'.lammpstrj'
            infile_all = open(infilename_all, "r")
        except:
            print('Oh, lammpstrj-file! Where art thou?')
            skippedfiles += 1
            continue # Skipping this file if it does not exist
    # Moving on, if the file
    filescounter += 1
    lines = infile_all.readlines() # This takes some time
    # Getting the number of lines, etc.
    totlines = len(lines)         # Total number of lines
    lineend = totlines-1          # Index of last element
    
    # Extracting the number of atoms:
    words = lines[3].split()
    Nall = int(words[0])
    N    = Nall

    chain_pos = np.zeros((M,Ninch,3)) # All atoms in chains. One time step because the first is the same for all configs
    
    skiplines   = 9             # If we hit 'ITEM:', skip this many steps...
    skipelem    = 0
    sampleevery = 0
    skiplines  += (Nall+skiplines)*sampleevery
    i           = 0
    t_index     = -1
    
    time_start = time.process_time()
    counter = 0
    while i<totlines:
        words = lines[i].split()
        if words[0]=='ITEM:':
            if words[1]=='TIMESTEP':
                t_index+=1
                if t_index==2:
                    break
                i_array     = np.zeros(9)
                i+=skiplines
            elif words[1]=='NUMBER': # These will never kick in. 
                i+=7
            elif words[1]=='BOX':
                i+=5
            elif words[1]=='ATOMS':
                i+=1
        elif len(words)<8:
            i+=1
        else:
            # Find properties
            # Order:  id  type mol ux  uy  uz  vx  vy   vz
            #         [0] [1]  [2] [3] [4] [5] [6] [7]  [8]
            atomtype = int(words[1]) 
            if atomtype==1 or atomtype==2: # Chain bead. Treat
                molID    = int(words[2])-1
                x        = float(words[3])
                y        = float(words[4])
                z        = float(words[5])
                # Overwriting the first time step. Not that efficient, but won't use this script much
                chain_pos[molID,int(i_array[molID]),0] = x 
                chain_pos[molID,int(i_array[molID]),1] = y
                chain_pos[molID,int(i_array[molID]),2] = z
                i_array[molID] += 1
            i+=1
    infile_all.close()
    
    # Find radius of gyration:
    for ic in range(M): # Looping over chains
        # Finding average r:
        x_avg = np.mean(chain_pos[ic,:,0])
        y_avg = np.mean(chain_pos[ic,:,1])
        z_avg = np.mean(chain_pos[ic,:,2])
        r_avg = np.array([x_avg,y_avg,z_avg])
        # Preparing the averaging:
        rg_cumul = 0
        for im in range(Ninch): # Looping over all beads in the chain ic
            rdiff    = chain_pos[ic,im,:]-r_avg
            rg_cumul += np.dot(rdiff,rdiff)
        rg_cumul = np.sqrt(rg_cumul/Ninch)
        radgyrs_all.append(rg_cumul) 
        radgyrs_avg += rg_cumul 
        outfile_radgyrs_all.write('%.5f ' % rg_cumul)
    outfile_radgyrs_all.write('\n') # New line for each time step
            
outfile_radgyrs_all.close()

NRg = len(radgyrs_all)
radgyrs_avg/=NRg
radgyrs_rms=0
for i in range(NRg):
    radgyrs_rms += (radgyrs_all[i]-radgyrs_avg)**2
radgyrs_rms = np.sqrt(radgyrs_rms/(NRg-1))

outfile_radgyrs_avg = open(outfilename_radgyrs_avg,'w')
outfile_radgyrs_avg.write('%.16e %.16e' % (radgyrs_avg,radgyrs_rms))
outfile_radgyrs_avg.close()

print('d:',spacing)
print('Rg:', radgyrs_avg, ' +/-', radgyrs_rms)

test1 = np.mean(chain_pos[1,:,0])
x_av = 0
for im in range(Ninch):
    x_av+=chain_pos[1,im,0]
x_av/=Ninch

