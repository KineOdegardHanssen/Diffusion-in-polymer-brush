import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
import numpy as np
import random
import math
import time
import os
import glob
import copy

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
#
damp = 10
# Input parameters for file selection: 
popup_plots = False
long    = False
spacing = 4
psigma  = 1

# Extracting the correct names (all of them)
T        = 3
plotdirs = False
zhigh          = 250
zlow           = -50
confignrs    = np.arange(1,1001)
Nseeds       = len(confignrs) 
maxz_av      = 0
filescounter = 0
if long==True:
    Nsteps   = 10001
else:
    Nsteps   = 2001
unitlength   = 1e-9
unittime     = 2.38e-11 # s
timestepsize = 0.00045*unittime
print('timestepsize:', timestepsize)
istart = 63 # For equilibration

endlocation_in           = '/Diffusion_bead_near_grid/Spacing'+str(spacing)+'/damp%i_diffseedLgv/Brush/Sigma_bead_' % damp+str(psigma) + '/'
endlocation              = endlocation_in +'Nocut/'
filestext                = 'config'+str(confignrs[0])+'to'+str(confignrs[-1])
# Text files
outfilename_vacf     = endlocation+'vacf_'+filestext+'_nocut'
outfilename_Dvacf    = endlocation+'Dvacf_'+filestext+'_nocut'

# Plots
plotname             = endlocation+filestext+'vacf_nocut'

if long==True:
    outfilename_vacf     = outfilename_vacf+'_long.txt'
    outfilename_Dvacf    = outfilename_Dvacf+'_long.txt'
    
    # Plots
    plotname             = plotname+'_long.png'
else:
    outfilename_vacf     = outfilename_vacf+'.txt'
    outfilename_Dvacf    = outfilename_Dvacf+'.txt'
    
    # Plots
    plotname             = plotname+'.png'

## Setting arrays
# These are all squared:
# All together:
allRs     = []
average_counter = np.zeros(Nsteps-istart)
# Velocity autocorrelation function:
vacf_tot = np.zeros(Nsteps-istart)
vacf_x   = np.zeros(Nsteps-istart)
vacf_y   = np.zeros(Nsteps-istart)
vacf_z   = np.zeros(Nsteps-istart)
vacf_par = np.zeros(Nsteps-istart)


# Separated by seed:
Nins          = []
# This is not squared, obviously:
alltimes  = []

skippedfiles = 0


for confignr in confignrs:
    print('On config number:', confignr)
    if long==True:
        infilename_all  = endlocation_in+'long/'+'all_confignr'+str(confignr)+'_long.lammpstrj'
        infilename_free = endlocation_in+'long/'+'freeatom_confignr'+str(confignr)+'_long.lammpstrj'
    else:
        infilename_all  = endlocation_in+'all_confignr'+str(confignr)+'.lammpstrj'
        infilename_free = endlocation_in+'freeatom_confignr'+str(confignr)+'.lammpstrj'
    
    # Read in:
    #### Automatic part
    ## Find the extent of the polymers: Max z-coord of beads in the chains
    try:
        infile_all = open(infilename_all, "r")
    except:
        try:
            infilename_all = endlocation_in+'long/'+'all_confignr'+str(confignr)+'.lammpstrj'
            infilename_free = endlocation_in+'long/'+'freeatom_confignr'+str(confignr)+'.lammpstrj'
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
    
    skiplines   = 9             # If we hit 'ITEM:', skip this many steps...
    skipelem    = 0
    sampleevery = 0
    i           = int(math.ceil(skipelem*(Nall+9)))
    skiplines  += (Nall+skiplines)*sampleevery
    
    # For now: Only find the largest z-position among the beads in the chain.
    maxz = -1000 # No z-position is this small.
    
    time_start = time.process_time()
    counter = 0
    while i<totlines:
        words = lines[i].split()
        if (words[0]=='ITEM:'):
            if words[1]=='TIMESTEP':
                i+=skiplines
            elif words[1]=='NUMBER': 
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
            ind      = int(words[0])-1 # Atom ids go from zero to N-1.
            atomtype = int(words[1]) 
            z        = float(words[5])
            if atomtype==2: # Moving polymer bead. Test if this is larger than maxz:
                if z>maxz:
                    maxz = z
            i+=1
    extent_polymers = maxz
    maxz_av += maxz
    infile_all.close()
    
    ## Find the position of the free particle: 
    infile_free = open(infilename_free, "r")
    lines = infile_free.readlines() # This takes some time
    # Getting the number of lines, etc.
    totlines = len(lines)         # Total number of lines
    lineend = totlines-1          # Index of last element
    
    # Extracting the number of atoms:
    words = lines[3].split()
    Nall = int(words[0])
    N    = Nall
    
    skiplines   = 9             # If we hit 'ITEM:', skip this many steps...
    skipelem    = 0
    sampleevery = 0
    i           = int(math.ceil(skipelem*(Nall+9)))
    skiplines  += (Nall+skiplines)*sampleevery
    
    
    # Setting arrays for treatment:
    vx = [] # x-component of the velocity. 
    vy = []
    vz = []
    positions = [] # Fill with xu,yu,zu-coordinates. One position for each time step. 
    times     = []
    
    maxz = -1000 # No z-position is this small.
    
    time_start = time.process_time()
    counter = 0
    zs_fortesting = []
    while i<totlines:
        words = lines[i].split()
        if (words[0]=='ITEM:'):
            if words[1]=='TIMESTEP':
                words2 = lines[i+1].split() # The time step is on the next line
                t = float(words2[0])
                times.append(t)
                i+=skiplines
            elif words[1]=='NUMBER':
                i+=1
            elif words[1]=='BOX':
                i+=1
            elif words[1]=='ATOMS':
                i+=1
        elif len(words)<8:
            i+=1
        else:
            # Find properties
            # Order:  id  type mol ux  uy  uz  vx  vy   vz
            #         [0] [1]  [2] [3] [4] [5] [6] [7]  [8]
            ind      = int(words[0])-1 # Atom ids go from zero to N-1.
            x        = float(words[3])
            y        = float(words[4])
            z        = float(words[5])
            positions.append(np.array([x,y,z]))
            vx.append(float(words[6]))
            vy.append(float(words[7]))
            vz.append(float(words[8]))
            zs_fortesting.append(z)
            counter+=1
            i+=1
    infile_free.close()
    dt = (times[1]-times[0])*timestepsize # This might be handy
    times_single = np.arange(Nsteps)#*dt
    times_single_real = np.arange(Nsteps)*dt
    
    time_end = time.process_time()
    
    if max(zs_fortesting)>zhigh:
        continue
    if min(zs_fortesting)<zlow:
        continue
    
    Nin   = Nsteps
    
    alltimes.append(0)
    vx0 = vx[istart]
    vy0 = vy[istart]
    vz0 = vz[istart]
    vtot0 = np.array([vx0,vy0,vz0])
    for j in range(istart,Nin):
        i = j-istart      
        # Velocity:
        vxi   = vx[i]
        vyi   = vy[i]
        vzi   = vz[i]
        vtoti = np.array([vxi,vyi,vzi])
        vacf_tot[i] += np.dot(vtoti,vtot0)
        vacf_x[i]   += vxi*vx0
        vacf_y[i]   += vyi*vy0
        vacf_z[i]   += vzi*vz0
        vacf_par[i] += vxi*vx0+vyi*vy0
        average_counter[i] +=1
    
    Nins.append(Nin)
    

print('filescounter:', filescounter)
maxz_av /= filescounter

print('maxz_av:', maxz_av)

allRs      = np.array(allRs)
alltimes   = np.array(alltimes)
Nins       = np.array(Nins)

Ninbrush = Nsteps # Default, in case it does not exit the brush
for i in range(Nsteps-istart):
    counter = average_counter[i]
    if counter!=0:
        vacf_tot[i] /=counter
        vacf_x[i]   /=counter
        vacf_y[i]   /=counter
        vacf_z[i]   /=counter
        vacf_par[i] /=counter
    else:
       Ninbrush = i-1
       break

times_single      = times_single[istart:Ninbrush]
times_single_real = times_single_real[istart:Ninbrush]
Ninbrush = Ninbrush-istart
# Velocity autocorrelation function:
vacf_tot = vacf_tot[0:Ninbrush]
vacf_x   = vacf_x[0:Ninbrush]
vacf_y   = vacf_y[0:Ninbrush]
vacf_z   = vacf_z[0:Ninbrush]
vacf_par = vacf_par[0:Ninbrush]

## Average, SI units:
# Velocity:
vacf_tot_SI  = vacf_tot*unitlength/unittime 
vacf_x_SI = vacf_x*unitlength/unittime 
vacf_y_SI = vacf_y*unitlength/unittime 
vacf_z_SI = vacf_z*unitlength/unittime 
vacf_par_SI = vacf_par*unitlength/unittime


outfile_vacf = open(outfilename_vacf,'w')
outfile_vacf.write('Time step; Time_SI; vacf_tot_SI;  vacf_z_SI;  vacf_par_SI;  vacf_x_SI;  vacf_y_SI\n')
for i in range(Ninbrush):
     outfile_vacf.write('%.5e %.5e %.5e %.5e %.5e %.5e %.5e\n' % (times_single[i],times_single_real[i],vacf_tot_SI[i],vacf_z_SI[i],vacf_par_SI[i],vacf_x_SI[i],vacf_y_SI[i])) 
outfile_vacf.close()

# Find d by the integral:
Dtot = 0
Dx   = 0
Dy   = 0
Dz   = 0
Dpar = 0
for i in range(Ninbrush):
    Dtot += vacf_tot_SI[i]
    Dx   += vacf_x_SI[i]
    Dy   += vacf_y_SI[i]
    Dz   += vacf_z_SI[i]
    Dpar += vacf_par_SI[i]
Dtot *= dt
Dx   *= dt
Dy   *= dt
Dz   *= dt
Dpar *= dt

print('spacing:',spacing)
print('Dtot:',Dtot)
print('Dx:',Dx)
print('Dy:',Dy)
print('Dz:',Dz)
print('Dpar:',Dpar)

outfile_Dvacf = open(outfilename_Dvacf,'w')
outfile_Dvacf.write('d   Dtot   Dz   Dpar   Dx   Dy\n')
outfile_Dvacf.write('%.2f %.5e %.5e %.5e %.5e %.5e\n' % (spacing,Dtot,Dz,Dpar,Dx,Dy))
outfile_Dvacf.close()

plt.figure(figsize=(6,5))
plt.plot(times_single_real, vacf_tot_SI, label=r'$<\vec{v}(0)\cdot\vec{v}(t)>$')
plt.plot(times_single_real, vacf_x_SI, label=r'$<v_x(0)\cdot v_x(t)>$')
plt.plot(times_single_real, vacf_y_SI, label=r'$<v_y(0)\cdot v_y(t)>$')
plt.plot(times_single_real, vacf_z_SI, label=r'$<v_z(0)\cdot v_z(t)>$')
plt.plot(times_single_real, vacf_par_SI, label=r'$<v_\parallel\cdot v_\parallel(t)>$')
plt.xlabel(r'Time [s]')
plt.ylabel(r'VACF')
plt.title(r'VACF vs time, d=%.2f' % spacing)
plt.tight_layout()
plt.savefig(plotname)
plt.show()
