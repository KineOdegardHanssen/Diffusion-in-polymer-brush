import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
import numpy as np

damp = 10
# Input parameters for file selection: 
popup_plots = False
spacings = [1,1.25,1.5,2,3,4,5,6,7,8,10,15,25,50,75,100]
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
Nsteps   = 2001
unitlength   = 1e-9
unittime     = 2.38e-11 # s
timestepsize = 0.00045*unittime

Nd = len(spacings)
Dvacf_tot = np.zeros(Nd)
Dvacf_z   = np.zeros(Nd)
Dvacf_par = np.zeros(Nd)

basepath       = '/Diffusion_bead_near_grid/'
filestext      = 'config'+str(confignrs[0])+'to'+str(confignrs[-1])
outfolder      = basepath + 'D_vs_d/Brush/Sigma_bead_'+str(psigma)+'/Nocut/'
outfilename    = outfolder + 'D_from_vacf_vs_d_dynamic_nocut_divbydim.txt'
plotname       = outfolder + 'D_from_vacf_vs_d_dynamic_nocut_divbydim.png'
plotname_short = outfolder + 'D_from_vacf_vs_d_dynamic_nocut_short_divbydim.png'

outfile = open(outfilename,'w')

outfile.write('spacing   Dvacf_tot   Dvacf_z   Dvacf_par\n')
for i in range(Nd):
    spacing = spacings[i]
    endlocation_in = basepath + '/Spacing'+str(spacing)+'/damp%i_diffseedLgv/Brush/Sigma_bead_' % damp+str(psigma) + '/'
    endlocation    = endlocation_in +'Nocut/'
    # In-file:
    infilename     = endlocation+'Dvacf_'+filestext+'_nocut'
    infilename = infilename+'.txt'
    infile = open(infilename,'r')
    lines  = infile.readlines()
    words  = lines[1].split()
    Dvacf_tot[i] = float(words[1])/6.
    Dvacf_z[i]   = float(words[2])/2.
    Dvacf_par[i] = float(words[3])/4.
    outfile.write('%.2f %.5e %.5e %.5e\n' % (spacing, Dvacf_tot[i], Dvacf_z[i], Dvacf_par[i]))
outfile.close()

plt.figure(figsize=(6,4),dpi=300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(spacings, Dvacf_tot, '-o', label=r'$D_{tot}$')
plt.plot(spacings, Dvacf_z, '-*', label=r'$D_{z}$')
plt.plot(spacings, Dvacf_par, '-d', label=r'$D_{\parallel}$')
plt.xlabel(r'Spacing $d$ (nm)', fontsize=12)
plt.ylabel(r'Diffusion constant $D$ (m$^2$/s)', fontsize=12)
plt.legend(loc='lower center',fontsize=12)
plt.tight_layout()
plt.savefig(plotname)
plt.show()


## Short plot:
N10 = 0
for i in range(Nd):
    if spacings[i]==10:
        N10 += 1
        break
    N10 += 1
    
plt.figure(figsize=(6,4),dpi=300)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(spacings[:N10], Dvacf_tot[:N10], '-o', label=r'$D_{tot}$')
plt.plot(spacings[:N10], Dvacf_z[:N10], '-*', label=r'$D_{z}$')
plt.plot(spacings[:N10], Dvacf_par[:N10], '-d', label=r'$D_{\parallel}$')
plt.xlabel(r'Spacing $d$ (nm)', fontsize=12)
plt.ylabel(r'Diffusion constant $D$ (m$^2$/s)', fontsize=12)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(plotname_short)
plt.show()
