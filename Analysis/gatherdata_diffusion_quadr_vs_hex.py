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

# Input parameters for file selection: 
psigma   = 1 # For instance 
spacings = [3,3.5,4]
damp     = 10
N        = len(spacings)
# Input booleans for file selection:
bulkdiffusion = False
substrate     = False
confignrs     = np.arange(1,1001)

########## Ds ################
## Quadr: ######
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
## Hex: #########
DRs_h = np.zeros(N)
Dxs_h = np.zeros(N)
Dys_h = np.zeros(N)
Dzs_h = np.zeros(N)
Dparallel_h = np.zeros(N)
# Ds, stdv
DRs_stdv_h = np.zeros(N)
Dxs_stdv_h = np.zeros(N)
Dys_stdv_h = np.zeros(N)
Dzs_stdv_h = np.zeros(N)
Dparallel_stdv_h = np.zeros(N)


##### bs: ####################
## Quadr: #######
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
## Hex: #########
bRs_h = np.zeros(N)
bxs_h = np.zeros(N)
bys_h = np.zeros(N)
bzs_h = np.zeros(N)
bparallel_h = np.zeros(N)
# bs, stdv
bRs_stdv_h = np.zeros(N)
bxs_stdv_h = np.zeros(N)
bys_stdv_h = np.zeros(N)
bzs_stdv_h = np.zeros(N)
bparallel_stdv_h = np.zeros(N)


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
    filestext    = '_config'+str(confignrs[0])+'to'+str(confignrs[-1])

endlocation_out = '/Diffusion_bead_near_grid/hex_vs_quadr/'
outfilename  = endlocation_out+'quadr_vs_hex.txt'
plotname     = endlocation_out+'quadr_vs_hex.png'
plotname_fit = endlocation_out+'quadr_vs_hex_fit.png'
indfilename  = endlocation_out+'quadr_vs_hex_fitindices.txt'

outfile = open(outfilename, 'w')
outfile.write('d   D_R2_q   sigmaD_R2_q  b_R2_q sigmab_R2_q; D_z2_q  sigmaD_z2_q b_z2_q  sigmaD_z2_q; D_par2_q sigmaD_par2_q  b_par2_q  sigmab_par2_q |    D_R2_h   sigmaD_R2_h  b_R2_h sigmab_R2_h; D_z2_h  sigmaD_z2_h b_z2_h  sigmaD_z2_h; D_par2_h sigmaD_par2_h  b_par2_h  sigmab_par2_h\n')

indexfile = open(indfilename, 'w')
indexfile.write('Start_index_R_q     end_index_R_q     Start_index_ort_q     end_index_ort_q     Start_index_par_q     end_index_par_q | start_index_R_h     end_index_R_h     Start_index_ort_h     end_index_ort_h     Start_index_par_h     end_index_par_h\n')

for i in range(N):
    spacing = spacings[i]
    outfilename_d   = endlocation_out+'quadr_vs_hex_d'+str(spacing)+'.txt'
    inlocation_base = '/Diffusion_bead_near_grid/Spacing'+str(spacing)+'/damp%i_diffseedLgv/' % damp +parentfolder+ 'Sigma_bead_' +str(psigma) + '/'
    inlocation_q    = inlocation_base+'Nocut/'
    inlocation_h    = inlocation_base+'Hexagonal/Nocut/'
    
    infilename_q = inlocation_q+'diffusion'+filestext+'_nocut.txt' 
    metaname_q   = inlocation_q+'diffusion_metadata'+filestext+'_nocut.txt'
    
    infilename_h = inlocation_h+'diffusion'+filestext+'_hex_nocut.txt' 
    metaname_h   = inlocation_h+'diffusion_metadata'+filestext+'_hex_nocut.txt'
    
    # Read in:
    #### Automatic part
    ### Quadr:
    ## Find the extent of the polymers: Max z-coord of beads in the chains
    infile_q = open(infilename_q, "r")
    lines = infile_q.readlines() # This takes some time
    # Getting the number of lines, etc.
    line = lines[1]
    words = line.split()
    
    # Ds
    DRs[i] = float(words[0])
    Dzs[i] = float(words[4])
    Dparallel[i] = float(words[8])
    # Ds, stdv
    DRs_stdv[i] = float(words[1])
    Dzs_stdv[i] = float(words[5])
    Dparallel_stdv[i] = float(words[9])
    
    # bs
    bRs[i] = float(words[2])
    bzs[i] = float(words[6])
    bparallel[i] = float(words[10])
    
    # bs, stdv
    bRs_stdv[i] = float(words[3])
    bzs_stdv[i] = float(words[7])
    bparallel_stdv[i] = float(words[11])
    
    infile_q.close()
    
    ### Hex:
    ## Find the extent of the polymers: Max z-coord of beads in the chains
    infile_h = open(infilename_h, "r")
    lines = infile_h.readlines() # This takes some time
    # Getting the number of lines, etc.
    line = lines[1]
    words = line.split()
    
    # Ds
    DRs_h[i] = float(words[0])
    Dzs_h[i] = float(words[4])
    Dparallel_h[i] = float(words[8])
    # Ds, stdv
    DRs_stdv_h[i] = float(words[1])
    Dzs_stdv_h[i] = float(words[5])
    Dparallel_stdv_h[i] = float(words[9])
    
    # bs
    bRs_h[i] = float(words[2])
    bzs_h[i] = float(words[6])
    bparallel_h[i] = float(words[10])
    
    # bs, stdv
    bRs_stdv_h[i] = float(words[3])
    bzs_stdv_h[i] = float(words[7])
    bparallel_stdv_h[i] = float(words[11])
    
    infile_h.close()
    
    outfile.write('%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e |  %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e \n' % (spacing, DRs[i], DRs_stdv[i], bRs[i], bRs_stdv[i], Dzs[i], Dzs_stdv[i], bzs[i], bzs_stdv[i], Dparallel[i], Dparallel_stdv[i], bparallel[i], bparallel[i], DRs_h[i], DRs_stdv_h[i], bRs_h[i], bRs_stdv_h[i], Dzs_h[i], Dzs_stdv_h[i], bzs_h[i], bzs_stdv_h[i], Dparallel_h[i], Dparallel_stdv_h[i], bparallel_h[i], bparallel_h[i]))
    
    # Outfile for this specific d (nice and neat):
    outfile_d = open(outfilename_d, 'w')
    outfile_d.write('Quadr (avg rms) | Hex (avg rms)\n')
    outfile_d.write('R %.5e %.5e | %.5e %.5e\n' % (DRs[i], DRs_stdv[i],DRs_h[i], DRs_stdv_h[i]))
    outfile_d.write('z %.5e %.5e | %.5e %.5e\n' % (Dzs[i], Dzs_stdv[i],Dzs_h[i], Dzs_stdv_h[i]))
    outfile_d.write('par %.5e %.5e | %.5e %.5e\n' % (Dparallel[i], Dparallel_stdv[i],Dparallel_h[i], Dparallel_stdv_h[i]))
    outfile_d.close()
    
    # Quadr:
    metafile = open(metaname_q, 'r')
    mlines   = metafile.readlines()
    startindex_R_q   = int(mlines[0].split()[1])
    endindex_R_q     = int(mlines[1].split()[1])
    startindex_ort_q = int(mlines[2].split()[1])
    endindex_ort_q   = int(mlines[3].split()[1])
    startindex_par_q = int(mlines[4].split()[1])
    endindex_par_q   = int(mlines[5].split()[1])
    metafile.close()
    # Hex:
    metafile = open(metaname_h, 'r')
    mlines   = metafile.readlines()
    startindex_R_h   = int(mlines[0].split()[1])
    endindex_R_h     = int(mlines[1].split()[1])
    startindex_ort_h = int(mlines[2].split()[1])
    endindex_ort_h   = int(mlines[3].split()[1])
    startindex_par_h = int(mlines[4].split()[1])
    endindex_par_h   = int(mlines[5].split()[1])
    metafile.close()
    indexfile.write('%i %i %i %i %i %i |  %i %i %i %i %i %i\n' % (startindex_R_q, endindex_R_q, startindex_ort_q, endindex_ort_q, startindex_par_q, endindex_par_q, startindex_R_h, endindex_R_h, startindex_ort_h, endindex_ort_h, startindex_par_h, endindex_par_h))

outfile.close()

spacings = np.array(spacings)
    
plt.figure(figsize=(6.4,4),dpi=300)
ax = plt.subplot(111)
# Quadr.
ax.plot(spacings, DRs, '-o', color='r', label=r'$D_R$, quadr.')
ax.plot(spacings, Dzs, '-d',color='b', label=r'$D_\perp$, quadr.')
ax.plot(spacings, Dparallel, '-s', color='g', label=r'$D_\parallel$, quadr.')
# Hex.
ax.plot(spacings, DRs_h, '-X', color='lightcoral', label=r'$D_R$, hex.')
ax.plot(spacings, Dzs_h, '-v', color='c', label=r'$D_\perp$, hex.')
ax.plot(spacings, Dparallel_h, '-<', color='limegreen', label=r'$D_\parallel$, hex.')
plt.xlabel(r'$d$ (nm)')
plt.ylabel(r'Diffusion constant $D$ (m$^2$/s)')
#plt.title('Diffusion constant $D$, dynamic %s' % systemtype)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig(plotname)
plt.show()

