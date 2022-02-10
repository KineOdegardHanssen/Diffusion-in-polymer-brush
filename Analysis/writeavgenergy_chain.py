import matplotlib.pyplot as plt                     # To plot
from scipy.optimize import curve_fit
from pylab import *
import numpy as np
import random
import math
import time
import os
import glob


spacings  = [50]#[1,3,10,50]#[1,10,50]#[3,10,50]
dampfac   = 10
sigmabead = 1
confignrs = np.arange(1,1001)
skipbool  = False

for spacing in spacings:
    avgavgE      = 0
    avgElist     = []
    avgsizeE     = 0
    avgsizeElist = []
    avgNE        = 0
    NElist       = []
    Nconfigs  = 0
    Nc_size   = 0
    for confignr in confignrs:
        filename = '/Diffusion_bead_near_grid/Spacing%s/damp%i_diffseedLgv/Brush/Sigma_bead_%s/log-files/log.confignr%i_pechain_printevery10' % (str(spacing),dampfac,str(sigmabead),confignr)
        try:
            infile = open(filename,'r')
        except:
            continue
        lines  = infile.readlines()
        
        totlines = len(lines)
        
        steps = []
        energy = []    
        
        i  = 0
        NE = 0
        startread = False
        while i<totlines:
            words = lines[i].split()
            if startread==True:
                if words[0]!='Loop' and words[0]!='ERROR' and words[0]!='WARNING:':
                    # Find properties  
                    energy_this = abs(float(words[1]))
                    steps.append(float(words[0]))
                    energy.append(energy_this) # Count neg.pot. energy as energy   
                    if energy_this>0: 
                         NE+=1
                elif words[0]=='ERROR':
                    skipbool = True
                else:
                    i = totlines 
                    break
            if len(words)>0:
                if words[0]=='Step' and words[1]=='c_pechainprint':
                    startread=True
            i+=1
        infile.close()
        if skipbool==True:
            skipbool=False
            continue
        
        Nconfigs+=1
        energy = np.array(energy)
        N = len(steps)
        # Compute quantities
        Esum  = np.sum(energy)
        avgE  = Esum/N  
        sizeE = Esum/NE
        NE    = NE/N
        # Accumulate for averages
        avgavgE +=avgE
        avgNE   +=NE
        if NE!=0:
            avgsizeE+=sizeE
            Nc_size+=1
        # Append to lists
        avgElist.append(avgE)
        if NE!=0:
            avgsizeElist.append(sizeE)
        NElist.append(NE)
    # Finalize averages
    avgavgE/=Nconfigs
    avgNE/=Nconfigs
    avgsizeE/=Nc_size
    rmsE     = 0
    rmsNE    = 0
    rmssizeE = 0
    for i in range(Nconfigs):
        rmsE    +=(avgElist[i]-avgavgE)**2
        rmsNE   +=(NElist[i]-avgNE)**2
    for i in range(len(avgsizeElist)):
        rmssizeE+=(avgsizeElist[i]-avgsizeE)**2
    rmsE=np.sqrt(rmsE/(Nconfigs-1))
    rmsNE=np.sqrt(rmsNE/(Nconfigs-1))
    rmssizeE=np.sqrt(rmssizeE/(Nconfigs-1))
    print('spacing:',spacing, '; avgE:',avgavgE, 'rmsE:', rmsE, '; avgNE:',avgNE, 'rmsNE:', rmsNE, '; avgsizeE:',avgsizeE, 'rmssizeE:', rmssizeE )
