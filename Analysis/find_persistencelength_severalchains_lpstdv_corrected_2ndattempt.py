import matplotlib.pyplot as plt               # To plot
from scipy.optimize import curve_fit
import numpy as np
import random
import math
import time

# Function for curve_fit
def costheta_exponential(s,P):
    return np.exp(-s/P)

start_time = time.process_time()

arctest     = False

# Max number of iterations in curve_fit: # (Use default unless it crashes)
thismaxfev  = 5000

### Input file parameters
Kangle      = 14
Kbond       = 140
factor      = Kangle/float(Kbond)
T           = 310
M           = 9
N           = 101 
psigma      = 1
charge      = -1
mass        = 1
K           = Kangle
systemtest  = False
lotsofchains = False

### Input and output file names

spacing     = 4
damp        = 10
gridspacing = spacing
T           = 3
flubbedup   = False
folder      = '/Diffusion_bead_near_grid/Spacing'+str(spacing)+'/damp%i_diffseedLgv/Brush/Sigma_bead_' % damp+str(psigma) + '/'
basename    = folder+'freeatom_confignr1.lammpstrj'
basename2 = basename

infilename   = basename+'.lammpstrj'
plot1name    = 'costheta_'+basename2+'.png'
plot2name    = 'angledistr_'+basename2+'.png'
plot3name    = 'costheta_actualdistance_'+basename2+'.png'
outfilename  = 'bond_lengths_'+basename2+'.txt'
outfilename2 = 'persistence_length_'+basename2+'.txt'
outfilename3 = 'costhetas_neighbourbonds_'+basename2+'.txt'
outfilename4 = 'ree_last_'+basename2+'.txt'
outfilename5 = 'ree_average_'+basename2+'.txt'
outfilename6 = 'persistence_length_actualdistance_'+basename2+'.txt'
outfilename7 = 'endzs_'+basename2+'.txt'
outfilename8 = 'bondlengths_systemwide_'+basename2+'.txt'


### Opening files
outfile3 = open(outfilename3,'w')
outfile4 = open(outfilename4,'w')
outfile5 = open(outfilename5,'w')
outfile6 = open(outfilename6,'w')

# The log file in LAMMPS contains a lot of lines detailing the initiation of the system. We want to access the information that comes after that.
linestart           = 6 #17            # The line the data starts at.
printeverynthstep   = 100
L                   = 10            # The size of our simulation box. LxLxL
startat             = 50            # To equilibrate. The number of ns we want to skip
dt                  = 0.00045       # The time step of our simulation. 0.00045 ns default for nano
skiplines           = 9             # If we hit 'ITEM:', skip this many steps...
skipelem            = 0             # The number of elements we skip in order to equilibrate (set to 0 if the .lammpstrj file should be equilibrated)
sampleevery         = 0             # Sample every 10th of the values written to file # The program gets WAY too slow if this is too small.
timefac             = dt*printeverynthstep*1e-9*sampleevery

#### Automatic part
infile = open(infilename, "r")
lines = infile.readlines()
# Getting the number of lines, etc.
totlines = len(lines)         # Total number of lines
lineend = totlines-1          # Index of last element

# Extracting the number of atoms:
words = lines[3].split()
print("words:", words)
print("words[0]:", words[0])
Nall = int(words[0])

words = lines[5].split()
xmin = float(words[0])
xmax = float(words[1])

words = lines[6].split()
ymin = float(words[0])
ymax = float(words[1])

words = lines[7].split()
zmin = float(words[0])
zmax = float(words[1])

Lx = xmax-xmin
Ly = ymax-ymin
Lz = zmax-zmin

halfLx = 0.5*Lx
halfLy = 0.5*Ly
halfLz = 0.5*Lz


## Making an array of indices:
index1 = np.zeros(M*N) # The chain index
index2 = np.zeros(M*N) # The atom (in chain) index
counter = 0
for i in range(M):
    for j in range(N):
        index1[counter] = i
        index2[counter] = j
        counter += 1

xes = []
ys  = []
zs  = []

x_curr = np.zeros((M,N))
y_curr = np.zeros((M,N))
z_curr = np.zeros((M,N))
tempzs = np.zeros(M)
times = []

indexshift = 0
if M==1 or flubbedup==True:
    indexshift = -1

# Extracting data
i = int(math.ceil(skipelem*(Nall+9)))
j = 0
testcounter = 0
skiplines += (Nall+skiplines)*sampleevery
while i<totlines:
    words = lines[i].split()
    if (words[0]=='ITEM:' and words[1]=='TIMESTEP'):
        if words[1]=='TIMESTEP':
            words2 = lines[i+1].split() # The time step is on the next line
            t = float(words2[0])
            times.append(t)
            if t!=0:
                counter=0
                xes.append(x_curr)
                ys.append(y_curr)
                zs.append(z_curr)
                # Reset for safety
                x_curr    = np.zeros((M,N))
                y_curr    = np.zeros((M,N))
                z_curr    = np.zeros((M,N))
                j+=1
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
        # Order:  id  type mol  x   y   z  vx   vy   vz
        #         [0] [1]  [2] [3] [4] [5] [6] [7]  [8]
        ind   = int(words[0])-1 # Atom ids go from zero to N-1.
        x     = float(words[3+indexshift])
        y     = float(words[4+indexshift])
        z     = float(words[5+indexshift])
        # Making array indices:
        ind1  = int(index1[ind])
        ind2  = int(index2[ind])
        # Add to lists
        x_curr[ind1,ind2] = x 
        y_curr[ind1,ind2] = y
        z_curr[ind1,ind2] = z
        i+=1 
        counter+=1

if systemtest==False:
    # Removing the first redundant element of position lists (neccessary if t=0 is not the first time step in the file/that we read)
    xes.pop(0)
    ys.pop(0)
    zs.pop(0)
if sampleevery==0: # Because sampleevery!=0 can mess with the program, and leave an empty x_curr after the read-in loop
    xes.append(x_curr)
    ys.append(y_curr)
    zs.append(z_curr)

infile.close()

print('Done with read-in loop')

# Finding the bond vectors for each time step.
# This will be an awfully blocky code, it seems

Nt = len(xes)

# Oh, vey.
bondvecs        = np.zeros((Nt, M, N-1, 3))
length_bondvecs = np.zeros((Nt, M, N-1))
ree_vec         = np.zeros((Nt, M))
ree2_vec        = np.zeros((Nt, M))
endzs		= np.zeros((Nt, M))

# A test:
infilename_zendtest = 'test_zend_from_find_persistencelength.txt'
infile_zendtest     = open(infilename_zendtest,'w')
bindex1 = 0
bindex2 = 0
for i in range(Nt):    # Loop over time steps
    for k in range(M): # Loop over chains
        xes_thistime = xes[i]
        ys_thistime  = ys[i]
        zs_thistime  = zs[i]
        xes_this     = xes_thistime[k,:]
        ys_this      = ys_thistime[k,:]
        zs_this      = zs_thistime[k,:]
        x1 = xes_this[0]
        y1 = ys_this[0]
        z1 = zs_this[0]
        bindex1 = 0
        ree_dx  = 0
        ree_dy  = 0
        ree_dz  = 0
        for j in range(N-1): # Looping over atoms in the chain to get bonds
            bindex2 = j+1
            x2 = xes_this[j+1]
            y2 = ys_this[j+1]
            z2 = zs_this[j+1]
            dx = x2-x1
            dy = y2-y1
            dz = z2-z1
            if abs(dx)>halfLx:
                if dx>0:
                    dx-=Lx
                else:
                    dx+=Lx
            if abs(dy)>halfLy:
                if dy>0:
                    dy-=Ly
                else:
                    dy+=Ly
            if abs(dz)>halfLz:
                if dz>0:
                    dz-=Lz
                else:
                    dz+=Lz
            bondvecs[i,k,j,0] = dx # x-component of bond vector j for time i
            bondvecs[i,k,j,1] = dy # y-component of bond vector j for time i
            bondvecs[i,k,j,2] = dz # z-component of bond vector j for time i
            ree_dx           += dx
            ree_dy           += dy
            ree_dz           += dz
            veclen2              = dx**2+dy**2+dz**2
            length_bondvecs[i,k,j] = np.sqrt(veclen2)
            
            # Test (verified):
            chainno1 = index1[bindex1]
            chainno2 = index1[bindex2]
            if abs(chainno1-chainno2)>0:
                print("Danger! Made bond between atoms in different chains! Difference:", chainno1-chainno2)
            # Next bond: The end particle is now the start particle 
            x1 = x2
            y1 = y2
            z1 = z2
            bindex1   = bindex2
        ree_len2      = ree_dx**2+ree_dy**2+ree_dz**2
        ree_vec[i,k]  = np.sqrt(ree_len2)
        ree2_vec[i,k] = ree_len2
        endzs[i,k]    = ree_dz
        infile_zendtest.write('%.16f\n' % ree_dz)
infile_zendtest.close()

print("Made bonds")
end_time = time.process_time()

print("Time:", end_time-start_time, " s")



### Finding the persistence length
# Want <cos(theta)> of sites that are at distance i apart
# We want this to be a time average
Nb = N-1
costheta = np.zeros((M,Nb))  # This matrix will hold the average costheta for each chain (M-index) and bond separation s (Nb-index). It is an average over time and pairs that fulfill s.
counters = np.zeros((M,Nb))
costheta[:,0] = 1            # Distance 0: <cos(theta)> = 1

costheta_chainsorted             = [] # Contains ct for all separations s, with contributions from every pair of bonds. Each element corresponds to one chain at one point in time
costheta_chainsorted_separations = [] # Lists the separations of costheta_chainsorted. On the same form
costheta_chainsorted_distances   = [] # Lists the distances of costheta_chainsorted. On the same form
costheta_chainmaster             = [] # Lists the chain index of costheta_chainsorted
costheta_timemaster              = [] # Lists the time index of costheta_chainsorted
costheta_eachMandt               = np.zeros((Nt,M,Nb)) # Have <cos(theta(s))> for each M and t
costheta_eachMandt_rms           = np.zeros((Nt,M,Nb))
costheta_totav                   = np.zeros(Nb)
costheta_totav_rms               = np.zeros(Nb)
cos_max                          = np.zeros(Nb)-1000
cos_min                          = np.zeros(Nb)+1000
cos_maxes_eachMandt              = np.zeros((Nt,M,Nb))-1000
cos_mins_eachMandt               = np.zeros((Nt,M,Nb))+1000
# Setting all \cos\theta(s=0)=1
cos_max[0]                       = 1
cos_min[0]                       = 1
costheta_totav[0]                = 1

for i in range(Nt):
    for j in range(M):
        cos_maxes_eachMandt[i,j,0] = 1
        cos_mins_eachMandt[i,j,0]  = 1
        costheta_eachMandt[i,j,0]  = 1

# Averaging over bonds as well:

for i in range(Nt):  # Looping over the time steps
    for l in range(M):           # Looping over the chains
        costheta_allvalues_unzipped            = []
        costheta_allvalues_separation_unzipped = []
        costheta_allvalues_distance_unzipped   = []
        for j in range(1,Nb):    # The distance between the bonds to be measured
            for k in range(Nb-j): # The position of the first bond
                bond1        = bondvecs[i,l,k]
                bond2        = bondvecs[i,l,k+j]
                distance     = np.sum(length_bondvecs[i,l,(k+1):(k+j+1)]) # Getting the distance between the two vectors
                length1      = length_bondvecs[i,l,k]
                length2      = length_bondvecs[i,l,k+j] 
                dotprod      = np.dot(bond1,bond2)         # We only find the angle between bond vectors in the same time step
                ct           = dotprod/(length1*length2)
                if(length1*length2==0):
                    print('bond1:', bond1, '; bond2:', bond2)
                costheta[l,j]             += ct
                counters[l,j]             += 1
                costheta_totav[j]         += ct
                costheta_eachMandt[i,l,j] += ct
                costheta_allvalues_unzipped.append(ct)
                costheta_allvalues_separation_unzipped.append(j)
                costheta_allvalues_distance_unzipped.append(distance)
        costheta_chainmaster.append(l)
        costheta_timemaster.append(i)
        costheta_chainsorted.append(costheta_allvalues_unzipped)
        costheta_chainsorted_separations.append(costheta_allvalues_separation_unzipped)
        costheta_chainsorted_distances.append(costheta_allvalues_distance_unzipped)

for i in range(1,Nb):
    costheta_totav[i] /= (M*Nt*(Nb-i))
    for j in range(M):
        costheta[j,i] /= Nt*(Nb-i)
        for k in range(Nt):
            costheta_eachMandt[k,j,i] /= (Nb-i)

costheta1_av_system  = np.mean(costheta[:,1]) # <cos(theta(s=1))>
costheta1_counter    = 0
# Finding the rms values:
costheta_rms         = np.zeros((M,Nb))
costheta1_rms_system = 0
#costheta_rms[0] = 0   # No variation here. # Don't actually need to set this
thiscounter = 0

# Finding RMS of costheta and costheta_eachMandt
Nthis = np.zeros((M,Nb))
for i1 in range(len(costheta_chainsorted)):                     # Looping over a set of time steps and chains
    costheta_allvalues      = costheta_chainsorted[i1]              # All ct for one chain for one time step.
    costheta_allseparations = costheta_chainsorted_separations[i1]  # List of the separation for those ct
    k                       = costheta_chainmaster[i1]              # Extracting the chain index for the current data
    j                       = costheta_timemaster[i1]               # Extracting the time index for the current data
    thiscounter += 1
    for i2 in range(len(costheta_allvalues)):                   # Looking at the data one by one
        cai = costheta_allvalues[i2]                                # Getting one cosine value
        i = costheta_allseparations[i2]                             # Getting the corresponding separation
        Nthis[k,i] += 1 
        costheta_rms[k,i] += (costheta[k,i]-cai)**2                 # Getting the contribution to the rms
        costheta_eachMandt_rms[j,k,i] += (costheta_eachMandt[j,k,i]-cai)**2
        costheta_totav_rms[i] += (costheta_totav[i]-cai)**2
        # Total max/min:
        if cai>cos_max[i]:
            cos_max[i]    = cai
        if cai<cos_min[i]:
            cos_min[i]    = cai
        # Max/min for chain and time:
        if cai>cos_maxes_eachMandt[j,k,i]:
            cos_maxes_eachMandt[j,k,i]    = cai
        if cai<cos_mins_eachMandt[j,k,i]:
            cos_mins_eachMandt[j,k,i]     = cai
        if i==1:
            costheta1_rms_system += (costheta1_av_system-cai)**2
            costheta1_counter    += 1
#                                                               # Getting the final value
costheta1_rms_system      = np.sqrt(costheta1_rms_system/(costheta1_counter-1))

for i in range(1,Nb):
    costheta_totav_rms[i] = np.sqrt(costheta_totav_rms[i]/(Nt*M*(Nb-i)-1)) 
    for k in range(M):
        costheta_rms[k,i] = np.sqrt(costheta_rms[k,i]/(Nthis[k,i]-1))
        for j in range(Nt):
            costheta_eachMandt_rms[j,k,i] = np.sqrt(costheta_eachMandt_rms[j,k,i]/(Nb-i)) 

print("Found cos(theta)")

for i in range(M):
    outfile3.write('%.16f %.16f\n' % (costheta[i,1],costheta_rms[i,1]))
outfile3.write('System wide: %.16f %.16f' % (costheta1_av_system,costheta1_rms_system))
outfile3.close()

#### Write ree last to file
for i in range(M):
    outfile4.write('%.16e\n' % (ree_vec[-1,i]))

outfile4.close()
#### Finding ree average and stdv                   # This might take some time
ree_av         = np.zeros(M)
ree2_av        = np.zeros(M)
ree_tot_av     = 0
ree2_tot_av    = 0

for i in range(M):
    ree_av[i]  = np.mean(ree_vec[:,i])
    ree2_av[i] = np.mean(ree2_vec[:,i])

ree_tot_av     = np.mean(ree_av)
ree2_tot_av    = np.mean(ree2_av)

# Stdv
ree_stdv       = np.zeros(M)
ree2_stdv      = np.zeros(M)
ree_tot_stdv   = 0
ree2_tot_stdv  = 0
for j in range(M):
    for i in range(Nt):
        ree_stdv[j]   += (ree_vec[i,j]  - ree_av[j])**2
        ree2_stdv[j]  += (ree2_vec[i,j] - ree2_av[j])**2
        ree_tot_stdv  += (ree_vec[i,j]  - ree_tot_av)**2
        ree2_tot_stdv += (ree2_vec[i,j] - ree2_tot_av)**2
    ree_stdv[j]  = np.sqrt(ree_stdv[j]/float(Nt-1))
    ree2_stdv[j] = np.sqrt(ree2_stdv[j]/float(Nt-1))
ree_tot_stdv  = np.sqrt(ree_tot_stdv/(M*Nt-1))
ree2_tot_stdv = np.sqrt(ree2_tot_stdv/(M*Nt-1))

for i in range(M):
    outfile5.write('<r_ee>: %.16e %.16e <r_ee^2>: %.16e %.16e\n' % (ree_av[i],ree_stdv[i],ree2_av[i],ree2_stdv[i]))
outfile5.write('System total: %.16e %.16e %.16e %.16e' % (ree_tot_av,ree_tot_stdv,ree2_tot_av,ree2_tot_stdv))
outfile5.close()


# For plotting:
separation         = np.arange(Nb)
# Make each bond count as separation 1
persistencelength  = np.zeros(M)
pl_stdv            = np.zeros(M)
allvals_pl         = np.zeros((M,Nt)) 
pl_stdv_all        = 0
pl_all             = 0         
# Make each bond count with its length
persistencelength2 = np.zeros(M)
pl_stdv2           = np.zeros(M)
allvals_pl2        = np.zeros((M,Nt)) 
pl_stdv_all2       = 0
pl_all2            = 0 

#Ntimeav = M*Nt/100. # 100 bins
for k in range(len(costheta_chainmaster)):
    i = costheta_chainmaster[k]                                  # Getting chain index
    j = costheta_timemaster[k]                                   # Getting time index
    costhetas_this        = costheta_chainsorted[k]              # Getting list of cosines
    separations_this      = costheta_chainsorted_separations[k]  # Getting list of separations
    distances_this        = costheta_chainsorted_distances[k]    # Getting list of distances
    # First: Separation fit:
    popt, pcov            = curve_fit(costheta_exponential, separations_this, costhetas_this, maxfev=thismaxfev)
    this_pl               = popt[0]
    uncertainty           = np.sqrt(pcov[0])
    persistencelength[i] += this_pl                              # persistencelength is the average over fits found from {all ct in one time step and one chain}
    allvals_pl[i,j]       = this_pl
    # Test of new implementation, separation fit: Averaging first:
    popt, pcov            = curve_fit(costheta_exponential, separation, costheta_eachMandt[j,i,:], maxfev=thismaxfev)
    this_pl_TEST          = popt[0]                           # persistencelength is the average over fits found from {all ct in one time step and one chain}
    if i==1 and j==0: # Have ONE plot that shows what's going on:
        separations_this = np.array(separations_this)
        fittedgraph_this = costheta_exponential(separations_this, this_pl)
        x_eline = np.zeros(2)
        y_eline = np.zeros(2)+ 1./np.exp(1)
        x_eline[0] = 0
        x_eline[1] = 100
        plt.figure(figsize=(6,5))
        plt.plot(separations_this, costhetas_this, '.', label='Values')
        plt.plot(separations_this, fittedgraph_this, label='Fit')
        plt.xlabel(r'Bond distance $s$', fontsize=16)
        plt.ylabel(r'<cos($\theta$)>', fontsize=16)
        plt.tight_layout(pad=2.0)
        plt.legend(loc="upper right")
        plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$', fontsize=16)
        #plt.show()
        plt.savefig('costheta_onefit_data_fit_allpoints_1_Ktheta%i' %Kangle)
        #
        plt.figure(figsize=(6,5))
        plt.plot(separation, costheta_eachMandt[j,i,:], label='Mean values')
        plt.plot(separation, cos_maxes_eachMandt[j,i,:], '.', label='Max. values')
        plt.plot(separation, cos_mins_eachMandt[j,i,:], '.', label='Min. values')
        plt.plot(separations_this, fittedgraph_this, label='Fit')
        plt.xlabel(r'Bond distance $s$', fontsize=16)
        plt.ylabel(r'<cos($\theta$)>', fontsize=16)
        plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
        plt.legend(loc="upper right")
        plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$', fontsize=16)
        #plt.show()
        plt.savefig('costheta_onefit_data_fit_Ktheta%i' %Kangle)
        #
        if min(costheta_eachMandt[j,i,:])>0:
            plt.figure(figsize=(6,5))
            plt.plot(separation, costheta_eachMandt[j,i,:], label='Mean values')
            plt.plot(separation, cos_maxes_eachMandt[j,i,:], '.', label='Max. values')
            plt.plot(separation, cos_mins_eachMandt[j,i,:], '.', label='Min. values')
            plt.plot(separations_this, fittedgraph_this, label='Fit')
            plt.yscale('log')
            plt.xlabel(r'Bond distance $s$', fontsize=16)
            plt.ylabel(r'<cos($\theta$)>', fontsize=16)
            plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
            plt.legend(loc="upper right")
            plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$', fontsize=16)
            #plt.show()
            plt.savefig('log_costheta_onefit_data_fit_Ktheta%i' %Kangle)
        #
        # TEST:
        fittedgraph_this = costheta_exponential(separation, this_pl_TEST)
        plt.figure(figsize=(6,5))
        plt.plot(separation, costheta_eachMandt[j,i,:], label='Values')
        plt.plot(separation, fittedgraph_this, label='Fit')
        #plt.plot(x_eline, y_eline, '--')
        plt.xlabel(r'Bond distance $s$', fontsize=16)
        plt.ylabel(r'<cos($\theta$)>', fontsize=16)
        plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
        plt.legend(loc="upper right")
        plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$ using <cos($\theta$)>', fontsize=16)
        #plt.show()
        plt.savefig('costheta_onefit_data_fit_AVERAGINGFIRST_Ktheta%i' %Kangle)
    if i==1 and j==1:
        # TEST:
        separations_this = np.array(separations_this)
        fittedgraph_this = costheta_exponential(separations_this, this_pl)
        plt.figure(figsize=(6,5))
        plt.plot(separations_this, costhetas_this, '.', label='Values')
        plt.plot(separations_this, fittedgraph_this, label='Fit')
        plt.xlabel(r'Bond distance $s$', fontsize=16)
        plt.ylabel(r'<cos($\theta$)>', fontsize=16)
        plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
        plt.legend(loc="upper right")
        plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$', fontsize=16)
        #plt.show()
        plt.savefig('costheta_onefit_data_fit_allpoints_2_Ktheta%i' %Kangle)
        #
        fittedgraph_this = costheta_exponential(separation, this_pl_TEST)
        plt.figure(figsize=(6,5))
        plt.plot(separation, costheta_eachMandt[j,i,:], label='Values')
        plt.plot(separation, fittedgraph_this, label='Fit')
        plt.xlabel(r'Bond distance $s$', fontsize=16)
        plt.ylabel(r'<cos($\theta$)>', fontsize=16)
        plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
        plt.legend(loc="upper right")
        plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$ using <cos($\theta$)>', fontsize=16)
        #plt.show()
        plt.savefig('costheta_onefit_data_fit_AVERAGINGFIRST_2_Ktheta%i' %Kangle)
        if min(costheta_eachMandt[j,i,:])>0:
            plt.figure(figsize=(6,5))
            plt.plot(separation, costheta_eachMandt[j,i,:], label='Mean values')
            plt.plot(separation, cos_maxes_eachMandt[j,i,:], '.', label='Max. values')
            plt.plot(separation, cos_mins_eachMandt[j,i,:], '.', label='Min. values')
            plt.plot(separation, fittedgraph_this, label='Fit')
            plt.yscale('log')
            plt.xlabel(r'Bond distance $s$', fontsize=16)
            plt.ylabel(r'<cos($\theta$)>', fontsize=16)
            plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
            plt.legend(loc="upper right")
            plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$', fontsize=16)
            #plt.show()
            plt.savefig('log_costheta_onefit_data_fit_2_Ktheta%i' %Kangle)
    if i==1 and j==2:
        # TEST:
        separations_this = np.array(separations_this)
        fittedgraph_this = costheta_exponential(separations_this, this_pl)
        plt.figure(figsize=(6,5))
        plt.plot(separations_this, costhetas_this, '.', label='Values')
        plt.plot(separations_this, fittedgraph_this, label='Fit')
        plt.xlabel(r'Bond distance $s$', fontsize=16)
        plt.ylabel(r'<cos($\theta$)>', fontsize=16)
        plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
        plt.legend(loc="upper right")
        plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$', fontsize=16)
        #plt.show()
        plt.savefig('costheta_onefit_data_fit_allpoints_3_Ktheta%i' %Kangle)
        #
        fittedgraph_this = costheta_exponential(separation, this_pl_TEST)
        plt.figure(figsize=(6,5))
        plt.plot(separation, costheta_eachMandt[j,i,:], label='Values')
        plt.plot(separation, fittedgraph_this, label='Fit')
        plt.xlabel(r'Bond distance $s$', fontsize=16)
        plt.ylabel(r'<cos($\theta$)>', fontsize=16)
        plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
        plt.legend(loc="upper right")
        plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$ using <cos($\theta$)>', fontsize=16)
        #plt.show()
        plt.savefig('costheta_onefit_data_fit_AVERAGINGFIRST_3_Ktheta%i' %Kangle)
        if min(costheta_eachMandt[j,i,:])>0:
            plt.figure(figsize=(6,5))
            plt.plot(separation, costheta_eachMandt[j,i,:], label='Mean values')
            plt.plot(separation, cos_maxes_eachMandt[j,i,:], '.', label='Max. values')
            plt.plot(separation, cos_mins_eachMandt[j,i,:], '.', label='Min. values')
            plt.plot(separation, fittedgraph_this, label='Fit')
            plt.yscale('log')
            plt.xlabel(r'Bond distance $s$', fontsize=16)
            plt.ylabel(r'<cos($\theta$)>', fontsize=16)
            plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
            plt.legend(loc="upper right")
            plt.title(r'<cos($\theta$)> vs $s$, finding one $l_p$', fontsize=16)
            #plt.show()
            plt.savefig('log_costheta_onefit_data_fit_3_Ktheta%i' %Kangle)
    # Distance fit:
    popt2, pcov2           = curve_fit(costheta_exponential, distances_this, costhetas_this, maxfev=thismaxfev)
    this_pl2               = popt2[0]
    persistencelength2[i] += this_pl2
    allvals_pl2[i,j]       = this_pl2
persistencelength /=float(Nt)
persistencelength2/=float(Nt)

stdv_test = False
counter   = 1
runcumul  = 0
runningav = 0
pl_all  = np.mean(persistencelength)
pl_all2 = np.mean(persistencelength2)
for i in range(M):
    for j in range(Nt):
        # Separations:
        pl_stdv[i]   += (persistencelength[i]-allvals_pl[i,j])**2
        pl_stdv_all  += (pl_all-allvals_pl[i,j])**2
        if stdv_test==True:
            runcumul  += allvals_pl[i,j]
            runningav = runcumul/counter 
            if allvals_pl[i,j]>20:
                print("l:,",i,"t:",j,"stdv,",counter," values:",np.sqrt(pl_stdv_all/float(counter)), "; pl_av =", pl_all, "; this pl-value:", allvals_pl[i,j], "av. of these:", runningav )
        counter+=1
        # Distances:
        pl_stdv2[i]  += (persistencelength2[i]-allvals_pl2[i,j])**2
        pl_stdv_all2 += (pl_all2-allvals_pl2[i,j])**2
    pl_stdv[i]        = np.sqrt(pl_stdv[i]/float(Nt-1))
    pl_stdv2[i]       = np.sqrt(pl_stdv2[i]/float(Nt-1))
pl_stdv_all           = np.sqrt(pl_stdv_all/float(M*Nt-1))
pl_stdv_all2          = np.sqrt(pl_stdv_all2/float(M*Nt-1))
fittedgraph     = costheta_exponential(separation, persistencelength[-1]) # Graph of the exponential function of the last chain (averaged over time)
fittedgraph_tot = costheta_exponential(separation, np.mean(persistencelength))

# Standard deviation in the mean persistence length:
pl_stdv_mean  = 0
pl_stdv_mean2 = 0
for i in range(M):
    pl_stdv_mean  += (persistencelength[i]-pl_all)**2
    pl_stdv_mean2 += (persistencelength2[i]-pl_all2)**2
pl_stdv_mean  = np.sqrt(pl_stdv_mean/(M-1))
pl_stdv_mean2 = np.sqrt(pl_stdv_mean2/(M-1))

print("pl_stdv_mean:", pl_stdv_mean)
print("pl_stdv_mean2:", pl_stdv_mean2)


lastpls      = allvals_pl[:,-1]
mean_lastpls = np.mean(lastpls)
lastpls_rms  = 0
for i in range(M):
    lastpls_rms += (lastpls[i]-mean_lastpls)**2
lastpls_rms = np.sqrt(lastpls_rms/(M-1))

lastpls2      = allvals_pl2[:,-1]
mean_lastpls2 = np.mean(lastpls2)
lastpls_rms2  = 0
for i in range(M):
    lastpls_rms2 += (lastpls2[i]-mean_lastpls2)**2
lastpls_rms2 = np.sqrt(lastpls_rms2/(M-1))

print("rms from the last values:", lastpls_rms)
print("Nt:", Nt)
fittedgraph2     = costheta_exponential(separation, persistencelength2[-1])
fittedgraph2_tot = costheta_exponential(separation, np.mean(persistencelength2))


print("K:", K)
print("Persistence length (from separation fit):", np.mean(pl_all), " (mean of chains)")
print("Persistence length 2 (from distance fit):", pl_all2, "\ndiff pl 1 and pl2:", (pl_all-pl_all2))
print("Percent deviation, diff pl 1 and pl2:", (pl_all-pl_all2)/pl_all)

end_time = time.process_time()
print("Time:", end_time-start_time, " s")


print(np.mean(length_bondvecs))

# Persistence length line
x_persistencelength = np.zeros(2) + persistencelength[-1]
y_persistencelength = np.zeros(2)
y_persistencelength[0] = 0
y_persistencelength[1] = 1

# e line:
x_eline = np.zeros(2)
y_eline = np.zeros(2)+ 1./np.exp(1)
x_eline[0] = 0
x_eline[1] = 100


print('costheta_totav:',costheta_totav)
plt.figure(figsize=(6,5))
plt.errorbar(separation, costheta_totav, yerr=costheta_totav_rms, fmt="none", capsize=2, label='Values')
plt.plot(separation, costheta_totav, '.', label=r'Mean $\cos\theta$')
plt.plot(separation, cos_max, '.', label=r'Max $\cos\theta$')
plt.plot(separation, cos_min, '.', label=r'Min $\cos\theta$')
if arctest==False:
    plt.plot(separation, fittedgraph_tot, label='Fit')
else:
    plt.plot(separation,np.cos(np.pi*separation/(2*(Nb-1))), label='cos(pi*s/2N-1)')
plt.xlabel(r'Bond distance $s$', fontsize=16)
plt.ylabel(r'<cos($\theta$)>', fontsize=16)
plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
plt.legend(loc="upper right")
plt.title(r'<cos($\theta$)> vs separation', fontsize=16)
plt.savefig(plot1name)

plot1oldname = 'lastchainvalues_'+plot1name
plt.figure(figsize=(6,5))
plt.errorbar(separation, costheta[-1,:], yerr=costheta_rms[-1,:], fmt="none", capsize=2, label='Values')
plt.plot(separation, costheta[-1,:], '.')
if arctest==False:
    plt.plot(separation, fittedgraph, label='Fit')
else:
    plt.plot(separation,np.cos(np.pi*separation/(2*Nb)), label='cos(pi*s/2N)')
plt.xlabel(r'Bond distance $s$', fontsize=16)
plt.ylabel(r'<cos($\theta$)>', fontsize=16)
plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
plt.legend(loc="upper right")
plt.title(r'<cos($\theta$)> vs separation', fontsize=16)
plt.savefig(plot1oldname)


plot1name_log = 'log_'+plot1name
plt.figure(figsize=(6,5))
plt.plot(separation, costheta_totav, label='Values')
if arctest==False:
    plt.plot(separation, fittedgraph_tot, label='Exp. fit')
else:
    plt.plot(separation,np.cos(np.pi*separation/(2*(Nb-1))), label='cos(pi*s/2N-1)')
plt.yscale('log')
plt.xlabel(r'Bond distance $s$', fontsize=16)
plt.ylabel(r'<cos($\theta$)>', fontsize=16)
plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
plt.legend(loc="upper right")
plt.title(r'<cos($\theta$)> vs separation', fontsize=16)
plt.savefig(plot1name_log)

if lotsofchains==True:
    plot1sevname = 'severalfits_'+plot1name
    plt.figure(figsize=(6,5))
    plt.plot(separation, costheta_totav, '.', label=r'Mean $\cos\theta$')
    if arctest==False:
        fg0  = costheta_exponential(separation, allvals_pl[0,0])
        fg7  = costheta_exponential(separation, allvals_pl[0,4])
        fg8  = costheta_exponential(separation, allvals_pl[1,4])
        fg1  = costheta_exponential(separation, allvals_pl[1,1])
        fg6  = costheta_exponential(separation, allvals_pl[1,0])
        fg5  = costheta_exponential(separation, allvals_pl[2,3])
        fg10 = costheta_exponential(separation, allvals_pl[int(M/8), int(Nt/8)])
        fg4  = costheta_exponential(separation, allvals_pl[int(M/4), int(Nt/4)])
        fg2  = costheta_exponential(separation, allvals_pl[int(M/2), int(Nt/2)])
        fg9  = costheta_exponential(separation, allvals_pl[int(3*M/4), int(3*Nt/4)])
        fg3  =  costheta_exponential(separation, allvals_pl[M-1,Nt-1])
        plt.plot(separation, fg0,'--', label='Fit m=0,t=0')
        plt.plot(separation, fg7,'--', label='Fit m=0,t=4')
        plt.plot(separation, fg8,'--', label='Fit m=1,t=4')
        plt.plot(separation, fg6,'--', label='Fit m=1,t=0')
        plt.plot(separation, fg1,'--', label='Fit m=1,t=1')
        plt.plot(separation, fg5,'--', label='Fit m=2,t=3')
        plt.plot(separation, fg8,'--', label='Fit m=M/8,t=Nt/8')
        plt.plot(separation, fg4,'--', label='Fit m=M/4,t=Nt/4')
        plt.plot(separation, fg2,'--', label='Fit m=M/2,t=Nt/2')
        plt.plot(separation, fg9,'--', label='Fit m=3M/2,t=3Nt/2')
        plt.plot(separation, fg3,'--', label='Fit m=M-1,t=Nt-1')
    else:
        plt.plot(separation,np.cos(np.pi*separation/(2*Nb)), label='cos(pi*s/2N)')
    plt.xlabel(r'Bond distance $s$', fontsize=16)
    plt.ylabel(r'<cos($\theta$)>', fontsize=16)
    plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
    plt.legend(loc="upper right")
    plt.title(r'<cos($\theta$)> vs separation', fontsize=16) # (last chain)', fontsize=16)
    plt.savefig(plot1sevname)
     
    plot1avname = 'averages_shortlegend_'+plot1name
    plt.figure(figsize=(6,5))
    plt.plot(separation, costheta_totav, '.', label=r'Mean $\cos\theta$')
    plt.plot(separation, costheta_eachMandt[0,0,:], '--', label=r'Averages')
    plt.plot(separation, costheta_eachMandt[0,1,:], '--')#, label=r'Average, t=0, m=1')
    plt.plot(separation, costheta_eachMandt[1,0,:], '--')#, label=r'Average, t=1, m=0')
    plt.plot(separation, costheta_eachMandt[1,1,:], '--')#, label=r'Average, t=1, m=1')
    plt.plot(separation, costheta_eachMandt[2,3,:], '--')#, label=r'Average, t=2, m=3')
    plt.plot(separation, costheta_eachMandt[int(Nt/4),int(M/4),:], '--')#, label=r'Average, t=Nt/4, m=M/4')
    plt.plot(separation, costheta_eachMandt[int(Nt/2),int(M/2),:], '--')#, label=r'Average, t=Nt/2, m=M/2')
    plt.plot(separation, costheta_eachMandt[int(3*Nt/4),int(3*M/4),:], '--')#, label=r'Average, t=3Nt/4, m=3M/4')
    plt.plot(separation, costheta_eachMandt[Nt-1,M-1,:], '--')#, label=r'Average, t=Nt-1, m=M-1')
    plt.xlabel(r'Bond distance $s$', fontsize=16)
    plt.ylabel(r'<cos($\theta$)>', fontsize=16)
    plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
    plt.legend(loc="lower left")
    plt.title(r'<cos($\theta$)> vs separation', fontsize=16) 
    plt.savefig(plot1avname)



plt.figure(figsize=(6,5))
plt.plot(separation, fittedgraph_tot, label='Fit, unit b')
plt.plot(separation, fittedgraph2_tot, label='Fit, exact b')
plt.xlabel(r'Bond distance $s$', fontsize=16)
plt.ylabel(r'<cos($\theta$)>', fontsize=16)
plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
plt.legend(loc="upper right")
plt.title(r'<cos($\theta$)> vs separation', fontsize=16)
plt.savefig(plot3name)

'''
plt.figure(figsize=(6,5))
plt.plot(costheta_allvalues[5], '.')
plt.xlabel(r'Index', fontsize=16)
plt.ylabel(r'cos($\theta$)', fontsize=16)
plt.tight_layout(pad=2.0)#, w_pad=0.0, h_pad=0.5)
plt.title(r'Values of cos($\theta$) for separation 5', fontsize=16)
plt.savefig(plot2name)
'''
outfile  = open(outfilename,'w')

# Have a costheta chain- and system file too.
bondlength_chain_vals     = np.zeros(M)
bondlength_chain_vals_rms = np.zeros(M)
bondlength_av_system      = np.mean(length_bondvecs)
bondlength_rms_system     = 0 


outfile.write('Nt: %i Nbonds: %i\n' % (Nt, N-1))
for i in range(Nt):
    outfile.write('Time: %.16e i: %i\n' % (times[i],i))
    for k in range(M):
        for j in range(N-1):
            bondlength_chain_vals[k] += length_bondvecs[i,k,j]
            bondlength_rms_system    += (bondlength_av_system-length_bondvecs[i,k,j])**2
            outfile.write('%i %i %.16e\n' % (j+1, k,length_bondvecs[i,k,j]))
bondlength_chain_vals /= Nt*(N-1)
bondlength_rms_system  = np.sqrt(bondlength_rms_system/(M*Nt*(N-1)-1))
outfile.close()

for i in range(Nt):
    for k in range(M):
        for j in range(N-1):
            bondlength_chain_vals_rms[k] += (bondlength_chain_vals[k]-length_bondvecs[i,k,j])**2
for i in range(M):
    bondlength_chain_vals_rms[i] = np.sqrt(bondlength_chain_vals_rms[k]/(Nt*(N-1)-1))

outfile8 = open(outfilename8,'w')
for i in range(M):
    outfile8.write('%.16e %.16e\n' % (bondlength_chain_vals[i],bondlength_chain_vals_rms[i]))
outfile8.write('System wide: %.16e %.16e' % (bondlength_av_system, bondlength_rms_system))

outfile2 = open(outfilename2,'w')
outfile2.write('The average persistence length for all chains: K=%i: %.16e %.16e\n' % (K, pl_all, pl_stdv_all))
for i in range(M):
    outfile2.write('%.16e %.16e %.16e\n' % (persistencelength[i], pl_stdv[i], persistencelength[i]*np.mean(length_bondvecs[:,i,:])))
outfile2.write('RMS of average persistence length of chains: %.16e (Taking the rms of the average chain persistencelengths)\n' % pl_stdv_mean)

outfile2.close()

outfile6.write('The average persistence length2 for all chains: K=%i: %.16e %.16e\n' % (K, pl_all2, pl_stdv_all2))
for i in range(M):
    outfile6.write('%.16e %.16e\n' % (persistencelength2[i], pl_stdv2[i]))
outfile6.write('RMS of average persistence length2 of chains: %.16e (Taking the rms of the average chain persistencelengths)\n' % pl_stdv_mean2)
outfile6.close()

outfile7 = open(outfilename7, 'w')

if M==9:
    print("Largest end z:",max(endzs[0])) 
    print("Largest end z:",max(endzs[1]))
    print("Largest end z:",max(endzs[2]))
    print("Largest end z:",max(endzs[3]))
    print("Largest end z:",max(endzs[4]))
    print("Largest end z:",max(endzs[5]))
    print("Largest end z:",max(endzs[6]))
    print("Largest end z:",max(endzs[7]))
    print("Largest end z:",max(endzs[8]))
    
    print("End zs:",endzs[1])

lastz_by_chain = np.zeros(M)
for i in range(Nt):
    thistimestep_lastz = endzs[i]
    for j in range(M):
        lastz_by_chain[j] += thistimestep_lastz[j] 
lastz_by_chain /= Nt
lastz_totalav   = np.mean(lastz_by_chain)

lastz_totalrms     = 0
lastz_by_chain_rms = np.zeros(M)
for i in range(Nt):
    thistimestep_lastz = endzs[i]
    for j in range(M):
        lastz_by_chain_rms[j] += (lastz_by_chain[j]-thistimestep_lastz[j])**2
        lastz_totalrms        += (lastz_totalav-thistimestep_lastz[j])**2
lastz_totalrms = np.sqrt(lastz_totalrms/(M*Nt-1))

for j in range(M):
    lastz_by_chain_rms[j] = np.sqrt(lastz_by_chain_rms[j]/Nt)
    outfile7.write('%.16e %.16e\n' % (lastz_by_chain[j],lastz_by_chain_rms[j]))
outfile7.write('Total average, end: %.16e %.16e' % (lastz_totalav,lastz_totalrms))

outfile7.close()
print("Script finished.")
